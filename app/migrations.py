"""Database creation and migration (schema versions tracked with PRAGMA user_version).

    version 0, no tables      -> fresh install: create schema v2
    version 0, v1 'predictions' table (machine_id TEXT, alert_status on each row)
                              -> back the file up, then migrate to v2 in ONE transaction
    version 2                 -> nothing to do

Run by hand (optional; the app also does this at start-up):
    python -m app.migrations --db instance/predictions.db            # migrate if needed
    python -m app.migrations --db instance/predictions.db --dry-run  # report what would happen, change nothing

What the v1 -> v2 migration preserves
  * every prediction, with its original id, timestamp, readings, probability, model and recommendation
  * machines are created from the distinct v1 machine_id values (case-insensitive); the machine's type is the type of
    its first prediction. If a machine was submitted with different types, each prediction keeps the type it was run with
    and the conflict is counted in the report.
  * alerts are rebuilt by replaying the v1 rows, in order, through the v2 lifecycle (duplicates collapse into one alert
    per machine; a later Normal reading resolves the alert). v1 acknowledgements are carried over.
  * the original table is kept as `legacy_predictions_v1` so nothing is ever lost.
"""
import argparse
import os
import shutil
import sqlite3
from datetime import datetime

from app import db

LEGACY_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    machine_id      TEXT    NOT NULL,
    machine_type    TEXT    NOT NULL,
    air_temp        REAL    NOT NULL,
    process_temp    REAL    NOT NULL,
    rpm             REAL    NOT NULL,
    torque          REAL    NOT NULL,
    tool_wear       REAL    NOT NULL,
    probability     REAL    NOT NULL,
    risk_level      TEXT    NOT NULL CHECK (risk_level IN ('Normal','Warning','Critical')),
    severity        INTEGER NOT NULL,
    model_key       TEXT    NOT NULL,
    model_label     TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    failure_type    TEXT,
    recommendation  TEXT    NOT NULL,
    alert_status    TEXT    NOT NULL DEFAULT 'none' CHECK (alert_status IN ('none','open','acknowledged'))
);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_machine ON predictions(machine_id);
CREATE INDEX IF NOT EXISTS idx_pred_risk    ON predictions(risk_level);
"""


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} - {"sqlite_sequence"}


def _backup(path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = f"{path}.v1-backup-{stamp}"
    src, dst = sqlite3.connect(path), sqlite3.connect(dest)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    return dest


def migrate_v1_to_v2(conn):
    """Run inside an open transaction. Returns a report dict. Raises if the data would not reconcile."""
    report = {"predictions": 0, "machines": 0, "machine_type_conflicts": 0, "alerts_created": 0,
              "alerts_resolved": 0, "alerts_active": 0, "acknowledged_carried_over": 0}
    conn.execute("ALTER TABLE predictions RENAME TO legacy_predictions_v1")
    for stmt in db.SCHEMA_STATEMENTS:
        conn.execute(stmt)
    legacy = conn.execute("SELECT * FROM legacy_predictions_v1 ORDER BY id").fetchall()
    machines = {}
    for r in legacy:
        key = r["machine_id"].lower()
        if key not in machines:
            cur = conn.execute("INSERT INTO machines (machine_code, machine_type, created_at) VALUES (?,?,?)",
                               (r["machine_id"], r["machine_type"], r["created_at"]))
            machines[key] = {"id": cur.lastrowid, "type": r["machine_type"]}
        elif machines[key]["type"] != r["machine_type"]:
            report["machine_type_conflicts"] += 1
    for r in legacy:
        m = machines[r["machine_id"].lower()]
        conn.execute(
            """INSERT INTO predictions (id, machine_id, created_at, source, machine_type, air_temp, process_temp, rpm, torque,
                   tool_wear, probability, risk_level, severity, health_score, model_key, model_label, model_version,
                   recommendation, details_json)
               VALUES (?,?,?,'migrated',?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)""",
            (r["id"], m["id"], r["created_at"], r["machine_type"], r["air_temp"], r["process_temp"], r["rpm"], r["torque"],
             r["tool_wear"], r["probability"], r["risk_level"], r["severity"], round(100 * (1 - r["probability"])),
             r["model_key"], r["model_label"], r["model_version"], r["recommendation"]))
        reason = f"{r['risk_level']}: predicted failure probability {r['probability'] * 100:.1f}% ({r['model_label']}). Migrated from the v1 database."
        res = db.apply_alert_lifecycle(conn, m["id"], r["id"], r["risk_level"], r["recommendation"], reason, r["created_at"])
        if res["action"] == "opened":
            report["alerts_created"] += 1
        if r["alert_status"] == "acknowledged" and res["alert_id"] and r["risk_level"] != "Normal":
            if db.acknowledge_alert(conn, res["alert_id"], "Carried over from the v1 acknowledgement.", r["created_at"], commit=False):
                report["acknowledged_carried_over"] += 1
    report["predictions"] = len(legacy)
    report["machines"] = len(machines)
    new_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if new_count != len(legacy):
        raise RuntimeError(f"Migration check failed: {len(legacy)} v1 rows but {new_count} v2 rows")
    report["alerts_resolved"] = conn.execute("SELECT COUNT(*) FROM alerts WHERE status='resolved'").fetchone()[0]
    report["alerts_active"] = conn.execute("SELECT COUNT(*) FROM alerts WHERE status IN ('open','acknowledged')").fetchone()[0]
    return report


def ensure_schema(path, dry_run=False):
    """Create or migrate the database at `path`. Returns {'status', 'report', 'backup'}."""
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # explicit transactions: DDL + data changes commit or roll back together
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = _tables(conn)
        if version == db.SCHEMA_VERSION:
            return {"status": "current", "report": None, "backup": None}
        if not tables:
            if dry_run:
                return {"status": "would_create", "report": None, "backup": None}
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN")
            for stmt in db.SCHEMA_STATEMENTS:
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION}")
            conn.execute("COMMIT")
            return {"status": "created", "report": None, "backup": None}
        if "predictions" in tables and "machines" not in tables:
            if dry_run:
                n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                return {"status": "would_migrate", "report": {"predictions": n}, "backup": None}
            backup = _backup(path) if exists else None
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN")
            try:
                report = migrate_v1_to_v2(conn)
                conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            conn.execute("PRAGMA foreign_keys = ON")
            return {"status": "migrated", "report": report, "backup": backup}
        raise RuntimeError(f"{path}: unrecognised database schema (version {version}, tables {sorted(tables)}). "
                           "Refusing to modify it.")
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create or migrate the application database.")
    ap.add_argument("--db", default=os.environ.get("DATABASE_PATH", "instance/predictions.db"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.db)), exist_ok=True)
    print(ensure_schema(a.db, dry_run=a.dry_run))
