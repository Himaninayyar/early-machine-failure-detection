"""Safe upgrade of an existing v1 database."""
import glob
import os
import sqlite3

import pytest

from app import db, migrations
from conftest import make_app


def legacy_db(path):
    c = sqlite3.connect(path)
    c.executescript(migrations.LEGACY_V1_SCHEMA)

    def add(ts, mid, typ, prob, lvl, sev, status):
        c.execute("""INSERT INTO predictions(created_at,machine_id,machine_type,air_temp,process_temp,rpm,torque,tool_wear,probability,
                     risk_level,severity,model_key,model_label,model_version,failure_type,recommendation,alert_status)
                     VALUES (?,?,?,300,310,1500,40,100,?,?,?,'rf','Random Forest','v1',NULL,'rec',?)""", (ts, mid, typ, prob, lvl, sev, status))
    add("2026-11-01 10:00:00", "PUMP-9", "L", 0.95, "Critical", 2, "open")
    add("2026-11-01 10:05:00", "PUMP-9", "L", 0.97, "Critical", 2, "open")       # duplicate open alert in v1
    add("2026-11-01 10:10:00", "pump-9", "L", 0.02, "Normal", 0, "none")          # same machine, different case; resolves the alert
    add("2026-11-02 09:00:00", "AUTO-0001", "M", 0.50, "Warning", 1, "acknowledged")
    add("2026-11-02 09:05:00", "AUTO-0002", "M", 0.40, "Warning", 1, "open")
    add("2026-11-03 09:00:00", "LATHE", "L", 0.10, "Normal", 0, "none")
    add("2026-11-03 09:30:00", "LATHE", "H", 0.80, "Critical", 2, "open")         # same machine submitted with another type
    c.commit()
    c.close()
    return path


@pytest.fixture()
def legacy(tmp_path):
    return legacy_db(str(tmp_path / "legacy.db"))


def test_dry_run_changes_nothing(legacy):
    before = open(legacy, "rb").read()
    out = migrations.ensure_schema(legacy, dry_run=True)
    assert out["status"] == "would_migrate" and out["report"]["predictions"] == 7 and open(legacy, "rb").read() == before


def test_migration_preserves_data_and_reports(legacy):
    out = migrations.ensure_schema(legacy)
    assert out["status"] == "migrated" and os.path.exists(out["backup"])
    assert out["report"] == {"predictions": 7, "machines": 4, "machine_type_conflicts": 1, "alerts_created": 4, "alerts_resolved": 1,
                             "alerts_active": 3, "acknowledged_carried_over": 1}
    c = db.connect(legacy)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert [r[0] for r in c.execute("SELECT id FROM predictions ORDER BY id")] == list(range(1, 8))      # ids preserved
    assert c.execute("SELECT COUNT(*) FROM legacy_predictions_v1").fetchone()[0] == 7                     # original table kept
    assert c.execute("PRAGMA foreign_key_check").fetchall() == []
    m = {r["machine_code"]: r["machine_type"] for r in c.execute("SELECT * FROM machines")}
    assert m == {"PUMP-9": "L", "AUTO-0001": "M", "AUTO-0002": "M", "LATHE": "L"}                         # case-insensitive merge, first type wins
    lathe = c.execute("SELECT machine_type FROM predictions WHERE id = 7").fetchone()[0]
    assert lathe == "H"                                                                                   # the type each prediction was run with is kept
    r = c.execute("SELECT * FROM predictions WHERE id = 1").fetchone()
    assert (r["source"], r["health_score"], r["probability"], r["model_label"]) == ("migrated", 5, 0.95, "Random Forest")


def test_migrated_alerts_follow_the_new_lifecycle(legacy):
    migrations.ensure_schema(legacy)
    c = db.connect(legacy)
    rows = {r["machine_code"]: r for r in c.execute("SELECT a.*, m.machine_code FROM alerts a JOIN machines m ON m.id = a.machine_id")}
    assert (rows["PUMP-9"]["status"], rows["PUMP-9"]["occurrences"], rows["PUMP-9"]["resolution"]) == ("resolved", 2, "auto_normal_reading")   # duplicates collapsed, then resolved
    assert rows["AUTO-0001"]["status"] == "acknowledged" and rows["AUTO-0002"]["status"] == "open" and rows["LATHE"]["status"] == "open"
    assert c.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0] >= 6                             # history recorded


def test_migration_is_idempotent_and_result_is_immutable(legacy):
    migrations.ensure_schema(legacy)
    assert migrations.ensure_schema(legacy)["status"] == "current"
    c = db.connect(legacy)
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        c.execute("UPDATE predictions SET probability = 0 WHERE id = 1")
    assert len(glob.glob(legacy + ".v1-backup-*")) == 1


def test_failed_migration_rolls_back_and_leaves_the_database_usable(legacy, monkeypatch):
    def boom(conn):
        conn.execute("ALTER TABLE predictions RENAME TO legacy_predictions_v1")
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(migrations, "migrate_v1_to_v2", boom)
    with pytest.raises(RuntimeError):
        migrations.ensure_schema(legacy)
    c = sqlite3.connect(legacy)
    assert c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 7        # rolled back: original table intact
    assert c.execute("PRAGMA user_version").fetchone()[0] == 0


def test_unrecognised_schema_is_never_modified(tmp_path):
    p = str(tmp_path / "other.db")
    c = sqlite3.connect(p); c.execute("CREATE TABLE something_else (x)"); c.commit(); c.close()
    with pytest.raises(RuntimeError, match="unrecognised"):
        migrations.ensure_schema(p)


def test_app_starts_on_a_v1_database_and_serves_migrated_history(tiny_models, legacy):
    app = make_app(tiny_models, legacy)
    c = app.test_client()
    assert app.extensions["db_error"] is None
    assert c.get("/history").status_code == 200 and "PUMP-9" in c.get("/history").get_data(as_text=True)
    assert c.get("/machines").status_code == 200 and c.get("/alerts").status_code == 200
    page = c.get("/predictions/1")
    assert page.status_code == 200 and "No explanation was stored" in page.get_data(as_text=True)    # honest about missing snapshot
