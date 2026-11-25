"""SQLite storage (standard-library sqlite3, no ORM). Schema version 2.

Tables
  machines      one row per physical machine; identity is `machine_code` (case-insensitive, unique).
  predictions   one row per analysis; IMMUTABLE (database triggers forbid UPDATE and DELETE), so history never changes.
  alerts        at most ONE active (open / acknowledged) alert per machine, enforced by a partial unique index.
  alert_events  append-only history of everything that happened to an alert.

Alert lifecycle (see docs/ARCHITECTURE.md)
  Warning/Critical reading, no active alert   -> new alert (open)
  Warning/Critical reading, active alert      -> same alert: occurrences + 1; a severity rise re-opens an acknowledged alert
  Normal reading, active alert                -> alert resolved automatically (resolution = auto_normal_reading)
  User actions                                -> Acknowledge (open -> acknowledged), Resolve (open/acknowledged -> resolved)
"""
import json
import sqlite3
from datetime import datetime

SCHEMA_VERSION = 2
RANK = {"Normal": 0, "Warning": 1, "Critical": 2}

SCHEMA_STATEMENTS = [
    """CREATE TABLE machines (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
        machine_type TEXT NOT NULL CHECK (machine_type IN ('L','M','H')),
        created_at   TEXT NOT NULL
    )""",
    """CREATE TABLE predictions (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id     INTEGER NOT NULL REFERENCES machines(id),
        created_at     TEXT NOT NULL,                 -- local time 'YYYY-MM-DD HH:MM:SS'
        source         TEXT NOT NULL DEFAULT 'form' CHECK (source IN ('form','api','batch','migrated')),
        machine_type   TEXT NOT NULL CHECK (machine_type IN ('L','M','H')),   -- type used as model input
        air_temp       REAL NOT NULL, process_temp REAL NOT NULL, rpm REAL NOT NULL,
        torque         REAL NOT NULL, tool_wear    REAL NOT NULL,
        probability    REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
        risk_level     TEXT NOT NULL CHECK (risk_level IN ('Normal','Warning','Critical')),
        severity       INTEGER NOT NULL CHECK (severity IN (0,1,2)),
        health_score   INTEGER NOT NULL CHECK (health_score BETWEEN 0 AND 100),
        model_key      TEXT NOT NULL, model_label TEXT NOT NULL, model_version TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        details_json   TEXT                           -- snapshot: indicators, contributions, mode estimates, warnings
    )""",
    """CREATE TABLE alerts (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id            INTEGER NOT NULL REFERENCES machines(id),
        status                TEXT NOT NULL CHECK (status IN ('open','acknowledged','resolved')),
        opened_at             TEXT NOT NULL,
        opening_prediction_id INTEGER NOT NULL REFERENCES predictions(id),
        latest_prediction_id  INTEGER NOT NULL REFERENCES predictions(id),
        current_level         TEXT NOT NULL CHECK (current_level IN ('Warning','Critical')),
        peak_level            TEXT NOT NULL CHECK (peak_level IN ('Warning','Critical')),
        occurrences           INTEGER NOT NULL DEFAULT 1,
        reason                TEXT NOT NULL,
        recommendation        TEXT NOT NULL,
        acknowledged_at       TEXT,
        resolved_at           TEXT,
        resolution            TEXT CHECK (resolution IN ('manual','auto_normal_reading'))
    )""",
    """CREATE TABLE alert_events (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id      INTEGER NOT NULL REFERENCES alerts(id),
        created_at    TEXT NOT NULL,
        event_type    TEXT NOT NULL CHECK (event_type IN
                      ('opened','repeat','escalated','downgraded','acknowledged','resolved','auto_resolved')),
        prediction_id INTEGER REFERENCES predictions(id),
        note          TEXT
    )""",
    "CREATE UNIQUE INDEX uq_one_active_alert_per_machine ON alerts(machine_id) WHERE status IN ('open','acknowledged')",
    "CREATE INDEX ix_predictions_machine_time ON predictions(machine_id, created_at)",
    "CREATE INDEX ix_predictions_created ON predictions(created_at)",
    "CREATE INDEX ix_predictions_risk ON predictions(risk_level)",
    "CREATE INDEX ix_alerts_status ON alerts(status, current_level)",
    "CREATE INDEX ix_alert_events_alert ON alert_events(alert_id)",
    """CREATE TRIGGER trg_predictions_no_update BEFORE UPDATE ON predictions
       BEGIN SELECT RAISE(ABORT, 'saved predictions are immutable'); END""",
    """CREATE TRIGGER trg_predictions_no_delete BEFORE DELETE ON predictions
       BEGIN SELECT RAISE(ABORT, 'saved predictions cannot be deleted'); END""",
    """CREATE TRIGGER trg_alert_events_no_update BEFORE UPDATE ON alert_events
       BEGIN SELECT RAISE(ABORT, 'alert history is append-only'); END""",
    """CREATE TRIGGER trg_alert_events_no_delete BEFORE DELETE ON alert_events
       BEGIN SELECT RAISE(ABORT, 'alert history is append-only'); END""",
    """CREATE TRIGGER trg_machines_identity BEFORE UPDATE OF machine_code, machine_type ON machines
       BEGIN SELECT RAISE(ABORT, 'machine identity is immutable'); END""",
]


class MachineTypeConflict(Exception):
    def __init__(self, code, existing_type):
        super().__init__(f"Machine {code} is registered as type {existing_type}.")
        self.code, self.existing_type = code, existing_type


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(path):
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path):
    """Create the database, or migrate an older one (with a backup). Safe to call at every start-up."""
    from app import migrations
    return migrations.ensure_schema(path)


# ----------------------------------------------------------------------------- machines
def get_machine(conn, code):
    return conn.execute("SELECT * FROM machines WHERE machine_code = ? COLLATE NOCASE", (code,)).fetchone()


def get_or_create_machine(conn, code, machine_type, now=None):
    """A machine keeps ONE identity: the same code always returns the same row. A different type is refused."""
    row = get_machine(conn, code)
    if row is not None:
        if row["machine_type"] != machine_type:
            raise MachineTypeConflict(row["machine_code"], row["machine_type"])
        return row, False
    cur = conn.execute("INSERT INTO machines (machine_code, machine_type, created_at) VALUES (?,?,?)",
                       (code, machine_type, now or now_str()))
    return conn.execute("SELECT * FROM machines WHERE id = ?", (cur.lastrowid,)).fetchone(), True


# ----------------------------------------------------------------------------- alert lifecycle
def _event(conn, alert_id, event_type, now, prediction_id=None, note=None):
    conn.execute("INSERT INTO alert_events (alert_id, created_at, event_type, prediction_id, note) VALUES (?,?,?,?,?)",
                 (alert_id, now, event_type, prediction_id, note))


def active_alert(conn, machine_id):
    return conn.execute("SELECT * FROM alerts WHERE machine_id = ? AND status IN ('open','acknowledged')",
                        (machine_id,)).fetchone()


def apply_alert_lifecycle(conn, machine_id, prediction_id, level, recommendation, reason, now):
    """Update the machine's alert after a new prediction. Returns {'action': ..., 'alert_id': ...}."""
    active = active_alert(conn, machine_id)
    if level == "Normal":
        if active is None:
            return {"action": "none", "alert_id": None}
        conn.execute("UPDATE alerts SET status='resolved', resolved_at=?, resolution='auto_normal_reading' WHERE id=?",
                     (now, active["id"]))
        _event(conn, active["id"], "auto_resolved", now, prediction_id, "A later reading was Normal.")
        return {"action": "auto_resolved", "alert_id": active["id"]}

    if active is None:
        cur = conn.execute(
            """INSERT INTO alerts (machine_id, status, opened_at, opening_prediction_id, latest_prediction_id,
                                   current_level, peak_level, occurrences, reason, recommendation)
               VALUES (?, 'open', ?, ?, ?, ?, ?, 1, ?, ?)""",
            (machine_id, now, prediction_id, prediction_id, level, level, reason, recommendation))
        _event(conn, cur.lastrowid, "opened", now, prediction_id, reason)
        return {"action": "opened", "alert_id": cur.lastrowid}

    prev, aid = active["current_level"], active["id"]
    peak = level if RANK[level] > RANK[active["peak_level"]] else active["peak_level"]
    if RANK[level] > RANK[prev]:
        conn.execute("""UPDATE alerts SET status='open', acknowledged_at=NULL, current_level=?, peak_level=?,
                        occurrences=occurrences+1, latest_prediction_id=?, reason=?, recommendation=? WHERE id=?""",
                     (level, peak, prediction_id, reason, recommendation, aid))
        _event(conn, aid, "escalated", now, prediction_id,
               f"Severity rose from {prev} to {level}." + (" The alert was re-opened." if active["status"] == "acknowledged" else ""))
        return {"action": "escalated", "alert_id": aid}
    if RANK[level] < RANK[prev]:
        conn.execute("""UPDATE alerts SET current_level=?, occurrences=occurrences+1, latest_prediction_id=?, reason=?,
                        recommendation=? WHERE id=?""", (level, prediction_id, reason, recommendation, aid))
        _event(conn, aid, "downgraded", now, prediction_id, f"Severity fell from {prev} to {level}.")
        return {"action": "downgraded", "alert_id": aid}
    conn.execute("UPDATE alerts SET occurrences=occurrences+1, latest_prediction_id=? WHERE id=?", (prediction_id, aid))
    _event(conn, aid, "repeat", now, prediction_id, f"Another {level} reading.")
    return {"action": "repeat", "alert_id": aid}


def acknowledge_alert(conn, alert_id, note=None, now=None, commit=True):
    now = now or now_str()
    cur = conn.execute("UPDATE alerts SET status='acknowledged', acknowledged_at=? WHERE id=? AND status='open'", (now, alert_id))
    if cur.rowcount != 1:
        return False
    _event(conn, alert_id, "acknowledged", now, None, note)
    if commit:
        conn.commit()
    return True


def resolve_alert(conn, alert_id, note=None, now=None, commit=True):
    now = now or now_str()
    cur = conn.execute("""UPDATE alerts SET status='resolved', resolved_at=?, resolution='manual'
                          WHERE id=? AND status IN ('open','acknowledged')""", (now, alert_id))
    if cur.rowcount != 1:
        return False
    _event(conn, alert_id, "resolved", now, None, note)
    if commit:
        conn.commit()
    return True


# ----------------------------------------------------------------------------- predictions
def _pct(p):
    """Same display rule as the pct filter: never print a misleading exact 0% or 100%."""
    return "<0.1%" if p < 0.0005 else (">99.9%" if p > 0.9995 else f"{p * 100:.1f}%")


def alert_reason(result):
    text = f"{result['risk_level']}: predicted failure probability {_pct(result['probability'])} ({result['model_label']})."
    codes = [i["code"] for i in result.get("details", {}).get("indicators", [])]
    return text + (f" Rule-based indicators: {', '.join(codes)}." if codes else "")


def record_prediction(conn, *, machine_code, machine_type, readings, result, source="form", now=None, commit=True):
    """Save one analysis and update the machine's alert, in a single transaction."""
    now = now or now_str()
    try:
        machine, created = get_or_create_machine(conn, machine_code, machine_type, now)
        cur = conn.execute(
            """INSERT INTO predictions (machine_id, created_at, source, machine_type, air_temp, process_temp, rpm, torque,
                   tool_wear, probability, risk_level, severity, health_score, model_key, model_label, model_version,
                   recommendation, details_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (machine["id"], now, source, machine_type, readings["air_temp"], readings["process_temp"], readings["rpm"],
             readings["torque"], readings["tool_wear"], result["probability"], result["risk_level"], result["severity"],
             result["health_score"], result["model_key"], result["model_label"], result["model_version"],
             result["recommendation"], json.dumps(result.get("details", {}))))
        pid = cur.lastrowid
        alert = apply_alert_lifecycle(conn, machine["id"], pid, result["risk_level"], result["recommendation"],
                                      alert_reason(result), now)
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"prediction_id": pid, "machine_id": machine["id"], "machine_code": machine["machine_code"],
            "machine_created": created, "alert": alert}


SORT_COLUMNS = {"created_at": "p.created_at", "machine": "m.machine_code", "probability": "p.probability",
                "risk": "p.severity", "model": "p.model_label", "health": "p.health_score"}
_SELECT_PRED = "SELECT p.*, m.machine_code FROM predictions p JOIN machines m ON m.id = p.machine_id"


def _where(filters):
    clauses, params = [], []
    f = filters or {}
    if f.get("risk"):
        clauses.append(f"p.risk_level IN ({','.join('?' * len(f['risk']))})")
        params += list(f["risk"])
    if f.get("model"):
        clauses.append("p.model_key = ?")
        params.append(f["model"])
    if f.get("machine"):
        clauses.append("m.machine_code LIKE ?")
        params.append(f"%{f['machine']}%")
    if f.get("machine_exact"):
        clauses.append("m.machine_code = ? COLLATE NOCASE")
        params.append(f["machine_exact"])
    if f.get("date_from"):
        clauses.append("date(p.created_at) >= date(?)")
        params.append(f["date_from"])
    if f.get("date_to"):
        clauses.append("date(p.created_at) <= date(?)")
        params.append(f["date_to"])
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def list_predictions(conn, filters=None, sort="created_at", direction="desc", limit=25, offset=0):
    where, params = _where(filters)
    col = SORT_COLUMNS.get(sort, "p.created_at")  # whitelist: user input never reaches the SQL text
    order = "ASC" if direction == "asc" else "DESC"
    return conn.execute(f"{_SELECT_PRED}{where} ORDER BY {col} {order}, p.id {order} LIMIT ? OFFSET ?",
                        params + [limit, offset]).fetchall()


def count_predictions(conn, filters=None):
    where, params = _where(filters)
    return conn.execute(f"SELECT COUNT(*) FROM predictions p JOIN machines m ON m.id = p.machine_id{where}", params).fetchone()[0]


def previous_prediction(conn, machine_id, pid):
    """The machine's analysis made just before prediction `pid` (for 'change since last analysis')."""
    return conn.execute("""SELECT id, created_at, probability, risk_level, health_score FROM predictions
                           WHERE machine_id = ? AND id < ? ORDER BY id DESC LIMIT 1""", (machine_id, pid)).fetchone()


def get_prediction(conn, pid):
    return conn.execute(f"{_SELECT_PRED} WHERE p.id = ?", (pid,)).fetchone()


EXPORT_COLUMNS = [  # (csv header, SQL expression)
    ("Timestamp", "p.created_at"), ("Machine ID", "m.machine_code"), ("Machine Type", "p.machine_type"),
    ("Probability", "p.probability"), ("Alert Level", "p.risk_level"), ("Severity", "p.severity"),
    ("Health Score", "p.health_score"), ("Air Temperature [K]", "p.air_temp"), ("Process Temperature [K]", "p.process_temp"),
    ("Rotational Speed [rpm]", "p.rpm"), ("Torque [Nm]", "p.torque"), ("Tool Wear [min]", "p.tool_wear"),
    ("Model", "p.model_label"), ("Model Version", "p.model_version"), ("Recommendation", "p.recommendation"),
    ("Source", "p.source"), ("Details", "p.details_json"),
]


def export_rows(conn, filters=None):
    where, params = _where(filters)
    cols = ", ".join(f"{expr} AS c{i}" for i, (_, expr) in enumerate(EXPORT_COLUMNS))
    return conn.execute(f"SELECT {cols} FROM predictions p JOIN machines m ON m.id = p.machine_id{where} ORDER BY p.id DESC",
                        params).fetchall()


# ----------------------------------------------------------------------------- machines (queries)
_MACHINE_SELECT = """
SELECT m.id, m.machine_code, m.machine_type, m.created_at,
       (SELECT COUNT(*) FROM predictions WHERE machine_id = m.id) AS n_predictions,
       lp.id AS last_prediction_id, lp.created_at AS last_analysed, lp.probability, lp.risk_level, lp.severity, lp.health_score,
       a.id AS alert_id, a.status AS alert_status, a.current_level AS alert_level
FROM machines m
LEFT JOIN predictions lp ON lp.id = (SELECT MAX(id) FROM predictions WHERE machine_id = m.id)
LEFT JOIN alerts a ON a.machine_id = m.id AND a.status IN ('open','acknowledged')"""


def list_machines(conn, search=None, attention_only=False):
    clauses, params = [], []
    if search:
        clauses.append("m.machine_code LIKE ?")
        params.append(f"%{search}%")
    if attention_only:
        clauses.append("lp.severity >= 1")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return conn.execute(f"{_MACHINE_SELECT}{where} ORDER BY lp.severity DESC, lp.created_at DESC, m.machine_code", params).fetchall()


def machine_detail(conn, code):
    m = conn.execute(f"{_MACHINE_SELECT} WHERE m.machine_code = ? COLLATE NOCASE", (code,)).fetchone()
    if m is None:
        return None
    trend = conn.execute("""SELECT id, created_at, probability, risk_level, health_score, model_label FROM predictions
                            WHERE machine_id = ? ORDER BY id DESC LIMIT 200""", (m["id"],)).fetchall()[::-1]
    return {"machine": m, "trend": [dict(r) for r in trend]}


def dashboard_stats(conn):
    """Every number on the dashboard is computed here from stored data; nothing is pre-filled."""
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    s = {"total_predictions": total, "machines": 0, "status": {"Normal": 0, "Warning": 0, "Critical": 0},
         "attention": 0, "avg_health": None, "active_alerts": 0, "open_alerts": 0, "attention_machines": [], "recent": []}
    if total == 0:
        return s
    machines = list_machines(conn)
    s["machines"] = len(machines)
    healths = []
    for m in machines:
        if m["risk_level"]:
            s["status"][m["risk_level"]] += 1
            healths.append(m["health_score"])
    s["attention"] = s["status"]["Warning"] + s["status"]["Critical"]
    s["avg_health"] = round(sum(healths) / len(healths)) if healths else None
    s["attention_machines"] = [dict(m) for m in machines if (m["severity"] or 0) >= 1][:8]
    c = alert_counts(conn)
    s["active_alerts"], s["open_alerts"] = c["open"] + c["acknowledged"], c["open"]
    s["recent"] = [dict(r) for r in list_predictions(conn, limit=8)]
    return s


# ----------------------------------------------------------------------------- alerts (queries)
_ALERT_SELECT = """SELECT a.*, m.machine_code, m.machine_type, lp.probability AS latest_probability, lp.created_at AS latest_at
FROM alerts a JOIN machines m ON m.id = a.machine_id JOIN predictions lp ON lp.id = a.latest_prediction_id"""


def _alert_where(view):
    if view == "active":
        return " WHERE a.status IN ('open','acknowledged')"
    if view == "resolved":
        return " WHERE a.status = 'resolved'"
    return ""


def list_alerts(conn, view="active", limit=25, offset=0):
    order = "a.resolved_at DESC, a.id DESC" if view == "resolved" else \
        "(a.status = 'open') DESC, (a.current_level = 'Critical') DESC, a.opened_at DESC, a.id DESC"
    return conn.execute(f"{_ALERT_SELECT}{_alert_where(view)} ORDER BY {order} LIMIT ? OFFSET ?", (limit, offset)).fetchall()


def count_alerts(conn, view="active"):
    return conn.execute(f"SELECT COUNT(*) FROM alerts a{_alert_where(view)}").fetchone()[0]


def alert_counts(conn):
    r = conn.execute("""SELECT COALESCE(SUM(status='open'),0) AS open, COALESCE(SUM(status='acknowledged'),0) AS acknowledged,
                        COALESCE(SUM(status='resolved'),0) AS resolved,
                        COALESCE(SUM(status='open' AND current_level='Critical'),0) AS open_critical,
                        COALESCE(SUM(status='open' AND current_level='Warning'),0) AS open_warning FROM alerts""").fetchone()
    return dict(r)


def get_alert(conn, alert_id):
    return conn.execute(f"{_ALERT_SELECT} WHERE a.id = ?", (alert_id,)).fetchone()


def alert_events(conn, alert_id):
    return conn.execute("""SELECT e.*, p.probability, p.risk_level FROM alert_events e
                           LEFT JOIN predictions p ON p.id = e.prediction_id WHERE e.alert_id = ? ORDER BY e.id""",
                        (alert_id,)).fetchall()
