"""Schema, machine identity, alert lifecycle, history immutability, queries."""
import sqlite3

import pytest

from app import db, migrations

READ = dict(air_temp=300.0, process_temp=310.0, rpm=1500.0, torque=40.0, tool_wear=100.0)


@pytest.fixture()
def conn(tmp_path):
    path = str(tmp_path / "t.db")
    assert db.init_db(path)["status"] == "created"
    c = db.connect(path)
    yield c
    c.close()


def result(p, level=None, model="rf", details=None):
    level = level or ("Critical" if p >= 0.7 else "Warning" if p >= 0.3 else "Normal")
    return {"probability": p, "risk_level": level, "severity": db.RANK[level], "health_score": round(100 * (1 - p)), "model_key": model,
            "model_label": "Random Forest", "model_version": "v1", "recommendation": f"{level} rec", "details": details or {}}


def add(conn, code="M-1", p=0.1, mtype="L", ts="2026-11-01 10:00:00", **kw):
    return db.record_prediction(conn, machine_code=code, machine_type=mtype, readings=READ, result=result(p, **kw), now=ts)


def alerts(conn, code=None):
    sql = "SELECT a.* FROM alerts a JOIN machines m ON m.id = a.machine_id" + (" WHERE m.machine_code = ?" if code else "")
    return conn.execute(sql, (code,) if code else ()).fetchall()


# ---------------------------------------------------------------- schema
def test_fresh_database_has_version_foreign_keys_and_indexes(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    idx = {r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type='index'")}
    assert {"uq_one_active_alert_per_machine", "ix_predictions_machine_time", "ix_alerts_status"} <= idx
    assert db.init_db(conn.execute("PRAGMA database_list").fetchone()[2])["status"] == "current"


def test_foreign_keys_are_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("""INSERT INTO predictions (machine_id, created_at, machine_type, air_temp, process_temp, rpm, torque, tool_wear,
            probability, risk_level, severity, health_score, model_key, model_label, model_version, recommendation)
            VALUES (999,'2026-01-01 00:00:00','L',1,1,1,1,1,0.5,'Warning',1,50,'rf','RF','v','r')""")


def test_check_constraints_reject_bad_values(conn):
    bad = {**result(0.5), "risk_level": "Bogus"}
    with pytest.raises(sqlite3.IntegrityError):
        db.record_prediction(conn, machine_code="A", machine_type="L", readings=READ, result=bad)
    assert conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0           # the whole transaction was rolled back


# ---------------------------------------------------------------- machine identity
def test_same_machine_id_keeps_one_identity_across_predictions(conn):
    a, b, c = add(conn, "PUMP-9"), add(conn, "PUMP-9"), add(conn, "pump-9")       # case-insensitive
    assert a["machine_id"] == b["machine_id"] == c["machine_id"]
    assert a["machine_created"] and not b["machine_created"]
    assert conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 3


def test_machine_type_is_part_of_identity(conn):
    add(conn, "M-1", mtype="L")
    with pytest.raises(db.MachineTypeConflict) as e:
        add(conn, "M-1", mtype="H")
    assert e.value.existing_type == "L"
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1         # rolled back, nothing half-saved


def test_machine_identity_cannot_be_edited(conn):
    add(conn, "M-1")
    for sql in ("UPDATE machines SET machine_code='X'", "UPDATE machines SET machine_type='H'"):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute(sql)


# ---------------------------------------------------------------- history immutability
def test_saved_predictions_cannot_be_changed_or_deleted(conn):
    pid = add(conn, p=0.95)["prediction_id"]
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        conn.execute("UPDATE predictions SET probability = 0.01 WHERE id = ?", (pid,))
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        conn.execute("DELETE FROM predictions WHERE id = ?", (pid,))
    assert db.get_prediction(conn, pid)["probability"] == 0.95


def test_prediction_preserves_every_required_field(conn):
    pid = add(conn, "M-7", p=0.55, details={"indicators": [{"code": "HDF"}]}, ts="2026-11-05 09:30:00")["prediction_id"]
    r = db.get_prediction(conn, pid)
    assert (r["machine_code"], r["machine_type"], r["created_at"], r["probability"], r["risk_level"]) == ("M-7", "L", "2026-11-05 09:30:00", 0.55, "Warning")
    assert (r["air_temp"], r["rpm"], r["tool_wear"], r["model_label"], r["model_version"], r["recommendation"]) == (300.0, 1500.0, 100.0, "Random Forest", "v1", "Warning rec")
    assert "HDF" in r["details_json"] and r["health_score"] == 45


# ---------------------------------------------------------------- alert lifecycle
def test_warning_or_critical_opens_one_alert_normal_does_not(conn):
    add(conn, "A", 0.05)
    assert alerts(conn) == []
    r = add(conn, "A", 0.5)
    assert r["alert"]["action"] == "opened" and len(alerts(conn)) == 1
    a = alerts(conn)[0]
    assert (a["status"], a["current_level"], a["occurrences"]) == ("open", "Warning", 1)


def test_repeated_readings_do_not_create_duplicate_alerts(conn):
    add(conn, "A", 0.5); add(conn, "A", 0.55); add(conn, "A", 0.6)
    a = alerts(conn)
    assert len(a) == 1 and a[0]["occurrences"] == 3
    assert [e["event_type"] for e in db.alert_events(conn, a[0]["id"])] == ["opened", "repeat", "repeat"]


def test_database_itself_refuses_a_second_active_alert(conn):
    add(conn, "A", 0.5)
    mid = conn.execute("SELECT id FROM machines").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("""INSERT INTO alerts (machine_id, status, opened_at, opening_prediction_id, latest_prediction_id, current_level,
                        peak_level, reason, recommendation) VALUES (?,'open','t',1,1,'Warning','Warning','r','r')""", (mid,))


def test_escalation_reopens_an_acknowledged_alert(conn):
    add(conn, "A", 0.5)
    aid = alerts(conn)[0]["id"]
    assert db.acknowledge_alert(conn, aid, "looking into it")
    assert alerts(conn)[0]["status"] == "acknowledged"
    add(conn, "A", 0.55)                                              # same severity: stays acknowledged
    assert alerts(conn)[0]["status"] == "acknowledged"
    r = add(conn, "A", 0.9)                                           # severity rises: back to open
    a = alerts(conn)[0]
    assert r["alert"]["action"] == "escalated" and (a["status"], a["current_level"], a["peak_level"]) == ("open", "Critical", "Critical")
    assert a["acknowledged_at"] is None


def test_downgrade_is_recorded_and_keeps_peak(conn):
    add(conn, "A", 0.9); r = add(conn, "A", 0.4)
    a = alerts(conn)[0]
    assert r["alert"]["action"] == "downgraded" and (a["current_level"], a["peak_level"], a["status"]) == ("Warning", "Critical", "open")


def test_later_normal_reading_resolves_the_alert_and_a_new_problem_opens_a_new_one(conn):
    add(conn, "A", 0.9)
    r = add(conn, "A", 0.02)
    a = alerts(conn)[0]
    assert r["alert"]["action"] == "auto_resolved" and (a["status"], a["resolution"]) == ("resolved", "auto_normal_reading") and a["resolved_at"]
    add(conn, "A", 0.8)
    rows = alerts(conn, "A")
    assert len(rows) == 2 and [x["status"] for x in rows] == ["resolved", "open"]           # history preserved, new alert for the new episode
    assert db.alert_counts(conn) == {"open": 1, "acknowledged": 0, "resolved": 1, "open_critical": 1, "open_warning": 0}


def test_manual_acknowledge_and_resolve_rules(conn):
    add(conn, "A", 0.9)
    aid = alerts(conn)[0]["id"]
    assert db.acknowledge_alert(conn, aid) and not db.acknowledge_alert(conn, aid)          # only once
    assert db.resolve_alert(conn, aid, "fan replaced") and not db.resolve_alert(conn, aid)
    assert not db.acknowledge_alert(conn, aid)                                              # resolved alerts cannot be acknowledged
    ev = db.alert_events(conn, aid)
    assert [e["event_type"] for e in ev] == ["opened", "acknowledged", "resolved"] and ev[-1]["note"] == "fan replaced"


def test_alert_history_is_append_only(conn):
    add(conn, "A", 0.9)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE alert_events SET note='x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM alert_events")


def test_alerts_of_different_machines_are_independent(conn):
    add(conn, "A", 0.9); add(conn, "B", 0.5); add(conn, "A", 0.01)
    assert {a["machine_id"]: a["status"] for a in alerts(conn)} == {1: "resolved", 2: "open"}


# ---------------------------------------------------------------- queries
def test_dashboard_is_empty_then_reflects_latest_status_per_machine(conn):
    s = db.dashboard_stats(conn)
    assert s["total_predictions"] == 0 and s["machines"] == 0 and s["recent"] == [] and s["avg_health"] is None
    add(conn, "A", 0.9); add(conn, "A", 0.05); add(conn, "B", 0.5); add(conn, "C", 0.1)
    s = db.dashboard_stats(conn)
    assert (s["machines"], s["total_predictions"], s["attention"]) == (3, 4, 1)
    assert s["status"] == {"Normal": 2, "Warning": 1, "Critical": 0}
    assert s["active_alerts"] == 1 and s["open_alerts"] == 1 and s["avg_health"] == round((95 + 50 + 90) / 3)
    assert [m["machine_code"] for m in s["attention_machines"]] == ["B"]


def test_filters_sorting_and_pagination(conn):
    add(conn, "A", 0.9, ts="2026-11-01 10:00:00"); add(conn, "B", 0.5, ts="2026-11-05 10:00:00"); add(conn, "C", 0.1, ts="2026-11-10 10:00:00")
    assert db.count_predictions(conn, {"risk": ["Critical", "Warning"]}) == 2
    assert db.count_predictions(conn, {"machine": "B"}) == 1 and db.count_predictions(conn, {"machine_exact": "b"}) == 1
    assert db.count_predictions(conn, {"date_from": "2026-11-04", "date_to": "2026-11-06"}) == 1
    assert [r["machine_code"] for r in db.list_predictions(conn, sort="probability", direction="desc")] == ["A", "B", "C"]
    assert len(db.list_predictions(conn, limit=2)) == 2 and len(db.list_predictions(conn, limit=2, offset=2)) == 1


def test_user_input_never_reaches_sql_text(conn):
    add(conn, "A", 0.9)
    db.list_predictions(conn, sort="id; DROP TABLE predictions", direction="desc; DROP TABLE x")
    assert db.count_predictions(conn, {"machine": "' OR 1=1 --"}) == 0
    assert db.count_predictions(conn) == 1


def test_machine_list_and_detail(conn):
    add(conn, "A", 0.9, ts="2026-11-01 10:00:00"); add(conn, "A", 0.4, ts="2026-11-02 10:00:00"); add(conn, "B", 0.1)
    ms = {m["machine_code"]: m for m in db.list_machines(conn)}
    assert ms["A"]["n_predictions"] == 2 and ms["A"]["risk_level"] == "Warning" and ms["A"]["alert_status"] == "open"
    d = db.machine_detail(conn, "a")
    assert [round(t["probability"], 1) for t in d["trend"]] == [0.9, 0.4] and db.machine_detail(conn, "nope") is None


def test_previous_prediction(conn):
    first = add(conn, "A", 0.2)["prediction_id"]; second = add(conn, "A", 0.3)["prediction_id"]
    assert db.previous_prediction(conn, 1, second)["id"] == first and db.previous_prediction(conn, 1, first) is None


def test_export_rows_use_proposal_column_order(conn):
    add(conn, "A", 0.9)
    assert [h for h, _ in db.EXPORT_COLUMNS][:6] == ["Timestamp", "Machine ID", "Machine Type", "Probability", "Alert Level", "Severity"]
    assert len(db.export_rows(conn)[0]) == len(db.EXPORT_COLUMNS)
