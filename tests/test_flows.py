"""Frontend-to-backend flows through the real Flask app (HTTP forms, redirects, pages, JSON API)."""
import csv
import io
import re

import pytest

from app import db
from conftest import DATA, HEAT, NORMAL, csv_file, make_app, post, set_probability


def rows(app, sql, *a):
    c = db.connect(app.config["DATABASE"])
    try:
        return c.execute(sql, a).fetchall()
    finally:
        c.close()


def analyze(client, data=None, **over):
    return post(client, "/analyze", {**(data or NORMAL), **over})


def follow(client, resp):
    assert resp.status_code == 302, resp.get_data(as_text=True)[:300]
    return client.get(resp.headers["Location"])


# ---------------------------------------------------------------- empty states
def test_first_time_user_sees_meaningful_empty_states_and_no_fake_numbers(client):
    dash = client.get("/").get_data(as_text=True)
    assert "Nothing has been analysed yet" in dash and "kpi-strip" not in dash
    assert "No machines yet" in client.get("/machines").get_data(as_text=True)
    assert "No analyses saved yet" in client.get("/history").get_data(as_text=True)
    assert "No active alerts" in client.get("/alerts").get_data(as_text=True)
    assert client.get("/history/export.csv").get_data(as_text=True).count("\n") == 1        # header only


def test_every_page_loads(client):
    for url in ["/", "/analyze", "/analyze/batch", "/analyze/batch/template.csv", "/machines", "/alerts", "/history", "/insights", "/about"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/nope").status_code == 404 and client.get("/predictions/999").status_code == 404
    assert client.get("/machines/ghost").status_code == 404 and client.get("/alerts/999").status_code == 404


# ---------------------------------------------------------------- analyze -> result
def test_analyze_form_to_result_page(client, app):
    set_probability(app, 0.596)
    r = analyze(client, machine_id="PRESS-7")
    assert re.fullmatch(r"/predictions/\d+", r.headers["Location"])
    page = follow(client, r).get_data(as_text=True)
    for expected in ("59.6%", "Warning", "Schedule inspection", "Health score", "40", "What to do", "What drove this result",
                     "Possible failure modes", "What if the readings were different", "Technical details", "PRESS-7"):
        assert expected in page, expected
    assert "does not establish a causal relationship" in page and "not an independently validated" in page
    saved = rows(app, "SELECT * FROM predictions")[0]
    assert saved["probability"] == pytest.approx(0.596) and saved["risk_level"] == "Warning" and saved["health_score"] == 40
    assert saved["model_label"] and saved["model_version"] and saved["recommendation"].startswith("Schedule inspection")


def test_result_page_does_not_expose_model_choice_but_advanced_option_exists(client):
    page = client.get("/analyze").get_data(as_text=True)
    assert "Advanced: choose a different model" in page and "recommended" in page
    assert "Prediction model" in page.split("Advanced: choose a different model")[1]


def test_invalid_input_is_explained_and_nothing_is_saved(client, app):
    r = analyze(client, rpm="abc", torque="", tool_wear="-5", machine_id="")
    html = r.get_data(as_text=True)
    assert r.status_code == 422 and "must be a number" in html and "is required" in html and "Enter a machine ID" in html
    assert 'value="abc"' in html                                                   # the user's input is kept
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 0 and rows(app, "SELECT COUNT(*) c FROM machines")[0]["c"] == 0


def test_out_of_range_reading_is_accepted_with_a_visible_warning(client, app):
    page = follow(client, analyze(client, rpm="3500")).get_data(as_text=True)
    assert "outside the range of the training data" in page


# ---------------------------------------------------------------- machine identity
def test_one_machine_keeps_one_identity_across_analyses(client, app):
    for _ in range(3):
        assert analyze(client, machine_id="Pump-A").status_code == 302
    analyze(client, machine_id="PUMP-a")                                           # same machine, different case
    assert rows(app, "SELECT COUNT(*) c FROM machines")[0]["c"] == 1
    assert rows(app, "SELECT COUNT(DISTINCT machine_id) c FROM predictions")[0]["c"] == 1
    page = client.get("/machines").get_data(as_text=True)
    assert page.count(">Pump-A</a>") == 1 and "AUTO-" not in page            # listed once, as one machine
    assert "4" in client.get("/machines/pump-a").get_data(as_text=True)


def test_machine_type_conflict_is_refused_with_a_clear_message(client, app):
    analyze(client, machine_id="LATHE", type="L")
    r = analyze(client, machine_id="LATHE", type="H")
    assert r.status_code == 422 and "registered as type L" in r.get_data(as_text=True)
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 1


def test_analyze_again_prefills_the_machine(client):
    analyze(client, machine_id="M-9", type="M")
    html = client.get("/analyze?machine=M-9").get_data(as_text=True)
    assert 'value="M-9"' in html and 'value="M" selected' in html


# ---------------------------------------------------------------- alert lifecycle through the UI
def test_alert_lifecycle_over_http(client, app):
    set_probability(app, 0.9)
    follow(client, analyze(client, machine_id="A"))
    follow(client, analyze(client, machine_id="A"))                                # same machine, still critical: no duplicate alert
    assert [r["occurrences"] for r in rows(app, "SELECT * FROM alerts")] == [2]
    alerts = client.get("/alerts").get_data(as_text=True)
    assert "Critical" in alerts and "Acknowledge" in alerts and "Resolve" in alerts

    assert follow(client, post(client, "/alerts/1/acknowledge", {"note": "on it"})).status_code == 200
    assert rows(app, "SELECT status FROM alerts")[0]["status"] == "acknowledged"
    detail = client.get("/alerts/1").get_data(as_text=True)
    assert "on it" in detail and "Acknowledged" in detail and "Alert history" in detail

    set_probability(app, 0.02)
    analyze(client, machine_id="A")                                                # later Normal reading
    a = rows(app, "SELECT * FROM alerts")[0]
    assert a["status"] == "resolved" and a["resolution"] == "auto_normal_reading"
    assert "No active alerts" in client.get("/alerts").get_data(as_text=True)
    assert "Resolved automatically" in client.get("/alerts/1").get_data(as_text=True)
    resolved_view = client.get("/alerts?view=resolved").get_data(as_text=True)
    assert ">A</a>" in resolved_view and "Resolved" in resolved_view

    set_probability(app, 0.5)
    analyze(client, machine_id="A")
    assert [r["status"] for r in rows(app, "SELECT status FROM alerts ORDER BY id")] == ["resolved", "open"]


def test_manual_resolve_and_invalid_actions(client, app):
    set_probability(app, 0.9)
    analyze(client, machine_id="A")
    assert post(client, "/alerts/1/resolve", {"note": "fixed"}).status_code == 302
    a = rows(app, "SELECT * FROM alerts")[0]
    assert (a["status"], a["resolution"]) == ("resolved", "manual")
    flashed = follow(client, post(client, "/alerts/1/acknowledge")).get_data(as_text=True)
    assert "could not be changed" in flashed                                          # acknowledging a resolved alert is refused, visibly
    assert rows(app, "SELECT status FROM alerts")[0]["status"] == "resolved"
    assert post(client, "/alerts/999/acknowledge").status_code == 404


def test_normal_machine_never_creates_an_alert(client, app):
    set_probability(app, 0.05)
    analyze(client)
    assert rows(app, "SELECT COUNT(*) c FROM alerts")[0]["c"] == 0


def test_dashboard_numbers_match_stored_data(client, app):
    set_probability(app, 0.9); analyze(client, machine_id="A"); analyze(client, machine_id="A")
    set_probability(app, 0.5); analyze(client, machine_id="B")
    set_probability(app, 0.05); analyze(client, machine_id="C")
    html = client.get("/").get_data(as_text=True)
    vals = re.findall(r'<div class="value[^"]*">([^<]+)</div>', html)
    assert vals == ["3", "2", "2", "4"]                                               # machines, attention, active alerts, analyses
    assert "Machines requiring attention" in html and "B" in html


# ---------------------------------------------------------------- history immutability
def test_history_does_not_change_when_models_are_replaced(client, app):
    set_probability(app, 0.9)
    pid = int(analyze(client, machine_id="A").headers["Location"].rsplit("/", 1)[1])
    before = client.get(f"/predictions/{pid}").get_data(as_text=True)
    reg = app.extensions["models"]
    reg.version = "NEW-VERSION"; set_probability(app, 0.01)                           # "retrained and replaced"
    after = client.get(f"/predictions/{pid}").get_data(as_text=True)
    assert "90.0%" in after and "90.0%" in before and "1.0%" not in after
    assert "not the one currently loaded" in after                                    # what-if honestly disabled for the old model version
    row = rows(app, "SELECT probability, model_version FROM predictions")[0]
    assert row["probability"] == pytest.approx(0.9) and row["model_version"] != "NEW-VERSION"
    assert "90.0%" in client.get("/history").get_data(as_text=True)


# ---------------------------------------------------------------- what-if
def test_what_if_api(client, app):
    set_probability(app, 0.5)
    pid = int(analyze(client, machine_id="A").headers["Location"].rsplit("/", 1)[1])
    r = client.post("/api/what-if", json={"prediction_id": pid, "changes": {"torque": "30", "tool_wear": 10}})
    body = r.get_json()
    assert r.status_code == 200 and body["probability"] == 0.5 and body["delta"] == 0 and "does not prove" in body["disclaimer"]
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 1                # nothing saved
    bad = client.post("/api/what-if", json={"prediction_id": pid, "changes": {"torque": "-5"}})
    assert bad.status_code == 422 and "torque" in bad.get_json()["errors"]
    assert client.post("/api/what-if", json={"prediction_id": 999, "changes": {}}).status_code == 404
    assert client.post("/api/what-if", json={"nonsense": 1}).status_code == 400


def test_what_if_unavailable_for_a_different_model_version(client, app):
    pid = int(analyze(client).headers["Location"].rsplit("/", 1)[1])
    app.extensions["models"].version = "other"
    r = client.post("/api/what-if", json={"prediction_id": pid, "changes": {"torque": 30}})
    assert r.status_code == 503 and "older model version" in r.get_json()["error"]


# ---------------------------------------------------------------- batch
GOOD = "machine_id,type,air_temp,process_temp,rpm,torque,tool_wear\nB-1,L,303.7,312.4,1513,40.1,135\nB-2,M,298.1,308.6,1551,42.8,0\nB-1,L,303.7,312.1,1363,51.8,90\n"


def upload(client, text, **kw):
    return client.post("/analyze/batch", data={"csrf_token": "tok", "file": csv_file(text)}, content_type="multipart/form-data", **kw)


def test_batch_saves_valid_rows_and_updates_machines_and_alerts(client, app):
    set_probability(app, 0.9)
    r = upload(client, GOOD)
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Rows read" in html and "Results" in html
    assert rows(app, "SELECT COUNT(*) c FROM predictions WHERE source='batch'")[0]["c"] == 3
    assert rows(app, "SELECT COUNT(*) c FROM machines")[0]["c"] == 2                  # B-1 appears twice but is one machine
    assert rows(app, "SELECT occurrences FROM alerts WHERE machine_id = 1")[0]["occurrences"] == 2


def test_batch_reports_bad_rows_with_line_numbers_and_saves_nothing_for_them(client, app):
    set_probability(app, 0.05)
    text = GOOD + "B-3,X,300,310,1500,40,10\nB-4,L,300,310,abc,40,10\nB-2,H,298.1,308.6,1551,42.8,0\n"
    html = upload(client, text).get_data(as_text=True)
    assert "Rejected rows" in html and ">5<" in html and ">6<" in html and ">7<" in html
    assert "registered as type M" in html                                             # machine type conflict inside the file
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 3


@pytest.mark.parametrize("text,msg", [("machine_id,type\nA,L\n", "Missing column"), ("", "Missing column"),
                                     ("machine_id,type,air_temp,process_temp,rpm,torque,tool_wear\n", "no data rows")])
def test_batch_file_problems_are_explained(client, text, msg):
    r = upload(client, text)
    assert r.status_code == 422 and msg in r.get_data(as_text=True)


def test_batch_row_limit_and_encoding(tiny_models, tmp_path):
    app = make_app(tiny_models, tmp_path / "t.db", BATCH_MAX_ROWS=2)
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    assert "more than 2 rows" in upload(c, GOOD).get_data(as_text=True)
    r = c.post("/analyze/batch", data={"csrf_token": "tok", "file": (io.BytesIO(b"\xff\xfe\x00bad"), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 422 and "UTF-8" in r.get_data(as_text=True)
    assert c.post("/analyze/batch", data={"csrf_token": "tok"}, content_type="multipart/form-data").status_code == 422


def test_batch_template_is_valid_input_for_batch(client, app):
    tpl = client.get("/analyze/batch/template.csv").get_data(as_text=True)
    assert tpl.splitlines()[0] == "machine_id,type,air_temp,process_temp,rpm,torque,tool_wear"
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 0
    assert "Rejected rows" not in upload(client, tpl).get_data(as_text=True)
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 3


# ---------------------------------------------------------------- history, export, machine pages
def test_history_filters_and_export(client, app):
    set_probability(app, 0.9); analyze(client, machine_id="CRIT-1")
    set_probability(app, 0.05); analyze(client, machine_id="OK-1")
    h = client.get("/history?risk=Critical").get_data(as_text=True)
    assert "CRIT-1" in h and "OK-1" not in h
    assert "OK-1" in client.get("/history?machine=ok").get_data(as_text=True)
    assert "No analyses match" in client.get("/history?machine=zzz").get_data(as_text=True)
    flashed = client.get("/history?date_from=bad&model=svm", follow_redirects=True).get_data(as_text=True)
    assert "not a valid date" in flashed and "Unknown model" in flashed
    r = client.get("/history/export.csv")
    rows_ = list(csv.reader(io.StringIO(r.get_data(as_text=True))))
    assert r.mimetype == "text/csv" and rows_[0][:6] == ["Timestamp", "Machine ID", "Machine Type", "Probability", "Alert Level", "Severity"]
    assert "Health Score" in rows_[0] and "Model Version" in rows_[0] and "Rule Indicators" in rows_[0] and len(rows_) == 3
    f = list(csv.reader(io.StringIO(client.get("/history/export.csv?risk=Critical").get_data(as_text=True))))
    assert len(f) == 2 and f[1][1] == "CRIT-1" and f[1][4] == "Critical"


def test_machine_page_shows_trend_and_history(client, app):
    for p in (0.1, 0.5, 0.9):
        set_probability(app, p); analyze(client, machine_id="TREND-1")
    html = client.get("/machines/TREND-1").get_data(as_text=True)
    assert "Risk trend" in html and 'id="chart-trend"' in html and "Recent analyses" in html and "alert #1" in html
    analyze(client, machine_id="SOLO")
    assert "at least twice" in client.get("/machines/SOLO").get_data(as_text=True)


def test_previous_analysis_comparison_on_result_page(client, app):
    set_probability(app, 0.2); analyze(client, machine_id="A")
    set_probability(app, 0.5); r = analyze(client, machine_id="A")
    page = follow(client, r).get_data(as_text=True)
    assert "Last time: 20.0%" in page and "+30.0 points" in page


# ---------------------------------------------------------------- JSON API
def test_api_predict_and_errors(client, app):
    set_probability(app, 0.9)
    r = client.post("/api/predict", json={**HEAT, "air_temp": 303.7})
    b = r.get_json()
    assert r.status_code == 200 and b["risk_level"] == "Critical" and b["health_score"] == 10 and b["alert"]["action"] == "opened"
    assert {"prediction_id", "machine_id", "probability", "recommendation", "model", "model_version", "warnings", "indicators"} <= set(b)
    assert rows(app, "SELECT source FROM predictions")[0]["source"] == "api"
    bad = client.post("/api/predict", json={"type": "L"})
    assert bad.status_code == 422 and "machine_id" in bad.get_json()["errors"]
    assert client.post("/api/predict", data="x").status_code == 415
    assert client.post("/api/predict", data="[1]", content_type="application/json").status_code == 400
    assert client.get("/api/health").get_json()["database"]["ok"] is True


def test_legacy_urls_redirect(client):
    assert client.get("/predict").headers["Location"].endswith("/analyze") and client.get("/models").headers["Location"].endswith("/insights")


def test_model_insights_page_reports_all_four_models(client):
    html = client.get("/insights").get_data(as_text=True)
    for s in ("Logistic Regression", "Random Forest", "XGBoost", "Soft Voting Ensemble", "Which model is used, and why", "Soft voting ensemble",
              "Failure-mode classifiers", "Are the 30% and 70% thresholds reasonable?", "Loaded models", "Preliminary results"):
        assert s in html, s


# ---------------------------------------------------------------- xlsx support (converted to CSV, then the normal CSV path)
def _xlsx_bytes(rows):
    import openpyxl
    wb = openpyxl.Workbook()
    for r in rows:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _post_file(c, data, name):
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    return c.post("/analyze/batch", data={"csrf_token": "tok", "file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_batch_accepts_xlsx_same_as_csv(client, app):
    hdr = ["machine_id", "type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    r = _post_file(client, _xlsx_bytes([hdr, ["XL-1", "L", 303.7, 312.4, 1513.0, 40.1, 135], [], ["XL-2", "M", 298.1, 308.6, 1551, 42.8, 0]]), "b.xlsx")
    assert r.status_code == 200 and "Rejected rows" not in r.get_data(as_text=True)
    assert rows(app, "SELECT COUNT(*) c FROM predictions")[0]["c"] == 2


def test_batch_rejects_bad_xlsx_and_unknown_types(client):
    assert _post_file(client, b"not a workbook", "b.xlsx").status_code == 422
    assert "Unsupported file type" in _post_file(client, b"a,b", "b.txt").get_data(as_text=True)


def test_ensure_csv_converts_once_and_load_data_reads_it(tmp_path):
    import shutil, pandas as pd
    from ml.data_io import ensure_csv
    from ml.features import load_data
    src = pd.read_csv(DATA).head(50)
    x = tmp_path / "d.xlsx"
    src.to_excel(x, index=False)
    out = ensure_csv(str(x))
    assert out.endswith("d.csv") and (tmp_path / "d.csv").exists()
    assert len(load_data(str(x))) == 50 and ensure_csv(out) == out


def test_batch_accepts_the_original_ai4i_dataset_columns(client, app):
    import pandas as pd
    d = pd.read_csv(DATA).head(5)
    r = _post_file(client, d.to_csv(index=False).encode(), "ai4i2020.csv")      # UDI, Product ID, Air temperature [K], ... as in the dataset
    page = r.get_data(as_text=True)
    assert r.status_code == 200 and "Missing column" not in page
    saved = rows(app, "SELECT m.machine_code FROM predictions p JOIN machines m ON m.id = p.machine_id ORDER BY p.id")
    assert [x["machine_code"] for x in saved] == list(d["Product ID"])


# ---------------------------------------------------------------- maintenance notice shows up even when the status is Normal, and never alerts
def test_tool_wear_notice_on_result_page_without_changing_status_or_alerts(client, app):
    set_probability(app, 0.05)
    r = analyze(client, machine_id="WORN-1", tool_wear="215")
    page = follow(client, r).get_data(as_text=True)
    assert "Maintenance notice: High tool wear" in page and "consider replacement" in page
    saved = rows(app, "SELECT risk_level FROM predictions")[0]
    assert saved["risk_level"] == "Normal" and rows(app, "SELECT COUNT(*) c FROM alerts")[0]["c"] == 0
    set_probability(app, 0.05)
    page2 = follow(client, analyze(client, machine_id="FRESH-1", tool_wear="20")).get_data(as_text=True)
    assert "Maintenance notice" not in page2


def test_tool_wear_notice_in_api_and_batch(client, app):
    set_probability(app, 0.05)
    with client.session_transaction() as s:
        s["csrf"] = "tok"
    api = client.post("/api/predict", json={"machine_id": "API-W", "type": "L", "air_temp": 298, "process_temp": 308, "rpm": 1500,
                                              "torque": 40, "tool_wear": 230}, headers={"X-CSRF-Token": "tok"})
    body = api.get_json() if api.status_code == 200 else {}
    assert api.status_code == 200 and body["maintenance"][0]["code"] == "TOOL_WEAR" and body["risk_level"] == "Normal"
    csv_bytes = b"machine_id,type,air_temp,process_temp,rpm,torque,tool_wear\nB1,L,298,308,1500,40,205\nB2,L,298,308,1500,40,10\n"
    page = _post_file(client, csv_bytes, "b.csv").get_data(as_text=True)
    assert "1 machine with high tool wear" in page
