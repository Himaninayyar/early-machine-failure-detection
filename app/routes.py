"""Web routes and JSON API. Business logic lives in service.py, predictor.py, risk.py and db.py."""
import csv
import io
import json
import math
import os
from datetime import datetime

from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for)

from app import db, risk, service
from ml import data_io
from app.validation import FIELDS, TYPE_HELP

bp = Blueprint("main", __name__)
PER_PAGE = 25
MODEL_ORDER = ["lr", "rf", "xgb", "ens"]

# Real records from the dataset's held-out test split (never used to train or tune the models).
EXAMPLES = [
    {"label": "Normal operation", "note": "No failure recorded in the dataset", "machine_id": "DEMO-NORMAL",
     "values": dict(type="L", air_temp=303.7, process_temp=312.4, rpm=1513, torque=40.1, tool_wear=135)},
    {"label": "Heat dissipation problem", "note": "Failure recorded in the dataset", "machine_id": "DEMO-HEAT",
     "values": dict(type="L", air_temp=303.7, process_temp=312.1, rpm=1363, torque=51.8, tool_wear=90)},
    {"label": "Overstrain problem", "note": "Failure recorded in the dataset", "machine_id": "DEMO-LOAD",
     "values": dict(type="L", air_temp=299.1, process_temp=310.2, rpm=1317, torque=54.8, tool_wear=231)},
    {"label": "Borderline reading", "note": "Failure recorded in the dataset", "machine_id": "DEMO-BORDER",
     "values": dict(type="M", air_temp=302.1, process_temp=310.7, rpm=1376, torque=46.3, tool_wear=210)},
]


# ----------------------------------------------------------------------------- helpers
def conn():
    if current_app.extensions.get("db_error"):
        raise service.ServiceUnavailable(f"The database could not be opened ({current_app.extensions['db_error']}). "
                                         "Check DATABASE_PATH and the file's permissions.")
    if "db" not in g:
        g.db = db.connect(current_app.config["DATABASE"])
    return g.db


def registry():
    return current_app.extensions["models"]


@bp.app_context_processor
def template_helpers():
    def url_with(**changes):
        """Current URL with some query parameters replaced (None removes one). Used by sort/pagination links."""
        params = {k: request.args.getlist(k) for k in request.args if k not in changes}
        params.update({k: v for k, v in changes.items() if v is not None})
        return url_for(request.endpoint, **params)
    return {"url_with": url_with, "FIELDS": FIELDS, "TYPE_HELP": TYPE_HELP, "risk": risk}


def load_details(row):
    try:
        return json.loads(row["details_json"]) if row["details_json"] else {}
    except ValueError:
        return {}


def page_number():
    try:
        return max(1, int(request.args.get("page", 1)))
    except ValueError:
        return 1


def parse_date(text, label):
    if not text:
        return ""
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return text
    except ValueError:
        flash(f"{label} '{text}' is not a valid date and was ignored.", "warning")
        return ""


def parse_filters():
    reg = registry()
    a = request.args
    model = a.get("model", "")
    if model and model not in reg.labels:
        flash(f"Unknown model '{model}' was ignored.", "warning")
        model = ""
    return {"risk": [r for r in a.getlist("risk") if r in risk.LEVELS], "model": model,
            "machine": a.get("machine", "").strip()[:40],
            "date_from": parse_date(a.get("date_from", ""), "Start date"), "date_to": parse_date(a.get("date_to", ""), "End date")}


def form_ctx(**extra):
    reg = registry()
    ctx = {"examples": EXAMPLES, "errors": {}, "form": {}, "ranges": reg.ranges,
           "models": {k: v for k, v in reg.labels.items() if k in reg.keys}, "default_model": reg.default_model,
           "known_machines": [] if current_app.extensions.get("db_error") else
           [{"code": m["machine_code"], "type": m["machine_type"]} for m in db.list_machines(conn())]}
    ctx.update(extra)
    return ctx


# ----------------------------------------------------------------------------- pages
@bp.route("/")
def dashboard():
    return render_template("dashboard.html", s=db.dashboard_stats(conn()))


@bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        try:
            _result, stored = service.analyze_and_store(registry(), conn(), request.form)
        except service.ValidationFailed as exc:
            return render_template("analyze.html", **form_ctx(form=request.form, errors=exc.errors, warnings=exc.warnings)), 422
        return redirect(url_for("main.result", pid=stored["prediction_id"]))
    form = {}
    code = request.args.get("machine", "").strip()
    if code:  # "Analyze again" from a machine page: machine ID and type are pre-filled
        m = db.get_machine(conn(), code)
        form = {"machine_id": m["machine_code"], "type": m["machine_type"]} if m else {"machine_id": code[:40]}
    return render_template("analyze.html", **form_ctx(form=form))


@bp.route("/predict")
def legacy_predict():
    return redirect(url_for("main.analyze"), code=301)


@bp.route("/predictions/<int:pid>")
def result(pid):
    row = db.get_prediction(conn(), pid)
    if row is None:
        abort(404)
    reg = registry()
    details = load_details(row)
    reliable = set(reg.meta.get("failure_modes_reliable", []))
    estimates = {c: p for c, p in (details.get("mode_estimates") or {}).items() if c in reliable}
    modes = risk.failure_mode_rows(estimates, details.get("indicators", []))
    what_if_ok = row["model_key"] in reg.keys and row["model_version"] == reg.version
    if "maintenance" not in details:   # analysed before maintenance notices existed: derive from the stored reading
        details["maintenance"] = risk.maintenance_notices({"tool_wear": row["tool_wear"]}, (reg.eda or {}).get("tool_wear_rates"))
    return render_template("result.html", p=row, d=details, modes=modes, what_if_ok=what_if_ok, level=risk.LEVELS[row["risk_level"]],
                           alert=db.active_alert(conn(), row["machine_id"]), prev=db.previous_prediction(conn(), row["machine_id"], pid),
                           has_modes=reg.modes is not None, mode_note=risk.RNF_NOTE, reliable_modes=sorted(reliable),
                           health_formula=risk.HEALTH_FORMULA, health_note=risk.HEALTH_NOTE)


@bp.route("/analyze/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", summary=None, error=None)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return render_template("batch.html", summary=None, error="Choose a CSV or Excel (.xlsx) file to upload."), 422
    max_rows = current_app.config["BATCH_MAX_ROWS"]
    try:
        raw = upload.read()
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext in data_io.XLSX_EXT:
            raw = data_io.xlsx_bytes_to_csv_bytes(raw, max_rows=max_rows)   # from here on it is just CSV
        elif ext not in data_io.CSV_EXT:
            raise service.ValidationFailed("Unsupported file type. Upload a .csv or .xlsx file.")
        summary = service.analyze_batch(registry(), conn(), raw, max_rows=max_rows)
    except data_io.DataFileError as exc:
        return render_template("batch.html", summary=None, error=str(exc)), 422
    except service.ValidationFailed as exc:
        return render_template("batch.html", summary=None, error=exc.message), 422
    return render_template("batch.html", summary=summary, error=None)


@bp.route("/analyze/batch/template.csv")
def batch_template():
    rows = [service.BATCH_COLUMNS] + [[e["machine_id"], *[e["values"][k] for k in service.BATCH_COLUMNS[1:]]] for e in EXAMPLES[:3]]
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=batch_template.csv"})


@bp.route("/machines")
def machines():
    q = request.args.get("q", "").strip()[:40]
    attention = request.args.get("attention") == "1"
    return render_template("machines.html", machines=db.list_machines(conn(), search=q or None, attention_only=attention), q=q,
                           attention=attention, total=len(db.list_machines(conn())))


@bp.route("/machines/<code>")
def machine_detail(code):
    detail = db.machine_detail(conn(), code)
    if detail is None:
        abort(404)
    m = detail["machine"]
    recent = db.list_predictions(conn(), {"machine_exact": m["machine_code"]}, limit=15)
    return render_template("machine.html", m=m, trend=detail["trend"], recent=recent, alert=db.active_alert(conn(), m["id"]))


@bp.route("/alerts")
def alerts():
    view = request.args.get("view", "active")
    view = view if view in ("active", "resolved", "all") else "active"
    total = db.count_alerts(conn(), view)
    pages = max(1, math.ceil(total / PER_PAGE))
    page = min(page_number(), pages)
    return render_template("alerts.html", rows=db.list_alerts(conn(), view, PER_PAGE, (page - 1) * PER_PAGE), total=total, page=page,
                           pages=pages, view=view, counts=db.alert_counts(conn()), per_page=PER_PAGE)


@bp.route("/alerts/<int:alert_id>")
def alert_detail(alert_id):
    a = db.get_alert(conn(), alert_id)
    if a is None:
        abort(404)
    return render_template("alert.html", a=a, events=db.alert_events(conn(), alert_id))


def _alert_action(alert_id, fn, done_text):
    note = (request.form.get("note") or "").strip()[:300] or None
    if not fn(conn(), alert_id, note):
        flash("That alert could not be changed (it may already be acknowledged or resolved).", "warning")
    else:
        flash(done_text, "success")
    back = request.form.get("next", "")
    return redirect(back if back.startswith("/alerts") else url_for("main.alert_detail", alert_id=alert_id))


@bp.post("/alerts/<int:alert_id>/acknowledge")
def alert_acknowledge(alert_id):
    if db.get_alert(conn(), alert_id) is None:
        abort(404)
    return _alert_action(alert_id, db.acknowledge_alert, f"Alert #{alert_id} acknowledged.")


@bp.post("/alerts/<int:alert_id>/resolve")
def alert_resolve(alert_id):
    if db.get_alert(conn(), alert_id) is None:
        abort(404)
    return _alert_action(alert_id, db.resolve_alert, f"Alert #{alert_id} resolved.")


@bp.route("/history")
def history():
    f = parse_filters()
    sort = request.args.get("sort", "created_at")
    direction = "asc" if request.args.get("dir") == "asc" else "desc"
    total = db.count_predictions(conn(), f)
    pages = max(1, math.ceil(total / PER_PAGE))
    page = min(page_number(), pages)
    export_params = {k: request.args.getlist(k) for k in ("risk", "model", "machine", "date_from", "date_to") if any(request.args.getlist(k))}
    return render_template("history.html", rows=db.list_predictions(conn(), f, sort, direction, PER_PAGE, (page - 1) * PER_PAGE),
                           total=total, page=page, pages=pages, f=f, sort=sort, direction=direction, per_page=PER_PAGE,
                           any_saved=db.count_predictions(conn()) > 0, export_params=export_params,
                           models=registry().labels, levels=list(risk.LEVELS))


@bp.route("/history/export.csv")
def export_csv():
    """CSV of the (optionally filtered) history, newest first."""
    headers = [h for h, _ in db.EXPORT_COLUMNS if h != "Details"] + ["Rule Indicators", "Model Failure-Mode Estimates (>=50%)"]
    reliable = set(registry().meta.get("failure_modes_reliable", []))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for r in db.export_rows(conn(), parse_filters()):
        vals = list(r)
        try:
            d = json.loads(vals[-1]) if vals[-1] else {}
        except ValueError:
            d = {}
        base = [round(v, 4) if i == 3 else ("" if v is None else v) for i, v in enumerate(vals[:-1])]
        ind = " ".join(i["code"] for i in d.get("indicators", []))
        est = " ".join(f"{c}:{p:.2f}" for c, p in (d.get("mode_estimates") or {}).items() if c in reliable and p >= 0.5)
        w.writerow(base + [ind, est])
    name = f"predictions_{datetime.now():%Y%m%d_%H%M}.csv"
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={name}"})


@bp.route("/insights")
def insights():
    reg = registry()
    m = reg.metrics
    order = [k for k in MODEL_ORDER if k in m.get("test", {})]
    per_type = (m.get("evaluation") or {}).get("per_type") or {}
    type_curves = {k: {"test": v.get("pr_curve_test"), "cv": v.get("pr_curve_cv")} for k, v in per_type.items() if v.get("available")} or None
    return render_template("insights.html", m=m, order=order, labels=reg.labels, default=reg.meta.get("default_model"),
                           meta=reg.meta, status=reg.status(), health_formula=risk.HEALTH_FORMULA, health_note=risk.HEALTH_NOTE,
                           twf=reg.twf_study, type_curves=type_curves)


@bp.route("/models")
def legacy_models():
    return redirect(url_for("main.insights"), code=301)


@bp.route("/about")
def about():
    reg = registry()
    return render_template("about.html", eda=reg.eda, m=reg.metrics, meta=reg.meta, health_formula=risk.HEALTH_FORMULA,
                           health_note=risk.HEALTH_NOTE)


# ----------------------------------------------------------------------------- JSON API
@bp.post("/api/predict")
def api_predict():
    """Same workflow as the form. Body: JSON with machine_id, type, air_temp, process_temp, rpm, torque, tool_wear
    and optional model. The analysis is saved to history (and may open/update the machine's alert)."""
    if not request.is_json:
        return jsonify(error="Send a JSON body with Content-Type: application/json", code="unsupported_media_type"), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Body must be a JSON object", code="bad_request"), 400
    form = {k: "" if v is None else str(v) for k, v in payload.items()}
    result, stored = service.analyze_and_store(registry(), conn(), form, source="api")
    d = result["details"]
    return jsonify(prediction_id=stored["prediction_id"], machine_id=stored["machine_code"], machine_created=stored["machine_created"],
                   probability=result["probability"], risk_level=result["risk_level"], health_score=result["health_score"],
                   action=result["action"], recommendation=result["recommendation"], model=result["model_label"],
                   model_version=result["model_version"], warnings=d["input_warnings"], indicators=[i["code"] for i in d["indicators"]],
                   maintenance=d.get("maintenance", []), alert=stored["alert"])


@bp.post("/api/what-if")
def api_what_if():
    """Body: {"prediction_id": 12, "changes": {"torque": 60}}. Re-runs the model that made that prediction. Nothing is saved."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("changes"), dict):
        return jsonify(error="Body must be JSON like {\"prediction_id\": 1, \"changes\": {\"torque\": 55}}", code="bad_request"), 400
    row = db.get_prediction(conn(), int(payload["prediction_id"])) if str(payload.get("prediction_id", "")).isdigit() else None
    if row is None:
        return jsonify(error="Unknown prediction_id", code="not_found"), 404
    out = service.what_if(registry(), row, payload["changes"])
    out["disclaimer"] = ("This shows how the model's estimate responds to the change. It does not prove that changing the real "
                         "machine would cause the same change in failure risk.")
    return jsonify(out)


@bp.get("/api/health")
def api_health():
    """Operational status: models, failure-mode models, database."""
    reg = registry()
    status = reg.status()
    status["database"] = {"ok": not current_app.extensions.get("db_error"), "error": current_app.extensions.get("db_error")}
    return jsonify(status), (200 if status["ready"] and status["database"]["ok"] else 503)
