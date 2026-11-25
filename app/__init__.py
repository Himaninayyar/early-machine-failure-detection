"""Early Machine Failure Detection - Flask application factory.

Configuration comes from environment variables (see docs/ARCHITECTURE.md):
    APP_ENV                 development (default) | production
    SECRET_KEY              required in production; in development a random per-process key is used
    DATABASE_PATH           SQLite file (default: instance/predictions.db)
    MODELS_DIR, REPORTS_DIR trained-model and EDA folders
    SESSION_COOKIE_SECURE   1 to send the session cookie over HTTPS only (default 1 in production)
    MAX_UPLOAD_BYTES        batch CSV size limit (default 1,000,000)
"""
import hmac
import logging
import os
import secrets
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, render_template, request, session
from werkzeug.exceptions import HTTPException

from app import db
from app.predictor import ModelRegistry
from app.service import ServiceError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger(__name__)


def _flag(name, default):
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=True)
    config = dict(config or {})
    env = os.environ.get("APP_ENV", "development").lower()
    testing = bool(config.get("TESTING"))

    secret = config.get("SECRET_KEY") or os.environ.get("SECRET_KEY")
    if not secret:
        if env == "production" and not testing:
            raise RuntimeError("SECRET_KEY must be set when APP_ENV=production (e.g. python -c \"import secrets; print(secrets.token_hex(32))\").")
        secret = secrets.token_hex(32)  # random per process: sessions reset on restart, nothing is hard-coded
        logging.getLogger(__name__).warning("SECRET_KEY not set: using a random per-process key (fine for development).")

    app.config.update(
        APP_ENV=env, SECRET_KEY=secret,
        DATABASE=os.environ.get("DATABASE_PATH") or os.path.join(app.instance_path, "predictions.db"),
        MODELS_DIR=os.environ.get("MODELS_DIR") or os.path.join(ROOT, "models"),
        REPORTS_DIR=os.environ.get("REPORTS_DIR") or os.path.join(ROOT, "reports"),
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_flag("SESSION_COOKIE_SECURE", env == "production"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_BYTES", 1_000_000)), BATCH_MAX_ROWS=1000,
    )
    app.config.update(config)
    os.makedirs(os.path.dirname(os.path.abspath(app.config["DATABASE"])), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # --- database: a failure here must not stop the app; pages report it instead ----------------
    app.extensions["db_error"] = None
    try:
        status = db.init_db(app.config["DATABASE"])
        if status["status"] == "migrated":
            log.warning("Database migrated from v1: %s (backup: %s)", status["report"], status["backup"])
    except Exception as exc:  # noqa: BLE001
        app.extensions["db_error"] = f"{type(exc).__name__}: {exc}"
        log.error("Database could not be initialised: %s", exc)

    # --- models: loaded once, each independently; never raises ------------------------------------
    app.extensions["models"] = ModelRegistry(app.config["MODELS_DIR"], app.config["REPORTS_DIR"])

    @app.teardown_appcontext
    def close_db(_exc):
        from flask import g
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    # --- CSRF + Origin check ---------------------------------------------------------------------------
    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    @app.before_request
    def protect():
        if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
            return None
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            abort(403, "Cross-site request blocked.")
        if request.path.startswith("/api/"):
            if not request.is_json:  # the API accepts JSON only, so a plain form post can never reach it
                abort(415, "Send a JSON body with Content-Type: application/json.")
            return None  # JSON bodies cannot be sent cross-site without a CORS preflight, and no CORS headers are sent
        if request.is_json:
            return None
        expected = session.get("csrf")
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or ""
        # BOTH values must be present and non-empty: "" == "" must never pass
        if not expected or not sent or not hmac.compare_digest(sent, expected):
            abort(400, "Your session expired or the form was invalid. Reload the page and try again.")
        return None

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.method == "POST":
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    # --- template helpers ---------------------------------------------------------------------------------
    @app.template_filter("pct")
    def pct(p):
        """Probability 0-1 -> '59.6%'. Never print a misleading exact 0% or 100%."""
        if p is None:
            return "-"
        if p < 0.0005:
            return "<0.1%"
        if p > 0.9995:
            return ">99.9%"
        return f"{p * 100:.1f}%"

    @app.template_filter("plain")
    def plain(v):
        """Form value for display: 1376.0 -> 1376, 298.10 -> 298.1. Text the user typed (even invalid) is kept as is."""
        try:
            return f"{float(v):g}"
        except (TypeError, ValueError):
            return v if v is not None else ""

    @app.template_filter("num")
    def num(v, digits=1):
        return f"{v:,.{digits}f}" if v is not None else "-"

    nav_parent = {"main.result": "main.analyze", "main.batch": "main.analyze", "main.machine_detail": "main.machines",
                  "main.alert_detail": "main.alerts"}

    @app.context_processor
    def inject_globals():
        reg = app.extensions["models"]
        active = nav_parent.get(request.endpoint, request.endpoint)
        return {"csrf_token": csrf_token, "model_version": reg.version, "registry": reg, "active_nav": active,
                "default_model_label": reg.labels.get(reg.default_model, "none") if reg.default_model else "none available",
                "db_error": app.extensions["db_error"],
                "nav": [("main.dashboard", "Dashboard"), ("main.analyze", "Analyze Machine"), ("main.machines", "Machines"),
                        ("main.alerts", "Alerts"), ("main.history", "History"), ("main.insights", "Model Insights"),
                        ("main.about", "About")]}

    app.jinja_env.globals.update(zip=zip)
    from app.routes import bp
    app.register_blueprint(bp)

    # --- errors: understandable messages, JSON for /api, nothing silently ignored ----------------------------
    def wants_json():
        return request.path.startswith("/api/")

    def fail(code, message, errors=None, status=None):
        status = status or code
        if wants_json():
            return jsonify(error=message, code=code, errors=errors or {}), status
        return render_template("error.html", code=status, message=message), status

    @app.errorhandler(ServiceError)
    def service_error(e):
        return fail(e.code, e.message, e.errors, e.status)

    @app.errorhandler(HTTPException)
    def http_error(e):
        messages = {404: "That page does not exist.", 413: "That file is too large.", 405: "That action is not allowed here."}
        return fail(e.code, messages.get(e.code, e.description), status=e.code)

    @app.errorhandler(Exception)
    def unexpected(e):
        import sqlite3
        if isinstance(e, sqlite3.Error):
            log.exception("database error")
            return fail("database_error", "The database is unavailable or returned an error. Nothing was lost; try again, or see the server log.", status=503)
        log.exception("unhandled error")
        return fail("server_error", "Something went wrong on the server. The error has been logged.", status=500)

    return app
