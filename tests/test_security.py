"""CSRF, session/cookie settings, headers, injection and configuration."""
import os

import pytest

from app import create_app
from conftest import HEAT, NORMAL, make_app, post, set_probability


def fresh_client(app):
    return app.test_client()                    # no session at all


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_csrf_rejects_missing_empty_and_wrong_tokens_even_without_a_session(app, token):
    """Regression: '' == '' used to pass when there was no session token."""
    c = fresh_client(app)
    data = dict(NORMAL) if token is None else {**NORMAL, "csrf_token": token}
    assert c.post("/analyze", data=data).status_code == 400


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_csrf_rejects_bad_tokens_when_a_session_token_exists(client, token):
    data = dict(NORMAL) if token is None else {**NORMAL, "csrf_token": token}
    assert client.post("/analyze", data=data).status_code == 400


def test_csrf_accepts_the_real_token_and_protects_alert_actions(client, app):
    set_probability(app, 0.9)
    assert post(client, "/analyze", NORMAL).status_code == 302
    assert client.post("/alerts/1/acknowledge", data={}).status_code == 400
    assert client.post("/alerts/1/resolve", data={"csrf_token": ""}).status_code == 400
    assert post(client, "/alerts/1/acknowledge").status_code == 302


def test_real_page_token_round_trip(app):
    c = app.test_client()
    html = c.get("/analyze").get_data(as_text=True)
    token = html.split('name="csrf_token" value="')[1].split('"')[0]
    assert len(token) >= 16
    assert c.post("/analyze", data={**NORMAL, "csrf_token": token}).status_code == 302


def test_cross_site_origin_is_blocked_even_for_json(client):
    assert client.post("/api/predict", json=NORMAL, headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.post("/analyze", data={**NORMAL, "csrf_token": "tok"}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_security_headers_and_cookie_flags(app):
    c = app.test_client()
    r = c.get("/analyze")
    csp = r.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "'unsafe-inline'" not in csp.split("style-src")[0]
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
    cookie = r.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_pages_use_no_inline_scripts(app):
    c = app.test_client()
    for url in ["/", "/analyze", "/analyze/batch", "/machines", "/alerts", "/history", "/insights", "/about"]:
        html = c.get(url).get_data(as_text=True)
        scripts = [s.split(">")[0] for s in html.split("<script")[1:]]
        assert all("src=" in s or "application/json" in s for s in scripts), url


def test_production_requires_a_secret_key(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "production"); monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app({"DATABASE": str(tmp_path / "x.db")})
    monkeypatch.setenv("SECRET_KEY", "x" * 32)
    prod = create_app({"DATABASE": str(tmp_path / "x.db")})
    assert prod.config["SESSION_COOKIE_SECURE"] is True and prod.config["SECRET_KEY"] == "x" * 32


def test_no_hardcoded_secret_in_source():
    root = os.path.dirname(os.path.dirname(__file__))
    for name in ("__init__.py", "routes.py", "service.py", "db.py"):
        assert "dev-only" not in open(os.path.join(root, "app", name)).read()


def test_development_secret_is_random_per_process(monkeypatch, tmp_path, tiny_models):
    monkeypatch.delenv("SECRET_KEY", raising=False); monkeypatch.delenv("APP_ENV", raising=False)
    a = create_app({"DATABASE": str(tmp_path / "a.db"), "MODELS_DIR": tiny_models})
    b = create_app({"DATABASE": str(tmp_path / "b.db"), "MODELS_DIR": tiny_models})
    assert a.config["SECRET_KEY"] != b.config["SECRET_KEY"] and len(a.config["SECRET_KEY"]) >= 32


def test_environment_variables_configure_paths(monkeypatch, tmp_path, tiny_models):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "env.db")); monkeypatch.setenv("MODELS_DIR", tiny_models)
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1234")
    app = create_app({"TESTING": True})
    assert app.config["DATABASE"] == str(tmp_path / "env.db") and app.config["MAX_CONTENT_LENGTH"] == 1234 and os.path.exists(tmp_path / "env.db")


def test_xss_is_escaped_and_hostile_machine_ids_are_rejected(client, app):
    html = client.get("/machines?q=<script>alert(1)</script>").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    r = post(client, "/analyze", {**NORMAL, "machine_id": "<img src=x onerror=alert(1)>"})
    assert r.status_code == 422 and "<img src=x" not in r.get_data(as_text=True)
    assert client.get("/history?machine=<script>").status_code == 200


def test_sql_injection_attempts_are_harmless(client, app):
    set_probability(app, 0.5)
    post(client, "/analyze", NORMAL)
    for url in ["/history?machine=' OR '1'='1", "/history?sort=probability;DROP TABLE predictions--&dir=asc;DROP",
                "/machines?q=%27%20OR%201=1--", "/alerts?view=active'%20OR%20'1'='1", "/history?risk=Critical'%20OR%201=1"]:
        assert client.get(url).status_code == 200, url
    assert "PUMP-1" in client.get("/history").get_data(as_text=True)               # the table still exists and holds the data
    assert "PUMP-1" not in client.get("/history?machine=' OR '1'='1").get_data(as_text=True)


def test_oversized_upload_is_rejected_cleanly(tiny_models, tmp_path):
    app = make_app(tiny_models, tmp_path / "t.db", MAX_CONTENT_LENGTH=200)
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    import io
    r = c.post("/analyze/batch", data={"csrf_token": "tok", "file": (io.BytesIO(b"x" * 5000), "big.csv")}, content_type="multipart/form-data")
    assert r.status_code == 413 and "too large" in r.get_data(as_text=True)
