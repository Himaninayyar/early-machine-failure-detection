"""Model loading failures must degrade gracefully, never crash the application."""
import json
import os

import pytest

from app.predictor import ModelRegistry, ModelUnavailable
from conftest import HEAT, make_app, post


def test_all_models_load_and_checksums_verify(tiny_models):
    reg = ModelRegistry(tiny_models)
    assert reg.ready and reg.errors == {} and reg.keys == {"lr", "rf", "xgb", "ens"} and reg.modes is not None


def test_a_missing_model_file_disables_only_that_model_and_the_ensemble(models_copy):
    os.remove(os.path.join(models_copy, "xgb.joblib"))
    reg = ModelRegistry(models_copy)
    assert reg.keys == {"lr", "rf"} and "missing" in reg.errors["xgb"] and "xgb" in reg.errors["ens"]
    assert reg.ready and reg.default_model in {"lr", "rf"}
    with pytest.raises(ModelUnavailable):
        reg.get("xgb")


def test_a_corrupted_file_is_detected_by_checksum_and_explained(models_copy):
    path = os.path.join(models_copy, "rf.joblib")
    data = bytearray(open(path, "rb").read()); data[len(data) // 2] ^= 0xFF
    open(path, "wb").write(bytes(data))
    reg = ModelRegistry(models_copy)
    assert "rf" not in reg.keys and "checksum" in reg.errors["rf"] and "python -m ml.train" in reg.errors["rf"]
    assert {"lr", "xgb"} <= reg.keys


def test_a_garbage_file_with_no_recorded_checksum_is_still_caught(models_copy):
    meta_path = os.path.join(models_copy, "meta.json")
    meta = json.load(open(meta_path)); del meta["artifacts"]["lr"]["sha256"]; json.dump(meta, open(meta_path, "w"))
    open(os.path.join(models_copy, "lr.joblib"), "wb").write(b"this is not a model")
    reg = ModelRegistry(models_copy)
    assert "lr" not in reg.keys and "could not be loaded" in reg.errors["lr"] and reg.ready


def test_default_falls_back_to_best_available_by_training_ranking(models_copy):
    meta = json.load(open(os.path.join(models_copy, "meta.json")))
    designated = meta["default_model"]
    if designated == "ens":
        pytest.skip("ensemble cannot be removed independently of its members here")
    os.remove(os.path.join(models_copy, meta["artifacts"][designated]["file"]))
    reg = ModelRegistry(models_copy)
    assert reg.default_model != designated and reg.default_is_fallback and reg.default_model in reg.keys
    expected = next(k for k in meta["ranking"] if k in reg.keys)
    assert reg.default_model == expected


def test_failure_mode_models_are_optional(models_copy):
    os.remove(os.path.join(models_copy, "modes.joblib"))
    reg = ModelRegistry(models_copy)
    assert reg.ready and reg.modes is None and reg.modes_error
    r = reg.analyze(reg.default_model, dict(type="L", air_temp=303.7, process_temp=312.1, rpm=1363, torque=51.8, tool_wear=90))
    assert r["details"]["mode_estimates"] == {}


def test_missing_meta_json_leaves_the_registry_not_ready_but_does_not_raise(models_copy):
    os.remove(os.path.join(models_copy, "meta.json"))
    reg = ModelRegistry(models_copy)
    assert not reg.ready and any("meta.json" in p for p in reg.load_problems)


def test_app_starts_with_no_models_and_explains_instead_of_crashing(tmp_path):
    app = make_app(str(tmp_path / "nothing_here"), tmp_path / "t.db")
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    assert c.get("/").status_code == 200 and "Predictions are unavailable" in c.get("/").get_data(as_text=True)
    assert c.get("/insights").status_code == 200 and c.get("/about").status_code == 200
    r = post(c, "/analyze", HEAT)
    assert r.status_code == 503 and "No prediction model is available" in r.get_data(as_text=True)
    h = c.get("/api/health")
    assert h.status_code == 503 and h.get_json()["ready"] is False
    api = c.post("/api/predict", json={**HEAT})
    assert api.status_code == 503 and "error" in api.get_json()


def test_app_serves_predictions_when_only_some_models_loaded(models_copy, tmp_path):
    os.remove(os.path.join(models_copy, "xgb.joblib"))
    app = make_app(models_copy, tmp_path / "t.db")
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    assert post(c, "/analyze", HEAT).status_code == 302
    health = c.get("/api/health").get_json()
    assert health["models"]["xgb"]["available"] is False and health["ready"] is True
    r = post(c, "/analyze", {**HEAT, "model": "xgb"})
    assert r.status_code == 422 and "not available" in r.get_data(as_text=True)


def test_database_failure_is_reported_not_fatal(tiny_models, tmp_path):
    app = make_app(tiny_models, tmp_path)                      # the "database file" is a directory
    assert app.extensions["db_error"]
    c = app.test_client()
    page = c.get("/")
    assert page.status_code == 503 and "database" in page.get_data(as_text=True).lower()
    assert c.get("/about").status_code == 200                  # pages that do not need the database still work
    assert c.get("/api/health").status_code == 503
