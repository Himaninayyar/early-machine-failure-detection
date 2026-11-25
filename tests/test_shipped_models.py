"""Integrity and consistency of the models and reports that ship with the project (models/)."""
import hashlib
import json
import os

import pytest

from app.predictor import ModelRegistry
from conftest import ROOT

MODELS = os.path.join(ROOT, "models")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(MODELS, "meta.json")), reason="models not trained yet")

READING = dict(type="L", air_temp=303.7, process_temp=312.1, rpm=1363, torque=51.8, tool_wear=90)


@pytest.fixture(scope="module")
def reg():
    return ModelRegistry(MODELS, os.path.join(ROOT, "reports"))


def test_artifact_checksums_match_meta():
    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    for key, art in meta["artifacts"].items():
        h = hashlib.sha256(open(os.path.join(MODELS, art["file"]), "rb").read()).hexdigest()
        assert h == art["sha256"], key


def test_everything_loads_and_the_designated_default_is_available(reg):
    assert reg.ready and reg.errors == {} and not reg.default_is_fallback
    assert reg.keys == {"lr", "rf", "xgb", "ens"} and reg.default_model == reg.meta["default_model"] and reg.modes is not None


def test_this_is_a_full_run_not_a_preliminary_one(reg):
    assert reg.meta["profile"] == "full" and reg.metrics["preliminary"] is False
    assert reg.metrics["dataset"]["train_rows"] == 8000 and reg.metrics["dataset"]["test_rows"] == 2000
    assert reg.metrics["dataset"]["train_failures"] + reg.metrics["dataset"]["test_failures"] == 339


def test_meta_and_metrics_describe_the_same_training_run(reg):
    assert reg.meta["version"] == reg.metrics["version"] and reg.meta["default_model"] == reg.metrics["selection"]["default_model"]
    assert set(reg.metrics["test"]) == {"lr", "rf", "xgb", "ens"} and set(reg.metrics["tuning"]) == {"lr", "rf", "xgb"}


def test_the_default_model_follows_the_documented_selection_rule(reg):
    table = reg.metrics["selection"]["table"]
    best = max(r["cv_pr_auc"] for r in table)
    near = [r for r in table if best - r["cv_pr_auc"] <= 0.01]
    order = ["lr", "rf", "xgb", "ens"]
    expected = min(near, key=lambda r: (r["cv_brier"], order.index(r["model"])))["model"]
    assert reg.meta["default_model"] == expected


def test_all_four_models_predict_valid_probabilities_for_the_same_reading(reg):
    probs = {k: reg.predict_proba(k, READING) for k in ("lr", "rf", "xgb", "ens")}
    assert all(0 <= p <= 1 for p in probs.values())
    assert probs["ens"] == pytest.approx(sum(reg.meta["ensemble"]["weights"][k] * probs[k] for k in ("lr", "rf", "xgb")) /
                                         sum(reg.meta["ensemble"]["weights"].values()), abs=1e-9)


def test_reported_test_metrics_reproduce_from_the_shipped_models(reg):
    """Recompute the default model's test metrics from the saved artefacts: they must equal metrics.json."""
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split
    from ml.features import INPUT_COLS, load_data
    df = load_data(os.path.join(ROOT, "data", "raw", "ai4i2020.csv"))
    _, idx_te = train_test_split(df.index, test_size=0.2, stratify=df["failure"].values, random_state=42)
    Xte, yte = df.loc[idx_te, INPUT_COLS], df.loc[idx_te, "failure"].values
    key = reg.default_model
    p = reg.get(key).predict_proba(Xte)[:, 1]
    assert round(float(average_precision_score(yte, p)), 4) == reg.metrics["test"][key]["pr_auc"]


def test_failure_mode_gate_excludes_unreliable_classifiers(reg):
    for code in reg.meta["failure_modes_reliable"]:
        assert reg.metrics["failure_modes"][code]["oof_pr_auc"] >= 0.5
    assert "TWF" not in reg.meta["failure_modes_reliable"] or reg.metrics["failure_modes"]["TWF"]["oof_pr_auc"] >= 0.5


def test_libraries_match_requirements_file():
    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    req = open(os.path.join(ROOT, "requirements-lock.txt")).read()   # exact pins live in the lock file
    for lib in ("scikit-learn", "xgboost", "imbalanced-learn"):
        assert f"{lib}=={meta['libraries'][lib]}" in req, lib


# ---------------------------------------------------------------- per-type evaluation and tool-wear study (consistency with the rest of metrics.json)
def _metrics():
    return json.load(open(os.path.join(MODELS, "metrics.json")))


def test_evaluation_block_is_consistent_with_the_main_test_results():
    m = _metrics()
    ev, d = m["evaluation"], m["dataset"]
    t3 = m["test"][ev["model"]]["thresholds"]["0.3"]
    assert ev["missed"]["total"] == t3["fn"] == len(ev["missed"]["rows"])          # same 12 failures as the headline numbers
    assert ev["missed"]["failures"] == d["test_failures"]
    for code, t in ev["per_type"].items():
        x = t["test_at_0.5"]
        assert x["tp"] + x["fp"] + x["fn"] + x["tn"] == d["test_rows"] and x["tp"] + x["fn"] == t["positives_test"]
        assert [c["threshold"] for c in t["cv_thresholds"]] == [0.1, 0.3, 0.5, 0.7]
    assert ev["missed"]["with_tool_wear_notice"] == sum(r["notice"] for r in ev["missed"]["rows"])
    assert ev["notice"]["test_failures_with_notice"] >= ev["missed"]["with_tool_wear_notice"]


def test_twf_study_uses_training_data_only_and_adds_up():
    m = _metrics()
    s = json.load(open(os.path.join(ROOT, "reports", "twf_study.json")))
    assert s["train_rows"] == m["dataset"]["train_rows"] and "test set not used" in s["split"]
    assert sum(b["machines"] for b in s["bands"]) == s["train_rows"] and sum(b["twf"] for b in s["bands"]) == s["twf_in_training"]
    assert m["failure_modes"]["TWF"]["positives_train"] == s["twf_in_training"]
    const = next(c for c in s["candidates"] if c["key"] == "constant")
    assert const["pr_auc_mean"] == pytest.approx(s["chance_pr_auc"], abs=1e-3)


def test_wilson_interval_and_threshold_helper():
    from ml.evaluation import _tm
    from ml.twf_study import wilson
    import numpy as np
    lo, hi = wilson(5, 100)
    assert lo < 5 < hi and 0 <= lo and hi <= 100 and wilson(0, 0) == [None, None]
    x = _tm(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.6, 0.1]), 0.5)
    assert (x["tp"], x["fp"], x["fn"], x["tn"]) == (1, 1, 1, 1) and x["precision"] == 0.5 and x["recall"] == 0.5
