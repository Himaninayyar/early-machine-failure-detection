"""Individual models, the soft-voting ensemble, explanations and failure-mode classifiers (tiny real models)."""
import pickle

import numpy as np
import pandas as pd
import pytest

from app.predictor import ModelRegistry
from ml.ensemble import SoftVotingEnsemble
from ml.features import INPUT_COLS
from ml.pipelines import PortableXGBClassifier

READING = dict(type="L", air_temp=303.7, process_temp=312.1, rpm=1363, torque=51.8, tool_wear=90)


@pytest.fixture(scope="module")
def reg(tiny_models):
    return ModelRegistry(tiny_models)


@pytest.mark.parametrize("key", ["lr", "rf", "xgb", "ens"])
def test_every_model_returns_valid_probabilities(reg, key):
    frame = pd.DataFrame([READING, {**READING, "type": "M", "rpm": 1600}, {**READING, "type": "H", "torque": 30.0}])[INPUT_COLS]
    p = reg.get(key).predict_proba(frame)
    assert p.shape == (3, 2) and np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(axis=1), 1.0)


def test_ensemble_is_the_weighted_average_of_its_members(reg):
    frame = pd.DataFrame([READING])[INPUT_COLS]
    w = reg.meta["ensemble"]["weights"]
    expected = sum(w[k] * reg.models[k].predict_proba(frame)[0, 1] for k in ("lr", "rf", "xgb")) / sum(w.values())
    assert reg.models["ens"].predict_proba(frame)[0, 1] == pytest.approx(expected)


def test_ensemble_weights_are_normalised_and_validated():
    class M:
        def __init__(self, p): self.p = p
        def predict_proba(self, X): return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])
    ens = SoftVotingEnsemble({"a": M(0.2), "b": M(0.8)}, {"a": 3.0, "b": 1.0})
    assert ens.normalized_weights() == {"a": 0.75, "b": 0.25}
    assert ens.predict_proba(pd.DataFrame({"x": [1]}))[0, 1] == pytest.approx(0.35)
    with pytest.raises(ValueError):
        SoftVotingEnsemble({"a": M(0.2)}, {"a": 0.0}).normalized_weights()
    with pytest.raises(ValueError):
        SoftVotingEnsemble({}).normalized_weights()


def test_training_selected_the_default_model_from_validation_evidence(reg):
    sel = reg.metrics["selection"]
    assert sel["default_model"] == reg.meta["default_model"] and sel["default_model"] in sel["ranking"]
    assert all(set(r) <= {"model", "cv_pr_auc", "cv_roc_auc", "cv_brier", "cv_best_f1", "cv_best_f1_threshold",
                          "cv_recall_at_0.3", "cv_precision_at_0.3"} for r in sel["table"])   # no test-set keys in the selection evidence
    e = reg.metrics["ensemble"]
    assert pytest.approx(sum(e["weights"].values()), abs=0.01) == 1.0
    if e["tuned_weights_degenerate"]:
        assert not e["tuned_weights_adopted"]        # a one-model "ensemble" is never deployed


def test_portable_xgboost_pickle_roundtrip_is_exact_and_uses_json(reg):
    clf = reg.models["xgb"].calibrated_classifiers_[0].estimator.named_steps["clf"]
    assert isinstance(clf, PortableXGBClassifier)
    state = clf.__getstate__()
    assert "_Booster" not in state and state["_booster_json"].lstrip().startswith(b"{")
    clone = pickle.loads(pickle.dumps(reg.models["xgb"]))
    frame = pd.DataFrame([READING])[INPUT_COLS]
    assert np.allclose(clone.predict_proba(frame), reg.models["xgb"].predict_proba(frame))


@pytest.mark.parametrize("key", ["lr", "rf", "xgb", "ens"])
def test_shapley_contributions_add_up_exactly(reg, key):
    out = reg.explain(reg.models[key], READING)
    total = sum(c["contribution"] for c in out["contributions"])
    assert total == pytest.approx(out["prediction"] - out["base_value"], abs=1e-9)
    assert out["prediction"] == pytest.approx(reg.predict_proba(key, READING), abs=1e-9)
    assert {c["group"] for c in out["contributions"]} == {"Machine type", "Temperatures", "Rotational speed", "Torque", "Tool wear"}


def test_failure_mode_models_give_probabilities(reg):
    assert reg.modes is not None and set(reg.modes) <= {"TWF", "HDF", "PWF", "OSF"}
    est = reg._mode_estimates(pd.DataFrame([READING])[INPUT_COLS])[0]
    assert est and all(0 <= p <= 1 for p in est.values())
    assert "RNF" not in est                                    # random failures are deliberately not modelled


def test_analyze_returns_complete_result(reg):
    r = reg.analyze(reg.default_model, {**READING})
    assert {"probability", "risk_level", "health_score", "recommendation", "details"} <= set(r)
    assert r["health_score"] == round(100 * (1 - r["probability"]))
    assert r["details"]["contributions"] is not None and set(r["details"]) >= {"derived", "indicators", "mode_estimates"}
    assert "wear_torque" in r["details"]["derived"] and "strain" not in r["details"]["derived"]
