"""Shared fixtures.

tiny_models  : real (but tiny, ~30 s) trained artefacts built once per test session with ml.train --profile tiny
FixedModel   : stub with a fixed probability, so Normal / Warning / Critical flows can be tested deterministically
"""
import io
import os
import shutil

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "ai4i2020.csv")

NORMAL = dict(machine_id="PUMP-1", type="L", air_temp="303.7", process_temp="312.4", rpm="1513", torque="40.1", tool_wear="135")
HEAT = dict(machine_id="PUMP-1", type="L", air_temp="303.7", process_temp="312.1", rpm="1363", torque="51.8", tool_wear="90")


class FixedModel:
    """Stands in for a trained model; every row gets failure probability p."""
    classes_ = np.array([0, 1])

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    from ml import train
    out = str(tmp_path_factory.mktemp("tiny_models"))
    train.run("tiny", out, DATA)
    return out


@pytest.fixture()
def models_copy(tiny_models, tmp_path):
    """A private copy of the tiny artefacts that a test may damage."""
    dest = str(tmp_path / "models")
    shutil.copytree(tiny_models, dest)
    return dest


def make_app(models_dir, db_path, **extra):
    from app import create_app
    return create_app({"TESTING": True, "SECRET_KEY": "test-secret", "DATABASE": str(db_path), "MODELS_DIR": models_dir,
                       "REPORTS_DIR": os.path.join(ROOT, "reports"), **extra})


@pytest.fixture()
def app(tiny_models, tmp_path):
    return make_app(tiny_models, tmp_path / "t.db")


@pytest.fixture()
def client(app):
    c = app.test_client()
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    return c


def post(client, path, data=None, **kw):
    return client.post(path, data={**(data or {}), "csrf_token": "tok"}, **kw)


def set_probability(app, p):
    """Make every model return probability p (deterministic risk level)."""
    reg = app.extensions["models"]
    for k in list(reg.models):
        reg.models[k] = FixedModel(p)


def csv_file(text, name="batch.csv"):
    return (io.BytesIO(text.encode()), name)
