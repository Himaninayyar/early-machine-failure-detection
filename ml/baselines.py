"""Simple baselines the trained models must beat (no learning, nothing tuned on the test set).

  never_fails : always predicts "no failure". With 3.4% failures its accuracy looks great (96.6%) but it finds nothing.
  rule_based  : flags a machine if ANY documented AI4I failure condition holds (HDF, PWF, OSF, or tool wear >= 200 min).
                These are the same rules the app shows as "rule indicators" (app/risk.py); here they are vectorised.
  rule_physical : the same without the tool-wear flag (HDF, PWF, OSF only). Both rule baselines are reported; neither is tuned.

Called by ml/train.py (stored in metrics.json under "baselines"). To add them to the SHIPPED models without retraining:

    python -m ml.baselines            # recomputes the same 80/20 split (seed 42) and patches models/metrics.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ml.ensemble import SoftVotingEnsemble  # noqa: F401  (keeps pickled classes importable when this module is run directly)
from ml.features import INPUT_COLS, load_data

SEED = 42
OSF_LIMITS = {"L": 11000, "M": 12000, "H": 13000}  # same as app/risk.py


def rule_flags(X, include_tool_wear=True):
    """Boolean array: does any documented failure condition hold? X has the INPUT_COLS columns.
    include_tool_wear=False leaves out the 'tool wear >= 200 min' flag (the one condition that is only partly deterministic)."""
    temp_diff = X["process_temp"] - X["air_temp"]
    power = X["torque"] * X["rpm"] * 2 * np.pi / 60
    wear_torque = X["tool_wear"] * X["torque"]
    hdf = (temp_diff < 8.6) & (X["rpm"] < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = wear_torque > X["type"].map(OSF_LIMITS)
    twf = X["tool_wear"] >= 200
    flags = hdf | pwf | osf
    return (flags | twf).to_numpy() if include_tool_wear else flags.to_numpy()


def _scores(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4), "accuracy": round(float((tn + tp) / len(y)), 4)}


def compute(Xte, yte):
    """Baseline results on the held-out test rows."""
    yte = np.asarray(yte)
    return {
        "never_fails": {"label": "Always predict 'no failure'", **_scores(yte, np.zeros(len(yte), dtype=int)),
                        "note": "High accuracy, but it never finds a failure."},
        "rule_based": {"label": "All documented failure rules (HDF, PWF, OSF, tool wear >= 200 min)", **_scores(yte, rule_flags(Xte).astype(int)),
                       "note": "No machine learning. Tool wear >= 200 min is common, so this flags many healthy machines (low precision)."},
        "rule_physical": {"label": "Documented rules without tool wear (HDF, PWF, OSF only)", **_scores(yte, rule_flags(Xte, False).astype(int)),
                          "note": "No machine learning. Shows how much of the target is a fixed rule in this synthetic dataset."},
        "test_rows": int(len(yte)), "test_failures": int(yte.sum()),
    }


def patch_metrics(models_dir="models", data_path="data/raw/ai4i2020.csv"):
    """Add 'baselines' to an existing metrics.json using the same split the training run used (full profile)."""
    path = os.path.join(models_dir, "metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    df = load_data(data_path)
    _, idx_te = train_test_split(df.index, test_size=0.2, stratify=df["failure"].values, random_state=SEED)
    ds = metrics["dataset"]
    if len(idx_te) != ds["test_rows"] or int(df.loc[idx_te, "failure"].sum()) != ds["test_failures"]:
        raise SystemExit("The recomputed test split does not match metrics.json (different data or a non-full profile). Retrain instead.")
    metrics["baselines"] = compute(df.loc[idx_te, INPUT_COLS], df.loc[idx_te, "failure"].values)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=1)
    os.replace(tmp, path)
    return metrics["baselines"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--data", default="data/raw/ai4i2020.csv")
    a = ap.parse_args()
    print(json.dumps(patch_metrics(a.models, a.data), indent=1))
