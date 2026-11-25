"""Per-failure-type evaluation, missed-failure analysis and precision-recall curves.

Added to metrics.json under "evaluation" (shown on Model Insights):
  per_type        for TWF, HDF, PWF, OSF: precision / recall / F1 and a confusion matrix on the test set (mode model at 0.5),
                  a threshold comparison from cross-validation on the TRAINING split (not the test set), PR curves (test and cross-validated)
  overall_by_type how many failures of each type the overall model (default, threshold 30%) catches
  missed          every test failure the default model scored below 30%: readings, probability, recorded failure types,
                  and whether the tool-wear maintenance notice would have flagged it
  notice          how the tool-wear notice relates to real failures in the test set

Training (ml/train.py) calls build(). To add this to the SHIPPED models without retraining:

    python -m ml.evaluation
"""
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ml.ensemble import SoftVotingEnsemble
from ml.features import INPUT_COLS, load_data

SEED = 42
TYPES = ["TWF", "HDF", "PWF", "OSF"]          # RNF (random failures) cannot be modelled, see risk.RNF_NOTE
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
NOTICE_WEAR = 200                             # same limit as app/risk.py TOOL_WEAR_NOTICE_MIN


def _tm(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(pr, 4), "recall": round(rc, 4),
            "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}


def _curve(y, p, n=60):
    if y.sum() == 0:
        return None
    pr, rc, _ = precision_recall_curve(y, p)
    idx = np.unique(np.linspace(0, len(pr) - 1, min(n, len(pr))).astype(int))
    return {"recall": [round(float(rc[i]), 4) for i in idx], "precision": [round(float(pr[i]), 4) for i in idx]}


def build(df, idx_tr, idx_te, p_default_te, default_key, mode_models):
    """df has the renamed columns of ml.features.load_data plus failure and TWF/HDF/PWF/OSF/RNF flags."""
    Xtr, Xte = df.loc[idx_tr, INPUT_COLS], df.loc[idx_te, INPUT_COLS]
    yte = df.loc[idx_te, "failure"].values
    per_type = {}
    for code in TYPES:
        cal = mode_models.get(code)
        ytr_m, yte_m = df.loc[idx_tr, code].values, df.loc[idx_te, code].values
        if cal is None or ytr_m.sum() < 3:
            per_type[code] = {"available": False, "positives_train": int(ytr_m.sum()), "positives_test": int(yte_m.sum())}
            continue
        pm = cal.predict_proba(Xte)[:, 1]
        oof = cross_val_predict(clone(cal.estimator), Xtr, ytr_m, cv=StratifiedKFold(min(5, int(ytr_m.sum())), shuffle=True, random_state=SEED),
                                method="predict_proba")[:, 1]
        per_type[code] = {
            "available": True, "positives_train": int(ytr_m.sum()), "positives_test": int(yte_m.sum()),
            "test_at_0.5": _tm(yte_m, pm, 0.5),
            "test_pr_auc": round(float(average_precision_score(yte_m, pm)), 4) if yte_m.sum() else None,
            "cv_pr_auc": round(float(average_precision_score(ytr_m, oof)), 4),
            "cv_thresholds": [_tm(ytr_m, oof, t) for t in THRESHOLDS],
            "pr_curve_test": _curve(yte_m, pm), "pr_curve_cv": _curve(ytr_m, oof),
            "chance": round(float(ytr_m.mean()), 4),
        }
    flags = df.loc[idx_te, ["TWF", "HDF", "PWF", "OSF", "RNF"]]
    pred30 = p_default_te >= 0.3
    overall = {}
    for code in TYPES + ["RNF"]:
        m = flags[code].values == 1
        overall[code] = {"failures": int(m.sum()), "caught": int((m & pred30).sum()),
                         "recall": round(float((m & pred30).sum() / m.sum()), 4) if m.sum() else None}
    X = df.loc[idx_te]
    miss_idx = np.where((yte == 1) & ~pred30)[0]
    missed_rows = []
    for i in miss_idx:
        r = X.iloc[i]
        modes = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if r[c] == 1]
        missed_rows.append({"type": r["type"], "air_temp": float(r["air_temp"]), "process_temp": float(r["process_temp"]), "rpm": float(r["rpm"]),
                            "torque": float(r["torque"]), "tool_wear": float(r["tool_wear"]), "probability": round(float(p_default_te[i]), 4),
                            "modes": modes, "notice": bool(r["tool_wear"] >= NOTICE_WEAR)})
    missed_rows.sort(key=lambda r: -r["probability"])
    wear_hi = X["tool_wear"].values >= NOTICE_WEAR
    return {
        "model": default_key, "threshold": 0.3, "per_type": per_type, "overall_by_type": overall,
        "missed": {"total": int(len(missed_rows)), "failures": int(yte.sum()),
                   "with_tool_wear_notice": int(sum(r["notice"] for r in missed_rows)), "rows": missed_rows},
        "notice": {"limit_min": NOTICE_WEAR, "test_machines_with_notice": int(wear_hi.sum()),
                   "test_failures_with_notice": int(((yte == 1) & wear_hi).sum()), "test_failures": int(yte.sum()),
                   "failure_rate_with_notice_pct": round(100 * float(yte[wear_hi].mean()), 1) if wear_hi.sum() else None,
                   "failure_rate_without_notice_pct": round(100 * float(yte[~wear_hi].mean()), 1) if (~wear_hi).sum() else None},
        "note": "Test-set figures are descriptive: each type has few failures in the test set (see counts), so they carry wide uncertainty. "
                "Threshold comparisons use cross-validation on the training split only.",
    }


def _default_probs(models_dir, meta, Xte):
    key = meta["default_model"]
    load = lambda k: joblib.load(os.path.join(models_dir, meta["artifacts"][k]["file"]))  # noqa: E731
    if key == "ens":
        members = meta["ensemble"]["members"]
        model = SoftVotingEnsemble({m: load(m) for m in members}, meta["ensemble"]["weights"])
    else:
        model = load(key)
    return key, model.predict_proba(Xte)[:, 1]


def patch_metrics(models_dir="models", data_path="data/raw/ai4i2020.csv"):
    """Add 'evaluation' to an existing metrics.json using the same split as training (full profile), without retraining."""
    mpath = os.path.join(models_dir, "metrics.json")
    metrics = json.load(open(mpath))
    meta = json.load(open(os.path.join(models_dir, "meta.json")))
    df = load_data(data_path)
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=df["failure"].values, random_state=SEED)
    ds = metrics["dataset"]
    if len(idx_te) != ds["test_rows"] or int(df.loc[idx_te, "failure"].sum()) != ds["test_failures"]:
        raise SystemExit("The recomputed test split does not match metrics.json (different data or a non-full profile). Retrain instead.")
    key, p = _default_probs(models_dir, meta, df.loc[idx_te, INPUT_COLS])
    modes = joblib.load(os.path.join(models_dir, meta["artifacts"]["modes"]["file"]))
    metrics["evaluation"] = build(df, idx_tr, idx_te, p, key, modes)
    tmp = mpath + ".tmp"
    json.dump(metrics, open(tmp, "w"), indent=1)
    os.replace(tmp, mpath)
    return metrics["evaluation"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--data", default="data/raw/ai4i2020.csv")
    a = ap.parse_args()
    e = patch_metrics(a.models, a.data)
    print(json.dumps({"overall_by_type": e["overall_by_type"], "missed": {k: v for k, v in e["missed"].items() if k != "rows"}, "notice": e["notice"]}, indent=1))
    for t, v in e["per_type"].items():
        print(t, v["test_at_0.5"] if v.get("available") else "unavailable")
