"""Tool-wear failure (TWF) study. Uses the TRAINING split only; the test set is never touched.

Question: can anything beat "tool wear is high" at telling which worn tools will fail?
  1. TWF rate by tool-wear band (training data), with 95% Wilson intervals.
  2. Repeated stratified cross-validation (5 folds x 5 repeats) of simple estimates against the shipped failure-mode model.

It also shows how to judge any future model change without using the test set: add a candidate to `candidates()` and re-run.

    python -m ml.twf_study            # writes reports/twf_study.json (shown on the Model Insights page)
"""
import json
import os

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.ensemble import SoftVotingEnsemble  # noqa: F401  (keeps pickled classes importable)
from ml.features import INPUT_COLS, load_data

SEED = 42
BANDS = [(0, 150), (150, 200), (200, 210), (210, 220), (220, 230), (230, 240), (240, 300)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0, c - h)) * 100, 1), round(float(min(1, c + h)) * 100, 1)]


def band_table(df):
    rows = []
    for lo, hi in BANDS:
        m = (df["tool_wear"] >= lo) & (df["tool_wear"] < hi)
        n, k, f = int(m.sum()), int(df.loc[m, "TWF"].sum()), int(df.loc[m, "failure"].sum())
        rows.append({"band": f"{lo}-{hi}" if hi < 300 else f"{lo}+", "machines": n, "twf": k,
                     "twf_rate_pct": round(100 * k / n, 1) if n else None, "twf_ci95_pct": wilson(k, n),
                     "any_failure_rate_pct": round(100 * f / n, 1) if n else None})
    return rows


def _hinge(X):
    X = np.asarray(X, dtype=float)
    w = X[:, [0]]
    return np.hstack([(w >= 200).astype(float), np.maximum(w - 200, 0)])


def _hinge_torque(X):
    X = np.asarray(X, dtype=float)
    w, tq = X[:, [0]], X[:, [1]]
    return np.hstack([(w >= 200).astype(float), np.maximum(w - 200, 0), tq, np.maximum(w - 200, 0) * tq])


def candidates(rf_base):
    """name -> (estimator that returns a score, columns it uses, short description)."""
    lr = lambda f: make_pipeline(FunctionTransformer(f), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))  # noqa: E731
    return {
        "constant": (None, [], "Always the training TWF rate (no information)"),
        "wear_200": (None, ["tool_wear"], "Binary: tool wear 200 min or more"),
        "wear_only": (None, ["tool_wear"], "Score = tool wear itself (higher = riskier)"),
        "hinge_lr": (lr(_hinge), ["tool_wear"], "Logistic regression on 'wear >= 200' and minutes above 200"),
        "hinge_torque_lr": (lr(_hinge_torque), ["tool_wear", "torque"], "Same plus torque and (minutes above 200) x torque"),
        "shipped_rf": (rf_base, INPUT_COLS, "Shipped failure-mode model: Random Forest on all readings and derived features"),
    }


def evaluate(df_tr, rf_base, repeats=5, folds=5):
    y = df_tr["TWF"].values
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    names = candidates(rf_base)
    res = {k: {"ap": [], "roc": []} for k in names}
    for tr, te in cv.split(df_tr, y):
        for name, (est, cols, _) in names.items():
            if name == "constant":
                score = np.full(len(te), y[tr].mean())
            elif name == "wear_200":
                score = (df_tr.iloc[te]["tool_wear"].values >= 200).astype(float)
            elif name == "wear_only":
                score = df_tr.iloc[te]["tool_wear"].values.astype(float)
            else:
                m = clone(est).fit(df_tr.iloc[tr][cols], y[tr])
                score = m.predict_proba(df_tr.iloc[te][cols])[:, 1]
            res[name]["ap"].append(average_precision_score(y[te], score))
            res[name]["roc"].append(roc_auc_score(y[te], score) if len(set(score)) > 1 else 0.5)
    out = []
    for name, (_, cols, desc) in names.items():
        ap, roc = np.array(res[name]["ap"]), np.array(res[name]["roc"])
        out.append({"key": name, "description": desc, "pr_auc_mean": round(float(ap.mean()), 4), "pr_auc_sd": round(float(ap.std()), 4),
                    "roc_auc_mean": round(float(roc.mean()), 4), "folds": int(len(ap))})
    return out


def build(data_path="data/raw/ai4i2020.csv", models_dir="models", out_path="reports/twf_study.json"):
    import joblib
    df = load_data(data_path)
    idx_tr, _ = train_test_split(df.index, test_size=0.2, stratify=df["failure"].values, random_state=SEED)
    tr = df.loc[idx_tr]
    modes = joblib.load(os.path.join(models_dir, "modes.joblib"))
    rf_base = modes["TWF"].estimator            # unfitted pipeline with the shipped settings
    cands = evaluate(tr, rf_base)
    prevalence = float(tr["TWF"].mean())
    simple = [c for c in cands if c["key"] in ("wear_only", "hinge_lr", "hinge_torque_lr")]
    best = max(simple, key=lambda c: c["pr_auc_mean"])
    rf = next(c for c in cands if c["key"] == "shipped_rf")
    noise = max(best["pr_auc_sd"], rf["pr_auc_sd"])
    diff = best["pr_auc_mean"] - rf["pr_auc_mean"]
    if abs(diff) <= noise:
        verdict = (f"The best simple estimate ({best['key']}, PR-AUC {best['pr_auc_mean']:.3f}) and the shipped model ({rf['pr_auc_mean']:.3f}) differ by "
                   f"{abs(diff):.3f}, less than the fold-to-fold spread ({noise:.3f}). No candidate is reliably better.")
    elif diff > 0:
        verdict = (f"The best simple estimate ({best['key']}, PR-AUC {best['pr_auc_mean']:.3f}) beats the shipped model ({rf['pr_auc_mean']:.3f}) by more than the "
                   f"fold-to-fold spread ({noise:.3f}), but all of them stay far below the level needed to trust a TWF forecast.")
    else:
        verdict = (f"The shipped model ({rf['pr_auc_mean']:.3f}) beats every simple estimate (best {best['key']} {best['pr_auc_mean']:.3f}) by more than the "
                   f"fold-to-fold spread ({noise:.3f}), but it stays far below the level needed to trust a TWF forecast.")
    out = {"split": "training only (test set not used)", "train_rows": int(len(tr)), "twf_in_training": int(tr["TWF"].sum()),
           "twf_rate_pct": round(100 * prevalence, 2), "bands": band_table(tr), "candidates": cands, "chance_pr_auc": round(prevalence, 4),
           "verdict": verdict,
           "conclusion": "In this dataset a tool fails from wear at a random point once it is worn, so the readings only say that risk rises with wear "
                         "above about 200 min. With so few TWF examples no model can do better, so TWF is shown as a maintenance notice, "
                         "not as a forecast."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    r = build()
    print(json.dumps({k: r[k] for k in ("twf_in_training", "chance_pr_auc", "verdict")}, indent=1))
    for c in r["candidates"]:
        print(f"{c['key']:16s} PR-AUC {c['pr_auc_mean']:.3f} +/- {c['pr_auc_sd']:.3f}  ROC {c['roc_auc_mean']:.3f}")
    for b in r["bands"]:
        print(b["band"], b["machines"], b["twf"], b["twf_rate_pct"], b["twf_ci95_pct"])
