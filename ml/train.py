"""Train, tune, calibrate and evaluate the models.

Run from the project root:
    python -m ml.train                      # full run (about 10-15 minutes on one CPU core)
    python -m ml.train --profile quick      # smaller search budgets (smoke test on the real data)
    python -m ml.train --profile tiny --out /tmp/m   # seconds; used by the automated tests

Every CHOICE (feature set, imbalance strategy, hyperparameters, calibration method, ensemble weights,
final default model) is made on the TRAINING split only, using cross-validation. The held-out test
split is used once, at the end, only to REPORT performance.

Outputs in --out (default models/):
    lr.joblib, rf.joblib, xgb.joblib  fitted, calibrated models (XGBoost stored in portable JSON form)
    modes.joblib                      failure-mode classifiers (TWF, HDF, PWF, OSF)
    meta.json                         version, checksums, ensemble weights, ranges, background sample
    metrics.json                      all results shown on the Model Insights page
The soft-voting ensemble is NOT saved as a file: it is rebuilt from the members + weights in meta.json.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import sklearn
import xgboost
import imblearn
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)

from ml.ensemble import SoftVotingEnsemble
from ml import baselines, evaluation
from ml.data_io import ensure_csv
from ml.features import INPUT_COLS, NUMERIC_RAW, data_quality_report, load_data
from ml.pipelines import BASE_MODELS, MODEL_NAMES, SEED, STRATEGIES, build_pipeline, feature_names

warnings.filterwarnings("ignore")
ALERT_THRESHOLDS = [0.3, 0.5, 0.7]  # 0.3 / 0.7 are the Warning / Critical cut-offs from the project proposal
SCAN_THRESHOLDS = [round(float(t), 1) for t in np.arange(0.1, 0.95, 0.1)]
MODES = ["TWF", "HDF", "PWF", "OSF"]  # RNF is random by design and is deliberately NOT modelled

PROFILES = {
    "full":  dict(repeats=2, n_iter={"lr": 15, "rf": 15, "xgb": 30}, rf_trees=[200, 300, 400], xgb_trees=[100, 200, 300, 500],
                  outer_cv=5, cal_cv=5, boot=1000, perm=10, subsample=None, mode_trees=300),
    "quick": dict(repeats=1, n_iter={"lr": 5, "rf": 3, "xgb": 5}, rf_trees=[200, 300, 400], xgb_trees=[100, 200, 300, 500],
                  outer_cv=5, cal_cv=5, boot=200, perm=5, subsample=None, mode_trees=150),
    "tiny":  dict(repeats=1, n_iter={"lr": 2, "rf": 2, "xgb": 2}, rf_trees=[20, 30], xgb_trees=[20, 40],
                  outer_cv=3, cal_cv=3, boot=30, perm=2, subsample=3000, mode_trees=30),
}


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- metric helpers
def best_f1(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(th[i])


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0, "accuracy": round((tn + tp) / len(y), 4)}


def band_table(y, p):
    """How often do machines in each alert band actually fail? (a direct check of the 30/70 thresholds)"""
    rows = []
    for name, lo, hi in [("Normal", 0.0, 0.3), ("Warning", 0.3, 0.7), ("Critical", 0.7, 1.0001)]:
        m = (p >= lo) & (p < hi)
        rows.append({"level": name, "machines": int(m.sum()), "failures": int(y[m].sum()),
                     "observed_failure_rate": round(float(y[m].mean()), 4) if m.sum() else None})
    return rows


def downsample(a, b, n=150):
    if len(a) <= n:
        return [round(float(x), 5) for x in a], [round(float(x), 5) for x in b]
    idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return [round(float(a[i]), 5) for i in idx], [round(float(b[i]), 5) for i in idx]


def fold_metrics(y, p):
    f1, _ = best_f1(y, p)
    return {"ap": average_precision_score(y, p), "roc": roc_auc_score(y, p), "best_f1": f1}


def summarize_probs(y, p):
    f1, thr = best_f1(y, p)
    return {"pr_auc": round(float(average_precision_score(y, p)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 5),
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
            "best_f1": round(f1, 4), "best_f1_threshold": round(thr, 3)}


def threshold_scan(y, p):
    keep = ("precision", "recall", "f1", "tp", "fp", "fn")
    return [{"threshold": t, **{k: v for k, v in threshold_metrics(y, p, t).items() if k in keep},
             "flagged": int((p >= t).sum())} for t in SCAN_THRESHOLDS]


def native_importance(pipe, key):
    names = feature_names(pipe)
    clf = pipe.named_steps["clf"]
    if key == "lr":
        vals, kind = clf.coef_[0], "Standardised coefficient (log-odds per 1 SD)"
    elif key == "rf":
        vals, kind = clf.feature_importances_, "Impurity-based importance"
    else:
        vals, kind = clf.feature_importances_, "Gain-based importance"
    rows = sorted(({"feature": n, "value": round(float(v), 4)} for n, v in zip(names, vals)), key=lambda r: -abs(r["value"]))
    return {"kind": kind, "values": rows}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_atomic(obj, path):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- ensemble weights
def simplex_grid(step=10):
    for a in range(step + 1):
        for b in range(step + 1 - a):
            yield (a / step, b / step, (step - a - b) / step)


def choose_ensemble_weights(P, y):
    """P: out-of-fold calibrated probabilities (n x 3). Only TRAINING data is used here.

    Candidates: equal weights, and a coarse grid of weights (step 0.1). Tuned weights are adopted only if
    (a) PR-AUC improves by >= 0.005 over equal weights on the out-of-fold predictions AND
    (b) the improvement also holds in a split-half check (weights chosen on one half of the rows are
        evaluated on the other half), which guards against over-fitting the weights AND
    (c) the weights are a real blend: if one member gets weight >= 0.9 the "ensemble" would just be that
        model again, so equal weights are kept (and the tuned result is reported as an experiment).
    """
    def ap(p, yy):
        return float(average_precision_score(yy, p))

    equal = np.full(3, 1 / 3)
    grid = []
    for w in simplex_grid():
        p = P @ np.array(w)
        grid.append({"weights": list(w), "pr_auc": ap(p, y), "brier": float(brier_score_loss(y, p))})
    grid.sort(key=lambda r: (-r["pr_auc"], r["brier"]))
    best = grid[0]
    equal_ap = ap(P @ equal, y)
    gain = best["pr_auc"] - equal_ap

    gains = []
    for a, b in StratifiedKFold(2, shuffle=True, random_state=SEED).split(P, y):
        cand = max(simplex_grid(), key=lambda w: ap(P[a] @ np.array(w), y[a]))
        gains.append(ap(P[b] @ np.array(cand), y[b]) - ap(P[b] @ equal, y[b]))
    split_gain = float(np.mean(gains))
    degenerate = bool(max(best["weights"]) >= 0.9)  # one member carries (almost) all the weight: not a real ensemble
    adopted = bool(gain >= 0.005 and split_gain > 0 and not degenerate)
    weights = np.array(best["weights"]) if adopted else equal
    return weights, {
        "equal": {"weights": [round(float(x), 4) for x in equal], "oof_pr_auc": round(equal_ap, 4),
                  "oof_brier": round(float(brier_score_loss(y, P @ equal)), 5)},
        "best_on_oof": {"weights": best["weights"], "oof_pr_auc": round(best["pr_auc"], 4), "oof_brier": round(best["brier"], 5)},
        "gain_on_oof": round(gain, 4), "split_half_gain": round(split_gain, 4), "tuned_weights_adopted": adopted, "tuned_weights_degenerate": degenerate,
        "top_weight_sets": [{"weights": g["weights"], "oof_pr_auc": round(g["pr_auc"], 4), "oof_brier": round(g["brier"], 5)} for g in grid[:5]],
        "rule": "Validation-selected weights replace equal weights only if they beat them by >= 0.005 OOF PR-AUC, the gain also holds on a split-half check, and they are a real blend (no member has weight >= 0.9). Otherwise the ensemble uses equal weights.",
    }


# ----------------------------------------------------------------------------- main
def run(profile="full", out_dir="models", data_path="data/raw/ai4i2020.csv"):
    t0 = time.time()
    cfg = PROFILES[profile]
    os.makedirs(out_dir, exist_ok=True)
    data_path = ensure_csv(data_path)   # .xlsx -> .csv; everything below reads/hashes the CSV
    df = load_data(data_path)
    dq = data_quality_report(df)
    log(f"data quality: {dq}")
    if not dq["ok"]:
        raise ValueError(f"Data quality check failed: {dq}")
    if cfg["subsample"]:
        df = df.sample(cfg["subsample"], random_state=SEED)
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=df["failure"].values, random_state=SEED)
    Xtr, Xte = df.loc[idx_tr, INPUT_COLS], df.loc[idx_te, INPUT_COLS]
    ytr, yte = df.loc[idx_tr, "failure"].values, df.loc[idx_te, "failure"].values
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    log(f"profile={profile} | train {len(Xtr)} ({ytr.sum()} failures) | test {len(Xte)} ({yte.sum()} failures)")

    # Resumable: every expensive unit is checkpointed. If the run is interrupted, running the same command again
    # continues where it stopped. The cache is keyed on the profile, the data and the training code, so a stale
    # checkpoint can never be reused after the code changes. It is deleted when the run completes.
    src = "".join(open(os.path.join(os.path.dirname(__file__), f)).read() for f in ("train.py", "pipelines.py", "features.py", "ensemble.py"))
    cache_key = hashlib.sha256((profile + sha256_of(data_path) + src).encode()).hexdigest()
    cache_dir = os.path.join(out_dir, ".train_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def cached(name, compute):
        path = os.path.join(cache_dir, name + ".joblib")
        if os.path.exists(path):
            try:
                key, val = joblib.load(path)
                if key == cache_key:
                    return val
            except Exception:  # noqa: BLE001 - unreadable checkpoint: just recompute
                pass
        val = compute()
        joblib.dump((cache_key, val), path + ".tmp")
        os.replace(path + ".tmp", path)
        return val

    # ---------------------------------------------------------------- A. feature set + imbalance strategy (CV on train)
    cv_a = RepeatedStratifiedKFold(n_splits=cfg["outer_cv"], n_repeats=cfg["repeats"], random_state=SEED)
    def compute_a(model, use_wt, strategy):
        fm = []
        for tr, va in cv_a.split(Xtr, ytr):
            pipe = build_pipeline(model, strategy, use_wt, scale_pos_weight=spw)
            pipe.fit(Xtr.iloc[tr], ytr[tr])
            fm.append(fold_metrics(ytr[va], pipe.predict_proba(Xtr.iloc[va])[:, 1]))
        row = {"model": model, "use_wear_torque": use_wt, "strategy": strategy}
        for k in ("ap", "roc", "best_f1"):
            row[k] = round(float(np.mean([f[k] for f in fm])), 4)
            row[k + "_std"] = round(float(np.std([f[k] for f in fm])), 4)
        return row

    rows = []
    for model in BASE_MODELS:
        for use_wt in (False, True):
            for strategy in STRATEGIES:
                row = cached(f"A_{model}_{use_wt}_{strategy}", lambda: compute_a(model, use_wt, strategy))
                rows.append(row)
                log(f"[A] {model:3s} wt={use_wt!s:5s} {strategy:12s} AP={row['ap']:.3f}+-{row['ap_std']:.3f} ROC={row['roc']:.3f} bestF1={row['best_f1']:.3f}")

    def best_row(model, use_wt):
        cand = [r for r in rows if r["model"] == model and r["use_wear_torque"] == use_wt]
        top = max(r["ap"] for r in cand)
        near = [r for r in cand if top - r["ap"] <= 0.005]  # within 0.005 AP prefer the simpler strategy
        return min(near, key=lambda r: STRATEGIES.index(r["strategy"]))

    gain = float(np.mean([best_row(m, True)["ap"] - best_row(m, False)["ap"] for m in BASE_MODELS]))
    use_wt = bool(gain > 0.01)
    chosen_strategy = {m: best_row(m, use_wt)["strategy"] for m in BASE_MODELS}
    log(f"[A] mean AP gain from wear*torque proxy = {gain:+.3f} -> use_wear_torque={use_wt}; strategies: {chosen_strategy}")

    # ---------------------------------------------------------------- B. hyperparameter tuning (CV on train)
    space = {
        "lr": {"clf__C": loguniform(1e-2, 1e2)},
        "rf": {"clf__n_estimators": cfg["rf_trees"], "clf__max_depth": [None, 6, 10, 16],
               "clf__min_samples_leaf": [1, 2, 4, 8], "clf__max_features": ["sqrt", 0.5]},
        "xgb": {"clf__n_estimators": cfg["xgb_trees"], "clf__max_depth": [2, 3, 4, 6],
                "clf__learning_rate": [0.02, 0.05, 0.1, 0.2], "clf__subsample": [0.7, 0.85, 1.0],
                "clf__colsample_bytree": [0.7, 0.85, 1.0], "clf__min_child_weight": [1, 3, 5], "clf__reg_lambda": [1, 5, 10]},
    }
    cv_b = StratifiedKFold(cfg["outer_cv"], shuffle=True, random_state=SEED)
    tuned, tuning = {}, {}

    def compute_b(model):
        base = build_pipeline(model, chosen_strategy[model], use_wt, scale_pos_weight=spw)
        rs = RandomizedSearchCV(base, space[model], n_iter=cfg["n_iter"][model], scoring="average_precision",
                                cv=cv_b, random_state=SEED, n_jobs=1, refit=True)
        rs.fit(Xtr, ytr)
        info = {"strategy": chosen_strategy[model], "n_iter": cfg["n_iter"][model],
                "best_cv_pr_auc": round(float(rs.best_score_), 4),
                "best_params": {k.replace("clf__", ""): (float(v) if isinstance(v, (np.floating, float)) else v)
                                for k, v in rs.best_params_.items()}}
        return rs.best_estimator_, info

    for model in BASE_MODELS:
        tuned[model], tuning[model] = cached(f"B_{model}", lambda: compute_b(model))
        log(f"[B] {model}: cv PR-AUC={tuning[model]['best_cv_pr_auc']:.4f} params={tuning[model]['best_params']}")

    # ---------------------------------------------------------------- C. calibration (out-of-fold, train only)
    cv_c = StratifiedKFold(cfg["outer_cv"], shuffle=True, random_state=SEED + 1)  # same folds for every model
    calibration, final_models, chosen_cal, oof = {}, {}, {}, {}

    def compute_c(model):
        res, oof_by_method = [], {}
        for method in ("none", "sigmoid", "isotonic"):
            est = clone(tuned[model]) if method == "none" else CalibratedClassifierCV(
                estimator=clone(tuned[model]), method=method, cv=cfg["cal_cv"])
            p = cross_val_predict(est, Xtr, ytr, cv=cv_c, method="predict_proba")[:, 1]
            oof_by_method[method] = p
            res.append({"method": method, **summarize_probs(ytr, p)})
        best = min((r for r in res if r["method"] != "none"), key=lambda r: r["brier"])
        final = CalibratedClassifierCV(estimator=clone(tuned[model]), method=best["method"], cv=cfg["cal_cv"]).fit(Xtr, ytr)
        return res, best["method"], oof_by_method[best["method"]], final

    for model in BASE_MODELS:
        res, method, oof[model], final_models[model] = cached(f"C_{model}", lambda: compute_c(model))
        chosen_cal[model] = method
        calibration[model] = {"options": res, "chosen": method}
        for r in res:
            log(f"[C] {model:3s} {r['method']:8s} OOF PR-AUC={r['pr_auc']:.3f} Brier={r['brier']:.5f}")

    # ---------------------------------------------------------------- D. soft-voting ensemble (weights from OOF only)
    P = np.column_stack([oof[m] for m in BASE_MODELS])
    w, ens_info = choose_ensemble_weights(P, ytr)
    ens_weights = {m: float(x) for m, x in zip(BASE_MODELS, w)}
    oof["ens"] = P @ w
    ens_info["members"] = BASE_MODELS
    ens_info["weights"] = {m: round(x, 4) for m, x in ens_weights.items()}
    log(f"[D] ensemble weights {ens_info['weights']} adopted_tuned={ens_info['tuned_weights_adopted']} "
        f"(oof gain {ens_info['gain_on_oof']:+.4f}, split-half {ens_info['split_half_gain']:+.4f})")
    final_models["ens"] = SoftVotingEnsemble({m: final_models[m] for m in BASE_MODELS}, ens_weights)

    # ---------------------------------------------------------------- E. model selection (train-only evidence)
    order = ["lr", "rf", "xgb", "ens"]
    sel = []
    for m in order:
        s = summarize_probs(ytr, oof[m])
        t3 = threshold_metrics(ytr, oof[m], 0.3)
        sel.append({"model": m, "cv_pr_auc": s["pr_auc"], "cv_roc_auc": s["roc_auc"], "cv_brier": s["brier"],
                    "cv_best_f1": s["best_f1"], "cv_best_f1_threshold": s["best_f1_threshold"],
                    "cv_recall_at_0.3": t3["recall"], "cv_precision_at_0.3": t3["precision"]})
    top_ap = max(s["cv_pr_auc"] for s in sel)
    near = [s for s in sel if top_ap - s["cv_pr_auc"] <= 0.01]
    # near-ties: better calibrated (lower Brier) wins, because the alert thresholds rely on honest probabilities;
    # still tied -> the simpler model (order lr < rf < xgb < ens)
    default_model = min(near, key=lambda s: (s["cv_brier"], order.index(s["model"])))["model"]
    ranking = [s["model"] for s in sorted(sel, key=lambda s: (-s["cv_pr_auc"], s["cv_brier"]))]
    thresholds_oof = {m: threshold_scan(ytr, oof[m]) for m in order}
    log(f"[E] default model = {default_model} | ranking {ranking}")

    # ---------------------------------------------------------------- F. one-time test evaluation
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, len(yte), len(yte)) for _ in range(cfg["boot"])]
    test = {}
    for model in order:
        est = final_models[model]
        p = est.predict_proba(Xte)[:, 1]
        pu = tuned[model].predict_proba(Xte)[:, 1] if model in tuned else None
        fpr, tpr, _ = roc_curve(yte, p)
        pr_, rc_, _ = precision_recall_curve(yte, p)
        edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
        rel = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m_ = (p >= lo) & (p < hi)
            if m_.sum():
                rel.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m_.sum()),
                            "mean_predicted": round(float(p[m_].mean()), 4), "observed": round(float(yte[m_].mean()), 4)})
        boots = {"pr_auc": [], "roc_auc": [], "recall_at_0.3": [], "precision_at_0.3": [], "f1_at_0.3": []}
        for idx in boot_idx:
            yb, pb = yte[idx], p[idx]
            if yb.sum() == 0:
                continue
            boots["pr_auc"].append(average_precision_score(yb, pb))
            boots["roc_auc"].append(roc_auc_score(yb, pb))
            tm = threshold_metrics(yb, pb, 0.3)
            boots["recall_at_0.3"].append(tm["recall"]); boots["precision_at_0.3"].append(tm["precision"]); boots["f1_at_0.3"].append(tm["f1"])
        ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in boots.items()}
        perm = permutation_importance(est, Xte, yte, scoring="average_precision", n_repeats=cfg["perm"], random_state=SEED, n_jobs=1)
        perm_rows = sorted(({"feature": c, "mean": round(float(perm.importances_mean[i]), 4), "std": round(float(perm.importances_std[i]), 4)}
                            for i, c in enumerate(INPUT_COLS)), key=lambda r: -r["mean"])
        fx, fy = downsample(fpr, tpr)
        rx, ry = downsample(rc_, pr_)
        native = native_importance(tuned[model], model) if model in tuned else None
        test[model] = {
            "label": MODEL_NAMES[model],
            "strategy": chosen_strategy.get(model), "calibration": chosen_cal.get(model),
            "roc_auc": round(float(roc_auc_score(yte, p)), 4), "pr_auc": round(float(average_precision_score(yte, p)), 4),
            "brier": round(float(brier_score_loss(yte, p)), 5),
            "log_loss": round(float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))), 4),
            "uncalibrated": None if pu is None else {"brier": round(float(brier_score_loss(yte, pu)), 5),
                                                      "pr_auc": round(float(average_precision_score(yte, pu)), 4),
                                                      "roc_auc": round(float(roc_auc_score(yte, pu)), 4)},
            "thresholds": {str(t): threshold_metrics(yte, p, t) for t in ALERT_THRESHOLDS},
            "bands": band_table(yte, p), "ci95": ci,
            "roc_curve": {"fpr": fx, "tpr": fy}, "pr_curve": {"recall": rx, "precision": ry},
            "reliability": rel, "importance": {"permutation": perm_rows, "native": native},
        }
        flags = df.loc[idx_te, ["TWF", "HDF", "PWF", "OSF", "RNF"]]
        missed = (yte == 1) & (p < 0.3)
        test[model]["missed_at_0.3_by_mode"] = {
            **{c: int(flags[c].values[missed].sum()) for c in flags.columns},
            "no_mode_flag": int((flags.values[missed].sum(axis=1) == 0).sum()), "missed_total": int(missed.sum())}
        t3 = test[model]["thresholds"]["0.3"]
        log(f"[F] {model:3s} test PR-AUC={test[model]['pr_auc']:.3f} ROC={test[model]['roc_auc']:.3f} Brier={test[model]['brier']:.5f} "
            f"| @0.3 P={t3['precision']:.3f} R={t3['recall']:.3f} F1={t3['f1']:.3f} (TP={t3['tp']} FP={t3['fp']} FN={t3['fn']})")

    tuned_w = dict(zip(BASE_MODELS, ens_info["best_on_oof"]["weights"]))
    pt = SoftVotingEnsemble({m: final_models[m] for m in BASE_MODELS}, tuned_w).predict_proba(Xte)[:, 1]
    tt = threshold_metrics(yte, pt, 0.3)
    ens_info["tuned_variant_test"] = {"weights": tuned_w, "pr_auc": round(float(average_precision_score(yte, pt)), 4),
                                      "brier": round(float(brier_score_loss(yte, pt)), 5),
                                      "recall_at_0.3": tt["recall"], "precision_at_0.3": tt["precision"],
                                      "note": "Reported for transparency only; these weights were chosen on training data and are not deployed unless they pass the rule above."}

    # ---------------------------------------------------------------- G. failure-mode classifiers (separate, one-vs-rest)
    modes_models, modes_metrics = {}, {}
    for code in MODES:
        ym_tr, ym_te = df.loc[idx_tr, code].values, df.loc[idx_te, code].values
        n_pos = int(ym_tr.sum())
        info = {"positives_train": n_pos, "positives_test": int(ym_te.sum()),
                "model": "Random Forest (balanced class weights), sigmoid-calibrated"}
        if n_pos < 3:
            info["available"] = False
            modes_metrics[code] = info
            log(f"[G] {code}: only {n_pos} training positives, skipped")
            continue
        base = build_pipeline("rf", "none", use_wt, n_estimators=cfg["mode_trees"], min_samples_leaf=2,
                              class_weight="balanced_subsample")
        info["oof_pr_auc"] = round(float(average_precision_score(ym_tr, cross_val_predict(
            clone(base), Xtr, ym_tr, cv=StratifiedKFold(min(cfg["outer_cv"], n_pos), shuffle=True, random_state=SEED),
            method="predict_proba")[:, 1])), 4)
        cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=max(2, min(cfg["cal_cv"], n_pos))).fit(Xtr, ym_tr)
        modes_models[code] = cal
        pm = cal.predict_proba(Xte)[:, 1]
        info.update(available=True,
                    test_pr_auc=round(float(average_precision_score(ym_te, pm)), 4) if ym_te.sum() else None,
                    test_at_threshold_0_5=threshold_metrics(ym_te, pm, 0.5))
        modes_metrics[code] = info
        log(f"[G] {code}: train positives {n_pos}, test positives {int(ym_te.sum())}, "
            f"OOF PR-AUC {info['oof_pr_auc']}, test PR-AUC {info['test_pr_auc']}")

    # ---------------------------------------------------------------- G2. per-failure-type evaluation, missed failures, PR curves
    default_est = (SoftVotingEnsemble({m: final_models[m] for m in BASE_MODELS}, ens_info["weights"])
                   if default_model == "ens" else final_models[default_model])
    evaluation_block = evaluation.build(df, idx_tr, idx_te, default_est.predict_proba(Xte)[:, 1], default_model, modes_models)
    log(f"[G2] missed {evaluation_block['missed']['total']} of {evaluation_block['missed']['failures']} test failures at 30%; "
        f"{evaluation_block['missed']['with_tool_wear_notice']} of them have tool wear >= 200 min")

    # ---------------------------------------------------------------- H. save artefacts
    stamp = datetime.now().strftime("%Y%m%d-%H%M") + ("" if profile == "full" else f"-{profile}")
    artifacts = {}
    for key in BASE_MODELS:
        path = os.path.join(out_dir, f"{key}.joblib")
        dump_atomic(final_models[key], path)
        artifacts[key] = {"file": f"{key}.joblib", "sha256": sha256_of(path), "bytes": os.path.getsize(path)}
    if modes_models:
        path = os.path.join(out_dir, "modes.joblib")
        dump_atomic(modes_models, path)
        artifacts["modes"] = {"file": "modes.joblib", "sha256": sha256_of(path), "bytes": os.path.getsize(path)}
    ranges = {c: {"min": float(Xtr[c].min()), "max": float(Xtr[c].max()), "median": float(Xtr[c].median())} for c in NUMERIC_RAW}
    background = Xtr.sample(min(30, len(Xtr)), random_state=SEED)[INPUT_COLS].to_dict(orient="records")
    meta = {
        "version": stamp, "profile": profile, "default_model": default_model, "ranking": ranking, "seed": SEED,
        "use_wear_torque": use_wt, "model_names": MODEL_NAMES, "input_columns": INPUT_COLS, "ranges": ranges,
        "type_counts": {k: int(v) for k, v in Xtr["type"].value_counts().items()},
        "thresholds": {"warning": 0.3, "critical": 0.7},
        "ensemble": {"members": BASE_MODELS, "weights": ens_info["weights"]},
        "artifacts": artifacts, "background": background, "failure_modes": MODES,
        "failure_modes_reliable": [c for c in MODES if modes_metrics[c].get("available") and modes_metrics[c].get("oof_pr_auc", 0) >= 0.5],
        "failure_modes_rule": "A failure-mode estimate is shown to users only if its cross-validated PR-AUC on the training split is >= 0.5.",
        "data_sha256": sha256_of(data_path), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "libraries": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                      "imbalanced-learn": imblearn.__version__, "numpy": np.__version__},
    }
    metrics = {
        "trained_at": meta["trained_at"], "profile": profile, "version": stamp, "preliminary": profile != "full",
        "data_quality": dq,
        "dataset": {"train_rows": len(Xtr), "test_rows": len(Xte), "train_failures": int(ytr.sum()), "test_failures": int(yte.sum()),
                    "baseline_accuracy_if_always_no_failure": round(float((yte == 0).mean()), 4)},
        "feature_engineering": {"use_wear_torque": use_wt, "mean_pr_auc_gain_from_wear_torque": round(gain, 4)},
        "ablation": rows, "tuning": tuning, "calibration": calibration, "ensemble": ens_info,
        "selection": {"default_model": default_model, "ranking": ranking, "table": sel,
                      "basis": "Cross-validated (out-of-fold, calibrated) PR-AUC on the training split; models within 0.01 PR-AUC of the best are separated by Brier score, then by simplicity. The test set is not used."},
        "thresholds_oof": thresholds_oof, "test": test, "failure_modes": modes_metrics,
        "baselines": baselines.compute(Xte, yte), "evaluation": evaluation_block,
    }
    write_json_atomic(metrics, os.path.join(out_dir, "metrics.json"))
    write_json_atomic(meta, os.path.join(out_dir, "meta.json"))  # written last: an interrupted run leaves the old set intact
    shutil.rmtree(cache_dir, ignore_errors=True)
    log(f"done in {(time.time() - t0) / 60:.1f} min of work in this session -> {out_dir}/ (version {stamp})")
    return meta, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="full")
    ap.add_argument("--out", default="models")
    ap.add_argument("--data", default="data/raw/ai4i2020.csv")
    a = ap.parse_args()
    run(a.profile, a.out, a.data)
