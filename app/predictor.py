"""Loads the trained models and turns validated readings into predictions.

Models are loaded ONCE at start-up and each one INDEPENDENTLY: a missing, corrupted or incompatible file only makes that
one model unavailable (with a clear reason); the rest keep working. If the designated default model is unavailable, the
best available model (by training-set ranking) is used and the page says so. If nothing loads, the web app still starts and
reports the problem instead of crashing.
"""
import hashlib
import json
import logging
import os
from math import factorial

import joblib
import numpy as np
import pandas as pd

from app import risk
from ml.compat import library_mismatches
from ml.ensemble import SoftVotingEnsemble
from ml.features import INPUT_COLS  # importing ml.* also makes the pickled pipeline classes resolvable

log = logging.getLogger(__name__)

# sensor groups used for local explanations: air and process temperature act together through their difference
GROUPS = {"Machine type": ["type"], "Temperatures": ["air_temp", "process_temp"], "Rotational speed": ["rpm"],
          "Torque": ["torque"], "Tool wear": ["tool_wear"]}


class ModelUnavailable(Exception):
    def __init__(self, key, reason):
        super().__init__(f"Model '{key}' is unavailable: {reason}")
        self.key, self.reason = key, reason


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, models_dir, reports_dir=None):
        self.models_dir = models_dir
        self.meta, self.metrics, self.eda, self.twf_study = {}, {}, None, None
        self.models, self.errors = {}, {}        # key -> fitted model / key -> reason it is unavailable
        self.modes, self.modes_error = None, None
        self.labels = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "ens": "Soft Voting Ensemble"}
        self.load_problems = []                    # problems that are not tied to one model (e.g. meta.json unreadable)

        self._read_json("meta.json", "meta")
        self._read_json("metrics.json", "metrics")
        if reports_dir and os.path.exists(os.path.join(reports_dir, "eda.json")):
            try:
                with open(os.path.join(reports_dir, "eda.json")) as f:
                    self.eda = json.load(f)
            except (OSError, ValueError) as exc:
                log.warning("Could not read eda.json: %s", exc)
        if reports_dir and os.path.exists(os.path.join(reports_dir, "twf_study.json")):
            try:
                with open(os.path.join(reports_dir, "twf_study.json")) as f:
                    self.twf_study = json.load(f)
            except (OSError, ValueError) as exc:
                log.warning("Could not read twf_study.json: %s", exc)
        self.labels = self.meta.get("model_names", self.labels)
        self.version = self.meta.get("version", "unknown")
        self.ranges = self.meta.get("ranges", {})
        try:
            self._background = pd.DataFrame(self.meta["background"])[INPUT_COLS] if self.meta.get("background") else None
        except (KeyError, ValueError) as exc:
            self._background = None
            self.load_problems.append(f"background sample in meta.json is unusable ({exc}); explanations are disabled.")

        self.version_warnings = library_mismatches(self.meta.get("libraries")) if self.meta.get("libraries") else []
        for w in self.version_warnings:
            log.warning("Library version check: %s", w)
        members = self.meta.get("ensemble", {}).get("members", ["lr", "rf", "xgb"])
        for key in members:
            self._load_member(key)
        self._build_ensemble(members)
        self._load_modes()
        log.info("Models available: %s | unavailable: %s", sorted(self.models), self.errors or "none")

    # ------------------------------------------------------------------ loading
    def _read_json(self, name, attr):
        try:
            with open(os.path.join(self.models_dir, name)) as f:
                setattr(self, attr, json.load(f))
        except (OSError, ValueError) as exc:
            self.load_problems.append(f"{name} could not be read ({type(exc).__name__}: {exc}).")

    def _artifact_path(self, key):
        art = self.meta.get("artifacts", {}).get(key)
        if not art:
            return None, None
        return os.path.join(self.models_dir, art["file"]), art.get("sha256")

    def _load_member(self, key):
        path, expected = self._artifact_path(key)
        if path is None:
            self.errors[key] = "not listed in meta.json"
            return
        if not os.path.exists(path):
            self.errors[key] = f"file {os.path.basename(path)} is missing"
            return
        try:
            if expected and _sha256(path) != expected:
                self.errors[key] = (f"{os.path.basename(path)} does not match its recorded checksum "
                                    "(the file is corrupted or was replaced). Re-download it or retrain with: python -m ml.train")
                return
            model = joblib.load(path)
            if self._background is not None:  # probe: a loaded-but-broken model must not be offered to users
                p = model.predict_proba(self._background.iloc[:1])[:, 1]
                if not (0.0 <= float(p[0]) <= 1.0):
                    raise ValueError("probe prediction outside [0, 1]")
            self.models[key] = model
        except Exception as exc:  # noqa: BLE001 - any failure must only disable this one model
            self.errors[key] = (f"could not be loaded ({type(exc).__name__}: {exc}). Library versions must match "
                                f"meta.json: {self.meta.get('libraries', {})}. Retrain with: python -m ml.train")
            log.error("Model %s failed to load: %s", key, exc)

    def _build_ensemble(self, members):
        missing = [m for m in members if m not in self.models]
        if "ens" not in self.labels or not self.meta:
            return
        if missing:
            self.errors["ens"] = f"needs {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} unavailable"
            return
        weights = self.meta["ensemble"]["weights"]
        self.models["ens"] = SoftVotingEnsemble({m: self.models[m] for m in members}, weights)

    def _load_modes(self):
        path, expected = self._artifact_path("modes")
        if path is None or not os.path.exists(path):
            self.modes_error = "failure-mode models not found"
            return
        try:
            if expected and _sha256(path) != expected:
                raise ValueError("checksum mismatch")
            self.modes = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            self.modes_error = f"failure-mode models could not be loaded ({exc})"
            log.error(self.modes_error)

    # ------------------------------------------------------------------ status
    @property
    def keys(self):
        return set(self.models)

    @property
    def default_model(self):
        """The designated default if it loaded; otherwise the best available model by training-set ranking."""
        designated = self.meta.get("default_model")
        if designated in self.models:
            return designated
        for key in self.meta.get("ranking", []) + list(self.models):
            if key in self.models:
                return key
        return None

    @property
    def default_is_fallback(self):
        return self.default_model is not None and self.default_model != self.meta.get("default_model")

    @property
    def ready(self):
        return self.default_model is not None

    def status(self):
        all_keys = list(self.labels)
        return {"ready": self.ready, "version": self.version, "designated_default": self.meta.get("default_model"),
                "active_default": self.default_model, "default_is_fallback": self.default_is_fallback,
                "models": {k: {"label": self.labels[k], "available": k in self.models, "error": self.errors.get(k)} for k in all_keys},
                "failure_modes": {"available": self.modes is not None, "error": self.modes_error},
                "problems": self.load_problems, "version_warnings": self.version_warnings}

    def get(self, key):
        if key not in self.models:
            raise ModelUnavailable(key, self.errors.get(key, "unknown model"))
        return self.models[key]

    # ------------------------------------------------------------------ prediction
    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows)[INPUT_COLS]

    def predict_proba(self, key, values):
        """Calibrated failure probability (0-1) for one validated reading (used by what-if)."""
        return float(self.get(key).predict_proba(self._frame([values]))[0, 1])

    def _mode_estimates(self, frame):
        """{code: probability} per row from the failure-mode classifiers (empty dicts if they are unavailable)."""
        if self.modes is None:
            return [{} for _ in range(len(frame))]
        cols = {code: m.predict_proba(frame)[:, 1] for code, m in self.modes.items()}
        return [{code: float(cols[code][i]) for code in cols} for i in range(len(frame))]

    def explain(self, model, values):
        """Exact Shapley values over the five sensor groups (interventional, background = typical training readings).

        Contribution of a group = its average effect on the failure probability when it is switched from a typical
        background reading to this machine's value, averaged over all orders of switching. The contributions add up
        exactly to  (this probability - the average probability of the background readings).
        This describes the MODEL's behaviour; it does not prove a physical cause.
        """
        if self._background is None:
            return None
        bg, names = self._background, list(GROUPS)
        n, B = len(names), len(bg)
        rows = []
        for mask in range(1 << n):
            for b in range(B):
                r = bg.iloc[b].to_dict()
                for i, g in enumerate(names):
                    if mask >> i & 1:
                        for c in GROUPS[g]:
                            r[c] = values[c]
                rows.append(r)
        v = model.predict_proba(self._frame(rows))[:, 1].reshape(1 << n, B).mean(axis=1)
        out = []
        for i, g in enumerate(names):
            phi = 0.0
            for mask in range(1 << n):
                if mask >> i & 1:
                    continue
                k = bin(mask).count("1")
                phi += factorial(k) * factorial(n - k - 1) / factorial(n) * (v[mask | 1 << i] - v[mask])
            out.append({"group": g, "contribution": float(phi)})
        return {"base_value": float(v[0]), "prediction": float(v[-1]),
                "contributions": sorted(out, key=lambda r: -abs(r["contribution"]))}

    def analyze_many(self, key, rows, explain=False):
        """Predict for several validated readings. `explain=True` adds Shapley contributions (slower; single analyses only)."""
        model = self.get(key)
        probs = model.predict_proba(self._frame(rows))[:, 1]
        levels = [risk.classify(float(p)) for p in probs]
        elevated = [i for i, lv in enumerate(levels) if lv != "Normal"]
        modes = {}
        if elevated:
            for i, est in zip(elevated, self._mode_estimates(self._frame([rows[i] for i in elevated]))):
                modes[i] = est
        wear_rates = (self.eda or {}).get("tool_wear_rates")
        results = []
        for i, (v, p, level) in enumerate(zip(rows, probs, levels)):
            p = float(p)
            indicators = risk.rule_indicators(v, wear_rates) if level != "Normal" else []
            results.append({
                "probability": p, "risk_level": level, "severity": risk.LEVELS[level]["severity"],
                "health_score": risk.health_score(p), "action": risk.LEVELS[level]["action"],
                "guidance": risk.LEVELS[level]["guidance"], "recommendation": risk.recommendation(level, indicators),
                "model_key": key, "model_label": self.labels[key], "model_version": self.version,
                "details": {"derived": risk.derived_values(v), "indicators": indicators, "mode_estimates": modes.get(i, {}),
                            "maintenance": risk.maintenance_notices(v, wear_rates),
                            "contributions": self.explain(model, v) if explain else None, "input_warnings": []},
            })
        return results

    def analyze(self, key, values):
        return self.analyze_many(key, [values], explain=True)[0]
