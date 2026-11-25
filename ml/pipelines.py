"""Builds the full preprocessing + (resampling) + model pipelines."""
import xgboost
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.features import TYPES, FeatureBuilder

SEED = 42
MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost", "ens": "Soft Voting Ensemble"}
BASE_MODELS = ["lr", "rf", "xgb"]
STRATEGIES = ["none", "class_weight", "smote"]


class PortableXGBClassifier(XGBClassifier):
    """XGBClassifier that is pickled with its booster in XGBoost's JSON model format.

    XGBoost's default pickle is a raw memory snapshot that is not guaranteed to load across
    platforms or builds (e.g. trained on Linux, loaded on Windows). JSON is the documented
    portable format, so the saved models load the same everywhere.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        booster = state.pop("_Booster", None)
        state["_booster_json"] = bytes(booster.save_raw(raw_format="json")) if booster is not None else None
        return state

    def __setstate__(self, state):
        state = dict(state)
        raw = state.pop("_booster_json", None)
        self.__dict__.update(state)
        if raw is not None:
            booster = xgboost.Booster()
            booster.load_model(bytearray(raw))
            self._Booster = booster


def make_classifier(model, strategy, scale_pos_weight=1.0, **params):
    weighted = strategy == "class_weight"
    if model == "lr":
        p = dict(max_iter=3000, class_weight="balanced" if weighted else None)
        p.update(params)
        return LogisticRegression(**p)
    if model == "rf":
        p = dict(n_estimators=200, random_state=SEED, n_jobs=1,
                 class_weight="balanced_subsample" if weighted else None)
        p.update(params)
        return RandomForestClassifier(**p)
    if model == "xgb":
        p = dict(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED,
                 n_jobs=1, eval_metric="logloss", tree_method="hist",
                 scale_pos_weight=scale_pos_weight if weighted else 1.0)
        p.update(params)
        return PortableXGBClassifier(**p)
    raise ValueError(model)


def build_pipeline(model, strategy="none", use_wear_torque=False, scale_pos_weight=1.0, **params):
    """raw dataframe -> features -> scale + ordinal type -> [SMOTENC] -> one-hot type -> classifier.

    The machine type is ordinal-encoded (H=0, L=1, M=2) and treated by SMOTENC as ONE categorical
    column, so every synthetic sample gets a valid type. One-hot encoding happens AFTER resampling,
    so the three dummy columns are always consistent (exactly one is 1).
    SMOTENC lives INSIDE the pipeline: during cross-validation it is fitted on the training folds
    only and is skipped at prediction time (never touches validation/test/inference data).
    """
    fb = FeatureBuilder(use_wear_torque=use_wear_torque)
    num = fb.numeric_columns()
    n = len(num)
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("type", OrdinalEncoder(categories=[TYPES]), ["type"]),
    ])
    onehot = ColumnTransformer([
        ("num", "passthrough", list(range(n))),
        ("type", OneHotEncoder(categories=[[float(i) for i in range(len(TYPES))]], sparse_output=False), [n]),
    ])
    steps = [("features", fb), ("prep", prep)]
    if strategy == "smote":
        steps.append(("smote", SMOTENC(categorical_features=[n], random_state=SEED)))
    steps += [("onehot", onehot), ("clf", make_classifier(model, strategy, scale_pos_weight, **params))]
    return Pipeline(steps)


def feature_names(pipe):
    """Names of the columns the classifier sees, in order."""
    fb = pipe.named_steps["features"]
    return fb.numeric_columns() + [f"type_{t}" for t in TYPES]
