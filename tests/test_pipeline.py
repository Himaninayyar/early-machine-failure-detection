"""Preprocessing consistency, categorical encoding and resampling (the SMOTENC fix)."""
import numpy as np
import pytest
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from conftest import DATA
from ml.features import INPUT_COLS, TYPES, FeatureBuilder, load_data
from ml.pipelines import build_pipeline, feature_names


@pytest.fixture(scope="module")
def data():
    df = load_data(DATA)
    X, y = df[INPUT_COLS], df["failure"].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_step_order_resamples_before_one_hot_and_only_for_smote():
    smote = [n for n, _ in build_pipeline("rf", "smote", True).steps]
    assert smote == ["features", "prep", "smote", "onehot", "clf"]
    for strategy in ("none", "class_weight"):
        assert "smote" not in [n for n, _ in build_pipeline("rf", strategy, True).steps]


def test_smotenc_treats_machine_type_as_one_categorical_column(data):
    pipe = build_pipeline("rf", "smote", True)
    n_num = len(FeatureBuilder(True).numeric_columns())
    assert list(pipe.named_steps["smote"].categorical_features) == [n_num]       # one column, not three dummies


def test_synthetic_samples_have_valid_machine_type_and_consistent_one_hot(data):
    Xtr, _, ytr, _ = data
    pipe = build_pipeline("rf", "smote", True)
    Z = pipe.named_steps["prep"].fit_transform(pipe.named_steps["features"].fit_transform(Xtr))
    Zr, yr = pipe.named_steps["smote"].fit_resample(Z, ytr)
    assert len(Zr) > len(Z)                                           # synthetic rows were created
    assert set(np.unique(Zr[:, -1])) <= {0.0, 1.0, 2.0}               # a valid type code for every row, including synthetic
    onehot = pipe.named_steps["onehot"].fit_transform(Zr)
    assert onehot.shape[1] == len(feature_names(pipe))
    assert np.allclose(onehot[:, -3:].sum(axis=1), 1.0)               # exactly one type column is 1, for ALL rows


def test_old_order_one_hot_then_smotenc_produced_invalid_rows(data):
    """Documents the bug that was fixed: SMOTENC on three independent one-hot columns invents rows with no type at all."""
    Xtr, _, ytr, _ = data
    fb = FeatureBuilder(True)
    num = fb.numeric_columns()
    prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(categories=[TYPES], sparse_output=False), ["type"])])
    Z = prep.fit_transform(fb.fit_transform(Xtr))
    n = len(num)
    Zr, _ = SMOTENC(categorical_features=[n, n + 1, n + 2], random_state=42).fit_resample(Z, ytr)
    synthetic = Zr[len(Z):, -3:]
    assert (~np.isclose(synthetic.sum(axis=1), 1.0)).sum() > 0


def test_resampling_never_touches_validation_folds_or_inference(data):
    Xtr, _, ytr, _ = data
    pipe = build_pipeline("lr", "smote", True)
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    p = cross_val_predict(pipe, Xtr.iloc[:3000], ytr[:3000], cv=cv, method="predict_proba")
    assert p.shape == (3000, 2)                                       # one prediction per REAL row, none for synthetic rows
    pipe.fit(Xtr.iloc[:3000], ytr[:3000])
    assert pipe.predict_proba(Xtr.iloc[3000:3100]).shape == (100, 2)  # the sampler is skipped at prediction time


def test_inference_uses_identical_transformations_as_training(data):
    """The same fitted pipeline gives the same answer for one row, a batch, and a shuffled batch."""
    Xtr, Xte, ytr, _ = data
    pipe = build_pipeline("rf", "none", True, n_estimators=20).fit(Xtr, ytr)
    batch = pipe.predict_proba(Xte.iloc[:50])[:, 1]
    singles = np.array([pipe.predict_proba(Xte.iloc[[i]])[0, 1] for i in range(50)])
    assert np.allclose(batch, singles)
    perm = np.random.default_rng(0).permutation(50)
    assert np.allclose(pipe.predict_proba(Xte.iloc[:50].iloc[perm])[:, 1], batch[perm])


@pytest.mark.parametrize("t,idx", [("H", 0), ("L", 1), ("M", 2)])
def test_machine_type_encoding(data, t, idx):
    Xtr, _, ytr, _ = data
    pipe = build_pipeline("lr", "none", True).fit(Xtr, ytr)
    row = Xtr.iloc[[0]].copy()
    row["type"] = t
    Z = pipe[:-1].transform(row)
    assert Z[0, -3:].tolist() == [1.0 if i == idx else 0.0 for i in range(3)]
    assert feature_names(pipe)[-3:] == ["type_H", "type_L", "type_M"]


def test_pipeline_is_reproducible_with_fixed_seed(data):
    Xtr, Xte, ytr, _ = data
    a = build_pipeline("rf", "smote", True, n_estimators=20).fit(Xtr.iloc[:2000], ytr[:2000]).predict_proba(Xte.iloc[:30])
    b = build_pipeline("rf", "smote", True, n_estimators=20).fit(Xtr.iloc[:2000], ytr[:2000]).predict_proba(Xte.iloc[:30])
    assert np.allclose(a, b)


# ---------------------------------------------------------------- schema check, baselines, library-version check
import pandas as pd  # noqa: E402


def test_schema_check_explains_missing_and_bad_columns():
    from ml.features import DataSchemaError, validate_schema
    raw = pd.read_csv(DATA)
    with pytest.raises(DataSchemaError, match="did you mean 'torque nm'"):
        validate_schema(raw.rename(columns={"Torque [Nm]": "torque nm"}))
    with pytest.raises(DataSchemaError, match="'TWF'"):
        validate_schema(raw.drop(columns=["TWF"]))
    bad = raw.copy()
    bad["Tool wear [min]"] = bad["Tool wear [min]"].astype(object)
    bad.loc[3, "Tool wear [min]"] = "abc"
    with pytest.raises(DataSchemaError, match="non-numeric"):
        validate_schema(bad)
    with pytest.raises(DataSchemaError, match="no data rows"):
        validate_schema(raw.iloc[0:0])
    assert len(validate_schema(raw.rename(columns=str.upper))) == len(raw)   # harmless header differences are fixed


def test_rule_baseline_matches_the_apps_rule_indicators():
    from app import risk
    from ml.baselines import rule_flags
    from ml.features import INPUT_COLS, load_data
    X = load_data(DATA).sample(400, random_state=1)[INPUT_COLS]
    flags = rule_flags(X)
    for (_, r), f in zip(X.iterrows(), flags):
        assert bool(risk.rule_indicators(r.to_dict())) == bool(f)


def test_baselines_never_fails_has_zero_recall_and_high_accuracy():
    from ml.baselines import compute
    from ml.features import INPUT_COLS, load_data
    df = load_data(DATA).head(1000)
    b = compute(df[INPUT_COLS], df["failure"].values)
    assert b["never_fails"]["recall"] == 0 and b["never_fails"]["accuracy"] > 0.9
    assert b["rule_physical"]["precision"] >= b["rule_based"]["precision"]


def test_library_version_check_warns_only_on_differences():
    from ml.compat import installed_versions, library_mismatches
    now = installed_versions()
    assert library_mismatches(dict(now)) == []
    w = library_mismatches({**now, "scikit-learn": "0.0.1"})
    assert any("scikit-learn" in x and "0.0.1" in x for x in w) and any("requirements-lock.txt" in x for x in w)
    assert library_mismatches({**now, "python": "3.12.3"}) == []   # the Python version alone does not warn
