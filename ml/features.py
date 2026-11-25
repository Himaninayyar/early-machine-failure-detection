"""Data loading, data-quality checks and feature engineering shared by training and the web app.

FeatureBuilder is part of every saved sklearn pipeline, so the web app applies exactly the same
transformations as training (no train/serve mismatch).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.data_io import ensure_csv

# Raw CSV column -> clean name used everywhere else
RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
}
INPUT_COLS = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
NUMERIC_RAW = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
TYPES = ["H", "L", "M"]
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # NEVER used as model inputs (they define the target: leakage)


class DataSchemaError(ValueError):
    """The dataset does not have the columns / value types this project needs. The message says exactly what to fix."""


# Columns every training dataset must have (the AI4I 2020 names). UDI and Product ID are optional identifiers.
REQUIRED_RAW = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                "Tool wear [min]", "Machine failure"] + MODE_COLS
NUMERIC_REQUIRED = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
                    "Tool wear [min]", "Machine failure"] + MODE_COLS


def _norm(name):
    return " ".join(str(name).replace("\ufeff", "").split()).lower()


def validate_schema(df):
    """Check a raw dataframe BEFORE it is used. Fixes harmless header differences (case, extra spaces, BOM), refuses anything
    else with a clear message: missing columns (with close-name suggestions), non-numeric cells, empty data.
    Returns the dataframe with canonical column names and numeric columns converted to numbers."""
    if len(df) == 0:
        raise DataSchemaError("The dataset has a header but no data rows.")
    by_norm = {_norm(c): c for c in df.columns}
    rename, missing = {}, []
    for need in REQUIRED_RAW:
        have = by_norm.get(_norm(need))
        if have is None:
            missing.append(need)
        elif have != need:
            rename[have] = need
    if missing:
        import difflib
        extra = [c for c in df.columns if c not in REQUIRED_RAW and c not in rename]
        hints = []
        for m in missing:
            close = difflib.get_close_matches(m, extra, n=1, cutoff=0.5)
            hints.append(f"'{m}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise DataSchemaError("Missing required column(s): " + ", ".join(hints) + ". Columns found: " + ", ".join(map(str, df.columns))
                              + ". Rename the columns to the AI4I 2020 names (see data/raw/ai4i2020.csv).")
    df = df.rename(columns=rename)
    for c in NUMERIC_REQUIRED:
        num = pd.to_numeric(df[c], errors="coerce")
        bad = num.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
        if bad.any():
            rows = ", ".join(str(i + 2) for i in df.index[bad][:5])   # +2: header is line 1 in the file
            raise DataSchemaError(f"Column '{c}' has {int(bad.sum())} non-numeric value(s) (file lines {rows}{', ...' if bad.sum() > 5 else ''}).")
        df[c] = num
    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    return df


def load_data(path):
    """Return the dataframe with clean column names. UDI / Product ID (identifiers) are dropped.
    Accepts .csv or .xlsx; an .xlsx is converted to CSV first, so the data is always READ as CSV.
    The columns and value types are checked first (validate_schema), so a wrong file fails with a clear message."""
    df = validate_schema(pd.read_csv(ensure_csv(path)))
    df = df.rename(columns=RENAME)
    return df.drop(columns=[c for c in ("UDI", "Product ID") if c in df.columns])


def data_quality_report(df):
    """Missing values, duplicates and invalid values. Used by training (fails loudly) and the EDA."""
    feats = df[INPUT_COLS]
    report = {
        "rows": int(len(df)),
        "missing_values": int(df[INPUT_COLS + ["failure"]].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(feats.duplicated().sum()),
        "invalid_type": int((~df["type"].isin(TYPES)).sum()),
        "non_positive_rpm_or_torque": int(((df["rpm"] <= 0) | (df["torque"] <= 0)).sum()),
        "negative_tool_wear": int((df["tool_wear"] < 0).sum()),
        "process_not_above_air": int((df["process_temp"] <= df["air_temp"]).sum()),
        "target_not_binary": int((~df["failure"].isin([0, 1])).sum()),
    }
    report["ok"] = all(report[k] == 0 for k in report if k not in ("rows", "ok", "duplicate_rows", "duplicate_feature_rows")) \
        and report["duplicate_rows"] == 0
    return report


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds derived features (all computed from the five raw sensor readings).

    temp_diff   : process temperature - air temperature (K)                   [derived measurement]
    power_w     : approximate mechanical power = torque * 2*pi*rpm/60 (W)      [derived physical quantity]
    wear_torque : tool wear * torque (min*Nm), a load-on-worn-tool PROXY.      [heuristic proxy, NOT physical strain]
                  It is the product the dataset documentation uses for the overstrain failure condition.
    """

    def __init__(self, use_wear_torque=False):
        self.use_wear_torque = use_wear_torque

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["temp_diff"] = X["process_temp"] - X["air_temp"]
        X["power_w"] = X["torque"] * X["rpm"] * 2 * np.pi / 60
        if self.use_wear_torque:
            X["wear_torque"] = X["tool_wear"] * X["torque"]
        return X

    def numeric_columns(self):
        cols = NUMERIC_RAW + ["temp_diff", "power_w"]
        return cols + ["wear_torque"] if self.use_wear_torque else cols
