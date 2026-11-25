import numpy as np
import pandas as pd
import pytest

from conftest import DATA
from ml.features import INPUT_COLS, MODE_COLS, FeatureBuilder, data_quality_report, load_data


def reading(**over):
    base = dict(type="L", air_temp=300.0, process_temp=310.0, rpm=1500.0, torque=40.0, tool_wear=100.0)
    return pd.DataFrame([{**base, **over}])


def test_temperature_difference():
    out = FeatureBuilder().transform(reading(air_temp=298.1, process_temp=308.6))
    assert out["temp_diff"].iloc[0] == pytest.approx(10.5)


def test_power_formula_torque_times_angular_speed():
    out = FeatureBuilder().transform(reading(torque=40.0, rpm=1500.0))
    assert out["power_w"].iloc[0] == pytest.approx(40.0 * 1500.0 * 2 * np.pi / 60)   # about 6283 W


def test_wear_torque_proxy_only_when_enabled_and_named_honestly():
    assert "wear_torque" not in FeatureBuilder(False).transform(reading()).columns
    out = FeatureBuilder(True).transform(reading(tool_wear=100.0, torque=40.0))
    assert out["wear_torque"].iloc[0] == 4000.0
    assert "strain" not in out.columns            # it is a load proxy, not physical strain
    assert FeatureBuilder(True).numeric_columns()[-1] == "wear_torque"


def test_transform_does_not_modify_input():
    df = reading()
    FeatureBuilder(True).transform(df)
    assert list(df.columns) == INPUT_COLS


def test_no_leakage_columns_in_model_inputs():
    assert not set(INPUT_COLS) & (set(MODE_COLS) | {"failure", "UDI", "Product ID"})
    assert INPUT_COLS == ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]


def test_real_dataset_passes_quality_checks():
    q = data_quality_report(load_data(DATA))
    assert q["ok"] and q["rows"] == 10000 and q["missing_values"] == 0 and q["duplicate_rows"] == 0


def test_quality_report_detects_problems():
    df = load_data(DATA).head(100).copy()
    df.loc[df.index[0], "air_temp"] = np.nan
    df.loc[df.index[1], "type"] = "X"
    df.loc[df.index[2], "rpm"] = -5
    q = data_quality_report(df)
    assert not q["ok"] and q["missing_values"] == 1 and q["invalid_type"] == 1 and q["non_positive_rpm_or_torque"] == 1
