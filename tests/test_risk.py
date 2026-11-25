import pytest

from app import risk


@pytest.mark.parametrize("p,expected", [(0.0, "Normal"), (0.2999, "Normal"), (0.30, "Warning"), (0.6999, "Warning"),
                                        (0.70, "Critical"), (1.0, "Critical")])
def test_thresholds_and_boundaries(p, expected):
    assert risk.classify(p) == expected


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_out_of_range_probability_rejected(bad):
    with pytest.raises(ValueError):
        risk.classify(bad)


def test_health_score_formula():
    assert [risk.health_score(p) for p in (0.0, 0.25, 0.596, 1.0)] == [100, 75, 40, 0]
    assert "not an independently validated" in risk.HEALTH_NOTE


R = dict(type="L", air_temp=302.0, process_temp=310.3, rpm=1300, torque=45.0, tool_wear=60)


def test_rule_indicators_and_naming():
    codes = {i["code"] for i in risk.rule_indicators(R)}
    assert codes == {"HDF"}                                           # dT 8.3 K and 1300 rpm
    assert {i["code"] for i in risk.rule_indicators({**R, "tool_wear": 230, "torque": 55.0})} >= {"OSF", "TWF"}
    assert "PWF" in {i["code"] for i in risk.rule_indicators({**R, "rpm": 2800, "torque": 40.0, "air_temp": 300.0, "process_temp": 310.0})}
    assert risk.rule_indicators({**R, "rpm": 1500, "air_temp": 298.1, "process_temp": 308.6}) == []


def test_failure_mode_rows_separate_model_estimate_from_rule_check():
    rows = risk.failure_mode_rows({"HDF": 0.9, "OSF": 0.25, "PWF": 0.05}, risk.rule_indicators(R))
    by = {r["code"]: r for r in rows}
    assert set(by) == {"HDF", "OSF"} and by["HDF"]["rule_indicator"] and not by["OSF"]["rule_indicator"]
    assert rows[0]["code"] == "HDF" and "RNF" not in by


def test_rnf_is_explained_not_predicted():
    assert "cannot be predicted" in risk.RNF_NOTE and "RNF" not in risk.MODE_INFO


def test_recommendation_text():
    assert risk.recommendation("Normal", []) == "Continue operation."
    assert "Inspect the cooling system" in risk.recommendation("Critical", risk.rule_indicators(R))


# ---------------------------------------------------------------- maintenance notice (separate from status and alerts)
def test_tool_wear_notice_threshold_and_wording():
    assert risk.maintenance_notices({"tool_wear": 199.9}) == []
    n = risk.maintenance_notices({"tool_wear": 200}, {"ge_200_pct": 15.4, "lt_200_pct": 2.3})
    assert len(n) == 1 and n[0]["code"] == "TOOL_WEAR"
    assert "Inspect the tool and consider replacement" in n[0]["detail"] and "not a prediction" in n[0]["detail"] and "15.4%" in n[0]["detail"]
