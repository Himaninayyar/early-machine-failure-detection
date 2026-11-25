from app.validation import validate

RANGES = {"air_temp": {"min": 295.3, "max": 304.5}, "process_temp": {"min": 305.7, "max": 313.8}, "rpm": {"min": 1168, "max": 2886},
          "torque": {"min": 3.8, "max": 76.6}, "tool_wear": {"min": 0, "max": 253}}
OK = {"machine_id": "M-001", "type": "M", "air_temp": "298.1", "process_temp": "308.6", "rpm": "1551", "torque": "42.8", "tool_wear": "0"}


def run(**over):
    return validate({**OK, **over}, RANGES, {"rf", "lr"}, "rf")


def test_valid_input_uses_default_model():
    clean, errors, warnings = run()
    assert not errors and not warnings and clean["rpm"] == 1551.0 and clean["model"] == "rf" and clean["machine_id"] == "M-001"


def test_machine_id_is_required():
    assert "machine_id" in run(machine_id="")[1] and "machine_id" in run(machine_id="   ")[1]


def test_missing_and_non_numeric():
    _, errors, _ = run(torque="", rpm="abc")
    assert "torque" in errors and "rpm" in errors


def test_impossible_values_rejected():
    _, errors, _ = run(rpm="-5", tool_wear="-1", torque="1e9")
    assert {"rpm", "tool_wear", "torque"} <= set(errors)


def test_nan_and_inf_rejected():
    _, errors, _ = run(air_temp="nan", process_temp="inf")
    assert "air_temp" in errors and "process_temp" in errors


def test_bad_type_and_model():
    _, errors, _ = run(type="X", model="svm")
    assert "type" in errors and "model" in errors


def test_outside_training_range_is_warning_not_error():
    _, errors, warnings = run(rpm="3500")
    assert not errors and len(warnings) == 1


def test_machine_id_characters():
    for bad in ("<script>", "a;b", "x" * 41, "-start"):
        assert "machine_id" in run(machine_id=bad)[1], bad
    assert not run(machine_id="Pump 7_a.b-c")[1]
