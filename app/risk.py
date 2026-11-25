"""Risk classification, health score, rule-based indicators and maintenance suggestions.

Three DIFFERENT kinds of information are produced here and in predictor.py. They are never mixed up:
  1. ML-based failure prediction      -> a calibrated failure probability from the trained model, then Normal/Warning/Critical.
  2. Rule-based failure indicators    -> simple checks of the failure conditions documented for the AI4I dataset
                                         (verified against the data). They are NOT model output.
  3. Suggested maintenance actions    -> generic suggestions attached to (1) and (2). They are not guaranteed diagnoses.

Risk bands (project proposal): Normal 0-30%, Warning 30-70%, Critical 70-100%.
Boundaries: the lower bound of each band is inclusive (exactly 30% is Warning, exactly 70% is Critical).
"""
import math

WARNING_AT = 0.30
CRITICAL_AT = 0.70

LEVELS = {
    "Normal":   {"severity": 0, "action": "Continue operation",
                 "guidance": "No maintenance action is indicated by the model. Keep analysing the machine as new readings arrive."},
    "Warning":  {"severity": 1, "action": "Schedule inspection",
                 "guidance": "Elevated failure risk. Schedule an inspection soon and review the readings highlighted below."},
    "Critical": {"severity": 2, "action": "Immediate maintenance",
                 "guidance": "High failure risk. Reduce load or stop the machine if it is safe to do so, and carry out maintenance immediately."},
}

HEALTH_FORMULA = "Health score = round(100 x (1 - calibrated failure probability)), from 0 (certain failure) to 100."
HEALTH_NOTE = ("The health score is only a re-expression of the model's failure probability on a 0-100 scale. "
               "It is not an independently validated or physical measure of machine health.")


def classify(probability):
    """probability in [0, 1] -> 'Normal' | 'Warning' | 'Critical'."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be between 0 and 1, got {probability}")
    if probability >= CRITICAL_AT:
        return "Critical"
    if probability >= WARNING_AT:
        return "Warning"
    return "Normal"


def health_score(probability):
    return int(min(100, max(0, round(100 * (1 - probability)))))


# ---------------------------------------------------------------------------------------------
# Rule-based indicators (documented AI4I failure conditions). HDF, PWF and OSF reproduce the dataset's
# flags exactly (0 mismatches, see reports/eda.json); TWF is only partly deterministic.
# ---------------------------------------------------------------------------------------------
OSF_LIMITS = {"L": 11000, "M": 12000, "H": 13000}  # tool wear [min] x torque [Nm]

MODE_INFO = {
    "TWF": {"name": "Tool wear failure", "action": "Inspect the tool and consider replacing it."},
    "HDF": {"name": "Heat dissipation failure", "action": "Inspect the cooling system and airflow."},
    "PWF": {"name": "Power failure", "action": "Check the drive, power supply and load."},
    "OSF": {"name": "Overstrain failure", "action": "Reduce the load and inspect the tool and spindle."},
}
RNF_NOTE = ("Random failures (RNF) are random by design in this dataset (about 0.1% of records, independent of the readings), "
            "so they cannot be predicted from sensor data and are not modelled.")


def derived_values(v):
    """Derived quantities for a validated reading dict (same formulas as ml/features.py)."""
    return {"temp_diff": v["process_temp"] - v["air_temp"],
            "power_w": v["torque"] * v["rpm"] * 2 * math.pi / 60,
            "wear_torque": v["tool_wear"] * v["torque"]}


def rule_indicators(v, wear_rates=None):
    """Documented failure conditions that are present in this reading (rule-based, not ML)."""
    d = derived_values(v)
    found = []
    if d["temp_diff"] < 8.6 and v["rpm"] < 1380:
        found.append({"code": "HDF", "title": "Heat dissipation condition",
                      "detail": f"Temperature difference is {d['temp_diff']:.1f} K, at or below the 8.6 K limit, and speed is "
                                f"{v['rpm']:g} rpm, below the 1380 rpm limit.",
                      "action": MODE_INFO["HDF"]["action"]})
    if d["power_w"] < 3500 or d["power_w"] > 9000:
        side = "below 3,500 W" if d["power_w"] < 3500 else "above 9,000 W"
        found.append({"code": "PWF", "title": "Power condition",
                      "detail": f"Estimated power is {d['power_w']:,.0f} W, {side}.", "action": MODE_INFO["PWF"]["action"]})
    limit = OSF_LIMITS[v["type"]]
    if d["wear_torque"] > limit:
        found.append({"code": "OSF", "title": "Overstrain condition",
                      "detail": f"Tool wear x torque is {d['wear_torque']:,.0f} min*Nm, above the {limit:,} limit for type {v['type']}.",
                      "action": MODE_INFO["OSF"]["action"]})
    if v["tool_wear"] >= 200:
        extra = ""
        if wear_rates:
            extra = (f" In the training data, {wear_rates['ge_200_pct']}% of records at or above 200 min failed, "
                     f"versus {wear_rates['lt_200_pct']}% below.")
        found.append({"code": "TWF", "title": "High tool wear",
                      "detail": f"Tool wear is {v['tool_wear']:g} min (200 min or more)." + extra, "action": MODE_INFO["TWF"]["action"]})
    return found


TOOL_WEAR_NOTICE_MIN = 200   # minutes; the documented tool-wear limit of the dataset


def maintenance_notices(v, wear_rates=None):
    """Maintenance notices are separate from the failure probability and the Normal / Warning / Critical status.
    They never change the status and never open an alert: they only make a known maintenance condition visible,
    even when the overall model says Normal (tool-wear failures are largely unpredictable from the readings)."""
    out = []
    if v.get("tool_wear") is not None and v["tool_wear"] >= TOOL_WEAR_NOTICE_MIN:
        context = ""
        if wear_rates:
            context = (f" In the training data, {wear_rates['ge_200_pct']}% of records at or above {TOOL_WEAR_NOTICE_MIN} min failed, "
                       f"versus {wear_rates['lt_200_pct']}% below.")
        out.append({"code": "TOOL_WEAR", "title": "High tool wear",
                    "detail": f"Tool wear is {v['tool_wear']:g} min ({TOOL_WEAR_NOTICE_MIN} min or more). Inspect the tool and consider replacement."
                              + context + " This is a maintenance notice, not a prediction that the machine will fail."})
    return out


def recommendation(level, indicators):
    """One-line recommendation stored with each prediction."""
    text = LEVELS[level]["action"] + "."
    if level != "Normal" and indicators:
        text += " " + " ".join(f"{c['title']}: {c['action']}" for c in indicators)
    return text


def failure_mode_rows(estimates, indicators, min_probability=0.20):
    """Rows for the 'possible failure modes' table: ML estimate and rule indicator side by side.

    estimates : {code: probability from the failure-mode classifier} (may be empty if unavailable)
    Only modes that are estimated at >= min_probability or flagged by a rule are listed.
    """
    flagged = {i["code"] for i in indicators}
    rows = []
    for code, info in MODE_INFO.items():
        p = (estimates or {}).get(code)
        if code in flagged or (p is not None and p >= min_probability):
            rows.append({"code": code, "name": info["name"], "model_estimate": p, "rule_indicator": code in flagged,
                         "action": info["action"]})
    return sorted(rows, key=lambda r: -(r["model_estimate"] or 0))
