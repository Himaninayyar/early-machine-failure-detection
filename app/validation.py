"""Input validation for sensor readings (form, CSV batch and JSON API all use this).

Hard errors  : missing / non-numeric / physically impossible values (the analysis is refused).
Soft warnings: values outside the range seen in training data (analysis is made, with a caution).
"""
import math
import re

FIELDS = {  # key: (label, unit, hard_min, hard_max, what it measures)
    "air_temp":     ("Air temperature", "K", 250.0, 350.0, "Temperature of the surrounding air. 0 deg C = 273.15 K."),
    "process_temp": ("Process temperature", "K", 250.0, 400.0, "Temperature of the machining process. Normally a few kelvin above the air temperature."),
    "rpm":          ("Rotational speed", "rpm", 1.0, 10000.0, "Spindle speed in revolutions per minute."),
    "torque":       ("Torque", "Nm", 0.0, 500.0, "Turning force applied by the spindle, in newton-metres."),
    "tool_wear":    ("Tool wear", "min", 0.0, 2000.0, "Minutes the cutting tool has already been in use."),
}
MACHINE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,39}$")
TYPE_HELP = {"L": "Low quality variant", "M": "Medium quality variant", "H": "High quality variant"}


def validate(form, ranges, available_models, default_model=None):
    """Return (clean_values, errors, warnings). `errors` maps a field name to a message."""
    errors, warnings, clean = {}, [], {}

    machine_id = (form.get("machine_id") or "").strip()
    if not machine_id:
        errors["machine_id"] = "Enter a machine ID. It identifies the machine across all its analyses."
    elif not MACHINE_ID_RE.match(machine_id):
        errors["machine_id"] = "Machine ID may use letters, numbers, spaces, dot, dash and underscore (max 40 characters)."
    clean["machine_id"] = machine_id

    t = (form.get("type") or "").strip().upper()
    if t not in TYPE_HELP:
        errors["type"] = "Select a machine type (L, M or H)."
    clean["type"] = t

    for key, (label, unit, lo, hi, _help) in FIELDS.items():
        raw = str(form.get(key) if form.get(key) is not None else "").strip()
        if raw == "":
            errors[key] = f"{label} is required."
            continue
        try:
            v = float(raw)
        except ValueError:
            errors[key] = f"{label} must be a number."
            continue
        if not math.isfinite(v):
            errors[key] = f"{label} must be a finite number."
        elif not lo <= v <= hi:
            errors[key] = f"{label} must be between {lo:g} and {hi:g} {unit}."
        else:
            clean[key] = v
            r = ranges[key]
            if v < r["min"] or v > r["max"]:
                warnings.append(f"{label} ({v:g} {unit}) is outside the range of the training data "
                                f"({r['min']:g}-{r['max']:g} {unit}). The prediction is an extrapolation and less reliable.")

    if "air_temp" in clean and "process_temp" in clean and clean["process_temp"] <= clean["air_temp"]:
        warnings.append("Process temperature is not higher than air temperature, which is unusual for this machine type.")

    model = (form.get("model") or "").strip() or default_model
    if model not in available_models:
        errors["model"] = "That model is not available."
    clean["model"] = model
    return clean, errors, warnings
