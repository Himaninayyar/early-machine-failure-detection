"""The one workflow behind the form, the JSON API and the CSV batch upload:

    validate -> predict (ML) -> risk level + health + recommendation -> save (machine, prediction, alert) -> result

Routes stay thin and never talk to the model or the database directly for this.
"""
import csv
import io
import logging
import sqlite3

from app import db
from app.predictor import ModelUnavailable
from app.validation import validate

log = logging.getLogger(__name__)
BATCH_COLUMNS = ["machine_id", "type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
# The AI4I 2020 training dataset uses different column names. A file with them is accepted as is (Product ID becomes the
# machine ID; UDI, Machine failure, TWF... are ignored), so the original dataset can be uploaded without editing.
AI4I_HEADERS = {"product id": "machine_id", "air temperature [k]": "air_temp", "process temperature [k]": "process_temp",
                "rotational speed [rpm]": "rpm", "torque [nm]": "torque", "tool wear [min]": "tool_wear"}


class ServiceError(Exception):
    status, code = 500, "error"

    def __init__(self, message, errors=None, warnings=None):
        super().__init__(message)
        self.message, self.errors, self.warnings = message, errors or {}, warnings or []


class ValidationFailed(ServiceError):
    status, code = 422, "validation_failed"


class ServiceUnavailable(ServiceError):
    status, code = 503, "service_unavailable"


def _require_ready(registry):
    if not registry.ready:
        problems = " ".join(registry.load_problems) or "; ".join(f"{k}: {v}" for k, v in registry.errors.items())
        raise ServiceUnavailable("No prediction model is available, so analyses cannot be made right now. "
                                 f"{problems} Run 'python -m ml.train' to create the models.")


def analyze_and_store(registry, conn, form, *, source="form", now=None):
    """Returns (result, stored). Raises ValidationFailed / ServiceUnavailable / ServiceError with a user-readable message."""
    _require_ready(registry)
    clean, errors, warnings = validate(form, registry.ranges, registry.keys, registry.default_model)
    if errors:
        raise ValidationFailed("Some inputs need attention.", errors, warnings)
    try:
        result = registry.analyze(clean["model"], clean)
    except ModelUnavailable as exc:
        raise ServiceUnavailable(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("prediction failed")
        raise ServiceError("The prediction could not be computed. The error has been logged.") from exc
    result["details"]["input_warnings"] = warnings
    try:
        stored = db.record_prediction(conn, machine_code=clean["machine_id"], machine_type=clean["type"], readings=clean,
                                      result=result, source=source, now=now)
    except db.MachineTypeConflict as exc:
        raise ValidationFailed("Some inputs need attention.", {"type": str(exc)}, warnings) from exc
    except sqlite3.Error as exc:
        log.exception("database error while saving a prediction")
        raise ServiceUnavailable("The prediction could not be saved because the database is unavailable.") from exc
    return result, stored


def analyze_batch(registry, conn, raw_bytes, *, max_rows=1000, now=None):
    """Validate and analyse every row of an uploaded CSV with the default model. Valid rows are saved (in one
    transaction); invalid rows are reported with their line number and are NOT saved."""
    _require_ready(registry)
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationFailed("The file is not valid UTF-8 text. Save it as CSV (UTF-8).") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames and "product id" in [h.strip().lower() for h in reader.fieldnames]:
        reader.fieldnames = [AI4I_HEADERS.get(h.strip().lower(), h) for h in reader.fieldnames]
    header = [h.strip().lower() for h in (reader.fieldnames or [])]
    missing = [c for c in BATCH_COLUMNS if c not in header]
    if missing:
        raise ValidationFailed(f"Missing column(s): {', '.join(missing)}. Required columns: {', '.join(BATCH_COLUMNS)}.")

    valid, rejected, seen_types, n = [], [], {}, 0
    for line, raw in enumerate(reader, start=2):
        n += 1
        if n > max_rows:
            raise ValidationFailed(f"The file has more than {max_rows} rows. Upload at most {max_rows} rows at a time (for example, the first {max_rows}).")
        row = {(k or "").strip().lower(): (v or "") for k, v in raw.items()}
        row.pop("model", None)  # batch analyses always use the default model
        clean, errors, warnings = validate(row, registry.ranges, registry.keys, registry.default_model)
        if not errors:
            key = clean["machine_id"].lower()
            existing = db.get_machine(conn, clean["machine_id"])
            known = seen_types.get(key) or (existing["machine_type"] if existing else None)
            if known and known != clean["type"]:
                errors["type"] = f"Machine {clean['machine_id']} is registered as type {known}."
            else:
                seen_types[key] = clean["type"]
        if errors:
            rejected.append({"line": line, "machine_id": row.get("machine_id", ""), "errors": errors})
        else:
            valid.append((line, clean, warnings))
    if n == 0:
        raise ValidationFailed("The file has no data rows.")

    saved = []
    if valid:
        try:
            results = registry.analyze_many(registry.default_model, [c for _, c, _ in valid], explain=False)
            for (line, clean, warnings), result in zip(valid, results):
                result["details"]["input_warnings"] = warnings
                stored = db.record_prediction(conn, machine_code=clean["machine_id"], machine_type=clean["type"],
                                              readings=clean, result=result, source="batch", now=now, commit=False)
                saved.append({"line": line, "machine_id": clean["machine_id"], "probability": result["probability"],
                              "risk_level": result["risk_level"], "health_score": result["health_score"],
                              "prediction_id": stored["prediction_id"], "alert": stored["alert"]["action"],
                              "recommendation": result["recommendation"],
                              "maintenance": [n["code"] for n in result["details"].get("maintenance", [])]})
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            log.exception("database error during batch save")
            raise ServiceUnavailable("The batch could not be saved because the database is unavailable. Nothing was saved.") from exc
        except ModelUnavailable as exc:
            raise ServiceUnavailable(str(exc)) from exc
    counts = {"Normal": 0, "Warning": 0, "Critical": 0}
    for s in saved:
        counts[s["risk_level"]] += 1
    return {"rows": n, "saved": saved, "rejected": rejected, "counts": counts,
            "maintenance_count": sum(1 for s in saved if s["maintenance"]), "model": registry.labels[registry.default_model]}


def what_if(registry, base_row, changes):
    """Re-run the SAME model that produced a saved prediction with some readings changed. Nothing is saved.

    Reading how the probability moves shows how the MODEL responds to the change. It does not establish
    that changing the real machine would cause the same change in failure risk."""
    from app import risk
    key = base_row["model_key"]
    if key not in registry.keys:
        raise ServiceUnavailable(f"The model used for this analysis ({base_row['model_label']}) is not available now.")
    if base_row["model_version"] != registry.version:
        raise ServiceUnavailable("This analysis was made with an older model version, so what-if analysis is not available for it. "
                                 "Analyse the machine again to use it.")
    form = {"machine_id": "what-if", "type": base_row["machine_type"], "air_temp": base_row["air_temp"],
            "process_temp": base_row["process_temp"], "rpm": base_row["rpm"], "torque": base_row["torque"],
            "tool_wear": base_row["tool_wear"], "model": key}
    form.update({k: v for k, v in changes.items() if k in ("air_temp", "process_temp", "rpm", "torque", "tool_wear")})
    clean, errors, warnings = validate(form, registry.ranges, registry.keys, key)
    if errors:
        raise ValidationFailed("Some what-if values are invalid.", errors, warnings)
    p = registry.predict_proba(key, clean)
    level = risk.classify(p)
    return {"probability": p, "risk_level": level, "health_score": risk.health_score(p), "action": risk.LEVELS[level]["action"],
            "baseline_probability": base_row["probability"], "delta": p - base_row["probability"], "warnings": warnings,
            "indicators": [i["code"] for i in risk.rule_indicators(clean)] if level != "Normal" else []}
