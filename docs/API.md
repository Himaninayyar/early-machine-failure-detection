# JSON API

Base URL: `http://127.0.0.1:5000`. All API routes accept and return **JSON only**. A request with another content type gets `415`.
There is no authentication (see the project limitations). Browsers cannot call the API cross-site (no CORS headers; a cross-site `Origin` is rejected with 403).

## Error format

```json
{ "error": "Some inputs need attention.", "code": "validation_failed", "errors": { "torque": "Torque is required." } }
```

| Status | Meaning |
|---|---|
| 400 | Body is not a JSON object |
| 404 | Unknown `prediction_id` |
| 415 | Not `application/json` |
| 422 | Validation failed (`errors` lists each field) |
| 503 | No model available, model version changed, or database unavailable (`error` explains) |

## POST /api/predict

Runs the same workflow as the Analyze form and **saves** the analysis (and may open, update or resolve the machine's alert).

Request:

```json
{ "machine_id": "PUMP-07", "type": "L", "air_temp": 303.7, "process_temp": 312.1, "rpm": 1363, "torque": 51.8, "tool_wear": 90 }
```

`model` is optional (`lr`, `rf`, `xgb`, `ens`); the default model is used if omitted. Units: kelvin, rpm, Nm, minutes.

Response `200`:

```json
{
  "prediction_id": 12, "machine_id": "PUMP-07", "machine_created": true,
  "probability": 0.9996, "risk_level": "Critical", "health_score": 0,
  "action": "Immediate maintenance",
  "recommendation": "Immediate maintenance. Heat dissipation condition: Inspect the cooling system and airflow.",
  "model": "Random Forest", "model_version": "20261117-0629",
  "warnings": [], "indicators": ["HDF"], "maintenance": [],
  "alert": { "action": "opened", "alert_id": 3 }
}
```

`alert.action` is one of `opened`, `repeat`, `escalated`, `downgraded`, `auto_resolved`, `none`.
`indicators` are rule-based checks, not machine-learning output. `maintenance` lists separate maintenance notices (currently `TOOL_WEAR` when tool wear is 200 min or more); a notice never changes `risk_level` or alerts.

```
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" \
  -d '{"machine_id":"PUMP-07","type":"L","air_temp":303.7,"process_temp":312.1,"rpm":1363,"torque":51.8,"tool_wear":90}'
```

## POST /api/what-if

Re-runs the model that produced a saved analysis with some readings changed. **Nothing is saved.**

```json
{ "prediction_id": 12, "changes": { "torque": 30 } }
```

Response `200`: `probability`, `risk_level`, `health_score`, `action`, `baseline_probability`, `delta`, `warnings`, `indicators`, `disclaimer`.
The result shows how the *model* responds; it does not establish cause and effect. Returns `503` if the analysis was made with a different model version than the one loaded now.

## GET /api/health

Operational status: `ready`, `version`, `designated_default`, `active_default`, `default_is_fallback`, per-model `available` / `error`, `failure_modes`, `database`, `version_warnings` (library versions that differ from training).
Returns `200` when ready and `503` when no model is available or the database is unavailable.

## Browser-only routes (not JSON)

| Route | Purpose |
|---|---|
| `GET/POST /analyze` | Analyze form |
| `GET/POST /analyze/batch`, `GET /analyze/batch/template.csv` | CSV or .xlsx upload (columns `machine_id,type,air_temp,process_temp,rpm,torque,tool_wear`; max 1000 rows) and template |
| `GET /predictions/<id>` | Stored result page |
| `GET /machines`, `GET /machines/<code>` | Machine list and detail with trend |
| `GET /alerts`, `GET /alerts/<id>`, `POST /alerts/<id>/acknowledge`, `POST /alerts/<id>/resolve` | Alerts (POST needs the CSRF token) |
| `GET /history`, `GET /history/export.csv` | History with filters (`machine`, `risk`, `model`, `date_from`, `date_to`, `sort`, `dir`, `page`) and CSV export |
| `GET /insights`, `GET /about` | Model Insights and project information |
