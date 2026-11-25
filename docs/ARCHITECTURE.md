# Architecture

## 1. The workflow

```
Sensor input ──► Validation ──► Machine identity ──► Preprocessing ──► Feature engineering ──► Model ──► Calibrated probability
  (form / CSV /     validation.py   db.get_or_create_   (inside the saved   (inside the saved       (LR / RF /        (0 to 1)
   JSON API)        service.py      machine             sklearn pipeline)   sklearn pipeline)      XGB / ensemble)
                                                                                                        │
   Dashboard ◄── Database ◄── Recommendation ◄── Risk class + health score + rule indicators + mode estimates + Shapley explanation
   Machines      db.py           risk.py              risk.py                  predictor.py
   Alerts / History / CSV
```

| Step | What happens | File / function |
|---|---|---|
| Input | Form POST, CSV upload or JSON | `routes.analyze`, `routes.batch`, `routes.api_predict` |
| Validation | Machine ID required, type L/M/H, five numbers, hard physical limits (errors), training-range check (warning) | `validation.validate` |
| Orchestration | One workflow for all three inputs | `service.analyze_and_store`, `service.analyze_batch` |
| Preprocessing + features | Derived features, scaling, type encoding. Part of the **saved pipeline**, so training and serving cannot differ | `ml/features.py FeatureBuilder`, `ml/pipelines.py build_pipeline` |
| Prediction | Calibrated failure probability from the chosen model | `predictor.ModelRegistry.analyze_many` |
| Risk | Normal < 30% <= Warning < 70% <= Critical; health = round(100 x (1 - p)) | `risk.classify`, `risk.health_score` |
| Indicators / failure modes | Rule checks (not ML) and per-mode classifier estimates (ML, only reliable modes) shown separately | `risk.rule_indicators`, `predictor._mode_estimates`, `risk.failure_mode_rows` |
| Explanation | Exact Shapley values over five sensor groups | `predictor.ModelRegistry.explain` |
| Storage | Machine, prediction snapshot and alert lifecycle in **one transaction** | `db.record_prediction` |
| Display | Pages read only stored data (the result page never recomputes) | `routes.*`, `templates/` |

## 2. Components

* **Web (Flask)** - `app/routes.py` is thin; `app/service.py` holds the workflow; templates are server-rendered Jinja; Chart.js draws charts from JSON embedded in the page; no inline scripts (Content-Security-Policy).
* **ML service** - `app/predictor.py` (`ModelRegistry`) loads each model **independently** at start-up: file present? checksum matches `meta.json`? loads? probe prediction valid? A failure disables only that model. The ensemble is rebuilt from its members and the weights in `meta.json`, so it is unavailable only if a member is. If the designated default is unavailable the best available model (training ranking) is used and a banner says so. If nothing loads the app still starts and reports the problem.
* **Database (SQLite)** - `app/db.py`.
* **Training (offline)** - `ml/train.py` writes `models/*.joblib`, `meta.json` (version, seed, library versions, checksums, ensemble weights, ranges, background sample) and `metrics.json`. `meta.json` is written last, so an interrupted run never leaves a half-updated set.

## 3. Database schema (version 2)

```
machines(id, machine_code UNIQUE NOCASE, machine_type, created_at)           identity; code/type immutable (trigger)
predictions(id, machine_id -> machines, created_at, source, machine_type, 5 readings, probability, risk_level, severity,
            health_score, model_key, model_label, model_version, recommendation, details_json)   IMMUTABLE (triggers)
alerts(id, machine_id -> machines, status, opened_at, opening_prediction_id -> predictions, latest_prediction_id -> predictions,
       current_level, peak_level, occurrences, reason, recommendation, acknowledged_at, resolved_at, resolution)
alert_events(id, alert_id -> alerts, created_at, event_type, prediction_id -> predictions, note)   append-only (triggers)
```

* Foreign keys are enforced (`PRAGMA foreign_keys = ON` on every connection) and checked by `CHECK` constraints.
* **One machine, one identity**: the same code (case-insensitive) is always the same row; a different type is refused.
* **At most one active alert per machine**, enforced by the partial unique index `uq_one_active_alert_per_machine`.
* `details_json` stores the indicators, Shapley contributions and failure-mode estimates **as computed at the time**, so a saved result never changes when models are retrained.
* Indexes: `(machine_id, created_at)`, `created_at`, `risk_level` on predictions; `(status, current_level)` on alerts; `alert_id` on events.

### Alert lifecycle

```
(no active alert) --Warning/Critical reading--> OPEN
OPEN / ACKNOWLEDGED --another reading at same level--> same alert, occurrences + 1 (event "repeat")
OPEN / ACKNOWLEDGED --reading at a higher level-----> same alert, level raised, ACKNOWLEDGED becomes OPEN again (event "escalated")
OPEN / ACKNOWLEDGED --reading at a lower level------> same alert, level lowered (event "downgraded")
OPEN --user: Acknowledge--> ACKNOWLEDGED
OPEN / ACKNOWLEDGED --user: Resolve--> RESOLVED (resolution = manual)
OPEN / ACKNOWLEDGED --a later Normal reading--> RESOLVED (resolution = auto_normal_reading)
RESOLVED --machine reaches Warning/Critical again--> a NEW alert (the old one stays as history)
```

### Migrations

`python -m app.migrations` (also run at start-up). A fresh database is created at version 2. A v1 database is backed up, then migrated in one
transaction: prediction ids and all stored values are preserved; machines are built from the distinct v1 machine IDs (case-insensitive; the
first prediction's type becomes the machine's type and conflicts are reported); alerts are rebuilt by replaying the v1 rows through the v2
lifecycle (duplicate open alerts collapse into one; later Normal readings resolve; v1 acknowledgements are carried over); the original table
is kept as `legacy_predictions_v1`. Anything unrecognised is refused, never modified.

## 4. Machine-learning pipeline

1. **Data**: `data/raw/ai4i2020.csv` (or any `.csv` / `.xlsx` with the same columns: an `.xlsx` is converted to CSV by `ml/data_io.py`, then the schema is checked by `ml.features.validate_schema`, so the data is always read as CSV); `UDI`, `Product ID` and the failure-mode columns are never model inputs. A data-quality check (missing, duplicates, invalid values) runs first and stops training if it fails.
2. **Split**: one stratified 80/20 split (seed 42). Everything below uses the training part only, except the final report.
3. **Features** (inside the pipeline): `temp_diff = process - air` (K), `power_w = torque x 2 pi rpm / 60` (W), optionally `wear_torque = tool_wear x torque` (a load proxy, **not** physical strain). The proxy is used only if it raises mean cross-validated PR-AUC by more than 0.01.
4. **Encoding**: numeric features scaled; machine type ordinal-encoded, then (after any resampling) one-hot encoded.
5. **Imbalance**: none / class weights / SMOTENC compared by cross-validated PR-AUC. SMOTENC sees machine type as **one** categorical column and runs inside the pipeline, so it only ever resamples training folds.
6. **Tuning**: randomized search, scored on PR-AUC, 5-fold cross-validation.
7. **Calibration**: none / sigmoid / isotonic compared by out-of-fold Brier score; the lowest-Brier calibrator is kept.
8. **Ensemble**: soft voting over the three calibrated models. Weights are chosen only from out-of-fold training predictions (grid, step 0.1), and replace equal weights only if the gain is at least 0.005 PR-AUC, also holds on a split-half check, and the weights form a real blend.
9. **Selection**: highest validation PR-AUC; models within 0.01 are separated by Brier score, then simplicity. The test set is not used.
10. **Test report**: one evaluation on the held-out set (also compared with two no-ML baselines, `ml/baselines.py`) with bootstrap confidence intervals; metrics at 30%, 50% and 70%; alert-band check; reliability curve; permutation importance.
11. **Failure-mode classifiers**: one calibrated Random Forest each for TWF, HDF, PWF and OSF (RNF is not modelled). Shown to users only if validation PR-AUC >= 0.5.

## 5. Security model

| Concern | Mechanism |
|---|---|
| CSRF | Per-session token required (non-empty on both sides) for every form POST; JSON API accepts only `application/json`; cross-site `Origin` rejected |
| Secrets | `SECRET_KEY` from the environment; required in production; random per process in development; nothing hard-coded |
| Session cookie | HttpOnly, SameSite=Lax, Secure in production |
| Headers | Content-Security-Policy (scripts from self only), X-Content-Type-Options, X-Frame-Options, Referrer-Policy |
| SQL injection | Every value is a bound parameter; sort columns are whitelisted; no user text reaches SQL text |
| XSS | Jinja auto-escaping; user text set with `textContent` in JavaScript; machine IDs restricted by a pattern |
| Input | Server-side validation of every field for form, CSV and JSON; upload size and row limits; UTF-8 check |
| Integrity | Triggers make predictions and alert history immutable; foreign keys and CHECK constraints |
| Availability | Independent model loading, checksums, fallback default, database errors reported (503) instead of crashing |

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| One model file missing, corrupt or incompatible | That model is unavailable with the reason; others work; the default falls back to the best available |
| All models unavailable | App starts; banner; analyses return 503 with the reason; `/api/health` returns 503 |
| Failure-mode models unavailable | Analyses work without the failure-mode table |
| Database cannot be opened | App starts; database pages return 503 with the reason; About and Model Insights still work |
| Invalid input | 422, field-level messages, nothing saved |
| Machine type conflict | 422 with the registered type |
| Prediction or save error | Logged; understandable message; the transaction is rolled back |

## 7. Not implemented (and why)

* Live IoT monitoring - inputs are manual; the architecture has no ingestion layer.
* Remaining useful life / time to failure - the dataset has no time dimension.
* User accounts - out of scope for this project.
