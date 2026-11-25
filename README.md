# Early Machine Failure Detection Using Machine Learning

A predictive-maintenance web application. You enter a machine's sensor readings (or upload a CSV); a trained model estimates the
probability that the machine is in a failing state, and the application turns that into a **Normal / Warning / Critical** status, a
**health score**, a **recommended action**, an explanation of **what drove the result**, and an **alert** that is tracked until it is resolved.

* Dataset: AI4I 2020 Predictive Maintenance (10,000 records, 339 failures = 3.39%). It is **synthetic**.
* Stack: Python, Flask (Jinja templates), Bootstrap 5, Chart.js, SQLite, scikit-learn, XGBoost, imbalanced-learn.
* Models: Logistic Regression, Random Forest, XGBoost and a Soft Voting Ensemble, each evaluated independently.
* Inputs are entered by hand or uploaded. This is **not** live monitoring, and the probability is **not** a time-to-failure.

Documentation: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (how it works), [`docs/API.md`](docs/API.md) (JSON API),
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) (status and latest test results) and the workbook
[`docs/Project_Documentation_and_Change_Log.xlsx`](docs/Project_Documentation_and_Change_Log.xlsx) (full project record, change log, viva preparation).

---

## 1. Install

Requires **Python 3.12 or newer** (tested on 3.12 and 3.14). Use a virtual environment **and make sure it is active** before installing.

```
python -m venv venv
```

Activate it (the command depends on your shell; your prompt should then start with `(venv)`):

| Shell | Command |
|---|---|
| Git Bash (Windows) | `source venv/Scripts/activate` |
| Command Prompt | `venv\Scripts\activate.bat` |
| PowerShell | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

```
pip install -r requirements.txt
```

`requirements.txt` allows compatible versions. The saved models were trained with the exact versions in `requirements-lock.txt`
(`pip install -r requirements-lock.txt` for a reproducible install). If your installed versions differ, a warning is shown on Model Insights;
models usually still load, and you can retrain with `python -m ml.train` to match your environment.

## 2. Models

Trained models are included in `models/`. To retrain them (about 10-15 minutes on one CPU core):

```
python -m ml.eda                 # dataset summary -> reports/eda.json (About page)
python -m ml.train               # full run -> models/ (lr, rf, xgb, modes, meta.json, metrics.json)
```

**Using your own dataset (CSV or Excel).** Put a `.csv` or `.xlsx` in `data/raw/` and pass it with `--data`:

```
python -m ml.train --data data/raw/my_data.xlsx     # also works for .csv; python -m ml.eda <file> for the summary
```

An `.xlsx` is converted once to a `.csv` beside it (`python -m ml.data_io file.xlsx` does only the conversion; `--sheet NAME` picks a sheet,
default is the first). After that the data is **only ever read as CSV**. The file must have the AI4I column names; the columns and value types are checked first and a wrong file fails with a clear message (missing columns with a "did you mean" hint, non-numeric cells with line numbers). The batch upload page also accepts `.xlsx`, and the original AI4I file (its column names are recognised; up to 1000 rows per upload).

Training is **resumable**: if it is interrupted, run the same command again and it continues from its checkpoints.
`--profile quick` uses smaller search budgets; `--profile tiny` (seconds) is only for the automated tests. Results from non-`full`
profiles are marked *preliminary* on the Model Insights page. Everything is made reproducible by fixed random seeds (`SEED = 42`).

Every choice in training (features, imbalance handling, hyperparameters, calibration, ensemble weights, default model) is made with
cross-validation on the **training split only**; the held-out test split is used once, to report results.

**Baselines.** Model Insights compares the models with "always predict no failure" and with the documented failure rules alone (no ML).
For the shipped models these were added without retraining: `python -m ml.baselines`. Training does this automatically.

**Tool-wear maintenance notice.** When tool wear is 200 min or more, the result page, batch results and API show "High tool wear: inspect the tool and
consider replacement", whatever the status. It is a maintenance reminder, not a failure prediction, and it never changes the status or creates an alert.

**Evaluation extras.** Model Insights shows results for each failure type (precision, recall, confusion matrix), the list of failures the model missed, precision-recall curves by
failure type and a tool-wear study. Training produces them. For the shipped models:
`python -m ml.evaluation` (adds them to `models/metrics.json`, no retraining) and `python -m ml.twf_study` (writes `reports/twf_study.json`, about 40 s).
**How the test set is used:** every choice uses cross-validation on the training split; the test set is looked at once to report results. To judge a future model change, use repeated
cross-validation as in `ml/twf_study.py`, not the test set.

## 3. Run

```
flask --app app run
```

Open http://127.0.0.1:5000. The database (`instance/predictions.db`) is created automatically on first start. If the file is a database from
the previous version (v1), it is **backed up and migrated automatically** (see section 5). `flask run` is the development server; for a
real deployment use a production WSGI server (for example waitress on Windows or gunicorn on Linux) with `APP_ENV=production`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `development` | `production` requires `SECRET_KEY` and turns on secure cookies |
| `SECRET_KEY` | random per process (development) | Signs the session cookie. **Required in production.** Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_PATH` | `instance/predictions.db` | SQLite file |
| `MODELS_DIR` / `REPORTS_DIR` | `models/` / `reports/` | Trained artefacts / EDA summary |
| `SESSION_COOKIE_SECURE` | `1` in production, else `0` | Send the session cookie over HTTPS only |
| `MAX_UPLOAD_BYTES` | `1000000` | CSV batch size limit |

See `.env.example`. Set them as real environment variables (the app does not read `.env` files).

## 4. Tests

```
python -m pytest tests -q
```

162 tests: features and data quality, preprocessing and resampling (including the SMOTENC fix), every model and the ensemble, Shapley
explanations, model-loading failures, dataset schema check, xlsx/CSV input, baselines, library-version check, database constraints, machine identity, alert lifecycle, history immutability, the v1 to v2
migration, CSRF and other security checks, and the whole web flow (form to result to history to alerts, CSV batch, what-if, JSON API, export).
The first run trains tiny models (about 30 seconds) for the model tests.

## 5. Database

SQLite, schema version 2 (`PRAGMA user_version`): `machines`, `predictions` (immutable), `alerts`, `alert_events` (append-only).
Created or upgraded automatically at start-up; to do it by hand, or to preview a migration without changing anything:

```
python -m app.migrations --db instance/predictions.db --dry-run
python -m app.migrations --db instance/predictions.db
```

A v1 database is copied to `<file>.v1-backup-<timestamp>` first, migrated in one transaction (rolled back on any error), and the original
table is kept as `legacy_predictions_v1`. Details: `docs/ARCHITECTURE.md`.

## 6. Using the application

1. **Dashboard** - machines tracked, machines needing attention, active alerts, average health, recent analyses (empty-state guide for a first-time user).
2. **Analyze Machine** - machine ID, type and five sensor readings (units, explanations and ranges are shown). Example readings can be loaded with one click.
   A CSV of many readings can be uploaded instead (template provided).
3. **Result** - failure probability, status, health score, recommended action, what drove the result, possible failure modes, what-if exploration and technical details.
4. **Machines** - every machine, its latest status, and a per-machine risk trend and history.
5. **Alerts** - one alert per machine while it is Warning or Critical; acknowledge, resolve, and see each alert's history.
6. **History** - search, filter, sort and export every analysis to CSV.
7. **Model Insights** - model comparison (including the ensemble), metrics, confusion matrices, calibration, thresholds, failure-mode classifiers, feature importance and methodology.

## 7. Folder structure

```
app/               Flask application
  __init__.py        app factory: config from environment, CSRF, security headers, error handling
  routes.py          pages and JSON API (thin)
  service.py         the shared workflow: validate -> predict -> store (form, API and CSV batch all use it)
  predictor.py       model registry: independent loading, checksums, fallback, Shapley explanations
  risk.py            thresholds, health score, rule-based indicators, suggested actions
  validation.py      input rules
  db.py              schema, machine identity, alert lifecycle, queries
  migrations.py      create / migrate the database (CLI: python -m app.migrations)
  templates/ static/
ml/                features.py, pipelines.py, ensemble.py, train.py, eda.py
data/raw/          ai4i2020.csv
models/            trained artefacts + meta.json + metrics.json (generated by ml/train.py)
reports/           eda.json, training log
tests/             pytest suite
docs/              ARCHITECTURE.md, API.md, PROJECT_STATUS.md, Project_Documentation_and_Change_Log.xlsx, build_workbook.py
```

## 8. Limitations

* The dataset is synthetic and its failure modes follow fixed rules. Scores are likely much higher than real machines would give; nothing was validated on external or real data.
* Records are independent snapshots: the probability is not remaining useful life or time to failure.
* Manual or CSV input only; no live sensor feed.
* The test set has 68 failures, so differences between close models are within statistical noise.
* The overall model estimates failure risk, not failure type. The "possible failure modes" table combines per-mode classifiers (only where they were reliable in validation) with rule checks. Both are suggestions to investigate, not diagnoses. Random failures (RNF) cannot be identified.
* The health score is `round(100 x (1 - probability))`. It is not an independently validated measure of machine health.
* "What-if" shows how the model responds to a changed reading; it does not establish cause and effect.
* There are no user accounts or permissions.

## 9. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `pip` changes packages of other projects | The virtual environment was not active. Activate it (section 1) and look for `(venv)` in the prompt. |
| `XGBoostError: input stream corrupted` while starting | Old XGBoost model files could not be read on this platform. The current models store XGBoost in its portable JSON format. If it still happens, the app now starts anyway and the Model Insights page names the unavailable model; run `python -m ml.train` to rebuild the models on your machine. |
| A model shows as *Unavailable* (checksum mismatch) | A model file is damaged or was replaced. Re-copy `models/`, or retrain. |
| "Predictions are unavailable" banner | No model could be loaded. See Model Insights for the reason, or run `python -m ml.train`. |
| `SECRET_KEY must be set` | You set `APP_ENV=production`. Set `SECRET_KEY` (section 3). |
| Port 5000 in use | `flask --app app run --port 5001` |
