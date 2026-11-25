# Project status

Last updated: 2026-11-24

## Current status

The project is a working predictive-maintenance web application (Flask, SQLite, scikit-learn, XGBoost) with the audit-driven improvements
implemented and verified as described below. The full record (what changed, why, how it works, viva preparation) is in
`docs/Project_Documentation_and_Change_Log.xlsx`, which is rebuilt by `python docs/build_workbook.py`.

## Completed phases

| Phase | Status |
|---|---|
| Phase 1 - Understand and audit | Done. Confirmed issues: SMOTENC after one-hot (115 invalid synthetic rows), CSRF accepted empty tokens, a new machine per prediction, duplicate and stale alerts, result page recomputed from current models, hard-coded development secret, one bad model file stopped the app, misleading 'strain' name. |
| Phase 2 - Plan | Done (prioritised table in the delivery message). |
| Phase 3 - Implement | Done: ML pipeline, ensemble, failure-mode classifiers, backend, database and migration, UI redesign, tests, documentation. |
| Phase 4 - Verify | Done on Linux (Python 3.12 and 3.14). Not done on Windows. |

## Latest test results

`python -m pytest tests -q` -> **162 passed of 162** (latest run, Python 3.13 on Linux). The first 148 tests also passed on Python 3.12 and 3.14 (2026-11-18).

| File | Passed |
|---|---|
| tests/test_db.py | 23 of 23 |
| tests/test_features.py | 7 of 7 |
| tests/test_flows.py | 35 of 35 |
| tests/test_migration.py | 7 of 7 |
| tests/test_models.py | 14 of 14 |
| tests/test_pipeline.py | 14 of 14 |
| tests/test_registry.py | 10 of 10 |
| tests/test_risk.py | 14 of 14 |
| tests/test_security.py | 18 of 18 |
| tests/test_shipped_models.py | 12 of 12 |
| tests/test_validation.py | 8 of 8 |

Also verified by hand: a browser walk-through of every page in Chromium (1440 px and 390 px) with the shipped models and no JavaScript errors;
an interrupted-and-resumed training run; a v1 to v2 migration with duplicate alerts, mixed-case IDs, a type conflict and an acknowledged alert.

## Final selected model

**Random Forest**, model version `20261117-0629` (profile `full`), chosen from cross-validated, calibrated PR-AUC on the training
split only (`models/metrics.json` -> `selection`). Held-out test set at the 30% threshold: precision 0.966, recall 0.824,
F1 0.889, PR-AUC 0.900. The soft-voting ensemble was evaluated and did **not** beat it.

## Current phase

Hand-over. Waiting for the project team to review.

## Pending tasks and open items

* Confirm on the Windows machine that the models load (XGBoost is now stored in portable JSON form; the original failure could not be reproduced on Linux).
* Decide when to drop the `legacy_predictions_v1` table kept by the migration.
* Known limitations: synthetic dataset and no external validation; 68 test failures; manual input only; non-nested hyperparameter tuning; not load-tested.

Full list: sheet *11 Pending & Limitations* of the workbook.
