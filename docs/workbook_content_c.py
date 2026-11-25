"""Change log, decisions, viva questions and pending work. Tests named in the change log are verified against the real pytest report."""

D = "2026-11-18"
D2 = "2026-11-24"
T0 = 148   # size of the suite when the 2026-11-18 rows were written; later additions are logged under D2
P_AUDIT, P_IMPL, P_VERIFY = "Phase 1 - Audit", "Phase 3 - Implement", "Phase 4 - Verify"
OK_T = "Implemented, tested"
OK_M = "Implemented, verified manually"


def changelog(M, meta, T):
    """Rows: date, phase, file, previous, new, reason, technical, [tests], status, manual-check text."""
    lab = meta["model_names"]
    R = []
    add = R.append
    # ------------------------------------------------------------------ machine learning
    add([D, P_IMPL + " (ML)", "ml/pipelines.py: build_pipeline",
         "Steps were features -> scale + one-hot machine type -> SMOTENC with categorical_features = the last three one-hot columns. SMOTENC treats each dummy as a separate categorical column. Measured with the old code: 115 of 7,458 synthetic rows had all three type columns equal to 0 (an impossible machine type).",
         "Steps are features -> scale + ordinal machine type -> SMOTENC (machine type is ONE categorical column) -> one-hot -> classifier. Every synthetic row has a valid type and exactly one type column equal to 1.",
         "Resampling must not invent impossible data; the SMOTE comparison reported earlier was contaminated.",
         "SMOTENC picks the category of a synthetic sample by majority vote of neighbours, per categorical column. With three independent 0/1 columns the three votes can all be 0. With one ordinal column the vote returns a valid category; one-hot encoding happens afterwards inside the pipeline. The sampler still runs only during fit, so only training folds are resampled.",
         ["test_step_order_resamples_before_one_hot_and_only_for_smote", "test_smotenc_treats_machine_type_as_one_categorical_column", "test_synthetic_samples_have_valid_machine_type_and_consistent_one_hot", "test_old_order_one_hot_then_smotenc_produced_invalid_rows", "test_resampling_never_touches_validation_folds_or_inference"],
         OK_T, "Before/after counts printed during the audit (115 invalid rows before, 0 after)."])
    add([D, P_IMPL + " (ML)", "ml/pipelines.py: PortableXGBClassifier",
         "XGBoost models were saved with XGBoost's default pickle (a raw memory snapshot). On the user's Windows machine loading xgb.joblib failed with 'XGBoostError: input stream corrupted'.",
         "XGBoost is pickled in XGBoost's JSON model format (portable across platforms and builds) and restored exactly.",
         "XGBoost's documentation says the memory snapshot is not guaranteed portable; JSON is the supported portable format.",
         "Subclass of XGBClassifier whose __getstate__ stores booster.save_raw('json') and whose __setstate__ rebuilds the booster with load_model. The failure could not be reproduced on Linux (Python 3.12 and 3.14 both loaded the old file), so the root cause is inferred, not proven.",
         ["test_portable_xgboost_pickle_roundtrip_is_exact_and_uses_json"], "Implemented, tested on Linux only; NOT yet confirmed on the user's Windows machine", "Loaded under Python 3.12 and 3.14 (Linux)."])
    add([D, P_IMPL + " (ML)", "ml/features.py: FeatureBuilder, load_data",
         "The derived feature tool wear x torque was named 'strain' (and use_strain) in code, features and metrics.",
         "Renamed to wear_torque (use_wear_torque). Documented as a load proxy on a worn tool, not physical strain.",
         "It is not a measurement of strain; the name was misleading.",
         "Column, parameter, metric keys, UI labels and docs renamed together; models were retrained because the pickled pipeline contains the parameter name.",
         ["test_wear_torque_proxy_only_when_enabled_and_named_honestly", "test_analyze_returns_complete_result"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/features.py: data_quality_report; ml/train.py",
         "Missing values, duplicates and invalid values were checked only informally in the EDA script.",
         "A data-quality report (missing, duplicates, invalid type, impossible values, target not binary) runs before every training run and stops training if a check fails; the counts are stored in metrics.json and shown on Model Insights.",
         "Training on bad data should fail loudly.",
         "data_quality_report returns counts and an overall ok flag; train.run raises ValueError if not ok.",
         ["test_real_dataset_passes_quality_checks", "test_quality_report_detects_problems"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/ensemble.py (new): SoftVotingEnsemble",
         "There was no ensemble.",
         "A weighted average of the calibrated probabilities of Logistic Regression, Random Forest and XGBoost. It is not saved as a file; the application rebuilds it from the loaded members and the weights in meta.json.",
         "Requested experiment: does combining models beat the best single model?",
         "Subclass of ClassifierMixin and BaseEstimator with predict_proba, weights normalised to sum to 1. Rebuilding at load time means one damaged member cannot hide inside an ensemble file.",
         ["test_ensemble_is_the_weighted_average_of_its_members", "test_ensemble_weights_are_normalised_and_validated", "test_every_model_returns_valid_probabilities"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/train.py: ensemble weights",
         "-",
         "Weights are chosen only from out-of-fold training predictions: equal weights versus a 0.1-step grid. Tuned weights replace equal weights only if they gain at least 0.005 PR-AUC, also gain on a split-half check, and no member carries weight 0.9 or more. The tuned variant's test result is reported for transparency but never used.",
         "Avoid using test data to choose weights, avoid over-fitting the weights and avoid calling a single model an 'ensemble'.",
         f"Result of the real run: weights lr 0 / rf 1.0 / xgb 0 gained {M['ensemble']['gain_on_oof']:+.3f} PR-AUC but are degenerate (Random Forest alone), so equal weights are deployed; the equal-weight ensemble scores below Random Forest.",
         ["test_training_selected_the_default_model_from_validation_evidence", "test_shipped_models.py::test_the_default_model_follows_the_documented_selection_rule"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/train.py: model selection",
         "The default model was chosen with an ad-hoc rule on cross-validated PR-AUC for three models.",
         "Four candidates are ranked by out-of-fold, calibrated PR-AUC on the training split; models within 0.01 are separated by Brier score and then simplicity; the test set is not used. The designated default, ranking and evidence are stored in meta.json and metrics.json.",
         "Selection must follow the predictive-maintenance objective without hard-coding Random Forest and without using test data.",
         "OOF predictions for all models use identical folds so they can be compared and blended; the rule is re-checked by a test that recomputes the winner from metrics.json.",
         ["test_the_default_model_follows_the_documented_selection_rule", "test_training_selected_the_default_model_from_validation_evidence"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/train.py: threshold evidence and missed failures",
         "The 30% / 70% thresholds were stated but not examined.",
         "metrics.json now holds an out-of-fold threshold scan for every model, the observed failure rate in each alert band, and which failure modes the missed failures belong to.",
         "Threshold choices need evidence; the recall ceiling needs an explanation.",
         "threshold_scan on out-of-fold probabilities; band_table on test; missed_at_0.3_by_mode from the mode flags.",
         ["test_reported_test_metrics_reproduce_from_the_shipped_models"], OK_T, "Shown on Model Insights (viewed in browser)."])
    add([D, P_IMPL + " (ML)", "ml/train.py: failure-mode classifiers",
         "Only the binary target was modelled; failure-specific guidance came from rules only.",
         "One calibrated Random Forest each for TWF, HDF, PWF and OSF (RNF deliberately not modelled). An estimate is shown to users only if its validation PR-AUC is at least 0.5; TWF fails this gate.",
         "Provide failure-specific information without claiming the binary model identifies failure types.",
         f"Validation PR-AUC: " + ", ".join(f"{c} {v['oof_pr_auc']:.3f}" for c, v in M["failure_modes"].items()) + f". Shown: {', '.join(meta['failure_modes_reliable'])}.",
         ["test_failure_mode_models_give_probabilities", "test_failure_mode_gate_excludes_unreliable_classifiers"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/train.py: artefacts, versioning, reproducibility",
         "No checksums; version was a timestamp only; files written directly.",
         "meta.json records version, profile, seed, library versions, data checksum, per-file sha256, ensemble weights, training ranges and a background sample; files are written atomically and meta.json last.",
         "Detect damaged files, tie results to code/data/library versions, and never leave a half-updated model set.",
         "sha256 of each artefact; os.replace for atomic writes.",
         ["test_artifact_checksums_match_meta", "test_libraries_match_requirements_file", "test_pipeline_is_reproducible_with_fixed_seed"], OK_T, ""])
    add([D, P_IMPL + " (ML)", "ml/train.py: resumable training and profiles",
         "An interrupted training run had to start again from the beginning (about 7 minutes at that time).",
         "Each expensive unit (every cross-validation configuration, each tuning search, each calibration) is checkpointed under a cache keyed on profile, data and code; rerunning the command resumes. Profiles: full, quick, tiny; non-full results are marked preliminary.",
         "The sandbox restored files but not running processes between work sessions; resumability also helps users on slow machines.",
         "joblib checkpoints with a sha256 cache key; cache deleted on success.",
         ["test_this_is_a_full_run_not_a_preliminary_one"], OK_M, "An interrupted tiny run (killed after 12 s, 11 checkpoints) resumed and produced identical results."])
    add([D, P_IMPL + " (ML)", "models/*, reports/train_log_full.txt",
         "Models trained with the flawed pipeline (version 20261115-1026).",
         f"Retrained with the corrected pipeline, full profile: version {meta['version']}. Default model {lab[meta['default_model']]}.",
         "The corrected pipeline, features and ensemble change the artefacts.",
         "Full run executed in checkpointed chunks (about 4.5 minutes of work in the final session after warm caches).",
         ["test_this_is_a_full_run_not_a_preliminary_one", "test_meta_and_metrics_describe_the_same_training_run", "test_everything_loads_and_the_designated_default_is_available"], OK_T, ""])
    # ------------------------------------------------------------------ backend
    add([D, P_IMPL + " (Backend)", "app/__init__.py: CSRF protection",
         "A POST with no session and an EMPTY csrf_token was accepted: hmac.compare_digest('', '') is True. Verified during the audit: the request returned 302 instead of 400.",
         "A POST is accepted only if the token is present, non-empty and equal to the session token. Cross-site Origin headers are rejected. The JSON API accepts only application/json.",
         "Missing or empty tokens must never pass.",
         "before_request now requires 'expected and sent and compare_digest(sent, expected)'.",
         ["test_csrf_rejects_missing_empty_and_wrong_tokens_even_without_a_session", "test_csrf_rejects_bad_tokens_when_a_session_token_exists", "test_csrf_accepts_the_real_token_and_protects_alert_actions", "test_real_page_token_round_trip", "test_cross_site_origin_is_blocked_even_for_json"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/__init__.py: configuration and secrets",
         "SECRET_KEY fell back to the hard-coded string 'dev-only-change-me'; no cookie flags; paths fixed in code.",
         "Configuration comes from environment variables (APP_ENV, SECRET_KEY, DATABASE_PATH, MODELS_DIR, REPORTS_DIR, SESSION_COOKIE_SECURE, MAX_UPLOAD_BYTES). Production refuses to start without SECRET_KEY; development uses a random per-process key. Session cookie is HttpOnly, SameSite=Lax and Secure in production.",
         "No development secrets in production, no hard-coded paths.",
         "create_app reads os.environ and test overrides; .env.example documents the variables.",
         ["test_production_requires_a_secret_key", "test_no_hardcoded_secret_in_source", "test_development_secret_is_random_per_process", "test_environment_variables_configure_paths", "test_security_headers_and_cookie_flags"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/__init__.py: security headers, errors",
         "No Content-Security-Policy or other security headers; unhandled errors showed a generic page; JSON API errors were HTML.",
         "Content-Security-Policy (own scripts only), X-Content-Type-Options, X-Frame-Options, Referrer-Policy; understandable HTML or JSON errors (400, 404, 413, 415, 422, 500, 503).",
         "Defence in depth and understandable errors.",
         "after_request adds headers; HTTPException, ServiceError and generic handlers choose HTML or JSON by path.",
         ["test_security_headers_and_cookie_flags", "test_pages_use_no_inline_scripts", "test_oversized_upload_is_rejected_cleanly", "test_xss_is_escaped_and_hostile_machine_ids_are_rejected", "test_sql_injection_attempts_are_harmless"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/predictor.py: ModelRegistry loading",
         "All models loaded in one loop; any exception (for example the XGBoost error) stopped the whole application from starting.",
         "Each model loads independently (file present, sha256 matches meta.json, loads, probe prediction in [0, 1]). A failure disables only that model and records the reason. The ensemble needs its members. The default falls back to the best available model by training ranking; if nothing loads the application still starts and reports it.",
         "One corrupted artefact must not take the application down; the default must be available before predictions are accepted.",
         "ModelRegistry.models / errors dictionaries, default_model property, ready flag; service refuses analyses with HTTP 503 if not ready.",
         ["test_a_missing_model_file_disables_only_that_model_and_the_ensemble", "test_a_corrupted_file_is_detected_by_checksum_and_explained", "test_a_garbage_file_with_no_recorded_checksum_is_still_caught", "test_default_falls_back_to_best_available_by_training_ranking", "test_missing_meta_json_leaves_the_registry_not_ready_but_does_not_raise", "test_app_starts_with_no_models_and_explains_instead_of_crashing", "test_app_serves_predictions_when_only_some_models_loaded", "test_failure_mode_models_are_optional"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/predictor.py: explain",
         "The 'effect of each reading' replaced a reading by its median and was recomputed from the current models whenever a result page was opened.",
         "Exact Shapley values over five sensor groups (type, temperatures, speed, torque, tool wear) against a background sample of typical training readings, computed once at prediction time and stored.",
         "Single replacement misses interactions; effects were not additive; results could change after retraining.",
         "All 32 subsets of groups are evaluated in one batch call; contributions sum exactly to the prediction minus the background mean (checked to 1e-9).",
         ["test_shapley_contributions_add_up_exactly"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/service.py (new)",
         "Routes called the model and the database directly; the API repeated the form logic.",
         "One workflow validate -> predict -> store used by the form, the JSON API and the CSV upload; raises ValidationFailed / ServiceUnavailable with readable messages.",
         "Avoid duplicated logic and make errors consistent.",
         "analyze_and_store, analyze_batch, what_if.",
         ["test_analyze_form_to_result_page", "test_api_predict_and_errors", "test_batch_saves_valid_rows_and_updates_machines_and_alerts"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/service.py: analyze_batch (new)",
         "No batch prediction.",
         "A CSV of readings is validated row by row; valid rows are predicted in one call and saved in one transaction; rejected rows are listed with line numbers; limits on size, rows and encoding.",
         "Evaluated as a genuinely useful addition.",
         "csv.DictReader, per-row validate, machine-type conflict check within the file and against stored machines, then registry.analyze_many(explain=False).",
         ["test_batch_saves_valid_rows_and_updates_machines_and_alerts", "test_batch_reports_bad_rows_with_line_numbers_and_saves_nothing_for_them", "test_batch_file_problems_are_explained", "test_batch_row_limit_and_encoding", "test_batch_template_is_valid_input_for_batch"], OK_T, "Uploaded through the browser: 1 row accepted, 2 rejected."])
    add([D, P_IMPL + " (Backend)", "app/service.py: what_if (new)",
         "No what-if analysis.",
         "Re-runs the model that produced a saved analysis with changed readings; nothing is saved; the answer carries a non-causal disclaimer; unavailable if the model version has changed.",
         "Evaluated as a genuinely useful addition; must not imply causation.",
         "POST /api/what-if; validated with the same rules; uses the stored model key and version.",
         ["test_what_if_api", "test_what_if_unavailable_for_a_different_model_version"], OK_T, "Used in the browser (torque 51.8 -> 30)."])
    add([D, P_IMPL + " (Backend)", "app/validation.py",
         "Machine ID optional (blank became AUTO-n); model field required to be one of the models.",
         "Machine ID required and pattern-checked; model optional (default model); field help text added for the UI.",
         "Supports stable machine identity and a simpler normal workflow.",
         "validate(form, ranges, available_models, default_model).",
         ["test_machine_id_is_required", "test_valid_input_uses_default_model", "test_machine_id_characters", "test_bad_type_and_model"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/risk.py",
         "Thresholds, rule-based 'condition checks' and a 'strain' value; no health score.",
         "Adds health score (round(100 x (1 - p)), documented as not independently validated), renames checks to rule_indicators, adds failure_mode_rows (ML estimate and rule check side by side), RNF note, wear_torque naming.",
         "Distinguish ML prediction, rule-based indicators and suggested actions; never present suggestions as diagnoses.",
         "Pure functions with no framework dependency.",
         ["test_health_score_formula", "test_rule_indicators_and_naming", "test_failure_mode_rows_separate_model_estimate_from_rule_check", "test_rnf_is_explained_not_predicted", "test_thresholds_and_boundaries"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/routes.py: result page",
         "After POST the page recomputed the prediction from the stored readings with the CURRENT models and, if different, replaced the numbers with stored ones.",
         "The result page at /predictions/<id> renders the stored snapshot only. What-if is offered only while the same model version is loaded.",
         "Historical results must not change when models are retrained or replaced.",
         "details_json holds the explanation, indicators and mode estimates; Post/Redirect/Get avoids duplicate saves.",
         ["test_history_does_not_change_when_models_are_replaced", "test_analyze_form_to_result_page"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/routes.py: JSON API",
         "/api/predict returned fields tied to the old schema; non-JSON requests got a CSRF error.",
         "/api/predict, /api/what-if and /api/health; JSON only (415 otherwise); JSON errors with field details; 503 when not ready.",
         "Clear API contract.",
         "request.is_json; handlers return JSON for paths under /api/.",
         ["test_api_predict_and_errors", "test_what_if_api"], OK_T, ""])
    add([D, P_IMPL + " (Backend)", "app/routes.py: CSV export",
         "Export had a Failure Type column that was always empty.",
         "Export follows the proposal's column order and adds machine type, health score, model version, source, rule indicators and failure-mode estimates; it respects the active filters.",
         "Complete, honest export.",
         "export_rows joins machines; details_json parsed for indicators.",
         ["test_history_filters_and_export", "test_export_rows_use_proposal_column_order"], OK_T, ""])
    # ------------------------------------------------------------------ database
    add([D, P_IMPL + " (Database)", "app/db.py: machines table",
         "machine_id was a text column on every prediction. A blank ID became a new 'AUTO-<row id>' for every prediction: four blank submissions created four machines (verified in the audit).",
         "machines table with a unique, case-insensitive machine_code; predictions reference it by foreign key; the machine's type is immutable and a conflicting type is refused.",
         "A machine must keep one identity across analyses; separate machine identity from prediction record IDs.",
         "get_or_create_machine inside record_prediction's transaction; trigger trg_machines_identity.",
         ["test_same_machine_id_keeps_one_identity_across_predictions", "test_machine_type_is_part_of_identity", "test_machine_identity_cannot_be_edited", "test_one_machine_keeps_one_identity_across_analyses", "test_machine_type_conflict_is_refused_with_a_clear_message"], OK_T, ""])
    add([D, P_IMPL + " (Database)", "app/db.py: alert lifecycle",
         "Every Warning/Critical prediction was a separate 'open' alert; alerts stayed open after a later Normal reading (verified: Critical, Critical, Normal left two open alerts).",
         "One alert per machine while active, with statuses open, acknowledged, resolved. Repeats update the alert; a higher level re-opens an acknowledged alert; a lower level is recorded; a Normal reading resolves it; a new problem after resolution opens a new alert. Every step is an alert_events row.",
         "Prevent duplicate alerts, avoid stale alerts and preserve alert history.",
         "apply_alert_lifecycle plus a partial unique index uq_one_active_alert_per_machine, so the database itself refuses a second active alert.",
         ["test_warning_or_critical_opens_one_alert_normal_does_not", "test_repeated_readings_do_not_create_duplicate_alerts", "test_database_itself_refuses_a_second_active_alert", "test_escalation_reopens_an_acknowledged_alert", "test_downgrade_is_recorded_and_keeps_peak", "test_later_normal_reading_resolves_the_alert_and_a_new_problem_opens_a_new_one", "test_manual_acknowledge_and_resolve_rules", "test_alerts_of_different_machines_are_independent", "test_alert_lifecycle_over_http"], OK_T, ""])
    add([D, P_IMPL + " (Database)", "app/db.py: immutability and integrity",
         "Any code (or a person with the file) could UPDATE or DELETE saved predictions; foreign keys were not enforced; no CHECK constraints.",
         "Triggers raise an error on UPDATE/DELETE of predictions and on any change to alert_events; PRAGMA foreign_keys = ON on every connection; CHECK constraints on levels, statuses and ranges; indexes on machine/time, time, risk, alert status.",
         "Historical results must never change; data must stay consistent.",
         "SQL triggers with RAISE(ABORT, ...).",
         ["test_saved_predictions_cannot_be_changed_or_deleted", "test_alert_history_is_append_only", "test_foreign_keys_are_enforced", "test_check_constraints_reject_bad_values", "test_fresh_database_has_version_foreign_keys_and_indexes", "test_prediction_preserves_every_required_field"], OK_T, ""])
    add([D, P_IMPL + " (Database)", "app/db.py: dashboard and queries",
         "Dashboard counted rows by risk level; 'machines' meant distinct text IDs.",
         "Machine counts use each machine's LATEST analysis; active alerts and average health are computed from stored data; empty database yields zeros and an empty state.",
         "Displayed statistics must represent actual stored data.",
         "list_machines joins the latest prediction and active alert; dashboard_stats.",
         ["test_dashboard_is_empty_then_reflects_latest_status_per_machine", "test_dashboard_numbers_match_stored_data", "test_user_input_never_reaches_sql_text"], OK_T, ""])
    add([D, P_IMPL + " (Database)", "app/migrations.py (new)",
         "No schema versioning or migration; a v1 database could not be used by the new application.",
         "PRAGMA user_version tracks the schema. A fresh database is created at version 2; a v1 database is backed up, then migrated in one transaction (prediction ids and values preserved; machines rebuilt case-insensitively; alerts rebuilt by replaying through the new lifecycle; acknowledgements carried over; original table kept as legacy_predictions_v1). Dry run available; unrecognised schemas are never modified.",
         "Preserve existing data wherever possible with a safe migration.",
         "python -m app.migrations [--dry-run]; also run at start-up. Migration test uses a v1 database containing duplicate alerts, mixed-case IDs, a type conflict and an acknowledged alert.",
         ["test_dry_run_changes_nothing", "test_migration_preserves_data_and_reports", "test_migrated_alerts_follow_the_new_lifecycle", "test_migration_is_idempotent_and_result_is_immutable", "test_failed_migration_rolls_back_and_leaves_the_database_usable", "test_unrecognised_schema_is_never_modified", "test_app_starts_on_a_v1_database_and_serves_migrated_history"], OK_T, ""])
    add([D, P_IMPL + " (Database)", "app/__init__.py: database failures",
         "A database error at start-up would have crashed the application.",
         "The failure is recorded; the application starts, shows a banner and returns HTTP 503 for pages that need the database; About and Model Insights still work.",
         "Handle database failures and give understandable errors.",
         "db_error stored in app.extensions; routes call conn() which raises ServiceUnavailable.",
         ["test_database_failure_is_reported_not_fatal"], OK_T, ""])
    # ------------------------------------------------------------------ UI
    add([D, P_IMPL + " (UI)", "templates/base.html, _macros.html, static/css/app.css",
         "Seven technical pages; model comparison details and dataset statistics were spread across pages; sidebar background was transparent on wide screens.",
         "Sections: Dashboard, Analyze Machine, Machines, Alerts, History, Model Insights, About. Health bar, status chips, banners for degraded operation, skip link, focus outlines. Sidebar background forced dark.",
         "The application felt complicated; it should read like a product.",
         "Bootstrap's offcanvas-lg forces a transparent background; overridden with !important.",
         ["test_every_page_loads", "test_pages_use_no_inline_scripts"], OK_M, "Viewed every page at 1440 px and 390 px in Chromium."])
    add([D, P_IMPL + " (UI)", "templates/dashboard.html",
         "Counts of predictions by risk level, a histogram and a per-model table; empty state only.",
         "Machines tracked, need attention, active alerts, average health, status bar, attention table, recent analyses; first-time guide.",
         "Show machine health, not prediction statistics.",
         "db.dashboard_stats.",
         ["test_dashboard_numbers_match_stored_data", "test_first_time_user_sees_meaningful_empty_states_and_no_fake_numbers"], OK_T, "Averages checked by hand against the browser view."])
    add([D, P_IMPL + " (UI)", "templates/analyze.html, static/js/main.js",
         "A form with model choice on the main page and no explanation of the readings.",
         "Two-step form: machine, then readings with units, meaning, typical and accepted ranges; example buttons; machine-type hint for known machines; model choice moved to collapsed Advanced options; Analyzing... state.",
         "Simple, understandable input with validation.",
         "Server-side validation with field messages; JavaScript only fills fields and shows state.",
         ["test_analyze_form_to_result_page", "test_invalid_input_is_explained_and_nothing_is_saved", "test_analyze_again_prefills_the_machine", "test_result_page_does_not_expose_model_choice_but_advanced_option_exists"], OK_T, "Examples clicked in the browser."])
    add([D, P_IMPL + " (UI)", "templates/result.html",
         "Technical result page (probability, bar, derived features, sensitivity bars).",
         "Answer first (probability, status, health score, bar, recommended action, change since last analysis), then 'What drove this result', 'Possible failure modes', what-if, and collapsed technical details.",
         "Most important information first; technical details expandable.",
         "Renders the stored snapshot only.",
         ["test_analyze_form_to_result_page", "test_previous_analysis_comparison_on_result_page"], OK_T, "Two display bugs found in the browser and fixed: '-0.0 pp' and a duplicated health number."])
    add([D, P_IMPL + " (UI)", "templates/machines.html, machine.html",
         "No machine-wise view.",
         "Machine list and machine page with status, health, alert, trend chart with 30%/70% lines and recent analyses.",
         "Machine-wise history and risk trends.",
         "db.list_machines, db.machine_detail; Chart.js.",
         ["test_machine_page_shows_trend_and_history", "test_machine_list_and_detail"], OK_T, "Chart edge clipping and legend icons fixed after viewing in the browser."])
    add([D, P_IMPL + " (UI)", "templates/alerts.html, alert.html",
         "Flat list of every Warning/Critical prediction with Acknowledge/Reopen.",
         "Counts, Active / Resolved / All views, reason, recommended action, Acknowledge and Resolve, alert detail with event timeline and optional note.",
         "Clear severity, reason, action, acknowledge, resolve and alert history.",
         "Alert lifecycle in db.py.",
         ["test_alert_lifecycle_over_http", "test_manual_resolve_and_invalid_actions"], OK_T, "Acknowledge clicked in the browser."])
    add([D, P_IMPL + " (UI)", "templates/history.html",
         "Table of predictions with machine ID text.",
         "Machine links, health score, model version on hover, filters, sorting, pagination, export that respects filters; table width fitted at 1440 px.",
         "Search and filter, machine-wise navigation, model version information, CSV export.",
         "-",
         ["test_history_filters_and_export", "test_filters_sorting_and_pagination"], OK_T, ""])
    add([D, P_IMPL + " (UI)", "templates/insights.html, static/js/charts.js",
         "Model Performance page with three models.",
         "Model Insights with four models: which model is used and why, metrics at three thresholds, ensemble comparison, curves, confusion matrices, threshold evidence, missed failures, failure-mode classifiers, importance, methodology and model status.",
         "Keep model comparison in a technical section; include ensemble and explainability.",
         "Reads metrics.json and meta.json; no numbers are typed into templates.",
         ["test_model_insights_page_reports_all_four_models"], OK_T, "Threshold and model toggles used in the browser."])
    # ------------------------------------------------------------------ tests, docs, verification
    add([D, P_IMPL + " (Tests)", "tests/ (all files)",
         "54 tests, written for the v1 schema and routes.",
         f"{T0} tests: features, preprocessing and resampling, all models and the ensemble, explanations, loading failures, database, migration, security and web flows. A small real model set is trained once per session (about 30 s); a fixed-probability stub makes Normal / Warning / Critical flows deterministic.",
         "Review existing tests and add missing ones; do not claim anything is tested that was not executed.",
         "conftest.py fixtures: tiny_models, models_copy, FixedModel.",
         ["test_every_page_loads"], OK_T, f"Full suite run on Python 3.12 and 3.14: {T0} passed on each."])
    add([D, P_IMPL + " (Docs)", "README.md, docs/ARCHITECTURE.md, docs/API.md, docs/PROJECT_STATUS.md, .env.example, this workbook",
         "A README for the earlier version.",
         "Installation per shell (including the virtual-environment activation that caused a problem), environment variables, training, database, running, tests, API, architecture, limitations, troubleshooting.",
         "A new developer must be able to install, train, test and run the project from the documentation.",
         "The workbook is built by docs/build_workbook.py from models/metrics.json, models/meta.json and the pytest report, and fails if a cited test did not pass.",
         [], "Written", "Commands in the README were exercised (install check, tests, run, migrations --dry-run)."])
    add([D, P_VERIFY, "Whole project", "-", f"Full test suite: {T0} passed on Python 3.12 and on Python 3.14 (Linux). Browser walk-through with the shipped models in Chromium (desktop and 390 px): no JavaScript errors.",
         "Verify before claiming.", "Test report parsed into this workbook.", ["test_every_page_loads"], OK_T, "Not verified on Windows; the XGBoost loading fix is untested there."])
    # ------------------------------------------------------------------ 2026-11-24 additions
    b = M["baselines"]
    x = M["test"][meta["default_model"]]["thresholds"]["0.3"]
    add([D2, P_IMPL + " (Data)", "ml/data_io.py (new); ml/features.py: load_data; ml/train.py; ml/eda.py; app/routes.py: batch; templates/batch.html",
         "Training, EDA and batch upload read CSV only.",
         "A dataset can be an .xlsx or a .csv. An .xlsx is converted once to a .csv next to it (first sheet, or --sheet NAME), and from then on the data is only ever read as CSV. The batch upload page also accepts .xlsx (converted in memory, same row and size limits).",
         "Allow Excel input without changing how the rest of the project reads data.",
         "ensure_csv() returns CSV paths unchanged and converts .xlsx/.xlsm with openpyxl in read-only mode (cached results, blank rows skipped, 1551.0 written as 1551). Uploads use xlsx_bytes_to_csv_bytes() and then the existing CSV code; unknown extensions and invalid workbooks give a 422 message. openpyxl was added to requirements.",
         ["test_batch_accepts_xlsx_same_as_csv", "test_batch_rejects_bad_xlsx_and_unknown_types", "test_ensure_csv_converts_once_and_load_data_reads_it"], OK_T, ""])
    add([D2, P_IMPL + " (Data)", "ml/features.py: validate_schema, DataSchemaError, load_data",
         "A dataset with a wrong or misspelt column failed later with a pandas KeyError or silently produced NaN.",
         "load_data checks the columns and value types first. Harmless header differences (case, extra spaces, BOM) are fixed; missing columns are listed with a 'did you mean' hint; non-numeric cells are reported with file line numbers; an empty dataset is refused.",
         "A custom dataset must fail early with a message a person can act on.",
         "validate_schema returns a dataframe with canonical column names and numeric columns converted; UDI and Product ID stay optional. The existing data-quality report (missing values, impossible values, duplicates) still runs afterwards in training.",
         ["test_schema_check_explains_missing_and_bad_columns", "test_real_dataset_passes_quality_checks"], OK_T, ""])
    add([D2, P_IMPL + " (ML)", "ml/baselines.py (new); ml/train.py; templates/insights.html",
         "Model results were shown without a reference point other than 'always predict no failure' accuracy.",
         f"Two baselines are computed on the same test split and shown on Model Insights: always 'no failure' (accuracy {b['never_fails']['accuracy'] * 100:.1f}%, recall 0) and documented failure rules without machine learning. All rules (HDF, PWF, OSF, tool wear >= 200 min): precision {b['rule_based']['precision']:.3f}, recall {b['rule_based']['recall']:.3f}, F1 {b['rule_based']['f1']:.3f}. Physical rules only (HDF, PWF, OSF): precision {b['rule_physical']['precision']:.3f}, recall {b['rule_physical']['recall']:.3f}, F1 {b['rule_physical']['f1']:.3f}. The default model ({lab[meta['default_model']]}) at 30%: precision {x['precision']:.3f}, recall {x['recall']:.3f}, F1 {x['f1']:.3f}.",
         "Accuracy alone is misleading at 3.4% failures, and a trained model should be compared with what a fixed rule already achieves.",
         "The rule baseline reuses the app's documented conditions, vectorised, and is tested to agree with app/risk.py. It is not tuned on the test set. The result is reported as it is: in this synthetic dataset the target is generated by these rules, so a rules-only baseline can match or beat a trained model. This says more about the dataset than about the models. For the shipped models the baselines were added without retraining (python -m ml.baselines recomputes the identical split and refuses if it does not match metrics.json).",
         ["test_rule_baseline_matches_the_apps_rule_indicators", "test_baselines_never_fails_has_zero_recall_and_high_accuracy"], OK_T, ""])
    add([D2, P_IMPL + " (Deployment)", "ml/compat.py (new); app/predictor.py; requirements.txt; requirements-lock.txt (new)",
         "requirements.txt pinned exact versions, so a newer Python or library could not install, and a version mismatch with the saved models was silent.",
         "requirements.txt now allows compatible ranges; requirements-lock.txt holds the exact versions the shipped models were trained with. At start-up the installed versions are compared with models/meta.json and any difference is logged and shown on Model Insights (and in /api/health as version_warnings).",
         "Easier installation without hiding a real risk: saved scikit-learn and XGBoost models are only guaranteed to load with the training versions.",
         "library_mismatches compares scikit-learn, xgboost, imbalanced-learn and numpy exactly and Python at major.minor. The warning never blocks start-up; the existing per-model load check still decides whether a model is usable.",
         ["test_library_version_check_warns_only_on_differences", "test_libraries_match_requirements_file"], OK_T, ""])
    add([D2, P_IMPL + " (Data)", "app/service.py: analyze_batch; templates/batch.html",
         "Uploading the original AI4I dataset on the Batch page failed with 'Missing column(s)' because its headers differ from the batch format.",
         "The Batch page now also accepts the AI4I column names (Product ID, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]). Product ID becomes the machine ID; UDI, Machine failure and the failure-mode flags are ignored. The row limit message now says to upload at most 1000 rows at a time.",
         "The dataset the project was built on should be usable in the app without editing it.",
         "Header aliasing happens before validation, so every existing check (types, ranges, one type per machine) still applies. The 1000-row limit is unchanged, so the 10,000-row file must be uploaded in parts.",
         ["test_batch_accepts_the_original_ai4i_dataset_columns"], OK_T, ""])
    ev = M["evaluation"]
    import json as _json, os as _os
    tw = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reports", "twf_study.json")))
    add([D2, P_IMPL + " (App)", "app/risk.py: maintenance_notices; app/predictor.py; app/routes.py; app/service.py; templates/result.html, batch.html",
         "Tool-wear failures were rated Normal by the model (it caught 1 of 10 in the test set) and nothing on the result page mentioned the worn tool unless the status was Warning or Critical.",
         "A separate maintenance notice 'High tool wear: inspect the tool and consider replacement' appears on the result page, in batch results (a Maintenance column and a count) and in the API (maintenance field) whenever tool wear is 200 min or more, whatever the status.",
         "Make a known maintenance condition visible without pretending the model can predict it.",
         f"The notice is stored in each analysis (details.maintenance); older analyses derive it from the stored reading. It never changes the status, the probability or the alerts. In the test set, {ev['notice']['test_failures_with_notice']} of {ev['notice']['test_failures']} failures occurred among the {ev['notice']['test_machines_with_notice']} machines with the notice ({ev['notice']['failure_rate_with_notice_pct']}% failed versus {ev['notice']['failure_rate_without_notice_pct']}% of the rest). {ev['missed']['with_tool_wear_notice']} of the {ev['missed']['total']} failures the model missed had the notice.",
         ["test_tool_wear_notice_threshold_and_wording", "test_tool_wear_notice_on_result_page_without_changing_status_or_alerts", "test_tool_wear_notice_in_api_and_batch"], OK_T, ""])
    add([D2, P_IMPL + " (ML)", "ml/evaluation.py (new); ml/train.py; templates/insights.html; static/js/charts.js",
         "Model Insights showed one overall table, PR curves for the overall models and the number of missed failures by type, but no per-failure-type precision and recall, confusion matrices, list of missed failures or per-type threshold comparison.",
         "Model Insights now has results for each failure type (precision, recall, F1, confusion matrix, share caught by the overall model), a table of every missed failure with its readings and whether the notice would flag it, precision-recall curves by failure type (cross-validation and test), and a per-type threshold comparison.",
         "Per-type evaluation and false-negative analysis are what show where the model fails.",
         "Threshold comparisons use out-of-fold predictions on the training split only. Test-set figures are descriptive and carry wide uncertainty because each type has few failures. The block is produced by training; python -m ml.evaluation adds it to existing models without retraining, using the identical split. The global 30% / 70% thresholds are unchanged.",
         ["test_evaluation_block_is_consistent_with_the_main_test_results", "test_wilson_interval_and_threshold_helper"], OK_T, ""])
    add([D2, P_IMPL + " (ML)", "ml/twf_study.py (new); reports/twf_study.json; templates/insights.html",
         "The tool-wear failure model had a validation PR-AUC of about 0.06 and no analysis of why.",
         "A study on the training split only: tool-wear failure rate by wear band with 95% intervals, and 5x5 repeated cross-validation of simple wear-based estimates against the shipped failure-mode model. " + tw["verdict"],
         "Improving TWF was proposed. The evidence shows it cannot be improved reliably, so the project documents the limit instead of claiming a better model.",
         f"{tw['twf_in_training']} tool-wear failures in training. Almost all occur at 200 min or more but only about 1 in 20 such tools fails from wear, rising slowly with wear. The main Random Forest, the failure-mode models and the thresholds are unchanged. The test set was not used. The same script is the template for judging any later model change without touching the test set.",
         ["test_twf_study_uses_training_data_only_and_adds_up"], OK_T, ""])
    return R


def decisions(M, meta, T):
    lab = meta["model_names"]
    d = meta["default_model"]
    return [
        ["Technology stack", "Flask + Jinja + Bootstrap/custom CSS + Chart.js + SQLite", "Streamlit; React + FastAPI", "The team knew Flask; server-rendered pages give full design control with one codebase; SQLite needs no server.", "Streamlit gives little control over the design requested; React + FastAPI doubles the code and adds a build step for no needed benefit.", "app/"],
        ["Selection metric", "PR-AUC (with precision, recall, F1, Brier reported)", "Accuracy; ROC-AUC; F1 at a fixed threshold", "Only 3.4% of records are failures; PR-AUC measures ranking quality on the rare class and is threshold-independent.", "Accuracy is 96.6% for a model that predicts 'no failure' always; ROC-AUC is flattered by the many easy non-failures; F1 depends on a threshold.", "ml/train.py"],
        ["Selection evidence", "Out-of-fold predictions on the TRAINING split only", "Pick the best on the test set", "The test set must give an unbiased final report.", "Choosing on the test set would make the reported result optimistic.", "ml/train.py"],
        ["Final default model", f"{lab[d]}", "Ensemble; XGBoost; Logistic Regression", f"Highest validation PR-AUC ({M['selection']['table'][1]['cv_pr_auc']:.3f}) and lowest Brier score; captures the rule-like thresholds in the data.", "Ensemble is lower on validation (Logistic Regression drags it down); XGBoost lower PR-AUC and Brier; Logistic Regression cannot model the U-shaped relationships.", "models/meta.json"],
        ["Imbalance handling", "None (plain training) plus probability calibration", "Class weights; SMOTENC", "Compared by cross-validated PR-AUC; neither alternative beat plain training by more than the 0.005 tolerance for any model.", "Class weights and SMOTENC changed ranking little (or hurt) and distort probabilities, which then need correcting anyway.", "ml/train.py, metrics.json 'ablation'"],
        ["SMOTENC order", "Ordinal-encode machine type, resample with one categorical column, then one-hot encode", "Keep one-hot before SMOTENC; drop SMOTE", "SMOTENC must see machine type as one categorical variable to generate valid samples.", "One-hot before SMOTENC produced impossible machine types (115 of 7,458 synthetic rows in the audit).", "ml/pipelines.py"],
        ["Probability calibration", "Sigmoid or isotonic, chosen per model by out-of-fold Brier score", "No calibration; fixed method", "The 30%/70% thresholds mean something only if probabilities are honest.", "Raw forest and boosting scores are not calibrated.", "ml/train.py"],
        ["Alert thresholds", "Normal < 30%, Warning 30% to < 70%, Critical >= 70%", "Thresholds optimised on data", "Given by the project proposal; the validation scan shows the recall/precision trade-off is reasonable; lower bound inclusive.", "No maintenance cost model exists to justify an optimum; tuning on the test set would be leakage.", "app/risk.py"],
        ["Ensemble construction", "Soft voting; weights only if validation-justified and a real blend", "Hard voting; stacking; always tuned weights", "Probabilities are calibrated so averaging is meaningful; the guard prevents over-fitting weights and one-model 'ensembles'.", "Stacking adds a trained meta-model on only 271 failures; always-tuned weights collapsed to Random Forest alone.", "ml/ensemble.py, ml/train.py"],
        ["Failure-type information", "Separate per-mode classifiers (reliable modes only) shown beside rule checks, both labelled as suggestions", "Claim the binary model predicts failure type; rules only; one multi-class model", "Keeps ML estimates, rules and suggestions distinct; honest about reliability (TWF excluded, RNF not modelled).", "A binary model cannot identify modes; rules alone are not ML; multi-class on 339 failures with overlapping modes would be unreliable.", "ml/train.py, app/risk.py"],
        ["Explanations", "Exact Shapley values over five sensor groups", "SHAP library; single-feature replacement; model coefficients only", "Additive, handles interactions, works identically for all four models, no new dependency, computed exactly in ~0.2 s.", "SHAP adds a dependency; single replacement ignores interactions; coefficients exist only for Logistic Regression.", "app/predictor.py"],
        ["Machine identity", "Separate machines table; machine ID required; type immutable", "Free-text ID per prediction; auto-generated ID when blank", "A machine needs one identity across analyses for history, trends and alerts.", "Auto IDs created a new machine per prediction.", "app/db.py"],
        ["Alert model", "One active alert per machine with open/acknowledged/resolved and an event history; auto-resolve on a Normal reading", "One alert per prediction; manual resolve only", "Avoids duplicates and stale alerts; history preserved. The partial unique index enforces it in the database.", "Per-prediction alerts flood the list; manual-only resolution leaves alerts open after recovery.", "app/db.py"],
        ["History immutability", "Database triggers plus a stored snapshot on each prediction", "Application-level convention only; recompute on view", "Guarantees history cannot change when models are retrained, even through mistakes.", "Recomputing on view changes old results after retraining.", "app/db.py"],
        ["Model persistence", "joblib files with sha256 in meta.json; XGBoost in JSON form; ensemble rebuilt at load", "One bundle file; XGBoost default pickle; pickle the ensemble", "Independent loading isolates failures; checksums catch damage; JSON is portable.", "A single bundle fails entirely if anything is wrong; default XGBoost pickle failed on the user's machine.", "app/predictor.py, ml/pipelines.py"],
        ["Security", "Per-session CSRF token (non-empty both sides), Origin check, CSP, env-based secrets", "Framework CSRF extension; hard-coded secret", "Small, auditable code with no new dependency; secrets never in source.", "Extra dependency not justified for this scope.", "app/__init__.py"],
        ["Health score", "round(100 x (1 - probability))", "Weighted composite of sensors", "Transparent and honest: it is the probability on a friendlier scale.", "A composite would imply a validated physical measure that the data cannot support.", "app/risk.py"],
        ["Not implemented", "Live IoT monitoring, remaining useful life, user accounts", "-", "The dataset has no time dimension; inputs are manual; accounts are out of scope.", "Implementing them would require data or architecture the project does not have.", "-"],
    ]


def viva(M, meta, eda, T):
    lab = meta["model_names"]
    t = M["test"]
    sel = {r["model"]: r for r in M["selection"]["table"]}
    x = lambda k: t[k]["thresholds"]["0.3"]
    d = meta["default_model"]
    e = M["ensemble"]
    q = []
    A = q.append
    # overview
    A(["Project overview", "What does your project do?", "It takes a machine's current sensor readings, estimates the probability that the machine is failing using a trained model, classifies it as Normal, Warning or Critical, recommends an action, explains which readings drove the result and raises and tracks an alert. All results are stored and can be reviewed and exported."])
    A(["Project overview", "How is this different from reactive and scheduled maintenance?", "Reactive maintenance repairs after a breakdown; scheduled maintenance acts at fixed intervals regardless of condition. Predictive maintenance uses the machine's condition to decide when to act, so work is done when risk is elevated, not too late and not needlessly."])
    A(["Project overview", "Is this real-time monitoring?", "No. Readings are entered by hand or uploaded as a CSV. There is no live sensor connection. We say so in the application and the documentation."])
    A(["Project overview", "Does the probability tell us when the machine will fail?", "No. The dataset has independent snapshots with no time order, so the model estimates whether the current readings look like a failing state. It is not remaining useful life or time to failure."])
    A(["Project overview", "What are the inputs and outputs?", "Inputs: machine ID, machine type (L/M/H), air and process temperature (K), rotational speed (rpm), torque (Nm), tool wear (min). Outputs: failure probability, status, health score, recommended action, Shapley explanation, possible failure modes, alert and stored history."])
    # dataset
    A(["Dataset", "Which dataset did you use and how big is it?", f"AI4I 2020 Predictive Maintenance: {eda['rows']:,} records, {eda['failures']} failures ({eda['failure_rate_pct']}%). It is synthetic, generated to resemble industrial data."])
    A(["Dataset", "Why does it matter that the dataset is synthetic?", "Its failure modes follow fixed rules (for example heat dissipation failure always happens when the temperature difference is below 8.6 K and speed below 1380 rpm). Tree models can learn such rules almost perfectly, so our scores are probably much higher than on real machines. We cannot claim real-world generalisation without external validation."])
    A(["Dataset", "How did you handle missing values and duplicates?", f"We check them before every training run: missing values {M['data_quality']['missing_values']}, duplicate rows {M['data_quality']['duplicate_rows']}, invalid types or impossible values none. If a check failed, training would stop."])
    A(["Dataset", "What are the failure modes TWF, HDF, PWF, OSF and RNF?", "TWF tool wear failure, HDF heat dissipation failure, PWF power failure, OSF overstrain failure, RNF random failure. They are 0/1 flags that make up the failure label. We never use them as inputs because that would leak the answer."])
    A(["Dataset", "Why is RNF not predicted?", f"It is random by design: only {eda['rule_checks']['RNF']['of_which_machine_failure']} of the {eda['rule_checks']['RNF']['flagged']} RNF-flagged records is even labelled a machine failure, so no model can predict it from sensor data."])
    A(["Dataset", "What is data leakage and how did you avoid it?", "Leakage is when information from outside the training data (or the answer itself) influences the model. We avoid it by dropping identifier and failure-mode columns, fitting scaling, resampling, tuning, calibration, ensemble weights and model selection on training folds only, and using the test set once for reporting."])
    # features
    A(["Feature engineering", "Which features did you engineer and why?", "Temperature difference = process - air (heat generated by the process); approximate power = torque x 2 pi x rpm / 60 (the dataset's power failure depends on it); and wear x torque (load on a worn tool). They express relationships the failure rules use, making them easier to learn."])
    A(["Feature engineering", "Is wear x torque the physical strain?", "No. It is a heuristic proxy: the product of tool wear and torque, which the dataset documentation uses for its overstrain condition. We call it a load proxy, not strain."])
    A(["Feature engineering", "Did the engineered features help?", f"The wear x torque proxy raised mean cross-validated PR-AUC by {M['feature_engineering']['mean_pr_auc_gain_from_wear_torque']:+.3f}, above our pre-set 0.01 rule, so it is used. Temperature difference and power come from the proposal and are always included."])
    A(["Feature engineering", "How is the machine type encoded?", "It is ordinal-coded first (H=0, L=1, M=2), then one-hot encoded after any resampling, so SMOTENC can treat it as one categorical variable."])
    A(["Feature engineering", "How do you make sure training and prediction use the same transformations?", "All preprocessing and feature engineering are steps of one saved scikit-learn pipeline. The web application calls that same object, so there is no separate re-implementation that could drift."])
    # imbalance
    A(["Class imbalance", "How did you handle the 3.4% failure rate?", "First by evaluating with PR-AUC, precision, recall, F1 and Brier instead of accuracy. We then compared plain training, class weights and SMOTENC by cross-validation. None improved PR-AUC meaningfully, so we use plain training and calibrate the probabilities."])
    A(["Class imbalance", "What is SMOTE / SMOTENC?", "SMOTE creates synthetic minority samples by interpolating between a minority sample and one of its nearest minority neighbours. SMOTENC extends it to datasets with categorical columns by choosing the category of the synthetic sample by majority vote of the neighbours."])
    A(["Class imbalance", "What was wrong with your original SMOTENC use?", "We one-hot encoded the machine type into three columns before SMOTENC, which treated each as an independent categorical column; the three votes could all be 0, producing impossible rows (115 of 7,458 synthetic rows). We fixed it by encoding the type as one ordinal column, resampling, and one-hot encoding afterwards."])
    A(["Class imbalance", "Why must resampling happen inside cross-validation?", "If synthetic samples were created before splitting, copies and near-copies of validation points would appear in training, inflating scores. Inside the pipeline, SMOTENC is fitted on training folds only and is skipped when predicting."])
    # LR
    A(["Logistic Regression", "How does Logistic Regression work?", "It computes a weighted sum of the features, z, and converts it to a probability with the sigmoid p = 1/(1+e^-z). It learns the weights by minimising log loss with an L2 penalty."])
    A(["Logistic Regression", "Why is it the weakest model here?", f"Failure risk in the data is U-shaped (high at both low and high speed, torque and power) and depends on interactions; a single linear boundary cannot represent that. Validation PR-AUC {sel['lr']['cv_pr_auc']:.3f} against {sel['rf']['cv_pr_auc']:.3f} for Random Forest."])
    A(["Logistic Regression", "What does the regularisation parameter C do?", "C is the inverse of the penalty strength. Small C forces small weights (simpler model); large C allows larger weights (more flexible, may over-fit)."])
    A(["Logistic Regression", "Why include a weak model at all?", "As the interpretable baseline from the proposal: it shows how much the non-linear models add and anchors the comparison."])
    # RF
    A(["Random Forest", "How does Random Forest work?", "It trains many decision trees on bootstrap samples of the rows, letting each split consider only a random subset of the features, and averages their predicted probabilities. Averaging many different trees reduces variance."])
    A(["Random Forest", "How does a decision tree decide a split?", "It tries thresholds on features and picks the one that best separates failures from non-failures (lowest Gini impurity), then repeats on each side until a stopping rule is reached."])
    A(["Random Forest", "What are its hyperparameters in your model?", f"n_estimators {M['tuning']['rf']['best_params']['n_estimators']}, max_depth {M['tuning']['rf']['best_params']['max_depth']}, min_samples_leaf {M['tuning']['rf']['best_params']['min_samples_leaf']}, max_features {M['tuning']['rf']['best_params']['max_features']}. More trees give a steadier average, depth and leaf size control over-fitting, max_features controls how different the trees are."])
    A(["Random Forest", "Why does Random Forest suit this dataset?", "The dataset's failure rules are threshold rules and interactions (for example low temperature difference AND low speed), which trees represent naturally without scaling."])
    A(["Random Forest", "Why do you calibrate Random Forest probabilities?", "The average of tree votes is not a true probability. Calibration (here isotonic or sigmoid, whichever has the lower out-of-fold Brier score) maps scores to observed failure rates, which the Warning and Critical thresholds need."])
    # XGB
    A(["XGBoost", "How does XGBoost work?", "It builds trees one after another; each new small tree is fitted to the gradient of the loss of the current ensemble (the direction that reduces the errors still left). Its output, scaled by a small learning rate, is added to the running sum, which a sigmoid turns into a probability."])
    A(["XGBoost", "What is the difference between bagging (Random Forest) and boosting (XGBoost)?", "Bagging trains independent trees on resampled data and averages them to reduce variance. Boosting trains trees sequentially, each correcting the previous ones, to reduce bias; it is more flexible but easier to over-fit."])
    A(["XGBoost", "Why did XGBoost not win?", f"On validation it had lower PR-AUC ({sel['xgb']['cv_pr_auc']:.3f} vs {sel['rf']['cv_pr_auc']:.3f}) and a worse Brier score. On the test set the two are statistically indistinguishable (intervals overlap; only {M['dataset']['test_failures']} test failures). With few positives, boosting can over-fit."])
    A(["XGBoost", "What do learning_rate, max_depth and subsample do?", "learning_rate is the step size of each tree's contribution (small is safer but needs more trees); max_depth limits each tree's complexity; subsample is the fraction of rows used per tree (adds randomness and reduces over-fitting)."])
    # ensemble
    A(["Ensemble learning", "What is a soft voting ensemble?", "It averages the predicted probabilities of several models (optionally weighted); hard voting would instead count their final class votes."])
    A(["Ensemble learning", "Why might an ensemble help, and did it?", f"If members make different errors, averaging cancels some of them. Here it did not help: the equal-weight ensemble's validation PR-AUC is {sel['ens']['cv_pr_auc']:.3f} against {sel['rf']['cv_pr_auc']:.3f} for Random Forest, because Logistic Regression is much weaker and lowers the average."])
    A(["Ensemble learning", "How did you choose the ensemble weights without using the test set?", "From out-of-fold calibrated probabilities on the training data only: equal weights versus a 0.1-step grid. Tuned weights would be used only if they beat equal weights by 0.005 PR-AUC, also gained on a split-half check, and formed a real blend."])
    A(["Ensemble learning", "What happened with the tuned weights?", f"The best weights on validation data were lr 0, rf 1, xgb 0 (gain {e['gain_on_oof']:+.3f}) - the Random Forest alone. That is not an ensemble, so the rule rejected it and equal weights are deployed. Reporting this is more honest than calling Random Forest an ensemble."])
    # metrics
    A(["Evaluation metrics", "Why not use accuracy?", f"Only {eda['failure_rate_pct']}% of machines fail, so predicting 'no failure' every time is {M['dataset']['baseline_accuracy_if_always_no_failure'] * 100:.1f}% accurate. Even our weakest model scores {x('lr')['accuracy'] * 100:.1f}% accuracy while catching only {x('lr')['recall'] * 100:.0f}% of failures."])
    A(["Evaluation metrics", "Explain precision and recall in this context.", "Precision: of machines we flag, how many really fail (low precision = false alarms and wasted inspections). Recall: of machines that really fail, how many we flag (low recall = missed failures and unplanned downtime). In maintenance a missed failure usually costs more, so we favour recall."])
    A(["Evaluation metrics", "What is PR-AUC and why is it the main metric?", "The area under the precision-recall curve. It measures how well failures are ranked above non-failures while ignoring the many easy non-failures, so it is informative for imbalanced data. A random model scores about the failure rate."])
    A(["Evaluation metrics", "What is the Brier score?", "The mean squared difference between predicted probability and the actual 0/1 outcome; lower is better. It tells us whether probabilities are honest, which the thresholds rely on."])
    A(["Evaluation metrics", "Read a confusion matrix.", f"For the default model at 30%: {x(d)['tp']} failures caught (true positives), {x(d)['fn']} missed (false negatives), {x(d)['fp']} false alarms (false positives) and {x(d)['tn']} correct non-failures (true negatives)."])
    A(["Evaluation metrics", "What are your final results?", f"Default model ({lab[d]}) on the held-out test set at the 30% threshold: precision {x(d)['precision']:.3f}, recall {x(d)['recall']:.3f}, F1 {x(d)['f1']:.3f}, PR-AUC {t[d]['pr_auc']:.3f} (95% CI {t[d]['ci95']['pr_auc'][0]:.2f}-{t[d]['ci95']['pr_auc'][1]:.2f}), Brier {t[d]['brier']:.4f}. The default was chosen on validation data, not on these figures."])
    A(["Evaluation metrics", "Why is recall not 100%?", f"The missed failures are mostly tool wear failures (TWF): {t[d]['missed_at_0.3_by_mode']['TWF']} of {t[d]['missed_at_0.3_by_mode']['missed_total']} missed. In this dataset TWF only happens to some machines within a range of tool wear and cannot be told apart from healthy machines in that range."])
    A(["Evaluation metrics", "What is probability calibration?", "Adjusting model scores so that predicted probabilities match observed frequencies: among machines predicted at 30%, about 30% should fail. We compared no calibration, sigmoid and isotonic by out-of-fold Brier score."])
    A(["Evaluation metrics", "How do you avoid choosing models with the test set?", "Every choice (features, imbalance handling, hyperparameters, calibration, ensemble weights, default model) uses cross-validation on the training split. The test set is evaluated once, after the choices are fixed. A unit test recomputes the selected default from the validation table to prove it."])
    A(["Evaluation metrics", "What is cross-validation?", "The training data is split into five folds; each fold is held out in turn while the model trains on the other four. The held-out predictions give an honest performance estimate from training data alone. Stratified folds keep the failure percentage constant."])
    A(["Evaluation metrics", "What are the limitations of your evaluation?", f"Synthetic data; only {M['dataset']['test_failures']} test failures (wide confidence intervals); hyperparameters tuned on the same data used for validation figures (slightly optimistic for all models); one split; no external validation; thresholds from the proposal rather than a cost model."])
    # thresholds / alerts
    A(["Alerts and thresholds", "How are Normal, Warning and Critical defined?", "Failure probability below 30% is Normal, 30% up to 70% is Warning, 70% and above is Critical (lower bound inclusive), as in the project proposal."])
    A(["Alerts and thresholds", "Why those thresholds?", "They come from the proposal. We checked them on validation data: lowering the threshold from 70% to 30% raises recall with modest loss of precision and little change in F1, which suits maintenance. We have no cost model, so they are a reasonable judgement, not an optimum."])
    A(["Alerts and thresholds", "Why are there so few Warning cases?", "In this synthetic data failures are close to deterministic, so the models are decisive and predictions cluster near 0% and 100%. In a real setting more machines would fall in between."])
    A(["Alerts and thresholds", "Describe the alert lifecycle.", "A Warning/Critical reading opens one alert for the machine. More readings update it (occurrences up; a higher level re-opens an acknowledged alert; a lower level is recorded). A user can acknowledge or resolve it; a later Normal reading resolves it automatically. A new problem after resolution opens a new alert. Every step is stored in alert_events."])
    A(["Alerts and thresholds", "How do you stop duplicate alerts?", "In code (apply_alert_lifecycle updates the existing alert) and in the database: a partial unique index allows only one open or acknowledged alert per machine."])
    A(["Alerts and thresholds", "What is the health score?", "round(100 x (1 - failure probability)). It is only the probability on a friendlier 0-100 scale and is not an independently validated measure of machine health."])
    # failure modes
    A(["Failure modes", "Does your model predict the type of failure?", "The main model does not; it is binary. We show 'possible failure modes' from separate per-mode classifiers (only those reliable in validation) next to rule checks of the documented failure conditions. Both are suggestions to investigate, never diagnoses."])
    A(["Failure modes", "Why is TWF not shown as a model estimate?", f"Its classifier's validation PR-AUC is only {M['failure_modes']['TWF']['oof_pr_auc']:.2f} (below our 0.5 gate), so it is not reliable. TWF is only partly deterministic in the data."])
    A(["Failure modes", "What is the difference between a rule check and a model estimate?", "A rule check tests a fixed condition from the dataset documentation (for example temperature difference below 8.6 K and speed below 1380 rpm); it is not machine learning. A model estimate is a probability from a trained classifier."])
    # explainability
    A(["Explainability", "How do you explain a single prediction?", "With exact Shapley values over five sensor groups: each group's average effect on the probability across all orders of switching it from a typical reading to this machine's value. They add up to the difference between this prediction and the typical one."])
    A(["Explainability", "Do the explanations prove what causes a failure?", "No. They describe how the model responds to each reading. The what-if feature is the same: it shows model behaviour, not cause and effect."])
    # backend
    A(["Backend", "How is the model used in the web application?", "At start-up the ModelRegistry loads each saved model once; requests only call predict_proba. Nothing is retrained per request."])
    A(["Backend", "What happens if a model file is corrupted?", "Each file is checked against a sha256 in meta.json and loaded independently. A bad file disables only that model with a reason; the default falls back to the best available model and a banner says so. If nothing loads, the app still starts and reports it."])
    A(["Backend", "How do you protect against CSRF?", "Each form carries a random per-session token; a POST is accepted only if the token is present, non-empty and equal to the session token. We also reject cross-site Origin headers, and the JSON API accepts only application/json. An earlier bug accepted empty tokens because comparing two empty strings is true."])
    A(["Backend", "How do you prevent SQL injection and XSS?", "All values are bound parameters and sortable columns come from a whitelist, so user text never becomes SQL. Jinja escapes output, JavaScript uses textContent, machine IDs follow a strict pattern and a Content-Security-Policy blocks inline scripts."])
    A(["Backend", "How is configuration handled?", "Environment variables: APP_ENV, SECRET_KEY (required in production), DATABASE_PATH, MODELS_DIR and others. Nothing secret or machine-specific is in the source."])
    A(["Backend", "Why Post/Redirect/Get?", "After saving an analysis the server redirects to its result page, so refreshing the browser does not submit the form again and create a duplicate."])
    # database
    A(["Database", "Describe your database design.", "machines (identity), predictions (immutable snapshots), alerts (one active per machine) and alert_events (append-only history), with foreign keys, CHECK constraints and indexes."])
    A(["Database", "How do you keep a machine's identity consistent?", "Machine ID is required and unique case-insensitively in the machines table; every prediction references it. The machine's type is part of its identity and cannot change; a conflicting type is refused."])
    A(["Database", "How is history protected from change?", "Database triggers abort any UPDATE or DELETE on predictions and on alert_events. Each prediction also stores its own explanation snapshot and model version, so retraining never alters old results."])
    A(["Database", "How did you migrate existing data?", "A versioned schema (PRAGMA user_version). A v1 database is backed up, then migrated in one transaction: ids and values preserved, machines rebuilt from the text IDs, alerts rebuilt by replaying the old rows through the new lifecycle, original table kept. If anything fails the transaction rolls back."])
    A(["Database", "Why SQLite rather than MySQL?", "It needs no server, ships with Python and supports the constraints, triggers, indexes and transactions we need. For many concurrent users a server database would be better."])
    # architecture
    A(["Architecture", "Walk through the flow from input to dashboard.", "Input (form, CSV or API) -> validation -> machine identity -> preprocessing and feature engineering (inside the pipeline) -> calibrated model probability -> risk class and health score -> indicators, mode estimates, explanation -> recommendation -> one database transaction (machine, prediction, alert) -> dashboard, machines, alerts, history and export read the stored data."])
    A(["Architecture", "Why separate the service layer from the routes?", "So the form, the JSON API and the CSV upload share one tested workflow, and routes stay thin."])
    A(["Architecture", "Why do you store the explanation with each prediction?", "So the result page and history are exactly what was shown at the time, regardless of later model changes."])
    # baselines (added 2026-11-24)
    A(["Evaluation metrics", "Does the model beat a simple baseline?", f"We compare with two baselines on the same test set. Predicting 'no failure' always is {M['baselines']['never_fails']['accuracy'] * 100:.1f}% accurate but finds nothing. Rules alone: all documented rules give recall {M['baselines']['rule_based']['recall']:.2f} but precision {M['baselines']['rule_based']['precision']:.2f}; the physical rules (HDF, PWF, OSF) give precision {M['baselines']['rule_physical']['precision']:.2f}, recall {M['baselines']['rule_physical']['recall']:.2f}, F1 {M['baselines']['rule_physical']['f1']:.3f}. The default model has F1 {x(meta['default_model'])['f1']:.3f} at 30%. So on this synthetic data a rules-only baseline is as good as, or slightly better than, the model. The dataset's failures are produced by those rules, so this is expected and we report it openly. What the model adds is a calibrated probability and a graded status."])
    import json as _json, os as _os
    tw = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reports", "twf_study.json")))
    A(["Failure types", "Why can't the model predict tool-wear failures, and what did you do about it?", f"In this dataset a tool fails from wear at a random point once it is worn, so the readings only say that risk rises above about 200 min. Only about 1 in 20 tools at that wear level fails from wear. We tested simple wear-based estimates against the shipped model with repeated cross-validation on the training split: {tw['verdict']} So instead of claiming a better model we added a separate maintenance notice (tool wear of 200 min or more), which would have flagged {M['evaluation']['missed']['with_tool_wear_notice']} of the {M['evaluation']['missed']['total']} failures the model missed."])
    A(["Evaluation metrics", "Why not lower the 30% threshold to catch more failures?", "Lowering it raises recall but floods the user with false alarms, and the thresholds should be judged on validation data (cross-validation on the training split), not on the test set. The per-type threshold table on Model Insights shows this trade-off. The tool-wear problem is better handled by a separate maintenance notice than by alarming on every worn tool."])
    A(["Evaluation metrics", "How was the test set used, and could it have leaked?", "Model, feature, threshold and calibration choices used cross-validation on the training split only. The test set was used once to report results; baselines, per-type results and the missed-failure list read it afterwards descriptively, and no model or threshold was changed because of them. Any future change should be judged by repeated cross-validation (see ml/twf_study.py) with the test set looked at once at the end. There is no second dataset to confirm results on."])
    # limitations
    A(["Limitations and future scope", "What are the main limitations?", "Synthetic data and no external validation; few test failures; manual input only; probability is not time to failure; failure type not diagnosed; single-user without accounts; not tested on Windows or under heavy load."])
    A(["Limitations and future scope", "What would you do with more time or real data?", "Validate on real, time-ordered sensor data; add remaining-useful-life modelling; add drift monitoring and periodic retraining; tune thresholds with a maintenance cost model; connect live sensors; add authentication and a production deployment."])
    A(["Limitations and future scope", "Why did you not implement live monitoring or remaining useful life?", "The dataset has no time dimension and inputs are manual, so those features would be unsupported claims."])
    return q


def pending(M, meta, T):
    return [
        ["Unverified", "Loading the models on the user's Windows machine (Python 3.14)", "OPEN - needs user confirmation", "XGBoost is now stored in JSON form and the application tolerates a failed model, but the original failure could not be reproduced on Linux, so the cause is inferred. Verified on Linux with Python 3.12 and 3.14 only.", "High"],
        ["Unverified", "Visual and interaction checks outside Chromium", "OPEN", "Pages were checked in Chromium at 1440 px and 390 px (emulated). Other browsers, real phones and a formal accessibility audit were not done.", "Medium"],
        ["Unverified", "Performance under load / concurrent users", "OPEN", "Not load-tested. SQLite allows one writer at a time; the Flask development server is single-process.", "Low"],
        ["Methodology", "Hyperparameter tuning is not nested", "KNOWN LIMITATION", "Validation figures are slightly optimistic for all models alike; the test set is unaffected. Nested cross-validation would remove this.", "Low"],
        ["Methodology", "Only 68 test failures", "KNOWN LIMITATION", "Wide confidence intervals; model differences are within noise.", "Medium"],
        ["Methodology", "A rules-only baseline matches the trained model", "KNOWN LIMITATION", f"Physical rules (HDF, PWF, OSF) reach F1 {M['baselines']['rule_physical']['f1']:.3f} on the test set versus {M['test'][meta['default_model']]['thresholds']['0.3']['f1']:.3f} for the default model, because the synthetic dataset's failures are generated by those rules. A real dataset (without known rules) is needed to show what machine learning adds.", "High"],
        ["Methodology", "Synthetic dataset, no external validation", "KNOWN LIMITATION", "Real-world performance is unknown.", "High"],
        ["Methodology", "Thresholds not cost-optimised", "KNOWN LIMITATION", "30% / 70% come from the proposal; no maintenance cost data.", "Low"],
        ["Methodology", "SMOTENC cannot create type H samples", "KNOWN LIMITATION", "H is rare; irrelevant to deployed models (no resampling used).", "Low"],
        ["Methodology", "TWF classifier unreliable", "BY DESIGN", f"Validation PR-AUC {M['failure_modes']['TWF']['oof_pr_auc']:.2f}; not shown to users.", "-"],
        ["Features not implemented", "Live IoT monitoring", "NOT IMPLEMENTED (by decision)", "Inputs are manual; no ingestion layer.", "-"],
        ["Features not implemented", "Remaining useful life / time to failure", "NOT IMPLEMENTED (by decision)", "No time dimension in the data.", "-"],
        ["Features not implemented", "User accounts, roles, audit of who acknowledged an alert", "NOT IMPLEMENTED", "Single-user prototype; notes are free text.", "Low"],
        ["Features not implemented", "Per-reading explanations for CSV batch rows", "NOT IMPLEMENTED (by decision)", "Keeps large uploads fast; the result page says so.", "Low"],
        ["Features not implemented", "Celsius input", "NOT IMPLEMENTED", "Inputs are in kelvin as in the dataset; a hint explains the conversion.", "Low"],
        ["Technical debt", "Timestamps are server local time without a time zone", "OPEN", "Fine for a single site; should be UTC for multi-site use.", "Low"],
        ["Technical debt", "Development server and no log rotation or backup schedule", "OPEN", "Documented: use a production WSGI server for deployment.", "Low"],
        ["Technical debt", "Old database table legacy_predictions_v1 is kept after migration", "AWAITING APPROVAL", "Kept for safety so no data can be lost. It can be dropped once the migrated data has been checked by the project team.", "Low"],
        ["Tests", "Tests not yet passed", "NONE", f"Latest run ({D2}): {T['passed']} of {T['total']} passed on Python 3.13 (Linux). The first {T0} also passed on Python 3.12 and 3.14 (2026-11-18).", "-"],
        ["Future improvements", "Nested cross-validation; drift monitoring; scheduled retraining; threshold tuning with a cost model; real-data validation", "PROPOSED", "-", "-"],
        ["Decisions awaiting approval", "Drop legacy_predictions_v1 after verification; adopt a production deployment target", "AWAITING APPROVAL", "-", "-"],
    ]
