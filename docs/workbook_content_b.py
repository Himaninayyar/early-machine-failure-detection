"""Narrative content for workbook sheets 5-7. Metrics come from models/metrics.json and models/meta.json."""


def f3(v):
    return f"{v:.3f}"


def f4(v):
    return f"{v:.4f}"


def pc(v):
    return f"{v * 100:.1f}%"


STRAT = {"none": "None", "class_weight": "Class weights", "smote": "SMOTENC"}


def evaluation(M, meta, eda, T):
    lab = meta["model_names"]
    order = ["lr", "rf", "xgb", "ens"]
    sel = {r["model"]: r for r in M["selection"]["table"]}
    te = M["test"]
    d = meta["default_model"]
    e = M["ensemble"]
    blocks = [
        ("heading", "Status of the numbers on this sheet"),
        ("table", ["Item", "Status"], [
            ["Results recorded here", f"FINAL. Produced by the full training profile, model version {meta['version']}, trained {meta['trained_at']}. They are read from models/metrics.json by docs/build_workbook.py, never typed by hand."],
            ["Preliminary results", "The 'tiny' and 'quick' training profiles exist only for the automated tests and smoke tests. Their numbers are meaningless and are not recorded here. The application labels any such run as preliminary."],
            ["Superseded (version 1) results", "Before the audit the project reported test PR-AUC of 0.451 (Logistic Regression), 0.900 (Random Forest), 0.901 (XGBoost) with Random Forest as default. That run used a flawed SMOTENC order and a feature called 'strain' and had no ensemble. It is superseded; the corrected pipeline's results are below (Random Forest again selected, from validation evidence this time)."],
        ]),
        ("heading", "Training configuration"),
        ("table", ["Setting", "Value"], [
            ["Random seed", str(meta["seed"]) + " (used for the split, cross-validation folds, model seeds and bootstrap)"],
            ["Split", f"Stratified 80/20: {M['dataset']['train_rows']} training / {M['dataset']['test_rows']} test machines; failures {M['dataset']['train_failures']} / {M['dataset']['test_failures']}"],
            ["Feature set and imbalance comparison", "Repeated stratified 5-fold cross-validation, 2 repeats (10 fits per configuration), 18 configurations (3 models x 2 feature sets x 3 imbalance strategies)"],
            ["Hyperparameter search", f"Randomized search, scoring = PR-AUC (average precision), 5-fold CV; settings tried: Logistic Regression {M['tuning']['lr']['n_iter']}, Random Forest {M['tuning']['rf']['n_iter']}, XGBoost {M['tuning']['xgb']['n_iter']}"],
            ["Calibration", "Out-of-fold comparison of none / sigmoid / isotonic (5-fold outer, 5-fold inner); lowest Brier score among the two calibrators kept"],
            ["Ensemble weights", "Equal weights versus a 0.1-step grid on out-of-fold probabilities, with split-half check"],
            ["Bootstrap confidence intervals", "1000 resamples of the test set (95% percentile interval)"],
            ["Permutation importance", "10 repeats on the test set, scoring PR-AUC"],
            ["Libraries", ", ".join(f"{k} {v}" for k, v in meta["libraries"].items())],
            ["Data checksum (sha256, first 16)", meta["data_sha256"][:16]],
        ]),
        ("heading", "Evaluation metrics explained"),
        ("table", ["Metric", "Plain-language meaning", "Why it matters here"], [
            ["Accuracy", "Share of all machines classified correctly.", f"Misleading: always predicting 'no failure' is {pc(M['dataset']['baseline_accuracy_if_always_no_failure'])} accurate. Reported, never used to choose a model."],
            ["Precision", "Of the machines the model flags as failing, the share that really fail.", "Low precision means false alarms and wasted inspections."],
            ["Recall (sensitivity)", "Of the machines that really fail, the share the model flags.", "A missed failure means unplanned downtime, so recall is the metric to protect."],
            ["F1-score", "Harmonic mean of precision and recall.", "One number balancing the two."],
            ["PR-AUC (average precision)", "Area under the precision-recall curve: how well failures are ranked above non-failures, focusing on the rare class. A random model scores about the failure rate (" + f"{M['dataset']['test_failures'] / M['dataset']['test_rows']:.3f}" + ").", "The primary selection metric because it ignores the huge number of easy non-failures."],
            ["ROC-AUC", "Probability that a random failure is ranked above a random non-failure.", "Looks flattering on imbalanced data; reported together with PR-AUC."],
            ["Brier score", "Mean squared difference between predicted probability and the 0/1 outcome (lower is better).", "Checks that probabilities are honest, which the 30% and 70% thresholds rely on. Constant prediction of the base rate scores about 0.033."],
            ["Confusion matrix", "Counts of true negatives, false positives (false alarms), false negatives (missed failures) and true positives at a chosen threshold.", "Shows exactly what kind of mistakes a model makes."],
        ]),
        ("heading", "RESULTS A. Model selection evidence (training data only, out-of-fold, calibrated)"),
        ("table", ["Model", "PR-AUC", "ROC-AUC", "Brier", "Best F1 (any threshold)", "Recall at 30%", "Precision at 30%", "Selected"],
         [[lab[k], f3(sel[k]["cv_pr_auc"]), f3(sel[k]["cv_roc_auc"]), f4(sel[k]["cv_brier"]), f3(sel[k]["cv_best_f1"]), f3(sel[k]["cv_recall_at_0.3"]), f3(sel[k]["cv_precision_at_0.3"]), "YES (default)" if k == d else ""] for k in order]),
        ("note", "Rule: highest validation PR-AUC; models within 0.01 of the best are separated by Brier score, then by simplicity. The test set is not used. Hyperparameters were tuned on the same training data, so validation figures are slightly optimistic for every model alike."),
        ("heading", "RESULTS B. Held-out test set (reported once)"),
        ("table", ["Model", "Threshold", "Precision", "Recall", "F1", "Accuracy", "True neg.", "False pos.", "False neg.", "True pos."],
         [[lab[k], th, f3(te[k]["thresholds"][th]["precision"]), f3(te[k]["thresholds"][th]["recall"]), f3(te[k]["thresholds"][th]["f1"]), f3(te[k]["thresholds"][th]["accuracy"]),
           te[k]["thresholds"][th]["tn"], te[k]["thresholds"][th]["fp"], te[k]["thresholds"][th]["fn"], te[k]["thresholds"][th]["tp"]] for th in ("0.3", "0.5", "0.7") for k in order]),
        ("table", ["Model", "PR-AUC (95% CI)", "ROC-AUC (95% CI)", "Brier", "Brier before calibration", "Recall at 30% (95% CI)", "Precision at 30% (95% CI)"],
         [[lab[k], f"{f3(te[k]['pr_auc'])} ({te[k]['ci95']['pr_auc'][0]:.2f}-{te[k]['ci95']['pr_auc'][1]:.2f})",
           f"{f3(te[k]['roc_auc'])} ({te[k]['ci95']['roc_auc'][0]:.3f}-{te[k]['ci95']['roc_auc'][1]:.3f})", f4(te[k]["brier"]), f4(te[k]["uncalibrated"]["brier"]) if te[k]["uncalibrated"] else "-",
           f"{f3(te[k]['thresholds']['0.3']['recall'])} ({te[k]['ci95']['recall_at_0.3'][0]:.2f}-{te[k]['ci95']['recall_at_0.3'][1]:.2f})",
           f"{f3(te[k]['thresholds']['0.3']['precision'])} ({te[k]['ci95']['precision_at_0.3'][0]:.2f}-{te[k]['ci95']['precision_at_0.3'][1]:.2f})"] for k in order]),
        ("note", f"Interpretation: Random Forest and XGBoost are statistically indistinguishable on the test set (overlapping intervals; only {M['dataset']['test_failures']} test failures). Logistic Regression is far weaker. The equal-weight ensemble is not better than Random Forest. Accuracy of the weakest model ({pc(te['lr']['thresholds']['0.3']['accuracy'])}) is almost the same as always predicting 'no failure' ({pc(M['dataset']['baseline_accuracy_if_always_no_failure'])}), which is why accuracy is not used."),
        ("heading", "RESULTS C. Class-imbalance strategy and feature set (cross-validation on training data, PR-AUC mean +/- std)"),
        ("table", ["Model", "Features", "Imbalance handling", "PR-AUC", "ROC-AUC", "Best F1"],
         [[lab[r["model"]], "With wear x torque" if r["use_wear_torque"] else "Standard", STRAT[r["strategy"]], f"{r['ap']:.3f} +/- {r['ap_std']:.3f}", f3(r["roc"]), f3(r["best_f1"])] for r in sorted(M["ablation"], key=lambda r: (["lr", "rf", "xgb"].index(r["model"]), r["use_wear_torque"]))]),
        ("note", f"Wear x torque proxy: mean PR-AUC gain {M['feature_engineering']['mean_pr_auc_gain_from_wear_torque']:+.4f} (adopted because > +0.01, a rule fixed beforehand). No class-imbalance method beat 'none' for any model by more than the 0.005 tolerance, so none is used. SMOTENC results are from the CORRECTED pipeline (machine type as one categorical column, one-hot after resampling)."),
        ("heading", "RESULTS D. Hyperparameter tuning"),
        ("table", ["Model", "Imbalance handling", "Settings tried", "Best CV PR-AUC", "Chosen settings"],
         [[lab[k], STRAT[v["strategy"]], v["n_iter"], f4(v["best_cv_pr_auc"]), ", ".join(f"{a}={b}" for a, b in v["best_params"].items())] for k, v in M["tuning"].items()]),
        ("heading", "RESULTS E. Probability calibration (out-of-fold, training data)"),
        ("table", ["Model", "Calibration", "PR-AUC", "Brier", "Log loss", "Used"],
         [[lab[k], o["method"], f3(o["pr_auc"]), f4(o["brier"]), f4(o["log_loss"]), "YES" if o["method"] == c["chosen"] else ""] for k, c in M["calibration"].items() for o in c["options"]]),
        ("heading", "RESULTS F. Soft voting ensemble experiment"),
        ("table", ["Variant", "Weights (lr, rf, xgb)", "Validation PR-AUC", "Validation Brier", "Used?"],
         [["Equal weights", ", ".join(f"{w:.2f}" for w in e["equal"]["weights"]), f3(e["equal"]["oof_pr_auc"]), f4(e["equal"]["oof_brier"]), "YES: this is the deployed ensemble"],
          ["Best weights found on validation data", ", ".join(f"{w:.1f}" for w in e["best_on_oof"]["weights"]), f3(e["best_on_oof"]["oof_pr_auc"]), f4(e["best_on_oof"]["oof_brier"]),
           "NO: degenerate (all weight on one model = Random Forest alone)" if e["tuned_weights_degenerate"] else ("YES" if e["tuned_weights_adopted"] else "NO")]] +
         [[f"Weight-grid rank {i + 1}", ", ".join(f"{w:.1f}" for w in r["weights"]), f3(r["oof_pr_auc"]), f4(r["oof_brier"]), ""] for i, r in enumerate(e["top_weight_sets"])]),
        ("note", f"Split-half check gain: {e['split_half_gain']:+.4f}; gain on validation: {e['gain_on_oof']:+.4f}. Rule: {e['rule']} Test-set figures for the tuned weights (reported only, never used for a decision): PR-AUC {f3(e['tuned_variant_test']['pr_auc'])}, recall at 30% {f3(e['tuned_variant_test']['recall_at_0.3'])}."),
        ("heading", f"RESULTS G. Threshold evidence for {lab[d]} (out-of-fold, training data)"),
        ("table", ["Flag when probability >=", "Precision", "Recall", "F1", "Machines flagged (of 8000)"],
         [[f"{r['threshold']:.0%}", f3(r["precision"]), f3(r["recall"]), f3(r["f1"]), r["flagged"]] for r in M["thresholds_oof"][d]]),
        ("note", "The 30% / 70% thresholds come from the project proposal. The table shows the trade-off: going from 70% down to 30% catches slightly more failures (higher recall) at the cost of more false alarms (lower precision); F1 changes little. A low Warning threshold favours recall, which suits maintenance where a missed failure usually costs more than an extra inspection. No cost model was available, so the thresholds are a judgement, not an optimum."),
        ("heading", f"RESULTS H. Do the alert bands mean what they say? ({lab[d]}, test set)"),
        ("table", ["Band", "Machines", "Actually failed", "Observed failure rate"],
         [[b["level"], b["machines"], b["failures"], "-" if b["observed_failure_rate"] is None else pc(b["observed_failure_rate"])] for b in te[d]["bands"]]),
        ("note", "Failure rate rises from Normal to Critical, as it should. The Warning band holds very few machines because failures in this synthetic data are close to deterministic, so predictions cluster near 0% or 100%."),
        ("heading", "RESULTS I. Which failures does each model miss at the 30% threshold? (test set)"),
        ("table", ["Model", "Missed", "TWF", "HDF", "PWF", "OSF", "RNF", "No mode recorded"],
         [[lab[k], te[k]["missed_at_0.3_by_mode"]["missed_total"]] + [te[k]["missed_at_0.3_by_mode"][c] for c in ("TWF", "HDF", "PWF", "OSF", "RNF")] + [te[k]["missed_at_0.3_by_mode"]["no_mode_flag"]] for k in order]),
        ("note", "Most misses are tool wear failures (TWF). In this dataset TWF happens only to some machines within a range of tool wear and cannot be told apart from healthy machines in that range from the sensor readings, so this is the main reason recall stays below 100%. A failure can carry more than one mode, so rows can add up to more than the number missed."),
        ("heading", "RESULTS J. Failure-mode classifiers (one calibrated Random Forest per mode)"),
        ("table", ["Mode", "Failures in training", "Failures in test", "Validation PR-AUC", "Test PR-AUC", "Shown to users"],
         [[c, v["positives_train"], v["positives_test"], f3(v["oof_pr_auc"]), "-" if v.get("test_pr_auc") is None else f3(v["test_pr_auc"]), "Yes" if c in meta["failure_modes_reliable"] else "No (validation PR-AUC below 0.5)"] for c, v in M["failure_modes"].items()]),
        ("note", "HDF, PWF and OSF are generated by fixed rules in this dataset, so near-perfect scores say little about real machines. TWF's classifier is not reliable and is not shown. RNF is random by design and is not modelled."),
        ("heading", "FINAL SELECTED MODEL AND REASONING"),
        ("table", ["Item", "Decision and reasoning"], [
            ["Default model", f"{lab[d]} (version {meta['version']})."],
            ["Why", f"It had the highest cross-validated, calibrated PR-AUC on the training data ({f3(sel[d]['cv_pr_auc'])} versus " + ", ".join(f"{lab[k]} {f3(sel[k]['cv_pr_auc'])}" for k in order if k != d) + f") and the lowest Brier score ({f4(sel[d]['cv_brier'])}). No other model was within 0.01 PR-AUC, so no tie-break was needed. The decision uses no test-set information. PR-AUC and recall/precision were considered, not accuracy."],
            ["Ensemble", "Evaluated honestly and not superior. Available as a selectable model but not the default."],
            ["Fallback", "If the default cannot be loaded, the best available model by validation ranking (" + ", ".join(lab[k] for k in meta["ranking"]) + ") is used and a banner says so."],
        ]),
        ("heading", "Limitations of the evaluation methodology"),
        ("table", ["Limitation", "Effect"], [
            ["Synthetic dataset", "Scores do not show real-world performance; failure rules in the data are simple, so tree models do very well."],
            ["Only 68 test failures", "Wide confidence intervals; differences between Random Forest and XGBoost are within noise."],
            ["Tuning is not nested", "Hyperparameters were tuned with cross-validation on the same training data that is then used for validation figures, so validation figures are slightly optimistic for all models alike. The test set is unaffected."],
            ["Ensemble weights selected on out-of-fold data", "A split-half check guards against over-fitting; the weights were not adopted anyway."],
            ["One train/test split", "A different split would give somewhat different test figures."],
            ["No external validation", "No data from another source or time period was available."],
            ["Thresholds not cost-optimised", "30% / 70% come from the proposal; no maintenance cost model exists."],
            ["SMOTENC never creates type H samples", "H is a rare category, so a nearest-neighbour majority vote never picks it. Irrelevant for the deployed models (they do not use resampling)."],
        ]),
    ]
    return blocks


def backend(M, meta, eda, T):
    return [
        ("heading", "Flask application structure"),
        ("table", ["File", "Responsibility"], [
            ["app/__init__.py", "Application factory: reads configuration from environment variables, initialises the database (reports failure instead of crashing), loads models, CSRF and Origin checks, security headers, template filters, error handlers."],
            ["app/routes.py", "Thin route functions for pages and the JSON API; no business logic."],
            ["app/service.py", "analyze_and_store (form/API), analyze_batch (CSV), what_if. Raises ValidationFailed / ServiceUnavailable with user-readable messages."],
            ["app/predictor.py", "ModelRegistry: independent model loading, checksum check, default and fallback, predictions, Shapley explanations, failure-mode estimates."],
            ["app/risk.py", "Thresholds, health score, rule-based indicators, mode rows, suggested actions."],
            ["app/validation.py", "Validation rules shared by form, CSV and JSON."],
            ["app/db.py", "Schema (version 2), machine identity, alert lifecycle, all queries."],
            ["app/migrations.py", "Create or migrate the database; backup; dry run; command-line interface."],
        ]),
        ("heading", "Routes and APIs"),
        ("table", ["Route", "Method", "Purpose", "Function"], [
            ["/", "GET", "Dashboard", "routes.dashboard"],
            ["/analyze", "GET, POST", "Analyze form; POST validates, predicts, stores and redirects to the result", "routes.analyze"],
            ["/predictions/<id>", "GET", "Stored result page", "routes.result"],
            ["/analyze/batch, /analyze/batch/template.csv", "GET, POST / GET", "CSV batch upload and template", "routes.batch, routes.batch_template"],
            ["/machines, /machines/<code>", "GET", "Machine list; machine detail with trend", "routes.machines, routes.machine_detail"],
            ["/alerts, /alerts/<id>", "GET", "Alert list (active / resolved / all); alert detail with history", "routes.alerts, routes.alert_detail"],
            ["/alerts/<id>/acknowledge, /alerts/<id>/resolve", "POST", "Alert actions (CSRF token required)", "routes.alert_acknowledge, routes.alert_resolve"],
            ["/history, /history/export.csv", "GET", "History with filters, sorting, pagination; CSV export", "routes.history, routes.export_csv"],
            ["/insights, /about", "GET", "Model Insights; project information", "routes.insights, routes.about"],
            ["/api/predict", "POST (JSON)", "Same workflow as the form; saves the analysis", "routes.api_predict"],
            ["/api/what-if", "POST (JSON)", "Re-run a saved analysis' model with changed readings; saves nothing", "routes.api_what_if"],
            ["/api/health", "GET", "Model, failure-mode and database status (200 or 503)", "routes.api_health"],
            ["/predict, /models", "GET", "Old URLs, redirect (301) to /analyze and /insights", "routes.legacy_predict, routes.legacy_models"],
        ]),
        ("heading", "Request / response flow for POST /analyze"),
        ("table", ["Step", "What happens"], [
            [1, "Browser posts the form with the CSRF token. before_request checks the token (must be present, non-empty and equal to the session token) and the Origin header."],
            [2, "routes.analyze calls service.analyze_and_store(registry, connection, form)."],
            [3, "Service checks a model is available, then validation.validate. On errors it raises ValidationFailed; the route re-renders the form with HTTP 422 and the user's input kept. Nothing is saved."],
            [4, "registry.analyze runs the model, the risk logic, the rule indicators, the failure-mode classifiers and the Shapley explanation."],
            [5, "db.record_prediction opens a transaction: get or create the machine, insert the prediction snapshot, apply the alert lifecycle, commit. Any error rolls everything back."],
            [6, "The route redirects (HTTP 303-style Post/Redirect/Get) to /predictions/<id>, so refreshing never creates a duplicate."],
            [7, "The result page renders the stored row; nothing is recomputed."],
        ]),
        ("heading", "Database tables (schema version 2)"),
        ("table", ["Table", "Columns", "Notes"], [
            ["machines", "id, machine_code (unique, case-insensitive), machine_type (L/M/H), created_at", "One row per physical machine. Code and type cannot be changed (trigger)."],
            ["predictions", "id, machine_id (FK), created_at, source (form/api/batch/migrated), machine_type, air_temp, process_temp, rpm, torque, tool_wear, probability, risk_level, severity, health_score, model_key, model_label, model_version, recommendation, details_json", "IMMUTABLE: triggers forbid UPDATE and DELETE. details_json holds the indicators, Shapley contributions and failure-mode estimates as computed at the time."],
            ["alerts", "id, machine_id (FK), status (open/acknowledged/resolved), opened_at, opening_prediction_id (FK), latest_prediction_id (FK), current_level, peak_level, occurrences, reason, recommendation, acknowledged_at, resolved_at, resolution (manual / auto_normal_reading)", "At most one ACTIVE (open or acknowledged) alert per machine: partial unique index uq_one_active_alert_per_machine."],
            ["alert_events", "id, alert_id (FK), created_at, event_type, prediction_id (FK), note", "Append-only history: opened, repeat, escalated, downgraded, acknowledged, resolved, auto_resolved."],
        ]),
        ("heading", "Relationships"),
        ("table", ["Relationship", "Meaning"], [
            ["machines 1 - N predictions", "Every analysis belongs to exactly one machine (foreign key enforced)."],
            ["machines 1 - N alerts", "A machine has at most one active alert, plus any number of resolved ones."],
            ["alerts 1 - N alert_events", "Each alert keeps its own timeline."],
            ["alerts -> predictions", "An alert points to the prediction that opened it and the latest one that updated it."],
        ]),
        ("heading", "Machine identification"),
        ("table", ["Topic", "Explanation"], [
            ["Before", "A blank machine ID produced a NEW 'AUTO-n' machine for every prediction, and the machine ID was just text repeated on each row."],
            ["Now", "Machine ID is required. The same ID (any letter case) is always the same machine row. The type belongs to the machine: submitting a different type for an existing machine is refused with a message. The Analyze page suggests known machines and fixes the type for them."],
        ]),
        ("heading", "Prediction storage"),
        ("table", ["Topic", "Explanation"], [
            ["What is stored", "Machine, readings, probability, risk level, severity, health score, model key/name/version, recommendation, timestamp and a JSON snapshot of the explanation, indicators and failure-mode estimates."],
            ["Immutability", "Triggers raise an error on any UPDATE or DELETE of predictions, so history cannot change when models are retrained. The result page shows the snapshot, not a recomputation."],
            ["Atomicity", "Machine creation, prediction insert and alert changes are one transaction."],
        ]),
        ("heading", "Alert creation, acknowledgement and resolution"),
        ("table", ["Situation", "What happens", "Event recorded"], [
            ["Warning/Critical reading, machine has no active alert", "A new alert is opened (status open).", "opened"],
            ["Warning/Critical reading at the same level, active alert exists", "Same alert; occurrences + 1. No duplicate alert.", "repeat"],
            ["Reading at a higher level (Warning -> Critical)", "Same alert; level raised; an acknowledged alert becomes open again.", "escalated"],
            ["Reading at a lower level", "Same alert; current level lowered; peak level kept.", "downgraded"],
            ["Normal reading, active alert exists", "Alert resolved automatically (resolution = auto_normal_reading).", "auto_resolved"],
            ["User clicks Acknowledge (open alert)", "Status acknowledged; time recorded; optional note.", "acknowledged"],
            ["User clicks Resolve (open or acknowledged)", "Status resolved (resolution = manual); optional note.", "resolved"],
            ["Machine reaches Warning/Critical after a resolved alert", "A NEW alert; the old one remains as history.", "opened"],
        ]),
        ("heading", "Error handling"),
        ("table", ["Situation", "Behaviour"], [
            ["Missing or invalid input", "HTTP 422; field-level messages; user's input kept; nothing saved. JSON API returns an errors object."],
            ["Model file missing / corrupted / incompatible", "Only that model is unavailable, with a reason on Model Insights and /api/health; the default falls back to the best available model and a banner says so."],
            ["No model available", "Application starts; banner; analysis returns HTTP 503 with an explanation; /api/health returns 503."],
            ["Database cannot be opened or errors", "Application starts; database pages return 503 with an explanation; the error is logged; transactions roll back."],
            ["Machine type conflict", "HTTP 422 naming the registered type."],
            ["Upload too large / bad encoding / too many rows / missing columns", "413 or 422 with a clear message; rejected rows reported with line numbers."],
            ["Unexpected exception", "Logged; generic message (HTML or JSON); no stack trace shown to the user."],
        ]),
        ("heading", "Security mechanisms"),
        ("table", ["Mechanism", "How it works", "Where"], [
            ["CSRF protection", "Per-session random token in every form; a POST is accepted only if the token is present, non-empty and equal to the session token (the earlier code accepted an empty token when no session existed). JSON API accepts only application/json; cross-site Origin is rejected.", "app/__init__.py: protect"],
            ["Secrets", "SECRET_KEY from the environment; required when APP_ENV=production; random per process in development; no secret in source.", "app/__init__.py"],
            ["Session cookie", "HttpOnly, SameSite=Lax, Secure in production.", "app/__init__.py"],
            ["Security headers", "Content-Security-Policy (scripts only from the application itself), X-Content-Type-Options, X-Frame-Options, Referrer-Policy.", "app/__init__.py: security_headers"],
            ["SQL injection", "All values are bound parameters; sortable columns come from a fixed whitelist; no user text becomes SQL.", "app/db.py"],
            ["XSS", "Jinja auto-escaping; JavaScript writes user-visible text with textContent; machine IDs restricted by a pattern; no inline scripts.", "templates, static/js/main.js"],
            ["Input validation", "Server-side for form, CSV and JSON; upload size and row limits.", "app/validation.py, app/service.py"],
            ["Data integrity", "Foreign keys, CHECK constraints, partial unique index, immutability triggers.", "app/db.py"],
        ]),
    ]


def features(M, meta, eda, T):
    H = ["Feature", "What the user sees", "What it does", "How it works internally", "Frontend files", "Backend files", "Status", "Verification"]
    R = [
        ["Dashboard", "Machines tracked, machines needing attention, active alerts, average health, status bar, attention table, recent analyses. A welcome guide when empty.", "Summarises stored data only.", "db.dashboard_stats computes every number from stored rows (machine status = latest analysis).", "templates/dashboard.html", "routes.dashboard, db.dashboard_stats", "Modified (redesigned)", "tests: test_dashboard_numbers_match_stored_data, test_first_time_user_sees_meaningful_empty_states_and_no_fake_numbers; manual browser check"],
        ["Analyze Machine form", "Machine ID, type, five readings with units, explanation, typical and accepted ranges; example buttons; advanced model choice (collapsed).", "Validates, predicts, stores and shows the result.", "Posts to /analyze; service.analyze_and_store.", "templates/analyze.html, static/js/main.js", "routes.analyze, service, validation", "Modified", "tests: test_analyze_form_to_result_page, test_invalid_input_is_explained_and_nothing_is_saved; manual browser check"],
        ["Example readings", "Four buttons that fill the form with real held-out dataset records.", "Lets a new user try the system.", "JSON embedded in the page; main.js fills fields with setters.", "analyze.html, main.js", "routes.EXAMPLES", "Modified", "manual browser check"],
        ["Result page", "Probability, status, health score, probability bar, recommended action, change since last analysis, active alert link; expandable technical details.", "Presents the stored result.", "routes.result renders the stored snapshot.", "templates/result.html", "routes.result, db.get_prediction", "Modified", "tests: test_analyze_form_to_result_page, test_previous_analysis_comparison_on_result_page; manual browser check"],
        ["What drove this result", "Bars showing each reading's contribution in percentage points.", "Explains the prediction.", "Exact Shapley values over five sensor groups against typical readings.", "result.html", "predictor.explain", "Modified (replaces the older 'typical value' sensitivity)", "tests: test_shapley_contributions_add_up_exactly"],
        ["Possible failure modes", "Table: mode, model estimate, rule check, suggested action; notes that these are suggestions.", "Gives specific things to inspect without claiming a diagnosis.", "Separate classifiers (reliable modes only) plus rule checks, shown side by side.", "result.html", "risk.failure_mode_rows, predictor._mode_estimates", "Added", "tests: test_failure_mode_rows_separate_model_estimate_from_rule_check, test_failure_mode_models_give_probabilities"],
        ["Health score", "0-100 number and bar with the probability.", "A friendlier restatement of the probability.", "round(100 x (1 - p)); documented as not independently validated.", "_macros.html, result.html", "risk.health_score", "Added", "tests: test_health_score_formula"],
        ["What-if analysis", "Editable readings and a Calculate button; shows new probability, status, change.", "Explores how the model responds; saves nothing; warns it is not causal.", "fetch() to /api/what-if; service.what_if re-runs the same model version.", "result.html, main.js", "routes.api_what_if, service.what_if", "Added", "tests: test_what_if_api, test_what_if_unavailable_for_a_different_model_version; manual browser check"],
        ["CSV batch analysis", "Upload page, template download, summary, rejected rows with line numbers.", "Analyses many readings in one go.", "service.analyze_batch validates each row, predicts in one call, saves valid rows in one transaction.", "templates/batch.html", "routes.batch, service.analyze_batch", "Added", "tests: test_batch_saves_valid_rows_and_updates_machines_and_alerts, test_batch_reports_bad_rows_with_line_numbers_and_saves_nothing_for_them; manual browser check"],
        ["Machines list", "Table of machines with status, probability, health, alert and counts; search and 'needs attention' filter.", "Machine-wise overview.", "db.list_machines joins each machine's latest prediction and active alert.", "templates/machines.html", "routes.machines", "Added", "tests: test_one_machine_keeps_one_identity_across_analyses, test_machine_list_and_detail"],
        ["Machine page with risk trend", "Current status, health, alert, a trend chart of probability with 30%/70% lines and recent analyses.", "Machine-wise history and risk trend.", "db.machine_detail; Chart.js line chart from embedded JSON.", "templates/machine.html, charts.js", "routes.machine_detail", "Added", "tests: test_machine_page_shows_trend_and_history; manual browser check"],
        ["Alerts list", "Open critical / warning / acknowledged / resolved counts; Active, Resolved, All views; reason; recommended action; Acknowledge and Resolve buttons.", "Manages the alert lifecycle.", "db.list_alerts, db.acknowledge_alert, db.resolve_alert.", "templates/alerts.html", "routes.alerts, alert_acknowledge, alert_resolve", "Modified", "tests: test_alert_lifecycle_over_http, test_manual_resolve_and_invalid_actions; manual browser check"],
        ["Alert detail and history", "Alert facts, optional note field, timeline of events.", "Preserves alert history.", "alert_events table, append-only.", "templates/alert.html", "routes.alert_detail, db.alert_events", "Added", "tests: test_alert_history_is_append_only, test_manual_acknowledge_and_resolve_rules"],
        ["Prediction history", "Searchable, filterable, sortable table with health and model; hover shows model version; CSV export respecting filters.", "Full record of analyses.", "db.list_predictions with whitelisted sorting and bound filters.", "templates/history.html", "routes.history, routes.export_csv", "Modified", "tests: test_history_filters_and_export, test_filters_sorting_and_pagination"],
        ["Model Insights", "Which model is used and why; metrics at 30/50/70%; ensemble comparison; curves; confusion matrices; thresholds; failure-mode classifiers; importance; methodology; model status.", "The technical section; keeps model comparison out of the normal workflow.", "Reads models/metrics.json and meta.json; Chart.js.", "templates/insights.html, charts.js", "routes.insights", "Modified", "tests: test_model_insights_page_reports_all_four_models; manual browser check"],
        ["About page", "Project summary, risk levels, dataset, exploratory charts, failure-mode facts, limitations, team.", "Project information.", "Reads reports/eda.json.", "templates/about.html", "routes.about", "Modified", "tests: test_every_page_loads"],
        ["JSON API", "-", "Programmatic predictions, what-if and health.", "JSON only; errors as JSON.", "-", "routes.api_*", "Modified", "tests: test_api_predict_and_errors, test_what_if_api"],
        ["Model status banners", "Banner when no model, or a fallback model, is in use; database problem banner.", "Explains degraded operation.", "registry.ready / default_is_fallback; db_error.", "base.html", "predictor, __init__", "Added", "tests: test_app_starts_with_no_models_and_explains_instead_of_crashing, test_database_failure_is_reported_not_fatal"],
        ["Loading, error and empty states", "Button shows 'Analyzing...'; friendly error pages; empty-state guides on every list.", "Clear feedback.", "main.js; error.html; per-page empty states.", "main.js, error.html, templates", "__init__ error handlers", "Modified", "tests: test_first_time_user_sees_meaningful_empty_states_and_no_fake_numbers; manual browser check"],
        ["Responsive layout", "Sidebar collapses to a menu on narrow screens; tables scroll sideways.", "Works on phones.", "Bootstrap offcanvas plus own CSS.", "base.html, app.css", "-", "Modified", "manual browser check at 1440 px and 390 px (Chromium emulation)"],
        ["Removed: model dropdown on the main form", "Model choice moved into a collapsed 'Advanced options'.", "Keeps the normal workflow simple.", "-", "analyze.html", "-", "Modified (moved, not removed)", "test_result_page_does_not_expose_model_choice_but_advanced_option_exists"],
        ["Removed: dashboard probability histogram and per-model table", "Replaced by the machine health overview.", "The old charts described predictions, not machines.", "-", "dashboard.html", "-", "Removed", "manual browser check"],
        ["Not implemented: live IoT monitoring, remaining useful life, user accounts", "-", "-", "The data and architecture do not support them.", "-", "-", "Not implemented (by decision)", "-"],
    ]
    return [("table", H, R)]
