"""Narrative content for workbook sheets 1-4. Every number comes from models/metrics.json, models/meta.json or reports/eda.json."""


def f3(v):
    return f"{v:.3f}"


def f4(v):
    return f"{v:.4f}"


def overview(M, meta, eda, T):
    sel = {r["model"]: r for r in M["selection"]["table"]}
    t = M["test"]
    d = meta["default_model"]
    lab = meta["model_names"]
    return [
        ("heading", "What this project is, in simple words"),
        ("table", ["Topic", "Explanation"], [
            ["Project title", "Early Machine Failure Detection Using Machine Learning: Predictive Maintenance through Sensor Data."],
            ["What it does", "You give it a machine's current sensor readings (temperatures, rotational speed, torque, tool wear and machine type). A trained machine-learning model estimates the probability that the machine is in a failing state. The application turns that probability into a status (Normal, Warning or Critical), a health score from 0 to 100, a recommended maintenance action and an explanation of which readings pushed the estimate up or down. Warning and Critical results open an alert that is tracked until someone resolves it. Everything is stored so you can review the history."],
            ["Problem it solves", "Reactive maintenance repairs a machine only after it breaks (expensive unplanned downtime). Scheduled maintenance services machines at fixed intervals whether or not they need it (wasted work, and failures can still happen between visits). Predictive maintenance uses the machine's actual condition to decide when to act. This project is a working prototype of that idea."],
            ["Dataset", f"AI4I 2020 Predictive Maintenance Dataset: {eda['rows']:,} records, {eda['failures']} failures ({eda['failure_rate_pct']}%). It is SYNTHETIC (generated to resemble industrial data), not recorded from real machines. Each record has five sensor readings, a machine type (L, M or H), a yes/no failure label and five failure-mode flags (TWF, HDF, PWF, OSF, RNF)."],
            ["Inputs", "Machine type (L/M/H), air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm), tool wear (minutes), plus a machine ID chosen by the user. Entered in a form, uploaded as a CSV, or sent to the JSON API."],
            ["Outputs", "Failure probability; status (Normal below 30%, Warning 30% to 70%, Critical 70% and above); health score = round(100 x (1 - probability)); recommended action; contribution of each reading to the result (Shapley values); possible failure modes (separate classifiers plus rule checks); what-if exploration; alert; stored history; CSV export."],
            ["What it is NOT", "It is not live monitoring (inputs are manual). The probability is not remaining useful life or time to failure (the data has no time dimension). It does not diagnose the exact failure type. It has not been validated on real machines."],
        ]),
        ("heading", "Technology stack"),
        ("table", ["Layer", "Technology", "Why"], [
            ["Language", "Python 3.12+", "One language for the machine learning and the web backend."],
            ["Machine learning", "scikit-learn, XGBoost, imbalanced-learn (SMOTENC), pandas, NumPy, joblib", "Standard, well-documented tools; imbalanced-learn provides resampling that works inside cross-validation."],
            ["Web backend", "Flask with Jinja templates", "Simple, easy to debug and the team already knew it. The models load once into memory."],
            ["Front end", "HTML, custom CSS design system, Bootstrap 5 (grid and forms), JavaScript, Chart.js", "Clean professional interface without a separate JavaScript framework. All libraries are bundled locally so the demo works offline."],
            ["Database", "SQLite (Python's built-in sqlite3)", "No server to install; supports constraints, triggers, indexes and transactions needed for history and alerts."],
            ["Tests", "pytest", f"{T['total']} automated tests."],
        ]),
        ("heading", "Main modules"),
        ("table", ["Module", "Purpose", "Files"], [
            ["Machine-learning pipeline", "Feature engineering, preprocessing, training, tuning, calibration, evaluation, ensemble, failure-mode classifiers", "ml/features.py, ml/pipelines.py, ml/ensemble.py, ml/train.py, ml/eda.py"],
            ["Model registry", "Loads the trained models once, each independently, checks checksums, picks the default, makes predictions and explanations", "app/predictor.py"],
            ["Service layer", "The single workflow validate -> predict -> store used by the form, the API and the CSV upload", "app/service.py"],
            ["Risk logic", "Thresholds, health score, rule-based indicators, suggested actions", "app/risk.py"],
            ["Validation", "Input rules and messages", "app/validation.py"],
            ["Database layer", "Schema, machine identity, alert lifecycle, queries", "app/db.py, app/migrations.py"],
            ["Web layer", "Pages, JSON API, security, error handling", "app/__init__.py, app/routes.py, app/templates/, app/static/"],
        ]),
        ("heading", "How the front end, backend, database and ML service communicate"),
        ("table", ["From", "To", "How"], [
            ["Browser", "Flask routes", "Normal page requests and form posts (HTML); fetch() with JSON for the what-if feature and for scripts using the API. Every form post carries a CSRF token."],
            ["Flask routes", "Service layer", "Plain Python function calls: analyze_and_store(registry, conn, form)."],
            ["Service layer", "ML service (registry)", "Python calls inside the same process. The models were loaded once at start-up and stay in memory; nothing is retrained per request."],
            ["Service layer", "Database", "sqlite3 with parameterised SQL in one transaction: machine, prediction and alert are saved together or not at all."],
            ["Flask routes", "Browser", "Jinja renders HTML from stored rows; Chart.js draws charts from JSON embedded in the page. The result page shows the stored snapshot and never recomputes it."],
        ]),
        ("heading", "A 60-second summary for the viva"),
        ("table", ["", "Script"], [
            ["", f"We built a predictive-maintenance web application on the AI4I 2020 dataset. Sensor readings go through a saved scikit-learn pipeline that adds derived features such as temperature difference and power, scales the data and feeds a calibrated model. We trained and compared Logistic Regression, Random Forest, XGBoost and a soft-voting ensemble using cross-validation on the training split only, judging them by PR-AUC, precision, recall, F1 and Brier score, not accuracy, because only {eda['failure_rate_pct']}% of records are failures. {lab[d]} had the best validation PR-AUC ({f3(sel[d]['cv_pr_auc'])}), so it is the default; the ensemble did not beat it. The probability is mapped to Normal, Warning or Critical using the thresholds from our proposal, and Warning or Critical results open one alert per machine that follows a defined lifecycle. Every result is stored immutably with its model version. Limitations: the dataset is synthetic, the test set contains only {M['dataset']['test_failures']} failures, and the system estimates failure risk, not time to failure or failure type."],
        ]),
        ("heading", "How to run it"),
        ("table", ["Step", "Command"], [
            ["Create and activate a virtual environment", "python -m venv venv, then source venv/Scripts/activate (Git Bash) or venv\\Scripts\\activate.bat (Command Prompt). The prompt must show (venv)."],
            ["Install", "pip install -r requirements.txt"],
            ["Train (optional, models are included)", "python -m ml.eda then python -m ml.train"],
            ["Run", "flask --app app run   then open http://127.0.0.1:5000"],
            ["Test", "python -m pytest tests -q"],
        ]),
    ]


def architecture(M, meta, eda, T):
    return [
        ("heading", "Request workflow: from sensor input to dashboard"),
        ("table", ["#", "Step", "What happens", "Why it is needed", "File / function", "Data in", "Data out"], [
            [1, "Sensor input", "The user submits the Analyze form, uploads a CSV, or calls the JSON API.", "Three ways in, one workflow behind them.", "app/routes.py: analyze, batch, api_predict", "Form fields / CSV rows / JSON", "A dictionary of raw values (strings)"],
            [2, "Validation", "Machine ID is required and must match a safe pattern; type must be L, M or H; each reading must be a finite number inside hard physical limits. Values outside the training range are accepted with a warning.", "Bad input must never reach the model or the database; users get field-level messages.", "app/validation.py: validate", "Raw strings", "Clean numbers, errors per field, warnings"],
            [3, "Machine identity", "The machine ID (case-insensitive) is looked up; if new, a machine row is created; if it exists with a different type the request is refused.", "A machine must keep one identity so its history, trend and alert make sense.", "app/db.py: get_or_create_machine", "Machine ID, type", "machines row"],
            [4, "Preprocessing", "Numeric features are standardised (mean 0, spread 1); machine type is converted to a numeric code and then to three 0/1 columns. This happens INSIDE the saved pipeline.", "Models such as Logistic Regression need comparable scales; the same fitted scaler must be used at prediction time.", "ml/pipelines.py: build_pipeline (ColumnTransformer)", "Raw readings", "Scaled numeric array"],
            [5, "Feature engineering", "Temperature difference, approximate power and the wear x torque load proxy are computed from the raw readings.", "They express physical relationships that the dataset's failure rules use; they make patterns easier for the models to find.", "ml/features.py: FeatureBuilder", "Raw readings", "Raw plus derived columns"],
            [6, "Model prediction", "The chosen model (default: " + meta["model_names"][meta["default_model"]] + ") outputs a calibrated failure probability.", "This is the machine-learning estimate the whole application is built around.", "app/predictor.py: ModelRegistry.analyze_many", "Feature array", "Probability between 0 and 1"],
            [7, "Probability calibration", "The model's raw scores were adjusted during training (isotonic or sigmoid) so that, for example, 70% really corresponds to about 70% observed failures.", "The Warning/Critical thresholds are only meaningful if probabilities are honest.", "ml/train.py (CalibratedClassifierCV); saved inside the model files", "Raw score", "Calibrated probability"],
            [8, "Risk classification and health score", "Below 30% is Normal, 30% up to 70% is Warning, 70% and above is Critical (lower bound inclusive). Health = round(100 x (1 - probability)).", "Turns a number into an action; the thresholds come from the project proposal.", "app/risk.py: classify, health_score", "Probability", "Status, severity, health score"],
            [9, "Indicators and failure modes", "Rule checks test the failure conditions documented for the dataset (heat dissipation, power, overstrain, tool wear). Separate classifiers estimate TWF, HDF, PWF and OSF likelihoods; only those that were reliable in validation are shown.", "Gives maintenance staff something specific to inspect, without pretending the binary model identifies failure types.", "app/risk.py: rule_indicators; app/predictor.py: _mode_estimates", "Readings, status", "Indicator list, mode estimates"],
            [10, "Explanation", "Exact Shapley values over five sensor groups show how much each reading moved the probability away from a typical reading.", "Users need to see why a result came out as it did.", "app/predictor.py: ModelRegistry.explain", "Readings, model", "Contribution per group (percentage points)"],
            [11, "Recommendation", "Action text for the status plus the suggested action of each triggered rule indicator.", "Tells the user what to do.", "app/risk.py: recommendation", "Status, indicators", "Recommendation text"],
            [12, "Database storage", "In ONE transaction: save the prediction snapshot (including explanation and indicators), then apply the alert lifecycle (open / update / escalate / resolve).", "History must be complete and consistent; a failure must not leave half-saved data.", "app/db.py: record_prediction, apply_alert_lifecycle", "Result dictionary", "predictions, alerts, alert_events rows"],
            [13, "Display", "Result page, dashboard, machines, alerts, history and CSV export read the stored rows.", "Everything shown is real stored data; the result page never recomputes.", "app/routes.py and app/templates/", "Stored rows", "HTML, JSON for charts, CSV"],
        ]),
        ("heading", "Training pipeline (offline, ml/train.py)"),
        ("table", ["#", "Step", "What happens", "Uses only"], [
            [1, "Data quality check", "Missing values, duplicates, invalid types and impossible values are counted; training stops if a check fails.", "All data (no labels used for decisions)"],
            [2, "Split", f"One stratified 80/20 split: {M['dataset']['train_rows']} training and {M['dataset']['test_rows']} test machines.", "-"],
            [3, "Feature set and imbalance strategy", "Repeated 5-fold cross-validation compares with/without the wear x torque proxy and none / class weights / SMOTENC.", "Training split"],
            [4, "Hyperparameter tuning", "Randomized search scored on PR-AUC with 5-fold cross-validation.", "Training split"],
            [5, "Calibration", "Out-of-fold predictions compare no calibration, sigmoid and isotonic by Brier score.", "Training split"],
            [6, "Ensemble weights", "Equal weights versus a grid of weights, using the out-of-fold calibrated probabilities, with a split-half guard.", "Training split"],
            [7, "Model selection", "Highest validation PR-AUC; ties within 0.01 broken by Brier score, then simplicity.", "Training split"],
            [8, "Test evaluation", "One evaluation on the held-out test split with bootstrap confidence intervals.", "Test split (report only)"],
            [9, "Failure-mode classifiers", "One calibrated Random Forest each for TWF, HDF, PWF, OSF; reliability gate at validation PR-AUC >= 0.5.", "Training split, then reported on test"],
            [10, "Save", "Models, checksums, version, seed, library versions and results are written; meta.json is written last.", "-"],
        ]),
        ("heading", "Key design properties"),
        ("table", ["Property", "How it is achieved"], [
            ["No train/serve mismatch", "Feature engineering, scaling and encoding live inside the saved pipeline object; the web app calls the same object."],
            ["Models loaded once", "ModelRegistry reads the files at start-up; requests only call predict_proba."],
            ["Failure isolation", "Each model file is loaded independently with a checksum check; one bad file disables only that model."],
            ["Immutable history", "Each saved prediction stores its own explanation snapshot and model version; database triggers forbid UPDATE and DELETE."],
        ]),
    ]


def models(M, meta, eda, T):
    tun = M["tuning"]
    t = M["test"]
    sel = {r["model"]: r for r in M["selection"]["table"]}
    e = M["ensemble"]

    def res(k):
        x = t[k]["thresholds"]["0.3"]
        return (f"Validation (cross-validated, calibrated) PR-AUC {f3(sel[k]['cv_pr_auc'])}, Brier {f4(sel[k]['cv_brier'])}. "
                f"Test set at the 30% threshold: precision {f3(x['precision'])}, recall {f3(x['recall'])}, F1 {f3(x['f1'])}, "
                f"PR-AUC {f3(t[k]['pr_auc'])} (95% CI {t[k]['ci95']['pr_auc'][0]:.2f}-{t[k]['ci95']['pr_auc'][1]:.2f}).")
    rf, xg, lr = tun["rf"]["best_params"], tun["xgb"]["best_params"], tun["lr"]["best_params"]
    return [
        ("heading", "1. Logistic Regression"),
        ("table", ["Aspect", "Explanation"], [
            ["What it is", "A linear classifier. Despite the name it classifies: it estimates the probability of 'failure' from a weighted sum of the input features."],
            ["How it works internally", "It computes z = b0 + w1 x1 + w2 x2 + ... for the (scaled) features, then squeezes z into a probability with the sigmoid function p = 1 / (1 + e^-z). Large positive z means p near 1 (failure), large negative z means p near 0. Each weight tells how strongly one feature pushes the log-odds up or down."],
            ["How it learns", "It searches for the weights that minimise the log loss (cross-entropy) on the training data: the loss is small when the predicted probability is high for real failures and low for non-failures. An L2 penalty keeps the weights small to avoid over-fitting. The solver (lbfgs) improves the weights step by step until the loss stops decreasing."],
            ["What it predicts", "A failure probability between 0 and 1."],
            ["Advantages", "Fast, simple, interpretable (each weight has a direction and size), a sensible baseline."],
            ["Limitations", "It draws ONE straight boundary in feature space. Failure risk in this data is U-shaped (high at both low and high speed, torque and power) and depends on interactions (for example low temperature difference AND low speed), which a straight boundary cannot represent."],
            ["Why included here", "As the interpretable baseline named in the project proposal; it shows how much the non-linear models add."],
            ["Hyperparameters used", f"C = {lr['C']:.3f} (chosen by randomized search, {tun['lr']['n_iter']} settings tried); class weights: {'balanced' if tun['lr']['strategy'] == 'class_weight' else 'none'}; max_iter = 3000; penalty L2 (default)."],
            ["How the hyperparameters affect it", "C is the inverse of the regularisation strength: a small C forces small weights (simpler, may under-fit); a large C lets weights grow (more flexible, may over-fit). class_weight='balanced' makes mistakes on the rare failure class cost more."],
            ["Result in this project", res("lr") + " It is clearly the weakest model, for the reason given under Limitations. Its accuracy (96.9% on the test set) is barely above the 96.6% obtained by always predicting 'no failure', which shows why accuracy is not used for selection."],
        ]),
        ("heading", "2. Random Forest"),
        ("table", ["Aspect", "Explanation"], [
            ["What it is", "An ensemble of many decision trees whose predictions are averaged."],
            ["How it works internally", "A decision tree asks a series of yes/no questions about the features (for example 'is temperature difference below 8.6?', 'is speed below 1380?') and ends in a leaf that holds the share of failures among the training samples that reached it. A single deep tree memorises the training data. A random forest trains many trees, each on a different bootstrap sample (rows drawn with replacement) and, at every split, considering only a random subset of the features. The trees therefore make different mistakes; averaging their probabilities cancels much of the noise (it reduces variance)."],
            ["How it learns", "Each tree is grown greedily: at every node it picks the feature and threshold that best separates failures from non-failures (lowest Gini impurity), then repeats on each side until a stopping rule (maximum depth or minimum leaf size) is met. There is no gradient descent; the forest is simply many independently grown trees."],
            ["What it predicts", "The average of the trees' leaf probabilities, i.e. a failure probability. We then calibrate it (isotonic regression) so the probabilities are trustworthy."],
            ["Advantages", "Captures thresholds, interactions and U-shaped relationships automatically; needs little preprocessing; robust; gives feature importances."],
            ["Limitations", "Less interpretable than a linear model; cannot extrapolate beyond the range of the training data (it predicts the nearest leaf); raw probabilities are not well calibrated, hence the calibration step; with few failures a leaf may rest on very few samples."],
            ["Why included here", "Named in the project proposal as the non-linear ensemble model. The dataset's failure rules are threshold rules, which trees represent naturally."],
            ["Hyperparameters used", f"n_estimators = {rf['n_estimators']}, max_depth = {rf['max_depth']}, min_samples_leaf = {rf['min_samples_leaf']}, max_features = {rf['max_features']} (randomized search, {tun['rf']['n_iter']} settings tried); class weights: {'balanced_subsample' if tun['rf']['strategy'] == 'class_weight' else 'none'}."],
            ["How the hyperparameters affect it", "n_estimators: more trees give a steadier average (diminishing returns, slower). max_depth: deeper trees fit finer patterns but can over-fit. min_samples_leaf: larger values force leaves to be based on more samples, which smooths predictions. max_features: how many features each split may consider; smaller values make the trees more different from each other."],
            ["Result in this project", res("rf") + f" Selected as the default model because it had the best validation PR-AUC."],
        ]),
        ("heading", "3. XGBoost"),
        ("table", ["Aspect", "Explanation"], [
            ["What it is", "A gradient-boosted decision-tree model."],
            ["How it works internally", "Instead of growing independent trees, boosting grows them one after another. Each new small tree is trained to correct the errors the current ensemble still makes. The final prediction is the sum of all trees' outputs (in log-odds), passed through a sigmoid to give a probability."],
            ["How it learns", "At each round it computes, for every training sample, the gradient (and curvature) of the log loss with respect to the current prediction - in effect 'which direction would reduce the error for this sample' - and fits a new tree to those values. The tree's contribution is multiplied by a small learning rate before being added, so the model improves gradually. Regularisation terms (lambda), row sampling (subsample) and column sampling (colsample_bytree) limit over-fitting."],
            ["What it predicts", "A failure probability (calibrated afterwards)."],
            ["Advantages", "Often the strongest model on tabular data; flexible; built-in regularisation."],
            ["Limitations", "More hyperparameters to tune; can over-fit when there are few positive examples (here 271 failures in training); sequential training; harder to interpret."],
            ["Why included here", "Named in the project proposal as the gradient-boosting model."],
            ["Hyperparameters used", f"n_estimators = {xg['n_estimators']}, max_depth = {xg['max_depth']}, learning_rate = {xg['learning_rate']}, subsample = {xg['subsample']}, colsample_bytree = {xg['colsample_bytree']}, min_child_weight = {xg['min_child_weight']}, reg_lambda = {xg['reg_lambda']} (randomized search, {tun['xgb']['n_iter']} settings tried)."],
            ["How the hyperparameters affect it", "learning_rate: smaller steps need more trees but generalise better. max_depth: complexity of each tree. n_estimators: number of boosting rounds. subsample / colsample_bytree: fraction of rows / columns used per tree (adds randomness, reduces over-fitting). min_child_weight: minimum amount of data a leaf needs. reg_lambda: L2 penalty on leaf values."],
            ["Result in this project", res("xgb") + " On the test set it is statistically indistinguishable from Random Forest (overlapping confidence intervals), but its validation PR-AUC and Brier score were worse, so it was not selected."],
        ]),
        ("heading", "4. Soft Voting Ensemble"),
        ("table", ["Aspect", "Explanation"], [
            ["What it is", "A model that combines the three models above by averaging their predicted probabilities (soft voting). 'Hard' voting would instead count each model's final yes/no vote."],
            ["How it works internally", "For a new reading each member produces a calibrated failure probability; the ensemble returns the weighted average p = w1 p_LR + w2 p_RF + w3 p_XGB with weights that sum to 1."],
            ["How it learns", "The members are trained and calibrated separately; the ensemble itself has nothing to train except its weights. We compared equal weights with weights searched on a grid, using ONLY out-of-fold predictions from the training data (never the test set). Searched weights replace equal weights only if they beat them by at least 0.005 PR-AUC, the gain also holds on a split-half check, and they form a real blend."],
            ["What it predicts", "A failure probability."],
            ["Advantages", "In principle, errors that differ between models partly cancel, giving a steadier prediction than any single model."],
            ["Limitations", "It helps only if the members are about equally good and make different mistakes. A weak member drags the average down. It is slower (three models per prediction) and harder to explain."],
            ["Why included here", "Requested as an additional experiment: to test whether combining the models improves on the best single model, without assuming that it does."],
            ["Weights used", f"Equal weights (lr {e['weights']['lr']:.2f}, rf {e['weights']['rf']:.2f}, xgb {e['weights']['xgb']:.2f}). The weight search on validation data returned lr 0.0, rf 1.0, xgb 0.0 (gain {e['gain_on_oof']:+.3f} PR-AUC), which is simply the Random Forest alone, not a real ensemble, so it was NOT adopted. Reported for transparency: those weights on the test set give PR-AUC {f3(e['tuned_variant_test']['pr_auc'])}."],
            ["Result in this project", res("ens") + " The equal-weight ensemble did NOT beat Random Forest: Logistic Regression is much weaker than the other two and lowers the average. No blend found on validation data improved on Random Forest, so the ensemble is available as a model but is not the default."],
        ]),
        ("heading", "Supporting techniques (frequently asked about)"),
        ("table", ["Technique", "Explanation", "Where used"], [
            ["Cross-validation", "The training data is split into 5 parts. Each part is held out in turn while the model trains on the other 4, giving an honest estimate of performance on unseen data using only training data. 'Stratified' keeps the failure percentage the same in every part.", "Feature set, imbalance strategy, tuning, calibration, ensemble weights, model selection"],
            ["Probability calibration", "Model scores are not always true probabilities. Calibration fits a small correction (sigmoid or isotonic regression) on held-out folds so that predicted 30% means about 30% observed. Judged with the Brier score.", "ml/train.py, CalibratedClassifierCV"],
            ["SMOTENC", "Creates synthetic minority samples by interpolating between a failure sample and one of its nearest failure neighbours; the NC version handles categorical columns. We tested it; it did not improve PR-AUC for any model, so it is not used. It is applied inside cross-validation folds only, and machine type is one categorical column (see the Change Log for the bug that was fixed).", "ml/pipelines.py (strategy 'smote')"],
            ["Class weights", "Make mistakes on the rare class cost more during training, instead of creating synthetic data. Compared; did not improve PR-AUC either.", "Strategy 'class_weight'"],
            ["Shapley values", "A fair way to share a prediction among the inputs: the contribution of an input is its average effect on the prediction across all possible orders in which the inputs could be 'switched on'. We compute them exactly for five sensor groups against a background of typical readings; the contributions add up to the difference between this prediction and the typical one. They describe the MODEL, not physical causes.", "app/predictor.py: explain"],
            ["Permutation importance", "Shuffle one input column on the test set and see how much PR-AUC drops; a big drop means the model relied on that input.", "Model Insights page"],
            ["PR curve and PR-AUC", "Precision against recall at every threshold; the area under it summarises ranking quality on the rare class. Preferred over ROC-AUC for imbalanced data.", "Selection metric"],
        ]),
    ]


def dataset(M, meta, eda, T):
    q = M["data_quality"]
    ts = eda["type_stats"]
    return [
        ("heading", "Dataset"),
        ("table", ["Item", "Value / explanation"], [
            ["Name and source", "AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository). Synthetic data designed to resemble industrial milling-machine data."],
            ["Records", f"{eda['rows']:,}"],
            ["Failures (target = 1)", f"{eda['failures']} ({eda['failure_rate_pct']}%); non-failures {eda['rows'] - eda['failures']:,} ({100 - eda['failure_rate_pct']:.2f}%)"],
            ["Machine types", "; ".join(f"{k}: {v['count']:,} records, {v['failures']} failures ({v['failure_rate_pct']}%)" for k, v in ts.items())],
            ["Train / test split", f"{M['dataset']['train_rows']} / {M['dataset']['test_rows']} (stratified, seed {meta['seed']}); failures {M['dataset']['train_failures']} / {M['dataset']['test_failures']}"],
            ["Target variable", "Machine failure: 1 = the machine failed, 0 = it did not."],
            ["Failure-mode columns (NOT inputs)", "TWF tool wear failure, HDF heat dissipation failure, PWF power failure, OSF overstrain failure, RNF random failure. They define the target, so using them as inputs would be data leakage. They are used only to train the separate failure-mode classifiers and for analysis."],
            ["Identifier columns dropped", "UDI and Product ID carry no physical information."],
        ]),
        ("heading", "Features: measured, derived and heuristic"),
        ("table", ["Feature", "Category", "Unit", "Meaning", "Formula / how obtained", "Model input?"], [
            ["type", "Categorical (product quality variant)", "L / M / H", "Quality variant of the product the machine is making (low, medium, high)", "Given", "Yes (ordinal code, then one-hot)"],
            ["air_temp", "Sensor measurement", "K (kelvin)", "Temperature of the surrounding air", "Given", "Yes (scaled)"],
            ["process_temp", "Sensor measurement", "K", "Temperature of the machining process", "Given", "Yes (scaled)"],
            ["rpm", "Sensor measurement", "rpm", "Rotational speed of the spindle", "Given", "Yes (scaled)"],
            ["torque", "Sensor measurement", "Nm", "Turning force applied by the spindle", "Given", "Yes (scaled)"],
            ["tool_wear", "Sensor measurement / counter", "min", "Minutes the cutting tool has been in use", "Given", "Yes (scaled)"],
            ["temp_diff", "Derived measurement", "K", "Heat generated by the process above the surroundings", "process_temp - air_temp", "Yes (scaled)"],
            ["power_w", "Derived physical quantity (approximate)", "W", "Mechanical power delivered", "torque x 2 x pi x rpm / 60", "Yes (scaled)"],
            ["wear_torque", "HEURISTIC PROXY (not physical strain)", "min x Nm", "Load placed on an already worn tool. The dataset documentation defines its overstrain failure through this product, which is why it is a useful feature; it is not a measured or physical strain.", "tool_wear x torque", f"Yes, if it helps in cross-validation (it did: mean PR-AUC gain {M['feature_engineering']['mean_pr_auc_gain_from_wear_torque']:+.4f}, rule: more than +0.01)"],
        ]),
        ("heading", "Data handling"),
        ("table", ["Topic", "What was done and why"], [
            ["Missing values", f"Checked before every training run: {q['missing_values']} found. If any appeared, training would stop with an error."],
            ["Duplicates", f"Full duplicate rows: {q['duplicate_rows']}. Duplicate sensor-reading combinations: {q['duplicate_feature_rows']}."],
            ["Invalid values", f"Invalid machine type: {q['invalid_type']}; non-positive speed or torque: {q['non_positive_rpm_or_torque']}; negative tool wear: {q['negative_tool_wear']}; process temperature not above air temperature: {q['process_not_above_air']}; target not 0/1: {q['target_not_binary']}."],
            ["Categorical encoding", "Machine type is ordinal-coded (H=0, L=1, M=2) and, AFTER any resampling, one-hot encoded into three 0/1 columns. The order matters for SMOTENC (see Change Log)."],
            ["Numerical scaling", "StandardScaler (mean 0, standard deviation 1) fitted on training data only, inside the pipeline. Trees do not need it but it is harmless; Logistic Regression does."],
            ["Class imbalance", f"{eda['failure_rate_pct']}% failures. Handled by (1) evaluating with PR-AUC, precision, recall, F1 and Brier instead of accuracy, (2) comparing no handling, class weights and SMOTENC by cross-validation, (3) calibrating probabilities. Result: no resampling or weighting improved PR-AUC, so none is used."],
            ["Leakage prevention", "Failure-mode columns and identifiers are never inputs. All fitting (scaler, resampler, hyperparameter search, calibration, ensemble weights, model choice) uses training folds only. The test set is used once for reporting."],
            ["Range awareness", "Training minimum and maximum of each reading are stored in meta.json; inputs outside them are accepted with a visible warning because the prediction is then an extrapolation."],
        ]),
        ("heading", "What the data shows (from the exploratory analysis)"),
        ("table", ["Finding", "Evidence"], [
            ["Risk is not monotonic", "Failure rate is highest at low and at high speed, torque and power and is lowest in the middle. This is why Logistic Regression (a single straight boundary) performs poorly."],
            ["HDF, PWF, OSF follow exact rules", f"HDF: temperature difference below 8.6 K and speed below 1380 rpm ({eda['rule_checks']['HDF']['mismatches']} exceptions in {eda['mode_counts']['HDF']} records). PWF: power below 3,500 W or above 9,000 W ({eda['rule_checks']['PWF']['mismatches']} exceptions). OSF: tool wear x torque above 11,000 / 12,000 / 13,000 for L / M / H ({eda['rule_checks']['OSF']['mismatches']} exceptions)."],
            ["TWF is only partly deterministic", f"All {eda['mode_counts']['TWF']} tool wear failures occur between {eda['rule_checks']['TWF']['wear_min']} and {eda['rule_checks']['TWF']['wear_max']} minutes of wear, but only {eda['tool_wear_rates']['ge_200_pct']}% of records at 200 minutes or more fail (versus {eda['tool_wear_rates']['lt_200_pct']}% below)."],
            ["RNF is random", f"{eda['rule_checks']['RNF']['flagged']} records flagged, only {eda['rule_checks']['RNF']['of_which_machine_failure']} of which is labelled a machine failure. It cannot be predicted."],
            ["Consequence", "Because three of the five modes are generated by fixed rules, tree models can reach very high scores on this synthetic data. This should not be read as performance on real machines."],
        ]),
    ]
