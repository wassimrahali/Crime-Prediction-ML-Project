# Crime Prediction Project — Wiki (What was implemented)

This document explains, in plain language, what was built in this workspace and how everything fits together.

The project goal (academic): use the US Homicide Reports dataset to **predict `Crime Solved`** as a first baseline, then extend to other targets (crime type / derived risk score) later.

---

## 1) What you have now (end-to-end)

You now have a complete, reproducible workflow:

1. **EDA** (understand the dataset)
2. **Preprocessing + feature engineering** (clean data + create useful signals)
3. **Baseline model training** (compare several models)
4. **Hyperparameter tuning** (Optuna for XGBoost + LightGBM)
5. **Explainability** (SHAP plots for the best saved model)

Artifacts produced along the way:
- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- `models/*_baseline.joblib` and `models/*_best.joblib`
- `reports/figures/shap_summary.png`

---

## 2) Key design choices (why things look the way they do)

### Target selection
- The baseline target is **`Crime Solved`** because it’s clean and binary (`Yes` / `No`).
- The code supports both the raw string version (`Yes/No`) **and** the encoded numeric version (`0/1`) saved in processed splits.

### Keep notebooks readable, move logic into `src/`
- Notebooks should show the story and the results.
- The reusable logic lives in `src/` so you don’t copy/paste between notebooks.

### Encoding strategy (important)
This dataset has many categorical variables, including high-cardinality columns.

- **One-Hot Encoding** for “normal” categoricals (Weapon, Relationship, Race, Sex, etc.).
- **Target Mean Encoding** for high-cardinality geography (`State`, `City`) to avoid enormous one-hot matrices.

This is implemented in a sklearn-compatible way so models can be trained as a single pipeline.

### Feature engineering
Feature engineering follows your architecture file:
- Month cyclical encoding (`month_sin`, `month_cos`) + `Season`
- Age binning (`VictimAgeBin`, `PerpetratorAgeBin`)
- Flags like `is_youth_victim`, `is_domestic`, `same_race`, `same_sex`, `multi_victim`
- Numeric `age_difference`

---

## 3) Folder structure (what each folder is for)

- `data/raw/` → original CSV (`database.csv`)
- `data/processed/` → engineered splits saved as parquet
- `notebooks/` → EDA, preprocessing, training, tuning, SHAP
- `src/` → reusable Python utilities (the “real pipeline”)
- `models/` → saved sklearn pipelines (`.joblib`)
- `reports/figures/` → exported plots

---

## 4) Notebooks (what each one does)

### `notebooks/01_eda.ipynb`
Purpose: understand the dataset.
- Loads the raw CSV
- Prints columns, data types, missingness
- Plots target distribution and a few basic distributions
- Plots top categories for `State`, `Weapon`, `Relationship`
- Plots a numeric correlation heatmap

### `notebooks/02_preprocessing.ipynb`
Purpose: create the “model-ready” splits.
- Loads raw CSV
- Cleans + engineers features (no encoding yet)
- Builds the target `Crime Solved` → {0,1}
- Creates **stratified train/val/test** (70/15/15)
- Saves parquet splits to `data/processed/`
- Verifies that the preprocessor can fit and transform

Notes:
- Parquet writing requires `pyarrow` available in the active environment/kernel.
- The sklearn preprocessing pipeline is designed to tolerate pandas missing values by converting them to `np.nan` before imputation.
- Feature names for explainability are available via `preprocessor.get_feature_names_out()`.

### `notebooks/03_baseline_models.ipynb`
Purpose: train several baseline models and compare them.
- Loads processed splits
- Trains:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Compares models using validation ROC-AUC
- Evaluates best model on test set (confusion matrix + ROC curve)
- Saves the best baseline pipeline to `models/<name>_baseline.joblib`
- Includes a simple “bias-audit” style table (ROC-AUC by `Victim Race` when sample size is sufficient)

### `notebooks/04_tuning.ipynb`
Purpose: tune the best tree models.
- Uses Optuna to tune XGBoost and LightGBM hyperparameters
- Uses ROC-AUC as objective
- Fits best params on the full training split
- Picks the best of XGB/LGB using validation ROC-AUC
- Evaluates on test set
- Saves tuned models:
  - `models/xgb_best.joblib`
  - `models/lgb_best.joblib`

Tip: this notebook has a `SAMPLE_N` switch so you can tune on a subset first to save time.

### `notebooks/05_explainability.ipynb`
Purpose: explain the best model.
- Loads the best available model from `models/` (prefers tuned models)
- Encodes a test sample using the pipeline’s preprocessor
- Computes SHAP values
- Produces:
  - Global SHAP summary plot (saved to `reports/figures/shap_summary.png`)
  - A per-sample waterfall plot

---

## 5) Source code (`src/`) — what each file contains

### `src/features.py`
Feature engineering only (pure pandas/numpy):
- temporal features
- age bins + youth flag
- similarity flags (`same_race`, `same_sex`) and context flags (`is_domestic`, `multi_victim`)

### `src/preprocessing.py`
Data preparation + sklearn preprocessing:
- load raw CSV
- clean raw values (e.g., `Perpetrator Age == 0` treated as unknown)
- create `Crime Solved` target (robust to raw Yes/No and processed 0/1)
- apply feature engineering
- build a `ColumnTransformer` with:
  - numeric imputation
  - one-hot encoding
  - target mean encoding for high-cardinality columns
- split + save splits as parquet

### `src/train.py`
Model training helpers:
- constructs sklearn `Pipeline(preprocess -> model)`
- baseline model factories (logreg/rf/xgb/lgb)
- save/load model pipelines via joblib

### `src/evaluate.py`
Evaluation helpers:
- ROC-AUC, classification report, confusion matrix
- ROC curve plot
- quick subgroup ROC-AUC computation for auditing
- helpers for plotting top categories

---

## 6) How to run (recommended order)

1. Activate the environment:
   - `venv312\Scripts\Activate.ps1`
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Run the notebooks in order:
   - `01_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_baseline_models.ipynb`
   - `04_tuning.ipynb` (optional but recommended)
   - `05_explainability.ipynb`

---

## 7) Common issues + fixes

### “Import could not be resolved” in VS Code
This usually means VS Code is analyzing with the wrong interpreter.
- Select the interpreter for this workspace as the `venv312` Python.

### Parquet save/load errors
Parquet needs `pyarrow` (already added to requirements).
- If needed: `pip install pyarrow`

If you installed `pyarrow` after opening a notebook kernel, restart the kernel and re-run the notebook cells.

### Scikit-learn warning: “Skipping features without any observed values”
This can happen when a column is entirely missing in the training split.
- It is a warning (not a crash): the imputer skips those all-missing features.
- If you want to eliminate the warning, drop columns that are all-missing in `X_train` before fitting.

### Running tests fails with “No module named pytest”
`pytest` is not currently pinned as a runtime dependency.
- Install: `python -m pip install pytest`
- Run: `python -m pytest -q`

### Tuning is slow
Optuna on 600k+ rows can be expensive.
- Keep `SAMPLE_N` enabled for the first tuning passes
- Increase trials only after you’re confident the pipeline works

---

## 8) Ethics & intended use

This project uses sensitive demographic data. The workflow includes a basic subgroup check, but that’s not a full fairness audit.

This work is intended for **academic pattern analysis** only.
Do not deploy for policing, enforcement, or profiling.
