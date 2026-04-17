# Crime Prediction ML Project — Full Architecture & Roadmap

> **Academic Purpose**: Predicting crime risk to protect young people in high-risk environments using the US Homicide Reports dataset (638,454 records, 24 features, from 1980).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Feature Engineering Plan](#3-feature-engineering-plan)
4. [Model Architecture](#4-model-architecture)
5. [Full Pipeline Roadmap](#5-full-pipeline-roadmap)
6. [Evaluation Strategy](#6-evaluation-strategy)
7. [Tools & Tech Stack](#7-tools--tech-stack)
8. [Ethical Considerations](#8-ethical-considerations)
9. [Folder Structure](#9-folder-structure)

---

## 1. Project Overview

### Goal
Build a machine learning system that predicts the **risk level of a criminal incident** based on contextual and demographic features — to help researchers and policymakers identify vulnerable youth environments.

### Problem Type
This is a **multi-class classification** problem:
- `Crime Type` or `Crime Solved` as the primary target
- Or a derived **risk score** (Low / Medium / High) as a custom label

### Target Variables (choose one based on objective)

| Option | Target Column | Type |
|--------|--------------|------|
| A | `Crime Type` | Multi-class classification |
| B | `Crime Solved` | Binary classification |
| C | Derived `Risk Score` | Multi-class (engineered label) |

> **Recommended**: Start with Option B (`Crime Solved`) as a binary baseline, then extend to Option A or C.

---

## 2. Dataset Analysis

### Dataset Summary

| Property | Value |
|----------|-------|
| Total Records | 638,454 |
| Total Features | 24 |
| Source | US Homicide Reports (Kaggle) |
| Time Range | 1980 onwards |

### All 24 Features

| # | Column | Type | Role |
|---|--------|------|------|
| 1 | Record ID | Numeric | ID (drop) |
| 2 | Agency Code | Categorical | Location proxy |
| 3 | Agency Name | Categorical | Location proxy |
| 4 | Agency Type | Categorical | Context |
| 5 | City | Categorical | Geography |
| 6 | State | Categorical | Geography |
| 7 | Year | Numeric | Temporal |
| 8 | Month | Numeric | Temporal (cyclical) |
| 9 | Incident | Numeric | Context |
| 10 | Crime Type | Categorical | **Feature / Target** |
| 11 | Crime Solved | Binary | **Target** |
| 12 | Victim | Categorical | Profile |
| 13 | Sex | Categorical | Victim sex |
| 14 | Victim Age | Numeric | Profile |
| 15 | Victim Race | Categorical | Profile |
| 16 | Victim Ethnicity | Categorical | Profile |
| 17 | Perpetrator Sex | Categorical | Profile |
| 18 | Perpetrator Age | Numeric | Profile |
| 19 | Perpetrator Race | Categorical | Profile |
| 20 | Perpetrator Ethnicity | Categorical | Profile |
| 21 | Relationship | Categorical | Social link |
| 22 | Weapon | Categorical | Crime context |
| 23 | Victim Count | Numeric | Severity |
| 24 | Perpetrator Count | Numeric | Severity |

---

## 3. Feature Engineering Plan

### 3.1 Drop / Ignore
- `Record ID` — unique identifier, no predictive value
- `Agency Name` — too high cardinality, use `Agency Type` instead

### 3.2 Temporal Features
```
Month → sin(2π × Month / 12), cos(2π × Month / 12)   # cyclical encoding
Year  → keep as numeric (captures trend over decades)
Season → derived from Month (Winter/Spring/Summer/Fall)
```

### 3.3 Categorical Encoding

| Column | Strategy | Reason |
|--------|----------|--------|
| State | Target Encoding | 50 states → too many for OHE |
| City | Target Encoding | Very high cardinality |
| Agency Type | One-Hot Encoding | Low cardinality |
| Crime Type | Label Encoding | When used as feature |
| Victim Race | One-Hot Encoding | Nominal |
| Perpetrator Race | One-Hot Encoding | Nominal |
| Weapon | One-Hot Encoding | Nominal, important feature |
| Relationship | One-Hot Encoding | Nominal, key predictor |
| Sex / Perpetrator Sex | Binary Encoding | Male=1, Female=0 |

### 3.4 Numeric Features
```
Victim Age       → bin into: [0–12, 13–17, 18–25, 26–40, 41–60, 60+]
Perpetrator Age  → same binning, handle "Unknown" (0 or NaN)
Victim Count     → keep numeric, log-transform if skewed
Perpetrator Count → keep numeric
```

### 3.5 Engineered Features (new columns)
```python
df['age_difference']    = df['Perpetrator Age'] - df['Victim Age']
df['same_race']         = (df['Victim Race'] == df['Perpetrator Race']).astype(int)
df['same_sex']          = (df['Sex'] == df['Perpetrator Sex']).astype(int)
df['is_youth_victim']   = (df['Victim Age'] <= 17).astype(int)   # KEY for your goal
df['is_domestic']       = df['Relationship'].isin(['Wife','Husband','Son','Daughter','Father','Mother']).astype(int)
df['multi_victim']      = (df['Victim Count'] > 1).astype(int)
```

### 3.6 Missing Value Strategy

| Column | Strategy |
|--------|----------|
| Perpetrator Age = 0 | Treat as Unknown → median imputation |
| Perpetrator Race = "Unknown" | Create separate category |
| Perpetrator Sex = "Unknown" | Create separate category |
| Victim Ethnicity | Mode imputation or Unknown category |

---

## 4. Model Architecture

### 4.1 Recommended Models (in order of priority)

#### Tier 1 — Start Here
| Model | Why |
|-------|-----|
| **XGBoost** | Best for tabular data, handles missing values, fast |
| **LightGBM** | Faster than XGBoost on 600K+ rows, excellent performance |
| **Random Forest** | Strong baseline, interpretable feature importance |

#### Tier 2 — Explore Later
| Model | Why |
|-------|-----|
| Logistic Regression | Fast linear baseline, good for explainability |
| CatBoost | Native categorical encoding, no manual encoding needed |
| Neural Network (MLP) | Captures non-linear interactions across all features |

#### Tier 3 — Advanced
| Model | Why |
|-------|-----|
| Stacking Ensemble | Combine XGBoost + LightGBM + RF for best accuracy |
| SHAP Explainer | Explain any model's predictions per-sample |

### 4.2 Recommended Architecture Diagram

```
Raw CSV (638K rows, 24 cols)
        │
        ▼
┌─────────────────────┐
│   Data Cleaning     │  Drop IDs, handle unknowns, fix dtypes
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  Cyclical time, age bins, derived flags
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Encoding & Scaling  │  OHE, Target Encoding, StandardScaler
└────────┬────────────┘
         │
    ┌────┴────┐
    │  Split  │  Train 70% / Val 15% / Test 15%  (stratified)
    └────┬────┘
         │
    ┌────┴──────────────────────┐
    │       Model Training      │
    │  XGBoost / LightGBM / RF  │
    └────┬──────────────────────┘
         │
    ┌────┴────────────┐
    │  Hyperparameter │  Optuna or GridSearchCV
    │  Tuning         │
    └────┬────────────┘
         │
    ┌────┴────────────┐
    │   Evaluation    │  Accuracy, F1, ROC-AUC, Confusion Matrix
    └────┬────────────┘
         │
    ┌────┴────────────┐
    │ Explainability  │  SHAP values, Feature Importance plot
    └─────────────────┘
```

---

## 5. Full Pipeline Roadmap

### Phase 1 — Setup & EDA (Week 1)

- [ ] Download dataset from Kaggle
- [ ] Set up Python environment (see stack below)
- [ ] Load data with `pandas`, check shape, dtypes
- [ ] Explore missing values: `df.isnull().sum()`
- [ ] Visualize distributions: age, crime type, weapon, state
- [ ] Plot correlation heatmap for numeric features
- [ ] Analyze class balance of target variable
- [ ] Document key findings in a notebook

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('database.csv')
print(df.shape)           # (638454, 24)
print(df.isnull().sum())
print(df['Crime Solved'].value_counts(normalize=True))
```

### Phase 2 — Data Preprocessing (Week 2)

- [ ] Drop `Record ID`, high-cardinality columns
- [ ] Handle `Perpetrator Age == 0` → replace with NaN → impute
- [ ] Encode all categorical variables (see Section 3.3)
- [ ] Apply cyclical encoding to `Month`
- [ ] Engineer all new features (Section 3.5)
- [ ] Split into train / val / test sets (stratified)
- [ ] Save processed splits as `.parquet` for fast loading

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Crime Solved', 'Record ID'])
y = df['Crime Solved'].map({'Yes': 1, 'No': 0})

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

### Phase 3 — Baseline Models (Week 3)

- [ ] Train Logistic Regression baseline
- [ ] Train Random Forest
- [ ] Train XGBoost
- [ ] Train LightGBM
- [ ] Compare all models on validation set
- [ ] Plot feature importance for best model
- [ ] Select top 2 models for tuning

```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=50)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

### Phase 4 — Hyperparameter Tuning (Week 4)

- [ ] Use Optuna for XGBoost and LightGBM tuning
- [ ] Tune: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- [ ] Use cross-validation (5-fold stratified)
- [ ] Save best model with `joblib`

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### Phase 5 — Explainability (Week 5)

- [ ] Apply SHAP to best model
- [ ] Generate global feature importance summary plot
- [ ] Generate individual prediction explanations
- [ ] Identify top risk factors for youth victims

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)                    # global
shap.force_plot(explainer.expected_value, shap_values[0]) # single prediction
```

### Phase 6 — Reporting & Documentation (Week 6)

- [ ] Write academic report (Introduction, Methodology, Results, Discussion)
- [ ] Create visualizations: confusion matrix, ROC curve, SHAP plots
- [ ] Document ethical limitations clearly
- [ ] Prepare presentation / poster

---

## 6. Evaluation Strategy

### Metrics to Use

| Metric | Why |
|--------|-----|
| **F1-Score (macro)** | Handles class imbalance fairly |
| **ROC-AUC** | Overall discrimination power |
| **Precision / Recall** | Per-class performance |
| **Confusion Matrix** | Visual error analysis |

### Handling Class Imbalance
If target is imbalanced (e.g., 80% solved vs 20% unsolved):

```python
# Option 1: Class weights
model = xgb.XGBClassifier(scale_pos_weight=4)  # ratio of negative/positive

# Option 2: SMOTE oversampling
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

---

## 7. Tools & Tech Stack

### Environment Setup

```bash
# Create virtual environment
python -m venv crime_env
source crime_env/bin/activate  # Windows: crime_env\Scripts\activate

# Install all dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost
pip install shap optuna imbalanced-learn
pip install matplotlib seaborn plotly
pip install jupyter notebook
```

### Full Stack

| Layer | Tool |
|-------|------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML models | `scikit-learn`, `XGBoost`, `LightGBM` |
| Explainability | `SHAP` |
| Hyperparameter tuning | `Optuna` |
| Imbalanced data | `imbalanced-learn` |
| Notebook | `Jupyter` |
| Version control | `git` + GitHub |

---

## 8. Ethical Considerations

This project deals with sensitive data involving race, ethnicity, age, and sex. The following principles must be observed:

- **No profiling**: The model must not be used to profile individuals. It is for pattern analysis only.
- **Bias auditing**: Always check model performance separately across racial and demographic groups to detect disparate impact.
- **Transparency**: Use SHAP to explain every prediction — black-box use is unacceptable in a justice context.
- **Limitations section**: Clearly state in your report that the model reflects historical patterns which may contain systemic biases.
- **Academic use only**: Do not deploy this model in any real-world law enforcement context.
- **Youth focus**: The `is_youth_victim` flag should be used to analyze vulnerability, not to target communities.

---

## 9. Folder Structure

```
crime_prediction/
│
├── data/
│   ├── raw/
│   │   └── database.csv              # original Kaggle dataset
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_preprocessing.ipynb        # Cleaning & feature engineering
│   ├── 03_baseline_models.ipynb      # First model experiments
│   ├── 04_tuning.ipynb               # Hyperparameter optimization
│   └── 05_explainability.ipynb       # SHAP analysis
│
├── src/
│   ├── preprocessing.py              # Reusable cleaning functions
│   ├── features.py                   # Feature engineering functions
│   ├── train.py                      # Model training script
│   └── evaluate.py                   # Metrics and plotting
│
├── models/
│   ├── xgboost_best.pkl
│   └── lightgbm_best.pkl
│
├── reports/
│   ├── figures/                      # All plots
│   └── final_report.pdf
│
├── requirements.txt
└── README.md
```

---

## Quick Start Checklist

- [ ] Download dataset from Kaggle: *Homicide Reports 1980–2014*
- [ ] Set up Python environment and install dependencies
- [ ] Run `01_eda.ipynb` to understand the data
- [ ] Run `02_preprocessing.ipynb` to clean and engineer features
- [ ] Run `03_baseline_models.ipynb` to get first results
- [ ] Run `04_tuning.ipynb` to optimize your best model
- [ ] Run `05_explainability.ipynb` to generate SHAP plots
- [ ] Write your academic report using the results

---

*Generated for academic research — crime risk prediction for youth protection.*
