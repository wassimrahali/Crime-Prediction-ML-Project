"""Training utilities.

Notebooks can import these helpers, but they are also usable from CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

try:
	import xgboost as xgb
except Exception:  # pragma: no cover
	xgb = None

try:
	import lightgbm as lgb
except Exception:  # pragma: no cover
	lgb = None

from .preprocessing import RAW_TARGET_COL, PreprocessSpec, build_preprocessor, make_target


@dataclass(frozen=True)
class TrainConfig:
	random_state: int = 42
	n_jobs: int = -1


def make_model(name: Literal["logreg", "rf", "xgb", "lgb"], *, cfg: TrainConfig | None = None) -> Any:
	cfg = cfg or TrainConfig()

	if name == "logreg":
		return LogisticRegression(
			max_iter=2000,
			n_jobs=cfg.n_jobs,
			class_weight="balanced",
			solver="lbfgs",
		)

	if name == "rf":
		return RandomForestClassifier(
			n_estimators=400,
			random_state=cfg.random_state,
			n_jobs=cfg.n_jobs,
			class_weight="balanced_subsample",
		)

	if name == "xgb":
		if xgb is None:
			raise ImportError("xgboost is not installed")
		return xgb.XGBClassifier(
			n_estimators=400,
			learning_rate=0.05,
			max_depth=6,
			subsample=0.9,
			colsample_bytree=0.9,
			reg_lambda=1.0,
			random_state=cfg.random_state,
			n_jobs=cfg.n_jobs,
			eval_metric="logloss",
		)

	if name == "lgb":
		if lgb is None:
			raise ImportError("lightgbm is not installed")
		return lgb.LGBMClassifier(
			n_estimators=600,
			learning_rate=0.05,
			num_leaves=63,
			subsample=0.9,
			colsample_bytree=0.9,
			random_state=cfg.random_state,
			n_jobs=cfg.n_jobs,
		)

	raise ValueError(f"Unknown model name: {name}")


def make_pipeline(
	df_features: pd.DataFrame,
	model_name: Literal["logreg", "rf", "xgb", "lgb"],
	*,
	model_params: dict[str, Any] | None = None,
	preprocess_spec: PreprocessSpec | None = None,
	cfg: TrainConfig | None = None,
) -> Pipeline:
	"""Create a preprocessing + model pipeline."""
	cfg = cfg or TrainConfig()
	preprocessor = build_preprocessor(df_features, spec=preprocess_spec)
	model = make_model(model_name, cfg=cfg)
	if model_params:
		model.set_params(**model_params)
	return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def fit_pipeline(
	pipeline: Pipeline,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	*,
	X_val: pd.DataFrame | None = None,
	y_val: pd.Series | None = None,
) -> Pipeline:
	"""Fit a pipeline.

	Note: Passing XGBoost early-stopping eval_set through a Pipeline is possible but
	error-prone; the notebooks keep fitting simple and reproducible.
	"""
	_ = X_val, y_val
	pipeline.fit(X_train, y_train)
	return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, path)


def load_model(path: str | Path) -> Pipeline:
	return joblib.load(Path(path))


def dataframe_X_y(df: pd.DataFrame, *, target_col: str = RAW_TARGET_COL):
	y = make_target(df, target_col=target_col)
	X = df.drop(columns=[target_col])
	# Remove rows with unknown target
	mask = y.notna()
	return X.loc[mask].reset_index(drop=True), y.loc[mask].astype(int).reset_index(drop=True)