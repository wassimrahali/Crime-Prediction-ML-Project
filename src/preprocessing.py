"""Preprocessing for the Crime Prediction project.

Responsibilities:
- Load raw CSV
- Basic cleaning / normalization
- Feature engineering (delegated to src.features)
- Train/val/test split
- Sklearn-compatible preprocessing (imputation + encoding)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import engineer_features


RAW_TARGET_COL = "Crime Solved"


@dataclass(frozen=True)
class DataPaths:
	raw_csv: Path
	processed_dir: Path

	@staticmethod
	def from_project_root(project_root: Path) -> "DataPaths":
		return DataPaths(
			raw_csv=project_root / "data" / "raw" / "database.csv",
			processed_dir=project_root / "data" / "processed",
		)


def load_raw_csv(path: str | Path, *, low_memory: bool = False) -> pd.DataFrame:
	path = Path(path)
	return pd.read_csv(path, low_memory=low_memory)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
	"""Lightweight cleaning that keeps semantics intact."""
	out = df.copy()

	# Standardize common columns to numeric where applicable
	for col in ["Year", "Month", "Incident", "Victim Age", "Perpetrator Age", "Victim Count", "Perpetrator Count"]:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce")

	# Perpetrator Age == 0 is 'Unknown' in this dataset
	if "Perpetrator Age" in out.columns:
		out.loc[out["Perpetrator Age"] == 0, "Perpetrator Age"] = np.nan

	# Normalize known string targets
	if RAW_TARGET_COL in out.columns:
		out[RAW_TARGET_COL] = out[RAW_TARGET_COL].astype(str).str.strip()

	return out


def make_target(df: pd.DataFrame, *, target_col: str = RAW_TARGET_COL) -> pd.Series:
	"""Create binary target for Crime Solved.

	Returns Int64 series with values in {0,1} and NA if unknown.
	"""
	ser = df[target_col]

	# If already encoded (0/1), keep it.
	if pd.api.types.is_numeric_dtype(ser):
		y_num = pd.to_numeric(ser, errors="coerce")
		return y_num.astype("Int64")

	y_raw = ser.astype(str).str.strip().str.lower()
	y = y_raw.map({"yes": 1, "no": 0})

	# If mapping failed (e.g., values are '0'/'1' strings), fall back to numeric coercion.
	if y.isna().all():
		y = pd.to_numeric(y_raw, errors="coerce")

	return y.astype("Int64")


def drop_leakage_and_ids(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	drop_cols: list[str] = []

	# IDs / high-cardinality identifiers
	for col in ["Record ID", "Agency Name"]:
		if col in out.columns:
			drop_cols.append(col)

	# This is a provenance field and typically not useful for prediction
	if "Record Source" in out.columns:
		drop_cols.append("Record Source")

	if drop_cols:
		out = out.drop(columns=drop_cols)
	return out


def prepare_dataframe_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
	"""End-to-end dataframe prep prior to encoding.

	This keeps categorical columns as object/category and numeric columns numeric.
	"""
	out = clean_raw(df)
	out = drop_leakage_and_ids(out)
	out = engineer_features(out)
	return out


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
	"""Target mean encoding for high-cardinality categoricals.

	Encodes each column into a single numeric feature based on mean(target) per category.
	- Fit uses training targets only.
	- Transform uses learned mapping; unseen categories map to global prior.

	Parameters
	----------
	smoothing : float
		Larger values shrink category means toward the global mean more strongly.
	min_samples_leaf : int
		Categories with fewer samples are shrunk more strongly.
	"""

	def __init__(self, smoothing: float = 10.0, min_samples_leaf: int = 20):
		self.smoothing = float(smoothing)
		self.min_samples_leaf = int(min_samples_leaf)

	def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[int] | pd.Series):
		if y is None:
			raise ValueError("TargetMeanEncoder requires y during fit().")
		X_df = pd.DataFrame(X)
		y_ser = pd.Series(y).astype(float)
		self._global_mean = float(y_ser.mean())
		self._mappings = {}

		for col in X_df.columns:
			col_ser = X_df[col].astype("string")
			stats = pd.DataFrame({"x": col_ser, "y": y_ser}).groupby("x")["y"].agg(["mean", "count"])  # type: ignore

			# Smoothed mean (similar to typical CatBoost / target encoding smoothing)
			counts = stats["count"].astype(float)
			means = stats["mean"].astype(float)
			smoothing = 1.0 / (1.0 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
			enc = self._global_mean * (1.0 - smoothing) + means * smoothing
			self._mappings[col] = enc.to_dict()

		self.n_features_in_ = X_df.shape[1]
		return self

	def transform(self, X: pd.DataFrame | np.ndarray):
		X_df = pd.DataFrame(X)
		out = np.zeros((X_df.shape[0], X_df.shape[1]), dtype=float)
		for i, col in enumerate(X_df.columns):
			mapping = self._mappings.get(col, {})
			series = X_df[col].astype("string")
			out[:, i] = series.map(mapping).fillna(self._global_mean).astype(float).to_numpy()
		return out

	def get_feature_names_out(self, input_features=None):
		if input_features is None:
			input_features = [f"te_{i}" for i in range(getattr(self, "n_features_in_", 0))]
		return np.array([f"te__{name}" for name in input_features], dtype=object)


@dataclass(frozen=True)
class PreprocessSpec:
	target_encode_cols: tuple[str, ...] = ("State", "City")
	onehot_cols: tuple[str, ...] = (
		"Agency Code",
		"Agency Type",
		"Crime Type",
		"Victim Sex",
		"Victim Race",
		"Victim Ethnicity",
		"Perpetrator Sex",
		"Perpetrator Race",
		"Perpetrator Ethnicity",
		"Relationship",
		"Weapon",
		"Season",
		"VictimAgeBin",
		"PerpetratorAgeBin",
	)
	passthrough_numeric: Literal["auto"] = "auto"


def _existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
	return [c for c in cols if c in df.columns]


def build_preprocessor(df: pd.DataFrame, *, spec: PreprocessSpec | None = None) -> ColumnTransformer:
	"""Build a sklearn ColumnTransformer for the prepared (engineered) dataframe."""
	spec = spec or PreprocessSpec()

	target_cols = _existing_cols(df, spec.target_encode_cols)
	ohe_cols = [c for c in _existing_cols(df, spec.onehot_cols) if c not in target_cols]

	ignore_cols = {RAW_TARGET_COL}
	feature_df = df.drop(columns=[c for c in ignore_cols if c in df.columns])
	numeric_cols = [
		c
		for c in feature_df.columns
		if c not in set(ohe_cols) | set(target_cols)
		and pd.api.types.is_numeric_dtype(feature_df[c])
	]
	# Also allow bool/int-like columns we created (Int64 dtype counts as numeric)

	numeric_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)

	ohe_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
		]
	)

	te_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("te", TargetMeanEncoder(smoothing=10.0, min_samples_leaf=20)),
		]
	)

	transformers = []
	if numeric_cols:
		transformers.append(("num", numeric_pipe, numeric_cols))
	if ohe_cols:
		transformers.append(("cat_ohe", ohe_pipe, ohe_cols))
	if target_cols:
		transformers.append(("cat_te", te_pipe, target_cols))

	return ColumnTransformer(
		transformers=transformers,
		remainder="drop",
		sparse_threshold=0.3,
		verbose_feature_names_out=True,
	)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
	try:
		names = preprocessor.get_feature_names_out()
		return [str(n) for n in names]
	except Exception:
		return []


def stratified_split(
	df: pd.DataFrame,
	y: pd.Series,
	*,
	test_size: float = 0.30,
	val_size: float = 0.50,
	random_state: int = 42,
):
	"""Create train/val/test split with a 70/15/15 default."""
	from sklearn.model_selection import train_test_split

	X_train, X_temp, y_train, y_temp = train_test_split(
		df, y, test_size=test_size, stratify=y, random_state=random_state
	)
	X_val, X_test, y_val, y_test = train_test_split(
		X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
	)
	return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
	processed_dir: str | Path,
	*,
	X_train: pd.DataFrame,
	X_val: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_val: pd.Series,
	y_test: pd.Series,
	format: Literal["parquet", "csv"] = "parquet",
):
	processed_dir = Path(processed_dir)
	processed_dir.mkdir(parents=True, exist_ok=True)

	train_df = X_train.copy()
	val_df = X_val.copy()
	test_df = X_test.copy()
	train_df[RAW_TARGET_COL] = y_train.astype("Int64")
	val_df[RAW_TARGET_COL] = y_val.astype("Int64")
	test_df[RAW_TARGET_COL] = y_test.astype("Int64")

	if format == "parquet":
		train_df.to_parquet(processed_dir / "train.parquet", index=False)
		val_df.to_parquet(processed_dir / "val.parquet", index=False)
		test_df.to_parquet(processed_dir / "test.parquet", index=False)
	else:
		train_df.to_csv(processed_dir / "train.csv", index=False)
		val_df.to_csv(processed_dir / "val.csv", index=False)
		test_df.to_csv(processed_dir / "test.csv", index=False)


def load_processed_split(processed_dir: str | Path, split: Literal["train", "val", "test"]) -> pd.DataFrame:
	processed_dir = Path(processed_dir)
	parquet_path = processed_dir / f"{split}.parquet"
	if parquet_path.exists():
		return pd.read_parquet(parquet_path)
	csv_path = processed_dir / f"{split}.csv"
	if csv_path.exists():
		return pd.read_csv(csv_path)
	raise FileNotFoundError(f"Could not find processed split '{split}' in {processed_dir}.")