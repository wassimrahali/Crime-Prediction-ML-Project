"""Feature engineering utilities for the Crime Prediction project.

These functions keep the notebooks/scripts consistent and reproducible.
They are intentionally lightweight and only depend on pandas/numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


YOUTH_AGE_CUTOFF = 17


@dataclass(frozen=True)
class FeatureConfig:
	youth_age_cutoff: int = YOUTH_AGE_CUTOFF
	domestic_relationships: tuple[str, ...] = (
		"Wife",
		"Husband",
		"Son",
		"Daughter",
		"Father",
		"Mother",
	)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add cyclical Month encoding and Season.

	Expects columns: 'Month' (1..12) and optionally 'Year'.
	"""
	out = df.copy()

	if "Month" in out.columns:
		month = pd.to_numeric(out["Month"], errors="coerce")
		month = month.clip(lower=1, upper=12)
		radians = 2.0 * np.pi * (month / 12.0)
		out["month_sin"] = np.sin(radians)
		out["month_cos"] = np.cos(radians)

		# Simple season mapping (Northern hemisphere)
		season = pd.Series(pd.NA, index=out.index, dtype="object")
		season = season.mask(month.isin([12, 1, 2]), "Winter")
		season = season.mask(month.isin([3, 4, 5]), "Spring")
		season = season.mask(month.isin([6, 7, 8]), "Summer")
		season = season.mask(month.isin([9, 10, 11]), "Fall")
		out["Season"] = season

	return out


def _bin_age(series: pd.Series) -> pd.Series:
	numeric = pd.to_numeric(series, errors="coerce")
	bins = [-np.inf, 12, 17, 25, 40, 60, np.inf]
	labels = ["0-12", "13-17", "18-25", "26-40", "41-60", "60+"]
	return pd.cut(numeric, bins=bins, labels=labels)


def add_age_features(df: pd.DataFrame, *, cfg: FeatureConfig | None = None) -> pd.DataFrame:
	"""Add age bins and age difference features.

	- Replaces perpetrator age '0' with NaN upstream (in preprocessing) where possible.
	- Adds victim/perpetrator age bins (categorical).
	- Adds age_difference numeric feature.
	- Adds is_youth_victim flag.
	"""
	cfg = cfg or FeatureConfig()
	out = df.copy()

	victim_age = pd.to_numeric(out.get("Victim Age"), errors="coerce")
	perp_age = pd.to_numeric(out.get("Perpetrator Age"), errors="coerce")

	if "Victim Age" in out.columns:
		out["VictimAgeBin"] = _bin_age(out["Victim Age"])
	if "Perpetrator Age" in out.columns:
		out["PerpetratorAgeBin"] = _bin_age(out["Perpetrator Age"])

	if "Victim Age" in out.columns and "Perpetrator Age" in out.columns:
		out["age_difference"] = perp_age - victim_age

	if "Victim Age" in out.columns:
		out["is_youth_victim"] = (victim_age <= cfg.youth_age_cutoff).astype("Int64")

	return out


def add_similarity_flags(df: pd.DataFrame, *, cfg: FeatureConfig | None = None) -> pd.DataFrame:
	"""Add engineered boolean/int flags based on demographic similarity and context."""
	cfg = cfg or FeatureConfig()
	out = df.copy()

	if "Victim Race" in out.columns and "Perpetrator Race" in out.columns:
		out["same_race"] = (out["Victim Race"] == out["Perpetrator Race"]).astype("Int64")

	if "Victim Sex" in out.columns and "Perpetrator Sex" in out.columns:
		out["same_sex"] = (out["Victim Sex"] == out["Perpetrator Sex"]).astype("Int64")

	if "Relationship" in out.columns:
		out["is_domestic"] = out["Relationship"].isin(cfg.domestic_relationships).astype("Int64")

	if "Victim Count" in out.columns:
		out["multi_victim"] = (pd.to_numeric(out["Victim Count"], errors="coerce") > 1).astype("Int64")

	return out


def engineer_features(df: pd.DataFrame, *, cfg: FeatureConfig | None = None) -> pd.DataFrame:
	"""End-to-end feature engineering (no encoding).

	This returns a dataframe with new columns added and is safe to call
	prior to train/val/test splitting.
	"""
	cfg = cfg or FeatureConfig()
	out = df
	out = add_temporal_features(out)
	out = add_age_features(out, cfg=cfg)
	out = add_similarity_flags(out, cfg=cfg)
	return out