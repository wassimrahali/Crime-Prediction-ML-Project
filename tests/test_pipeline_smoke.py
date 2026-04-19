import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_binary
from src.preprocessing import DataPaths, prepare_dataframe_for_modeling, stratified_split
from src.train import dataframe_X_y, fit_pipeline, make_pipeline


class TestPipelineSmoke(unittest.TestCase):
	def test_end_to_end_train_eval_logreg(self):
		project_root = Path(__file__).resolve().parents[1]
		paths = DataPaths.from_project_root(project_root)
		if not paths.raw_csv.exists():
			self.skipTest(f"Missing dataset at {paths.raw_csv}.")

		# Keep the smoke test fast and stable.
		raw = pd.read_csv(paths.raw_csv, low_memory=False, nrows=25000)
		df = prepare_dataframe_for_modeling(raw)
		X, y = dataframe_X_y(df)
		if len(X) < 2000:
			self.skipTest("Not enough labeled rows for smoke test.")

		# Ensure both classes are present (required for ROC-AUC).
		y_value_counts = y.value_counts(dropna=True)
		if len(y_value_counts) < 2:
			self.skipTest("Need at least two classes for ROC-AUC.")

		# Downsample for speed (while keeping stratification possible).
		if len(X) > 12000:
			sampled = pd.concat([X, y.rename("y")], axis=1).sample(12000, random_state=42)
			X = sampled.drop(columns=["y"]).reset_index(drop=True)
			y = sampled["y"].reset_index(drop=True)

		X_train, X_val, _, y_train, y_val, _ = stratified_split(
			X, y, test_size=0.30, val_size=0.50, random_state=42
		)
		pipe = make_pipeline(X_train, "logreg")
		pipe = fit_pipeline(pipe, X_train, y_train)

		res = evaluate_binary(pipe, X_val, y_val)
		self.assertTrue(np.isfinite(res.roc_auc))
		self.assertGreaterEqual(res.roc_auc, 0.0)
		self.assertLessEqual(res.roc_auc, 1.0)


if __name__ == "__main__":
	unittest.main()
