"""Evaluation utilities (metrics + plots)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	classification_report,
	confusion_matrix,
	roc_auc_score,
	roc_curve,
)


@dataclass(frozen=True)
class EvalResult:
	roc_auc: float
	report: dict[str, Any]
	confusion: np.ndarray


def predict_proba_positive(model, X):
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X)
		return proba[:, 1]
	if hasattr(model, "decision_function"):
		scores = model.decision_function(X)
		# Convert to pseudo-proba via logistic; for ranking metrics only
		return 1.0 / (1.0 + np.exp(-scores))
	raise ValueError("Model does not support probability-like predictions")


def evaluate_binary(model, X, y, *, threshold: float = 0.5) -> EvalResult:
	proba = predict_proba_positive(model, X)
	y_pred = (proba >= threshold).astype(int)
	roc_auc = float(roc_auc_score(y, proba))
	rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
	cm = confusion_matrix(y, y_pred)
	return EvalResult(roc_auc=roc_auc, report=rep, confusion=cm)


def plot_confusion(cm: np.ndarray, *, title: str = "Confusion matrix"):
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot(values_format="d", cmap="Blues")
	plt.title(title)
	plt.grid(False)
	plt.show()


def plot_roc_curve(model, X, y, *, title: str = "ROC curve"):
	proba = predict_proba_positive(model, X)
	fpr, tpr, _ = roc_curve(y, proba)
	auc = roc_auc_score(y, proba)
	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
	plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.show()


def group_metrics_by_column(
	*,
	df_with_target: pd.DataFrame,
	y_true: pd.Series,
	y_proba: np.ndarray,
	group_col: str,
	min_group_size: int = 2000,
):
	"""Compute ROC-AUC per subgroup to support bias auditing."""
	if group_col not in df_with_target.columns:
		raise KeyError(f"Missing group column: {group_col}")

	tmp = df_with_target[[group_col]].copy()
	tmp["y_true"] = y_true.to_numpy()
	tmp["y_proba"] = y_proba

	rows = []
	for value, g in tmp.groupby(group_col, dropna=False):
		if len(g) < min_group_size:
			continue
		try:
			auc = roc_auc_score(g["y_true"], g["y_proba"])
		except Exception:
			continue
		rows.append({group_col: value, "n": len(g), "roc_auc": float(auc)})

	return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def plot_top_categories(series: pd.Series, *, top_n: int = 20, title: str = "Top categories"):
	counts = series.astype("string").value_counts(dropna=False).head(top_n)
	plt.figure(figsize=(10, 5))
	sns.barplot(x=counts.values, y=counts.index, orient="h")
	plt.title(title)
	plt.xlabel("Count")
	plt.ylabel(series.name or "category")
	plt.tight_layout()
	plt.show()