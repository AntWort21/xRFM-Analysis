"""Metric computation shared by every model runner."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


LOWER_IS_BETTER = {"rmse", "mae", "mse", "log_loss"}


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    predictions: Any,
    probabilities: Any | None = None,
) -> dict[str, float | None]:
    """Compute report metrics for regression or classification."""

    try:
        from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    y_true = np.asarray(y_true)
    predictions_array = np.asarray(predictions)

    if task == "regression":
        y_pred = predictions_array.reshape(-1)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    labels = labels_from_predictions(predictions_array)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, labels)),
        "auc_roc": compute_auc_roc(y_true, predictions_array, probabilities, roc_auc_score),
    }
    return metrics


def labels_from_predictions(predictions: np.ndarray) -> np.ndarray:
    """Convert model output into class labels."""

    array = np.asarray(predictions)
    if array.ndim == 2 and array.shape[1] > 1:
        return np.argmax(array, axis=1).astype(int)
    array = array.reshape(-1)
    if np.issubdtype(array.dtype, np.floating):
        rounded = np.rint(array)
        if np.allclose(array, rounded, equal_nan=False):
            return rounded.astype(int)
        return (array >= 0.5).astype(int)
    return array.astype(int)


def compute_auc_roc(y_true: np.ndarray, predictions: np.ndarray, probabilities: Any | None, roc_auc_score: Any) -> float | None:
    """Compute AUC-ROC where scores/probabilities are available."""

    scores = np.asarray(probabilities) if probabilities is not None else np.asarray(predictions)
    try:
        if scores.ndim == 2 and scores.shape[1] > 1:
            if scores.shape[1] == 2:
                return float(roc_auc_score(y_true, scores[:, 1]))
            return float(roc_auc_score(y_true, scores, multi_class="ovr"))
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if np.unique(y_true).size == 2 and np.issubdtype(scores.dtype, np.number):
            return float(roc_auc_score(y_true, scores.reshape(-1)))
    except ValueError:
        return None
    return None


def primary_metric_name(task: str, common_config: Mapping[str, Any]) -> str:
    """Return the configured primary validation metric for a task."""

    return str(common_config.get("metrics", {}).get(task, {}).get("primary", "rmse" if task == "regression" else "accuracy"))


def validation_score(metrics: Mapping[str, float | None], task: str, common_config: Mapping[str, Any]) -> float:
    """Return the scalar validation score used for model selection."""

    metric_name = primary_metric_name(task, common_config)
    value = metrics.get(metric_name)
    if value is None:
        raise ValueError(f"Primary metric {metric_name!r} is unavailable for this run.")
    return float(value)


def higher_is_better(metric_name: str) -> bool:
    """Return whether larger values are better for a metric."""

    return metric_name not in LOWER_IS_BETTER


def is_better_score(candidate: float, incumbent: float | None, metric_name: str) -> bool:
    """Compare validation scores according to metric direction."""

    if incumbent is None:
        return True
    if higher_is_better(metric_name):
        return candidate > incumbent
    return candidate < incumbent

