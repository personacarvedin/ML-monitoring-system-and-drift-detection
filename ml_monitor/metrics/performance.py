"""
Model performance metrics for classification and regression.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_proba : predicted probabilities (optional, needed for AUC)
    average : sklearn averaging strategy
    """
    metrics: Dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )

    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                # binary
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=average
                    )
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Unified entry point.

    Parameters
    ----------
    task : "classification" | "regression"
    """
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba, **kwargs)
    elif task == "regression":
        return regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")