"""Split conformal intervals calibrated on a chronologically separate set."""
from __future__ import annotations

import numpy as np


def conformal_quantile(y_true: np.ndarray, y_pred: np.ndarray,
                       alpha: float = 0.10) -> np.ndarray:
    """Finite-sample absolute-residual quantiles, one independently per lead."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("calibration truth and prediction must be matching (n, horizon) arrays")
    if not 0 < alpha < 1 or len(y_true) == 0:
        raise ValueError("alpha must be in (0, 1) and calibration data cannot be empty")
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)
    rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return np.partition(residuals, rank - 1, axis=0)[rank - 1]


def prediction_interval(y_pred: np.ndarray, quantile: np.ndarray):
    y_pred = np.asarray(y_pred, dtype=float)
    quantile = np.asarray(quantile, dtype=float)
    return y_pred - quantile, y_pred + quantile


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    y_true, lower, upper = map(lambda x: np.asarray(x, dtype=float), (y_true, lower, upper))
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("truth and interval bounds must have matching shapes")
    return {"picp": float(np.mean((y_true >= lower) & (y_true <= upper))),
            "mean_interval_width": float(np.mean(upper - lower)),
            "median_interval_width": float(np.median(upper - lower))}
