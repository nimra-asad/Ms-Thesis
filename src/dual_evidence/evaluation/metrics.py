"""Forecast, selective-decision, and demand-response metrics."""
from __future__ import annotations

import numpy as np


def forecast_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.shape != p.shape or len(y) == 0:
        raise ValueError("forecast truth and predictions must have matching non-empty shapes")
    error = p - y
    denom = np.sum(np.abs(y))
    return {"mae_kw": float(np.mean(np.abs(error))),
            "rmse_kw": float(np.sqrt(np.mean(error ** 2))),
            "wape_pct": float(100 * np.sum(np.abs(error)) / denom) if denom else float("nan"),
            "bias_kw": float(np.mean(error))}


def selective_metrics(decisions, harmful, regrets) -> dict[str, float]:
    decisions = np.asarray(decisions, dtype=str)
    harmful = np.asarray(harmful, dtype=bool)
    regrets = np.asarray(regrets, dtype=float)
    if not len(decisions):
        return {"automation_coverage": float("nan"), "harmful_action_rate": float("nan"),
                "mean_regret": float("nan")}
    acted = decisions == "ACT"
    return {"automation_coverage": float(np.mean(acted)),
            "verify_rate": float(np.mean(decisions == "VERIFY")),
            "abstain_rate": float(np.mean(decisions == "ABSTAIN")),
            "harmful_action_rate": float(np.mean(harmful[acted])) if acted.any() else 0.0,
            "mean_regret": float(np.mean(regrets[acted])) if acted.any() else 0.0}
