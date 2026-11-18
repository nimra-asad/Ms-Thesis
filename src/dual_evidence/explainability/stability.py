"""Local TreeSHAP explanations and deterministic perturbation stability."""
from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np
from scipy.stats import spearmanr

from dual_evidence.forecasting.classical import LightGBMForecaster, _features
from dual_evidence.forecasting.windows import WindowSet


def feature_names(weather_names: tuple[str, ...]) -> list[str]:
    return ([f"previous_day_slot_{i+1}" for i in range(48)] +
            [f"previous_week_slot_{i+1}" for i in range(48)] +
            ["lag_1", "lag_2", "lag_48", "lag_336", "mean_24h", "std_24h",
             "mean_7d", "std_7d"] + list(weather_names) +
            ["future_time_sin", "future_time_cos", "future_weekday", "future_weekend",
             "future_month", "household_code", "city_code"])


def _spearman_rank(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0 if np.allclose(a, b) else 0.0
    value = spearmanr(a, b).statistic
    return float(np.nan_to_num(value, nan=0.0))


def explain_and_measure_stability(model: LightGBMForecaster, windows: WindowSet,
                                  perturbations: int = 2, seed: int = 42):
    """Return per-window stability, global rankings, and local lead-24 explanations.

    LightGBM's native `pred_contrib=True` computes exact TreeSHAP contributions.
    Two small perturbations are used: 1% demand noise and 0.1 standard-deviation
    weather noise. The unmodified forecasts used for evaluation are untouched.
    """
    if model.models is None:
        raise RuntimeError("forecaster must be fitted before explaining")
    n = len(windows)
    if n == 0:
        return np.empty(0), [], []
    rng = np.random.default_rng(seed)
    leads = (0, 11, 23, 35, 47)
    local_scores = np.zeros(n, dtype=np.float64)
    score_counts = np.zeros(n, dtype=np.int32)
    all_abs = []
    local_output: list[list[dict]] = [[] for _ in range(n)]
    names = feature_names(windows.feature_names)
    wx_scale = np.nanstd(windows.weather, axis=0) if windows.weather.shape[1] else np.array([])
    for lead in leads:
        logging.getLogger(__name__).info("TreeSHAP stability lead %d/48 (%d rows)", lead + 1, n)
        base_x = _features(windows, lead)
        base = model.models[lead].booster_.predict(base_x, pred_contrib=True)[:, :-1]
        all_abs.append(np.abs(base))
        if lead == 23:
            for i, contrib in enumerate(base):
                top = np.argsort(np.abs(contrib))[-10:][::-1]
                local_output[i] = [{"feature": names[j], "shap_value": float(contrib[j])}
                                   for j in top]
        for repeat in range(perturbations):
            hist = windows.history.copy()
            hist += rng.normal(0.0, np.maximum(np.abs(hist) * .01, .01), size=hist.shape).astype(np.float32)
            hist = np.maximum(hist, 0.0)
            weather = windows.weather.copy()
            if weather.shape[1]:
                weather += rng.normal(0.0, np.maximum(wx_scale * .10, .01), size=weather.shape).astype(np.float32)
            perturbed = replace(windows, history=hist, weather=weather)
            changed = model.models[lead].booster_.predict(
                _features(perturbed, lead), pred_contrib=True)[:, :-1]
            for i in range(n):
                local_scores[i] += _spearman_rank(np.abs(base[i]), np.abs(changed[i]))
                score_counts[i] += 1
    local_scores /= np.maximum(score_counts, 1)
    mean_abs = np.mean(np.concatenate(all_abs, axis=0), axis=0)
    ranking = sorted(({"feature": names[i], "mean_abs_shap": float(value)}
                      for i, value in enumerate(mean_abs)),
                     key=lambda row: row["mean_abs_shap"], reverse=True)
    return local_scores.astype(np.float32), ranking, local_output
