"""CVXPY simulated load shifting; never sends commands to real appliances."""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def tariff_vector() -> np.ndarray:
    """Illustrative configurable Pakistan PKR/kWh TOU profile (not a utility tariff)."""
    hour = np.arange(48) / 2
    return np.where((hour >= 17) & (hour < 22), 60.0,
                    np.where((hour >= 7) & (hour < 17), 40.0, 28.0))


def optimize_day_ahead(forecast_kw, actual_kw, *, flexible_share: float = .20,
                       max_shift_multiplier: float = 2.0, tariffs=None,
                       peak_penalty_pkr_per_kw: float = 50.0) -> dict:
    """Shift simulated evening appliance energy into a wider feasible window.

    Flexible energy is 20% of forecast demand during 17:00-22:00. It can move
    within 08:00-23:30, is energy-conserving, and is capped per slot at twice the
    forecast demand. The forecast objective combines illustrative tariff cost
    with a configurable penalty on maximum half-hour demand. Realized cost and
    peak changes are measured separately because forecast error can reverse either.
    """
    forecast = np.asarray(forecast_kw, dtype=float).reshape(-1)
    actual = np.asarray(actual_kw, dtype=float).reshape(-1)
    if forecast.shape != (48,) or actual.shape != (48,) or not np.isfinite(forecast).all() or not np.isfinite(actual).all() or np.any(actual < 0):
        raise ValueError("forecast and actual must be finite 48-step kW profiles; actual demand must be non-negative")
    # Tree-based regression can emit small negative values outside the training
    # range. Consumption is physically non-negative, so clip only for DR planning;
    # score the original forecasts unchanged in forecast_metrics.
    forecast = np.maximum(forecast, 0.0)
    if not 0 <= flexible_share <= 1:
        raise ValueError("flexible_share must be in [0, 1]")
    if not np.isfinite(peak_penalty_pkr_per_kw) or peak_penalty_pkr_per_kw < 0:
        raise ValueError("peak_penalty_pkr_per_kw must be finite and non-negative")
    tariff = np.asarray(tariffs if tariffs is not None else tariff_vector(), dtype=float)
    if tariff.shape != (48,):
        raise ValueError("tariffs must contain 48 half-hour prices")
    hour = np.arange(48) / 2
    source_mask = (hour >= 17) & (hour < 22)
    allowed_mask = (hour >= 8) & (hour < 23.5)
    baseline_flexible = forecast * flexible_share * source_mask
    energy = float(baseline_flexible.sum())
    if energy <= 1e-9:
        return _no_shift(forecast, actual, tariff, peak_penalty_pkr_per_kw)
    schedule = cp.Variable(48, nonneg=True)
    constraints = [cp.sum(schedule) == energy,
                   schedule <= np.maximum(forecast * max_shift_multiplier, 0.05),
                   schedule[~allowed_mask] == 0]
    planned_load = forecast - baseline_flexible + schedule
    objective = cp.Minimize(
        cp.sum(cp.multiply(tariff, planned_load)) * .5
        + float(peak_penalty_pkr_per_kw) * cp.max(planned_load)
        + .01 * cp.norm1(schedule[1:] - schedule[:-1]))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL")
    except cp.SolverError:
        problem.solve(solver="SCS", eps=1e-4, max_iters=20_000)
    if schedule.value is None:
        raise RuntimeError(f"CVXPY optimization failed: {problem.status}")
    planned = np.maximum(np.asarray(schedule.value).ravel(), 0.0)
    planned *= energy / max(planned.sum(), 1e-9)
    actual_flexible_energy = float((actual * flexible_share * source_mask).sum())
    applied = planned * (actual_flexible_energy / energy)
    actual_source_flex = actual * flexible_share * source_mask
    post = actual - actual_source_flex + applied
    base_cost = float(np.dot(actual, tariff) * .5)
    post_cost = float(np.dot(post, tariff) * .5)
    peak_reduction = float(actual.max() - post.max())
    return {"status": problem.status,
            "forecast_energy_cost_pkr": float(np.dot(planned_load.value, tariff) * .5),
            "forecast_peak_kw": float(np.max(planned_load.value)),
            "peak_penalty_pkr_per_kw": float(peak_penalty_pkr_per_kw),
            "baseline_actual_cost_pkr": base_cost, "optimized_actual_cost_pkr": post_cost,
            "cost_saving_pkr": base_cost-post_cost,
            "cost_saving_pct": 100*(base_cost-post_cost)/base_cost if base_cost else 0.0,
            "peak_reduction_kw": peak_reduction,
            "cost_increase": bool(post_cost > base_cost + 1e-6),
            "peak_increase": bool(peak_reduction < -1e-6),
            "flex_energy_kwh": actual_flexible_energy*.5,
            "scheduled_flexible_kw": applied.tolist(), "optimized_actual_kw": post.tolist()}


def _no_shift(forecast, actual, tariff, peak_penalty_pkr_per_kw):
    cost = float(np.dot(actual, tariff)*.5)
    return {"status": "no_flexible_energy",
            "forecast_energy_cost_pkr": float(np.dot(forecast, tariff)*.5),
            "forecast_peak_kw": float(np.max(forecast)),
            "baseline_actual_cost_pkr": cost, "optimized_actual_cost_pkr": cost,
            "cost_saving_pkr": 0.0, "cost_saving_pct": 0.0,
            "peak_reduction_kw": 0.0,
            "peak_penalty_pkr_per_kw": float(peak_penalty_pkr_per_kw),
            "cost_increase": False, "peak_increase": False,
            "flex_energy_kwh": 0.0, "scheduled_flexible_kw": [0.0]*48,
            "optimized_actual_kw": actual.tolist()}
