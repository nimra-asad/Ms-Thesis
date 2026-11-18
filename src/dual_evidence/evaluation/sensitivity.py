"""Evaluation-only sensitivity to long exact-zero runs flagged in preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_evidence.evaluation.metrics import forecast_metrics
from dual_evidence.forecasting.windows import WindowSet


def zero_run_window_mask(panel: pd.DataFrame, windows: WindowSet) -> np.ndarray:
    """Mark forecasts whose 48 observed targets intersect a flagged zero run."""
    if "suspect_zero_run" not in panel:
        return np.zeros(len(windows), dtype=bool)
    homes = sorted(panel.household_id.unique())
    series = {}
    for code, home in enumerate(homes):
        group = panel.loc[panel.household_id == home].sort_values("timestamp")
        series[code] = (pd.to_datetime(group.timestamp).to_numpy(dtype="datetime64[ns]"),
                        group.suspect_zero_run.fillna(False).to_numpy(dtype=bool))
    result = np.zeros(len(windows), dtype=bool)
    for i, (code, origin) in enumerate(zip(windows.household, windows.origin)):
        timestamps, flags = series[int(code)]
        start = int(np.searchsorted(timestamps, origin, side="left"))
        result[i] = bool(flags[start:start+48].any())
    return result


def evaluate_zero_run_sensitivity(panel: pd.DataFrame, windows: WindowSet,
                                  forecasts: dict[str, np.ndarray]) -> dict:
    """Compare overall test scores with a diagnostic subset excluding flagged horizons.

    This is explicitly an evaluation sensitivity check; training data are unchanged.
    """
    flagged = zero_run_window_mask(panel, windows)
    result = {"method": "exclude any test forecast whose 48 labels intersect a flagged >=24h zero run",
              "training_policy": "unchanged; zeros retained and flagged",
              "test_windows_total": len(windows),
              "test_windows_intersecting_flagged_zero_run": int(flagged.sum()),
              "test_windows_excluded": int((~flagged).sum()),
              "flagged_window_rate": float(flagged.mean()) if len(flagged) else 0.0,
              "models": {}}
    for name, prediction in forecasts.items():
        if prediction is None:
            continue
        result["models"][name] = {
            "overall": forecast_metrics(windows.y, prediction),
            "exclude_flagged_zero_windows": (forecast_metrics(windows.y[~flagged], prediction[~flagged])
                                              if (~flagged).any() else None),
            "flagged_zero_windows_only": (forecast_metrics(windows.y[flagged], prediction[flagged])
                                           if flagged.any() else None),
        }
    return result
