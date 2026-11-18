"""Secondary forecasting consistency check used only for VERIFY cases."""
from __future__ import annotations

import numpy as np


def verify_forecast(primary: np.ndarray, secondary: np.ndarray,
                    relative_tolerance: float = .25) -> dict:
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    if primary.shape != secondary.shape or primary.size == 0:
        raise ValueError("verification forecasts must be non-empty and shape matched")
    denom = np.maximum(np.abs(primary), 0.05)
    relative_error = float(np.mean(np.abs(primary - secondary) / denom))
    return {"agrees": relative_error <= relative_tolerance,
            "mean_relative_difference": relative_error,
            "tolerance": relative_tolerance}
