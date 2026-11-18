"""Leakage-aware daily 24-hour forecasting windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class WindowSet:
    history: np.ndarray
    weather: np.ndarray
    calendar: np.ndarray
    household: np.ndarray
    city: np.ndarray
    origin: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.y)


WEATHER_COLUMNS = (
    "weather_temperature", "weather_humidity", "weather_dew",
    "weather_precipitation", "weather_wind_speed", "weather_pressure",
    "weather_solar_radiation", "weather_uv_index", "weather_age_hours",
)


def time_boundaries(panel: pd.DataFrame,
                    ratios: Mapping[str, float] | None = None) -> dict[str, pd.Timestamp]:
    ratios = ratios or {"train": .70, "validation": .10, "calibration": .10, "test": .10}
    times = pd.Index(pd.to_datetime(panel["timestamp"]).dropna().unique()).sort_values()
    if len(times) < 10:
        raise ValueError("at least ten unique timestamps are required")
    c1 = round(len(times) * ratios["train"])
    c2 = round(len(times) * (ratios["train"] + ratios["validation"]))
    c3 = round(len(times) * (ratios["train"] + ratios["validation"] + ratios["calibration"]))
    return {"train_end": times[c1], "validation_end": times[c2],
            "calibration_end": times[c3]}


def build_windows(panel: pd.DataFrame, context_steps: int = 336,
                  horizon_steps: int = 48, stride_steps: int = 48,
                  ratios: Mapping[str, float] | None = None) -> dict[str, WindowSet]:
    """Build complete context/label windows; weather is the origin-time snapshot.

    Origins are midnight local timestamps at a daily stride. Each window must have
    a full context and 48 observed labels. No label or future weather enters X.
    """
    if context_steps < 48 or horizon_steps != 48 or stride_steps < 1:
        raise ValueError("expected context >= 48 and a 48-step daily horizon")
    boundaries = time_boundaries(panel, ratios)
    weather_cols = [c for c in WEATHER_COLUMNS if c in panel.columns]
    households = {value: index for index, value in enumerate(sorted(panel.household_id.unique()))}
    cities = {value: index for index, value in enumerate(sorted(panel.city.unique()))}
    all_rows: dict[str, list] = {name: [] for name in ("train", "validation", "calibration", "test")}
    frame = panel.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["household_id", "timestamp"])
    for home, group in frame.groupby("household_id", sort=True):
        group = group.reset_index(drop=True)
        stamps = pd.DatetimeIndex(group.timestamp)
        demand = pd.to_numeric(group.target_kw, errors="coerce").to_numpy(dtype=np.float32)
        wx = group[weather_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        wx = np.nan_to_num(wx, nan=0.0, posinf=0.0, neginf=0.0)
        for end in range(context_steps, len(group) - horizon_steps + 1, stride_steps):
            origin = stamps[end]
            if origin.hour != 0 or origin.minute != 0:
                continue
            history = demand[end-context_steps:end]
            labels = demand[end:end+horizon_steps]
            if not np.isfinite(history).all() or not np.isfinite(labels).all():
                continue
            if (stamps[end] - stamps[end-1]) != pd.Timedelta(minutes=30):
                continue
            label_end = stamps[end+horizon_steps-1]
            # Keep every 48-step label wholly inside its assigned chronological
            # partition; otherwise windows near a cutoff would leak future labels.
            if origin < boundaries["train_end"] and label_end < boundaries["train_end"]:
                split = "train"
            elif (origin >= boundaries["train_end"] and
                  label_end < boundaries["validation_end"]):
                split = "validation"
            elif (origin >= boundaries["validation_end"] and
                  label_end < boundaries["calibration_end"]):
                split = "calibration"
            elif origin >= boundaries["calibration_end"]:
                split = "test"
            else:
                continue
            date = origin
            calendar = np.array([date.dayofweek, date.month, int(date.dayofweek >= 5)], dtype=np.float32)
            row = (history.copy(), wx[end-1].copy(), calendar, households[home],
                   cities[group.city.iloc[0]], origin.to_datetime64(), labels.copy())
            all_rows[split].append(row)
    if not all_rows["train"]:
        raise ValueError("no complete training windows; inspect coverage and split settings")
    output = {}
    feature_names = tuple(weather_cols)
    for split, rows in all_rows.items():
        if not rows:
            output[split] = WindowSet(np.empty((0, context_steps), np.float32),
                                      np.empty((0, len(weather_cols)), np.float32),
                                      np.empty((0, 3), np.float32), np.empty(0, int),
                                      np.empty(0, int), np.empty(0, "datetime64[ns]"),
                                      np.empty((0, horizon_steps), np.float32), feature_names)
            continue
        values = list(zip(*rows))
        output[split] = WindowSet(np.stack(values[0]), np.stack(values[1]),
                                  np.stack(values[2]), np.asarray(values[3], dtype=np.int32),
                                  np.asarray(values[4], dtype=np.int32), np.asarray(values[5]),
                                  np.stack(values[6]), feature_names)
    return output
