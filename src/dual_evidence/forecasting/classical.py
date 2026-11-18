"""Seasonal-naive and pooled LightGBM multi-horizon forecasting."""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from lightgbm import LGBMRegressor

from .windows import WindowSet


def seasonal_naive(windows: WindowSet) -> np.ndarray:
    """Repeat the previous 24-hour profile for the next day."""
    return windows.history[:, -48:].copy()


def _features(windows: WindowSet, lead: int) -> np.ndarray:
    history = windows.history
    last_day = history[:, -48:]
    last_week_day = history[:, -336:-288] if history.shape[1] >= 336 else last_day
    hour = (lead * 30) % 1440 / 30
    day_offset = (lead * 30) // 1440
    weekday = (windows.calendar[:, 0] + day_offset) % 7
    month = windows.calendar[:, 1]
    time = np.column_stack([
        np.sin(2*np.pi*hour/48) * np.ones(len(history)),
        np.cos(2*np.pi*hour/48) * np.ones(len(history)),
        weekday / 6.0, (weekday >= 5).astype(np.float32), month / 12.0,
        windows.household / max(1, int(windows.household.max(initial=0)) + 1),
        windows.city / max(1, int(windows.city.max(initial=0)) + 1),
    ])
    summaries = np.column_stack([
        history[:, -1], history[:, -2], history[:, -48], history[:, -336],
        last_day.mean(axis=1), last_day.std(axis=1),
        history[:, -336:].mean(axis=1), history[:, -336:].std(axis=1),
    ])
    return np.concatenate([last_day, last_week_day, summaries, windows.weather, time], axis=1).astype(np.float32)


@dataclass
class LightGBMForecaster:
    n_estimators: int = 300
    learning_rate: float = 0.04
    num_leaves: int = 31
    max_depth: int = -1
    random_state: int = 42
    n_jobs: int = 2
    models: list | None = None

    def fit(self, train: WindowSet) -> "LightGBMForecaster":
        self.models = []
        for lead in range(48):
            model = LGBMRegressor(
                objective="regression", n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                max_depth=self.max_depth, random_state=self.random_state + lead,
                n_jobs=self.n_jobs, verbosity=-1,
            )
            model.fit(_features(train, lead), train.y[:, lead])
            self.models.append(model)
            if (lead + 1) % 8 == 0:
                logging.getLogger(__name__).info("LightGBM trained %d/48 forecast leads", lead + 1)
        return self

    def predict(self, windows: WindowSet) -> np.ndarray:
        if self.models is None:
            raise RuntimeError("fit must be called before predict")
        if not len(windows):
            return np.empty((0, 48), dtype=np.float32)
        return np.column_stack([
            model.predict(_features(windows, lead)) for lead, model in enumerate(self.models)
        ]).astype(np.float32)

    def predict_contributions(self, windows: WindowSet, leads=(0, 11, 23, 35, 47)):
        if self.models is None:
            raise RuntimeError("fit must be called before explain")
        return {lead: self.models[lead].booster_.predict(
            _features(windows, lead), pred_contrib=True
        ) for lead in leads}
