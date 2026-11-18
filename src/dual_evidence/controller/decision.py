"""Deterministic dual-evidence ACT/VERIFY/ABSTAIN decision policy."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    uncertainty_act_max: float
    explanation_act_min: float
    uncertainty_verify_max: float
    explanation_verify_min: float
    verification_relative_tolerance: float = 0.25


def select_thresholds(validation_width: np.ndarray, validation_stability: np.ndarray,
                      uncertainty_quantile: float = .70,
                      explanation_quantile: float = .30,
                      verify_uncertainty_quantile: float = .90,
                      verify_explanation_quantile: float = .10,
                      verification_relative_tolerance: float = .25) -> Thresholds:
    width = np.asarray(validation_width, dtype=float)
    stability = np.asarray(validation_stability, dtype=float)
    valid = np.isfinite(width) & np.isfinite(stability)
    if not valid.any():
        raise ValueError("validation evidence is empty or non-finite")
    return Thresholds(float(np.quantile(width[valid], uncertainty_quantile)),
                      float(np.quantile(stability[valid], explanation_quantile)),
                      float(np.quantile(width[valid], verify_uncertainty_quantile)),
                      float(np.quantile(stability[valid], verify_explanation_quantile)),
                      verification_relative_tolerance)


def decide(uncertainty_width: float, explanation_stability: float,
           thresholds: Thresholds, verification_agrees: bool | None = None) -> dict:
    """Apply fixed thresholds. VERIFY can transition only after a passed check."""
    if not np.isfinite(uncertainty_width) or not np.isfinite(explanation_stability):
        return {"decision": "ABSTAIN", "reason": "non-finite reliability evidence"}
    uncertainty_ok = uncertainty_width <= thresholds.uncertainty_act_max
    explanation_ok = explanation_stability >= thresholds.explanation_act_min
    if uncertainty_ok and explanation_ok:
        return {"decision": "ACT", "reason": "both evidence thresholds passed"}
    verify_region = (uncertainty_width <= thresholds.uncertainty_verify_max and
                     explanation_stability >= thresholds.explanation_verify_min)
    if not verify_region:
        return {"decision": "ABSTAIN", "reason": "evidence outside verification region"}
    if verification_agrees is None:
        return {"decision": "VERIFY", "reason": "ambiguous evidence requires secondary check"}
    if verification_agrees and uncertainty_ok and explanation_stability >= thresholds.explanation_verify_min:
        return {"decision": "ACT", "reason": "secondary model agreed; uncertainty gate passed"}
    return {"decision": "ABSTAIN", "reason": "secondary check did not clear both evidence gates"}


def thresholds_as_dict(value: Thresholds) -> dict:
    return asdict(value)


def decide_all(widths: Iterable[float], stabilities: Iterable[float], thresholds: Thresholds):
    return [decide(float(w), float(s), thresholds) for w, s in zip(widths, stabilities)]
