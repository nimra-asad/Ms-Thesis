"""Deterministic Dual-Evidence workflow tools invoked by the LLM coordinator.

The coordinator may choose the next stage. Numeric ACT/VERIFY/ABSTAIN routing,
PatchTST consistency, and demand-response simulation remain Python-owned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from dual_evidence.controller.decision import Thresholds, decide
from dual_evidence.optimization.demand_response import optimize_day_ahead
from dual_evidence.verification.secondary import verify_forecast


@dataclass
class DualEvidenceWorkflow:
    data: pd.DataFrame
    primary: np.ndarray
    secondary: np.ndarray
    thresholds: Thresholds
    flexible_share: float = 0.20
    peak_penalty_pkr_per_kw: float = 50.0
    dr_stride_days: int = 7
    results_dir: Path | None = None
    x_complete: bool = False
    verified: bool = False
    abstain_locked: bool = False
    dr_complete: bool = False
    summary: dict = field(default_factory=dict)

    @classmethod
    def from_cached_outputs(cls, root: Path) -> "DualEvidenceWorkflow":
        results = root / "results"
        decision_log = pd.read_csv(results / "decision_log.csv")
        forecast_rows = pd.read_csv(results / "test_forecasts.csv")
        grouped = (forecast_rows.groupby(["origin", "household_id_code"], sort=False)
            .agg(actual=("actual_kw", list), lightgbm=("lightgbm_kw", list))
            .reset_index().rename(columns={"household_id_code": "household_code"}))
        for frame in (decision_log, grouped):
            frame["origin"] = pd.to_datetime(frame.origin).dt.strftime("%Y-%m-%d %H:%M:%S")
        data = decision_log.merge(grouped, on=["origin", "household_code"],
                                  validate="one_to_one").reset_index(drop=True)
        with np.load(results / "forecast_cache_lightgbm.npz") as cache:
            primary = cache["test"].copy()
        with np.load(results / "forecast_cache_patchtst.npz") as cache:
            secondary = cache["test"].copy()
        thresholds = Thresholds(**json.loads((results / "decision_thresholds.json").read_text())["dual_evidence"])
        experiment = yaml.safe_load((root / "configs/experiment.yaml").read_text())
        return cls(data=data, primary=primary, secondary=secondary, thresholds=thresholds,
                   flexible_share=float(experiment.get("dr_flexible_share", .20)),
                   peak_penalty_pkr_per_kw=float(experiment.get("dr_peak_penalty_pkr_per_kw", 50.0)),
                   dr_stride_days=int(experiment.get("dr_evaluation_stride_days", 7)),
                   results_dir=results)

    def allowed_next_tools(self) -> list[str]:
        allowed = ["inspect_workflow_state"]
        if not self.x_complete:
            allowed.append("run_x_stage")
            return allowed
        if not self.verified:
            allowed.append("run_verify_stage")
        if not self.abstain_locked:
            allowed.append("apply_abstain_policy")
        if self.verified and self.abstain_locked and not self.dr_complete:
            allowed.append("simulate_act_cases")
        if self.verified and self.abstain_locked and self.dr_complete:
            allowed.append("finalize_run")
        return allowed

    def inspect_workflow_state(self) -> dict:
        routes = self.summary.get("initial_routes")
        return {
            "x_stage_complete": self.x_complete,
            "verify_stage_complete": self.verified,
            "abstain_stage_complete": self.abstain_locked,
            "act_simulation_complete": self.dr_complete,
            "initial_routes": routes,
            "pending_verify_cases": None if routes is None else int(routes["VERIFY"]),
            "allowed_next_tools": self.allowed_next_tools(),
            "policy_owner": "deterministic Python modules",
        }

    def run_x_stage(self) -> dict:
        if self.x_complete:
            raise RuntimeError("X stage was already evaluated")
        self.data["initial_dual_decision"] = [
            decide(float(row.normalized_interval_width), float(row.explanation_stability),
                   self.thresholds)["decision"]
            for row in self.data.itertuples()
        ]
        counts = self.data.initial_dual_decision.value_counts().to_dict()
        self.x_complete = True
        self.summary["initial_routes"] = {key: int(counts.get(key, 0)) for key in ("ACT", "VERIFY", "ABSTAIN")}
        return {
            "stage": "X",
            "test_origins": int(len(self.data)),
            "initial_routes": self.summary["initial_routes"],
            "interpretation": (
                "ACT cases may later enter simulation; VERIFY cases require PatchTST "
                "consistency; ABSTAIN cases must never be optimized."),
            "allowed_next_tools": self.allowed_next_tools(),
        }

    def run_verify_stage(self) -> dict:
        if not self.x_complete:
            raise RuntimeError("Run X stage before verification")
        if self.verified:
            raise RuntimeError("Verification stage was already run")
        pending = np.flatnonzero(self.data.initial_dual_decision.to_numpy() == "VERIFY")
        final_decisions, relative_differences, agrees, reasons = [], [], [], []
        for index, row in self.data.iterrows():
            check = (verify_forecast(self.primary[index], self.secondary[index],
                                     self.thresholds.verification_relative_tolerance)
                     if row.initial_dual_decision == "VERIFY" else None)
            final = decide(float(row.normalized_interval_width), float(row.explanation_stability),
                           self.thresholds, check["agrees"] if check is not None else None)
            final_decisions.append(final["decision"])
            reasons.append(final["reason"])
            relative_differences.append(None if check is None else check["mean_relative_difference"])
            agrees.append(None if check is None else check["agrees"])
        self.data["dual_decision"] = final_decisions
        self.data["dual_reason"] = reasons
        self.data["verification_relative_difference"] = relative_differences
        self.data["verification_agrees"] = agrees
        verify_mask = self.data.initial_dual_decision.to_numpy() == "VERIFY"
        verification_passed = int(sum(bool(value) for value in self.data.loc[verify_mask, "verification_agrees"]))
        verify_act = int(((self.data.initial_dual_decision == "VERIFY")
                          & (self.data.dual_decision == "ACT")).sum())
        verify_abstain = int(((self.data.initial_dual_decision == "VERIFY")
                              & (self.data.dual_decision == "ABSTAIN")).sum())
        self.verified = True
        self.summary["verification"] = {
            "verify_cases": int(len(pending)),
            "consistency_passed": verification_passed,
            "moved_to_act": verify_act,
            "moved_to_abstain": verify_abstain,
            "final_act_total": int((self.data.dual_decision == "ACT").sum()),
            "final_abstain_total": int((self.data.dual_decision == "ABSTAIN").sum()),
        }
        if self.results_dir is not None:
            self.data.to_csv(self.results_dir / "decision_log_agentic.csv", index=False)
        payload = dict(self.summary["verification"])
        payload["stage"] = "VERIFY"
        payload["allowed_next_tools"] = self.allowed_next_tools()
        return payload

    def apply_abstain_policy(self) -> dict:
        if not self.x_complete:
            raise RuntimeError("Run X stage before abstention handling")
        if self.abstain_locked:
            raise RuntimeError("Abstain policy was already applied")
        if self.verified:
            column = "dual_decision"
        else:
            column = "initial_dual_decision"
        abstain = int((self.data[column] == "ABSTAIN").sum())
        act = int((self.data[column] == "ACT").sum())
        verify = int((self.data[column] == "VERIFY").sum()) if column == "initial_dual_decision" else 0
        self.abstain_locked = True
        self.summary["abstain"] = {
            "locked_abstain_cases": abstain,
            "act_candidates_at_lock": act,
            "unresolved_verify_at_lock": verify,
            "optimization_permitted_for_abstain": False,
        }
        payload = dict(self.summary["abstain"])
        payload["stage"] = "ABSTAIN"
        payload["allowed_next_tools"] = self.allowed_next_tools()
        return payload

    def simulate_act_cases(self) -> dict:
        if not self.verified:
            raise RuntimeError("Resolve VERIFY cases before demand-response simulation")
        if not self.abstain_locked:
            raise RuntimeError("Lock the ABSTAIN policy before demand-response simulation")
        if self.dr_complete:
            raise RuntimeError("Demand-response stage was already run")
        rows = []
        skipped_abstain = 0
        for _, row in self.data.iterrows():
            if row.dual_decision == "ABSTAIN":
                skipped_abstain += 1
                continue
            if row.dual_decision != "ACT":
                raise RuntimeError("Unresolved non-ACT/non-ABSTAIN case reached simulation")
            if pd.Timestamp(row.origin).date().toordinal() % self.dr_stride_days != 0:
                continue
            result = optimize_day_ahead(row.lightgbm, row.actual,
                flexible_share=self.flexible_share,
                peak_penalty_pkr_per_kw=self.peak_penalty_pkr_per_kw)
            rows.append({"origin": row.origin, "household_code": int(row.household_code),
                         **{key: value for key, value in result.items()
                            if key not in ("scheduled_flexible_kw", "optimized_actual_kw")}})
        demand = pd.DataFrame(rows)
        if self.results_dir is not None:
            demand.to_csv(self.results_dir / "demand_response_agentic.csv", index=False)
        self.dr_complete = True
        self.summary["demand_response"] = {
            "evaluated_act_cases": int(len(demand)),
            "skipped_abstain_cases": int(skipped_abstain),
            "mean_cost_saving_pkr": float(demand.cost_saving_pkr.mean()) if len(demand) else 0.0,
            "mean_peak_reduction_kw": float(demand.peak_reduction_kw.mean()) if len(demand) else 0.0,
            "cost_increase_rate": float(demand.cost_increase.mean()) if len(demand) else 0.0,
            "peak_increase_rate": float(demand.peak_increase.mean()) if len(demand) else 0.0,
        }
        payload = dict(self.summary["demand_response"])
        payload["stage"] = "ACT_SIMULATION"
        payload["allowed_next_tools"] = self.allowed_next_tools()
        return payload

    def finalize_run(self) -> dict:
        if not (self.x_complete and self.verified and self.abstain_locked and self.dr_complete):
            raise RuntimeError("Cannot finalize until X, VERIFY, ABSTAIN, and ACT simulation complete")
        reference_path = None if self.results_dir is None else self.results_dir / "decision_log_patchtst.csv"
        if reference_path is not None and reference_path.exists():
            reference = pd.read_csv(reference_path)
            self.summary["matches_deterministic_reference"] = bool(
                len(reference) == len(self.data)
                and (reference.dual_decision.astype(str).to_numpy()
                     == self.data.dual_decision.astype(str).to_numpy()).all())
        return {"status": "complete", "stage": "FINALIZE", **self.summary}

    def tool_map(self) -> dict[str, Callable]:
        return {
            "inspect_workflow_state": self.inspect_workflow_state,
            "run_x_stage": self.run_x_stage,
            "run_verify_stage": self.run_verify_stage,
            "apply_abstain_policy": self.apply_abstain_policy,
            "simulate_act_cases": self.simulate_act_cases,
            "finalize_run": self.finalize_run,
        }


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "inspect_workflow_state",
        "description": "Inspect pending Dual-Evidence stages and which tools are currently allowed.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_x_stage",
        "description": "X stage: apply the frozen uncertainty-plus-SHAP controller and return ACT, VERIFY, and ABSTAIN counts.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "run_verify_stage",
        "description": "VERIFY stage: run the frozen PatchTST consistency check on VERIFY cases only, then re-apply Python ACT/ABSTAIN gates.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "apply_abstain_policy",
        "description": "ABSTAIN stage: lock uncertain cases so they cannot enter demand-response optimization.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "simulate_act_cases",
        "description": "Run the existing demand-response simulation only for final ACT cases. ABSTAIN cases must never be optimized.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "finalize_run",
        "description": "Return the completed workflow summary after X, VERIFY, ABSTAIN, and ACT simulation.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]

def legal_mutating_sequence(calls: list[str], verify_count: int) -> bool:
    """True when mutating tools obey X → (VERIFY ↔ ABSTAIN) → ACT simulation → finalize."""
    mutating = [name for name in calls if name != "inspect_workflow_state"]
    if "run_x_stage" not in mutating:
        return False
    prefix = mutating[:mutating.index("run_x_stage") + 1]
    if prefix != ["run_x_stage"]:
        return False
    remainder = [name for name in mutating if name != "run_x_stage"]
    if verify_count == 0:
        remainder = [name for name in remainder if name != "run_verify_stage"]
        if "apply_abstain_policy" not in remainder:
            return False
        tail_start = remainder.index("apply_abstain_policy") + 1
    else:
        if "run_verify_stage" not in remainder or "apply_abstain_policy" not in remainder:
            return False
        tail_start = max(remainder.index("run_verify_stage"),
                         remainder.index("apply_abstain_policy")) + 1
    return remainder[tail_start:] == ["simulate_act_cases", "finalize_run"]


COORDINATOR_SYSTEM_PROMPT = (
    "You are the workflow coordinator for a frozen Dual-Evidence research pipeline. "
    "You do not compute forecasts, SHAP values, intervals, or thresholds. "
    "Call exactly one tool per turn. Use inspect_workflow_state whenever you need "
    "pending counts or the current allowed_next_tools list. "
    "X stage (run_x_stage) must run before VERIFY or ABSTAIN. "
    "If X reports VERIFY > 0, you must call run_verify_stage before simulation. "
    "You must call apply_abstain_policy so abstaining cases never enter optimization. "
    "After VERIFY and ABSTAIN are complete, call simulate_act_cases, then finalize_run. "
    "You may inspect state between stages. Never invent numeric results."
)
