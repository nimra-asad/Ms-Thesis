"""Run the LLM coordinator over cached test evidence without retraining models.

The LLM selects the next approved workflow tool. All numerical decisions and
simulation outputs are computed by the existing deterministic Python modules.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from dual_evidence.agents.orchestrator import configured_coordinator
from dual_evidence.agents.workflow import (
    COORDINATOR_SYSTEM_PROMPT, TOOL_SCHEMAS, DualEvidenceWorkflow, legal_mutating_sequence)
from dual_evidence.logging_utils import configure_logging

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    load_dotenv(ROOT / ".env")
    configure_logging(ROOT / "logs")
    if not os.getenv("LLM_API_KEY") and not os.getenv("LLM_BASE_URL"):
        raise SystemExit("Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL in .env first.")

    workflow = DualEvidenceWorkflow.from_cached_outputs(ROOT)
    coordinator = configured_coordinator(workflow.tool_map())
    if coordinator.provider is None:
        raise SystemExit("LLM provider was not configured.")
    model = os.getenv("LLM_MODEL", "unknown")
    started = time.perf_counter()
    result = coordinator.orchestrate([
        {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Coordinate the Dual-Evidence experiment using cached trained-model "
            "outputs. Decide the next stage from tool results. Complete X, VERIFY "
            "if needed, ABSTAIN locking, ACT-only simulation, and finalize."
        )},
    ], TOOL_SCHEMAS, max_tool_calls=12)
    if not workflow.dr_complete:
        raise RuntimeError("LLM ended before completing all required workflow stages")
    elapsed_seconds = time.perf_counter() - started
    actual_order = [item["tool"] for item in result["tool_calls"]]
    verify_count = workflow.summary["initial_routes"]["VERIFY"]
    legal = legal_mutating_sequence(actual_order, verify_count)
    reference_path = RESULTS / "demand_response_patchtst_joint.csv"
    comparison = {}
    if reference_path.exists() and (RESULTS / "demand_response_agentic.csv").exists():
        import pandas as pd
        agentic = pd.read_csv(RESULTS / "demand_response_agentic.csv")
        reference = pd.read_csv(reference_path)
        dual = reference[reference.config == "dual_evidence_patchtst"] if "config" in reference.columns else reference
        comparison = {
            "agentic_dr_cases": int(len(agentic)),
            "reference_dual_evidence_dr_cases": int(len(dual)),
            "agentic_mean_cost_saving_pkr": float(agentic.cost_saving_pkr.mean()) if len(agentic) else 0.0,
            "reference_mean_cost_saving_pkr": float(dual.cost_saving_pkr.mean()) if len(dual) else 0.0,
        }
    output = {
        "provider": os.getenv("LLM_PROVIDER", "openai_compatible"),
        "model": model,
        "local_endpoint": os.getenv("LLM_BASE_URL", ""),
        "forecast_models_retrained": False,
        "numerical_policy_owner": "deterministic Python modules",
        "tool_order": actual_order,
        "legal_stage_sequence": legal,
        "model_turns": result["model_turns"],
        "elapsed_seconds": elapsed_seconds,
        "tool_calls": result["tool_calls"],
        "workflow": workflow.summary,
        "non_agentic_comparison": comparison,
        "agent_final_response": result["content"],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "agentic_orchestration.json").write_text(json.dumps(output, indent=2, default=str))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/agentic_tool_calls.json").write_text(json.dumps(result["tool_calls"], indent=2, default=str))
    workflow_summary = workflow.summary
    routes = workflow_summary["initial_routes"]
    verify = workflow_summary["verification"]
    abstain = workflow_summary["abstain"]
    demand = workflow_summary["demand_response"]
    match = workflow_summary.get("matches_deterministic_reference", False)
    report = [
        "# LLM Tool Orchestration Trial", "",
        f"- Provider/runtime: `{output['provider']}`; model: `{model}`.",
        f"- Model turns: {result['model_turns']}; approved tool calls: {len(result['tool_calls'])}; elapsed: {elapsed_seconds:.1f} seconds.",
        f"- Tool sequence: `{' -> '.join(actual_order)}`.",
        f"- Legal stage sequence (X before VERIFY/ABSTAIN; ACT simulation after both; finalize last): {legal}.",
        f"- Initial X-stage routes: ACT {routes['ACT']}, VERIFY {routes['VERIFY']}, ABSTAIN {routes['ABSTAIN']}.",
        f"- VERIFY: {verify['consistency_passed']}/{verify['verify_cases']} consistency checks passed; {verify['moved_to_act']} proceeded to ACT and {verify['moved_to_abstain']} to ABSTAIN.",
        f"- ABSTAIN lock: {abstain['locked_abstain_cases']} cases excluded from optimization.",
        f"- ACT-only demand response: {demand['evaluated_act_cases']} sampled cases; controller decisions match saved deterministic reference: {match}.",
        "",
        "The LLM selected among approved tools. Python retained ownership of forecast, explanation, threshold, controller, verification, and optimization calculations. Existing trained model outputs were reused; this trial did not retrain forecasting models. This is one workflow run, not a general agent-reliability evaluation.",
        "",
    ]
    if comparison:
        report.extend([
            "## Comparison with the non-agentic Dual-Evidence run", "",
            f"- Agentic sampled ACT cases: {comparison['agentic_dr_cases']}; non-agentic Dual-Evidence sampled ACT cases: {comparison['reference_dual_evidence_dr_cases']}.",
            f"- Mean cost saving (PKR/case): agentic {comparison['agentic_mean_cost_saving_pkr']:.4f}; non-agentic {comparison['reference_mean_cost_saving_pkr']:.4f}.",
            "",
        ])
    (ROOT / "reports/agentic_orchestration_results.md").write_text("\n".join(report), encoding="utf-8")
    logging.info("Agentic orchestration complete; legal_sequence=%s", legal)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
