import unittest

import numpy as np
import pandas as pd

from dual_evidence.agents.orchestrator import ResearchCoordinator, extract_tool_calls
from dual_evidence.agents.workflow import DualEvidenceWorkflow, legal_mutating_sequence
from dual_evidence.controller.decision import Thresholds


class FakeProvider:
    def __init__(self, names):
        self.names = list(names)
        self.calls = 0

    def complete(self, messages, tools=None):
        if not self.names:
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        name = self.names.pop(0)
        self.calls += 1
        return {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{self.calls}", "type": "function",
                            "function": {"name": name, "arguments": "{}"}}],
        }}]}


def _tiny_workflow() -> DualEvidenceWorkflow:
    data = pd.DataFrame({
        "origin": ["2024-10-13 00:00:00", "2024-10-14 00:00:00", "2024-10-15 00:00:00"],
        "household_code": [0, 0, 0],
        "normalized_interval_width": [0.10, 0.15, 0.90],
        "explanation_stability": [0.95, 0.55, 0.10],
        "actual": [[0.4] * 48, [0.4] * 48, [0.4] * 48],
        "lightgbm": [[0.4] * 48, [0.4] * 48, [0.4] * 48],
    })
    primary = np.full((3, 48), 0.4)
    secondary = np.full((3, 48), 0.41)
    thresholds = Thresholds(0.2, 0.7, 0.5, 0.4, 0.25)
    return DualEvidenceWorkflow(data=data, primary=primary, secondary=secondary,
                                thresholds=thresholds, dr_stride_days=1)


class AgenticOrchestrationTests(unittest.TestCase):
    def test_extract_tool_calls_parses_qwen_xml_fallback(self):
        message = {"content": '<tool_call>{"name": "run_x_stage", "arguments": {}}</tool_call>'}
        calls = extract_tool_calls(message)
        self.assertEqual(calls[0]["function"]["name"], "run_x_stage")

    def test_unapproved_tool_is_rejected(self):
        coordinator = ResearchCoordinator({"inspect_workflow_state": lambda: {}}, FakeProvider(["hack"]))
        with self.assertRaises(KeyError):
            coordinator.orchestrate([{"role": "user", "content": "go"}], [], max_tool_calls=2)

    def test_x_verify_abstain_gates(self):
        workflow = _tiny_workflow()
        self.assertEqual(workflow.inspect_workflow_state()["allowed_next_tools"],
                         ["inspect_workflow_state", "run_x_stage"])
        x_result = workflow.run_x_stage()
        self.assertEqual(x_result["initial_routes"], {"ACT": 1, "VERIFY": 1, "ABSTAIN": 1})
        with self.assertRaises(RuntimeError):
            workflow.simulate_act_cases()
        verify = workflow.run_verify_stage()
        self.assertEqual(verify["verify_cases"], 1)
        abstain = workflow.apply_abstain_policy()
        self.assertGreaterEqual(abstain["locked_abstain_cases"], 1)
        simulated = workflow.simulate_act_cases()
        self.assertEqual(simulated["skipped_abstain_cases"], 1)
        self.assertEqual(workflow.finalize_run()["status"], "complete")

    def test_coordinator_selects_stage_tools(self):
        workflow = _tiny_workflow()
        names = ["inspect_workflow_state", "run_x_stage", "run_verify_stage",
                 "apply_abstain_policy", "simulate_act_cases", "finalize_run"]
        coordinator = ResearchCoordinator(workflow.tool_map(), FakeProvider(names))
        result = coordinator.orchestrate([{"role": "user", "content": "coordinate"}], [],
                                         max_tool_calls=8)
        self.assertEqual([item["tool"] for item in result["tool_calls"]], names)
        self.assertTrue(workflow.dr_complete)

    def test_legal_sequence_allows_abstain_before_verify(self):
        order = ["inspect_workflow_state", "run_x_stage", "apply_abstain_policy",
                 "run_verify_stage", "simulate_act_cases", "finalize_run"]
        self.assertTrue(legal_mutating_sequence(order, verify_count=1))
        self.assertFalse(legal_mutating_sequence(
            ["run_verify_stage", "run_x_stage", "finalize_run"], verify_count=1))


if __name__ == "__main__":
    unittest.main()
