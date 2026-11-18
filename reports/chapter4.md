# Chapter 4 — Experimental Setup, Dual Evidence, and Agentic Orchestration

This chapter records implemented methods and measured outputs from the Dual-Evidence residential load-forecasting pipeline. Numeric tables that originate in executed Python modules are authoritative. The coordinator trial in §4.6 is a single local llama.cpp run, not a multi-seed agent benchmark.

## 4.1 Status of prior work (not re-run)

Forecasting, conformal intervals, TreeSHAP stability, and Dual-Evidence versus Uncertainty-only comparison were completed on the REWD-P panel before this orchestration layer. Cached artifacts under `results/` are reused; forecasting models are not retrained for the agent trial.

Executed test-set forecast metrics (994 origins, 48 half-hour steps):

| Model | MAE kW | RMSE kW | WAPE % | PICP |
|---|---:|---:|---:|---:|
| Seasonal Naive | 0.3057 | 0.6211 | 46.35 | 0.940 |
| LightGBM | 0.2791 | 0.5007 | 42.31 | 0.945 |
| PatchTST | 0.2486 | 0.4909 | 37.68 | 0.945 |
| Chronos-2 | 0.2269 | 0.4821 | 34.40 | 0.946 |

Frozen-threshold controller coverage on the test set, with PatchTST as the secondary verifier: ungated 100%, explanation-only 73.4%, Dual-Evidence 33.3%, Uncertainty-only 20.2%. The Uncertainty-only threshold was chosen on validation to match Dual-Evidence validation coverage. A separate matched-coverage diagnostic is in `reports/matched_coverage_test.md`.

## 4.2 LLM referenced by prior notes

Repository documentation already specified **Qwen3-1.7B** (GGUF `Qwen3-1.7B-Q4_K_M`), not a “Qwen 2.8” checkpoint. No Qwen 2.8 release is used in this project. The live local server alias at implementation time is `Qwen3-1.7B-Q4_K_M`.

## 4.3 Model selection for the coordinator

Candidates considered for a free, local, tool-calling coordinator on an Intel Mac CPU:

| Candidate | License / cost | Approx. size | Tool calling | Fit for this machine |
|---|---|---|---|---|
| Qwen3-1.7B Q4_K_M | Free weights; local inference | ~1.28 GB | Native OpenAI-style tools via llama.cpp `--jinja` | Selected |
| Qwen2.5-1.5B-Instruct | Free weights | ~1 GB | Strong templates, older family | Viable alternative |
| Llama 3.2 1B/3B Instruct | Free weights (community GGUF) | ~0.8–2 GB | Weaker small-model tool reliability | Heavier or less reliable |
| Hosted GPT-4o-mini | Provider charges | n/a | Strong | Rejected (not free, sends data off-box) |

**Selection:** Qwen3-1.7B Q4_K_M served by llama.cpp. Reasons: Apache-licensed weights, CPU-feasible quantization, measured native `tool_calls` on this host, no project data sent to a commercial API, and structured outputs that the Python allowlist can validate. Thinking/reasoning tokens are disabled for tool turns (`enable_thinking: false`) so the coordinator emits a function call instead of a chain-of-thought essay.

## 4.4 Agentic architecture

The coordinator is an OpenAI-compatible client (`src/dual_evidence/agents/orchestrator.py`) talking to `http://localhost:18080/v1`. Approved tools live in `DualEvidenceWorkflow` (`src/dual_evidence/agents/workflow.py`):

1. `inspect_workflow_state` — pending stages and currently allowed tools.
2. `run_x_stage` — frozen Dual-Evidence ACT/VERIFY/ABSTAIN routing (uncertainty width + SHAP stability).
3. `run_verify_stage` — PatchTST relative-error consistency on VERIFY cases only.
4. `apply_abstain_policy` — lock ABSTAIN cases out of demand-response.
5. `simulate_act_cases` — existing CVXPY simulation for sampled ACT cases.
6. `finalize_run` — summary and comparison with the saved non-agentic decision log.

The LLM chooses the next tool; it cannot change thresholds, SHAP scores, forecasts, or controller inequalities. Unapproved names are rejected. One tool is executed per model turn, with a bounded call budget.

```
LLM coordinator (Qwen3-1.7B)
        |  tool_calls
        v
Python allowlist  -->  X / VERIFY / ABSTAIN / ACT simulation
        |                 ^
        +---- cached LightGBM, PatchTST, SHAP, conformal artifacts
```

## 4.5 Implementation integration

- Existing `scripts/run_experiment.py` path is unchanged and still runs offline without an LLM.
- `scripts/run_agentic_orchestration.py` loads cached `results/` files only.
- Decision and DR CSVs from the agent trial are written as `decision_log_agentic.csv` and `demand_response_agentic.csv` so the prior PatchTST-joint artifacts remain auditable.
- Logging: coordinator turns go to `logs/pipeline.log`; tool traces to `logs/agentic_tool_calls.json`.

## 4.6 Experimental results (executed)

Command: `python scripts/run_agentic_orchestration.py` against the already-running local llama.cpp server (`Qwen3-1.7B-Q4_K_M` on `127.0.0.1:18080`). Forecast models were not retrained. Source traces: `results/agentic_orchestration.json`, `logs/agentic_tool_calls.json`, `reports/agentic_orchestration_results.md`.

| Quantity | Measured value |
|---|---|
| Model | Qwen3-1.7B-Q4_K_M (llama.cpp b12502) |
| Model turns | 7 |
| Approved tool calls | 6 |
| Wall time | 35.0 s |
| Tool sequence | inspect → X → VERIFY → ABSTAIN lock → ACT simulation → finalize |
| Legal stage sequence | True |
| X-stage routes (994 origins) | ACT 289, VERIFY 459, ABSTAIN 246 |
| PatchTST VERIFY checks passed | 406 / 459 |
| VERIFY → ACT / ABSTAIN | 42 / 417 |
| Final ACT / ABSTAIN | 331 / 663 |
| ABSTAIN cases excluded from DR | 663 |
| Sampled ACT DR cases | 48 |
| Dual-Evidence decisions match `decision_log_patchtst.csv` | True |
| Mean DR cost saving vs non-agentic Dual-Evidence | 31.6296 PKR/case on both runs |
| Mean DR peak reduction | −0.0345 kW (same as non-agentic Dual-Evidence) |

The coordinator inspected state first, then chose `run_x_stage`. After X reported 459 VERIFY cases it invoked verification rather than simulation. It then locked ABSTAIN before optimization. Python rejected any path that would optimize ABSTAIN or skip VERIFY. Numeric ACT/VERIFY/ABSTAIN labels and DR costs matched the prior non-agentic Dual-Evidence artifacts exactly; the LLM did not create a new operating point.

This does **not** show that the agent improved forecast accuracy or DR value. It shows that, on this one run, a local 1.7B model selected the legal tool sequence and that frozen Python policy reproduced the non-agentic Dual-Evidence outcomes.

## 4.7 Limitations and intended follow-up

- One local model, one workflow, one machine; not an agent benchmark.
- Stage choice is constrained by Python guards so a small model cannot skip VERIFY when VERIFY > 0.
- Demand-response remains a simulated schedule; no live appliance control.
- Numeric regret remains withheld from earlier DR work pending a common feasible oracle (see `KNOWN_ISSUES.md`).
