# LLM Tool Orchestration Trial

- Provider/runtime: `llama.cpp`; model: `Qwen3-1.7B-Q4_K_M`.
- Model turns: 7; approved tool calls: 6; elapsed: 35.0 seconds.
- Tool sequence: `inspect_workflow_state -> run_x_stage -> run_verify_stage -> apply_abstain_policy -> simulate_act_cases -> finalize_run`.
- Legal stage sequence (X before VERIFY/ABSTAIN; ACT simulation after both; finalize last): True.
- Initial X-stage routes: ACT 289, VERIFY 459, ABSTAIN 246.
- VERIFY: 406/459 consistency checks passed; 42 proceeded to ACT and 417 to ABSTAIN.
- ABSTAIN lock: 663 cases excluded from optimization.
- ACT-only demand response: 48 sampled cases; controller decisions match saved deterministic reference: True.

The LLM selected among approved tools. Python retained ownership of forecast, explanation, threshold, controller, verification, and optimization calculations. Existing trained model outputs were reused; this trial did not retrain forecasting models. This is one workflow run, not a general agent-reliability evaluation.

## Comparison with the non-agentic Dual-Evidence run

- Agentic sampled ACT cases: 48; non-agentic Dual-Evidence sampled ACT cases: 48.
- Mean cost saving (PKR/case): agentic 31.6296; non-agentic 31.6296.
