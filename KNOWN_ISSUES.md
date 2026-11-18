# Known Issues and Research Assumptions

## Completed work

- The complete local pipeline has run on the downloaded REWD-P dataset: four forecasting approaches, split-conformal intervals, TreeSHAP stability analysis, four controller configurations, simulated CVXPY demand response, sensitivity analysis, and report/figure generation. See [experiment_results.md](reports/experiment_results.md).
- The source archive has 69 energy CSV entries but only 59 distinct household series; ten entries are byte-identical copies. Metadata lists House1–House59, and there is no unique House60 source file. The analysis uses the 59 available homes and does not fabricate data.
- The full audit and preprocessing completed. Data quality counts and preprocessing rules are documented in [DEVELOPMENT_PROGRESS.md](DEVELOPMENT_PROGRESS.md) and the generated audit files.

## Assumptions requiring interpretation

- The synopsis does not specify split proportions, so chronological 70/10/10/10 train/validation/calibration/test splits are configured.
- Tariffs, 20% flexible-load share, source hours, and eligible shifting window are illustrative simulation assumptions, not utility tariffs or appliance measurements.
- Weather is aligned from the latest observed past value at each forecast origin; future realized weather is not treated as known.
- Exact zero readings are retained and long runs are flagged. The sensitivity report excludes flagged test windows for a comparison, but does not mask them during training or retrain a second model.
- Harmful action is currently defined as an increase in actual electricity cost. Peak change is reported separately.

## Demand-response comparison pending

- The first DR run's regret is invalid: the forecast-based schedule and perfect-information oracle used different feasible capacity limits. Do not interpret the recorded regret values as valid results.
- Approval was requested to align the oracle with the forecast-based schedule's feasible limits, keeping the cost-only objective. Once approved, the DR results must be rerun and the report updated. Peak reduction can remain negative under cost-only optimization; adding a peak penalty would change the research objective and needs separate approval.

## Runtime and optional components

- Chronos-2 weights are downloaded on first use. On this Intel Mac, the run uses one Torch CPU thread to avoid an OpenMP stall. The weights are cached for offline reruns.
- The optional hosted LLM narrative provider is disabled. Local Qwen3-1.7B tool orchestration ran once over cached outputs; numeric ACT/VERIFY/ABSTAIN and DR values remain Python-computed (`reports/chapter4.md`).
- Git commits could not be created because `.git` is read-only in this workspace.
