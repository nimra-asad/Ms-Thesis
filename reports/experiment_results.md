# Experimental Results

All rows below are calculated from the downloaded REWD-P source data. Test forecasts are 48 half-hour steps; conformal intervals use a separate calibration split. The source archive contains 59 unique households; the synopsis says 60. Long zero runs are preserved and flagged.

**Demand-response caveat:** Numeric regret is withheld because the forecast schedule and prior oracle did not share a common feasible set. The updated optimizer includes a forecast peak penalty, but realized peak increases remain possible under forecast error. The penalty coefficient was evaluated at one value only; no sensitivity sweep was run.

## Forecasting and interval results

| Model | MAE kW | RMSE kW | WAPE % | PICP | Mean width kW |
|---|---:|---:|---:|---:|---:|
| seasonal_naive | 0.3057 | 0.6211 | 46.35 | 0.940 | 2.4278 |
| lightgbm | 0.2791 | 0.5007 | 42.31 | 0.945 | 1.9532 |
| patchtst | 0.2486 | 0.4909 | 37.68 | 0.945 | 1.9095 |
| chronos2 | 0.2269 | 0.4821 | 34.40 | 0.946 | 1.8806 |

## Long-zero-run sensitivity

21 of 994 test windows intersect a long-zero flag. The table compares all test windows with the subset excluding those windows; this is an evaluation-only sensitivity check, with the training policy unchanged.

| Model | All-test MAE | Excluding flagged-window MAE | Flagged-window MAE |
|---|---:|---:|---:|
| seasonal_naive | 0.3057 | 0.3123 | 0.0002 |
| lightgbm | 0.2791 | 0.2842 | 0.0438 |
| patchtst | 0.2486 | 0.2539 | 0.0003 |
| chronos2 | 0.2269 | 0.2318 | 0.0001 |

## Updated controller and demand-response results

The controller and DR outcomes below supersede the initial Seasonal-Naive-verifier / cost-only run. They were recomputed from saved test forecasts with PatchTST as the secondary verifier and a forecast-side objective combining illustrative energy cost with a 50 PKR/kW peak penalty. No forecasting model was retrained.

| Configuration | ACT coverage | Cost increase rate | Peak increase rate | Mean cost saved (PKR/case) | Mean peak reduction (kW) | DR cases |
|---|---:|---:|---:|---:|---:|---:|
| ungated | 1.000 | 0.000 | 0.455 | 17.45 | 0.0000 | 143 |
| explanation_only | 0.734 | 0.000 | 0.495 | 19.59 | -0.0015 | 101 |
| dual_evidence_patchtst | 0.333 | 0.000 | 0.438 | 31.63 | -0.0345 | 48 |
| uncertainty_only | 0.202 | 0.000 | 0.345 | 36.72 | -0.0168 | 29 |

PatchTST passed the consistency tolerance in 406 of 459 VERIFY cases. The optimizer penalized the forecast-side maximum demand, but realized mean peak reduction remained near zero or negative; cost and peak effects must be interpreted separately. The one-in-seven simulation is illustrative, and only one peak-penalty coefficient was evaluated. Numeric regret is withheld because a common feasible oracle was not evaluated.

## Matched-coverage test risk–coverage results

At each predeclared operating point, both policies select the same number of cases based on evidence scores only; test outcomes do not determine selection. These are retrospective risk–coverage results, not frozen-threshold operational coverage estimates.

| Coverage | Origins per policy | Uncertainty-only MAE | Dual-evidence MAE | Uncertainty-only RMSE | Dual-evidence RMSE |
|---:|---:|---:|---:|---:|---:|
| 10% | 99 | 0.6597 | 0.4223 | 0.8900 | 0.6323 |
| 20% | 199 | 0.5342 | 0.4389 | 0.7514 | 0.6558 |
| 30% | 298 | 0.4895 | 0.4520 | 0.7127 | 0.6615 |
| 40% | 398 | 0.4320 | 0.4259 | 0.6527 | 0.6385 |
| 50% | 497 | 0.3971 | 0.3971 | 0.6162 | 0.6162 |

At 20.02% realized test coverage (199 origins), dual-evidence selection lowered LightGBM MAE by 0.0954 kW versus uncertainty-only. The conditional household-cluster bootstrap 95% interval was [-0.1841, -0.0415] kW across 47 households. This supports a descriptive ranking result on the held-out test set, not a causal estimate of SHAP value or proof of operational benefit. See `reports/matched_coverage_test.md` and `reports/patchtst_joint_dr_results.md` for methodology and full caveats.

## Execution status

| Model | Status |
|---|---|
| seasonal_naive | complete: executed |
| lightgbm | complete: executed |
| patchtst | complete: executed |
| chronos2 | complete: amazon/chronos-2 |

Tariffs and flexible-load share are illustrative simulation assumptions, not a utility tariff or measured appliance telemetry. DR optimization executes only for ACT decisions. A weekday fixed tariff is used throughout; no real appliance commands are issued.

Primary comparison is Dual-Evidence versus Uncertainty-Only. The Uncertainty-Only threshold is chosen on validation data to match Dual-Evidence validation coverage; test coverage is reported as observed.

## Agentic coordinator trial

A local Qwen3-1.7B coordinator selected Dual-Evidence workflow tools over cached outputs (no retraining). Sequence: inspect → X → VERIFY → ABSTAIN lock → ACT simulation → finalize (7 model turns, 35.0 s). Controller decisions and 48-case Dual-Evidence DR cost savings matched the non-agentic PatchTST-joint artifacts exactly. See `reports/chapter4.md` and `reports/agentic_orchestration_results.md`. This is orchestration evidence, not a forecast-accuracy gain.
