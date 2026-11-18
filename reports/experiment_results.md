# Experimental Results

All rows below are calculated from the downloaded REWD-P source data. Test forecasts are 48 half-hour steps; conformal intervals use a separate calibration split. The source archive contains 59 unique households; the synopsis says 60. Long zero runs are preserved and flagged.

**Demand-response caveat:** The current regret figures are invalid because the forecast-based schedule and perfect-information oracle used different feasible capacity limits. Do not interpret or cite those regret values. Oracle alignment is pending the user's research-design decision. The optimizer minimizes cost only; peak change is measured and may be negative (a peak increase). “Harmful action” currently means an increase in actual electricity cost.

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

## Selective controller and DR comparison

| Configuration | Automation | Harmful action | Cost saved PKR | Peak reduction kW | Regret PKR | DR cases |
|---|---:|---:|---:|---:|---:|---:|
| ungated | 1.000 | 0.000 | 21.17 | -0.0802 | -0.21 | 143 |
| explanation_only | 0.734 | 0.000 | 23.98 | -0.0970 | -0.17 | 101 |
| dual_evidence | 0.304 | 0.000 | 37.76 | -0.1751 | -0.07 | 42 |
| uncertainty_only | 0.202 | 0.000 | 42.55 | -0.1287 | -0.11 | 29 |

## Execution status

| Model | Status |
|---|---|
| seasonal_naive | complete: executed |
| lightgbm | complete: executed |
| patchtst | complete: executed |
| chronos2 | complete: amazon/chronos-2 |

Tariffs and flexible-load share are illustrative simulation assumptions, not a utility tariff or measured appliance telemetry. DR optimization executes only for ACT decisions. A weekday fixed tariff is used throughout; no real appliance commands are issued.

Primary comparison is Dual-Evidence versus Uncertainty-Only. The Uncertainty-Only threshold is chosen on validation data to match Dual-Evidence validation coverage; test coverage is reported as observed.
