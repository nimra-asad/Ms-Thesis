# Matched-Coverage Test Comparison

This is a post-hoc risk--coverage comparison on the held-out test origins. The uncertainty-only policy ranks cases by normalized interval width. The dual-evidence policy ranks by the weaker normalized margin across interval width and SHAP stability, using ACT thresholds fixed on validation. For each predeclared coverage, both methods select the same number of test origins. Test outcomes are not used for ranking or selection.

| Test coverage | Selected origins per method | Uncertainty-only MAE (kW) | Dual-evidence MAE (kW) | Uncertainty-only RMSE (kW) | Dual-evidence RMSE (kW) |
|---:|---:|---:|---:|---:|---:|
| 10% | 99 | 0.6597 | 0.4223 | 0.8900 | 0.6323 |
| 20% | 199 | 0.5342 | 0.4389 | 0.7514 | 0.6558 |
| 30% | 298 | 0.4895 | 0.4520 | 0.7127 | 0.6615 |
| 40% | 398 | 0.4320 | 0.4259 | 0.6527 | 0.6385 |
| 50% | 497 | 0.3971 | 0.3971 | 0.6162 | 0.6162 |

At 20% target coverage (199 origins; realized 20.02%), dual-evidence MAE was 0.4389 kW versus 0.5342 kW for uncertainty-only. The dual-selected and uncertainty-selected sets overlap on 125 origins. Conditional household-cluster bootstrap (n=47 households) 95% interval for the difference (dual minus uncertainty MAE) was [-0.1841, -0.0415] kW.

This is evidence that the joint reliability score ranked lower-error forecasts more effectively at matched coverage in this test set. It is not a causal estimate of SHAP’s value, does not establish operational savings, and should be presented as a retrospective risk--coverage diagnostic alongside the frozen-threshold results.
