# PatchTST Verification and Cost–Peak Demand-Response Results

This recomputation uses saved forecasts and frozen validation thresholds. PatchTST replaces Seasonal Naive as the secondary consistency verifier. Demand-response optimization minimizes forecast-based illustrative energy cost plus a 50 PKR/kW peak penalty. Cost and peak outcomes are reported separately; outcomes use the same deterministic one-in-seven test-date sample. No model was retrained. Numeric regret is withheld.

| Configuration | Test ACT coverage | ACT origins | Sampled DR cases | Mean cost saving (PKR) | Mean peak reduction (kW) | Cost increase rate | Peak increase rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| ungated | 1.000 | 994 | 143 | 17.45 | 0.0000 | 0.000 | 0.455 |
| uncertainty_only | 0.202 | 201 | 29 | 36.72 | -0.0168 | 0.000 | 0.345 |
| explanation_only | 0.734 | 730 | 101 | 19.59 | -0.0015 | 0.000 | 0.495 |
| dual_evidence_patchtst | 0.333 | 331 | 48 | 31.63 | -0.0345 | 0.000 | 0.438 |

PatchTST verification passed for 406 of 459 VERIFY cases. These figures are simulated and depend on the stated flexible-load assumptions and peak penalty.
