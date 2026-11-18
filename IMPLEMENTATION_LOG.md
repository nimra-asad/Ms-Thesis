# Implementation Log

## 2026-11-18 — Initial project setup

- Inspected the empty workspace and local Python/Git availability.
- Added a Python 3.11–3.12 package definition, optional model extras, project configuration, environment template, and Git exclusions.
- Added CLI/logging entry points, data acquisition and archive inspection modules, and setup smoke tests.
- Ran `PYTHONPATH=src python3.11 -m unittest discover -s tests -v`: 2 passed. Ran compileall: passed.
- Added synthetic archive inspection and download-cache regression coverage; current regression result is 4 passed. Loopback HTTP testing failed with `PermissionError` under sandbox networking, so the network-dependent test was replaced by an offline cache-reuse check.
- Added archive path-traversal rejection; current regression result is 5 passed.
- Installed the Python 3.11 scientific base stack after requesting network access. The first resolver selected CVXPY 1.9.3 and hit a concurrent package-cache lock while building SCS; narrowed the project constraint to `<1.8`, installed CVXPY 1.7.5, and verified imports. `pip check` reports no broken requirements.
- Git initialization failed because `.git` is read-only per workspace permissions, so commits cannot be created in this checkout.
- Downloaded and extracted the official LUMS source data. Full archive inspection found 69 energy CSVs, 10 byte-identical duplicate Peshawar files, and 59 unique households; the metadata workbook also lists House1–House59. This conflicts with the synopsis/LUMS overview claim of 60. A user decision is pending.
- Ran full streaming audit over all 59 unique household files and six city weather files. Corrected the interval check so energy is checked against one minute and weather against one hour. Audit found 30,707,062 energy rows; 330,900 duplicate timestamp rows; 685,975 missing target rows; 1,244,047 zero target rows; 5,241 energy gaps over one minute; no invalid timestamps or negative target values. The six weather files each have 12,456 hourly rows with no gaps or out-of-order records.
- Final source audit now also lists the longest consecutive exact-zero and missing-target episodes in `reports/data_audit.md`: 102,176 minutes of exact zeros and 42,390 minutes with missing target. These are quality flags, not yet removed from the primary cohort.
- Current regression result: `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` — 11 passed; `compileall` passed; `pip check` reports no broken requirements. The locally resolved CVXPY package was corrected from 1.9.3 to 1.7.5 to match the declared compatibility range.
- Installed the local project editable with `uv pip install --no-build-isolation --no-deps -e .`; CLI version and archive inspection commands succeeded. The first isolated editable install attempted to resolve wheel from PyPI and failed due DNS; disabling build isolation used the already installed build tools and succeeded.
- Implemented generic meter normalization, duplicate-timestamp averaging, half-hour aggregation, causal lags/calendar features, shared chronological splits, past-only weather alignment, and streaming CSV panel writes that refuse to overwrite existing outputs. First preprocessing test exposed pandas mixed-date inference failure; switched to explicit `format="mixed"`. Current regression: 10 passed and compileall passed. Real dataset preprocessing is not yet run.
- Real-data single-household smoke test: Islamabad House41 had 525,600 source rows and produced 17,520 half-hour intervals; all targets and aligned weather values were valid for this file, and the target conversion assertion passed. This is a preprocessing smoke test, not a research experiment.
- Added long-zero episode flagging and a 24-hour configurable threshold without modifying target values; current regression suite: 11 passed.
- Historical status note above predates the end-to-end experiment recorded below.

## End-to-end real-data run — 2026-11-18

- Prepared 1,031,562 half-hour rows from 59 distinct households. Used 12,084 train, 2,215 validation, 2,177 calibration, and 994 test day-ahead windows; windows crossing split cutoffs are excluded.
- Executed Seasonal Naive, pooled LightGBM (300 trees per lead), PatchTST (8 CPU epochs), and zero-shot Amazon Chronos-2 (local 456 MB checkpoint).
- Test MAE/RMSE: Seasonal Naive 0.3057/0.6211 kW; LightGBM 0.2791/0.5007; PatchTST 0.2486/0.4909; Chronos-2 0.2269/0.4821. WAPE values: 46.35%, 42.31%, 37.68%, 34.40%. 90% conformal PICP: 0.940, 0.945, 0.945, 0.946.
- Generated validation and test TreeSHAP explanations, global mean-absolute importance, local top-ten explanations, and perturbation stability scores. Controller thresholds were selected from validation only; calibration residuals came from the independent calibration split.
- Executed all four controller configurations; automation coverage on the test set was 100.0% ungated, 73.4% explanation-only, 30.4% dual-evidence, and 20.2% uncertainty-only. The uncertainty-only operating point matched dual-evidence validation coverage (53.36% vs. 53.41%); test coverage is separately reported.
- CVXPY test simulations were sampled at one day in seven and ran only for ACT cases. The first cost-only optimizer saved money on evaluated cases, but peak reduction was negative and regret was negative because the oracle used mismatched feasibility caps. This invalidates the regret comparison. The script clips physically impossible negative forecasts only when passed to the optimizer; raw forecasts remain unchanged for accuracy scoring. User approval is pending to correct the oracle feasible set.
- Test suite at the first full run: 18 tests passed; the final regression suite now passes 24 tests (including agentic coordinator unit tests). `results/model_status.json` confirms all four model runs completed. `reports/experiment_results.md` and figures show actual metrics, including the DR caveat.

## Agentic coordinator — 2026-11-18

- Continued from cached Dual-Evidence artifacts; did not retrain LightGBM, PatchTST, Chronos-2, or SHAP.
- Confirmed no “Qwen 2.8” reference in the repo; README already named Qwen3-1.7B-Q4_K_M. Selected that local GGUF for free CPU tool calling via llama.cpp.
- Added `DualEvidenceWorkflow` tools (inspect, X, VERIFY, ABSTAIN lock, ACT simulation, finalize), hardened OpenAI-compatible orchestration, and `scripts/run_agentic_orchestration.py`.
- Unit tests: 24 passed (`unittest discover -s tests -v`).
- Live run: `.venv/bin/python scripts/run_agentic_orchestration.py` against `http://127.0.0.1:18080/v1`. Measured sequence inspect → X → VERIFY → ABSTAIN → simulate → finalize; 994 origins routed ACT 289 / VERIFY 459 / ABSTAIN 246; 406/459 VERIFY checks passed; 48 sampled ACT DR cases; exact match to `decision_log_patchtst.csv` and 31.6296 PKR/case Dual-Evidence saving. Chapter 4: `reports/chapter4.md`.
