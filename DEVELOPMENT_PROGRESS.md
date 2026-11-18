# Development Progress

## Phase 1 — Environment and Project Setup

- **Status:** Complete (Git checkpoint limitation recorded)
- **Completed:** inspected empty workspace; identified macOS Intel, Python 3.11/3.13/3.15, Git 2.42, and `uv`; selected Python 3.11 target. Created initial package/configuration layout and documented configurable research assumptions.
- **Files:** README.md, pyproject.toml, .env.example, .gitignore, configs/*.yaml, `src/dual_evidence/` package skeleton, `tests/test_setup.py`, and `scripts/acquire_data.py`.
- **Tests:** Python 3.11 `unittest discover` — 2 passed; `compileall` passed.
- **Additional verification:** base scientific dependencies installed in `.venv`; `pip check` clean; editable package installation succeeded without build isolation; `rewdp --version` and `rewdp --inspect-data --data-dir data/raw` both ran successfully.
- **Outstanding:** standard Git checkpoint creation is blocked because workspace `.git` is explicitly read-only. Phase 1 code is otherwise set up.
- **Next:** Full end-to-end forecasting, explanation, decision, optimization, and evaluation are now implemented and run; see the completion section below.

## Phase 2 — Dataset Acquisition and Inspection

- **Status:** Complete; source count discrepancy documented
- **Completed:** downloaded and extracted the official LUMS energy archive (506 MiB), weather archive (1.0 MiB), and metadata workbook; implemented acquisition, safe extraction, source inventory, full streaming audit, and a data audit command. Initial full audit and duplicate-file analysis completed.
- **Files:** `src/dual_evidence/data/{acquisition,inspection,audit}.py`, `scripts/{acquire_data,audit_data}.py`, `data/processed/source_inspection.json`, `data/processed/data_audit.json`.
- **Tests:** 11 unit tests pass under the project venv; compileall passes. `pip check` reports no broken requirements; NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, LightGBM 4.7.0, and CVXPY 1.7.5 import successfully.
 - **Findings:** 69 energy CSVs reduce to 59 unique household files after identifying 10 byte-identical Peshawar copies. Metadata also lists House1–House59. All 59 unique files contain about 30.7M rows spanning 2023-07-16 to 2024-11-28. Target headers denote kW with three spelling variants. Audit found 330,900 duplicate timestamp rows, 685,975 rows with missing target values, 1,244,047 zero target rows, 5,241 energy gaps over one minute, and no invalid timestamps or negative targets. The longest flagged zero run was 102,176 minutes. All six city weather files have 12,456 hourly rows from 2023-07-01 to 2024-11-30 with no gaps/out-of-order rows.
- **Resolution:** the research run uses the 59 distinct available homes; it does not invent the absent House60. Long zeros are retained in the main panel and an evaluation sensitivity is reported.
- **Next:** model development (completed in the end-to-end run below).

## Phase 3 — Data Preprocessing

- **Status:** Complete on the full available cohort
- **Completed:** added normalization of the three observed target header variants, timestamp deduplication by mean kW, half-hour average-kW and kWh aggregation with a minimum-valid-minute rule, causal calendar/lag features, shared chronological splits, and exact-file deduplication.
- **Files:** `src/dual_evidence/data/preprocessing.py`, preprocessing unit tests.
- **Tests:** normalization, kW-to-kWh aggregation, low-coverage target handling, chronological splits, causal lag fields, exact-file deduplication, past-only weather alignment, and long-zero-run flagging pass in the 11-test suite. Real-data smoke test on Islamabad House41 processed 525,600 readings into 17,520 intervals with no missing targets or aligned-weather nulls.
- **Issue fixed:** pandas inferred only the first date format and rejected mixed date-only and minute timestamps; parsing now uses `format="mixed"` and the test passes.
- **Completed:** the panel was produced for all 59 unique homes, using 24 valid minute readings per interval, past-only weather alignment, and 24-hour long-zero flags.
- **Next:** see the experiment and DR oracle status below.

## Assumption register

The synopsis does not specify exact split ratios, tariff values, flexible appliance profiles, regret reference, or deployment-time weather availability. Provisional configurable choices are documented in README/configuration and must remain clearly labeled in experiment reports.

## End-to-end implementation and first real-data run (2026-11-18)

- **Environment:** restored Python 3.11.9 `.venv` after discovering its scripts referenced the former `Nimra <3` path. Pinned NumPy 1.26 and Transformers 4.x with the Intel macOS PyTorch 2.2 wheel. Chronos uses one CPU thread; standard HTTPS transfer was required because the default Xet downloader stalled. Its 456 MB model is now cached locally.
- **Phase 2:** full source audit completed. The archive has 69 energy CSV entries, 59 distinct source series and 10 byte-identical duplicate copies. There is no unique House60 series in the archive. Long zeros remain in the main dataset and are flagged.
- **Phase 3:** preprocessed all 59 distinct homes into `data/processed/panel_30min.csv` (1,031,562 half-hour rows). The 24-valid-minute threshold is retained. Weather alignment is past-only.
- **Forecast windows:** 12,084 train, 2,215 validation, 2,177 calibration, and 994 test. The window builder discards any 48-step target window that would cross a chronological split boundary.
- **Phases 3–12:** Seasonal Naive, pooled 48-lead LightGBM, CPU PatchTST and zero-shot Chronos-2 were executed. Conformal intervals use the separate calibration split; TreeSHAP contribution stability uses two deterministic perturbations. The validation-tuned dual-evidence controller, secondary consistency check, four configurations, and simulated CVXPY DR were executed. Results and forecast caches are under `results/`; figures and the report are under `reports/`.
- **Measured first-run test metrics:** Seasonal Naive MAE 0.3057 kW; LightGBM 0.2791; PatchTST 0.2486; Chronos-2 0.2269. 90% split-conformal PICP values were 0.940–0.946. These are actual outputs from 994 test origins and 48 steps per origin.
- **Long-zero sensitivity:** 21 of 994 test windows intersect a flagged long-zero episode. `results/zero_run_sensitivity.json` and the report compare full test scores with flagged windows excluded. This is evaluation-only; training data are unchanged.
- **Tests:** `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v` — 18 passed. PatchTST single-epoch smoke and Chronos one-window/32-window offline CPU probes passed. The four-model pipeline completed through report generation.
- **Pending research decision:** the first DR output has negative mean regret because the perfect-information oracle and forecast-based schedule used different capacity limits. The synopsis leaves this formulation open. User approval was requested to align both feasible sets while preserving the cost-only objective. The measured peak reduction is also negative under that cost-only objective; this is reported, not hidden. The DR comparison needs a final rerun after the oracle decision.
- **LLM:** local Python tool orchestration is implemented. No external LLM API key was configured, so the optional LLM narrative did not run; numerical results are Python-computed.
- **Git:** no commit checkpoint was created because the workspace `.git` directory is read-only in this environment.
