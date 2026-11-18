# Dual-Evidence Residential Load Forecasting

Reproducible research implementation for the REWD-P residential electricity and weather dataset.

## Status

The end-to-end pipeline and real-data experiment have been implemented and executed. One demand-response regret-reference issue is pending the user's research-design decision; see [DEVELOPMENT_PROGRESS.md](DEVELOPMENT_PROGRESS.md), [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md), and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for exact status and measured outputs.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[chronos,patchtst,dev]'
cp .env.example .env
```

Acquire and audit the official source data:

```bash
python scripts/acquire_data.py
python scripts/audit_data.py
# Uses the 59 distinct household series present in the archive; the missing House60 is not fabricated.
python scripts/prepare_data.py
rewdp --inspect-data --data-dir data/raw
```

The source archives, expanded dataset, processed data, and model checkpoints are excluded from Git.

## Research assumptions

The synopsis leaves some operational details open. Configured assumptions are: chronological 70/10/10/10 train/validation/calibration/test splits; an illustrative time-of-use tariff; explicitly simulated flexible appliance loads; a perfect-information minimum-cost schedule as the regret reference; and weather features restricted to values available at forecast time. These assumptions are not empirical claims and are labeled in the experiment outputs. The first DR run exposed an oracle feasibility mismatch; current regret values are invalid pending alignment and rerun.

No live appliance control is implemented or permitted. Demand response is simulation only.

## Run the full research workflow

Create the CPU-compatible Python environment (Python 3.11 is recommended):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[chronos,patchtst,dev]'
```

The full model set includes PyTorch and downloads the pretrained Chronos-2 weights on its first run. Then acquire and prepare the public LUMS files:

```bash
python scripts/acquire_data.py
python scripts/audit_data.py
python scripts/prepare_data.py
```

The audit scans millions of source records and prints progress. Preprocessing writes `data/processed/panel_30min.csv`; it refuses to overwrite an existing panel. Run the chronological model comparison and all four decision configurations with:

```bash
python scripts/run_experiment.py
```

Outputs are written to `results/`, `models/`, `reports/`, and `logs/`. To run the test suite:

```bash
python -m unittest discover -s tests -v
```

### Optional LLM coordinator

The research workflow runs without an LLM key. To enable the optional OpenAI-compatible report coordinator, copy `.env.example` to `.env` and set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. The LLM receives computed summaries only; Python modules retain ownership of forecasts, SHAP, thresholds, controller decisions, and optimization.

### Research scope and assumptions

The source archive contains 59 distinct household files (10 additional archive entries are exact copies), while the synopsis describes 60 homes. The missing 60th series is not fabricated. Exact zero runs are retained and flagged; half-hour bins need 24 valid minute readings. The split is chronological 70/10/10/10. Weather at each origin uses only the latest observed past value. The tariff and 20% flexible-load share are illustrative simulation assumptions, not utility or appliance measurements. Demand response is a simulation and sends no real control commands.
