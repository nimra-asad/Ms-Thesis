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

### Local pretrained LLM coordinator

The experiment calculations remain reproducible without an LLM. This Intel Mac uses the CPU build of `llama.cpp`. Qwen3 1.7B is an open, pretrained model family with agent/tool capabilities and multilingual support. For a lighter local run, use the Q4_K_M quantization (about 1.28 GB). Download the macOS Intel binary and Qwen3 GGUF model:

```bash
curl --location 'https://github.com/ggml-org/llama.cpp/releases/download/b12502/llama-b12502-bin-macos-x64.tar.gz' --output /tmp/llama-macos-x64.tar.gz
tar -xzf /tmp/llama-macos-x64.tar.gz -C /tmp
curl --location 'https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf' --output /tmp/Qwen3-1.7B-Q4_K_M.gguf
```

Start the local API server in one terminal:

```bash
/tmp/llama-b12502/llama-server --model /tmp/Qwen3-1.7B-Q4_K_M.gguf --alias Qwen3-1.7B-Q4_K_M --jinja --reasoning-format deepseek --host 127.0.0.1 --port 18080 --ctx-size 4096
```

In another terminal, copy `.env.example` to `.env` and run:

```bash
cp .env.example .env
python scripts/run_agentic_orchestration.py
```

The coordinator uses the local OpenAI-compatible endpoint at `http://localhost:18080/v1` to choose among inspect, X-stage routing, VERIFY, ABSTAIN locking, ACT simulation, and finalize. It coordinates the frozen controller, conditional PatchTST verification, and ACT-only demand-response simulation over already-computed model outputs. Forecasts, SHAP, thresholds, ACT/VERIFY/ABSTAIN decisions, and optimization remain deterministic Python outputs; the LLM cannot change test thresholds or bypass the Python policy. Calls and results are recorded under `logs/agentic_tool_calls.json`, `results/agentic_orchestration.json`, and `reports/chapter4.md`.

For an OpenAI-compatible hosted provider, override `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in `.env`. That option may incur provider charges; the local llama.cpp setup does not send project data to a hosted model.

### Research scope and assumptions

The source archive contains 59 distinct household files (10 additional archive entries are exact copies), while the synopsis describes 60 homes. The missing 60th series is not fabricated. Exact zero runs are retained and flagged; half-hour bins need 24 valid minute readings. The split is chronological 70/10/10/10. Weather at each origin uses only the latest observed past value. The tariff and 20% flexible-load share are illustrative simulation assumptions, not utility or appliance measurements. Demand response is a simulation and sends no real control commands.
