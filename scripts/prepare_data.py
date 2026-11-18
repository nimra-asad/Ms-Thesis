"""Aggregate REWD-P household meters to the configured 30-minute panel."""
from pathlib import Path

import yaml

from dual_evidence.data.preprocessing import write_processed_panel


if __name__ == "__main__":
    config = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
    summary = write_processed_panel(
        config.get("energy_dir", "data/raw/energy"),
        Path(config["processed_dir"]) / "panel_30min.csv",
        minimum_valid_minutes=config["minimum_valid_minutes_per_interval"],
        weather_root=config.get("weather_dir"),
        suspected_zero_run_hours=config.get("suspected_zero_run_hours", 24),
    )
    print(f"Prepared {summary['rows']:,} half-hour records across {summary['households']} unique households")
