"""Leakage-aware meter normalization, 30-minute aggregation, and chronological splits."""
from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_meter_frame(frame: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Normalize source column variants and average duplicate minute timestamps.

    The target remains mean demand in kW. Missing meter values remain missing so
    downstream complete-window logic can exclude affected observations honestly.
    """
    columns = {str(c).strip().lower(): c for c in frame.columns}
    time_col = columns.get("datetime")
    if time_col is None:
        raise ValueError(f"{source}: expected a datetime column")
    target_col = next(
        (c for c in frame.columns if re.sub(r"[^a-z]", "", str(c).lower()).startswith("usage")),
        None,
    )
    if target_col is None:
        raise ValueError(f"{source}: expected a Usage (kW) column")
    result = pd.DataFrame({
        # REWD-P includes both date-only midnights and minute-formatted rows.
        "timestamp": pd.to_datetime(frame[time_col], errors="coerce", format="mixed"),
        "target_kw": pd.to_numeric(frame[target_col], errors="coerce"),
    })
    result = result.dropna(subset=["timestamp"])
    if result["target_kw"].lt(0).any():
        raise ValueError(f"{source}: negative whole-house demand values found")
    result = result.groupby("timestamp", as_index=False, sort=True).agg(
        target_kw=("target_kw", "mean"), duplicate_rows=("target_kw", "size")
    )
    result["source"] = source
    return result


def aggregate_30min(
    meter: pd.DataFrame,
    minimum_valid_minutes: int = 24,
    expected_minutes: int = 30,
) -> pd.DataFrame:
    """Aggregate minute-level kW to interval mean kW and interval kWh.

    A half-hour bin with fewer than ``minimum_valid_minutes`` valid readings is
    retained with a missing target, preserving the gap for reporting and split
    filtering rather than imputing a target from future data.
    """
    if not 1 <= minimum_valid_minutes <= expected_minutes:
        raise ValueError("minimum_valid_minutes must be within [1, expected_minutes]")
    frame = meter.set_index("timestamp").sort_index()
    grouped = frame["target_kw"].resample("30min", label="left", closed="left")
    result = pd.DataFrame({
        "target_kw": grouped.mean(),
        "valid_minute_count": grouped.count(),
        "source_row_count": frame["duplicate_rows"].resample("30min", label="left", closed="left").sum(),
    })
    result.loc[result["valid_minute_count"] < minimum_valid_minutes, "target_kw"] = float("nan")
    result["target_kwh"] = result["target_kw"] * 0.5
    result["timestamp"] = result.index
    result = result.reset_index(drop=True)
    if "source" in meter and not meter.empty:
        result["source"] = meter["source"].iloc[0]
    result["zero_load"] = result["target_kw"].eq(0).fillna(False)
    return result


def add_calendar_and_lag_features(
    panel: pd.DataFrame,
    household_col: str = "household_id",
    target_col: str = "target_kw",
) -> pd.DataFrame:
    """Add causal demand lags and calendar fields to a household half-hour panel."""
    required = {"timestamp", household_col, target_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing feature input columns: {sorted(missing)}")
    result = panel.sort_values([household_col, "timestamp"]).copy()
    timestamp = pd.to_datetime(result["timestamp"])
    result["slot_of_day"] = timestamp.dt.hour * 2 + timestamp.dt.minute // 30
    result["day_of_week"] = timestamp.dt.dayofweek
    result["month"] = timestamp.dt.month
    result["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    result["slot_sin"] = (2 * math.pi * result["slot_of_day"] / 48).map(math.sin)
    result["slot_cos"] = (2 * math.pi * result["slot_of_day"] / 48).map(math.cos)
    grouped = result.groupby(household_col, sort=False)[target_col]
    for steps in (1, 2, 48, 336):
        result[f"lag_{steps}"] = grouped.shift(steps)
    result["rolling_mean_48"] = grouped.transform(lambda values: values.shift(1).rolling(48, min_periods=36).mean())
    return result


def chronological_split(
    panel: pd.DataFrame,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split by shared time cutoffs so all households use identical periods."""
    ratios = ratios or {"train": 0.70, "validation": 0.10, "calibration": 0.10, "test": 0.10}
    keys = ("train", "validation", "calibration", "test")
    if set(ratios) != set(keys) or any(value <= 0 for value in ratios.values()):
        raise ValueError(f"ratios must contain positive values for {keys}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to one")
    times = pd.Index(pd.to_datetime(panel["timestamp"]).dropna().unique()).sort_values()
    if len(times) < 10:
        raise ValueError("at least ten unique timestamps are required")
    cuts = [round(len(times) * sum(ratios[k] for k in keys[:i])) for i in range(1, 4)]
    boundaries = [times[0], times[cuts[0]], times[cuts[1]], times[cuts[2]], times[-1] + pd.Timedelta(nanoseconds=1)]
    stamps = pd.to_datetime(panel["timestamp"])
    return {
        key: panel.loc[(stamps >= boundaries[i]) & (stamps < boundaries[i + 1])].copy()
        for i, key in enumerate(keys)
    }


def load_unique_household_files(root: str | Path) -> list[Path]:
    """Find CSVs and discard only byte-identical duplicate copies, deterministically."""
    paths = sorted(Path(root).rglob("*.csv"))
    seen: set[str] = set()
    selected = []
    for path in paths:
        digest = _sha256(path)
        if digest not in seen:
            selected.append(path)
            seen.add(digest)
    return selected


def prepare_household_panels(
    root: str | Path,
    minimum_valid_minutes: int = 24,
    weather_root: str | Path | None = None,
    suspected_zero_run_hours: float = 24.0,
) -> Iterator[pd.DataFrame]:
    """Yield one clean half-hour panel per distinct source household file."""
    weather_index = None
    if weather_root is not None:
        weather_index = {
            path.stem.casefold(): pd.read_csv(path)
            for path in Path(weather_root).rglob("*.csv")
        }
    for path in load_unique_household_files(root):
        match = re.search(r"House\s*(\d+)", path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not infer household ID from {path.name}")
        meter = normalize_meter_frame(pd.read_csv(path), source=str(path))
        panel = aggregate_30min(meter, minimum_valid_minutes=minimum_valid_minutes)
        panel["household_id"] = f"House{int(match.group(1))}"
        panel["city"] = path.parent.name
        if weather_index is not None:
            weather_frame = weather_index.get(path.parent.name.casefold())
            if weather_frame is None:
                raise FileNotFoundError(f"No weather CSV for city {path.parent.name}")
            panel = align_weather_history(panel, weather_frame)
        panel = flag_suspicious_zero_runs(panel, suspected_zero_run_hours)
        yield panel


def write_processed_panel(
    root: str | Path,
    destination: str | Path,
    minimum_valid_minutes: int = 24,
    weather_root: str | Path | None = None,
    suspected_zero_run_hours: float = 24.0,
) -> dict[str, int]:
    """Write the aggregated panel incrementally to CSV to cap peak memory use."""
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing processed data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    household_count = row_count = 0
    try:
        temporary = Path(temporary_name)
        for household_count, panel in enumerate(
            prepare_household_panels(root, minimum_valid_minutes, weather_root,
                                     suspected_zero_run_hours), start=1
        ):
            panel.to_csv(temporary, mode="a", index=False, header=household_count == 1)
            row_count += len(panel)
        if household_count == 0:
            raise FileNotFoundError(f"No household CSVs found under {root}")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing processed data: {output}")
        temporary.replace(output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {"households": household_count, "rows": row_count}


def align_weather_history(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """As-of join the most recently observed city weather, never realized future weather.

    This supplies a weather snapshot available at each forecast origin; future
    horizon samples must carry only the origin snapshot unless a true weather
    forecast is supplied separately.
    """
    if "datetime" not in weather:
        raise ValueError("weather data must contain datetime")
    right = weather.copy()
    right["timestamp"] = pd.to_datetime(right.pop("datetime"), errors="coerce", format="mixed")
    right = right.dropna(subset=["timestamp"]).sort_values("timestamp")
    right = right.rename(columns={c: f"weather_{c.strip().lower().replace(' ', '_')}"
                                  for c in right.columns if c != "timestamp"})
    left = panel.sort_values("timestamp").copy()
    joined = pd.merge_asof(left, right, on="timestamp", direction="backward")
    weather_time = right["timestamp"]
    prior_idx = weather_time.searchsorted(pd.DatetimeIndex(joined["timestamp"]), side="right") - 1
    matched = pd.Series(pd.NaT, index=joined.index, dtype="datetime64[ns]")
    valid = prior_idx >= 0
    matched.loc[valid] = weather_time.iloc[prior_idx[valid]].to_numpy()
    joined["weather_age_hours"] = (
        pd.to_datetime(joined["timestamp"]) - matched
    ).dt.total_seconds() / 3600
    return joined


def flag_suspicious_zero_runs(
    panel: pd.DataFrame,
    threshold_hours: float = 24.0,
    household_col: str = "household_id",
) -> pd.DataFrame:
    """Flag, but do not alter, zero-demand intervals in long consecutive runs."""
    if threshold_hours <= 0:
        raise ValueError("threshold_hours must be positive")
    result = panel.sort_values([household_col, "timestamp"]).copy()
    result["zero_run_hours"] = 0.0
    result["suspect_zero_run"] = False
    threshold_steps = math.ceil(threshold_hours * 2)
    for _, indices in result.groupby(household_col, sort=False).groups.items():
        zero = result.loc[indices, "target_kw"].eq(0)
        run_ids = zero.ne(zero.shift(fill_value=False)).cumsum()
        lengths = zero.groupby(run_ids).transform("sum")
        is_long_zero = zero & lengths.ge(threshold_steps)
        result.loc[indices, "zero_run_hours"] = lengths.where(zero, 0).astype(float).to_numpy() * 0.5
        result.loc[indices, "suspect_zero_run"] = is_long_zero.to_numpy()
    return result
