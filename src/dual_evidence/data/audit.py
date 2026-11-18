"""Full streaming audit of the downloaded REWD-P household and weather files."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: str) -> datetime:
    value = value.strip()
    return datetime.fromisoformat(value)


def _audit_csv(path: Path, kind: str, progress: bool = False) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not headers or "datetime" not in [h.strip().lower() for h in headers]:
            raise ValueError(f"Missing datetime column in {path}")
        time_col = next(h for h in headers if h.strip().lower() == "datetime")
        target_col = None
        if kind == "energy":
            target_col = next((h for h in headers if re.sub(r"[^a-z]", "", h.lower()).startswith("usage")), None)
            if target_col is None:
                raise ValueError(f"Missing Usage column in {path}; headers={headers}")
        rows = 0
        invalid_time = duplicate_times = missing_target = zero_target = negative_target = 0
        out_of_order = gaps = 0
        date_only = 0
        previous_in_file: datetime | None = None
        deltas: Counter[float] = Counter()
        expected_delta_minutes = 1 if kind == "energy" else 60
        minimum: float | None = None
        maximum: float | None = None
        total = 0.0
        valid_target_count = 0
        seen_times: set[datetime] = set()
        minute_target: dict[datetime, list[float | int]] = {}
        first = last = None
        for row in reader:
            rows += 1
            if progress and rows % 100_000 == 0:
                print(f"  {path.name}: {rows:,} rows scanned", file=sys.stderr, flush=True)
            raw_time = (row.get(time_col) or "").strip()
            if len(raw_time) == 10:
                date_only += 1
            try:
                stamp = _parse_time(raw_time)
            except (ValueError, TypeError):
                invalid_time += 1
                continue
            if first is None or stamp < first:
                first = stamp
            if previous_in_file is not None:
                delta = (stamp - previous_in_file).total_seconds() / 60
                if delta < 0:
                    out_of_order += 1
                elif delta > expected_delta_minutes:
                    gaps += 1
                if delta > 0:
                    deltas[delta] += 1
            previous_in_file = stamp
            if stamp in seen_times:
                duplicate_times += 1
            seen_times.add(stamp)
            if last is None or stamp > last:
                last = stamp
            if target_col:
                raw = (row.get(target_col) or "").strip()
                if raw.upper() in {"", "NA", "N/A", "NAN", "NULL"}:
                    missing_target += 1
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    missing_target += 1
                    continue
                minimum = value if minimum is None else min(minimum, value)
                maximum = value if maximum is None else max(maximum, value)
                total += value
                valid_target_count += 1
                zero_target += value == 0.0
                negative_target += value < 0.0
                aggregate = minute_target.setdefault(stamp, [0.0, 0])
                aggregate[0] += value
                aggregate[1] += 1
    zero_runs: list[dict[str, Any]] = []
    missing_runs: list[dict[str, Any]] = []
    run_start: datetime | None = None
    run_last: datetime | None = None
    run_length = 0
    missing_start: datetime | None = None
    missing_last: datetime | None = None
    missing_length = 0
    zero_timestamps = 0
    missing_timestamps = 0
    if target_col:
        for stamp in sorted(seen_times):
            value_sum, count = minute_target.get(stamp, [0.0, 0])
            is_zero = count > 0 and value_sum == 0.0
            is_missing = count == 0
            if is_zero:
                zero_timestamps += 1
            if is_zero and run_last is not None and (stamp - run_last).total_seconds() == 60:
                run_length += 1
                run_last = stamp
            else:
                if run_start is not None:
                    zero_runs.append({"start": run_start.isoformat(sep=" "),
                                      "end": run_last.isoformat(sep=" "), "minutes": run_length})
                run_start = stamp if is_zero else None
                run_last = stamp if is_zero else None
                run_length = 1 if is_zero else 0
            if is_missing:
                missing_timestamps += 1
            if is_missing and missing_last is not None and (stamp - missing_last).total_seconds() == 60:
                missing_length += 1
                missing_last = stamp
            else:
                if missing_start is not None:
                    missing_runs.append({"start": missing_start.isoformat(sep=" "),
                                         "end": missing_last.isoformat(sep=" "),
                                         "minutes": missing_length})
                missing_start = stamp if is_missing else None
                missing_last = stamp if is_missing else None
                missing_length = 1 if is_missing else 0
        if run_start is not None:
            zero_runs.append({"start": run_start.isoformat(sep=" "),
                              "end": run_last.isoformat(sep=" "), "minutes": run_length})
        if missing_start is not None:
            missing_runs.append({"start": missing_start.isoformat(sep=" "),
                                 "end": missing_last.isoformat(sep=" "),
                                 "minutes": missing_length})
    return {
        "file": str(path), "kind": kind, "headers": headers, "rows": rows,
        "start": first.isoformat(sep=" ") if first else None,
        "end": last.isoformat(sep=" ") if last else None,
        "date_only_rows": date_only, "invalid_timestamps": invalid_time,
        "duplicate_timestamp_rows": duplicate_times,
        "out_of_order_timestamp_rows": out_of_order,
        "most_common_positive_deltas_minutes": deltas.most_common(5),
        "expected_interval_minutes": expected_delta_minutes,
        "gaps_over_expected_interval": gaps,
        "target_column": target_col, "missing_target": missing_target,
        "zero_target": zero_target, "negative_target": negative_target,
        "zero_unique_minute_timestamps": zero_timestamps,
        "zero_run_count": len(zero_runs),
        "longest_zero_runs": sorted(zero_runs, key=lambda run: run["minutes"], reverse=True)[:5],
        "missing_unique_minute_timestamps": missing_timestamps,
        "missing_run_count": len(missing_runs),
        "longest_missing_runs": sorted(missing_runs, key=lambda run: run["minutes"], reverse=True)[:5],
        "target_min": minimum if target_col else None,
        "target_max": maximum if target_col else None,
        "target_mean": total / valid_target_count if target_col and valid_target_count else None,
    }


def audit_dataset(energy_dir: str | Path, weather_dir: str | Path,
                  progress: bool = False) -> dict[str, Any]:
    energy_files = sorted(Path(energy_dir).rglob("*.csv"))
    weather_files = sorted(Path(weather_dir).rglob("*.csv"))
    unique_energy: dict[str, Path] = {}
    duplicate_files: list[dict[str, str]] = []
    for path in energy_files:
        digest = _sha256(path)
        if digest in unique_energy:
            duplicate_files.append({"duplicate": str(path), "identical_to": str(unique_energy[digest]), "sha256": digest})
        else:
            unique_energy[digest] = path
    energy_reports = []
    for index, path in enumerate(unique_energy.values(), start=1):
        if progress:
            print(f"Auditing household file {index}/{len(unique_energy)}: {path.name}",
                  file=sys.stderr, flush=True)
        energy_reports.append(_audit_csv(path, "energy", progress=progress))
    weather_reports = []
    for index, path in enumerate(weather_files, start=1):
        if progress:
            print(f"Auditing weather file {index}/{len(weather_files)}: {path.name}",
                  file=sys.stderr, flush=True)
        weather_reports.append(_audit_csv(path, "weather", progress=progress))
    return {
        "energy_csv_count": len(energy_files),
        "unique_energy_file_count": len(unique_energy),
        "duplicate_files": duplicate_files,
        "energy": energy_reports,
        "weather_csv_count": len(weather_files),
        "weather": weather_reports,
    }


def write_audit(report: dict[str, Any], destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else x), encoding="utf-8")
