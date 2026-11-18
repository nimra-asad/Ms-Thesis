import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from dual_evidence import __version__
from dual_evidence.cli import main
from dual_evidence.data.inspection import inspect_csv_archive
from dual_evidence.data.acquisition import download_file, extract_archive
from dual_evidence.data.audit import audit_dataset
from dual_evidence.data.preprocessing import (
    add_calendar_and_lag_features,
    align_weather_history,
    aggregate_30min,
    chronological_split,
    flag_suspicious_zero_runs,
    normalize_meter_frame,
    write_processed_panel,
)


class SetupTests(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(__version__, "0.1.0")

    def test_cli_inspect_data_creates_log(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            with zipfile.ZipFile(raw / "sample.zip", "w") as bundle:
                bundle.writestr("house.csv", "timestamp,load\n2025-01-01,1.2\n")
            with patch.object(sys, "argv", ["rewdp", "--inspect-data", "--log-dir", directory,
                                              "--data-dir", str(raw)]):
                main()
            self.assertTrue((Path(directory) / "pipeline.log").exists())
            self.assertTrue((Path(directory) / "processed" / "source_inspection.json").exists())

    def test_inspection_reads_csv_headers_and_sample_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("House1.csv", "timestamp,load\n2025-01-01 00:00,0.2\n")
            report = inspect_csv_archive(archive)
            member = report["members"][0]
            self.assertEqual(member["header"], ["timestamp", "load"])
            self.assertEqual(member["sample_rows"], [["2025-01-01 00:00", "0.2"]])

    def test_downloader_reuses_existing_file_without_network(self):
        payload = b"small deterministic fixture"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.bin"
            destination.write_bytes(payload)
            self.assertEqual(download_file("https://invalid.example/never-requested", destination).read_bytes(), payload)

    def test_archive_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escaped.txt", "no")
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                extract_archive(archive, Path(directory) / "out")

    def test_streaming_audit_counts_duplicates_and_zero_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            energy = root / "energy" / "Lahore"
            weather = root / "weather"
            energy.mkdir(parents=True)
            weather.mkdir()
            (energy / "House1.csv").write_text(
                "datetime,Usage (kW)\n2025-01-01,0\n2025-01-01 00:01:00,1\n"
                "2025-01-01 00:01:00,0\n2025-01-01 00:02:00,0\n"
                "2025-01-01 00:03:00,0\n2025-01-01 00:05:00,NA\n"
                "2025-01-01 00:06:00,NA\n", encoding="utf-8")
            (weather / "Lahore.csv").write_text(
                "datetime,Temperature\n2025-01-01,25\n", encoding="utf-8")
            report = audit_dataset(root / "energy", weather)
            household = report["energy"][0]
            self.assertEqual(report["unique_energy_file_count"], 1)
            self.assertEqual(household["duplicate_timestamp_rows"], 1)
            self.assertEqual(household["zero_target"], 4)
            self.assertEqual(household["zero_unique_minute_timestamps"], 3)
            self.assertEqual(household["longest_zero_runs"][0]["minutes"], 2)
            self.assertEqual(household["missing_unique_minute_timestamps"], 2)
            self.assertEqual(household["longest_missing_runs"][0]["minutes"], 2)
            self.assertEqual(household["target_min"], 0.0)
            self.assertEqual(household["target_max"], 1.0)

    def test_preprocessing_normalizes_duplicates_and_aggregates_kwh(self):
        import pandas as pd
        raw = pd.DataFrame({
            "datetime": ["2025-01-01", "2025-01-01 00:00:00", "2025-01-01 00:01:00"],
            "Usage..kW.": [1.0, 3.0, 2.0],
        })
        meter = normalize_meter_frame(raw)
        self.assertEqual(len(meter), 2)
        self.assertEqual(meter.loc[0, "target_kw"], 2.0)
        half_hour = aggregate_30min(meter, minimum_valid_minutes=2)
        self.assertEqual(half_hour.loc[0, "target_kw"], 2.0)
        self.assertEqual(half_hour.loc[0, "target_kwh"], 1.0)
        low_coverage = aggregate_30min(meter, minimum_valid_minutes=3)
        self.assertTrue(pd.isna(low_coverage.loc[0, "target_kw"]))

    def test_split_uses_shared_chronological_cutoffs(self):
        import pandas as pd
        panel = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="30min").repeat(2),
            "household_id": ["A", "B"] * 100,
            "target_kw": list(range(100)) * 2,
        })
        splits = chronological_split(panel)
        self.assertEqual(set(splits), {"train", "validation", "calibration", "test"})
        boundaries = [split.timestamp.min() for split in splits.values()]
        self.assertEqual(boundaries, sorted(boundaries))
        featured = add_calendar_and_lag_features(splits["train"])
        self.assertIn("lag_48", featured)
        self.assertIn("rolling_mean_48", featured)

    def test_panel_writer_deduplicates_identical_source_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "energy"
            root.mkdir()
            payload = "datetime,Usage (kW)\n2025-01-01,1\n"
            (root / "House1.csv").write_text(payload, encoding="utf-8")
            (root / "House1-copy.csv").write_text(payload, encoding="utf-8")
            output = Path(directory) / "panel.csv"
            summary = write_processed_panel(root, output, minimum_valid_minutes=1)
            self.assertEqual(summary, {"households": 1, "rows": 1})
            with self.assertRaises(FileExistsError):
                write_processed_panel(root, output, minimum_valid_minutes=1)

    def test_weather_alignment_uses_only_latest_observation_at_or_before_origin(self):
        import pandas as pd
        panel = pd.DataFrame({"timestamp": pd.to_datetime([
            "2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"
        ]), "target_kw": [1.0, 2.0, 3.0]})
        weather = pd.DataFrame({"datetime": [
            "2025-01-01 00:00", "2025-01-01 01:00"
        ], "Temperature": [20, 25]})
        joined = align_weather_history(panel, weather)
        self.assertEqual(joined["weather_temperature"].tolist(), [20, 20, 25])
        self.assertEqual(joined["weather_age_hours"].tolist(), [0.0, 0.5, 0.0])

    def test_long_zero_runs_are_flagged_without_changing_targets(self):
        import pandas as pd
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=51, freq="30min"),
            "household_id": "House1",
            "target_kw": [0.0] * 50 + [1.0],
        })
        flagged = flag_suspicious_zero_runs(frame, threshold_hours=24)
        self.assertEqual(flagged["suspect_zero_run"].sum(), 50)
        self.assertEqual(flagged["zero_run_hours"].iloc[0], 25.0)
        self.assertEqual(flagged["target_kw"].tolist(), frame["target_kw"].tolist())


if __name__ == "__main__":
    unittest.main()
