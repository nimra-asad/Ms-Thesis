"""Lightweight source archive inspection without loading full household files."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any


def inspect_csv_archive(archive: str | Path, sample_rows: int = 5) -> dict[str, Any]:
    """Report archive members, sizes, and CSV header/sample for each CSV member."""
    path = Path(archive)
    if not path.is_file():
        raise FileNotFoundError(path)
    report: dict[str, Any] = {"archive": str(path), "archive_bytes": path.stat().st_size, "members": []}
    with zipfile.ZipFile(path) as bundle:
        for member in bundle.infolist():
            item: dict[str, Any] = {"name": member.filename, "uncompressed_bytes": member.file_size}
            if member.filename.lower().endswith(".csv") and not member.is_dir():
                with bundle.open(member) as raw:
                    text = (line.decode("utf-8-sig", errors="replace") for line in raw)
                    reader = csv.reader(text)
                    item["header"] = next(reader, [])
                    item["sample_rows"] = [row for _, row in zip(range(sample_rows), reader)]
            report["members"].append(item)
    return report


def write_inspection_report(reports: list[dict[str, Any]], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
