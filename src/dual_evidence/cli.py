"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_evidence.data.inspection import inspect_csv_archive, write_inspection_report
from dual_evidence.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="REWD-P dual-evidence research pipeline")
    parser.add_argument("--version", action="version", version="dual-evidence 0.1.0")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--inspect-data", action="store_true", help="Inspect local raw dataset files")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    configure_logging(args.log_dir)
    if args.inspect_data:
        root = Path(args.data_dir)
        archives = sorted(root.glob("*.zip"))
        if not archives:
            parser.error(f"No complete ZIP archives found in {root}; partial downloads are not inspected")
        reports = [inspect_csv_archive(path) for path in archives]
        output = root.parent / "processed" / "source_inspection.json"
        write_inspection_report(reports, output)
        logging.getLogger(__name__).info("Inspected %d archives; report: %s", len(reports), output)
        print(json.dumps({"archives": len(reports), "report": str(output)}, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
