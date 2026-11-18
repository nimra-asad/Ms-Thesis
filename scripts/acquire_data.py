"""Download the source dataset from the LUMS Energy Institute."""
from dual_evidence.data.acquisition import acquire_all, extract_archive
from dual_evidence.data.inspection import inspect_csv_archive, write_inspection_report
from dual_evidence.logging_utils import configure_logging


if __name__ == "__main__":
    configure_logging()
    files = acquire_all()
    for path in files:
        print(f"Downloaded/reused: {path} ({path.stat().st_size:,} bytes)")
    reports = []
    for path in files:
        if path.suffix == ".zip":
            category = "energy" if "rewdp" in path.name else "weather"
            extracted = extract_archive(path, path.parent / category)
            reports.append(inspect_csv_archive(path))
            print(f"Extracted {len(extracted)} members to {path.parent / category}")
    write_inspection_report(reports, "data/processed/source_inspection.json")
    print("Inspection report: data/processed/source_inspection.json")
