"""Run the complete streaming quality audit over downloaded source files."""
from dual_evidence.data.audit import audit_dataset, write_audit


if __name__ == "__main__":
    print("Starting full source audit. This scans all unique energy and weather CSVs.", flush=True)
    report = audit_dataset("data/raw/energy", "data/raw/weather", progress=True)
    output = "data/processed/data_audit.json"
    write_audit(report, output)
    print(f"Audited {report['unique_energy_file_count']} unique households and "
          f"{report['weather_csv_count']} weather files; report: {output}")
