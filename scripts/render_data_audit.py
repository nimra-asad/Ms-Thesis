"""Render the machine-readable source audit as a concise human-readable report."""
import json
from pathlib import Path


def render(data: dict) -> str:
    energy = data["energy"]
    weather = data["weather"]
    sums = lambda key: sum(row[key] for row in energy)
    report = [
        "# REWD-P Source Data Audit",
        "",
        "This report describes the downloaded source files. It contains no forecast or decision results.",
        "",
        "## Household electricity files",
        "",
        f"- CSV files in archive: {data['energy_csv_count']}",
        f"- Unique file contents: {data['unique_energy_file_count']}",
        f"- Exact duplicate copies: {len(data['duplicate_files'])}",
        f"- Rows across unique households: {sums('rows'):,}",
        f"- Timestamp span: {min(r['start'] for r in energy)} to {max(r['end'] for r in energy)}",
        f"- Invalid timestamps: {sums('invalid_timestamps'):,}",
        f"- Duplicate timestamp rows: {sums('duplicate_timestamp_rows'):,}",
        f"- Out-of-order timestamp rows: {sums('out_of_order_timestamp_rows'):,}",
        f"- Missing target rows: {sums('missing_target'):,}",
        f"- Exact zero target readings: {sums('zero_target'):,}",
        f"- Negative target readings: {sums('negative_target'):,}",
        f"- Gaps longer than one minute: {sums('gaps_over_expected_interval'):,}",
        f"- Target column variants: {', '.join(sorted({r['target_column'] for r in energy}))}",
        "",
        "## Weather files",
        "",
    ]
    for row in weather:
        report.append(f"- {Path(row['file']).name}: {row['rows']:,} hourly rows, {row['start']} to {row['end']}; "
                      f"{row['gaps_over_expected_interval']} gaps over one hour")
    report += ["", "## Largest zero-consumption episodes", ""]
    runs = sorted(
        ((run["minutes"], Path(row["file"]).name, run["start"], run["end"])
         for row in energy for run in row.get("longest_zero_runs", [])), reverse=True
    )[:10]
    report.extend(f"- {minutes:,} consecutive minutes in {name}: {start} to {end}"
                  for minutes, name, start, end in runs)
    report += ["", "## Largest continuous missing-target episodes", ""]
    runs = sorted(
        ((run["minutes"], Path(row["file"]).name, run["start"], run["end"])
         for row in energy for run in row.get("longest_missing_runs", [])), reverse=True
    )[:10]
    report.extend(f"- {minutes:,} consecutive minutes in {name}: {start} to {end}"
                  for minutes, name, start, end in runs)
    report += [
        "",
        "## Source count discrepancy",
        "",
        "The archive and metadata workbook identify House1–House59. Ten additional Peshawar CSV entries are byte-identical copies, so the archive has 59 unique household series. This differs from the synopsis and LUMS overview statement of 60 households; the discrepancy is awaiting the user's decision.",
        "",
    ]
    return "\n".join(report)


if __name__ == "__main__":
    data = json.loads(Path("data/processed/data_audit.json").read_text(encoding="utf-8"))
    output = Path("reports/data_audit.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(data), encoding="utf-8")
    print(f"Wrote {output}")
