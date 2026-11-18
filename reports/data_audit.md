# REWD-P Source Data Audit

This report describes the downloaded source files. It contains no forecast or decision results.

## Household electricity files

- CSV files in archive: 69
- Unique file contents: 59
- Exact duplicate copies: 10
- Rows across unique households: 30,707,062
- Timestamp span: 2023-07-16 00:00:00 to 2024-11-28 23:59:00
- Invalid timestamps: 0
- Duplicate timestamp rows: 330,900
- Out-of-order timestamp rows: 0
- Missing target rows: 685,975
- Exact zero target readings: 1,244,047
- Negative target readings: 0
- Gaps longer than one minute: 5,241
- Target column variants: Usage (kW), Usage..kW., Usage_kW

## Weather files

- Islamabad.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour
- Karachi.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour
- Lahore.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour
- Multan.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour
- Peshawar.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour
- Skardu.csv: 12,456 hourly rows, 2023-07-01 00:00:00 to 2024-11-30 23:00:00; 0 gaps over one hour

## Largest zero-consumption episodes

- 102,176 consecutive minutes in House30.csv: 2024-03-16 01:04:00 to 2024-05-25 23:59:00
- 45,299 consecutive minutes in islamabad_House50.csv: 2024-05-22 13:03:00 to 2024-06-23 00:01:00
- 42,872 consecutive minutes in House21.csv: 2023-11-30 22:08:00 to 2023-12-30 16:39:00
- 31,671 consecutive minutes in House30.csv: 2024-02-16 00:09:00 to 2024-03-08 23:59:00
- 30,219 consecutive minutes in islamabad_House41.csv: 2024-09-27 11:34:00 to 2024-10-18 11:12:00
- 26,027 consecutive minutes in House27.csv: 2024-08-14 22:13:00 to 2024-09-01 23:59:00
- 26,027 consecutive minutes in House26.csv: 2024-08-14 22:13:00 to 2024-09-01 23:59:00
- 18,046 consecutive minutes in islamabad_House41.csv: 2024-10-18 11:14:00 to 2024-10-30 23:59:00
- 17,933 consecutive minutes in House22.csv: 2023-12-11 13:07:00 to 2023-12-23 23:59:00
- 13,356 consecutive minutes in skardu_House53.csv: 2024-05-17 17:45:00 to 2024-05-27 00:20:00

## Largest continuous missing-target episodes

- 42,390 consecutive minutes in islamabad_House45.csv: 2023-12-22 04:38:00 to 2024-01-20 15:07:00
- 30,347 consecutive minutes in multan_House15.csv: 2023-12-13 22:13:00 to 2024-01-03 23:59:00
- 20,571 consecutive minutes in multan_House19.csv: 2024-10-04 17:09:00 to 2024-10-18 23:59:00
- 20,456 consecutive minutes in islamabad_House42.csv: 2024-02-02 19:04:00 to 2024-02-16 23:59:00
- 16,193 consecutive minutes in multan_House19.csv: 2024-08-03 18:07:00 to 2024-08-14 23:59:00
- 14,400 consecutive minutes in multan_House19.csv: 2024-02-28 00:00:00 to 2024-03-08 23:59:00
- 14,277 consecutive minutes in islamabad_House49.csv: 2024-05-10 02:03:00 to 2024-05-19 23:59:00
- 12,410 consecutive minutes in lahore_House5.csv: 2024-01-05 09:10:00 to 2024-01-13 23:59:00
- 11,520 consecutive minutes in lahore_House8.csv: 2024-06-25 00:00:00 to 2024-07-02 23:59:00
- 11,366 consecutive minutes in islamabad_House45.csv: 2024-09-10 09:00:00 to 2024-09-18 06:25:00

## Source count discrepancy

The archive and metadata workbook identify House1–House59. Ten additional Peshawar CSV entries are byte-identical copies, so the archive has 59 unique household series. This differs from the synopsis and LUMS overview statement of 60 households; the discrepancy is awaiting the user's decision.
