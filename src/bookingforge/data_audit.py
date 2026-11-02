"""Read-only, standard-library hotel booking dataset audit.

Run: python src/bookingforge/data_audit.py --csv data/raw/hotels.csv --output reports/data_audit.json
Audit results are descriptive. They DO NOT certify that features are known at prediction time.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

REQUIRED = {
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "reservation_status", "reservation_status_date",
}
OUTCOME_COLUMNS = ("is_canceled", "reservation_status", "reservation_status_date")
TIMING_REVIEW = {
    "assigned_room_type": "May reflect assignments after booking creation.",
    "booking_changes": "May accumulate changes after booking creation.",
    "days_in_waiting_list": "May contain changes after booking creation.",
    "adr": "May reflect updated rates.",
    "total_of_special_requests": "May be updated after booking creation.",
    "required_car_parking_spaces": "May be updated after booking creation.",
    "previous_cancellations": "Check that counts only include prior events.",
    "previous_bookings_not_canceled": "Check that counts only include prior events.",
}
MONTHS = {m.casefold(): i for i, m in enumerate(calendar.month_name) if m}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _arrival_date(row: dict[str, str]) -> date:
    month = (row["arrival_date_month"] or "").strip().casefold()
    if month not in MONTHS:
        raise ValueError("Unknown arrival month")
    return date(int(row["arrival_date_year"]), MONTHS[month], int(row["arrival_date_day_of_month"]))


def _provisional_split(arrival: date) -> str:
    if arrival <= date(2016, 8, 31):
        return "train"
    if arrival <= date(2016, 12, 31):
        return "validation"
    if arrival <= date(2017, 8, 31):
        return "test"
    return "outside"


def audit_dataset(csv_path: str | Path) -> dict:
    """Stream aggregates; do not emit row-level records or train a model."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    missing = Counter()
    hotels = Counter()
    split_counts = Counter()
    split_canceled = Counter()
    hotel_splits = Counter()
    issues = Counter()
    status_vs_target = Counter()
    month_counts = Counter()
    earliest = latest = None
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Missing CSV header")
        columns = list(reader.fieldnames)
        absent = REQUIRED.difference(columns)
        if absent:
            raise ValueError(f"Required columns missing: {sorted(absent)}")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names in CSV header")
        for row in reader:
            row_count += 1
            for col in columns:
                raw = row.get(col)
                if raw is None or raw.strip().upper() in {"", "NA", "NULL"}:
                    missing[col] += 1
            hotel = (row.get("hotel") or "").strip() or "<missing>"
            hotels[hotel] += 1
            try:
                arrival = _arrival_date(row)
            except (ValueError, TypeError, OverflowError):
                issues["invalid_arrival_date"] += 1
                continue
            earliest = arrival if earliest is None else min(earliest, arrival)
            latest = arrival if latest is None else max(latest, arrival)
            split = _provisional_split(arrival)
            split_counts[split] += 1
            hotel_splits[(hotel, split)] += 1
            month_counts[arrival.strftime("%Y-%m")] += 1
            target = (row.get("is_canceled") or "").strip()
            if target not in {"0", "1"}:
                issues["invalid_cancellation_target"] += 1
            elif target == "1":
                split_canceled[split] += 1
            status_vs_target[((row.get("reservation_status") or "").strip(), target)] += 1
            try:
                if int(row["lead_time"]) < 0:
                    issues["negative_lead_time"] += 1
            except (ValueError, TypeError):
                issues["invalid_lead_time"] += 1
            try:
                nights = int(row["stays_in_weekend_nights"]) + int(row["stays_in_week_nights"])
                if nights == 0:
                    issues["zero_night_booking"] += 1
                elif nights < 0:
                    issues["negative_nights"] += 1
            except (ValueError, TypeError):
                issues["invalid_nights"] += 1

    return {
        "audit_version": "1.0",
        "dataset_name": "Hotel Booking Demand (TidyTuesday copy)",
        "input_filename": path.name,
        "sha256": _sha256(path),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "hotel_counts": dict(sorted(hotels.items())),
        "first_arrival": earliest.isoformat() if earliest else None,
        "last_arrival": latest.isoformat() if latest else None,
        "missing_by_column": dict(sorted(missing.items())),
        "provisional_split_counts": dict(sorted(split_counts.items())),
        "provisional_split_cancellation_rates": {
            key: round(split_canceled[key] / n, 6) if n else None
            for key, n in sorted(split_counts.items())
        },
        "hotel_split_counts": {
            f"{hotel} / {split}": n
            for (hotel, split), n in sorted(hotel_splits.items())
        },
        "arrival_month_counts": dict(sorted(month_counts.items())),
        "status_vs_target": {
            f"{status} / {target}": n
            for (status, target), n in sorted(status_vs_target.items())
        },
        "potential_data_quality_flags": dict(sorted(issues.items())),
        "forbidden_predictors": list(OUTCOME_COLUMNS),
        "features_requiring_timestamp_review": TIMING_REVIEW,
        "evaluation_status": "PROVISIONAL: arrival-based split does not prove observation-time availability or label maturity.",
        "known_limits": [
            "Identical rows are not automatically duplicates: the file lacks unique booking IDs.",
            "Historical feature snapshots are absent; feature timing is not proven.",
            "Arrival-based splits cannot guarantee labels were mature at training cutoff.",
            "Simulation of alternative hotel decisions requires explicit assumptions.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only BookingForge dataset audit")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/data_audit.json")
    args = parser.parse_args()
    report = audit_dataset(args.csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BookingForge data audit complete")
    print(f"Rows: {report['row_count']:,} | Columns: {report['column_count']}")
    print(f"Arrival range: {report['first_arrival']} to {report['last_arrival']}")
    print(f"Hotels: {report['hotel_counts']}")
    print(f"Provisional splits: {report['provisional_split_counts']}")
    print(f"Split cancellation rates: {report['provisional_split_cancellation_rates']}")
    print(f"Quality flags: {report['potential_data_quality_flags']}")
    print(f"Aggregate report: {output}")
    print("WARNING: Feature timing and label maturity remain unvalidated.")


if __name__ == "__main__":
    main()
