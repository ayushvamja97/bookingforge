"""Temporal and label-maturity diagnostics for the Hotel Booking Demand dataset.

Descriptive audit only. Reconstructed booking date = arrival date - lead_time.
This is a proxy, not a verified historical booking-creation timestamp.
Never use reservation_status or reservation_status_date as predictors.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

MONTH_NUMBERS = {month.casefold(): index for index, month in enumerate(calendar.month_name) if month}
CUTOFFS = {
    "train": date(2016, 8, 31),
    "validation": date(2016, 12, 31),
    "test": date(2017, 8, 31),
}
REQUIRED = {
    "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "lead_time", "reservation_status_date", "reservation_status",
    "is_canceled", "stays_in_week_nights", "stays_in_weekend_nights",
}


def parse_arrival(row: dict[str, str]) -> date:
    month = (row["arrival_date_month"] or "").strip().casefold()
    if month not in MONTH_NUMBERS:
        raise ValueError("Unknown month")
    return date(
        int(row["arrival_date_year"]),
        MONTH_NUMBERS[month],
        int(row["arrival_date_day_of_month"]),
    )


def arrival_split(arrival: date) -> str:
    if arrival <= CUTOFFS["train"]:
        return "train"
    if arrival <= CUTOFFS["validation"]:
        return "validation"
    if arrival <= CUTOFFS["test"]:
        return "test"
    return "outside"


def temporal_audit(csv_path: str | Path) -> dict:
    """Inspect time consistency without selecting features or fitting a model."""
    counters = Counter()
    outcomes_by_split = Counter()
    zero_night_status = Counter()
    events_after_cutoff = Counter()
    event_dates_before_arrival = Counter()
    first_creation = last_creation = None
    path = Path(csv_path)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Missing CSV header")
        missing = REQUIRED - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            counters["total_rows"] += 1
            try:
                arrival = parse_arrival(row)
            except (ValueError, TypeError, OverflowError):
                counters["invalid_arrival_date"] += 1
                continue

            split = arrival_split(arrival)
            status = (row["reservation_status"] or "").strip()
            target = (row["is_canceled"] or "").strip()
            outcomes_by_split[(split, status)] += 1

            try:
                nights = int(row["stays_in_weekend_nights"]) + int(row["stays_in_week_nights"])
                if nights == 0:
                    zero_night_status[status] += 1
            except (TypeError, ValueError):
                counters["invalid_stay_length"] += 1

            try:
                lead = int(row["lead_time"])
                if lead < 0:
                    raise ValueError("Negative lead time")
                created = arrival - timedelta(days=lead)
                first_creation = min(first_creation, created) if first_creation else created
                last_creation = max(last_creation, created) if last_creation else created
                counters["reconstructed_creation_count"] += 1
                if lead > 365:
                    counters["lead_time_over_365_days"] += 1
            except (TypeError, ValueError, OverflowError):
                created = None
                counters["invalid_creation_proxy"] += 1

            try:
                status_date = date.fromisoformat((row["reservation_status_date"] or "").strip())
                counters["parsed_final_status_dates"] += 1
            except (ValueError, TypeError):
                counters["invalid_status_date"] += 1
                continue

            if created is not None and status_date < created:
                counters["status_before_creation_proxy"] += 1

            if status_date < arrival:
                event_dates_before_arrival[status] += 1

            if target == "1" and status not in {"Canceled", "No-Show"}:
                counters["unexpected_positive_target_status"] += 1
            if target == "0" and status != "Check-Out":
                counters["unexpected_negative_target_status"] += 1

            if split in CUTOFFS and status_date > CUTOFFS[split]:
                events_after_cutoff[split] += 1

    return {
        "audit_version": "1.0",
        "total_rows": counters["total_rows"],
        "reconstructed_creation_range": {
            "earliest": first_creation.isoformat() if first_creation else None,
            "latest": last_creation.isoformat() if last_creation else None,
        },
        "counts": dict(sorted(counters.items())),
        "arrival_split_final_status_counts": {
            f"{split} / {status}": count
            for (split, status), count in sorted(outcomes_by_split.items())
        },
        "final_status_dated_after_own_split_cutoff": dict(sorted(events_after_cutoff.items())),
        "zero_night_booking_by_final_status": dict(sorted(zero_night_status.items())),
        "final_status_dated_before_arrival": dict(sorted(event_dates_before_arrival.items())),
        "interpretation": [
            "The creation date is reconstructed using lead_time, not verified against booking history.",
            "The final reservation-status date is a proxy for outcome availability.",
            "Counts after a cutoff indicate records whose final status was not yet observed at that cutoff.",
            "An arrival-date split alone does not prove label maturity, nor feature availability.",
            "A complete time-respecting training protocol still requires selecting an observation timestamp.",
        ],
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BookingForge temporal leakage diagnostics")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/temporal_audit.json")
    args = parser.parse_args()
    report = temporal_audit(args.csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BookingForge temporal audit complete")
    print(f"Rows checked: {report['total_rows']:,}")
    print(f"Reconstructed booking dates: {report['reconstructed_creation_range']}")
    print(f"Status dates AFTER own arrival-split cutoff: {report['final_status_dated_after_own_split_cutoff']}")
    print(f"Zero-night outcomes: {report['zero_night_booking_by_final_status']}")
    print(f"Temporal flags: {report['counts']}")
    print(f"Aggregate report: {output}")
    print("WARNING: This audit does not authorize predictive model training.")


if __name__ == "__main__":
    main()
