"""Booking-creation cohort diagnostics and a conservative predictor contract.

Historical records lack feature snapshots. Listed features are plausible
booking-time proxies, not verified values at booking creation.
"""
from __future__ import annotations
import argparse
import calendar
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
TRAIN_CUTOFF = date(2016, 8, 31)
VALID_CUTOFF = date(2016, 12, 31)
TEST_CUTOFF = date(2017, 8, 31)

PROVISIONAL_BOOKING_FEATURES = frozenset({
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "reserved_room_type",
})
REQUIRED = PROVISIONAL_BOOKING_FEATURES | frozenset({
    "is_canceled", "reservation_status_date",
})


def validate_booking_features(features: list[str]) -> None:
    """Fail closed if a pipeline requests unreviewed columns."""
    if not features:
        raise ValueError("At least one predictor is required")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate predictors not allowed")
    unapproved = set(features) - PROVISIONAL_BOOKING_FEATURES
    if unapproved:
        raise ValueError(f"Not approved as booking-time proxy features: {sorted(unapproved)}")


def reconstructed_creation(arrival: date, lead_time: str | int) -> date:
    lead = int(lead_time)
    if lead < 0:
        raise ValueError("Negative lead_time")
    return arrival - timedelta(days=lead)


def cohort_at(creation: date) -> str:
    if creation <= TRAIN_CUTOFF:
        return "train"
    if creation <= VALID_CUTOFF:
        return "validation"
    if creation <= TEST_CUTOFF:
        return "test"
    return "outside"


def parse_arrival(row: dict[str, str]) -> date:
    month = (row["arrival_date_month"] or "").strip().casefold()
    if month not in MONTHS:
        raise ValueError("Invalid arrival month")
    return date(int(row["arrival_date_year"]), MONTHS[month], int(row["arrival_date_day_of_month"]))


def audit_creation_cohorts(csv_path: str | Path) -> dict:
    """Produce aggregates only; this does not authorize model training."""
    counts = Counter()
    maturity = Counter()
    cancellations = Counter()
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Missing header")
        missing = REQUIRED - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        for row in reader:
            counts["total_rows"] += 1
            try:
                arrival = parse_arrival(row)
                created = reconstructed_creation(arrival, row["lead_time"])
                status_date = date.fromisoformat(row["reservation_status_date"].strip())
                if row["is_canceled"].strip() not in ("0", "1"):
                    raise ValueError("Invalid target")
            except (ValueError, TypeError, OverflowError, KeyError):
                counts["invalid_rows"] += 1
                continue

            cohort = cohort_at(created)
            counts[f"cohort_{cohort}"] += 1
            if row["is_canceled"].strip() == "1":
                cancellations[cohort] += 1
            if status_date < created:
                counts["final_status_before_reconstructed_creation"] += 1

            if cohort == "train":
                label = "train_known_at_train_cutoff" if status_date <= TRAIN_CUTOFF else "train_not_known_at_train_cutoff"
                maturity[label] += 1
            elif cohort == "validation":
                label = "validation_known_at_valid_cutoff" if status_date <= VALID_CUTOFF else "validation_not_known_at_valid_cutoff"
                maturity[label] += 1
            elif cohort == "test":
                label = "test_known_at_test_cutoff" if status_date <= TEST_CUTOFF else "test_not_known_at_test_cutoff"
                maturity[label] += 1

    cohorts = {name: counts[f"cohort_{name}"] for name in ("train", "validation", "test", "outside")}
    return {
        "audit_version": "1.0",
        "total_rows": counts["total_rows"],
        "creation_cohort_counts": cohorts,
        "creation_cohort_cancel_rates": {
            name: round(cancellations[name] / n, 6) if n else None for name, n in cohorts.items()
        },
        "label_maturity": dict(sorted(maturity.items())),
        "other_flags": {
            "invalid_rows": counts["invalid_rows"],
            "final_status_before_reconstructed_creation": counts["final_status_before_reconstructed_creation"],
        },
        "provisional_booking_features": sorted(PROVISIONAL_BOOKING_FEATURES),
        "warning": (
            "Creation date = arrival minus lead_time is reconstructed, not verified. "
            "Recorded features are not historical booking-time snapshots. "
            "Dataset inclusion is bounded by arrival date, so later creation cohorts "
            "can be right-truncated and biased. This is not an approved training protocol."
        ),
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BookingForge creation-time cohort audit")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/creation_cohorts.json")
    args = parser.parse_args()
    report = audit_creation_cohorts(args.csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BookingForge creation-cohort audit complete")
    print("Creation cohorts:", report["creation_cohort_counts"])
    print("Label maturity:", report["label_maturity"])
    print("Additional flags:", report["other_flags"])
    print("Report:", output)
    print("WARNING: Training protocol is still provisional.")


if __name__ == "__main__":
    main()
