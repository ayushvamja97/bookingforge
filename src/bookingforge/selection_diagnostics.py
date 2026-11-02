"""Retrospective selection-bias diagnostics for BookingForge's preliminary baseline.

This module is NOT a training pipeline. It deliberately inspects final outcomes
of train/validation cohort records *after* their cutoff to quantify selection.
It does not inspect final test-cohort outcomes or alter training eligibility.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .cohort import (
    TRAIN_CUTOFF, VALID_CUTOFF, cohort_at, parse_arrival, reconstructed_creation,
)

REQUIRED = {
    "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
    "lead_time", "reservation_status_date", "reservation_status",
    "is_canceled", "hotel",
}


def lead_time_bucket(days: int) -> str:
    if days < 0:
        raise ValueError("Negative lead time")
    if days <= 7:
        return "0-7"
    if days <= 30:
        return "8-30"
    if days <= 90:
        return "31-90"
    return "91+"


def selection_diagnostics(csv_path: str | Path) -> dict:
    """Measure outcome-dependent cohort selection, without touching test labels."""
    counts = Counter()
    positives = Counter()
    by_hotel = Counter()
    by_lead = Counter()
    by_status = Counter()

    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV missing header")
        missing = REQUIRED - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Required columns missing: {sorted(missing)}")

        for row in reader:
            counts["all_rows"] += 1
            try:
                arrival = parse_arrival(row)
                lead = int(row["lead_time"])
                creation = reconstructed_creation(arrival, lead)
            except (ValueError, TypeError, OverflowError):
                counts["invalid_creation_proxy"] += 1
                continue

            cohort = cohort_at(creation)
            if cohort not in {"train", "validation"}:
                counts["not_train_or_validation"] += 1
                # Do not access test cohort label or status fields.
                continue

            cutoff = TRAIN_CUTOFF if cohort == "train" else VALID_CUTOFF
            try:
                status_date = date.fromisoformat((row["reservation_status_date"] or "").strip())
                target = int(row["is_canceled"])
                if target not in (0, 1):
                    raise ValueError("Invalid label")
                if status_date < creation:
                    raise ValueError("Final status before reconstructed creation")
            except (ValueError, TypeError):
                counts[f"{cohort}_invalid_outcome_or_status_date"] += 1
                continue

            selected = status_date <= cutoff
            state = "matured" if selected else "not_matured"
            key = (cohort, state)
            counts[key] += 1
            positives[key] += target
            hotel = (row["hotel"] or "").strip() or "<missing>"
            by_hotel[(cohort, state, hotel)] += 1
            by_lead[(cohort, state, lead_time_bucket(lead))] += 1
            status = (row["reservation_status"] or "").strip() or "<missing>"
            by_status[(cohort, state, status)] += 1

    cohorts = {}
    for cohort in ("train", "validation"):
        mature = counts[(cohort, "matured")]
        other = counts[(cohort, "not_matured")]
        total = mature + other
        positive_mature = positives[(cohort, "matured")]
        positive_other = positives[(cohort, "not_matured")]
        cohorts[cohort] = {
            "records": total,
            "matured_records": mature,
            "not_matured_records": other,
            "matured_fraction": round(mature / total, 6) if total else None,
            "final_cancel_rate_matured": round(positive_mature / mature, 6) if mature else None,
            "final_cancel_rate_not_matured": round(positive_other / other, 6) if other else None,
            "final_cancel_rate_all_records_retrospective": round(
                (positive_mature + positive_other) / total, 6
            ) if total else None,
        }

    return {
        "name": "bookingforge_retrospective_selection_diagnostic_v1",
        "cohorts": cohorts,
        "final_status_by_cohort_and_maturity": {
            " / ".join(key): value for key, value in sorted(by_status.items())
        },
        "hotel_counts_by_cohort_and_maturity": {
            " / ".join(key): value for key, value in sorted(by_hotel.items())
        },
        "lead_time_counts_by_cohort_and_maturity": {
            " / ".join(key): value for key, value in sorted(by_lead.items())
        },
        "other_counts": {
            k: v for k, v in sorted(counts.items()) if isinstance(k, str)
        },
        "test_outcomes_accessed": False,
        "warning": (
            "Retrospective analysis uses *eventual* outcomes to demonstrate "
            "selection bias. These eventual outcomes are NOT permitted for a "
            "historical cutoff training run. Booking-creation dates are proxies "
            "and the arrival-bounded dataset remains selected/truncated. "
            "Validation metrics cannot be generalized to all bookings."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mature-vs-unresolved outcome selection")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/selection_diagnostics.json")
    args = parser.parse_args()
    report = selection_diagnostics(args.csv)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BookingForge retrospective selection diagnostics complete")
    for cohort, results in report["cohorts"].items():
        print(f"{cohort.upper()}: {results}")
    print("Final status by maturity:", report["final_status_by_cohort_and_maturity"])
    print(f"Aggregate report: {dest}")
    print("WARNING: Retrospective outcomes are used for diagnosis only, never training.")


if __name__ == "__main__":
    main()
