"""Descriptive audit for the *inner* historical chronological ablation cohorts.

This module makes NO fitting decisions and trains NO models. It inspects
outcomes only inside the historical inner training and scoring cohorts that
are already explicitly gated by historical label-availability dates.
Outer validation and final-test outcome columns remain sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .inner_year_ablation import (
    INNER_TRAIN_END, INNER_VALID_END, INNER_VALID_LABEL_OBSERVED_BY,
    INNER_VALID_START, load_inner_chronological,
)
from .train_baseline import make_frame, metrics_for


def population_description(rows, labels):
    """Class prevalence and arrival-year distribution of an eligible cohort."""
    if len(rows) == 0 or len(rows) != len(labels):
        raise ValueError("Require a nonempty cohort with aligned labels")
    frame = make_frame(rows)
    y = np.asarray(labels, dtype=np.int64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary outcome labels")
    years = np.asarray(frame["arrival_date_year"], dtype=np.float64)
    observed = np.isfinite(years)
    counts = Counter(
        int(year) for year in years[observed]
    )
    by_year = {}
    for year in sorted(counts):
        mask = years == year
        by_year[str(year)] = {
            "records": int(mask.sum()),
            "cancel_or_no_show_count": int(y[mask].sum()),
            "cancel_or_no_show_fraction": round(float(y[mask].mean()), 6),
        }
    hotel = {}
    for name in sorted(frame["hotel"].unique()):
        mask = frame["hotel"].to_numpy() == name
        hotel[str(name)] = {
            "records": int(mask.sum()),
            "cancel_or_no_show_fraction": round(float(y[mask].mean()), 6),
        }
    return {
        "records": int(len(y)),
        "cancel_or_no_show_count": int(y.sum()),
        "cancel_or_no_show_fraction": round(float(y.mean()), 6),
        "arrival_years": by_year,
        "hotel": hotel,
        "invalid_arrival_year_count": int((~observed).sum()),
    }


def diagnostic_report(csv_path):
    rows, labels, counts = load_inner_chronological(csv_path)
    train = population_description(rows["train"], labels["train"])
    valid = population_description(rows["validation"], labels["validation"])
    y_val = np.asarray(labels["validation"], dtype=np.int64)
    prior_p = float(train["cancel_or_no_show_count"] / train["records"])
    prior = np.full(y_val.shape, prior_p, dtype=np.float64)
    if np.unique(y_val).size == 2:
        baseline = metrics_for(y_val, prior)
    else:
        # Do not manufacture AUC on single-class synthetic fixtures.
        baseline = {
            "brier_score": round(float(np.mean((prior - y_val) ** 2)), 6),
            "roc_auc": None,
            "average_precision": None,
            "warning": "Only one class in scoring cohort",
        }
    path = Path(csv_path)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "experiment": "bookingforge_inner_training_cohort_composition_v1",
        "dataset_sha256": sha,
        "historical_training_cutoff": INNER_TRAIN_END.isoformat(),
        "inner_validation_creation_start": INNER_VALID_START.isoformat(),
        "inner_validation_creation_end": INNER_VALID_END.isoformat(),
        "inner_validation_status_cutoff": INNER_VALID_LABEL_OBSERVED_BY.isoformat(),
        "train": train,
        "validation": valid,
        "training_prior_on_inner_validation": baseline,
        "counted_ineligible_and_censored_records": counts,
        "outer_validation_labels_accessed": False,
        "final_test_outcomes_accessed": False,
        "warning": (
            "Both cohorts are selected on eventual outcome maturity. The "
            "source is arrival-window-bounded; reconstructed creation "
            "dates and original booking-time predictors are not verified. "
            "The constant-prior benchmark is descriptive, not a production estimate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Report inner split prevalence, year composition and simple prior score"
    )
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/inner_cohort_composition.json")
    args = parser.parse_args()
    report = diagnostic_report(args.csv)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nINNER HISTORICAL COHORT COMPOSITION COMPLETE")
    for name in ("train", "validation"):
        item = report[name]
        print(
            f"{name}: n={item['records']:,}, "
            f"cancellation/no-show prevalence={item['cancel_or_no_show_fraction']:.3%}"
        )
        print(f"  arrival years: {item['arrival_years']}")
        print(f"  hotels: {item['hotel']}")
    print("Inner prior baseline:", report["training_prior_on_inner_validation"])
    print("Maturity/selection counts:", report["counted_ineligible_and_censored_records"])
    print(f"Aggregate report: {dest}")
    print("Outer validation and final test labels were NOT accessed.")


if __name__ == "__main__":
    main()
