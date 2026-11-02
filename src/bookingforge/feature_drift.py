"""Label-blind input drift audit for the full retrospective validation cohort.

Training is restricted to outcome-mature historical records. Validation rows
match the full retrospective scoring cohort. NO labels, predictions or outcome
statistics appear in the output, and final test fields are never loaded.
This describes distribution shift; it does not attribute model causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .train_baseline import CATEGORICAL, NUMERIC, load_matured_cohorts, make_frame


def numeric_drift(train, validation):
    """Compare distributions against the TRAINING support and scale only."""
    a = np.asarray(train, dtype=np.float64)
    b = np.asarray(validation, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            "train_observed": int(len(a)),
            "validation_observed": int(len(b)),
            "insufficient_data": True,
        }
    sd = float(a.std())
    train_lo, train_hi = float(a.min()), float(a.max())
    mean_a, mean_b = float(a.mean()), float(b.mean())
    return {
        "train_observed": int(len(a)),
        "validation_observed": int(len(b)),
        "train_mean": round(mean_a, 6),
        "validation_mean": round(mean_b, 6),
        "train_std": round(sd, 6),
        "standardized_mean_shift": round((mean_b - mean_a) / sd, 6) if sd > 0 else None,
        "train_min": round(train_lo, 6),
        "train_max": round(train_hi, 6),
        "train_median": round(float(np.median(a)), 6),
        "validation_median": round(float(np.median(b)), 6),
        "train_p10_p90": [round(float(n), 6) for n in np.quantile(a, [0.1, 0.9])],
        "validation_p10_p90": [round(float(n), 6) for n in np.quantile(b, [0.1, 0.9])],
        "validation_fraction_outside_train_min_max": round(
            float(((b < train_lo) | (b > train_hi)).mean()), 6
        ),
        "validation_fraction_outside_train_p10_p90": round(
            float(((b < np.quantile(a, 0.1)) | (b > np.quantile(a, 0.9))).mean()), 6
        ),
    }


def categorical_drift(train, validation):
    """Measure unseen categories without consulting any validation labels."""
    a = [str(v).strip() if v is not None else "<missing>" for v in train]
    b = [str(v).strip() if v is not None else "<missing>" for v in validation]
    a = [v if v else "<missing>" for v in a]
    b = [v if v else "<missing>" for v in b]
    support = set(a)
    counts_a, counts_b = Counter(a), Counter(b)
    return {
        "train_distinct_categories": len(counts_a),
        "validation_distinct_categories": len(counts_b),
        "unseen_validation_category_values": sorted(set(b) - support)[:20],
        "unseen_validation_category_count": len(set(b) - support),
        "validation_fraction_unseen_category": round(
            sum(v not in support for v in b) / len(b), 6
        ) if b else None,
        "train_categories": dict(sorted(counts_a.items())),
        "validation_categories": dict(sorted(counts_b.items())),
    }


def audit_input_drift(csv_path: str | Path) -> dict:
    rows, _ignored_labels, counts = load_matured_cohorts(
        csv_path, validation_mode="all_eventual"
    )
    train = make_frame(rows["train"])
    valid = make_frame(rows["validation"])
    if len(train) == 0 or len(valid) == 0:
        raise ValueError("Training and validation cohorts must be nonempty")

    numeric = {
        name: numeric_drift(train[name], valid[name]) for name in NUMERIC
    }
    categorical = {
        name: categorical_drift(train[name], valid[name]) for name in CATEGORICAL
    }
    path = Path(csv_path)
    with path.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()

    return {
        "name": "bookingforge_feature_drift_v1",
        "dataset_sha256": fingerprint,
        "train_records": len(train),
        "validation_records": len(valid),
        "numeric_feature_shift": numeric,
        "categorical_feature_shift": categorical,
        "cohort_counts": counts,
        "individual_records_exported": False,
        "outcome_values_used_to_compute_drift": False,
        "cohort_eligibility_required_outcome_dates_and_training_labels": True,
        "final_test_outcomes_accessed": False,
        "interpretation": (
            "Descriptive covariate shift between training and validation input "
            "distributions. It cannot prove which features caused model errors. "
            "Training is selected by historical label maturity; validation is "
            "retrospective and the source dataset is arrival-window bounded."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="BookingForge training-vs-validation input drift")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/feature_drift.json")
    args = parser.parse_args()
    report = audit_input_drift(args.csv)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BookingForge label-blind feature drift audit complete")
    print(f"Training rows: {report['train_records']:,}; validation rows: {report['validation_records']:,}")
    print("Numeric features sorted by absolute standardized mean shift:")
    for name, values in sorted(
        report["numeric_feature_shift"].items(),
        key=lambda item: abs(item[1].get("standardized_mean_shift") or 0),
        reverse=True,
    ):
        print(
            f"  {name}: train_mean={values.get('train_mean')} "
            f"validation_mean={values.get('validation_mean')} "
            f"shift={values.get('standardized_mean_shift')} "
            f"outside_train_range={values.get('validation_fraction_outside_train_min_max')}"
        )
    print("Categorical drift:")
    for name, values in report["categorical_feature_shift"].items():
        print(
            f"  {name}: unseen_validation_fraction={values['validation_fraction_unseen_category']}; "
            f"training={values['train_categories']}; validation={values['validation_categories']}"
        )
    print(f"Aggregate report: {dest}")
    print("Diagnostic only. No model was trained and final test labels were not accessed.")


if __name__ == "__main__":
    main()
