"""Preliminary historical-cutoff training with retrospective validation.

Training uses ONLY outcomes known by the training cutoff. Validation uses
all eventual final labels of the distinct later-created cohort, regardless
of the validation cutoff; these labels are used for scoring, never fitting.
This avoids outcome-maturity selection in validation, but training remains
selected and the dataset has arrival-window selection and feature drift.
The final test cohort is never scored or inspected for labels here.

Run from repository root:
    python -m pip install -r requirements-ml.txt
    $env:PYTHONPATH = (Join-Path (Get-Location) "src")
    python -m bookingforge.train_baseline --csv data/raw/hotels.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .cohort import (
    PROVISIONAL_BOOKING_FEATURES,
    TRAIN_CUTOFF,
    VALID_CUTOFF,
    cohort_at,
    parse_arrival,
    reconstructed_creation,
    validate_booking_features,
)

NUMERIC = (
    "lead_time", "arrival_date_year", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
)
CATEGORICAL = ("hotel", "arrival_date_month", "reserved_room_type")
FEATURES = list(NUMERIC + CATEGORICAL)

if set(FEATURES) != PROVISIONAL_BOOKING_FEATURES:
    raise RuntimeError("Research predictor list is out of sync with the allowlist.")
validate_booking_features(FEATURES)


def load_matured_cohorts(csv_path: str | Path, validation_mode: str = "matured"):
    """Return outcome-mature train plus selected validation; never touch test labels.

    validation_mode="matured" preserves the historical cutoff diagnostic.
    validation_mode="all_eventual" retains every valid validation booking
    with a recorded final outcome for retrospective scoring ONLY. No later
    validation outcome is used for fitting any part of the pipeline.
    """
    if validation_mode not in {"matured", "all_eventual"}:
        raise ValueError("validation_mode must be matured or all_eventual")
    samples = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    counters = Counter()
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        expected = set(FEATURES) | {"is_canceled", "reservation_status_date"}
        missing = expected - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")

        for row in reader:
            counters["total_rows"] += 1
            try:
                arrival = parse_arrival(row)
                created = reconstructed_creation(arrival, row["lead_time"])
            except (TypeError, ValueError, OverflowError):
                counters["invalid_date_or_lead_time"] += 1
                continue

            cohort = cohort_at(created)
            counters[f"cohort_{cohort}"] += 1
            if cohort not in samples:
                # Preserve test separation: do not read test outcomes, including
                # reservation_status_date or is_canceled.
                continue

            cutoff = TRAIN_CUTOFF if cohort == "train" else VALID_CUTOFF
            try:
                known_at = date.fromisoformat((row["reservation_status_date"] or "").strip())
                label = int(row["is_canceled"])
                if label not in (0, 1):
                    raise ValueError("Invalid binary label")
            except (TypeError, ValueError):
                counters[f"{cohort}_invalid_label_or_status_date"] += 1
                continue

            if known_at > cutoff:
                counters[f"{cohort}_outcome_unknown_at_cutoff"] += 1
                if cohort == "train" or validation_mode == "matured":
                    continue
                counters["validation_scored_using_eventual_outcome"] += 1

            # Guard impossible chronology; raise rather than silently train on it.
            if known_at < created:
                counters[f"{cohort}_status_before_creation"] += 1
                continue

            record = {feature: row.get(feature, "") for feature in FEATURES}
            samples[cohort].append(record)
            labels[cohort].append(label)
            counters[f"{cohort}_included"] += 1

    return samples, labels, dict(sorted(counters.items()))


def make_frame(rows):
    import pandas as pd

    frame = pd.DataFrame(rows, columns=FEATURES)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in CATEGORICAL:
        frame[col] = frame[col].fillna("MISSING").replace("", "MISSING")
    return frame


def metrics_for(y_true, prob):
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
        accuracy_score,
        f1_score,
    )
    y_pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 6),
        "average_precision": round(float(average_precision_score(y_true, prob)), 6),
        "brier_score": round(float(brier_score_loss(y_true, prob)), 6),
        "log_loss": round(float(log_loss(y_true, prob, labels=[0, 1])), 6),
        "accuracy_at_0_5": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_at_0_5": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def run_experiment(csv_path: str | Path) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import numpy as np

    samples, labels, counts = load_matured_cohorts(csv_path, validation_mode="all_eventual")
    if not samples["train"] or not samples["validation"]:
        raise ValueError("No eligible train/validation rows after label maturity screening")

    x_train = make_frame(samples["train"])
    x_valid = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=int)
    y_valid = np.asarray(labels["validation"], dtype=int)
    if len(set(y_train.tolist())) != 2 or len(set(y_valid.tolist())) != 2:
        raise ValueError("Both classes must exist in each evaluation cohort")

    # No preprocessing operation sees validation features during fitting.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), list(NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
    )
    classifier = Pipeline([
        ("preprocess", preprocessor),
        ("classify", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
    ])
    classifier.fit(x_train, y_train)
    logistic_probs = classifier.predict_proba(x_valid)[:, 1]

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_probs = dummy.predict_proba(np.zeros((len(y_valid), 1)))[:, list(dummy.classes_).index(1)]

    p = Path(csv_path)
    with p.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()  # read-only dataset fingerprint

    return {
        "experiment": "bookingforge_retrospective_full_validation_baseline_v2",
        "dataset_file": p.name,
        "dataset_sha256": sha,
        "evaluation_status": "PRELIMINARY / TRAIN-CUTOFF LABELS + FULL RETROSPECTIVE VALIDATION / NOT DEPLOYMENT",
        "fit_information_cutoff": TRAIN_CUTOFF.isoformat(),
        "validation_creation_cohort_end": VALID_CUTOFF.isoformat(),
        "validation_label_observation": "eventual final outcomes; not used for fitting",
        "features": FEATURES,
        "counts": counts,
        "training_rows": len(y_train),
        "validation_rows": len(y_valid),
        "train_positive_fraction": round(float(y_train.mean()), 6),
        "validation_positive_fraction": round(float(y_valid.mean()), 6),
        "validation_only_metrics": {
            "prior_probability": metrics_for(y_valid, baseline_probs),
            "logistic_regression": metrics_for(y_valid, logistic_probs),
        },
        "test_set_evaluated": False,
        "limits": [
            "Creation dates reconstructed from arrival date minus lead time.",
            "Original dataset is bounded by arrival date, not creation date.",
            "Training labels are selected to be known by fit cutoff, so training remains biased.",
            "Validation includes all eventual outcomes but is only retrospective.",
            "Features are assumed booking-time proxies; historical snapshots are unavailable.",
            "Threshold 0.5 is not optimized or calibrated for a business decision.",
            "Metrics reflect retrospective classification on resolved cases, not production performance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BookingForge preliminary cancellation baseline")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/baseline_validation.json")
    args = parser.parse_args()
    report = run_experiment(args.csv)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("PRELIMINARY BookingForge ML baseline complete")
    print(f"Training records: {report['training_rows']:,}")
    print(f"Validation records: {report['validation_rows']:,}")
    print(f"Training cancel/no-show fraction: {report['train_positive_fraction']:.3%}")
    print(f"Validation cancel/no-show fraction: {report['validation_positive_fraction']:.3%}")
    for name, values in report["validation_only_metrics"].items():
        print(f"{name}: ROC AUC={values['roc_auc']:.4f}; Brier={values['brier_score']:.4f}; AP={values['average_precision']:.4f}")
    print(f"Report: {dest}")
    print("WARNING: Retrospective validation uses all eventual labels; training selection bias and source limitations remain.")
    print("Final test set was NOT evaluated.")


if __name__ == "__main__":
    main()
