"""Retrospective ARRIVAL-time algorithm benchmark, not booking-time prediction.

The original BookingForge booking-creation label-maturity cohort produced a
structural artifact: when a booking arriving in 2017 had already resolved
by March 2016, it could not have checked out. Here we define a different,
clearly labeled exercise: classify eventual status across separated ARRIVAL
date cohorts, with only training outcomes known before validation arrivals.
This removes that particular maturity-by-future-arrival artifact, but does
not establish original-booking-time feature availability or deployability.

Inner training: arrivals 2015-07-01 through 2016-08-31, final status
recorded by 2016-09-30. September 2016 arrivals are held out as a gap.
Retrospective validation: arrivals 2016-10-01 through 2016-12-31, all
eventual final outcomes. All arrivals in 2017+ are excluded BEFORE reading
any outcome fields. Model-selection contamination from earlier exploratory
validation investigations remains a limitation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from .cohort import parse_arrival, reconstructed_creation, validate_booking_features
from .mlp_numpy import NumpyMLPBinary
from .train_baseline import FEATURES, NUMERIC, CATEGORICAL, make_frame, metrics_for

TRAIN_FIRST_ARRIVAL = date(2015, 7, 1)
TRAIN_LAST_ARRIVAL = date(2016, 8, 31)
FIT_INFORMATION_CUTOFF = date(2016, 9, 30)
VALID_FIRST_ARRIVAL = date(2016, 10, 1)
VALID_LAST_ARRIVAL = date(2016, 12, 31)


def load_arrival_benchmark(csv_path: str | Path):
    """Select eligible pre-validation arrival cohorts, preserving sealed 2017."""
    samples = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    counters = Counter()
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Missing CSV header")
        required = set(FEATURES) | {"reservation_status_date", "is_canceled"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in reader:
            counters["total_rows"] += 1
            try:
                arrival = parse_arrival(row)
            except (TypeError, ValueError, OverflowError):
                counters["invalid_arrival_date"] += 1
                continue
            if TRAIN_FIRST_ARRIVAL <= arrival <= TRAIN_LAST_ARRIVAL:
                subset = "train"
            elif VALID_FIRST_ARRIVAL <= arrival <= VALID_LAST_ARRIVAL:
                subset = "validation"
            else:
                counters["excluded_outside_target_arrival_windows"] += 1
                # Never touch labels of 2017 arrivals, or the September gap.
                continue
            counters[f"{subset}_arrivals"] += 1
            try:
                creation = reconstructed_creation(arrival, row["lead_time"])
                observed = date.fromisoformat((row["reservation_status_date"] or "").strip())
                outcome = int(row["is_canceled"])
                if outcome not in (0, 1):
                    raise ValueError("Expected 0/1 target")
                if observed < creation:
                    raise ValueError("Final outcome precedes inferred creation")
            except (ValueError, TypeError, OverflowError):
                counters[f"{subset}_invalid_record"] += 1
                continue
            if subset == "train" and observed > FIT_INFORMATION_CUTOFF:
                counters["train_outcome_not_yet_known_at_fit_cutoff"] += 1
                continue
            if subset == "validation" and observed > VALID_LAST_ARRIVAL:
                counters["validation_outcome_observed_after_valid_arrival_window"] += 1
            record = {feature: row.get(feature, "") for feature in FEATURES}
            samples[subset].append(record)
            labels[subset].append(outcome)
            counters[f"{subset}_included"] += 1
    return samples, labels, dict(sorted(counters.items()))


def selected_features(include_year: bool):
    if type(include_year) is not bool:
        raise ValueError("include_year must be boolean")
    numeric = tuple(
        name for name in NUMERIC
        if include_year or name != "arrival_date_year"
    )
    categorical = tuple(CATEGORICAL)
    all_features = list(numeric + categorical)
    validate_booking_features(all_features)
    return numeric, categorical, all_features


def run_arrival_benchmark(csv_path: str | Path, mlp_epochs: int = 30):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if type(mlp_epochs) is not int or mlp_epochs < 1:
        raise ValueError("mlp_epochs must be a positive integer")
    samples, labels, counters = load_arrival_benchmark(csv_path)
    if not samples["train"] or not samples["validation"]:
        raise ValueError("Training and validation must have eligible bookings")
    train = make_frame(samples["train"])
    valid = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_valid = np.asarray(labels["validation"], dtype=np.int64)
    if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise ValueError("Both classes required in each cohort")

    prior = np.full(len(y_valid), y_train.mean(), dtype=np.float64)
    print(
        f"Arrival-based retrospective benchmark: train={len(y_train):,}, "
        f"validation={len(y_valid):,}.",
        flush=True,
    )
    variants = {}
    for include_year in (True, False):
        variant = "with_arrival_year" if include_year else "without_arrival_year"
        numeric, categorical, features = selected_features(include_year)
        print(f"Training {variant} on all eligible historical arrival records...", flush=True)
        preprocessing = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), list(numeric)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ],
            sparse_threshold=0.0,
        )
        x_train = np.asarray(preprocessing.fit_transform(train[features]), dtype=np.float64)
        x_valid = np.asarray(preprocessing.transform(valid[features]), dtype=np.float64)
        logistic = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42
        ).fit(x_train, y_train)
        logistic_prob = logistic.predict_proba(x_valid)[:, 1]
        mlp = NumpyMLPBinary(
            hidden_units=32, epochs=mlp_epochs, batch_size=256,
            learning_rate=0.001, weight_decay=0.0001, seed=42,
        )
        def progress(epoch, total, loss):
            if epoch == 1 or epoch == total or epoch % 5 == 0:
                print(f"  {variant} epoch {epoch}/{total}: objective={loss:.6f}", flush=True)
        mlp.fit(x_train, y_train, on_epoch=progress)
        mlp_prob = mlp.predict_proba_positive(x_valid)

        def summary(prob):
            score = metrics_for(y_valid, prob)
            score["mean_probability"] = round(float(prob.mean()), 6)
            score["fraction_above_0_99"] = round(float((prob > 0.99).mean()), 6)
            return score

        variants[variant] = {
            "features": features,
            "training_records_used": len(y_train),
            "validation_records_scored": len(y_valid),
            "encoded_dimension": int(x_train.shape[1]),
            "metrics": {
                "logistic_regression": summary(logistic_prob),
                "custom_numpy_mlp": summary(mlp_prob),
            },
            "mlp_training_objective_by_epoch": mlp.training_loss_history_,
        }
    path = Path(csv_path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "experiment": "bookingforge_retrospective_arrival_cohort_benchmark_v1",
        "evaluation_scope": "RETROSPECTIVE ARRIVAL-COHORT ALGORITHM STUDY ONLY",
        "dataset_sha256": sha,
        "train_arrival_period": [TRAIN_FIRST_ARRIVAL.isoformat(), TRAIN_LAST_ARRIVAL.isoformat()],
        "train_status_known_by": FIT_INFORMATION_CUTOFF.isoformat(),
        "validation_arrival_period": [VALID_FIRST_ARRIVAL.isoformat(), VALID_LAST_ARRIVAL.isoformat()],
        "validation_outcomes": "eventual status, used for scoring only",
        "train_records": len(y_train),
        "validation_records": len(y_valid),
        "train_outcome_rate": round(float(y_train.mean()), 6),
        "validation_outcome_rate": round(float(y_valid.mean()), 6),
        "prior_baseline": metrics_for(y_valid, prior),
        "variants": variants,
        "cohort_counters": counters,
        "final_2017_outcomes_accessed": False,
        "warnings": [
            "Different research task from booking-creation historical cutoff protocol; do not compare AUCs directly.",
            "Prediction time is NOT verified at booking creation: features are not historical immutable snapshots.",
            "Training is slightly selected for outcomes available by September 30 after arrival window.",
            "Validation eventually finalized outcome labels are used retrospectively for scoring only.",
            "Earlier validation exploration motivates this design; claims of independent generalization would be inappropriate.",
            "Arrival-window-bounded dataset, duplicate-booking audit and feature timing remain limitations.",
            "Final year 2017 arrival records were excluded before reading outcome/status columns.",
            "No cancellation-risk deployment, revenue, or operational recommendation is supported.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Retrospective arrival cohort: logistic and original MLP")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/arrival_benchmark.json")
    parser.add_argument("--mlp-epochs", type=int, default=30)
    args = parser.parse_args()
    result = run_arrival_benchmark(args.csv, mlp_epochs=args.mlp_epochs)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nARRIVAL-BASED RETROSPECTIVE BENCHMARK COMPLETE")
    print(
        f"Train: {result['train_records']:,} ({result['train_outcome_rate']:.1%} cancellations)"
        f"; validation: {result['validation_records']:,} "
        f"({result['validation_outcome_rate']:.1%} cancellations)"
    )
    print("Prior baseline:", result["prior_baseline"])
    for variant, info in result["variants"].items():
        print(variant)
        for model_name, m in info["metrics"].items():
            print(
                f"  {model_name}: AUC={m['roc_auc']:.4f}, AP={m['average_precision']:.4f}, "
                f"Brier={m['brier_score']:.4f}, mean_p={m['mean_probability']:.3f}, "
                f"P>0.99={m['fraction_above_0_99']:.1%}"
            )
    print("Cohort counters:", result["cohort_counters"])
    print(f"JSON report: {dest}")
    print("LIMITATION: This is arrival-based retrospective classification, NOT booking-time prediction.")
    print("2017 arrival labels were NOT accessed.")


if __name__ == "__main__":
    main()
