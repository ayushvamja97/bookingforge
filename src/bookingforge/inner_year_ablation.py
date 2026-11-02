"""Chronological, training-only ablation of arrival-year as a predictor.

The inner split is *within* the previously defined historical training creation
cohort. No outer validation or final-test outcomes are read or used. Training
only uses outcomes known by the inner fit cutoff (2016-03-31); inner validation
uses statuses known by 2016-08-31, retrospectively, for scoring only.

This inner evaluation is still selected by label maturity and is NOT a clean
prospective benchmark. Its purpose is a traceable model-design experiment.
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

from .cohort import (
    TRAIN_CUTOFF, parse_arrival, reconstructed_creation,
    validate_booking_features,
)
from .mlp_numpy import NumpyMLPBinary
from .train_baseline import CATEGORICAL, FEATURES, NUMERIC, make_frame, metrics_for

INNER_TRAIN_END = date(2016, 3, 31)
INNER_VALID_START = date(2016, 4, 1)
INNER_VALID_END = date(2016, 5, 31)
INNER_VALID_LABEL_OBSERVED_BY = TRAIN_CUTOFF


def load_inner_chronological(csv_path: str | Path):
    """Only historical training-era creation cohorts; exclude all future labels.

    Inner train: booking created <= March 31, status known by March 31.
    Inner validation: booked April 1-May 31, outcome known by August 31.
    Inner validation target is only accessed to evaluate *after* fitting.
    """
    samples = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    counts = Counter()
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Dataset is missing header")
        missing = set(FEATURES) | {"reservation_status_date", "is_canceled"}
        missing -= set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        for row in reader:
            counts["total_rows"] += 1
            try:
                created = reconstructed_creation(parse_arrival(row), row["lead_time"])
            except (ValueError, TypeError, OverflowError):
                counts["invalid_booking_creation"] += 1
                continue
            if created <= INNER_TRAIN_END:
                cohort = "train"
                label_cutoff = INNER_TRAIN_END
            elif INNER_VALID_START <= created <= INNER_VALID_END:
                cohort = "validation"
                label_cutoff = INNER_VALID_LABEL_OBSERVED_BY
            else:
                # Important: do not access outcomes for outer validation/test.
                counts["not_inner_cohort"] += 1
                continue

            counts[f"{cohort}_creation_records"] += 1
            try:
                known_at = date.fromisoformat((row["reservation_status_date"] or "").strip())
                label = int(row["is_canceled"])
                if label not in (0, 1):
                    raise ValueError("Not a binary target")
            except (TypeError, ValueError):
                counts[f"{cohort}_invalid_outcome"] += 1
                continue
            if known_at < created:
                counts[f"{cohort}_status_before_creation"] += 1
                continue
            if known_at > label_cutoff:
                counts[f"{cohort}_outcome_not_known_by_cutoff"] += 1
                continue

            samples[cohort].append({feature: row.get(feature, "") for feature in FEATURES})
            labels[cohort].append(label)
            counts[f"{cohort}_eligible"] += 1
    return samples, labels, dict(sorted(counts.items()))


def experiment_config(include_arrival_year: bool):
    if type(include_arrival_year) is not bool:
        raise ValueError("include_arrival_year must be boolean")
    numeric = tuple(
        name for name in NUMERIC
        if include_arrival_year or name != "arrival_date_year"
    )
    categorical = tuple(CATEGORICAL)
    chosen = list(numeric + categorical)
    validate_booking_features(chosen)
    return numeric, categorical, chosen


def model_comparison(csv_path: str | Path, mlp_epochs: int = 30) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if type(mlp_epochs) is not int or mlp_epochs < 1:
        raise ValueError("mlp_epochs must be a positive integer")

    samples, labels, counts = load_inner_chronological(csv_path)
    if not samples["train"] or not samples["validation"]:
        raise ValueError("Empty inner chronological training or validation cohort")

    train = make_frame(samples["train"])
    valid = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_valid = np.asarray(labels["validation"], dtype=np.int64)
    if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise ValueError("Both outcome classes required in both inner cohorts")

    print(f"INNER chronological train: {len(train):,} | validation: {len(valid):,}", flush=True)
    result = {}
    for with_year in (True, False):
        variant = "with_arrival_year" if with_year else "without_arrival_year"
        numeric, categorical, chosen = experiment_config(with_year)
        print(f"Fitting {variant} on INNER train only...", flush=True)
        preprocess = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), list(numeric)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ],
            sparse_threshold=0.0,
        )
        x_train = np.asarray(preprocess.fit_transform(train[chosen]), dtype=np.float64)
        x_valid = np.asarray(preprocess.transform(valid[chosen]), dtype=np.float64)

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

        def stats(pred):
            ans = metrics_for(y_valid, pred)
            ans["mean_predicted_probability"] = round(float(pred.mean()), 6)
            ans["fraction_above_0_99"] = round(float((pred > 0.99).mean()), 6)
            return ans

        result[variant] = {
            "features": chosen,
            "transformed_dimensions": int(x_train.shape[1]),
            "models": {
                "logistic_regression": stats(logistic_prob),
                "custom_numpy_mlp": stats(mlp_prob),
            },
            "mlp_final_training_objective": mlp.training_loss_history_[-1],
        }

    path = Path(csv_path)
    with path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "experiment": "bookingforge_training_only_chronological_year_ablation_v1",
        "dataset_sha256": fingerprint,
        "inner_train_end": INNER_TRAIN_END.isoformat(),
        "inner_valid_creation_start": INNER_VALID_START.isoformat(),
        "inner_valid_creation_end": INNER_VALID_END.isoformat(),
        "inner_valid_status_observed_by": INNER_VALID_LABEL_OBSERVED_BY.isoformat(),
        "inner_training_rows": int(len(train)),
        "inner_validation_rows": int(len(valid)),
        "inner_training_cancel_fraction": round(float(y_train.mean()), 6),
        "inner_validation_cancel_fraction": round(float(y_valid.mean()), 6),
        "mlp_epochs": mlp_epochs,
        "variants": result,
        "cohort_counts": counts,
        "outer_validation_labels_accessed": False,
        "final_test_outcomes_accessed": False,
        "warning": (
            "Both inner cohorts are selected by eventual label maturity. "
            "Creation dates and booking-time feature values are proxies. "
            "This is an exploratory training-only ablation, not a prospective "
            "generalization claim. The year ablation was motivated by prior "
            "inspection of outer validation diagnostics; outer validation "
            "should not be considered independent for future model selection."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Historical train-only arrival-year ablation")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/inner_year_ablation.json")
    parser.add_argument("--mlp-epochs", type=int, default=30)
    args = parser.parse_args()
    report = model_comparison(args.csv, mlp_epochs=args.mlp_epochs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print("\nTRAINING-ONLY ARRIVAL-YEAR ABLATION COMPLETE")
    print(
        f"Inner train: {report['inner_training_rows']:,}, "
        f"inner validation: {report['inner_validation_rows']:,}"
    )
    for variant, data in report["variants"].items():
        print(variant)
        for model, metrics in data["models"].items():
            print(
                f"  {model}: ROC AUC={metrics['roc_auc']:.4f}; "
                f"Brier={metrics['brier_score']:.4f}; "
                f"AP={metrics['average_precision']:.4f}; "
                f"mean predicted P={metrics['mean_predicted_probability']:.3f}; "
                f"P>0.99={metrics['fraction_above_0_99']:.1%}"
            )
    print(f"Aggregate report: {output}")
    print("Outer validation and final test outcomes were NOT accessed.")


if __name__ == "__main__":
    main()
