"""Post-fit, validation-label-blind input intervention for original BookingForge models.

The observed validation booking features are held fixed except for replacing
one column at a time with the median seen in training. We only inspect how
predicted probabilities change; NO validation target is used for selection,
fitting, metrics, or choosing a replacement. Replacements can create
unrealistic synthetic rows and are NOT causal effects or approved fixes.

All available historical training labels were known by 2016-08-31. Test
cohort target fields remain unread by the underlying loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .mlp_numpy import NumpyMLPBinary
from .train_baseline import CATEGORICAL, NUMERIC, load_matured_cohorts, make_frame


def prediction_shift(original, intervened) -> dict:
    """Compare aligned probability arrays without using outcome labels."""
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(intervened, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("Aligned, nonempty probability vectors required")
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a < 0).any() or (a > 1).any()
            or (b < 0).any() or (b > 1).any()):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    delta = b - a
    return {
        "n": int(len(a)),
        "original_mean_probability": round(float(a.mean()), 6),
        "intervened_mean_probability": round(float(b.mean()), 6),
        "mean_signed_delta": round(float(delta.mean()), 6),
        "mean_absolute_delta": round(float(np.abs(delta).mean()), 6),
        "fraction_absolute_delta_over_0_10": round(float((np.abs(delta) > 0.10).mean()), 6),
        "original_fraction_probability_over_0_99": round(float((a > 0.99).mean()), 6),
        "intervened_fraction_probability_over_0_99": round(float((b > 0.99).mean()), 6),
    }


def replace_numeric_with_training_median(training, validation, column):
    """Return an independent copy; replacement learned from training only."""
    if column not in training.columns or column not in validation.columns:
        raise ValueError(f"Unknown input column {column}")
    values = training[column].dropna()
    if values.empty:
        raise ValueError(f"No observed training values for {column}")
    median = float(values.median())
    result = validation.copy(deep=True)
    result.loc[:, column] = median
    return result, median


def run_sensitivity(csv_path: str | Path, mlp_epochs: int = 30) -> dict:
    """Train fixed original models and perform label-blind validation probes."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if type(mlp_epochs) is not int or mlp_epochs < 1:
        raise ValueError("mlp_epochs must be a positive integer")
    samples, labels, counts = load_matured_cohorts(
        csv_path, validation_mode="all_eventual"
    )
    train = make_frame(samples["train"])
    validation = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    # Validation labels are not read or used in the intervention analysis.
    if not len(train) or not len(validation) or len(set(y_train.tolist())) != 2:
        raise ValueError("Insufficient historical training/validation cohorts")

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), list(NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ],
        sparse_threshold=0.0,
    )
    x_train = np.asarray(preprocess.fit_transform(train), dtype=np.float64)
    x_valid = np.asarray(preprocess.transform(validation), dtype=np.float64)

    print("Refitting fixed logistic regression on historical training records...", flush=True)
    logistic = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42
    ).fit(x_train, y_train)
    print("Refitting original NumPy MLP on historical training records...", flush=True)
    mlp = NumpyMLPBinary(
        hidden_units=32, epochs=mlp_epochs,
        batch_size=256, learning_rate=0.001,
        weight_decay=0.0001, seed=42,
    )
    mlp.fit(
        x_train, y_train,
        on_epoch=lambda epoch, total, loss: print(
            f"  epoch {epoch}/{total}: training objective={loss:.6f}", flush=True
        ) if epoch == 1 or epoch == total or epoch % 5 == 0 else None,
    )
    baseline = {
        "logistic_regression": logistic.predict_proba(x_valid)[:, 1],
        "numpy_mlp": mlp.predict_proba_positive(x_valid),
    }
    observed_years = validation["arrival_date_year"].to_numpy()
    year_counts = Counter(str(int(year)) for year in observed_years if np.isfinite(year))

    interventions = {}
    # The lead-time intervention is a diagnostic comparison, not a proposed
    # modification of valid reservations. Do not use validation labels.
    for column in ("arrival_date_year", "lead_time"):
        changed, neutral = replace_numeric_with_training_median(
            train, validation, column
        )
        x_changed = np.asarray(preprocess.transform(changed), dtype=np.float64)
        altered = {
            "logistic_regression": logistic.predict_proba(x_changed)[:, 1],
            "numpy_mlp": mlp.predict_proba_positive(x_changed),
        }
        interventions[column] = {
            "training_median_replacement": neutral,
            "model_prediction_changes": {
                name: prediction_shift(baseline[name], altered[name])
                for name in baseline
            },
            "mlp_by_original_arrival_year": {
                str(int(year)): prediction_shift(
                    baseline["numpy_mlp"][observed_years == year],
                    altered["numpy_mlp"][observed_years == year],
                )
                for year in sorted(set(observed_years))
                if np.isfinite(year) and np.count_nonzero(observed_years == year)
            },
        }

    path = Path(csv_path)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "experiment": "bookingforge_label_blind_feature_intervention_v1",
        "dataset_sha256": sha,
        "train_records": len(train),
        "validation_records": len(validation),
        "observed_validation_arrival_year_counts": dict(sorted(year_counts.items())),
        "original_model_mean_probabilities": {
            name: round(float(prob.mean()), 6) for name, prob in baseline.items()
        },
        "interventions": interventions,
        "cohort_counts": counts,
        "mlp_epochs": mlp_epochs,
        "validation_labels_used_in_interventions": False,
        "validation_used_in_fitting_or_early_stopping": False,
        "final_test_outcomes_accessed": False,
        "limits": [
            "Replacing one feature creates synthetic, possibly unrealistic records.",
            "Prediction changes under a feature intervention are not causal effects.",
            "Historical training is selected by outcome maturity.",
            "Only provisional booking-time feature snapshots are available.",
            "No hyperparameter tuning, calibration, or validation-based feature selection is authorized.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect arrival-year prediction sensitivity")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/feature_sensitivity.json")
    args = parser.parse_args()
    report = run_sensitivity(args.csv)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nBOOKINGFORGE PREDICTION SENSITIVITY AUDIT COMPLETE")
    print("Validation arrival-year counts:", report["observed_validation_arrival_year_counts"])
    for column, result in report["interventions"].items():
        print(f"Replace {column} with training median {result['training_median_replacement']}")
        for name, metrics in result["model_prediction_changes"].items():
            print(
                f"  {name}: original mean={metrics['original_mean_probability']:.3f}; "
                f"modified mean={metrics['intervened_mean_probability']:.3f}; "
                f"average |change|={metrics['mean_absolute_delta']:.3f}; "
                f"modified p>0.99={metrics['intervened_fraction_probability_over_0_99']:.1%}"
            )
        for year, metrics in result["mlp_by_original_arrival_year"].items():
            print(
                f"  NumPy MLP — arrivals {year}: n={metrics['n']:,}; "
                f"original mean={metrics['original_mean_probability']:.3f}; "
                f"modified mean={metrics['intervened_mean_probability']:.3f}"
            )
    print(f"Aggregate report: {destination}")
    print("Diagnostic only: modified feature values are NOT a validated model fix.")
    print("Final test cohort labels remained sealed.")


if __name__ == "__main__":
    main()
