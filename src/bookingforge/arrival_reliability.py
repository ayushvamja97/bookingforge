"""Fixed-model, descriptive reliability audit of arrival-separated validation.

This does NOT select features, hyperparameters, thresholds, models, or
calibration functions. Fits the same preregistered algorithms as
bookingforge.arrival_benchmark and computes subgroup/decile reliability
diagnostics plus simple *paired row-bootstrap* uncertainty estimates.

2017 arrivals are excluded by load_arrival_benchmark BEFORE reading labels.
Legacy aggregate audits have inspected 2017 outcomes, so the overall
historical data is not an entirely pristine blind holdout.

Row bootstrap intervals condition on already-trained models and observed
validation rows; they ignore retraining variability, clustering, calendar
autocorrelation, model selection and repeated exploration. DESCRIPTIVE ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .arrival_benchmark import load_arrival_benchmark, selected_features
from .mlp_numpy import NumpyMLPBinary
from .train_baseline import make_frame, metrics_for


def _aligned_binary(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.size == 0 or y.shape != p.shape:
        raise ValueError("Expected nonempty, aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite values from 0 to 1")
    return y.astype(np.int64), p


def reliability_bins(y_true, probabilities, n_bins: int = 10):
    """Equal-width bins [0,.1),..., [.9,1], with explicit empty bins."""
    if type(n_bins) is not int or n_bins < 2 or n_bins > 100:
        raise ValueError("n_bins must be an integer between 2 and 100")
    y, p = _aligned_binary(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assigned = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        mask = assigned == i
        count = int(mask.sum())
        pred_mean = float(p[mask].mean()) if count else None
        true_rate = float(y[mask].mean()) if count else None
        if count:
            ece += (count / len(y)) * abs(pred_mean - true_rate)
        bins.append({
            "lower_inclusive": round(float(edges[i]), 6),
            "upper_inclusive_only_in_last_bin": round(float(edges[i + 1]), 6),
            "count": count,
            "predicted_mean": round(pred_mean, 6) if count else None,
            "observed_frequency": round(true_rate, 6) if count else None,
        })
    return {
        "bins": bins,
        "expected_calibration_error_equal_width": round(ece, 6),
        "total_records": len(y),
        "notes": "ECE is descriptive, bin-dependent and not used to fit a calibrator.",
    }


def subgroup_metrics(y_true, probabilities):
    """Scores with gracefully undefined AUC/AP when only one outcome class."""
    from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

    y, p = _aligned_binary(y_true, probabilities)
    has_both = len(np.unique(y)) == 2
    return {
        "count": int(len(y)),
        "outcome_rate": round(float(y.mean()), 6),
        "mean_predicted_probability": round(float(p.mean()), 6),
        "brier_score": round(float(brier_score_loss(y, p)), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6) if has_both else None,
        "average_precision": round(float(average_precision_score(y, p)), 6) if has_both else None,
        "fraction_predictions_above_0_99": round(float(np.mean(p > 0.99)), 6),
    }


def paired_bootstrap_mlp(y_true, with_year, without_year, *, replicates=300, seed=2026):
    """Resample paired ROWS to summarize score differences; never refit models.

    Difference = score(without year) - score(with year).
    For Brier a negative difference favors without-year; for AUC, positive
    favors without-year. The bootstrap is not an independent model test.
    """
    from sklearn.metrics import roc_auc_score

    if type(replicates) is not int or replicates < 20 or replicates > 2000:
        raise ValueError("replicates must be an integer in [20, 2000]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    y, a = _aligned_binary(y_true, with_year)
    _, b = _aligned_binary(y_true, without_year)
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Both outcome classes are required for paired AUC uncertainty")
    rng = np.random.default_rng(seed)
    sq_diff = (b - y) ** 2 - (a - y) ** 2
    brier_samples = np.empty(replicates, dtype=np.float64)
    auc_samples = []
    for i in range(replicates):
        idx = rng.integers(0, len(y), size=len(y))
        brier_samples[i] = float(sq_diff[idx].mean())
        if np.unique(y[idx]).size == 2:
            auc_samples.append(
                float(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
            )
    def quantile_interval(samples):
        if len(samples) < 10:
            return None
        return [round(float(v), 6) for v in np.quantile(samples, [0.025, 0.975])]
    original_auc_delta = roc_auc_score(y, b) - roc_auc_score(y, a)
    return {
        "comparison": "without_arrival_year minus with_arrival_year (NumPy MLP)",
        "brier_difference": round(float(sq_diff.mean()), 6),
        "brier_95_percentile_interval": quantile_interval(brier_samples),
        "roc_auc_difference": round(float(original_auc_delta), 6),
        "roc_auc_95_percentile_interval": quantile_interval(auc_samples),
        "requested_replicates": replicates,
        "successful_auc_replicates": len(auc_samples),
        "seed": seed,
        "caution": (
            "Exploratory paired row-resampling intervals, conditional on fixed "
            "fitted models and this already-examined cohort. They do not cover "
            "training uncertainty, temporal/group dependence, model selection, "
            "distribution shift, or independent replication; not significance claims."
        ),
    }


def run_reliability(csv_path: str | Path, *, mlp_epochs: int = 30,
                    bootstrap_replicates: int = 300) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if type(mlp_epochs) is not int or mlp_epochs < 1:
        raise ValueError("mlp_epochs must be a positive integer")
    if type(bootstrap_replicates) is not int or not 20 <= bootstrap_replicates <= 2000:
        raise ValueError("bootstrap_replicates must be an integer in [20, 2000]")

    rows, labels, counts = load_arrival_benchmark(csv_path)
    train = make_frame(rows["train"])
    valid = make_frame(rows["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_valid = np.asarray(labels["validation"], dtype=np.int64)
    if len(y_train) < 2 or len(y_valid) < 2 or np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise ValueError("Requires both outcome classes in train and validation")
    print(f"Reliability: fitting fixed models on {len(train):,} train rows; "
          f"describing {len(valid):,} retrospective validation rows", flush=True)

    predictions = {
        "training_prior": np.full(len(y_valid), float(y_train.mean()), dtype=np.float64)
    }
    for include_year in (True, False):
        numeric, categorical, columns = selected_features(include_year)
        suffix = "with_year" if include_year else "without_year"
        preprocess = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), list(numeric)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ], sparse_threshold=0.0,
        )
        xt = np.asarray(preprocess.fit_transform(train[columns]), dtype=np.float64)
        xv = np.asarray(preprocess.transform(valid[columns]), dtype=np.float64)
        logistic = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42
        ).fit(xt, y_train)
        predictions[f"logistic_{suffix}"] = logistic.predict_proba(xv)[:, 1]
        mlp = NumpyMLPBinary(
            hidden_units=32, epochs=mlp_epochs, batch_size=256,
            learning_rate=0.001, weight_decay=0.0001, seed=42,
        )
        print(f"  fitting NumPy MLP {suffix}, epochs={mlp_epochs}...", flush=True)
        mlp.fit(
            xt, y_train, on_epoch=lambda epoch, total, loss: print(
                f"    {suffix} epoch {epoch}/{total}, objective={loss:.6f}", flush=True
            ) if epoch == 1 or epoch == total or epoch % 5 == 0 else None,
        )
        predictions[f"mlp_{suffix}"] = mlp.predict_proba_positive(xv)

    summary = {}
    for name, p in predictions.items():
        summary[name] = {
            "full_validation": subgroup_metrics(y_valid, p),
            "reliability": reliability_bins(y_valid, p, n_bins=10),
            "hotel": {},
            "arrival_month": {},
        }
        for field, key in (("hotel", "hotel"), ("arrival_date_month", "arrival_month")):
            for value in sorted(valid[field].unique()):
                mask = (valid[field].to_numpy() == value)
                if mask.any():
                    summary[name][key][str(value)] = subgroup_metrics(y_valid[mask], p[mask])

    uncertainty = paired_bootstrap_mlp(
        y_valid, predictions["mlp_with_year"], predictions["mlp_without_year"],
        replicates=bootstrap_replicates, seed=2026,
    )
    csv_path = Path(csv_path)
    with csv_path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "experiment": "bookingforge_fixed_arrival_reliability_diagnostic_v1",
        "dataset_sha256": sha,
        "source_protocol": "arrival_benchmark.py; exactly the previously verified fit/validation periods",
        "training_records": len(y_train),
        "retrospective_validation_records": len(y_valid),
        "mlp_epochs": mlp_epochs,
        "fixed_models": summary,
        "paired_mlp_uncertainty": uncertainty,
        "cohort_counts": counts,
        "raw_booking_rows_exported": False,
        "outcomes_from_2017_arrivals_accessed_by_this_runner": False,
        "models_or_thresholds_selected_using_validation_labels": False,
        "calibrators_fit": False,
        "warning": (
            "Descriptive post-hoc reliability diagnostics on an already-examined "
            "retrospective arrival cohort. Not independent evidence for deployment "
            "or original booking-time prediction; legacy aggregate audits used "
            "2017 outcomes, so no pristine final holdout is claimed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Descriptive calibration and subgroup audit")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/arrival_reliability.json")
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    args = parser.parse_args()
    report = run_reliability(
        args.csv, mlp_epochs=args.mlp_epochs,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nARRIVAL RETROSPECTIVE RELIABILITY REPORT COMPLETE")
    for name, stats in report["fixed_models"].items():
        m = stats["full_validation"]
        print(f"  {name}: n={m['count']:,}, AUC={m['roc_auc']}, "
              f"Brier={m['brier_score']}, "
              f"calibration ECE={stats['reliability']['expected_calibration_error_equal_width']}")
        for hotel, values in stats["hotel"].items():
            print(f"    {hotel}: n={values['count']:,}, Brier={values['brier_score']}, AUC={values['roc_auc']}")
        for month, values in stats["arrival_month"].items():
            print(f"    {month}: n={values['count']:,}, Brier={values['brier_score']}, AUC={values['roc_auc']}")
    print("Paired descriptive MLP differences:", report["paired_mlp_uncertainty"])
    print(f"Aggregate JSON: {output}")
    print("No model tuning or calibration was performed; this is NOT booking-time validation.")


if __name__ == "__main__":
    main()
