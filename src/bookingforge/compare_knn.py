"""Full-cohort, reproducible comparison of prior, logistic, NumPy kNN and NumPy MLP.

No training or validation subsampling. Training only uses outcomes known by
2016-08-31; retrospective validation scores ALL eventual final outcomes in
the distinct 2016-09 to 2016-12 booking-creation cohort. Test labels sealed.

These are exploratory metrics on an arrival-bounded dataset with incomplete
historical booking snapshots and biased historically mature training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from .knn import NumpyKNNBinary
from .mlp_numpy import NumpyMLPBinary
from .train_baseline import (
    CATEGORICAL, NUMERIC, FEATURES, load_matured_cohorts, make_frame, metrics_for,
)


def run_full_comparison(csv_path: str | Path, k: int = 15, batch_size: int = 64,
                        mlp_hidden: int = 32, mlp_epochs: int = 30,
                        mlp_batch_size: int = 256, mlp_learning_rate: float = 0.001) -> dict:
    import numpy as np
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    start = time.perf_counter()
    samples, labels, counts = load_matured_cohorts(
        csv_path, validation_mode="all_eventual"
    )
    x_train_df = make_frame(samples["train"])
    x_valid_df = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_valid = np.asarray(labels["validation"], dtype=np.int64)
    if len(y_train) < k or len(y_valid) == 0:
        raise ValueError("Not enough eligible training/validation rows")
    if len(set(y_train.tolist())) != 2 or len(set(y_valid.tolist())) != 2:
        raise ValueError("Both classes must exist in training and validation")

    print(f"FULL train cohort: {len(y_train):,} | FULL validation cohort: {len(y_valid):,}", flush=True)
    print("Train-only imputation, scaling and encoding...", flush=True)

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
    x_train = np.asarray(preprocess.fit_transform(x_train_df), dtype=np.float64)
    x_valid = np.asarray(preprocess.transform(x_valid_df), dtype=np.float64)

    print("Fitting logistic regression on ALL eligible training records...", flush=True)
    logistic = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    logistic.fit(x_train, y_train)
    logistic_prob = logistic.predict_proba(x_valid)[:, list(logistic.classes_).index(1)]
    prior_prob = np.full(len(y_valid), float(y_train.mean()), dtype=np.float64)

    print("Evaluating original NumPy kNN on ALL validation records (batched, exact)...", flush=True)
    knn_start = time.perf_counter()
    knn = NumpyKNNBinary(k=k, batch_size=batch_size).fit(x_train, y_train)

    def report_progress(done: int, total: int) -> None:
        print(f"  kNN: {done:,}/{total:,} validation bookings ({done/total:.1%})", flush=True)

    knn_prob = knn.predict_proba_positive(x_valid, on_progress=report_progress)
    knn_seconds = time.perf_counter() - knn_start

    print("Training from-scratch NumPy MLP using only training features and labels...", flush=True)
    mlp = NumpyMLPBinary(
        hidden_units=mlp_hidden, epochs=mlp_epochs, batch_size=mlp_batch_size,
        learning_rate=mlp_learning_rate, weight_decay=0.0001, seed=42,
    )
    mlp_start = time.perf_counter()

    def report_epoch(epoch: int, total: int, loss: float) -> None:
        print(f"  MLP epoch {epoch}/{total}: training objective={loss:.6f}", flush=True)

    mlp.fit(x_train, y_train, on_epoch=report_epoch)
    mlp_prob = mlp.predict_proba_positive(x_valid)
    mlp_seconds = time.perf_counter() - mlp_start

    path = Path(csv_path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()

    return {
        "experiment": "bookingforge_original_knn_mlp_full_cohort_v2",
        "dataset_sha256": sha,
        "evaluation_status": "EXPLORATORY RETROSPECTIVE VALIDATION; NOT DEPLOYMENT-VALIDATED",
        "data_sampling": "NONE; all eligible train rows and all validation creation-cohort rows",
        "training_label_cutoff": "2016-08-31",
        "validation_creation_dates": "2016-09-01 to 2016-12-31",
        "validation_scoring_labels": "eventual final status; not used to fit models or preprocessing",
        "final_test_labels_accessed": False,
        "training_rows": len(y_train),
        "validation_rows": len(y_valid),
        "training_cancel_fraction": round(float(y_train.mean()), 6),
        "validation_cancel_fraction": round(float(y_valid.mean()), 6),
        "feature_columns": FEATURES,
        "encoded_dimension": int(x_train.shape[1]),
        "custom_knn": {
            "algorithm": "exact Euclidean distance, NumPy batched dot-product, uniform k-neighbor voting",
            "k": k,
            "batch_size": batch_size,
            "training_rows_used": len(y_train),
            "validation_rows_scored": len(y_valid),
            "execution_seconds": round(knn_seconds, 3),
        },
        "custom_numpy_mlp": {
            "implementation": "one hidden ReLU layer, logistic output, binary cross entropy",
            "optimizer": "mini-batch Adam from scratch with L2 penalty on weights",
            "hidden_units": mlp_hidden,
            "epochs": mlp_epochs,
            "batch_size": mlp_batch_size,
            "learning_rate": mlp_learning_rate,
            "weight_decay": mlp.weight_decay,
            "seed": mlp.seed,
            "trained_rows_used": len(y_train),
            "validation_rows_scored": len(y_valid),
            "training_objective_by_epoch": mlp.training_loss_history_,
            "fit_and_score_seconds": round(mlp_seconds, 3),
            "validation_used_for_training_or_early_stopping": False,
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation_metrics": {
            "prior_probability": metrics_for(y_valid, prior_prob),
            "logistic_regression": metrics_for(y_valid, logistic_prob),
            "custom_numpy_knn": metrics_for(y_valid, knn_prob),
            "custom_numpy_mlp": metrics_for(y_valid, mlp_prob),
        },
        "counts": counts,
        "total_execution_seconds": round(time.perf_counter() - start, 3),
        "limitations": [
            "Historical feature snapshots are not available; source values may have changed after creation.",
            "Dataset is arrival-bounded and does not capture all created reservations.",
            "Training outcomes are selected to be known at the historical cutoff.",
            "Validation includes all eventual outcomes, which are used ONLY retrospectively for scoring.",
            "k=15 and the NumPy MLP configuration were fixed before examining new results; no validation-based tuning or early stopping.",
            "Exact kNN has O(train_rows * validation_rows * features) evaluation cost.",
            "No causal, business revenue or production performance claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact full-data NumPy kNN vs logistic comparison")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/full_four_model_comparison.json")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    report = run_full_comparison(
        args.csv, k=args.k, batch_size=args.batch_size,
        mlp_hidden=args.mlp_hidden, mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size, mlp_learning_rate=args.mlp_learning_rate,
    )
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nFOUR-MODEL FULL COHORT COMPARISON COMPLETE")
    for name, metrics in report["validation_metrics"].items():
        print(f"  {name}: ROC AUC={metrics['roc_auc']:.4f}; AP={metrics['average_precision']:.4f}; Brier={metrics['brier_score']:.4f}")
    print(f"Full comparison report: {dest}")
    print("WARNING: Historical training-label selection and feature timing remain limitations.")
    print("Final test outcomes were not accessed.")


if __name__ == "__main__":
    main()
