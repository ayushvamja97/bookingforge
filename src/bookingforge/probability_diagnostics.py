"""Inspect probability behavior of the already-fixed NumPy neural baseline.

This is diagnostic, not a new model-selection experiment. Refit using the
predeclared hyperparameters and historic training labels only; inspect
training fit and FULL retrospective validation separately. No final test
labels or status dates are accessed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .train_baseline import CATEGORICAL, NUMERIC, load_matured_cohorts, make_frame
from .mlp_numpy import NumpyMLPBinary


def summarize_probabilities(y, prob):
    """Aggregate model confidence and scoring, never emit individual bookings."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob) or not len(y):
        raise ValueError("Labels and probabilities must be aligned, nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Invalid labels or predicted probabilities")
    percentiles = np.quantile(prob, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist()
    return {
        "n": int(len(y)),
        "target_prevalence": round(float(y.mean()), 6),
        "predicted_mean": round(float(prob.mean()), 6),
        "predicted_std": round(float(prob.std()), 6),
        "predicted_min_p01_p10_p50_p90_p99_max": [round(float(v), 6) for v in percentiles],
        "p_below_0_01_fraction": round(float((prob < 0.01).mean()), 6),
        "p_above_0_99_fraction": round(float((prob > 0.99).mean()), 6),
        "p_above_0_5_fraction": round(float((prob >= 0.5).mean()), 6),
        "brier": round(float(brier_score_loss(y, prob)), 6),
        "log_loss": round(float(log_loss(y, prob, labels=[0, 1])), 6),
        "roc_auc": round(float(roc_auc_score(y, prob)), 6) if len(np.unique(y)) == 2 else None,
        "brier_by_truth": {
            "ultimately_checked_out": round(float(np.mean(prob[y == 0] ** 2)), 6) if (y == 0).any() else None,
            "ultimately_canceled_or_no_show": round(float(np.mean((1 - prob[y == 1]) ** 2)), 6) if (y == 1).any() else None,
        },
    }


def run_diagnostic(csv_path: str | Path, mlp_epochs: int = 30):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    samples, labels, counts = load_matured_cohorts(csv_path, validation_mode="all_eventual")
    train = make_frame(samples["train"])
    valid = make_frame(samples["validation"])
    y_train = np.asarray(labels["train"], dtype=np.int64)
    y_valid = np.asarray(labels["validation"], dtype=np.int64)

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
    x_valid = np.asarray(preprocess.transform(valid), dtype=np.float64)

    print("Fitting logistic regression on historical training data...", flush=True)
    logistic = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    logistic.fit(x_train, y_train)
    train_logistic = logistic.predict_proba(x_train)[:, 1]
    valid_logistic = logistic.predict_proba(x_valid)[:, 1]

    print("Fitting NumPy MLP with original fixed configuration...", flush=True)
    mlp = NumpyMLPBinary(
        hidden_units=32, epochs=mlp_epochs,
        batch_size=256, learning_rate=0.001,
        weight_decay=0.0001, seed=42,
    )
    mlp.fit(
        x_train, y_train,
        on_epoch=lambda epoch, total, loss: print(
            f"  epoch {epoch}/{total}: training loss {loss:.6f}", flush=True
        ) if epoch == 1 or epoch == total or epoch % 5 == 0 else None,
    )
    train_mlp = mlp.predict_proba_positive(x_train)
    valid_mlp = mlp.predict_proba_positive(x_valid)

    model_summaries = {
        "logistic_regression": {
            "train": summarize_probabilities(y_train, train_logistic),
            "validation": summarize_probabilities(y_valid, valid_logistic),
        },
        "numpy_mlp": {
            "train": summarize_probabilities(y_train, train_mlp),
            "validation": summarize_probabilities(y_valid, valid_mlp),
        },
    }

    per_hotel = {}
    for name, p in (("logistic_regression", valid_logistic), ("numpy_mlp", valid_mlp)):
        per_hotel[name] = {}
        for hotel in sorted(valid["hotel"].unique()):
            mask = (valid["hotel"].to_numpy() == hotel)
            if mask.sum():
                per_hotel[name][hotel] = summarize_probabilities(y_valid[mask], p[mask])

    path = Path(csv_path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()

    return {
        "experiment": "bookingforge_probability_diagnostics_v1",
        "dataset_sha256": sha,
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "cohort_counts": counts,
        "model_summaries": model_summaries,
        "validation_by_hotel": per_hotel,
        "mlp_epochs": mlp_epochs,
        "validation_used_in_fitting_or_early_stopping": False,
        "test_outcomes_accessed": False,
        "status": "DIAGNOSTIC ONLY; no hyperparameter selection or calibration",
        "limits": [
            "Historical training uses only labels known at 2016-08-31 and remains selected.",
            "Validation uses eventual outcomes, retrospectively, for scoring only.",
            "Creation time is inferred from arrival minus lead time; immutable snapshots absent.",
            "The source dataset is selected by arrival window, not booking creation.",
            "Hotel subgroup analyses are descriptive, not independent hypothesis tests.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose neural-network probability collapse")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output", default="reports/probability_diagnostics.json")
    args = parser.parse_args()
    report = run_diagnostic(args.csv)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nBOOKINGFORGE PROBABILITY DIAGNOSTICS COMPLETE")
    for model, splits in report["model_summaries"].items():
        print(model)
        for split, stats in splits.items():
            print(
                f"  {split}: prevalence={stats['target_prevalence']:.3f}; "
                f"predicted_mean={stats['predicted_mean']:.3f}; "
                f"Brier={stats['brier']:.4f}; ROC AUC={stats['roc_auc']}; "
                f"p<0.01={stats['p_below_0_01_fraction']:.1%}; "
                f"p>0.99={stats['p_above_0_99_fraction']:.1%}"
            )
        for hotel, stats in report["validation_by_hotel"][model].items():
            print(
                f"  {hotel}: n={stats['n']:,}; "
                f"prevalence={stats['target_prevalence']:.3f}; "
                f"mean_p={stats['predicted_mean']:.3f}; "
                f"Brier={stats['brier']:.4f}"
            )
    print("Saved aggregate diagnostic:", dest)
    print("Final test outcome labels remain sealed.")


if __name__ == "__main__":
    main()
