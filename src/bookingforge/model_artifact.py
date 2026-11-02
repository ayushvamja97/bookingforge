"""Portable, pickle-free NumPy MLP research artifacts.

Train only on the fixed retrospective ARRIVAL-cohort historical training
population. Save model weights in .npz (allow_pickle=False when loading)
and train-fitted preprocessing statistics/categories in .json. Re-load
and verify exact (1e-12 absolute tolerance) validation-feature predictions
without using validation labels to fit or select the model.

These artifacts are *research classifiers only*, not operational
booking-creation cancellation-risk models. They are not calibrated for
deployment; historical immutable booking-time snapshots are unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .arrival_benchmark import (
    FIT_INFORMATION_CUTOFF,
    TRAIN_FIRST_ARRIVAL,
    TRAIN_LAST_ARRIVAL,
    load_arrival_benchmark,
    selected_features,
)
from .cohort import validate_booking_features
from .mlp_numpy import NumpyMLPBinary
from .train_baseline import make_frame

SCHEMA_VERSION = 1
WEIGHT_KEYS = ("W1", "b1", "W2", "b2")
FORBIDDEN_INPUT_FIELDS = {"is_canceled", "reservation_status_date", "reservation_status"}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class PortableMLP:
    """NumPy-only inference using serialized fixed feature encoding and weights."""

    def __init__(self, metadata: dict, weights: dict[str, np.ndarray]):
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported artifact schema")
        if metadata.get("research_scope") != "retrospective_arrival_cohort_only":
            raise ValueError("Artifact scope is missing or invalid")
        numeric, categorical, features = selected_features(
            metadata.get("include_arrival_year")
        )
        if list(numeric) != metadata.get("numeric_features"):
            raise ValueError("Artifact numeric feature schema mismatch")
        if list(categorical) != metadata.get("categorical_features"):
            raise ValueError("Artifact categorical feature schema mismatch")
        if features != metadata.get("features"):
            raise ValueError("Artifact feature schema mismatch")
        validate_booking_features(features)

        self.numeric_features = list(numeric)
        self.categorical_features = list(categorical)
        self.features = list(features)
        self.categories = metadata["categories"]
        self.medians = np.asarray(metadata["numeric_medians"], dtype=np.float64)
        self.means = np.asarray(metadata["numeric_means"], dtype=np.float64)
        self.scales = np.asarray(metadata["numeric_scales"], dtype=np.float64)
        if (
            len(self.categories) != len(categorical)
            or any(not isinstance(cat, list) or not cat for cat in self.categories)
            or any(len(set(cat)) != len(cat) for cat in self.categories)
        ):
            raise ValueError("Malformed categorical encoding")
        if any(not isinstance(v, str) for cats in self.categories for v in cats):
            raise ValueError("Categorical values must be text")
        expected_numeric = len(numeric)
        for array in (self.medians, self.means, self.scales):
            if array.shape != (expected_numeric,) or not np.isfinite(array).all():
                raise ValueError("Invalid numeric preprocessing array")
        if np.any(self.scales <= 0):
            raise ValueError("Numeric scale must be positive")

        expected_dim = expected_numeric + sum(len(cat) for cat in self.categories)
        loaded = {name: np.asarray(weights[name], dtype=np.float64) for name in WEIGHT_KEYS}
        hidden_units = loaded["W1"].shape[1] if loaded["W1"].ndim == 2 else 0
        if not hidden_units or (
            loaded["W1"].shape != (expected_dim, hidden_units)
            or loaded["b1"].shape != (hidden_units,)
            or loaded["W2"].shape != (hidden_units, 1)
            or loaded["b2"].shape != (1,)
        ):
            raise ValueError("Weight array shapes are incompatible with preprocessing")
        if not all(np.isfinite(value).all() for value in loaded.values()):
            raise ValueError("Nonfinite weights are not permitted")
        if metadata.get("transformed_dimension") != expected_dim:
            raise ValueError("Encoded feature dimension mismatch")
        self.metadata = metadata
        self.mlp = NumpyMLPBinary(hidden_units=hidden_units, epochs=1)
        self.mlp.params_ = loaded
        self.mlp.n_features_in_ = expected_dim

    def transform(self, rows: list[dict]) -> np.ndarray:
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected a nonempty list of feature dictionaries")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each input must be a feature dictionary")
            if FORBIDDEN_INPUT_FIELDS.intersection(row):
                raise ValueError("Outcome fields are forbidden during inference")
            missing = set(self.features) - set(row)
            if missing:
                raise ValueError(f"Missing required feature columns: {sorted(missing)}")
        frame = make_frame(rows)
        xnum = frame[self.numeric_features].to_numpy(dtype=np.float64)
        if np.isinf(xnum).any():
            raise ValueError("Infinite numeric values are not permitted")
        xnum = np.where(np.isnan(xnum), self.medians, xnum)
        xnum = (xnum - self.means) / self.scales
        parts = [xnum]
        for name, categories in zip(self.categorical_features, self.categories):
            values = frame[name].astype(str).to_numpy()
            onehot = (values[:, None] == np.asarray(categories, dtype=str)[None, :]).astype(np.float64)
            parts.append(onehot)
        encoded = np.column_stack(parts)
        if encoded.shape[1] != self.mlp.n_features_in_ or not np.isfinite(encoded).all():
            raise ValueError("Invalid transformed inputs")
        return encoded

    def predict_proba_positive(self, rows: list[dict]) -> np.ndarray:
        return self.mlp.predict_proba_positive(self.transform(rows))


def save_portable(directory: Path, stem: str, metadata: dict, mlp: NumpyMLPBinary) -> dict:
    """Save JSON + fixed-shape NumPy weights; no sklearn/pickle object stored."""
    if mlp.params_ is None:
        raise ValueError("Cannot export unfitted neural model")
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / (stem + ".npz")
    metadata_path = directory / (stem + ".json")
    with weights_path.open("wb") as handle:
        np.savez_compressed(handle, **{name: mlp.params_[name] for name in WEIGHT_KEYS})
    file_metadata = dict(metadata)
    file_metadata["weights_sha256"] = _sha256(weights_path)
    metadata_path.write_text(
        json.dumps(file_metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "model_weights": str(weights_path),
        "model_metadata": str(metadata_path),
        "weights_sha256": file_metadata["weights_sha256"],
    }


def load_portable(directory: str | Path, stem: str) -> PortableMLP:
    """Verify paired file integrity; refuse object arrays or pickle loading."""
    root = Path(directory)
    metadata_path, weights_path = root / (stem + ".json"), root / (stem + ".npz")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("weights_sha256") != _sha256(weights_path):
        raise ValueError("Artifact weights failed SHA256 integrity check")
    with np.load(weights_path, allow_pickle=False) as archived:
        if set(archived.files) != set(WEIGHT_KEYS):
            raise ValueError("Unexpected weight-array keys")
        arrays = {name: archived[name] for name in WEIGHT_KEYS}
    return PortableMLP(meta, arrays)


def export_arrival_models(csv_path: str | Path, output_dir: str | Path,
                          *, variant: str = "both", epochs: int = 30) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if variant not in {"both", "with_year", "without_year"}:
        raise ValueError("variant must be both, with_year or without_year")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    samples, labels, counts = load_arrival_benchmark(csv_path)
    y_train = np.asarray(labels["train"], dtype=np.int64)
    if not samples["train"] or not samples["validation"] or np.unique(y_train).size != 2:
        raise ValueError("Expected nonempty train/validation and two train classes")
    train = make_frame(samples["train"])
    validation = make_frame(samples["validation"])
    output_dir = Path(output_dir)
    chosen = (("with_year", True), ("without_year", False))
    if variant != "both":
        chosen = tuple((name, flag) for name, flag in chosen if name == variant)
    results = {}
    dataset_sha = _sha256(Path(csv_path))

    for label, include_year in chosen:
        numeric, categorical, features = selected_features(include_year)
        processor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), list(numeric)),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
            ],
            sparse_threshold=0.0,
        )
        x_train = np.asarray(processor.fit_transform(train[features]), dtype=np.float64)
        x_valid = np.asarray(processor.transform(validation[features]), dtype=np.float64)
        numeric_pipeline = processor.named_transformers_["numeric"]
        imputer = numeric_pipeline.named_steps["impute"]
        scaler = numeric_pipeline.named_steps["scale"]
        encoder = processor.named_transformers_["categorical"]
        print(f"Training fixed {label} MLP on {len(y_train):,} historical records...", flush=True)
        mlp = NumpyMLPBinary(
            hidden_units=32, epochs=epochs, batch_size=256,
            learning_rate=0.001, weight_decay=0.0001, seed=42,
        )
        mlp.fit(
            x_train, y_train,
            on_epoch=lambda epoch, total, loss: print(
                f"  {label}: epoch {epoch}/{total}, loss={loss:.6f}", flush=True
            ) if epoch == 1 or epoch == total or epoch % 5 == 0 else None,
        )
        categories = [[str(value) for value in col] for col in encoder.categories_]
        meta = {
            "schema_version": SCHEMA_VERSION,
            "research_scope": "retrospective_arrival_cohort_only",
            "include_arrival_year": include_year,
            "features": features,
            "numeric_features": list(numeric),
            "categorical_features": list(categorical),
            "numeric_medians": imputer.statistics_.astype(float).tolist(),
            "numeric_means": scaler.mean_.astype(float).tolist(),
            "numeric_scales": scaler.scale_.astype(float).tolist(),
            "categories": categories,
            "transformed_dimension": int(x_train.shape[1]),
            "dataset_sha256": dataset_sha,
            "train_arrival_first": TRAIN_FIRST_ARRIVAL.isoformat(),
            "train_arrival_last": TRAIN_LAST_ARRIVAL.isoformat(),
            "training_outcomes_known_by": FIT_INFORMATION_CUTOFF.isoformat(),
            "training_records": len(y_train),
            "training_outcome_rate": round(float(y_train.mean()), 6),
            "epochs": epochs,
            "hidden_units": 32,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "seed": 42,
            "training_objective_by_epoch": mlp.training_loss_history_,
            "warning": (
                "Exploratory retrospective ARRIVAL-date research only. "
                "Historical feature snapshots at reservation creation are unavailable. "
                "Uncalibrated for operational use, not a deployable booking-time predictor."
            ),
        }
        paths = save_portable(output_dir, "arrival_mlp_" + label, meta, mlp)
        reloaded = load_portable(output_dir, "arrival_mlp_" + label)
        expected = mlp.predict_proba_positive(x_valid)
        actual = reloaded.predict_proba_positive(samples["validation"])
        # Numerical equality check is against the exact sklearn transform.
        # Do not score or select either model using validation target labels.
        max_error = float(np.max(np.abs(expected - actual)))
        if max_error > 1e-12 or not np.isfinite(max_error):
            raise AssertionError(f"Reloaded {label} artifact changed probabilities by {max_error}")
        results[label] = {
            **paths,
            "validation_features_checked": len(actual),
            "max_absolute_prediction_difference": max_error,
            "trained_records": len(y_train),
        }
        print(f"  {label}: reload verification max difference={max_error:.3g}", flush=True)
    return {
        "experiment": "bookingforge_portable_numpy_arrival_mlp_v1",
        "research_only": True,
        "dataset_sha256": dataset_sha,
        "training_rows": len(y_train),
        "validation_features_used_to_check_reproducibility": len(samples["validation"]),
        "validation_outcomes_used_for_training_or_artifact_verification": False,
        "no_2017_arrival_outcomes_read_by_this_runner": True,
        "legacy_nominal_test_aggregate_audits_had_read_targets": True,
        "models": results,
        "cohort_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit and export original NumPy MLP research artifacts")
    parser.add_argument("--csv", default="data/raw/hotels.csv")
    parser.add_argument("--output-dir", default="models/arrival_v1")
    parser.add_argument("--report", default="reports/arrival_artifact_verification.json")
    parser.add_argument("--variant", choices=["both", "with_year", "without_year"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    report = export_arrival_models(
        args.csv, args.output_dir, variant=args.variant, epochs=args.epochs
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print("\nPORTABLE NUMPY MLP ARTIFACT VERIFICATION COMPLETE")
    print("Trained records:", report["training_rows"])
    for label, model in report["models"].items():
        print(f"  {label}: weights={model['model_weights']}; metadata={model['model_metadata']}; "
              f"maximum reload prediction difference={model['max_absolute_prediction_difference']:.3g}")
    print("Verification report:", report_path)
    print("Research-only historical arrival classifier; not validated at booking creation.")


if __name__ == "__main__":
    main()
