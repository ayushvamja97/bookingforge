"""Tests for portable original NumPy MLP artifact training/inference parity."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.arrival_benchmark import load_arrival_benchmark
from bookingforge.model_artifact import (
    PortableMLP,
    export_arrival_models,
    load_portable,
)
from bookingforge.train_baseline import FEATURES


class PortableModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.csv_path = self.directory / "synthetic.csv"
        self.outdir = self.directory / "models"
        self.records = [
            self.row("2016", "July", "1", "2016-07-02", "0"),
            self.row("2016", "July", "2", "2016-07-03", "1"),
            self.row("2016", "July", "3", "2016-07-04", "0", hotel="Resort Hotel"),
            self.row("2016", "July", "4", "2016-07-05", "1", hotel="Resort Hotel"),
            self.row("2016", "October", "11", "2016-10-12", "0"),
            self.row("2016", "October", "12", "2016-10-13", "1"),
            self.row("2016", "November", "11", "2017-01-03", "0", hotel="New Hotel"),
            self.row("2016", "November", "12", "2017-01-03", "1", hotel="Resort Hotel"),
            # These must be excluded before reading final status/label.
            self.row("2017", "January", "3", "invalid", "invalid"),
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
            )
            writer.writeheader()
            writer.writerows(self.records)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def row(year, month, day, status, target, hotel="City Hotel"):
        record = {feature: "0" for feature in FEATURES}
        record.update({
            "hotel": hotel,
            "lead_time": "0",
            "arrival_date_year": year,
            "arrival_date_month": month,
            "arrival_date_day_of_month": day,
            "reserved_room_type": "A",
            "adults": "2",
            "reservation_status_date": status,
            "is_canceled": target,
        })
        return record

    def _export(self, variant="both"):
        return export_arrival_models(
            self.csv_path, self.outdir,
            variant=variant, epochs=2,
        )

    def test_portable_reloaded_predictions_match_sklearn_transformed_mlp(self):
        report = self._export()
        self.assertEqual(set(report["models"]), {"with_year", "without_year"})
        self.assertEqual(report["training_rows"], 4)
        self.assertEqual(report["validation_features_used_to_check_reproducibility"], 4)
        self.assertFalse(report["validation_outcomes_used_for_training_or_artifact_verification"])
        self.assertTrue(report["no_2017_arrival_outcomes_read_by_this_runner"])
        rows, _, _ = load_arrival_benchmark(self.csv_path)
        for variant in ("with_year", "without_year"):
            info = report["models"][variant]
            self.assertLessEqual(info["max_absolute_prediction_difference"], 1e-12)
            loaded = load_portable(self.outdir, "arrival_mlp_" + variant)
            probabilities = loaded.predict_proba_positive(rows["validation"])
            self.assertEqual(probabilities.shape, (4,))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
            self.assertEqual(loaded.metadata["training_records"], 4)
            self.assertEqual(len(loaded.metadata["training_objective_by_epoch"]), 2)

    def test_tampered_weights_are_rejected(self):
        self._export("with_year")
        path = self.outdir / "arrival_mlp_with_year.npz"
        with path.open("ab") as handle:
            handle.write(b"unexpected bytes")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_portable(self.outdir, "arrival_mlp_with_year")

    def test_malformed_json_schema_is_rejected(self):
        self._export("without_year")
        path = self.outdir / "arrival_mlp_without_year.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["numeric_features"].append("is_canceled")
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            load_portable(self.outdir, "arrival_mlp_without_year")

    def test_inference_rejects_labels_and_missing_features(self):
        self._export("without_year")
        loaded = load_portable(self.outdir, "arrival_mlp_without_year")
        rows, _, _ = load_arrival_benchmark(self.csv_path)
        good = rows["validation"][0].copy()
        # The no-year variant genuinely does not require arrival_date_year.
        good.pop("arrival_date_year")
        self.assertEqual(loaded.predict_proba_positive([good]).shape, (1,))
        missing = dict(good)
        missing.pop("lead_time")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            loaded.predict_proba_positive([missing])
        with self.assertRaisesRegex(ValueError, "Outcome fields"):
            loaded.predict_proba_positive([dict(good, is_canceled="0")])
        with self.assertRaises(ValueError):
            loaded.predict_proba_positive([])

    def test_only_requested_variant_exported_and_config_guard(self):
        result = self._export("with_year")
        self.assertEqual(set(result["models"]), {"with_year"})
        self.assertFalse((self.outdir / "arrival_mlp_without_year.npz").exists())
        with self.assertRaises(ValueError):
            export_arrival_models(self.csv_path, self.outdir, variant="invalid", epochs=2)
        with self.assertRaises(ValueError):
            export_arrival_models(self.csv_path, self.outdir, variant="both", epochs=0)


if __name__ == "__main__":
    unittest.main()
