"""Chronological inner split and full-feature versus no-year ablation tests."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.inner_year_ablation import (
    experiment_config, load_inner_chronological, model_comparison,
)
from bookingforge.train_baseline import FEATURES


class InnerYearAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "synthetic.csv"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, year, month, day, status, label, *, lead="0"):
        entry = {name: "0" for name in FEATURES}
        entry.update({
            "hotel": "City Hotel",
            "lead_time": lead,
            "arrival_date_year": str(year),
            "arrival_date_month": month,
            "arrival_date_day_of_month": str(day),
            "reserved_room_type": "A",
            "adults": "2",
            "reservation_status_date": status,
            "is_canceled": label,
        })
        return entry

    def write(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_cutoff_maturity_and_outer_sealing(self):
        self.write([
            # Inner train: finalized by March 31.
            self.row(2016, "March", 20, "2016-03-22", "1"),
            self.row(2016, "March", 21, "2016-04-05", "0"),
            # Inner validation: eventual outcome known by original train cutoff.
            self.row(2016, "April", 15, "2016-08-25", "1"),
            self.row(2016, "April", 16, "2016-09-01", "0"),
            self.row(2016, "May", 2, "2016-06-03", "0"),
            # June booking excluded even though in the original training cohort.
            self.row(2016, "June", 2, "invalid-status", "invalid-label"),
            # Outer validation must be excluded BEFORE touching target.
            self.row(2016, "September", 2, "invalid-status", "invalid-label"),
            # Final test must also be excluded BEFORE touching target.
            self.row(2017, "January", 3, "invalid-status", "invalid-label"),
        ])
        samples, labels, counts = load_inner_chronological(self.path)
        self.assertEqual(labels, {"train": [1], "validation": [1, 0]})
        self.assertEqual(len(samples["validation"]), 2)
        self.assertEqual(counts["train_outcome_not_known_by_cutoff"], 1)
        self.assertEqual(counts["validation_outcome_not_known_by_cutoff"], 1)
        self.assertEqual(counts["not_inner_cohort"], 3)

    def test_ablation_drop_only_arrival_year(self):
        numeric, categorical, chosen = experiment_config(False)
        self.assertNotIn("arrival_date_year", numeric)
        self.assertNotIn("arrival_date_year", chosen)
        self.assertIn("lead_time", chosen)
        self.assertIn("hotel", categorical)
        full = experiment_config(True)[2]
        self.assertEqual(len(full) - len(chosen), 1)
        with self.assertRaises(ValueError):
            experiment_config("no")

    def test_missing_columns_guard(self):
        self.path.write_text("hotel,is_canceled\nCity Hotel,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            load_inner_chronological(self.path)

    def test_inner_models_only_score_inner_validation(self):
        self.write([
            self.row(2016, "March", 20, "2016-03-21", "1"),
            self.row(2016, "March", 21, "2016-03-22", "0"),
            self.row(2016, "March", 22, "2016-03-23", "1"),
            self.row(2016, "March", 23, "2016-03-24", "0"),
            self.row(2016, "April", 15, "2016-05-01", "1"),
            self.row(2016, "April", 16, "2016-05-02", "0"),
            self.row(2016, "May", 2, "2016-05-03", "1"),
            self.row(2016, "May", 3, "2016-05-04", "0"),
            self.row(2017, "January", 3, "invalid", "invalid"),
        ])
        report = model_comparison(self.path, mlp_epochs=2)
        self.assertEqual(report["inner_training_rows"], 4)
        self.assertEqual(report["inner_validation_rows"], 4)
        self.assertFalse(report["outer_validation_labels_accessed"])
        self.assertFalse(report["final_test_outcomes_accessed"])
        self.assertEqual(set(report["variants"]), {
            "with_arrival_year", "without_arrival_year"
        })
        for variant in report["variants"].values():
            self.assertEqual(set(variant["models"]), {
                "logistic_regression", "custom_numpy_mlp"
            })


if __name__ == "__main__":
    unittest.main()
