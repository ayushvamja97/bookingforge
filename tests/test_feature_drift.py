"""Deterministic feature drift tests, including sealed test cohort fields."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.feature_drift import numeric_drift, categorical_drift, audit_input_drift
from bookingforge.train_baseline import FEATURES


class FeatureDriftTests(unittest.TestCase):
    def test_numeric_shift_and_training_range(self):
        stats = numeric_drift([0, 2, 4], [2, 5])
        self.assertEqual(stats["train_mean"], 2.0)
        self.assertEqual(stats["validation_mean"], 3.5)
        self.assertAlmostEqual(stats["validation_fraction_outside_train_min_max"], 0.5)
        self.assertGreater(stats["standardized_mean_shift"], 0)

    def test_missing_values_do_not_distort_numeric_distribution(self):
        stats = numeric_drift([np.nan, 0.0, 4.0], [np.nan, 10.0])
        self.assertEqual(stats["train_observed"], 2)
        self.assertEqual(stats["validation_observed"], 1)
        self.assertEqual(stats["validation_fraction_outside_train_min_max"], 1.0)

    def test_categorical_new_values_identified(self):
        stats = categorical_drift(["A", "A", "B"], ["B", "C", "C"])
        self.assertEqual(stats["unseen_validation_category_values"], ["C"])
        self.assertAlmostEqual(stats["validation_fraction_unseen_category"], 2 / 3, places=6)

    def test_full_cohort_rows_without_test_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            def row(year, month, day, label, status):
                entry = {name: "0" for name in FEATURES}
                entry.update(
                    hotel="City Hotel", lead_time="0",
                    arrival_date_year=year,
                    arrival_date_month=month,
                    arrival_date_day_of_month=day,
                    reserved_room_type="A",
                    reservation_status_date=status,
                    is_canceled=label,
                )
                return entry
            records = [
                row("2016", "August", "31", "0", "2016-08-31"),
                row("2016", "September", "1", "1", "2017-01-03"),
                row("2017", "January", "2", "bad-target", "bad-status"),
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
                )
                writer.writeheader()
                writer.writerows(records)
            report = audit_input_drift(path)
            self.assertEqual(report["train_records"], 1)
            self.assertEqual(report["validation_records"], 1)
            self.assertFalse(report["final_test_outcomes_accessed"])
            self.assertFalse(report["individual_records_exported"])
            self.assertEqual(
                report["numeric_feature_shift"]["arrival_date_day_of_month"]["validation_mean"],
                1.0
            )


if __name__ == "__main__":
    unittest.main()
