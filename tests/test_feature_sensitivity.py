"""Synthetic tests for label-blind, training-median prediction sensitivity."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.feature_sensitivity import (
    prediction_shift,
    replace_numeric_with_training_median,
    run_sensitivity,
)
from bookingforge.train_baseline import FEATURES


class FeatureSensitivityTests(unittest.TestCase):
    def test_prediction_change_summary(self):
        values = prediction_shift([0.2, 0.95], [0.6, 0.98])
        self.assertEqual(values["n"], 2)
        self.assertAlmostEqual(values["original_mean_probability"], 0.575)
        self.assertAlmostEqual(values["intervened_mean_probability"], 0.79)
        self.assertAlmostEqual(values["mean_absolute_delta"], 0.215)
        self.assertEqual(values["original_fraction_probability_over_0_99"], 0.0)

    def test_intervention_does_not_modify_original_data(self):
        training = pd.DataFrame({"arrival_date_year": [2015, 2016, 2016]})
        validation = pd.DataFrame({"arrival_date_year": [2016, 2017]})
        altered, median = replace_numeric_with_training_median(
            training, validation, "arrival_date_year"
        )
        self.assertEqual(median, 2016.0)
        self.assertEqual(altered["arrival_date_year"].tolist(), [2016.0, 2016.0])
        self.assertEqual(validation["arrival_date_year"].tolist(), [2016, 2017])
        with self.assertRaises(ValueError):
            replace_numeric_with_training_median(training, validation, "unknown")

    def test_invalid_probability_shapes_rejected(self):
        for a, b in (([0.5], [0.5, 0.4]), ([], []),
                     ([float("nan")], [0.2]), ([1.1], [0.2])):
            with self.assertRaises(ValueError):
                prediction_shift(a, b)

    def test_retrospective_validation_probe_does_not_read_test_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            def row(year, month, day, final, target):
                result = {feature: "0" for feature in FEATURES}
                result.update({
                    "hotel": "City Hotel", "lead_time": "0",
                    "arrival_date_year": year, "arrival_date_month": month,
                    "arrival_date_day_of_month": day, "adults": "2",
                    "reserved_room_type": "A",
                    "reservation_status_date": final, "is_canceled": target,
                })
                return result
            rows = [
                row("2016", "August", "25", "2016-08-31", "0"),
                row("2016", "August", "26", "2016-08-31", "1"),
                row("2016", "August", "27", "2016-08-31", "0"),
                row("2016", "August", "28", "2016-08-31", "1"),
                row("2016", "September", "1", "2017-01-03", "1"),
                row("2016", "September", "2", "2017-01-05", "0"),
                row("2016", "September", "3", "2017-01-06", "0"),
                dict(row("2017", "January", "2", "invalid", "invalid")),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
                )
                writer.writeheader()
                writer.writerows(rows)
            report = run_sensitivity(path, mlp_epochs=2)
            self.assertEqual(report["train_records"], 4)
            self.assertEqual(report["validation_records"], 3)
            self.assertFalse(report["validation_labels_used_in_interventions"])
            self.assertFalse(report["final_test_outcomes_accessed"])
            self.assertEqual(
                set(report["interventions"]),
                {"arrival_date_year", "lead_time"},
            )
            self.assertEqual(
                report["observed_validation_arrival_year_counts"], {"2016": 3}
            )


if __name__ == "__main__":
    unittest.main()
