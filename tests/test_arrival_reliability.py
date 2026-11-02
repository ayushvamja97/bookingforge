"""Unit tests for descriptive, validation-only reliability and uncertainty audit."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.arrival_reliability import (
    paired_bootstrap_mlp, reliability_bins, run_reliability, subgroup_metrics,
)
from bookingforge.arrival_benchmark import run_arrival_benchmark
from bookingforge.train_baseline import FEATURES


class ArrivalReliabilityTests(unittest.TestCase):
    def test_reliability_bins_handle_edges_and_empty_bins(self):
        report = reliability_bins([0, 1, 0, 1], [0.0, 0.2, 0.9, 1.0], n_bins=5)
        self.assertEqual(sum(b["count"] for b in report["bins"]), 4)
        self.assertEqual(report["bins"][0]["count"], 1)
        self.assertEqual(report["bins"][-1]["count"], 2)
        self.assertEqual(report["total_records"], 4)
        self.assertGreaterEqual(report["expected_calibration_error_equal_width"], 0)
        self.assertIsNone(report["bins"][1]["predicted_mean"])

    def test_single_outcome_group_does_not_invent_auc(self):
        report = subgroup_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(report["roc_auc"])
        self.assertIsNone(report["average_precision"])
        self.assertAlmostEqual(report["brier_score"], 0.025)
        self.assertEqual(report["outcome_rate"], 0.0)

    def test_paired_bootstrap_uses_same_validation_rows(self):
        y = [0, 1, 0, 1, 0, 1]
        p_a = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
        p_b = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        first = paired_bootstrap_mlp(y, p_a, p_b, replicates=25, seed=9)
        second = paired_bootstrap_mlp(y, p_a, p_b, replicates=25, seed=9)
        self.assertEqual(first, second)
        self.assertLess(first["brier_difference"], 0)
        self.assertEqual(first["roc_auc_difference"], 0.0)
        lo, hi = first["brier_95_percentile_interval"]
        self.assertLess(hi, 0)

    def test_invalid_values_or_replicates_rejected(self):
        with self.assertRaises(ValueError):
            reliability_bins([0, 1], [0.5], n_bins=10)
        with self.assertRaises(ValueError):
            reliability_bins([0, 1], [0.5, 1.2], n_bins=10)
        with self.assertRaises(ValueError):
            reliability_bins([0, 1], [0.5, 0.5], n_bins=1)
        with self.assertRaises(ValueError):
            paired_bootstrap_mlp([0, 1], [0.1, 0.9], [0.2, 0.8], replicates=2)
        with self.assertRaises(ValueError):
            paired_bootstrap_mlp([0, 0], [0.1, 0.2], [0.2, 0.3], replicates=20)

    def test_reliability_reproduces_fixed_benchmark_without_future_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"

            def row(year, month, day, outcome_date, target, hotel="City Hotel"):
                item = {feature: "0" for feature in FEATURES}
                item.update({
                    "hotel": hotel,
                    "lead_time": "0",
                    "arrival_date_year": year,
                    "arrival_date_month": month,
                    "arrival_date_day_of_month": day,
                    "reserved_room_type": "A",
                    "adults": "2",
                    "reservation_status_date": outcome_date,
                    "is_canceled": target,
                })
                return item

            records = [
                row("2016", "July", "1", "2016-07-02", "0"),
                row("2016", "July", "2", "2016-07-03", "1"),
                row("2016", "July", "3", "2016-07-04", "0", "Resort Hotel"),
                row("2016", "July", "4", "2016-07-05", "1", "Resort Hotel"),
                row("2016", "October", "11", "2016-10-12", "0"),
                row("2016", "October", "12", "2016-10-13", "1"),
                row("2016", "November", "11", "2017-01-03", "0", "Resort Hotel"),
                row("2016", "November", "12", "2017-01-03", "1", "Resort Hotel"),
                # Future label and status must be ignored before parsing.
                row("2017", "January", "3", "invalid", "invalid"),
            ]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
                )
                writer.writeheader()
                writer.writerows(records)

            benchmark = run_arrival_benchmark(path, mlp_epochs=2)
            report = run_reliability(path, mlp_epochs=2, bootstrap_replicates=25)
            self.assertEqual(report["training_records"], 4)
            self.assertEqual(report["retrospective_validation_records"], 4)
            self.assertFalse(report["outcomes_from_2017_arrivals_accessed_by_this_runner"])
            self.assertFalse(report["models_or_thresholds_selected_using_validation_labels"])
            self.assertFalse(report["calibrators_fit"])
            expected = {
                "logistic_with_year": ("with_arrival_year", "logistic_regression"),
                "logistic_without_year": ("without_arrival_year", "logistic_regression"),
                "mlp_with_year": ("with_arrival_year", "custom_numpy_mlp"),
                "mlp_without_year": ("without_arrival_year", "custom_numpy_mlp"),
            }
            for name, (variant, algorithm) in expected.items():
                old = benchmark["variants"][variant]["metrics"][algorithm]
                new = report["fixed_models"][name]["full_validation"]
                self.assertEqual(old["brier_score"], new["brier_score"])
                self.assertEqual(old["roc_auc"], new["roc_auc"])
                self.assertEqual(sum(b["count"] for b in report["fixed_models"][name]["reliability"]["bins"]), 4)
                self.assertEqual(sum(x["count"] for x in report["fixed_models"][name]["hotel"].values()), 4)
                self.assertEqual(sum(x["count"] for x in report["fixed_models"][name]["arrival_month"].values()), 4)


if __name__ == "__main__":
    unittest.main()
