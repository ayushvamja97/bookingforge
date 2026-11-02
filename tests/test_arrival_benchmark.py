"""Synthetic edge-case and fit-integration tests for arrival-based benchmark."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.arrival_benchmark import (
    load_arrival_benchmark, run_arrival_benchmark, selected_features,
)
from bookingforge.train_baseline import FEATURES


class ArrivalBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp.name) / "arrival.csv"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, year, month, day, status, label):
        sample = {key: "0" for key in FEATURES}
        sample.update({
            "hotel": "City Hotel",
            "lead_time": "0",
            "arrival_date_year": str(year),
            "arrival_date_month": month,
            "arrival_date_day_of_month": str(day),
            "reserved_room_type": "A",
            "adults": "2",
            "reservation_status_date": status,
            "is_canceled": label,
        })
        return sample

    def write(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_train_fit_status_cutoff_and_2017_sealing(self):
        self.write([
            self.row(2016, "August", 15, "2016-08-17", "0"),
            self.row(2016, "August", 20, "2016-10-01", "1"),
            self.row(2016, "September", 15, "invalid", "invalid"),
            self.row(2016, "October", 10, "2017-01-03", "1"),
            self.row(2017, "January", 3, "invalid", "invalid"),
        ])
        samples, labels, counters = load_arrival_benchmark(self.csv_path)
        self.assertEqual(labels["train"], [0])
        self.assertEqual(labels["validation"], [1])
        self.assertEqual(counters["train_outcome_not_yet_known_at_fit_cutoff"], 1)
        self.assertEqual(counters["excluded_outside_target_arrival_windows"], 2)
        self.assertEqual(counters.get("validation_invalid_record", 0), 0)

    def test_only_approved_year_feature_removed(self):
        full = selected_features(True)[2]
        reduced = selected_features(False)[2]
        self.assertIn("arrival_date_year", full)
        self.assertNotIn("arrival_date_year", reduced)
        self.assertEqual(set(full) - set(reduced), {"arrival_date_year"})
        with self.assertRaises(ValueError):
            selected_features("false")

    def test_missing_columns_rejected(self):
        self.csv_path.write_text("hotel,is_canceled\nCity Hotel,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            load_arrival_benchmark(self.csv_path)

    def test_model_comparison_without_any_future_year_labels(self):
        records = [
            self.row(2016, "July", 1, "2016-07-02", "0"),
            self.row(2016, "July", 2, "2016-07-03", "1"),
            self.row(2016, "July", 3, "2016-07-04", "0"),
            self.row(2016, "July", 4, "2016-07-05", "1"),
            self.row(2016, "October", 11, "2016-10-12", "0"),
            self.row(2016, "October", 12, "2016-10-13", "1"),
            self.row(2016, "November", 11, "2017-01-03", "0"),
            self.row(2016, "November", 12, "2017-01-03", "1"),
            self.row(2017, "January", 3, "invalid", "invalid"),
        ]
        self.write(records)
        result = run_arrival_benchmark(self.csv_path, mlp_epochs=2)
        self.assertEqual(result["train_records"], 4)
        self.assertEqual(result["validation_records"], 4)
        self.assertFalse(result["final_2017_outcomes_accessed"])
        self.assertEqual(
            set(result["variants"]), {"with_arrival_year", "without_arrival_year"}
        )
        for info in result["variants"].values():
            self.assertEqual(info["training_records_used"], 4)
            self.assertEqual(info["validation_records_scored"], 4)
            self.assertEqual(
                set(info["metrics"]), {"logistic_regression", "custom_numpy_mlp"}
            )


if __name__ == "__main__":
    unittest.main()
