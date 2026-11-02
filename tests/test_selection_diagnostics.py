"""Synthetic-fixture tests for selection bias diagnostics."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.selection_diagnostics import selection_diagnostics, lead_time_bucket


class SelectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "synthetic.csv"
        self.fields = [
            "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
            "lead_time", "reservation_status_date", "reservation_status",
            "is_canceled", "hotel",
        ]

    def tearDown(self):
        self.temp.cleanup()

    def row(self, **updates):
        sample = {
            "arrival_date_year": "2016", "arrival_date_month": "August",
            "arrival_date_day_of_month": "31", "lead_time": "0",
            "reservation_status_date": "2016-08-31", "reservation_status": "Canceled",
            "is_canceled": "1", "hotel": "City Hotel",
        }
        sample.update(updates)
        return sample

    def write(self, records):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(records)

    def test_maturity_selection_changes_target_prevalence(self):
        self.write([
            self.row(),
            self.row(is_canceled="0", reservation_status="Check-Out",
                     reservation_status_date="2016-09-02"),
        ])
        report = selection_diagnostics(self.path)
        train = report["cohorts"]["train"]
        self.assertEqual(train["records"], 2)
        self.assertEqual(train["matured_records"], 1)
        self.assertEqual(train["not_matured_records"], 1)
        self.assertEqual(train["final_cancel_rate_matured"], 1.0)
        self.assertEqual(train["final_cancel_rate_not_matured"], 0.0)
        self.assertEqual(train["final_cancel_rate_all_records_retrospective"], 0.5)
        self.assertFalse(report["test_outcomes_accessed"])

    def test_test_labels_are_never_read(self):
        self.write([self.row(
            arrival_date_year="2017", arrival_date_month="January",
            arrival_date_day_of_month="1",
            reservation_status_date="invalid", is_canceled="invalid",
        )])
        report = selection_diagnostics(self.path)
        self.assertEqual(report["other_counts"]["not_train_or_validation"], 1)
        self.assertEqual(report["cohorts"]["train"]["records"], 0)
        self.assertEqual(report["cohorts"]["validation"]["records"], 0)
        self.assertFalse(report["test_outcomes_accessed"])

    def test_lead_time_buckets(self):
        self.assertEqual([lead_time_bucket(i) for i in (0, 7, 8, 30, 31, 90, 91)],
                         ["0-7", "0-7", "8-30", "8-30", "31-90", "31-90", "91+"])
        with self.assertRaises(ValueError):
            lead_time_bucket(-1)

    def test_missing_columns_rejected(self):
        self.path.write_text("hotel,is_canceled\nCity Hotel,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Required columns missing"):
            selection_diagnostics(self.path)


if __name__ == "__main__":
    unittest.main()
