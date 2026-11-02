"""Tests for the aggregate, non-training data audit."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.data_audit import audit_dataset


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sample.csv"
        self.columns = [
            "hotel", "is_canceled", "lead_time", "arrival_date_year",
            "arrival_date_month", "arrival_date_day_of_month",
            "stays_in_weekend_nights", "stays_in_week_nights",
            "reservation_status", "reservation_status_date", "country",
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, **updates):
        result = {
            "hotel": "City Hotel", "is_canceled": "0", "lead_time": "4",
            "arrival_date_year": "2016", "arrival_date_month": "August",
            "arrival_date_day_of_month": "31",
            "stays_in_weekend_nights": "1", "stays_in_week_nights": "1",
            "reservation_status": "Check-Out",
            "reservation_status_date": "2016-09-02", "country": "PRT",
        }
        result.update(updates)
        return result

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_counts_splits_and_missing(self):
        self.write_rows([
            self.row(),
            self.row(arrival_date_month="September", arrival_date_day_of_month="01",
                     is_canceled="1", reservation_status="Canceled", country=""),
            self.row(arrival_date_year="2017", arrival_date_month="January",
                     arrival_date_day_of_month="01"),
        ])
        report = audit_dataset(self.path)
        self.assertEqual(report["row_count"], 3)
        self.assertEqual(report["provisional_split_counts"], {
            "test": 1, "train": 1, "validation": 1
        })
        self.assertEqual(report["missing_by_column"], {"country": 1})
        self.assertEqual(report["provisional_split_cancellation_rates"]["validation"], 1.0)

    def test_invalid_date_is_counted(self):
        self.write_rows([self.row(arrival_date_month="Smarch")])
        self.assertEqual(
            audit_dataset(self.path)["potential_data_quality_flags"],
            {"invalid_arrival_date": 1},
        )

    def test_required_columns_guard(self):
        self.path.write_text("hotel,is_canceled\nCity Hotel,0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Required columns missing"):
            audit_dataset(self.path)

    def test_never_certifies_training_protocol(self):
        self.write_rows([self.row()])
        report = audit_dataset(self.path)
        self.assertIn("PROVISIONAL", report["evaluation_status"])
        self.assertIn("reservation_status", report["forbidden_predictors"])


if __name__ == "__main__":
    unittest.main()
