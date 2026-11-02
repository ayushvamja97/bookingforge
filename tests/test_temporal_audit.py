"""Synthetic fixtures for BookingForge temporal data-quality diagnostics."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.temporal_audit import temporal_audit, parse_arrival, arrival_split
from datetime import date


class TemporalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp.name) / "bookings.csv"
        self.fields = [
            "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month",
            "lead_time", "reservation_status_date", "reservation_status", "is_canceled",
            "stays_in_week_nights", "stays_in_weekend_nights",
        ]

    def tearDown(self):
        self.temp.cleanup()

    def row(self, **changes):
        example = dict(
            arrival_date_year="2016", arrival_date_month="August",
            arrival_date_day_of_month="31", lead_time="4",
            reservation_status_date="2016-09-02",
            reservation_status="Check-Out", is_canceled="0",
            stays_in_week_nights="2", stays_in_weekend_nights="0",
        )
        example.update(changes)
        return example

    def write_rows(self, rows):
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_arrival_month_and_boundary(self):
        self.assertEqual(parse_arrival(self.row()), date(2016, 8, 31))
        self.assertEqual(arrival_split(date(2016, 8, 31)), "train")
        self.assertEqual(arrival_split(date(2016, 9, 1)), "validation")
        self.assertEqual(arrival_split(date(2017, 1, 1)), "test")
        self.assertEqual(arrival_split(date(2017, 9, 1)), "outside")

    def test_final_status_after_training_cutoff(self):
        self.write_rows([self.row()])
        audit = temporal_audit(self.csv_path)
        self.assertEqual(audit["final_status_dated_after_own_split_cutoff"], {"train": 1})
        self.assertEqual(audit["reconstructed_creation_range"]["earliest"], "2016-08-27")
        self.assertFalse(audit["training_authorized"])

    def test_zero_nights_and_before_creation(self):
        self.write_rows([self.row(
            lead_time="0", reservation_status_date="2016-08-30",
            reservation_status="Canceled", is_canceled="1",
            stays_in_week_nights="0", stays_in_weekend_nights="0",
        )])
        audit = temporal_audit(self.csv_path)
        self.assertEqual(audit["zero_night_booking_by_final_status"], {"Canceled": 1})
        self.assertEqual(audit["counts"]["status_before_creation_proxy"], 1)
        self.assertEqual(audit["final_status_dated_before_arrival"], {"Canceled": 1})

    def test_invalid_date_and_required_columns(self):
        self.write_rows([self.row(reservation_status_date="bad-date")])
        self.assertEqual(temporal_audit(self.csv_path)["counts"]["invalid_status_date"], 1)
        self.csv_path.write_text("lead_time\n4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            temporal_audit(self.csv_path)


if __name__ == "__main__":
    unittest.main()
