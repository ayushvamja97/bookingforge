"""Synthetic-data tests for BookingForge creation-time cohorts."""
import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.cohort import (
    audit_creation_cohorts, cohort_at, reconstructed_creation,
    validate_booking_features, PROVISIONAL_BOOKING_FEATURES,
)


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sample.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_feature_allowlist_fails_closed(self):
        validate_booking_features(["hotel", "lead_time"])
        for feature in ("reservation_status", "booking_changes", "is_canceled"):
            with self.assertRaisesRegex(ValueError, "Not approved"):
                validate_booking_features([feature])
        with self.assertRaises(ValueError):
            validate_booking_features([])
        with self.assertRaises(ValueError):
            validate_booking_features(["hotel", "hotel"])

    def test_creation_and_cohort_boundaries(self):
        self.assertEqual(reconstructed_creation(date(2017, 1, 2), "2"), date(2016, 12, 31))
        self.assertEqual(cohort_at(date(2016, 8, 31)), "train")
        self.assertEqual(cohort_at(date(2016, 9, 1)), "validation")
        self.assertEqual(cohort_at(date(2017, 1, 1)), "test")
        self.assertEqual(cohort_at(date(2017, 9, 1)), "outside")
        with self.assertRaises(ValueError):
            reconstructed_creation(date(2017, 1, 1), "-3")

    def write(self, rows):
        fields = sorted(PROVISIONAL_BOOKING_FEATURES | {"is_canceled", "reservation_status_date"})
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def row(**changes):
        row = {col: "0" for col in PROVISIONAL_BOOKING_FEATURES}
        row.update(
            hotel="City Hotel", arrival_date_year="2016", arrival_date_month="August",
            arrival_date_day_of_month="31", lead_time="0", is_canceled="0",
            reservation_status_date="2016-09-02",
        )
        row.update(changes)
        return row

    def test_censored_labels_are_counted(self):
        self.write([self.row(), self.row(reservation_status_date="2016-08-30", is_canceled="1")])
        report = audit_creation_cohorts(self.path)
        self.assertEqual(report["creation_cohort_counts"]["train"], 2)
        self.assertEqual(report["label_maturity"], {
            "train_known_at_train_cutoff": 1, "train_not_known_at_train_cutoff": 1,
        })
        self.assertFalse(report["training_authorized"])

    def test_status_before_creation_is_flagged(self):
        self.write([self.row(reservation_status_date="2016-08-29")])
        report = audit_creation_cohorts(self.path)
        self.assertEqual(report["other_flags"]["final_status_before_reconstructed_creation"], 1)

    def test_invalid_records_do_not_enter_cohorts(self):
        self.write([self.row(arrival_date_month="NotAMonth")])
        report = audit_creation_cohorts(self.path)
        self.assertEqual(report["other_flags"]["invalid_rows"], 1)
        self.assertEqual(sum(report["creation_cohort_counts"].values()), 0)


if __name__ == "__main__":
    unittest.main()
