"""Tests for conservative, preliminary BookingForge cancellation-baseline data selection."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.train_baseline import (
    FEATURES, NUMERIC, CATEGORICAL, load_matured_cohorts
)


class BaselineDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "synthetic.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, **changes):
        sample = {name: "0" for name in FEATURES}
        sample.update({
            "hotel": "City Hotel", "lead_time": "0",
            "arrival_date_year": "2016", "arrival_date_month": "August",
            "arrival_date_day_of_month": "31", "reserved_room_type": "A",
            "stays_in_weekend_nights": "1",
            "reservation_status_date": "2016-08-31", "is_canceled": "1",
        })
        sample.update(changes)
        return sample

    def write(self, rows):
        columns = list(FEATURES) + ["reservation_status_date", "is_canceled"]
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_only_matured_labels_enter_training(self):
        self.write([
            self.row(),
            self.row(is_canceled="0", reservation_status_date="2016-09-02"),
            self.row(arrival_date_month="September", arrival_date_day_of_month="01",
                     reservation_status_date="2016-10-01", is_canceled="0"),
            self.row(arrival_date_month="September", arrival_date_day_of_month="02",
                     reservation_status_date="2017-01-04", is_canceled="1"),
        ])
        samples, labels, counters = load_matured_cohorts(self.path)
        self.assertEqual(labels["train"], [1])
        self.assertEqual(labels["validation"], [0])
        self.assertEqual(counters["train_outcome_unknown_at_cutoff"], 1)
        self.assertEqual(counters["validation_outcome_unknown_at_cutoff"], 1)
        self.assertEqual(len(samples["train"]), 1)

    def test_final_test_labels_are_not_examined(self):
        self.write([
            self.row(arrival_date_year="2017", arrival_date_month="January",
                     arrival_date_day_of_month="02",
                     is_canceled="not-a-label",
                     reservation_status_date="not-a-date"),
        ])
        samples, labels, counters = load_matured_cohorts(self.path)
        self.assertEqual(counters["cohort_test"], 1)
        self.assertEqual(counters.get("test_invalid_label_or_status_date", 0), 0)
        self.assertEqual(labels, {"train": [], "validation": []})
        self.assertEqual(samples, {"train": [], "validation": []})

    def test_features_do_not_include_outcomes(self):
        self.assertEqual(set(FEATURES), set(NUMERIC) | set(CATEGORICAL))
        self.assertFalse(set(FEATURES) & {"is_canceled", "reservation_status_date", "reservation_status"})
        self.assertNotIn("assigned_room_type", FEATURES)
        self.assertNotIn("booking_changes", FEATURES)

    def test_missing_header_fails(self):
        self.path.write_text("is_canceled,lead_time\n1,3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_matured_cohorts(self.path)


if __name__ == "__main__":
    unittest.main()
