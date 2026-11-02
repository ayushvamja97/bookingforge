"""Tests for training-only inner cohort composition and benchmark."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bookingforge.inner_cohort_diagnostics import (
    diagnostic_report, population_description,
)
from bookingforge.train_baseline import FEATURES


class InnerCohortDiagnosticTests(unittest.TestCase):
    def test_population_counts_and_years(self):
        rows = [
            {"arrival_date_year": "2015", "hotel": "City Hotel"},
            {"arrival_date_year": "2016", "hotel": "Resort Hotel"},
            {"arrival_date_year": "2016", "hotel": "City Hotel"},
        ]
        stats = population_description(rows, [0, 1, 1])
        self.assertEqual(stats["records"], 3)
        self.assertEqual(stats["cancel_or_no_show_count"], 2)
        self.assertEqual(stats["arrival_years"]["2016"]["records"], 2)
        self.assertEqual(stats["hotel"]["City Hotel"]["records"], 2)

    def test_wrong_lengths_and_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            population_description([{"arrival_date_year": "2016", "hotel": "City Hotel"}], [])
        with self.assertRaises(ValueError):
            population_description([], [])
        with self.assertRaises(ValueError):
            population_description([{"arrival_date_year": "2016", "hotel": "City Hotel"}], [2])

    def test_inner_historical_prior_scored_only_on_inner_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            def row(year, month, day, status, label):
                sample = {name: "0" for name in FEATURES}
                sample.update({
                    "hotel": "City Hotel", "lead_time": "0",
                    "arrival_date_year": year, "arrival_date_month": month,
                    "arrival_date_day_of_month": day, "reserved_room_type": "A",
                    "reservation_status_date": status, "is_canceled": label,
                })
                return sample

            records = [
                row("2016", "March", "20", "2016-03-22", "1"),
                row("2016", "March", "21", "2016-03-23", "0"),
                row("2016", "April", "15", "2016-05-01", "0"),
                row("2016", "May", "02", "2016-06-03", "1"),
                # This historical booking's eventual outcome is NOT fit-visible.
                row("2016", "March", "24", "2016-04-04", "1"),
                # Must skip outside rows without parsing final label fields.
                row("2016", "September", "02", "bad", "bad"),
                row("2017", "January", "02", "bad", "bad"),
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=FEATURES + ["reservation_status_date", "is_canceled"]
                )
                writer.writeheader()
                writer.writerows(records)
            report = diagnostic_report(path)
            self.assertEqual(report["train"]["records"], 2)
            self.assertEqual(report["validation"]["records"], 2)
            self.assertEqual(report["train"]["cancel_or_no_show_fraction"], 0.5)
            self.assertEqual(report["training_prior_on_inner_validation"]["brier_score"], 0.25)
            self.assertEqual(report["training_prior_on_inner_validation"]["roc_auc"], 0.5)
            self.assertFalse(report["outer_validation_labels_accessed"])
            self.assertFalse(report["final_test_outcomes_accessed"])

    def test_years_with_missing_values_are_flagged(self):
        rows = [
            {"arrival_date_year": "", "hotel": "City Hotel"},
            {"arrival_date_year": "2016", "hotel": "City Hotel"},
        ]
        result = population_description(rows, [1, 0])
        self.assertEqual(result["invalid_arrival_year_count"], 1)
        self.assertEqual(result["arrival_years"]["2016"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
