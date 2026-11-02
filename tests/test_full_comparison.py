"""A full-cohort comparison smoke test on synthetic data; no hidden sampling."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.compare_knn import run_full_comparison
from bookingforge.train_baseline import FEATURES


class FullComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "synthetic.csv"

    def tearDown(self):
        self.temp.cleanup()

    def row(self, day: int, month: str, label: str, final_date: str):
        row = {feature: "0" for feature in FEATURES}
        row.update({
            "hotel": "City Hotel", "lead_time": "0",
            "arrival_date_year": "2016",
            "arrival_date_month": month,
            "arrival_date_day_of_month": str(day),
            "stays_in_weekend_nights": "1",
            "stays_in_week_nights": "1",
            "adults": "2",
            "reserved_room_type": "A",
            "reservation_status_date": final_date,
            "is_canceled": label,
        })
        return row

    def test_uses_all_validation_rows_and_does_not_read_test_label(self):
        rows = [
            self.row(25, "August", "0", "2016-08-31"),
            self.row(26, "August", "1", "2016-08-31"),
            self.row(27, "August", "0", "2016-08-31"),
            self.row(28, "August", "1", "2016-08-31"),
            self.row(1, "September", "0", "2017-01-03"),
            self.row(2, "September", "1", "2017-01-04"),
            self.row(3, "September", "0", "2017-01-05"),
            self.row(4, "September", "1", "2017-01-06"),
            # Invalid target and status are safe because test cohort labels remain sealed.
            dict(self.row(2, "January", "garbage", "garbage"), arrival_date_year="2017"),
        ]
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FEATURES) + [
                "reservation_status_date", "is_canceled"
            ])
            writer.writeheader()
            writer.writerows(rows)

        report = run_full_comparison(self.path, k=1, batch_size=2)
        self.assertEqual(report["training_rows"], 4)
        self.assertEqual(report["validation_rows"], 4)
        self.assertFalse(report["final_test_labels_accessed"])
        self.assertEqual(
            set(report["validation_metrics"]),
            {"prior_probability", "logistic_regression", "custom_numpy_knn"},
        )
        self.assertEqual(report["custom_knn"]["training_rows_used"], 4)
        self.assertEqual(report["custom_knn"]["validation_rows_scored"], 4)


if __name__ == "__main__":
    unittest.main()
