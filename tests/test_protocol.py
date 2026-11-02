import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.protocol import TemporalSplit, validate_features


class ProtocolTests(unittest.TestCase):
    def test_split_boundaries(self):
        split = TemporalSplit(date(2016, 8, 31), date(2016, 12, 31), date(2017, 8, 31))
        self.assertEqual(split.assign(date(2016, 8, 31)), "train")
        self.assertEqual(split.assign(date(2016, 9, 1)), "validation")
        self.assertEqual(split.assign(date(2017, 1, 1)), "test")
        self.assertEqual(split.assign(date(2017, 9, 1)), "outside")

    def test_invalid_boundaries(self):
        with self.assertRaises(ValueError):
            TemporalSplit(date(2017, 1, 1), date(2016, 1, 1), date(2018, 1, 1))

    def test_outcome_leakage_blocked(self):
        with self.assertRaises(ValueError):
            validate_features(["lead_time", "reservation_status"])

    def test_non_outcome_features_not_automatically_approved(self):
        validate_features(["lead_time"])


if __name__ == "__main__":
    unittest.main()
