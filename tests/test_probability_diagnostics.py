"""Small deterministic tests for prediction-distribution diagnostics."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.probability_diagnostics import summarize_probabilities


class ProbabilityDiagnosticsTests(unittest.TestCase):
    def test_perfect_probabilities_have_zero_brier(self):
        s = summarize_probabilities([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(s["brier"], 0.0)
        self.assertEqual(s["roc_auc"], 1.0)
        self.assertEqual(s["target_prevalence"], 0.5)
        self.assertEqual(s["predicted_mean"], 0.5)
        self.assertEqual(s["p_below_0_01_fraction"], 0.5)
        self.assertEqual(s["p_above_0_99_fraction"], 0.5)

    def test_extreme_wrong_predictions_expose_calibration(self):
        s = summarize_probabilities([0, 1], [1.0, 0.0])
        self.assertEqual(s["brier"], 1.0)
        self.assertEqual(s["roc_auc"], 0.0)
        self.assertEqual(s["brier_by_truth"]["ultimately_checked_out"], 1.0)
        self.assertEqual(s["brier_by_truth"]["ultimately_canceled_or_no_show"], 1.0)

    def test_single_class_has_no_auc(self):
        s = summarize_probabilities([0, 0], [0.2, 0.3])
        self.assertIsNone(s["roc_auc"])
        self.assertIsNone(s["brier_by_truth"]["ultimately_canceled_or_no_show"])

    def test_invalid_probabilities_and_shapes_rejected(self):
        for labels, probabilities in (
            ([0, 1], [0.1]),
            ([0, 1], [float("nan"), 0.5]),
            ([0, 1], [-0.1, 0.5]),
            ([0, 2], [0.4, 0.5]),
            ([], []),
        ):
            with self.assertRaises(ValueError):
                summarize_probabilities(labels, probabilities)


if __name__ == "__main__":
    unittest.main()
