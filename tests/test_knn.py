"""Ground-truth examples for the from-scratch NumPy kNN implementation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.knn import NumpyKNNBinary


class KNNTests(unittest.TestCase):
    def test_one_neighbor_matches_nearest_training_point(self):
        model = NumpyKNNBinary(k=1, batch_size=1)
        model.fit([[0.0], [3.0], [10.0]], [0, 1, 1])
        np.testing.assert_array_equal(model.predict([[0.1], [9.9]]), [0, 1])

    def test_three_neighbors_vote_is_a_probability(self):
        model = NumpyKNNBinary(k=3)
        model.fit([[0.0], [1.0], [2.0], [9.0]], [0, 1, 1, 0])
        self.assertAlmostEqual(model.predict_proba_positive([[1.0]])[0], 2 / 3)

    def test_batch_size_does_not_change_predictions(self):
        train = [[0.0, 0.0], [3.0, 1.0], [5.0, 4.0], [8.0, 7.0]]
        labels = [0, 0, 1, 1]
        queries = [[1.0, 0.0], [6.0, 5.0], [9.0, 8.0]]
        first = NumpyKNNBinary(k=1, batch_size=1).fit(train, labels)
        second = NumpyKNNBinary(k=1, batch_size=20).fit(train, labels)
        np.testing.assert_array_equal(first.predict(queries), second.predict(queries))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=0)
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=1, batch_size=0)
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=1).predict([[0]])
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=3).fit([[1.0], [2.0]], [0, 1])
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=1).fit([[float("nan")]], [1])
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=1).fit([[1.0]], [2])
        with self.assertRaises(ValueError):
            NumpyKNNBinary(k=1).fit([[1.0]], [1]).predict([[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
