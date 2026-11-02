"""Finite-difference gradient verification and reproducibility for NumPy MLP."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bookingforge.mlp_numpy import NumpyMLPBinary


class NumpyMLPTests(unittest.TestCase):
    def test_numerical_gradient_matches_backpropagation(self):
        rng = np.random.default_rng(7)
        model = NumpyMLPBinary(hidden_units=4, epochs=1, weight_decay=0.03)
        model._initialize(3, rng)
        model.params_["b1"][:] = 2.0
        x = np.array([[0.1, 0.2, 0.1], [-0.1, 0.3, 0.2], [0.2, -0.1, 0.0]])
        y = np.array([1.0, 0.0, 1.0])
        _, gradients = model._loss_and_grads(x, y)
        eps = 1e-5
        for name in model.params_:
            param = model.params_[name]
            for index in np.ndindex(param.shape):
                previous = param[index]
                param[index] = previous + eps
                high, _ = model._loss_and_grads(x, y)
                param[index] = previous - eps
                low, _ = model._loss_and_grads(x, y)
                param[index] = previous
                self.assertAlmostEqual(
                    gradients[name][index], (high - low) / (2 * eps), places=5,
                    msg=f"gradient mismatch: {name}{index}",
                )

    def test_fit_is_reproducible(self):
        x = np.array([[0.0], [0.2], [0.8], [1.0]])
        y = np.array([0, 0, 1, 1])
        a = NumpyMLPBinary(hidden_units=5, epochs=15, seed=55).fit(x, y)
        b = NumpyMLPBinary(hidden_units=5, epochs=15, seed=55).fit(x, y)
        np.testing.assert_array_equal(a.predict_proba_positive(x), b.predict_proba_positive(x))
        self.assertEqual(a.training_loss_history_, b.training_loss_history_)

    def test_training_reduces_loss_without_validation(self):
        x = np.array([[-1.0], [-0.5], [0.5], [1.0]] * 5)
        y = np.array([0, 0, 1, 1] * 5)
        model = NumpyMLPBinary(hidden_units=8, epochs=70, learning_rate=0.01).fit(x, y)
        self.assertLess(model.training_loss_history_[-1], model.training_loss_history_[0])
        np.testing.assert_array_equal(model.predict(x), y)

    def test_callback_and_input_validation(self):
        callbacks = []
        model = NumpyMLPBinary(hidden_units=2, epochs=3, batch_size=2).fit(
            [[0.1], [0.5], [0.9]], [0, 0, 1],
            on_epoch=lambda epoch, total, loss: callbacks.append((epoch, total, loss))
        )
        self.assertEqual([entry[0] for entry in callbacks], [1, 2, 3])
        self.assertEqual(model.predict_proba_positive(np.empty((0, 1))).shape, (0,))
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            NumpyMLPBinary(hidden_units=0)
        with self.assertRaises(ValueError):
            NumpyMLPBinary(learning_rate=0.0)
        with self.assertRaises(ValueError):
            NumpyMLPBinary().fit([[float('nan')]], [1])
        with self.assertRaises(ValueError):
            NumpyMLPBinary().fit([[1.0]], [2])


if __name__ == "__main__":
    unittest.main()
