"""One-hidden-layer binary MLP implemented in NumPy with backpropagation.

No deep-learning framework is used. Supports mini-batch Adam and weight decay.
All optimizer updates use only the supplied training features and labels.
This is an educational/research implementation, not production-calibrated.
"""
from __future__ import annotations

import numpy as np


class NumpyMLPBinary:
    """ReLU hidden layer, sigmoid output, binary cross-entropy and Adam."""

    def __init__(self, hidden_units: int = 32, epochs: int = 30,
                 batch_size: int = 256, learning_rate: float = 0.001,
                 weight_decay: float = 0.0001, seed: int = 42):
        for name, value in (("hidden_units", hidden_units), ("epochs", epochs),
                            ("batch_size", batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        self.params_ = None
        self.training_loss_history_ = []
        self.n_features_in_ = None

    @staticmethod
    def _sigmoid(logit):
        return np.exp(-np.logaddexp(0.0, -logit))

    def _initialize(self, num_features: int, rng: np.random.Generator) -> None:
        self.params_ = {
            "W1": rng.standard_normal((num_features, self.hidden_units)) * np.sqrt(2.0 / num_features),
            "b1": np.zeros(self.hidden_units),
            "W2": rng.standard_normal((self.hidden_units, 1)) * np.sqrt(1.0 / self.hidden_units),
            "b2": np.zeros(1),
        }
        self.n_features_in_ = num_features

    def _forward(self, x: np.ndarray):
        if self.params_ is None:
            raise ValueError("MLP must be fitted first")
        p = self.params_
        hidden_pre = x @ p["W1"] + p["b1"]
        hidden = np.maximum(0.0, hidden_pre)
        logits = hidden @ p["W2"] + p["b2"]
        probs = self._sigmoid(logits)
        return hidden_pre, hidden, logits, probs

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        """Analytic gradients for binary cross-entropy with L2 on weights."""
        n = len(x)
        hidden_pre, hidden, logits, probs = self._forward(x)
        target = y.reshape(-1, 1)
        p = self.params_
        data_loss = np.mean(np.logaddexp(0.0, logits) - target * logits)
        reg_loss = 0.5 * self.weight_decay * (
            np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2)
        )
        dz2 = (probs - target) / n
        grads = {
            "W2": hidden.T @ dz2 + self.weight_decay * p["W2"],
            "b2": dz2.sum(axis=0),
        }
        dh = dz2 @ p["W2"].T
        dhidden_pre = dh * (hidden_pre > 0.0)
        grads["W1"] = x.T @ dhidden_pre + self.weight_decay * p["W1"]
        grads["b1"] = dhidden_pre.sum(axis=0)
        return float(data_loss + reg_loss), grads

    @staticmethod
    def _validate_xy(features, labels):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("Training features must be nonempty 2D")
        if y.ndim != 1 or len(y) != len(x) or not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be a 1D binary vector matching features")
        if not np.isfinite(x).all():
            raise ValueError("Training features must be finite")
        return x, y.astype(np.float64)

    def fit(self, features, labels, on_epoch=None):
        x, y = self._validate_xy(features, labels)
        rng = np.random.default_rng(self.seed)
        self._initialize(x.shape[1], rng)
        self.training_loss_history_ = []
        m = {key: np.zeros_like(value) for key, value in self.params_.items()}
        v = {key: np.zeros_like(value) for key, value in self.params_.items()}
        step = 0
        for epoch in range(1, self.epochs + 1):
            permutation = rng.permutation(len(x))
            for start in range(0, len(x), self.batch_size):
                idx = permutation[start:start + self.batch_size]
                _, gradients = self._loss_and_grads(x[idx], y[idx])
                step += 1
                for key in self.params_:
                    grad = gradients[key]
                    m[key] = 0.9 * m[key] + 0.1 * grad
                    v[key] = 0.999 * v[key] + 0.001 * (grad ** 2)
                    m_corrected = m[key] / (1 - 0.9 ** step)
                    v_corrected = v[key] / (1 - 0.999 ** step)
                    self.params_[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
            loss, _ = self._loss_and_grads(x, y)
            if not np.isfinite(loss):
                raise FloatingPointError("MLP training diverged; nonfinite loss")
            self.training_loss_history_.append(round(loss, 8))
            if on_epoch is not None:
                on_epoch(epoch, self.epochs, loss)
        return self

    def predict_proba_positive(self, features):
        if self.params_ is None:
            raise ValueError("Call fit before predict_proba_positive")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Query features must be 2D with fitted feature dimension")
        if not np.isfinite(x).all():
            raise ValueError("Query features must be finite")
        return self._forward(x)[3].ravel()

    def predict(self, features):
        return (self.predict_proba_positive(features) >= 0.5).astype(np.int64)
