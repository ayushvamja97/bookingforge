"""NumPy k-nearest-neighbors classifier for binary cancellation labels.

Neighbor selection and probability estimation are implemented here, not
delegated to scikit-learn's KNeighborsClassifier. Feature preprocessing must
happen separately, using training data only.
"""
from __future__ import annotations

import numpy as np


class NumpyKNNBinary:
    """Unweighted Euclidean kNN; P(positive) = fraction of positive neighbors."""

    def __init__(self, k: int = 15, batch_size: int = 64):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.k = k
        self.batch_size = batch_size
        self._train = None
        self._labels = None
        self._train_squared_norms = None

    def fit(self, features, labels):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("Training features must be a nonempty 2D matrix")
        if y.ndim != 1 or len(y) != x.shape[0] or not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must be a binary 1D vector matching the training rows")
        if not np.isfinite(x).all():
            raise ValueError("Training features must be finite")
        if self.k > x.shape[0]:
            raise ValueError("k cannot exceed the number of training rows")
        self._train = x.copy()
        self._labels = y.astype(np.float64, copy=True)
        self._train_squared_norms = np.einsum("ij,ij->i", self._train, self._train)
        return self

    def predict_proba_positive(self, features, on_progress=None):
        if self._train is None:
            raise ValueError("Call fit before predict_proba_positive")
        queries = np.asarray(features, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != self._train.shape[1]:
            raise ValueError("Query matrix must have the same feature dimension as training")
        if not np.isfinite(queries).all():
            raise ValueError("Query features must be finite")
        results = np.empty(queries.shape[0], dtype=np.float64)
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]
            # ||q-t||^2 = ||q||^2 + ||t||^2 - 2(q dot t).
            # Squared distances preserve neighbor rankings; sqrt is unnecessary.
            dist2 = (
                np.einsum("ij,ij->i", batch, batch)[:, None]
                + self._train_squared_norms[None, :]
                - 2.0 * (batch @ self._train.T)
            )
            np.maximum(dist2, 0.0, out=dist2)
            neighbors = np.argpartition(dist2, kth=self.k - 1, axis=1)[:, :self.k]
            results[start:start + len(batch)] = self._labels[neighbors].mean(axis=1)
            if on_progress is not None and ((start // self.batch_size) % 25 == 0 or start + len(batch) == len(queries)):
                on_progress(start + len(batch), len(queries))
        return results

    def predict(self, features):
        return (self.predict_proba_positive(features) >= 0.5).astype(np.int64)
