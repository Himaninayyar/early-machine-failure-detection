"""Soft-voting ensemble of already-fitted, already-calibrated classifiers."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SoftVotingEnsemble(ClassifierMixin, BaseEstimator):
    """Weighted average of the members' calibrated failure probabilities.

    Members are fitted (and calibrated) separately; this class only combines them. It is not saved
    to disk: the web app rebuilds it from the loaded members and the weights in meta.json, so one
    corrupted member file can never take the other models down.
    """

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators or {}
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def normalized_weights(self):
        names = list(self.estimators)
        if not names:
            raise ValueError("ensemble has no members")
        w = self.weights or {n: 1.0 for n in names}
        raw = np.array([float(w.get(n, 0.0)) for n in names])
        if raw.sum() <= 0:
            raise ValueError("ensemble weights must sum to a positive number")
        return dict(zip(names, raw / raw.sum()))

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        w = self.normalized_weights()
        p = sum(w[n] * est.predict_proba(X)[:, 1] for n, est in self.estimators.items())
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
