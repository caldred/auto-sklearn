"""Unit tests for InferenceGraph aggregation behavior."""

import numpy as np
import pandas as pd

from sklearn_meta.artifacts.inference import InferenceGraph
from sklearn_meta.spec.graph import GraphSpec
from sklearn_meta.spec.node import NodeSpec


class _FakeProbClassifier:
    def __init__(self, classes_=None, probas=None, preds=None):
        self.classes_ = np.asarray(classes_ if classes_ is not None else [0, 1])
        self._probas = np.asarray(probas) if probas is not None else None
        self._preds = np.asarray(preds) if preds is not None else self.classes_[:1]

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(self._preds)

    def predict_proba(self, X):
        return np.asarray(self._probas)


class _FakeVoteClassifier:
    def __init__(self, classes_=None, preds=None):
        self.classes_ = np.asarray(classes_ if classes_ is not None else ["ham", "spam"])
        self._preds = np.asarray(preds if preds is not None else ["ham"])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(self._preds)


def _make_inference_graph(estimator_class, models):
    graph = GraphSpec()
    graph.add_node(NodeSpec(name="clf", estimator_class=estimator_class))
    return InferenceGraph(
        graph=graph,
        node_models={"clf": models},
        selected_features={"clf": None},
        node_params={"clf": {}},
    )


class TestInferenceGraphClassifierAggregation:
    def test_predict_decodes_labels_from_aligned_probabilities(self):
        inference = _make_inference_graph(
            _FakeProbClassifier,
            [
                _FakeProbClassifier(
                    classes_=[0, 1],
                    probas=[[0.3, 0.7], [0.8, 0.2]],
                    preds=[1, 0],
                ),
                _FakeProbClassifier(
                    classes_=[1, 0],
                    probas=[[0.9, 0.1], [0.1, 0.9]],
                    preds=[1, 0],
                ),
            ],
        )

        X = pd.DataFrame({"x": [1, 2]})
        preds = inference.predict(X)

        np.testing.assert_array_equal(preds, np.array([1, 0]))

    def test_predict_uses_union_of_classes_when_first_fold_is_missing_one(self):
        inference = _make_inference_graph(
            _FakeProbClassifier,
            [
                _FakeProbClassifier(
                    classes_=[0, 1],
                    probas=[[0.55, 0.45]],
                    preds=[1],
                ),
                _FakeProbClassifier(
                    classes_=[0, 1, 2],
                    probas=[[0.05, 0.05, 0.90]],
                    preds=[2],
                ),
            ],
        )

        X = pd.DataFrame({"x": [1]})
        preds = inference.predict(X)

        np.testing.assert_array_equal(preds, np.array([2]))

    def test_predict_uses_deterministic_majority_vote_without_proba(self):
        inference = _make_inference_graph(
            _FakeVoteClassifier,
            [
                _FakeVoteClassifier(classes_=["ham", "spam"], preds=["spam", "ham"]),
                _FakeVoteClassifier(classes_=["ham", "spam"], preds=["ham", "spam"]),
            ],
        )

        X = pd.DataFrame({"x": [1, 2]})
        preds = inference.predict(X)

        np.testing.assert_array_equal(preds, np.array(["ham", "ham"]))
