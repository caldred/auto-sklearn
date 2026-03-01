"""Tests for top-level API helpers."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from sklearn_meta.api import GraphBuilder
from sklearn_meta.core.data.context import DataContext


class TestGraphBuilderFit:
    """Tests for GraphBuilder fit entrypoints."""

    def test_fit_delegates_to_fit_context(self, monkeypatch):
        """fit() should construct a DataContext and delegate to fit_context()."""
        X = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]})
        y = pd.Series([0, 1], name="target")
        groups = pd.Series([10, 11], name="game_id")

        builder = GraphBuilder("test")
        search_backend = object()
        executor = object()
        captured = {}
        sentinel = object()

        def fake_fit_context(self, ctx, search_backend=None, executor=None):
            captured["ctx"] = ctx
            captured["search_backend"] = search_backend
            captured["executor"] = executor
            return sentinel

        monkeypatch.setattr(GraphBuilder, "fit_context", fake_fit_context)

        result = builder.fit(
            X,
            y,
            groups=groups,
            search_backend=search_backend,
            executor=executor,
        )

        assert result is sentinel
        assert isinstance(captured["ctx"], DataContext)
        assert captured["ctx"].feature_cols == ("f0", "f1")
        assert captured["ctx"].target_col == "target"
        assert captured["ctx"].group_col == "game_id"
        assert captured["search_backend"] is search_backend
        assert captured["executor"] is executor

    def test_fit_context_uses_declared_feature_columns(self, mock_search_backend):
        """fit_context() should ignore identity columns that are not features."""
        X, y = make_classification(
            n_samples=60,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        feature_cols = [f"f{i}" for i in range(4)]
        feature_df = pd.DataFrame(X, columns=feature_cols)

        full_df = feature_df.copy()
        full_df["event_id"] = np.arange(len(full_df))
        full_df["batter_id"] = 1000 + np.arange(len(full_df))
        full_df["target"] = y

        ctx = DataContext(
            df=full_df,
            feature_cols=tuple(feature_cols),
            target_col="target",
        )

        fitted = (
            GraphBuilder("test")
            .add_model("rf", RandomForestClassifier)
            .with_fixed_params(n_estimators=10, random_state=42)
            .with_cv(n_splits=3, strategy="stratified", random_state=42)
            .with_tuning(strategy="none", metric="accuracy", greater_is_better=True)
            .fit_context(ctx, search_backend=mock_search_backend)
        )

        node = fitted.get_node("rf")
        assert fitted.metadata is not None
        assert fitted.metadata.data_shape == (len(full_df), len(feature_cols))
        assert fitted.metadata.feature_names == feature_cols
        assert node.models[0].n_features_in_ == len(feature_cols)
