"""Tests for DependencyEdge and DependencyType."""

import pytest

from auto_sklearn.core.model.dependency import DependencyEdge, DependencyType


class TestDependencyType:
    """Tests for DependencyType enum."""

    def test_all_types_defined(self):
        """Verify all expected dependency types exist."""
        expected_types = [
            DependencyType.PREDICTION,
            DependencyType.TRANSFORM,
            DependencyType.FEATURE,
            DependencyType.PROBA,
            DependencyType.BASE_MARGIN,
        ]

        assert len(DependencyType) == 5
        for dep_type in expected_types:
            assert dep_type in DependencyType

    def test_type_values(self):
        """Verify dependency type string values."""
        assert DependencyType.PREDICTION.value == "prediction"
        assert DependencyType.TRANSFORM.value == "transform"
        assert DependencyType.FEATURE.value == "feature"
        assert DependencyType.PROBA.value == "proba"
        assert DependencyType.BASE_MARGIN.value == "base_margin"


class TestDependencyEdgeCreation:
    """Tests for DependencyEdge creation."""

    def test_basic_creation(self):
        """Verify basic edge creation."""
        edge = DependencyEdge(source="A", target="B")

        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.dep_type == DependencyType.PREDICTION  # Default

    def test_creation_with_type(self):
        """Verify edge creation with explicit type."""
        edge = DependencyEdge(
            source="A",
            target="B",
            dep_type=DependencyType.PROBA,
        )

        assert edge.dep_type == DependencyType.PROBA

    def test_creation_with_column_name(self):
        """Verify edge creation with column name."""
        edge = DependencyEdge(
            source="A",
            target="B",
            column_name="custom_feature",
        )

        assert edge.column_name == "custom_feature"

    def test_empty_source_raises(self):
        """Verify empty source raises error."""
        with pytest.raises(ValueError, match="Source.*cannot be empty"):
            DependencyEdge(source="", target="B")

    def test_empty_target_raises(self):
        """Verify empty target raises error."""
        with pytest.raises(ValueError, match="Target.*cannot be empty"):
            DependencyEdge(source="A", target="")

    def test_self_loop_raises(self):
        """Verify self-loop raises error."""
        with pytest.raises(ValueError, match="Self-loops"):
            DependencyEdge(source="A", target="A")

    def test_type_from_string(self):
        """Verify dep_type can be set from string."""
        edge = DependencyEdge(
            source="A",
            target="B",
            dep_type="proba",
        )

        assert edge.dep_type == DependencyType.PROBA


class TestDependencyEdgeFeatureName:
    """Tests for feature_name property."""

    def test_feature_name_with_custom_column(self):
        """Verify feature_name returns custom column name."""
        edge = DependencyEdge(
            source="A",
            target="B",
            column_name="custom_name",
        )

        assert edge.feature_name == "custom_name"

    def test_feature_name_prediction_default(self):
        """Verify feature_name for prediction type."""
        edge = DependencyEdge(
            source="model_1",
            target="meta",
            dep_type=DependencyType.PREDICTION,
        )

        assert edge.feature_name == "pred_model_1"

    def test_feature_name_proba_default(self):
        """Verify feature_name for proba type."""
        edge = DependencyEdge(
            source="model_1",
            target="meta",
            dep_type=DependencyType.PROBA,
        )

        assert edge.feature_name == "proba_model_1"

    def test_feature_name_transform_default(self):
        """Verify feature_name for transform type."""
        edge = DependencyEdge(
            source="scaler",
            target="model",
            dep_type=DependencyType.TRANSFORM,
        )

        assert edge.feature_name == "trans_scaler"

    def test_feature_name_feature_default(self):
        """Verify feature_name for feature type."""
        edge = DependencyEdge(
            source="encoder",
            target="model",
            dep_type=DependencyType.FEATURE,
        )

        assert edge.feature_name == "feat_encoder"

    def test_feature_name_base_margin_default(self):
        """Verify feature_name for base_margin type."""
        edge = DependencyEdge(
            source="base_model",
            target="xgb",
            dep_type=DependencyType.BASE_MARGIN,
        )

        assert edge.feature_name == "margin_base_model"


class TestDependencyEdgeEquality:
    """Tests for edge equality and hashing."""

    def test_equality_same_edge(self):
        """Verify edges with same attributes are equal."""
        edge1 = DependencyEdge(
            source="A",
            target="B",
            dep_type=DependencyType.PREDICTION,
        )
        edge2 = DependencyEdge(
            source="A",
            target="B",
            dep_type=DependencyType.PREDICTION,
        )

        assert edge1 == edge2

    def test_inequality_different_source(self):
        """Verify edges with different sources are not equal."""
        edge1 = DependencyEdge(source="A", target="B")
        edge2 = DependencyEdge(source="C", target="B")

        assert edge1 != edge2

    def test_inequality_different_target(self):
        """Verify edges with different targets are not equal."""
        edge1 = DependencyEdge(source="A", target="B")
        edge2 = DependencyEdge(source="A", target="C")

        assert edge1 != edge2

    def test_inequality_different_type(self):
        """Verify edges with different types are not equal."""
        edge1 = DependencyEdge(source="A", target="B", dep_type=DependencyType.PREDICTION)
        edge2 = DependencyEdge(source="A", target="B", dep_type=DependencyType.PROBA)

        assert edge1 != edge2

    def test_hash_same_edge(self):
        """Verify edges with same attributes have same hash."""
        edge1 = DependencyEdge(source="A", target="B", dep_type=DependencyType.PREDICTION)
        edge2 = DependencyEdge(source="A", target="B", dep_type=DependencyType.PREDICTION)

        assert hash(edge1) == hash(edge2)

    def test_usable_in_set(self):
        """Verify edges can be used in sets."""
        edge1 = DependencyEdge(source="A", target="B")
        edge2 = DependencyEdge(source="A", target="B")
        edge3 = DependencyEdge(source="A", target="C")

        edges = {edge1, edge2, edge3}

        assert len(edges) == 2  # edge1 and edge2 are same


class TestDependencyEdgeRepr:
    """Tests for edge representation."""

    def test_repr(self):
        """Verify repr is informative."""
        edge = DependencyEdge(
            source="model_a",
            target="model_b",
            dep_type=DependencyType.PROBA,
        )

        repr_str = repr(edge)

        assert "model_a" in repr_str
        assert "model_b" in repr_str
        assert "proba" in repr_str


class TestDependencyTypeUseCases:
    """Tests for different dependency use cases."""

    def test_prediction_dependency_stacking(self):
        """Verify PREDICTION dependency for classic stacking."""
        # Base model predictions become features for meta-learner
        edge = DependencyEdge(
            source="rf_base",
            target="lr_meta",
            dep_type=DependencyType.PREDICTION,
        )

        assert edge.dep_type == DependencyType.PREDICTION
        assert edge.feature_name == "pred_rf_base"

    def test_proba_dependency_stacking(self):
        """Verify PROBA dependency for probability stacking."""
        # Base model probabilities become features for meta-learner
        edge = DependencyEdge(
            source="rf_base",
            target="lr_meta",
            dep_type=DependencyType.PROBA,
        )

        assert edge.dep_type == DependencyType.PROBA
        assert edge.feature_name == "proba_rf_base"

    def test_transform_dependency_pipeline(self):
        """Verify TRANSFORM dependency for pipeline flow."""
        # Preprocessor output flows to model
        edge = DependencyEdge(
            source="scaler",
            target="svm",
            dep_type=DependencyType.TRANSFORM,
        )

        assert edge.dep_type == DependencyType.TRANSFORM
        assert edge.feature_name == "trans_scaler"

    def test_base_margin_xgboost(self):
        """Verify BASE_MARGIN for XGBoost stacking."""
        # Base model predictions used as XGBoost base_margin
        edge = DependencyEdge(
            source="lr_init",
            target="xgb",
            dep_type=DependencyType.BASE_MARGIN,
        )

        assert edge.dep_type == DependencyType.BASE_MARGIN
        assert edge.feature_name == "margin_lr_init"
