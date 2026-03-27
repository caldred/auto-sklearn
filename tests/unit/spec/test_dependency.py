"""Tests for DependencyEdge and DependencyType."""

import pytest

from sklearn_meta.spec.dependency import (
    ConditionalSampleConfig,
    DependencyEdge,
    DependencyType,
)


class TestDependencyEdgeCreation:
    """Tests for DependencyEdge creation and validation."""

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

    def test_feature_name_distill_default(self):
        """Verify feature_name for distill type."""
        edge = DependencyEdge(
            source="teacher",
            target="student",
            dep_type=DependencyType.DISTILL,
        )

        assert edge.feature_name == "distill_teacher"


class TestDependencyEdgeBlocksTraining:
    """Tests for fit-time dependency semantics."""

    @pytest.mark.parametrize(
        ("dep_type", "expected"),
        [
            (DependencyType.PREDICTION, True),
            (DependencyType.PROBA, True),
            (DependencyType.TRANSFORM, True),
            (DependencyType.FEATURE, True),
            (DependencyType.BASE_MARGIN, True),
            (DependencyType.DISTILL, True),
        ],
    )
    def test_non_conditional_edges_block_training(self, dep_type, expected):
        edge = DependencyEdge(source="A", target="B", dep_type=dep_type)
        assert edge.blocks_training() is expected

    def test_conditional_sample_with_actual_targets_does_not_block_training(self):
        edge = DependencyEdge(
            source="A",
            target="B",
            dep_type=DependencyType.CONDITIONAL_SAMPLE,
            conditional_config=ConditionalSampleConfig(
                property_name="price",
                use_actual_during_training=True,
            ),
        )
        assert edge.blocks_training() is False

    def test_conditional_sample_without_actual_targets_blocks_training(self):
        edge = DependencyEdge(
            source="A",
            target="B",
            dep_type=DependencyType.CONDITIONAL_SAMPLE,
            conditional_config=ConditionalSampleConfig(
                property_name="price",
                use_actual_during_training=False,
            ),
        )
        assert edge.blocks_training() is True


class TestDependencyEdgeSerialization:
    """Tests verifying DependencyEdge attribute values survive to_dict/from_dict."""

    def test_basic_round_trip(self):
        """Verify source, target, and dep_type are preserved."""
        edge = DependencyEdge(
            source="model_a",
            target="model_b",
            dep_type=DependencyType.PREDICTION,
        )

        restored = DependencyEdge.from_dict(edge.to_dict())

        assert restored.source == "model_a"
        assert restored.target == "model_b"
        assert restored.dep_type == DependencyType.PREDICTION

    def test_round_trip_with_column_name(self):
        """Verify custom column_name is preserved."""
        edge = DependencyEdge(
            source="encoder",
            target="model",
            dep_type=DependencyType.FEATURE,
            column_name="encoded_category",
        )

        restored = DependencyEdge.from_dict(edge.to_dict())

        assert restored.column_name == "encoded_category"

    def test_round_trip_with_conditional_config(self):
        """Verify ConditionalSampleConfig attributes are preserved."""
        edge = DependencyEdge(
            source="price_q",
            target="volume_q",
            dep_type=DependencyType.CONDITIONAL_SAMPLE,
            conditional_config=ConditionalSampleConfig(
                property_name="price",
                use_actual_during_training=False,
            ),
        )

        restored = DependencyEdge.from_dict(edge.to_dict())

        assert restored.conditional_config is not None
        assert restored.conditional_config.property_name == "price"
        assert restored.conditional_config.use_actual_during_training is False

    @pytest.mark.parametrize("dep_type", list(DependencyType))
    def test_round_trip_each_dependency_type(self, dep_type):
        """Verify dep_type enum value survives for every DependencyType."""
        kwargs = {"source": "src", "target": "tgt", "dep_type": dep_type}
        if dep_type == DependencyType.CONDITIONAL_SAMPLE:
            kwargs["conditional_config"] = ConditionalSampleConfig(
                property_name="prop",
            )

        edge = DependencyEdge(**kwargs)
        restored = DependencyEdge.from_dict(edge.to_dict())

        assert restored.dep_type == dep_type
        assert restored.dep_type.value == dep_type.value

    def test_from_dict_restores_conditional_config_none(self):
        """Verify conditional_config is None when absent from dict."""
        data = {
            "source": "A",
            "target": "B",
            "dep_type": "prediction",
            "column_name": None,
        }

        restored = DependencyEdge.from_dict(data)

        assert restored.conditional_config is None


class TestDependencyEdgeConditionalSampleValidation:
    """Tests for CONDITIONAL_SAMPLE validation."""

    def test_conditional_sample_without_config_raises(self):
        """CONDITIONAL_SAMPLE without conditional_config raises ValueError."""
        with pytest.raises(ValueError, match="conditional_config is required"):
            DependencyEdge(
                source="A",
                target="B",
                dep_type=DependencyType.CONDITIONAL_SAMPLE,
            )
