"""Tests for NodeSpec."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.search.space import SearchSpace


class TestNodeSpecCreation:
    """Tests for NodeSpec creation and validation."""

    def test_basic_creation(self):
        """Verify basic node creation."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        assert node.name == "test"
        assert node.estimator_class == LogisticRegression
        assert node.output_type == OutputType.PREDICTION

    def test_creation_with_search_space(self):
        """Verify node creation with search space."""
        space = SearchSpace()
        space.add_float("C", 0.01, 10.0)

        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            search_space=space,
        )

        assert node.search_space is space
        assert node.has_search_space

    def test_creation_with_fixed_params(self):
        """Verify node creation with fixed params."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 1000, "random_state": 42},
        )

        assert node.fixed_params["max_iter"] == 1000
        assert node.fixed_params["random_state"] == 42

    def test_empty_name_raises(self):
        """Verify empty name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            NodeSpec(name="", estimator_class=LogisticRegression)

    def test_missing_estimator_raises(self):
        """Verify missing estimator raises error."""
        with pytest.raises(ValueError, match="required"):
            NodeSpec(name="test", estimator_class=None)


class TestNodeSpecOutputTypes:
    """Tests for different output types."""

    def test_prediction_output_type(self):
        """Verify prediction output type requires predict method."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            output_type=OutputType.PREDICTION,
        )

        assert node.output_type == OutputType.PREDICTION

    def test_proba_output_type(self):
        """Verify proba output type requires predict_proba method."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
        )

        assert node.output_type == OutputType.PROBA

    def test_transform_output_type(self):
        """Verify transform output type requires transform method."""
        node = NodeSpec(
            name="test",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        )

        assert node.output_type == OutputType.TRANSFORM

    def test_prediction_without_predict_raises(self):
        """Verify prediction output without predict method raises."""
        class NoPredict:
            def fit(self, X, y):
                pass

        with pytest.raises(ValueError, match="predict"):
            NodeSpec(
                name="test",
                estimator_class=NoPredict,
                output_type=OutputType.PREDICTION,
            )

    def test_proba_without_predict_proba_raises(self):
        """Verify proba output without predict_proba method raises."""
        class NoProba:
            def fit(self, X, y):
                pass

            def predict(self, X):
                pass

        with pytest.raises(ValueError, match="predict_proba"):
            NodeSpec(
                name="test",
                estimator_class=NoProba,
                output_type=OutputType.PROBA,
            )

    def test_transform_without_transform_raises(self):
        """Verify transform output without transform method raises."""
        class NoTransform:
            def fit(self, X, y):
                pass

        with pytest.raises(ValueError, match="transform"):
            NodeSpec(
                name="test",
                estimator_class=NoTransform,
                output_type=OutputType.TRANSFORM,
            )


class TestNodeSpecProperties:
    """Tests for NodeSpec properties."""

    def test_has_search_space_true(self):
        """Verify has_search_space is True when space has params."""
        space = SearchSpace()
        space.add_float("C", 0.01, 10.0)

        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            search_space=space,
        )

        assert node.has_search_space is True

    def test_has_search_space_false_none(self):
        """Verify has_search_space is False when space is None."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        assert node.has_search_space is False

    def test_has_search_space_false_empty(self):
        """Verify has_search_space is False when space is empty."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            search_space=SearchSpace(),
        )

        assert node.has_search_space is False

    def test_is_conditional_true(self):
        """Verify is_conditional is True when condition is set."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            condition=lambda ctx: ctx.n_rows > 100,
        )

        assert node.is_conditional is True

    def test_is_conditional_false(self):
        """Verify is_conditional is False when no condition."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        assert node.is_conditional is False


class TestNodeSpecShouldRun:
    """Tests for should_run method."""

    def test_should_run_no_condition(self, data_context):
        """Verify should_run returns True without condition."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        assert node.should_run(data_context) is True

    def test_should_run_condition_true(self, data_context):
        """Verify should_run returns True when condition is met."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            condition=lambda ctx: ctx.n_rows > 100,
        )

        assert node.should_run(data_context) is True

    def test_should_run_condition_false(self, data_context):
        """Verify should_run returns False when condition fails."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            condition=lambda ctx: ctx.n_rows > 10000,
        )

        assert node.should_run(data_context) is False


class TestNodeSpecCreateEstimator:
    """Tests for create_estimator method."""

    def test_create_estimator_no_params(self):
        """Verify estimator creation with no params."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        model = node.create_estimator()

        assert isinstance(model, LogisticRegression)

    def test_create_estimator_with_fixed_params(self):
        """Verify estimator creation with fixed params."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 500},
        )

        model = node.create_estimator()

        assert model.max_iter == 500

    def test_create_estimator_with_params(self):
        """Verify estimator creation with provided params."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        model = node.create_estimator({"C": 0.5, "max_iter": 200})

        assert model.C == 0.5
        assert model.max_iter == 200

    def test_create_estimator_params_override_fixed(self):
        """Verify provided params override fixed params."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 500},
        )

        model = node.create_estimator({"max_iter": 200})

        assert model.max_iter == 200

    def test_create_estimator_merges_params(self):
        """Verify fixed and provided params are merged."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 500, "random_state": 42},
        )

        model = node.create_estimator({"C": 0.5})

        assert model.C == 0.5
        assert model.max_iter == 500
        assert model.random_state == 42


class TestNodeSpecGetOutput:
    """Tests for get_output method."""

    def test_get_output_prediction(self, small_classification_data):
        """Verify get_output returns predictions for PREDICTION type."""
        X, y = small_classification_data

        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            output_type=OutputType.PREDICTION,
            fixed_params={"max_iter": 1000},
        )

        model = node.create_estimator()
        model.fit(X, y)

        output = node.get_output(model, X)

        assert output.shape == (len(X),)
        assert np.all(np.isin(output, [0, 1]))  # Binary predictions

    def test_get_output_proba(self, small_classification_data):
        """Verify get_output returns probabilities for PROBA type."""
        X, y = small_classification_data

        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
            fixed_params={"max_iter": 1000},
        )

        model = node.create_estimator()
        model.fit(X, y)

        output = node.get_output(model, X)

        assert output.shape == (len(X), 2)  # Binary probabilities
        np.testing.assert_array_almost_equal(output.sum(axis=1), np.ones(len(X)))

    def test_get_output_transform(self, small_classification_data):
        """Verify get_output returns transformed features for TRANSFORM type."""
        X, y = small_classification_data

        node = NodeSpec(
            name="test",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        )

        model = node.create_estimator()
        model.fit(X, y)

        output = node.get_output(model, X)

        assert output.shape == X.shape
        # Transformed data should be standardized
        np.testing.assert_array_almost_equal(output.mean(axis=0), np.zeros(X.shape[1]), decimal=1)


class TestNodeSpecEquality:
    """Tests for node equality and hashing."""

    def test_equality_same_name(self):
        """Verify nodes with same name are equal."""
        node1 = NodeSpec(name="test", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test", estimator_class=RandomForestClassifier)

        assert node1 == node2

    def test_equality_different_name(self):
        """Verify nodes with different names are not equal."""
        node1 = NodeSpec(name="test1", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test2", estimator_class=LogisticRegression)

        assert node1 != node2

    def test_hash_same_name(self):
        """Verify nodes with same name have same hash."""
        node1 = NodeSpec(name="test", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test", estimator_class=RandomForestClassifier)

        assert hash(node1) == hash(node2)

    def test_hash_different_name(self):
        """Verify nodes with different names have different hash."""
        node1 = NodeSpec(name="test1", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test2", estimator_class=LogisticRegression)

        assert hash(node1) != hash(node2)

    def test_usable_in_set(self):
        """Verify nodes can be used in sets."""
        node1 = NodeSpec(name="test", estimator_class=LogisticRegression)
        node2 = NodeSpec(name="test", estimator_class=RandomForestClassifier)

        nodes = {node1, node2}

        assert len(nodes) == 1  # Same name = same node


class TestNodeSpecRepr:
    """Tests for node representation."""

    def test_repr(self):
        """Verify repr is informative."""
        node = NodeSpec(
            name="my_model",
            estimator_class=RandomForestClassifier,
        )

        repr_str = repr(node)

        assert "my_model" in repr_str
        assert "RandomForestClassifier" in repr_str


class TestNodeSpecDistillation:
    """Tests for distillation support on NodeSpec."""

    def test_distillation_config_field(self):
        """Verify distillation_config field is set correctly."""
        from sklearn_meta.spec.distillation import DistillationConfig

        class MockXGB:
            def __init__(self, objective=None):
                pass
            def fit(self, X, y):
                pass
            def predict(self, X):
                pass

        config = DistillationConfig(temperature=5.0, alpha=0.3)
        node = NodeSpec(
            name="student",
            estimator_class=MockXGB,
            distillation_config=config,
        )

        assert node.distillation_config is config
        assert node.distillation_config.temperature == 5.0
        assert node.distillation_config.alpha == 0.3

    def test_is_distilled_true(self):
        """Verify is_distilled is True when config is set."""
        from sklearn_meta.spec.distillation import DistillationConfig

        class MockXGB:
            def __init__(self, objective=None):
                pass
            def fit(self, X, y):
                pass
            def predict(self, X):
                pass

        node = NodeSpec(
            name="student",
            estimator_class=MockXGB,
            distillation_config=DistillationConfig(),
        )

        assert node.is_distilled is True

    def test_is_distilled_false(self):
        """Verify is_distilled is False when no config."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
        )

        assert node.is_distilled is False

    def test_distillation_rejects_incompatible_estimator(self):
        """Verify distillation rejects estimators without objective param."""
        from sklearn_meta.spec.distillation import DistillationConfig

        with pytest.raises(ValueError, match="does not support custom objectives"):
            NodeSpec(
                name="test",
                estimator_class=LogisticRegression,
                distillation_config=DistillationConfig(),
            )


class TestOutputTypeConstants:
    """Tests for OutputType constants."""

    def test_output_type_values(self):
        """Verify OutputType values are correct strings."""
        assert OutputType.PREDICTION == "prediction"
        assert OutputType.PROBA == "proba"
        assert OutputType.TRANSFORM == "transform"
