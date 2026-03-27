"""Tests for NodeSpec."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn_meta.spec.node import NodeSpec, OutputType
from sklearn_meta.search.space import SearchSpace
from sklearn_meta.search.parameter import FloatParameter, IntParameter


class TestNodeSpecCreation:
    """Tests for NodeSpec creation and validation."""

    def test_empty_name_raises(self):
        """Verify empty name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            NodeSpec(name="", estimator_class=LogisticRegression)

    def test_missing_estimator_raises(self):
        """Verify missing estimator raises error."""
        with pytest.raises(ValueError, match="required"):
            NodeSpec(name="test", estimator_class=None)


class TestNodeSpecOutputTypes:
    """Tests for output type validation."""

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

    def test_has_search_space_false_empty(self):
        """Empty SearchSpace should report has_search_space=False (non-obvious edge case)."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            search_space=SearchSpace(),
        )

        assert node.has_search_space is False


class TestNodeSpecShouldRun:
    """Tests for should_run method."""

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


class TestNodeSpecDistillation:
    """Tests for distillation support on NodeSpec."""

    def test_distillation_rejects_incompatible_estimator(self):
        """Verify distillation rejects estimators without objective param."""
        from sklearn_meta.spec.distillation import DistillationConfig

        with pytest.raises(ValueError, match="does not support custom objectives"):
            NodeSpec(
                name="test",
                estimator_class=LogisticRegression,
                distillation_config=DistillationConfig(),
            )


class MockQuantileRegressor:
    """Mock estimator that supports quantile regression parameters."""

    def __init__(self, objective=None, quantile_alpha=0.5, **kwargs):
        self.objective = objective
        self.quantile_alpha = quantile_alpha

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MockDistillableEstimator:
    """Mock estimator that accepts an objective parameter for distillation tests."""

    def __init__(self, objective=None):
        self.objective = objective

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TestNodeSpecSerializationValues:
    """Tests verifying NodeSpec attribute values survive to_dict/from_dict."""

    def test_output_type_proba_preserved(self):
        """Verify PROBA output_type round-trips correctly."""
        node = NodeSpec(
            name="lr_proba",
            estimator_class=LogisticRegression,
            output_type=OutputType.PROBA,
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.output_type == OutputType.PROBA

    def test_output_type_transform_preserved(self):
        """Verify TRANSFORM output_type round-trips correctly."""
        node = NodeSpec(
            name="scaler",
            estimator_class=StandardScaler,
            output_type=OutputType.TRANSFORM,
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.output_type == OutputType.TRANSFORM

    def test_fixed_params_preserved(self):
        """Verify fixed_params are preserved exactly."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fixed_params={"max_iter": 500, "random_state": 42, "C": 0.1},
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.fixed_params == {"max_iter": 500, "random_state": 42, "C": 0.1}

    def test_fit_params_preserved(self):
        """Verify fit_params are preserved exactly."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            fit_params={"sample_weight": "auto", "verbose": True},
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.fit_params == {"sample_weight": "auto", "verbose": True}

    def test_feature_cols_preserved(self):
        """Verify feature_cols list is preserved."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            feature_cols=["age", "income", "score"],
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.feature_cols == ["age", "income", "score"]

    def test_description_preserved(self):
        """Verify description string is preserved."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            description="Baseline logistic regression model",
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.description == "Baseline logistic regression model"

    def test_plugins_preserved(self):
        """Verify plugins list is preserved."""
        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            plugins=["shap_explainer", "calibration"],
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.plugins == ["shap_explainer", "calibration"]

    def test_search_space_parameter_attributes_preserved(self):
        """Verify search space parameter attributes (low, high, log, step) survive."""
        space = SearchSpace()
        space.add_float("C", 0.01, 10.0, log=True)
        space.add_int("max_iter", 100, 1000, step=100)

        node = NodeSpec(
            name="test",
            estimator_class=LogisticRegression,
            search_space=space,
        )

        restored = NodeSpec.from_dict(node.to_dict())

        c_param = restored.search_space.get_parameter("C")
        assert isinstance(c_param, FloatParameter)
        assert c_param.low == 0.01
        assert c_param.high == 10.0
        assert c_param.log is True
        assert c_param.step is None

        iter_param = restored.search_space.get_parameter("max_iter")
        assert isinstance(iter_param, IntParameter)
        assert iter_param.low == 100
        assert iter_param.high == 1000
        assert iter_param.step == 100

    def test_distillation_config_round_trips(self):
        """Verify distillation_config alpha and temperature are preserved."""
        from sklearn_meta.spec.distillation import DistillationConfig

        node = NodeSpec(
            name="student",
            estimator_class=MockDistillableEstimator,
            distillation_config=DistillationConfig(alpha=0.3, temperature=5.0),
        )

        restored = NodeSpec.from_dict(node.to_dict())

        assert restored.distillation_config is not None
        assert restored.distillation_config.alpha == 0.3
        assert restored.distillation_config.temperature == 5.0

    def test_quantile_output_type_dispatches_to_quantile_node_spec(self):
        """Verify NodeSpec.from_dict with output_type='quantiles' returns QuantileNodeSpec."""
        from sklearn_meta.spec.quantile import QuantileNodeSpec

        node = QuantileNodeSpec(
            name="quantile_price",
            property_name="price",
            estimator_class=MockQuantileRegressor,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        data = node.to_dict()
        restored = NodeSpec.from_dict(data)

        assert isinstance(restored, QuantileNodeSpec)
        assert restored.output_type == OutputType.QUANTILES
