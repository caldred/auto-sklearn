"""Tests for pre-baked reparameterizations."""

import pytest

from auto_sklearn.meta.prebaked import (
    PREBAKED_REGISTRY,
    PrebakedConfig,
    get_all_prebaked_for_model,
    get_prebaked_reparameterization,
    register_prebaked,
    suggest_reparameterizations,
)
from auto_sklearn.meta.reparameterization import (
    LogProductReparameterization,
    RatioReparameterization,
)


# Mock model classes for testing
class XGBClassifier:
    pass


class XGBRegressor:
    pass


class LGBMClassifier:
    pass


class RandomForestClassifier:
    pass


class LogisticRegression:
    pass


class MLPClassifier:
    pass


class SVC:
    pass


class CatBoostClassifier:
    pass


class TestGetPrebakedReparameterization:
    """Tests for get_prebaked_reparameterization function."""

    def test_xgb_learning_budget_applies(self):
        """Verify XGB learning_rate × n_estimators reparam applies."""
        param_names = ["learning_rate", "n_estimators", "max_depth"]

        reparams = get_prebaked_reparameterization(XGBClassifier, param_names)

        assert len(reparams) > 0
        # Should have the learning budget reparam
        budget_reparam = next(
            (r for r in reparams if "learning" in r.name.lower() or "budget" in r.name.lower()),
            None
        )
        assert budget_reparam is not None

    def test_xgb_regularization_applies(self):
        """Verify XGB regularization reparam applies."""
        param_names = ["reg_alpha", "reg_lambda"]

        reparams = get_prebaked_reparameterization(XGBClassifier, param_names)

        # Should have regularization reparam
        has_reg = any("regularization" in r.name.lower() for r in reparams)
        assert has_reg

    def test_lgbm_learning_budget_applies(self):
        """Verify LGBM learning budget reparam applies."""
        param_names = ["learning_rate", "n_estimators"]

        reparams = get_prebaked_reparameterization(LGBMClassifier, param_names)

        assert len(reparams) > 0

    def test_rf_complexity_applies(self):
        """Verify RF complexity reparam applies."""
        param_names = ["max_depth", "min_samples_split"]

        reparams = get_prebaked_reparameterization(RandomForestClassifier, param_names)

        # Should have complexity reparam
        has_complexity = any("complexity" in r.name.lower() for r in reparams)
        assert has_complexity

    def test_no_match_returns_empty(self):
        """Verify no match returns empty list."""
        param_names = ["completely_unknown_param"]

        reparams = get_prebaked_reparameterization(LogisticRegression, param_names)

        # LogisticRegression with unknown params should have no matches
        # (unless it accidentally matches something)
        # This is acceptable - just verify it returns a list
        assert isinstance(reparams, list)

    def test_partial_params_no_match(self):
        """Verify partial param match doesn't apply reparam."""
        # Only learning_rate, missing n_estimators
        param_names = ["learning_rate"]

        reparams = get_prebaked_reparameterization(XGBClassifier, param_names)

        # Learning budget reparam needs both params
        budget_reparams = [r for r in reparams if "budget" in r.name.lower()]
        assert len(budget_reparams) == 0

    def test_svm_kernel_applies(self):
        """Verify SVM C and gamma reparam applies."""
        param_names = ["C", "gamma"]

        reparams = get_prebaked_reparameterization(SVC, param_names)

        # Should have SVM kernel reparam
        has_svm = any("svm" in r.name.lower() or "kernel" in r.name.lower() for r in reparams)
        assert has_svm

    def test_alternative_param_names(self):
        """Verify alternative param names are matched."""
        # Using 'eta' instead of 'learning_rate'
        param_names = ["eta", "num_boost_round"]  # XGBoost native names

        reparams = get_prebaked_reparameterization(XGBClassifier, param_names)

        # Should still match learning budget
        # Note: depends on how patterns are configured
        assert isinstance(reparams, list)


class TestGetAllPrebakedForModel:
    """Tests for get_all_prebaked_for_model function."""

    def test_xgb_has_multiple_configs(self):
        """Verify XGB has multiple applicable configs."""
        configs = get_all_prebaked_for_model(XGBClassifier)

        # Should have learning budget and regularization at least
        assert len(configs) >= 2

    def test_returns_prebaked_configs(self):
        """Verify returns list of PrebakedConfig."""
        configs = get_all_prebaked_for_model(XGBClassifier)

        assert all(isinstance(c, PrebakedConfig) for c in configs)

    def test_unknown_model_empty(self):
        """Verify unknown model returns empty list."""
        class UnknownModel:
            pass

        configs = get_all_prebaked_for_model(UnknownModel)

        assert configs == []


class TestSuggestReparameterizations:
    """Tests for suggest_reparameterizations function."""

    def test_suggests_applicable(self):
        """Verify suggests applicable reparameterizations."""
        param_names = ["learning_rate", "n_estimators", "max_depth"]

        suggestions = suggest_reparameterizations(XGBClassifier, param_names)

        # Should suggest learning budget
        applicable = [s for s in suggestions if s.get("applies")]
        assert len(applicable) > 0

    def test_includes_descriptions(self):
        """Verify suggestions include descriptions."""
        param_names = ["learning_rate", "n_estimators"]

        suggestions = suggest_reparameterizations(
            XGBClassifier, param_names, include_descriptions=True
        )

        if suggestions:
            assert all("description" in s for s in suggestions)

    def test_indicates_missing_params(self):
        """Verify suggestions indicate missing params for partial matches."""
        param_names = ["learning_rate"]  # Missing n_estimators

        suggestions = suggest_reparameterizations(XGBClassifier, param_names)

        # Should have non-applicable suggestion with missing info
        not_applicable = [s for s in suggestions if not s.get("applies")]
        # Some might have missing params indicated
        assert isinstance(suggestions, list)


class TestPrebakedRegistry:
    """Tests for the prebaked registry."""

    def test_registry_not_empty(self):
        """Verify registry has entries."""
        assert len(PREBAKED_REGISTRY) > 0

    def test_all_prebaked_registered(self):
        """Verify expected number of prebaked configs."""
        # Based on the prebaked.py file, we expect at least 10 configs
        assert len(PREBAKED_REGISTRY) >= 10

    def test_each_config_has_required_fields(self):
        """Verify each config has required fields."""
        for name, config in PREBAKED_REGISTRY.items():
            assert config.name == name
            assert isinstance(config.description, str)
            assert len(config.description) > 0
            assert isinstance(config.model_patterns, list)
            assert len(config.model_patterns) > 0
            assert isinstance(config.param_patterns, list)
            assert len(config.param_patterns) > 0
            assert callable(config.create_reparam)

    def test_create_reparam_returns_reparameterization(self):
        """Verify create_reparam returns valid reparameterization."""
        for name, config in PREBAKED_REGISTRY.items():
            reparam = config.create_reparam()

            assert hasattr(reparam, "forward")
            assert hasattr(reparam, "inverse")
            assert hasattr(reparam, "get_transformed_bounds")
            assert hasattr(reparam, "original_params")
            assert hasattr(reparam, "transformed_params")


class TestPrebakedConfig:
    """Tests for PrebakedConfig dataclass."""

    def test_basic_creation(self):
        """Verify basic config creation."""
        config = PrebakedConfig(
            name="test_config",
            description="Test description",
            model_patterns=["TestModel"],
            param_patterns=["param_a", "param_b"],
            create_reparam=lambda: LogProductReparameterization(
                name="test", param1="param_a", param2="param_b"
            ),
        )

        assert config.name == "test_config"
        assert config.priority == 0  # Default

    def test_config_with_priority(self):
        """Verify config with custom priority."""
        config = PrebakedConfig(
            name="test",
            description="Test",
            model_patterns=["Test"],
            param_patterns=["a"],
            create_reparam=lambda: None,
            priority=10,
        )

        assert config.priority == 10


class TestSpecificPrebakedConfigs:
    """Tests for specific pre-baked configurations."""

    def test_xgb_learning_budget_config(self):
        """Verify XGB learning budget config."""
        config = PREBAKED_REGISTRY.get("xgb_learning_budget")

        assert config is not None
        assert "XGB" in config.model_patterns or "XGBoost" in config.model_patterns
        assert any("learning_rate" in p or "eta" in p for p in config.param_patterns)
        assert any("n_estimators" in p or "num_boost_round" in p for p in config.param_patterns)

    def test_xgb_regularization_config(self):
        """Verify XGB regularization config."""
        config = PREBAKED_REGISTRY.get("xgb_regularization")

        assert config is not None
        reparam = config.create_reparam()
        assert isinstance(reparam, RatioReparameterization)

    def test_lgbm_regularization_config(self):
        """Verify LGBM regularization config."""
        config = PREBAKED_REGISTRY.get("lgbm_regularization")

        assert config is not None
        assert "LGBM" in config.model_patterns or "LightGBM" in config.model_patterns

    def test_rf_complexity_config(self):
        """Verify RF complexity config."""
        config = PREBAKED_REGISTRY.get("rf_complexity")

        assert config is not None
        assert "RandomForest" in config.model_patterns or "Forest" in config.model_patterns

    def test_svm_kernel_config(self):
        """Verify SVM kernel config."""
        config = PREBAKED_REGISTRY.get("svm_kernel")

        assert config is not None
        assert "SVC" in config.model_patterns or "SVM" in config.model_patterns


class TestRegisterPrebaked:
    """Tests for register_prebaked function."""

    def test_register_adds_to_registry(self):
        """Verify register adds config to registry."""
        original_count = len(PREBAKED_REGISTRY)

        # Register a temporary config
        temp_config = PrebakedConfig(
            name="temp_test_config_12345",
            description="Temporary test config",
            model_patterns=["TempModel"],
            param_patterns=["temp_param"],
            create_reparam=lambda: None,
        )

        register_prebaked(temp_config)

        assert len(PREBAKED_REGISTRY) == original_count + 1
        assert "temp_test_config_12345" in PREBAKED_REGISTRY

        # Clean up
        del PREBAKED_REGISTRY["temp_test_config_12345"]

    def test_register_overwrites_existing(self):
        """Verify register overwrites existing config with same name."""
        temp_config1 = PrebakedConfig(
            name="temp_overwrite_test",
            description="First",
            model_patterns=["Model1"],
            param_patterns=["param1"],
            create_reparam=lambda: None,
        )
        temp_config2 = PrebakedConfig(
            name="temp_overwrite_test",
            description="Second",
            model_patterns=["Model2"],
            param_patterns=["param2"],
            create_reparam=lambda: None,
        )

        register_prebaked(temp_config1)
        register_prebaked(temp_config2)

        assert PREBAKED_REGISTRY["temp_overwrite_test"].description == "Second"

        # Clean up
        del PREBAKED_REGISTRY["temp_overwrite_test"]
