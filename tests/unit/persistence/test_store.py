"""Tests for ArtifactStore ABC."""

import pytest
from abc import ABC

from sklearn_meta.persistence.store import ArtifactStore


class TestArtifactStoreABC:
    """Tests for ArtifactStore abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Verify ArtifactStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ArtifactStore()

    def test_is_abstract(self):
        """Verify ArtifactStore is an ABC."""
        assert issubclass(ArtifactStore, ABC)

    def test_has_required_abstract_methods(self):
        """Verify all expected abstract methods exist."""
        expected_methods = [
            "save_model",
            "load_model",
            "save_fitted_node",
            "save_fitted_graph",
            "list_artifacts",
            "delete_artifact",
        ]
        for method_name in expected_methods:
            assert hasattr(ArtifactStore, method_name), f"Missing method: {method_name}"
