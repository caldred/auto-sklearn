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
