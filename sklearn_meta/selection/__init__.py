"""Feature selection components."""

from sklearn_meta.selection.importance import (
    ImportanceExtractor,
    TreeImportanceExtractor,
    LinearImportanceExtractor,
    PermutationImportanceExtractor,
)
from sklearn_meta.selection.shadow import ShadowFeatureSelector
from sklearn_meta.selection.selector import FeatureSelector

__all__ = [
    "ImportanceExtractor",
    "TreeImportanceExtractor",
    "LinearImportanceExtractor",
    "PermutationImportanceExtractor",
    "ShadowFeatureSelector",
    "FeatureSelector",
]
