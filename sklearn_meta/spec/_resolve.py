"""Shared utility for resolving dotted class paths to Python types."""

from __future__ import annotations

import importlib
from typing import Type


def resolve_class_path(class_path: str) -> Type:
    """Resolve a dotted class path like ``'sklearn.linear_model.LogisticRegression'``.

    Tries progressively shorter module prefixes so that nested classes
    (e.g. ``module.Outer.Inner``) are handled correctly.

    Args:
        class_path: Fully qualified class path.

    Returns:
        The resolved Python type.

    Raises:
        ImportError: If the class cannot be found.
    """
    path_parts = class_path.split(".")
    for split_idx in range(len(path_parts) - 1, 0, -1):
        module_path = ".".join(path_parts[:split_idx])
        attr_parts = path_parts[split_idx:]
        try:
            obj = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name != module_path:
                raise
            continue

        try:
            for attr_name in attr_parts:
                obj = getattr(obj, attr_name)
        except AttributeError:
            continue

        return obj  # type: ignore[return-value]

    raise ImportError(f"Could not resolve class path '{class_path}'")


def get_class_path(cls: Type) -> str:
    """Return the fully qualified dotted path for *cls*."""
    return f"{cls.__module__}.{cls.__qualname__}"
