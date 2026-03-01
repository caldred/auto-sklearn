"""Shared manifest read/write helpers for save/load serialization."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_FILENAME = "manifest.json"


def to_json_safe(value: Any, path: str = "root") -> Any:
    """Convert nested values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist(), path)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_str = str(key)
            result[key_str] = to_json_safe(item, f"{path}.{key_str}")
        return result
    raise TypeError(
        f"Object at {path} of type {type(value).__name__} is not JSON serializable"
    )


def write_manifest(directory: Path, manifest: dict) -> None:
    """Write manifest.json to a directory.

    Creates the directory (and parents) if it doesn't exist, then writes
    the manifest dict as indented JSON.

    Args:
        directory: Target directory path.
        manifest: Manifest data to serialize.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(directory: Path) -> dict:
    """Read manifest.json from a directory with unified error handling.

    Args:
        directory: Directory containing manifest.json.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If manifest.json is missing.
        ValueError: If manifest.json contains invalid JSON.
    """
    manifest_path = directory / MANIFEST_FILENAME
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{MANIFEST_FILENAME} not found in {directory}. "
            f"Expected a directory created by save()."
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Corrupt {MANIFEST_FILENAME} in {directory}: {e}"
        ) from e
