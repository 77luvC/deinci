from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_SCHEMA_VERSION = 1

# Bump this when a change to encoding/scoring logic (not just inputs) should
# invalidate every existing cache file, e.g. a change to `max_length` or
# normalization in `SciNCLEncoder`.


def slugify(value: str) -> str:
    """Turn an arbitrary identifier (e.g. a HF model name) into a filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def hash_dataframe(df: pd.DataFrame, columns: list[str]) -> str:
    """Return a stable short hash of the given columns of a DataFrame.

    Row order is sorted before hashing so the result only depends on content,
    not on incidental ordering from upstream joins or CSV reads.
    """
    subset = df[columns].fillna("").astype(str).sort_values(columns).reset_index(drop=True)
    payload = subset.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON cache manifest, returning None if missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a JSON cache manifest, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def manifest_matches(manifest: dict[str, Any] | None, expected: dict[str, Any], required_files: list[Path]) -> bool:
    """Return whether a cache manifest is valid for reuse.

    A manifest is valid only when every key in `expected` matches exactly and
    every file it should have produced still exists on disk; a manifest whose
    files were deleted out-of-band must not be trusted.
    """
    if manifest is None:
        return False
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    return all(f.exists() for f in required_files)
