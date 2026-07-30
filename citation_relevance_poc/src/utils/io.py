from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if needed and return it as a Path.

    Assumes `path` points to a directory rather than a file. Existing
    directories are accepted, which makes this safe to call before every write.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table into a DataFrame.

    The suffix determines the parser; unsupported extensions fail explicitly so
    pipeline stages do not silently read a file with the wrong format.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as CSV or Parquet, creating parent directories first.

    The table is written without the pandas index because downstream pipeline
    stages expect all identifiers to be normal columns. Unsupported extensions
    raise a ValueError rather than guessing a serialization format.
    """
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
