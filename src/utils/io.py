from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def ensure_dir(path: str | Path) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = ensure_parent(path)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
        return
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=index)
        return
    raise ValueError(f"Unsupported file format: {path}")
