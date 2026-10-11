"""
Small helpers shared by the preprocessing stages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalise_series(series: pd.Series) -> pd.Series:
    """Normalise strings for consistent matching."""
    return series.astype(str).str.strip().str.lower()


def read_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
