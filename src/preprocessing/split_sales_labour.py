"""
Splits anonymised raw rota files into separate sales and labour datasets.

Purpose
-------
Reads raw input files from data/raw/, detects the row where labour data begins,
and writes two separate outputs per source file:
- Sales data -> data/interim/Sales/
- Labour data -> data/interim/Labour/

Notes
-----
- This script is designed for the public portfolio version of the project.
- It assumes the raw input files are stored locally.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
SALES_DIR = OUTPUT_DIR / "sales"
LABOUR_DIR = OUTPUT_DIR / "labour"

SALES_DIR.mkdir(parents=True, exist_ok=True)
LABOUR_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

EXTS = {".csv", ".xlsx"}
TARGET_METRIC = "wages"
TARGET_SUBDIV = "forecast"
REQUIRED_COLS = ["Metric", "Subdivision"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalise_series(series: pd.Series) -> pd.Series:
    """Normalise text for reliable matching."""
    return series.astype(str).str.strip().str.lower()


def read_file(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def find_split_index(df: pd.DataFrame) -> int | None:
    """
    Find the first row where labour data begins.

    The split point is defined as the first occurrence where:
    Metric == 'wages' and Subdivision == 'forecast'
    """
    metric = normalise_series(df["Metric"])
    subdiv = normalise_series(df["Subdivision"])

    mask = (metric == TARGET_METRIC) & (subdiv == TARGET_SUBDIV)
    idx = np.where(mask.to_numpy())[0]

    return int(idx[0]) if len(idx) else None


def write_output(df_sales: pd.DataFrame, df_labour: pd.DataFrame, source_path: Path) -> None:
    """Write split sales and labour data to separate folders."""
    stem = source_path.stem

    if source_path.suffix.lower() == ".csv":
        df_sales.to_csv(SALES_DIR / f"{stem}_sales.csv", index=False)
        df_labour.to_csv(LABOUR_DIR / f"{stem}_labour.csv", index=False)
    else:
        df_sales.to_excel(SALES_DIR / f"{stem}_sales.xlsx", index=False)
        df_labour.to_excel(LABOUR_DIR / f"{stem}_labour.xlsx", index=False)


def build_file_list(input_dir: Path) -> list[Path]:
    """Return valid raw input files from the raw data directory."""
    return sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTS
            and not p.name.startswith("~$")
            and p.name.lower() != "split_summary.csv"
        ]
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def split_sales_and_labour(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    """
    Split all raw files in the input directory into sales and labour outputs.

    Returns
    -------
    pd.DataFrame
        Summary of processing results.
    """
    summary: list[dict] = []
    files = build_file_list(input_dir)

    if not files:
        raise FileNotFoundError(f"No supported raw files found in: {input_dir}")

    for path in files:
        try:
            df = read_file(path)

            # Skip files that are not raw rota-style files
            if not all(col in df.columns for col in REQUIRED_COLS):
                print(f"[SKIP] {path.name} (missing required raw columns)")
                summary.append(
                    {
                        "file": path.name,
                        "status": "SKIP_NOT_RAW",
                        "rows_total": len(df),
                    }
                )
                continue

            split_idx = find_split_index(df)

            if split_idx is None:
                print(f"[NO SPLIT] {path.name}")
                summary.append(
                    {
                        "file": path.name,
                        "status": "NO_SPLIT",
                        "rows_total": len(df),
                    }
                )
                continue

            df_sales = df.iloc[:split_idx].copy()
            df_labour = df.iloc[split_idx:].copy()

            write_output(df_sales, df_labour, path)

            print(f"[OK] {path.name} split at row {split_idx}")
            summary.append(
                {
                    "file": path.name,
                    "status": "OK",
                    "rows_total": len(df),
                    "split_idx_0based": split_idx,
                    "sales_rows": len(df_sales),
                    "labour_rows": len(df_labour),
                }
            )

        except Exception as exc:
            print(f"[ERROR] {path.name}: {exc}")
            summary.append(
                {
                    "file": path.name,
                    "status": "ERROR",
                    "error": repr(exc),
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nDone. Summary written to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    split_sales_and_labour()