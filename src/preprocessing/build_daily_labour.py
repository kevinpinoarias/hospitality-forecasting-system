"""
Builds daily labour totals from split weekly labour files.

Purpose
-------
Reads weekly labour files from data/interim/labour/, aggregates actual and
forecast labour hours and wages by day, and writes both:
- per-file daily outputs
- a master combined daily labour dataset

Outputs
-------
data/processed/labour/daily_labour_totals_master.csv
data/processed/labour/daily_labour_build_summary.csv
data/processed/labour/per_file/*.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.preprocessing.common import build_daily_dataset, normalise_series


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

from src.config import CONFIG

INPUT_DIR = CONFIG.path("interim_labour_dir")
OUTPUT_DIR = CONFIG.path("processed_labour_dir")
PER_FILE_DIR = OUTPUT_DIR / "per_file"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PER_FILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATE_COL = "Date of business"
METRIC_COL = "Metric"
SUBDIV_COL = "Subdivision"
VALUE_COL = "Value"
DEPT_COL = "Department"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_sum(series: pd.Series) -> float:
    """Safely sum numeric values, coercing invalid entries to zero."""
    return pd.to_numeric(series, errors="coerce").fillna(0).sum()


def build_daily_from_week(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one weekly labour file into daily totals."""
    needed = {DATE_COL, METRIC_COL, SUBDIV_COL, VALUE_COL}
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce").dt.date

    metric = normalise_series(df[METRIC_COL])
    subdiv = normalise_series(df[SUBDIV_COL])

    def aggregate(metric_keyword: str, subdiv_keyword: str) -> pd.Series:
        subset = df[
            metric.str.contains(metric_keyword, na=False, regex=False)
            & subdiv.str.contains(subdiv_keyword, na=False, regex=False)
        ]
        return subset.groupby(DATE_COL, dropna=False)[VALUE_COL].apply(safe_sum)

    total_hours = aggregate("hours", "actual").rename("total_hours")
    total_wages = aggregate("wages", "actual").rename("total_wages")
    forecast_hours = aggregate("hours", "forecast").rename("forecast_hours")
    forecast_wages = aggregate("wages", "forecast").rename("forecast_wages")

    out = pd.concat(
        [total_hours, total_wages, forecast_hours, forecast_wages],
        axis=1
    ).reset_index()

    out = out.rename(columns={DATE_COL: "date"}).sort_values("date").reset_index(drop=True)

    if DEPT_COL in df.columns and len(df):
        out["department"] = str(df[DEPT_COL].iloc[0]).strip()

    cols = [
        "date",
        "department",
        "total_hours",
        "total_wages",
        "forecast_hours",
        "forecast_wages",
    ]
    cols = [c for c in cols if c in out.columns]

    out = out[cols]
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_daily_labour(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    """
    Build daily labour outputs from all weekly labour files.

    Returns
    -------
    pd.DataFrame
        Master daily labour dataset across all files.
    """
    return build_daily_dataset(
        label="labour",
        input_dir=input_dir,
        output_dir=OUTPUT_DIR,
        per_file_dir=PER_FILE_DIR,
        per_file_suffix="_daily_labour.csv",
        master_name="daily_labour_totals_master.csv",
        summary_name="daily_labour_build_summary.csv",
        build_week=build_daily_from_week,
    )


if __name__ == "__main__":
    build_daily_labour()