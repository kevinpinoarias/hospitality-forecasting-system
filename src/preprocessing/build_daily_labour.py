"""
Builds daily labour totals from split weekly labour files.

Purpose
-------
Reads weekly labour files from data/interim/Labour/, aggregates actual and
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


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "Labour"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "labour"
PER_FILE_DIR = OUTPUT_DIR / "per_file"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PER_FILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PATTERN = "*.csv"

DATE_COL = "Date of business"
METRIC_COL = "Metric"
SUBDIV_COL = "Subdivision"
VALUE_COL = "Value"
DEPT_COL = "Department"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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
    all_days: list[pd.DataFrame] = []
    summary: list[dict] = []

    files = sorted(input_dir.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir}")

    for path in files:
        if path.name.lower().startswith("daily_") or path.name.lower().endswith("summary.csv"):
            continue

        try:
            df = read_file(path)
            daily = build_daily_from_week(df)
            daily["source_file"] = path.name

            per_file_out = PER_FILE_DIR / f"{path.stem}_daily_labour.csv"
            daily.to_csv(per_file_out, index=False)

            all_days.append(daily)

            summary.append(
                {
                    "file": path.name,
                    "days_found": len(daily),
                    "min_date": str(daily["date"].min()) if len(daily) else "",
                    "max_date": str(daily["date"].max()) if len(daily) else "",
                }
            )

            print(f"[OK] {path.name}: {len(daily)} days -> {per_file_out.name}")

        except Exception as exc:
            print(f"[ERROR] {path.name}: {exc}")
            summary.append({"file": path.name, "error": repr(exc)})

    if not all_days:
        raise ValueError("No daily labour datasets were built successfully.")

    master = pd.concat(all_days, ignore_index=True)

    master_out = OUTPUT_DIR / "daily_labour_totals_master.csv"
    master.to_csv(master_out, index=False)
    print(f"\nMaster written to: {master_out}")

    summary_out = OUTPUT_DIR / "daily_labour_build_summary.csv"
    pd.DataFrame(summary).to_csv(summary_out, index=False)
    print(f"Summary written to: {summary_out}")

    return master


if __name__ == "__main__":
    build_daily_labour()