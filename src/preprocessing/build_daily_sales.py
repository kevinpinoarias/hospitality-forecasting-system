"""
Builds daily sales totals from split weekly sales files.

Purpose
-------
Reads weekly sales files from data/interim/Sales/, filters to food and beverage
streams, aggregates actual and forecast sales by day, and writes both:
- per-file daily outputs
- a master combined daily sales dataset

Outputs
-------
data/processed/sales/daily_sales_totals_master.csv
data/processed/sales/daily_build_summary.csv
data/processed/sales/per_file/*.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Known closure days
# ---------------------------------------------------------------------

# The venue is confirmed always closed on these dates (month, day) - the raw
# exports simply have no row at all for them, which is indistinguishable
# from "unknown" unless handled explicitly. Since we know the true sales
# figure for a closure day (zero), this fills them in rather than leaving a
# gap that downstream code would otherwise have to guess about or skip.
KNOWN_CLOSURE_MONTH_DAYS = {
    (12, 25),  # Christmas Day
    (1, 1),    # New Year's Day
}


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "Sales"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sales"
PER_FILE_DIR = OUTPUT_DIR / "per_file"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PER_FILE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PATTERN = "*.csv"

DATE_COL = "Date of business"
SUBDIV_COL = "Subdivision"
STREAM_COL = "Stream"
VALUE_COL = "Value"
DEPT_COL = "Department"

INCLUDE_STREAMS = {"food", "beverage"}


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


def build_daily_from_week(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one weekly sales file into daily totals."""
    needed = {DATE_COL, SUBDIV_COL, STREAM_COL, VALUE_COL}
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    df = df.copy()

    # Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce").dt.date

    # Filter to relevant sales streams
    df = df[normalise_series(df[STREAM_COL]).isin(INCLUDE_STREAMS)].copy()

    subdiv = normalise_series(df[SUBDIV_COL])

    forecast_df = df[subdiv.eq("forecast")].copy()
    actual_df = df[subdiv.eq("actual")].copy()

    forecast_daily = (
        forecast_df.groupby(DATE_COL, dropna=False)[VALUE_COL]
        .sum()
        .rename("forecast_sales")
    )

    actual_daily = (
        actual_df.groupby(DATE_COL, dropna=False)[VALUE_COL]
        .sum()
        .rename("total_sales")
    )

    out = pd.concat([actual_daily, forecast_daily], axis=1).reset_index()
    out = out.rename(columns={DATE_COL: "date"})

    # Add department if present
    if DEPT_COL in df.columns and len(df):
        out["department"] = str(df[DEPT_COL].iloc[0]).strip()

    cols = ["date", "department", "total_sales", "forecast_sales"]
    cols = [c for c in cols if c in out.columns]

    out = out[cols].sort_values("date").reset_index(drop=True)
    return out


def fill_known_closure_days(master: pd.DataFrame) -> pd.DataFrame:
    """
    Insert an explicit zero-sales/zero-forecast row for each known closure
    date (Christmas Day, New Year's Day) falling inside the dataset's date
    range, if it isn't already present. This does not guess a value - it
    records a fact (the venue was closed, so sales were zero) rather than
    leaving a silent gap that looks identical to "unknown."
    """
    out = master.copy()
    out["date"] = pd.to_datetime(out["date"])

    existing_dates = set(out["date"].dt.date)
    min_year, max_year = out["date"].dt.year.min(), out["date"].dt.year.max()

    department = out["department"].mode().iloc[0] if "department" in out.columns and len(out) else None

    new_rows = []
    for year in range(min_year, max_year + 1):
        for month, day in KNOWN_CLOSURE_MONTH_DAYS:
            closure_date = pd.Timestamp(year=year, month=month, day=day)
            if not (out["date"].min() <= closure_date <= out["date"].max()):
                continue
            if closure_date.date() in existing_dates:
                continue

            row = {
                "date": closure_date,
                "total_sales": 0.0,
                "forecast_sales": 0.0,
                "source_file": "synthetic_known_closure_day",
            }
            if department is not None:
                row["department"] = department
            new_rows.append(row)

    if not new_rows:
        return out

    out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)

    print(f"Filled {len(new_rows)} known closure day(s) with zero sales:")
    for row in new_rows:
        print(f"  {row['date'].date()}")

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_daily_sales(input_dir: Path = INPUT_DIR) -> pd.DataFrame:
    """
    Build daily sales outputs from all weekly sales files.

    Returns
    -------
    pd.DataFrame
        Master daily sales dataset across all files.
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

            per_file_out = PER_FILE_DIR / f"{path.stem}_daily.csv"
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
        raise ValueError("No daily sales datasets were built successfully.")

    master = pd.concat(all_days, ignore_index=True)
    master = fill_known_closure_days(master)

    master_out = OUTPUT_DIR / "daily_sales_totals_master.csv"
    master.to_csv(master_out, index=False)
    print(f"\nMaster written to: {master_out}")

    summary_out = OUTPUT_DIR / "daily_build_summary.csv"
    pd.DataFrame(summary).to_csv(summary_out, index=False)
    print(f"Summary written to: {summary_out}")

    return master


if __name__ == "__main__":
    build_daily_sales()