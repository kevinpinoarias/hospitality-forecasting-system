"""
Small helpers shared by the preprocessing stages.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

EXTS = {".csv", ".xlsx"}


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


def build_file_list(input_dir: Path) -> list[Path]:
    """Return the CSV/Excel files in a directory, skipping Excel lock files
    (~$...) and hidden files such as the anonymise manifest."""
    return sorted(
        [
            p for p in input_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTS
            and not p.name.startswith("~$")
            and not p.name.startswith(".")
        ]
    )


def build_daily_dataset(
    *,
    label: str,
    input_dir: Path,
    output_dir: Path,
    per_file_dir: Path,
    per_file_suffix: str,
    master_name: str,
    summary_name: str,
    build_week: Callable[[pd.DataFrame], pd.DataFrame],
    finish_master: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    The loop shared by the daily sales and daily labour builds: turn every
    weekly file in `input_dir` into daily rows with `build_week`, write one
    per-file CSV each, then write the combined master and a build summary.

    `finish_master` is an optional last step on the combined table (sales
    uses it to fill the known closure days). Returns the master table.
    """
    all_days: list[pd.DataFrame] = []
    summary: list[dict] = []

    files = build_file_list(input_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir}")

    for path in files:
        if path.name.lower().startswith("daily_") or path.name.lower().endswith("summary.csv"):
            continue

        try:
            df = read_file(path)
            daily = build_week(df)
            daily["source_file"] = path.name

            per_file_out = per_file_dir / f"{path.stem}{per_file_suffix}"
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
        raise ValueError(f"No daily {label} datasets were built successfully.")

    master = pd.concat(all_days, ignore_index=True)
    if finish_master is not None:
        master = finish_master(master)

    master_out = output_dir / master_name
    master.to_csv(master_out, index=False)
    print(f"\nMaster written to: {master_out}")

    summary_out = output_dir / summary_name
    pd.DataFrame(summary).to_csv(summary_out, index=False)
    print(f"Summary written to: {summary_out}")

    return master
