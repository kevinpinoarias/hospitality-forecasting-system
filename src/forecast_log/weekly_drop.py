"""
The weekly drop: take the newest Rotaready export(s), anonymise them with the existing pipeline and rebuild the engineered features, so the next runs know the latest sales.

  python -m src.forecast_log.weekly_drop --incoming "D:\\path\\to\\folder\\with\\the\\new\\csv"

Only the anonymised daily data is used afterwards; the raw export stays in data/raw_private (local, never committed). The step refuses to continue when the new files leave a gap in the days
(the forecasts' history features need unbroken days) or when the rebuilt features do not extend the data. It never touches the models: a version is fitted once and not refitted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.forecast_log.run_forecasts import FEATURES
from src.models.final_model import DATE_COL, TARGET_COL

RAW_PRIVATE = PROJECT_ROOT / "data" / "raw_private"
PIPELINE = [
    "from src.ingestion.anonymize_raw import anonymize_raw; anonymize_raw()",
    "from src.preprocessing.split_sales_labour import split_sales_and_labour; split_sales_and_labour()",
    "from src.preprocessing.build_daily_sales import build_daily_sales; build_daily_sales()",
    "from src.preprocessing.build_daily_labour import build_daily_labour; build_daily_labour()",
    "from src.features.build_features import build_feature_dataset; build_feature_dataset()",
]


def last_sales_day(path: Path = FEATURES) -> pd.Timestamp | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=[DATE_COL]).dropna(subset=[TARGET_COL])
    return frame[DATE_COL].max() if len(frame) else None


def find_gaps(path: Path = FEATURES) -> list[pd.Timestamp]:
    """Days missing between the first and last day of the engineered features."""
    frame = pd.read_csv(path, parse_dates=[DATE_COL])
    full = pd.date_range(frame[DATE_COL].min(), frame[DATE_COL].max(), freq="D")
    return list(full.difference(pd.DatetimeIndex(frame[DATE_COL])))


def copy_new_exports(incoming: Path, destination: Path = RAW_PRIVATE) -> list[str]:
    """Copy exports not already in the private folder (by file name); returns the names copied."""
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(incoming.glob("Rotaready-CostControlRawData*.csv")):
        if not (destination / f.name).exists():
            shutil.copy2(f, destination / f.name)
            copied.append(f.name)
    return copied


def run_pipeline() -> None:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "WANDB_MODE": "disabled"}
    for code in PIPELINE:
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"pipeline step failed: {code.split(';')[0]}\n{done.stderr[-800:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", type=Path, required=True, help="folder holding the new Rotaready-CostControlRawData*.csv exports")
    args = parser.parse_args()
    before = last_sales_day()
    copied = copy_new_exports(args.incoming)
    if not copied:
        print("[drop] no new export files in the incoming folder (nothing copied)")
        return 0
    print(f"[drop] copied {len(copied)} new file(s): {', '.join(copied)}")
    run_pipeline()
    after = last_sales_day()
    gaps = find_gaps()
    print(f"[drop] sales now run through {after.date() if after is not None else None} (before: {before.date() if before is not None else None})")
    if after is None or (before is not None and after <= before):
        print("[drop] WARNING: the data does not extend past the previous last day")
        return 1
    if gaps:
        print(f"[drop] WARNING: {len(gaps)} missing day(s) in the data, first {gaps[0].date()}: the next forecasts would use broken history; fix before running")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
