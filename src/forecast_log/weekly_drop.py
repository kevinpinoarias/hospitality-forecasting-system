"""
The weekly drop: anonymise the newest Rotaready export(s) with the existing pipeline and rebuild the engineered features, so the next runs know the latest sales.

  python -m src.forecast_log.weekly_drop                                                   exports already placed in data/raw_private
  python -m src.forecast_log.weekly_drop --incoming "D:\\path\\to\\folder\\with\\the\\new\\csv"   copy new exports from another folder first

The pipeline runs when data/raw_private holds an export the anonymiser has not processed yet (new, or changed since: its manifest keys on file name and content hash).
Only the anonymised data (data/raw) is used afterwards; the raw export stays in data/raw_private (local, never committed). The step refuses to continue when the new files leave a gap in the days
(the forecasts' history features need unbroken days) or when the rebuilt features do not extend the data. It never touches the models: a version is fitted once and not refitted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.forecast_log.run_forecasts import FEATURES
from src.ingestion.anonymize_raw import build_file_list, load_manifest
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


def pending_exports(folder: Path = RAW_PRIVATE) -> list[str]:
    """Exports in the private folder that the anonymiser has not processed yet: not in its manifest, or with different content since."""
    if not folder.exists():
        return []
    manifest = load_manifest(folder / ".anonymize_manifest.json")
    return [p.name for p in build_file_list(folder) if manifest.get(p.name, {}).get("sha256") != hashlib.sha256(p.read_bytes()).hexdigest()]


def run_pipeline() -> None:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "WANDB_MODE": "disabled"}
    for code in PIPELINE:
        done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"pipeline step failed: {code.split(';')[0]}\n{done.stderr[-800:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", type=Path, help="optional: a folder holding new Rotaready-CostControlRawData*.csv exports to copy into data/raw_private first")
    args = parser.parse_args()
    before = last_sales_day()
    if args.incoming:
        copied = copy_new_exports(args.incoming)
        print(f"[drop] copied {len(copied)} new file(s) from {args.incoming}" + (f": {', '.join(copied)}" if copied else ""))
    pending = pending_exports()
    if not pending:
        print("[drop] no unprocessed export in data/raw_private (nothing to do)")
        return 0
    print(f"[drop] {len(pending)} export(s) not processed yet: {', '.join(pending)}")
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
