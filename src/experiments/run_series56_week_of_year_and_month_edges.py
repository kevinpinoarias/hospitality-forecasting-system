"""
Series 56 - do the week of the year and the first and last days of the month help the weekly log's models?

The last two calendar candidates. Day of year is already in the model (day_of_year plus its sine and cosine), so week of year is the same fact in
coarser steps; the reason to try it is the one the weekday flags showed: the trees are four splits deep and a ready-made column can save splits.
The month-edge flags are the payday idea: sales around the 1st and the last days of the month might move with salaries and the bar's month-end
trade, and the one-way error screen (run_error_calendar_screen.py) saw an excess at the month end that turned out to be event days. Written four
ways, appended after each piece's own inputs (so the baseline is a prefix of every variant):

  none                today's piece, nothing added
  week_of_year        the ISO week number, 1 to 53 (Monday-to-Sunday weeks; 29-31 December can be week 1 of the next year, and 1-3 January
                      week 52 or 53 of the last, as pandas gives it)
  week_of_year_circle week_of_year_sin and week_of_year_cos, angle 2 pi (week - 1) / 52.18, so week 53 sits next to week 1
  edges_three_days    first_three_days (day 1 to 3) and last_three_days (the last three days of the month)
  edges_one_day       is_month_start (the 1st) and is_month_end (the last day)

Every variant goes into every piece of the weekly log, exactly as in Series 54 and 55 (Series 54's docstring has the pieces, the post and hydro
models, the correction and the baselines; the hydro and Saturday-fallback baselines are Series 54's `none` fits, same seeds and data).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece; in how many folds it is better; the new days
  expected   little or nothing: day of year already carries the season, and Series 54-55 and the error screen found nothing by day or week of month
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version

Run:  python -m src.experiments.run_series56_week_of_year_and_month_edges list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series56"
RESULTS = OUT_DIR / "series56_week_of_year_and_month_edges.jsonl"
VARIANT_COLUMNS = {
    "none": [],
    "week_of_year": ["week_of_year"],
    "week_of_year_circle": ["week_of_year_sin", "week_of_year_cos"],
    "edges_three_days": ["first_three_days", "last_three_days"],
    "edges_one_day": ["is_month_start", "is_month_end"],
}
NEW_COLUMNS = {c for cols in VARIANT_COLUMNS.values() for c in cols}


def piece_features(piece: str, variant: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[piece][0] + VARIANT_COLUMNS[variant]


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    day, length = out[DATE_COL].dt.day, out[DATE_COL].dt.days_in_month
    week = out[DATE_COL].dt.isocalendar().week.astype(int)
    out["week_of_year"] = week
    angle = 2 * np.pi * (week - 1) / 52.18
    out["week_of_year_sin"], out["week_of_year_cos"] = np.sin(angle), np.cos(angle)
    out["first_three_days"] = (day <= 3).astype(int)
    out["last_three_days"] = (length - day <= 2).astype(int)
    out["is_month_start"] = (day == 1).astype(int)
    out["is_month_end"] = (day == length).astype(int)
    return out


def init_worker56() -> None:
    s47.init_worker47()
    clash = NEW_COLUMNS & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANT_COLUMNS:
            if variant == "none":
                continue  # every baseline exists already: Series 46-47 for the post pieces, Series 54's `none` for the other two
            features = piece_features(piece, variant)
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), size):
                    chunk = SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANT_COLUMNS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} baseline {len(piece_features(piece, 'none'))} columns, widest {max(len(piece_features(piece, v)) for v in VARIANT_COLUMNS)}")
        dates = pd.to_datetime(["2025-12-29", "2026-01-01", "2026-01-03", "2026-02-28", "2026-03-01", "2026-03-31", "2026-12-31", "2027-01-01"])
        show = add_columns(pd.DataFrame({DATE_COL: dates}))
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker56, "series56")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
