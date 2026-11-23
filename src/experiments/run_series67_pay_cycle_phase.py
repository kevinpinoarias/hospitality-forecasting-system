"""
Series 67 - a pay-cycle phase: does the model gain from knowing which week of a two- or four-week cycle it is?

Some people are paid every two weeks or every four weeks. If a venue's guests are paid on those cycles, sales would rise on alternate (or every
fourth) weekends, and no monthly calendar column can see it. The anchor week (which week is a pay week) is unknown, so the model is given the week number
counted from a fixed date, folded into 2 or 4 phases, and left to find which phase pays.

A free look at the deployed post model's errors came first (no fits; 585 backtest days, errors after the correction, weeks counted from Monday 1 January 2024),
with a permutation test on the spread between phases:
  parity (2 phases)   Friday  +288 / −434 (p = 0.017);  Saturday +134 / +114 (p = 0.95);  Sunday −77 / −40 (p = 0.91);  Monday-Thursday p = 0.22
  mod 4               Friday  p = 0.081;  Saturday p = 0.34;  Sunday p = 0.20;  Friday-Sunday together p = 0.060
One test in ten is below 0.05, which chance alone gives about half the time per ten tests; and a fortnightly pay cycle would show on the Saturday more than on the Friday, which it
does not. So the screen is weak and does not fit the story. The model test is run anyway, as one cheap series, because the screen is the only one of the free checks that showed anything.

Variants, each ONE column added on top of the piece's inputs and run on its own:

  week_parity   the number of whole weeks since Monday 1 January 2024, modulo 2 (0 or 1)
  week_mod4     the same, modulo 4 (0 to 3)

Series 54's design: every variant in every piece of the weekly log (the post model's four pieces, the hydro Monday-to-Thursday piece, the raw
Saturday fallback), 100 seeds (200-299), 585 days and the 56 new days, the two models assembled and corrected, read by judgment on the net effect.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at    for each variant, the change in the post and hydro models' MAE after the correction with a 95% moving-block bootstrap interval; by weekday and
             piece (a pay-cycle effect should show on Friday and Saturday together); folds better; the new days
  expected   nothing: the screen does not fit a pay-cycle story. A gain on Friday alone, with Saturday and Sunday flat, would point to something other than pay
  would matter  an interval below zero in both models, and the new days not worse. The phase is a function of the date only, so a real effect would hold in the live forecasts
  limits     no pass/fail gate; nothing in production changes; with 84 weeks in the backtest the trees could use the phase to memorise individual weeks

Run:  python -m src.experiments.run_series67_pay_cycle_phase list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series67"
RESULTS = OUT_DIR / "series67_pay_cycle_phase.jsonl"
VARIANTS = ["none", "week_parity", "week_mod4"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = VARIANTS[1:]
EPOCH = pd.Timestamp("2024-01-01")  # a Monday


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return cols + ([variant] if variant != "none" else [])


def add_phase_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    week = (out[DATE_COL].dt.normalize() - EPOCH).dt.days // 7
    out["week_parity"], out["week_mod4"] = week % 2, week % 4
    return out


def init_worker67() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_phase_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANTS:
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
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, each variant {len(piece_features(piece, VARIANTS[1]))}")
        df, _ = load_protocol()
        df = add_phase_columns(df)
        assert (df.loc[df[DATE_COL] == EPOCH, "week_parity"] == 0).all() and EPOCH.dayofweek == 0
        # the phase must be constant within a Monday-to-Sunday week and step by one each week
        week = (df[DATE_COL].dt.normalize() - EPOCH).dt.days // 7
        assert (df.groupby(week)["week_mod4"].nunique() == 1).all(), "the phase changes inside a week"
        print(f"{len(df)} days; week_parity counts {df['week_parity'].value_counts().sort_index().to_dict()}; week_mod4 counts {df['week_mod4'].value_counts().sort_index().to_dict()}")
        print(df.loc[df[DATE_COL].between("2026-10-26", "2026-11-09"), [DATE_COL, "week_parity", "week_mod4"]].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker67, "series67")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
