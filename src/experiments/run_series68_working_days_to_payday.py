"""
Series 68 - working days to and from payday, instead of calendar days.

The payday columns count calendar days. A payday on a Friday and one on a Tuesday are both "payday", but the weekend that follows them sits at different
calendar distances, and a bank holiday after payday shifts the count. Counting only working days (Monday to Friday, not Scottish bank holidays) gives the days a
payday sits "behind" or "ahead" in the working week: the Saturday, the Sunday and the Monday after a Friday payday are all one working day after it.

Both counts use the same payday as the deployed columns (the last working day of the month) and are true distances, 0 on payday itself (the deployed
`days_since_payday` is not; see Series 59 and 62). The columns are ADDED on top of the deployed 38, flaw included, never replacing them.

Variants, each run on its own:

  wd_since   working days from the latest payday to the day (0 on payday, 1 for the next working day and the weekend after a Friday payday)
  wd_to      working days from the day to the next payday on or after it (0 on payday)
  wd_pair    both columns together (two columns more than deployed)

Series 54's design: every variant in every piece of the weekly log (the post model's four pieces, the hydro Monday-to-Thursday piece, the raw
Saturday fallback), 100 seeds (200-299), 585 days and the 56 new days, the two models assembled and corrected, read by judgment on the net effect.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at    for each variant, the change in the post and hydro models' MAE after the correction with a 95% moving-block bootstrap interval; by weekday and
             piece (Friday, Saturday and Sunday above all); folds better; the new days
  expected   little or nothing: the model already knows the weekday and the calendar distance, so it can combine them; Series 59-64 found no payday encoding that
             beat the deployed columns
  would matter  an interval below zero in both models, and the new days not worse
  limits     no pass/fail gate; nothing in production changes; a new column would change the serving code too

Run:  python -m src.experiments.run_series68_working_days_to_payday list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.run_series61_payday_rules import month_payday
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series68"
RESULTS = OUT_DIR / "series68_working_days_to_payday.jsonl"
VARIANTS = ["none", "wd_since", "wd_to", "wd_pair"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = ["wd_since", "wd_to"]
EXTRA = {"wd_since": ["wd_since"], "wd_to": ["wd_to"], "wd_pair": ["wd_since", "wd_to"]}


def piece_features(piece: str, variant: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[piece][0] + EXTRA.get(variant, [])


def add_working_day_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Working days since the latest and to the next last-working-day payday; `busday_count(a, b)` counts the working days in [a, b)."""
    out = frame.copy()
    d = out[DATE_COL].dt.normalize()
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision, years=range(d.min().year - 1, d.max().year + 2))
    bank = np.array(sorted(calendar.keys()), dtype="datetime64[D]")
    periods = pd.period_range(d.min().to_period("M") - 2, d.max().to_period("M") + 2, freq="M")
    paydays = np.sort(np.array([month_payday(p, "last_working_day", calendar) for p in periods], dtype="datetime64[D]"))
    days = d.to_numpy().astype("datetime64[D]")
    latest = paydays[np.searchsorted(paydays, days, side="right") - 1]
    upcoming = paydays[np.searchsorted(paydays, days, side="left")]
    out["wd_since"] = np.busday_count(latest, days, holidays=bank)
    out["wd_to"] = np.busday_count(days, upcoming, holidays=bank)
    return out


def init_worker68() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_working_day_columns(add_hydro_thu(_STATE["df"]))


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
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, " + ", ".join(
                f"{v} {len(piece_features(piece, v))}" for v in VARIANTS[1:]))
        df, _ = load_protocol()
        df = add_working_day_columns(df)
        assert ((df["wd_since"] == 0) == (df["is_payday"] == 1)).all(), "wd_since is not 0 exactly on the deployed paydays"
        # wd_to is 0 on payday, and also on a weekend or bank-holiday day with no working day between it and the payday (a Saturday before a Monday payday)
        assert (df.loc[df["is_payday"] == 1, "wd_to"] == 0).all(), "wd_to is not 0 on the deployed paydays"
        calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision, years=range(2023, 2028))
        extra_zero = df[(df["wd_to"] == 0) & (df["is_payday"] == 0)][DATE_COL]
        assert all(x.dayofweek >= 5 or x in calendar for x in extra_zero), "a working day has wd_to 0 without being payday"
        print(f"wd_to is 0 on {int((df['wd_to'] == 0).sum())} days: {int(df['is_payday'].sum())} paydays and {len(extra_zero)} weekend or bank-holiday days just before a payday")
        assert (df[["wd_since", "wd_to"]] >= 0).all().all()
        print(f"{len(df)} days; wd_since range {int(df['wd_since'].min())} to {int(df['wd_since'].max())}, wd_to range {int(df['wd_to'].min())} to {int(df['wd_to'].max())}")
        fri = df[(df["is_payday"] == 1) & (df[DATE_COL].dt.dayofweek == 4)][DATE_COL].iloc[0]
        show = df[df[DATE_COL].between(fri - pd.Timedelta(days=3), fri + pd.Timedelta(days=4))]
        print(show[[DATE_COL, "is_payday", "days_since_payday", "wd_since", "wd_to"]].assign(weekday=show[DATE_COL].dt.day_name()).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker68, "series68")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
