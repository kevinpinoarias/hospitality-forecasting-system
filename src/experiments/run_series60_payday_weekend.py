"""
Series 60 - make the "weekend after an early payday" signal explicit, in the deployed system and next to the repaired columns.

Series 59 found that repairing the four payday columns made the model WORSE (post model after the correction +4.87 pounds a day, interval 0.71 to
9.63), and that dropping them was worse still (+7.81). The repair changes the columns on only about 50 of 1,006 days, 17 of them the days AFTER an
early payday inside the same month (the weekend or bank-holiday days at month end). On those days the deployed columns read nonsense (the stopwatch
`days_since_payday` 29-31 and the countdown `days_to_payday` -1, -2) that no ordinary day produces, so it may be working as an accidental flag
for "the weekend after an early payday". On the deployed model's backtest errors these are the hardest days: 9 of the 585 days, mean absolute error
1,546 against 845 for the rest, Saturdays under-predicted (+1,292 to +2,789 on three of six) and Sundays over-predicted (about -3,000 on two of three).

Series 60 tests that reading by writing the signal down explicitly. Variants 1-3 keep the deployed columns exactly as they are (flaw included) and
add one column each:

  after_payday_flag    1 on a day after this month's payday, inside the same month (the days the flaw touches); 0 otherwise
  after_payday_count   the same days counted 1, 2, 3 (days after this month's payday); 0 otherwise
  payday_weekend_wide  1 on a Friday, Saturday or Sunday whose Monday-to-Sunday week contains the latest payday; this also catches a Friday
                       payday whose weekend falls in the next month, which the first two cannot see

Variant 4 applies the Series 59 repair (`repair_all`: the three repaired columns in the old positions) and then adds `after_payday_flag`, to put
back the signal the repair removed:

  repair_plus_flag     repair_all + after_payday_flag

Each variant is run on its own, never in combination (the three signals overlap and there are about 17 such days in training, so combinations
would mostly add noise). The new column is appended at the end of the piece's inputs; every variant goes into every piece of the weekly log
exactly as in Series 54-59 (Series 54's docstring has the pieces, the post and hydro models, the correction and the baselines).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for variants 1-3, the change in the post model's and the hydro model's MAE after the correction against the deployed columns, with a
             95% moving-block bootstrap interval; where it comes from by weekday and piece (Saturday and Sunday above all); in how many folds it
             is better; the new days. For variant 4, the same against the deployed columns AND against `repair_all` from Series 59
  expected   little: the backtest holds 9 days of this kind, so any effect will carry a wide interval. If an explicit flag helps, the Saturday and
             Sunday pieces (which are separate models) should show it with opposite signs on the days after payday
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version, and the
             live API builds the same columns, so a new column would change the serving code too

Run:  python -m src.experiments.run_series60_payday_weekend list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series59_payday_repair as s59
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel, test_days
from src.features.build_features import add_payday_features
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series60"
RESULTS = OUT_DIR / "series60_payday_weekend.jsonl"
VARIANTS = ["none", "after_payday_flag", "after_payday_count", "payday_weekend_wide", "repair_plus_flag"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
EXTRA_COLUMNS = {"after_payday_flag": ["after_payday_flag"], "after_payday_count": ["after_payday_count"],
                 "payday_weekend_wide": ["payday_weekend_wide"], "repair_plus_flag": ["after_payday_flag"]}
NEW_COLUMNS = ["after_payday_flag", "after_payday_count", "payday_weekend_wide"]
SWAP = {"days_since_payday": "days_since_payday_fixed", "days_to_payday": "days_to_payday_fixed",
        "is_payday_window_pm3": "is_payday_window_pm3_fixed"}


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    if variant == "repair_plus_flag":
        cols = [SWAP.get(c, c) for c in cols]
    return cols + EXTRA_COLUMNS.get(variant, [])


def add_payday_weekend_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The three explicit "after an early payday" columns, from the last-working-day payday of each calendar month."""
    out = frame.copy()
    d = out[DATE_COL].dt.normalize()
    dates = add_payday_features(out[[DATE_COL]])  # `payday` is this month's, `prev_payday` last month's
    payday, prev = pd.to_datetime(dates["payday"]), pd.to_datetime(dates["prev_payday"])
    after = d > payday
    out["after_payday_flag"] = after.astype(int)
    out["after_payday_count"] = np.where(after, (d - payday).dt.days, 0)
    latest = payday.where(d >= payday, prev)
    week_start = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
    out["payday_weekend_wide"] = ((d.dt.dayofweek >= 4) & (latest >= week_start)).astype(int)
    return out


def init_worker60() -> None:
    s47.init_worker47()
    clash = (set(NEW_COLUMNS) | set(s59.NEW_COLUMNS)) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_payday_weekend_columns(s59.add_repaired_columns(add_hydro_thu(_STATE["df"])))


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


def compare_with_repair() -> None:
    """Variant 4 against `repair_all` from Series 59: does the flag win back what the repair lost?"""
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, dow = days["actual"].to_numpy(), pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    b60 = s54.load_bags("backtest", RESULTS, VARIANT_COLUMNS, s54.RESULTS)
    b59 = s54.load_bags("backtest", s59.RESULTS, s59.VARIANT_COLUMNS, s54.RESULTS)

    def package(source: dict, variant: str, mon_thu: str) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [source[("Friday", variant)], source[("Saturday", variant)], source[("Sunday", variant)]],
                         default=source[(mon_thu, variant)])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    print("\n\nREPAIR PLUS FLAG AGAINST THE SERIES 59 REPAIR (after the correction, all 585 days; negative = repair_plus_flag better)")
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        deployed, repair, plus = package(b60, "none", mon_thu), package(b59, "repair_all", mon_thu), package(b60, "repair_plus_flag", mon_thu)
        for label, other in (("vs repair_all", repair), ("vs deployed", deployed)):
            diff = np.abs(plus - a) - np.abs(other - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            print(f"  {name:12s} {label:14s} change {change:6.2f}  [{lo:6.2f}, {hi:6.2f}]   Sat {diff[dow == 5].mean():7.2f}  Sun {diff[dow == 6].mean():7.2f}")


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)
    compare_with_repair()


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, " + ", ".join(
                f"{v} {len(piece_features(piece, v))}" for v in VARIANTS[1:]))
        df, _ = load_protocol()
        df = s59.add_repaired_columns(df)
        df = add_payday_weekend_columns(df)
        flag = df["after_payday_flag"] == 1
        assert ((df["after_payday_count"] > 0) == flag).all(), "the count and the flag disagree"
        assert (df.loc[df["is_payday"] == 1, "after_payday_flag"] == 0).all(), "a payday itself carries the flag"
        wide = df["payday_weekend_wide"] == 1
        assert df.loc[wide, DATE_COL].dt.dayofweek.ge(4).all(), "the wide flag is on a weekday before Friday"
        print(f"\n{len(df)} days; after_payday_flag on {int(flag.sum())}, payday_weekend_wide on {int(wide.sum())}; "
              f"flag days not covered by the wide flag: {int((flag & ~wide).sum())}; wide days without the flag: {int((wide & ~flag).sum())}")
        print("after_payday_flag by weekday:", df.loc[flag, DATE_COL].dt.day_name().value_counts().to_dict())
        print("after_payday_count values:", df.loc[flag, "after_payday_count"].value_counts().sort_index().to_dict())
        backtest = df[(df[DATE_COL] < CUTOFF)]
        print(f"before the 56 new days: flag on {int((backtest['after_payday_flag'] == 1).sum())}, wide on {int((backtest['payday_weekend_wide'] == 1).sum())}; "
              f"new days: flag on {int((df.loc[df[DATE_COL] >= CUTOFF, 'after_payday_flag'] == 1).sum())}")
        show = df[df[DATE_COL].between("2024-03-25", "2024-04-01")]
        print(show[[DATE_COL, "is_payday", "days_since_payday", "days_to_payday", "after_payday_flag", "after_payday_count",
                    "payday_weekend_wide"]].to_string(index=False))
        jan = df[df[DATE_COL].between("2025-01-30", "2025-02-03")]
        print(jan[[DATE_COL, "is_payday", "after_payday_flag", "after_payday_count", "payday_weekend_wide"]].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker60, "series60")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
