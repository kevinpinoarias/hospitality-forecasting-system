"""
Series 64 - the parts of the original payday plan that Series 59-63 did not run: symmetric windows, a graded distance, and other payday dates
as complete column sets.

The plan written before this family was started had four items. Series 59 ran the first (fix the flaw), Series 60 and 61 ran the fourth (the
payday weekend), and Series 61 ran only a narrow slice of the second and third (a "payday and the day after" flag, and a mix of three rules). This
series runs what is left, with no pattern taken from the model's errors: it is the plan as written, so the selection bias of Series 60-61 does not apply.

A. Window shapes around the last-working-day payday (the deployed rule), each ADDED as one new column next to the deployed 38 (flaw included):

  lwd_window_pm1       1 on the day before payday, payday and the day after; 0 otherwise
  lwd_window_pm2       1 from 2 days before to 2 days after payday
  lwd_window_pm5       1 from 5 days before to 5 days after payday
  lwd_signed_distance  signed days to the nearest payday (negative before, positive after), clipped to -7..+7; a tie goes to "after"

  Distances are true ones: days since the latest payday and days to the next payday on or after the day, both 0 on payday itself. The deployed
  `is_payday_window_pm3` is the +-3 version of the same idea, built from the flawed columns, and stays in as it is.

B. Other payday dates, rebuilt as complete column sets that REPLACE the four deployed payday columns in their positions (column order and count
   unchanged). For a date rule, payday is that day of the month, or the working day before it when it falls on a weekend or a Scottish bank holiday:

  def_25   the 25th
  def_28   the 28th
  def_15   the 15th

  Each rebuilds days_since_payday (days since the latest payday), days_to_payday (days to the next payday on or after the day), is_payday (1 on
  the payday) and is_payday_window_pm3 (within 3 days either side), all correct, none of the flaw.

  The right comparator for B is Series 59's `repair_all`, which is the same four columns, correct, for the last working day: it shows what changing the
  DATE costs or gains once the columns are built the same way. The report prints each B variant against the deployed model and against `repair_all`.

What to keep in mind (set before any result was seen)
  Series 59 and 62 found that replacing the deployed payday columns with correct ones costs 3 to 6 pounds a day, and Series 63 found that even adding
  a redundant column moves the score by about 3. Series 19 put the shift from column order alone at about 2. So a B variant that is 2-3 pounds worse
  than the deployed model is the expected cost of replacing the columns, not a verdict on the date. Read B against `repair_all`, not against the deployed
  model, and A against the deployed model.

READING (by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 new days from 7 September 2026
  look at    for each variant, the change in the post and hydro models' MAE after the correction with a 95% moving-block bootstrap interval; by
             weekday and piece; folds better; the new days; for B also against `repair_all`
  expected   A: nothing (Series 61's 15th control and mix windows were flat). B: all three worse than the deployed model by 3-6 pounds; def_28 and
             def_25 closer to `repair_all` than def_15, if the research on pay dates points the right way
  would matter   a B variant clearly better than `repair_all` (interval below zero) AND not worse than the deployed model: the date rule would
             then be worth a closer look. An A variant better than the deployed model with an interval below zero and a better direction on the 56 new days
  limits     no pass/fail gate; nothing in production changes; the live API builds the same columns, so any new column would change the serving code too

Run:  python -m src.experiments.run_series64_payday_plan_rest list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series59_payday_repair as s59
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.run_series61_payday_rules import days_around
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, run_parallel, test_days
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series64"
RESULTS = OUT_DIR / "series64_payday_plan_rest.jsonl"
WINDOWS = {"lwd_window_pm1": 1, "lwd_window_pm2": 2, "lwd_window_pm5": 5}
DEFS = {"def_25": 25, "def_28": 28, "def_15": 15}
VARIANTS = ["none", *WINDOWS, "lwd_signed_distance", *DEFS]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
DEPLOYED_FOUR = ["days_since_payday", "days_to_payday", "is_payday", "is_payday_window_pm3"]


def def_columns(rule: int) -> list[str]:
    return [f"r{rule}_days_since", f"r{rule}_days_to", f"r{rule}_is_payday", f"r{rule}_window_pm3"]


NEW_COLUMNS = list(WINDOWS) + ["lwd_signed_distance"] + [c for rule in DEFS.values() for c in def_columns(rule)]


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    if variant in DEFS:
        swap = dict(zip(DEPLOYED_FOUR, def_columns(DEFS[variant])))
        return [swap.get(c, c) for c in cols]
    return cols + ([variant] if variant != "none" else [])


def add_plan_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    since, to = days_around(out[DATE_COL], "last_working_day", calendar)
    for name, radius in WINDOWS.items():
        out[name] = ((since <= radius) | (to <= radius)).astype(int)
    out["lwd_signed_distance"] = np.clip(np.where(since <= to, since, -to), -7, 7)
    for rule in DEFS.values():
        s, t = days_around(out[DATE_COL], rule, calendar)
        out[f"r{rule}_days_since"], out[f"r{rule}_days_to"] = s, t
        out[f"r{rule}_is_payday"] = (s == 0).astype(int)
        out[f"r{rule}_window_pm3"] = ((s <= 3) | (t <= 3)).astype(int)
    return out


def init_worker64() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_plan_columns(add_hydro_thu(_STATE["df"]))


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


def compare_defs_with_repair() -> None:
    """The three date rules against Series 59's `repair_all` (the last working day, same four columns, correct)."""
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, dow = days["actual"].to_numpy(), pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    b64 = s54.load_bags("backtest", RESULTS, VARIANT_COLUMNS, s54.RESULTS)
    b59 = s54.load_bags("backtest", s59.RESULTS, s59.VARIANT_COLUMNS, s54.RESULTS)

    def package(source: dict, variant: str, mon_thu: str) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [source[("Friday", variant)], source[("Saturday", variant)], source[("Sunday", variant)]],
                         default=source[(mon_thu, variant)])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    print("\n\nOTHER PAYDAY DATES AGAINST THE SERIES 59 REPAIR (after the correction, all 585 days; negative = the date rule is better than the last working day)")
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        repair = package(b59, "repair_all", mon_thu)
        for variant in DEFS:
            diff = np.abs(package(b64, variant, mon_thu) - a) - np.abs(repair - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            print(f"  {name:12s} {variant:8s} vs repair_all change {change:6.2f}  [{lo:6.2f}, {hi:6.2f}]   Fri {diff[dow == 4].mean():7.2f}  "
                  f"Sat {diff[dow == 5].mean():7.2f}  Sun {diff[dow == 6].mean():7.2f}")


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)
    compare_defs_with_repair()


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, windows/distance +1, defs "
                  f"{len(piece_features(piece, 'def_25'))}")
            for v in DEFS:
                assert len(piece_features(piece, v)) == len(piece_features(piece, "none")), "a date rule changed the number of columns"
                assert not set(DEPLOYED_FOUR) & set(piece_features(piece, v)), "a deployed payday column is still there"
        df, _ = load_protocol()
        df = add_plan_columns(df)
        df = s59.add_repaired_columns(df)
        # the last-working-day columns built here must equal Series 59's correct ones
        calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
        since, to = days_around(df[DATE_COL], "last_working_day", calendar)
        assert (since == df["days_since_payday_fixed"]).all() and (to == df["days_to_payday_fixed"]).all(), "differs from the Series 59 repair"
        assert (df["lwd_window_pm1"] <= df["lwd_window_pm2"]).all() and (df["lwd_window_pm2"] <= df["lwd_window_pm5"]).all()
        assert ((since == 0) == (df["is_payday"] == 1)).all(), "the last-working-day payday is not the deployed one"
        print(f"\n{len(df)} days; the last-working-day distances equal the Series 59 repair on every day")
        for name in list(WINDOWS) + [f"r{rule}_window_pm3" for rule in DEFS.values()] + [f"r{rule}_is_payday" for rule in DEFS.values()]:
            print(f"  {name:22s} on {int((df[name] == 1).sum()):4d} of {len(df)} days")
        print("  lwd_signed_distance values", sorted(df["lwd_signed_distance"].unique().tolist()))
        for rule in DEFS.values():
            print(f"  rule {rule}: days_since max {int(df[f'r{rule}_days_since'].max())}, days_to max {int(df[f'r{rule}_days_to'].max())}, "
                  f"paydays that are not on the {rule}th: {int((df.loc[df[f'r{rule}_is_payday'] == 1, DATE_COL].dt.day != rule).sum())}")
        show = df[df[DATE_COL].between("2025-09-24", "2025-10-01")]
        print(show[[DATE_COL, "is_payday", "lwd_window_pm1", "lwd_window_pm2", "lwd_signed_distance", "r25_is_payday", "r25_days_since",
                    "r28_is_payday", "r28_days_since", "r28_days_to"]].to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker64, "series64")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
