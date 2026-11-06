"""
Series 59 - repair the current payday columns, and measure what the payday family is worth, before any new payday idea is tried.

The model has four payday columns, all built on one assumption (payday is the last working day of the month): `is_payday`, `days_to_payday`,
`days_since_payday` and `is_payday_window_pm3`. Two of them misbehave on the days at and after payday within the same month:

  days_since_payday  counts from the PREVIOUS month's payday, so it reads 28-33 on payday itself and about 29-31 on the weekend or bank-holiday
                     days after an early payday, where it should read 0, then 1 or 2
  days_to_payday     on those days after payday it is NEGATIVE (-1, -2: the date arithmetic rolls forward to the end of the SAME month and finds
                     the payday that has already gone), and only on a month's last calendar day does it jump to the true ~30 days to next month's

Before new definitions are compared, the model to beat should carry the best version of the current definition:

  none               today's columns (the deployed definition)
  fix_since_only     days_since_payday counts from the most recent payday (today if it is payday); nothing else changes
  repair_all         days_since_payday as above, days_to_payday counting to the next payday on or after today (never negative), and
                     is_payday_window_pm3 recomputed from the two (0 to 3 days to the next payday, or 0 to 3 days since the most recent one);
                     is_payday is unchanged
  drop_all_payday    the four payday columns removed: what the whole family is worth today

The repaired columns replace the old ones in the same positions, so column order is unchanged. Every variant goes into every piece of the weekly log
exactly as in Series 54-56 (Series 54's docstring has the pieces, the post and hydro models, the correction and the baselines; the hydro and
Saturday-fallback baselines are Series 54's `none` fits).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece; in how many folds it is better; the new days; for the repair, the number of
             days whose columns change
  expected   little: the repair changes the columns on only a few dozen days (payday itself and the few days after it); dropping all four may be level
             (Series 2 found `days_to_payday` and `days_since_payday` unhelpful on the old model)
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version,
             and the live API builds the same columns, so a repair would change the serving code too

Run:  python -m src.experiments.run_series59_payday_repair list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, run_parallel
from src.features.build_features import add_payday_features
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series59"
RESULTS = OUT_DIR / "series59_payday_repair.jsonl"
PAYDAY_COLUMNS = ["days_since_payday", "is_payday_window_pm3", "is_payday", "days_to_payday"]
VARIANTS = ["none", "fix_since_only", "repair_all", "drop_all_payday"]
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names
NEW_COLUMNS = ["days_since_payday_fixed", "days_to_payday_fixed", "is_payday_window_pm3_fixed"]


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    swap = {"fix_since_only": {"days_since_payday": "days_since_payday_fixed"},
            "repair_all": {"days_since_payday": "days_since_payday_fixed", "days_to_payday": "days_to_payday_fixed",
                           "is_payday_window_pm3": "is_payday_window_pm3_fixed"}}
    if variant == "drop_all_payday":
        return [c for c in cols if c not in PAYDAY_COLUMNS]
    return [swap.get(variant, {}).get(c, c) for c in cols]


def add_repaired_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The payday columns counted from the most recent and to the next payday, and the window flag recomputed from them."""
    out = frame.copy()
    d = out[DATE_COL].dt.normalize()
    dates = add_payday_features(out[[DATE_COL]])  # `payday` is this month's, `prev_payday` last month's
    latest = dates["payday"].where(d >= dates["payday"], dates["prev_payday"])
    # next month's payday for every day: one payday per calendar month, from the first day's month to the month after the last day's
    months = pd.period_range(d.min().to_period("M"), d.max().to_period("M") + 1, freq="M")
    helper = add_payday_features(pd.DataFrame({DATE_COL: months.to_timestamp()}))
    payday_of = dict(zip(months, pd.to_datetime(helper["payday"])))
    following = d.dt.to_period("M").map(lambda m: payday_of[m + 1])
    upcoming = dates["payday"].where(d <= dates["payday"], following)
    since, to = (d - latest).dt.days, (upcoming - d).dt.days
    out["days_since_payday_fixed"], out["days_to_payday_fixed"] = since, to
    out["is_payday_window_pm3_fixed"] = (to.between(0, 3) | since.between(0, 3)).astype(int)
    return out


def init_worker59() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_repaired_columns(add_hydro_thu(_STATE["df"]))


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
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, drop_all_payday {len(piece_features(piece, 'drop_all_payday'))}")
        from src.experiments.series19_21_common import load_protocol

        df, _ = load_protocol()
        df = add_repaired_columns(df)
        old_window = (df["days_to_payday"].between(0, 3) | df["days_since_payday"].between(0, 3)).astype(int)
        assert (old_window == df["is_payday_window_pm3"]).all(), "the window flag is not what the code says it is"
        changed = df["days_since_payday"] != df["days_since_payday_fixed"]
        to_changed = df["days_to_payday"] != df["days_to_payday_fixed"]
        flag_changed = df["is_payday_window_pm3"] != df["is_payday_window_pm3_fixed"]
        print(f"\n{len(df)} days; days_since_payday changes on {int(changed.sum())} days, days_to_payday on {int(to_changed.sum())} "
              f"({int((df['days_to_payday'] < 0).sum())} of them negative today), the window flag on {int(flag_changed.sum())}")
        print("days_since_payday changed, by weekday:", df.loc[changed, DATE_COL].dt.day_name().value_counts().to_dict())
        print("never negative and in range after the repair:", bool((df["days_to_payday_fixed"] >= 0).all()), bool((df["days_since_payday_fixed"] >= 0).all()),
              int(df["days_to_payday_fixed"].max()), int(df["days_since_payday_fixed"].max()))
        print(df.loc[changed | to_changed, [DATE_COL, "days_since_payday", "days_since_payday_fixed", "days_to_payday", "days_to_payday_fixed",
                                            "is_payday_window_pm3", "is_payday_window_pm3_fixed"]].head(14).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker59, "series59")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
