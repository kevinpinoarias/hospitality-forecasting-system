"""
Series 49 - does a leap-year-safe day of year change anything?

`day_of_year` counts from 1 January, so in a leap year every date after 29 February is one higher than the same date in other years (1 March is
day 61 in 2024 and 60 in 2025), and day 366 (31 December 2024) lands on the same point of the sine/cosine circle as day 1 because the circle's
period is 365. The data holds one leap year, 2024.

The leap-safe version numbers the days as if every year had 365: after 29 February of a leap year the count is one lower, and 29 February takes
28 February's number (59). 1 March is then always day 60, Christmas Day always 359, and 31 December 2024 is day 365. `day_of_year`,
`day_of_year_sin` and `day_of_year_cos` are all recomputed from it; every other column is untouched.

PRE-REGISTERED (written before any result was seen)
  H49        the leap-safe day of year lowers the final model's MAE
  protocol   the final model's own, as Series 46: log1p target, 300 iterations, depth 4, 10 rolling-origin folds, 585 test days, 100 seeds (200-299)
  reading    pooled MAE over the 585 days and the change against the final model with a 95% moving-block bootstrap interval, the same by test
             year, and how many training rows the change touches in each fold
  expected   a tie: the 585 test days are all in 2025 and 2026, so the change can only act on how the 2024 training rows line up with later dates
  decision   better only if the interval excludes zero; a tie is reported as a tie. A tie does not say the change is wrong: it is a correctness
             fix for the next leap year (2028), and whether to adopt it on those grounds is Kevin's call.

Run:  python -m src.experiments.run_series49_leap_safe_day_of_year list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import (
    _STATE, DATE_COL, FINAL_FEATURES, FINAL_PARAMS, bagged_log1p_predictions, init_worker, load_protocol, read_done, run_parallel, test_days,
)
from src.features.build_features import add_cyclical

CHUNK = 5
VERSIONS = ("as_today", "leap_safe")
OUT_DIR = CONFIG.path("experiments_dir") / "series49"
RESULTS = OUT_DIR / "series49_leap_safe_day_of_year.jsonl"


def leap_safe_day_of_year(dates: pd.Series) -> pd.Series:
    """Day of year counted as if every year had 365 days (29 February takes 28 February's number)."""
    doy = dates.dt.dayofyear
    after_leap_day = dates.dt.is_leap_year & (dates.dt.month > 2)
    safe = doy - after_leap_day.astype(int)
    return safe.mask(dates.dt.is_leap_year & (dates.dt.month == 2) & (dates.dt.day == 29), 59)


def with_leap_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_year"] = leap_safe_day_of_year(out[DATE_COL])
    return add_cyclical(out, "day_of_year", 365)


def init_worker49() -> None:
    init_worker()
    _STATE["df_leap_safe"] = with_leap_safe(_STATE["df"])


def fit_task(task: dict) -> dict:
    started = time.time()
    df = _STATE["df_leap_safe"] if task["version"] == "leap_safe" else _STATE["df"]
    per_seed = bagged_log1p_predictions(df, _STATE["folds"], list(FINAL_FEATURES), FINAL_PARAMS, task["seeds"])
    return {"id": task["id"], "version": task["version"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    return [{"id": f"{v}_s{SEEDS[i]}-{SEEDS[i + CHUNK - 1]}", "version": v, "seeds": SEEDS[i:i + CHUNK]}
            for v in VERSIONS for i in range(0, len(SEEDS), CHUNK)]


def load_bags() -> dict[str, np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for v in VERSIONS:
        per_seed = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["version"] == v], axis=0)
        assert per_seed.shape[0] == len(SEEDS), (v, per_seed.shape)
        out[v] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    year = pd.to_datetime(days["date"]).dt.year.to_numpy()
    bags = load_bags()
    safe = with_leap_safe(df)
    changed = (safe["day_of_year"] != df["day_of_year"]).to_numpy()
    print(f"Rows in the model data: {len(df)}; day-of-year number differs on {int(changed.sum())} of them "
          f"({df.loc[changed, DATE_COL].min().date()} to {df.loc[changed, DATE_COL].max().date()}), all in leap year 2024.")
    print("Training rows touched per fold:", {int(k): int((changed & (df[DATE_COL] < f.train_end).to_numpy()).sum()) for k, f in enumerate(folds)}, "\n")

    diff = np.abs(bags["leap_safe"] - a) - np.abs(bags["as_today"] - a)
    change, lo, hi = masked_block_bootstrap(diff, np.ones(len(a)))
    print(f"Final model MAE {np.abs(bags['as_today'] - a).mean():.2f}; leap-safe {np.abs(bags['leap_safe'] - a).mean():.2f}; "
          f"change {change:+.2f} (95% interval {lo:+.2f} to {hi:+.2f})")
    for y in sorted(set(year)):
        print(f"  test year {y}: change {diff[year == y].mean():+.2f} over {int((year == y).sum())} days")
    print("  folds better:", int(sum(diff[fold == k].mean() < 0 for k in range(10))), "of 10")
    print("\nDECISION RULE (interval excluding zero):", "leap-safe is better" if hi < 0 else "tie" if lo <= 0 <= hi else "leap-safe is worse")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"change": change, "lo": lo, "hi": hi}]).to_csv(OUT_DIR / "leap_safe.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        dates = pd.Series(pd.to_datetime(["2024-02-28", "2024-02-29", "2024-03-01", "2024-12-25", "2024-12-31", "2025-03-01", "2025-12-25"]))
        print(f"{len(tasks)} tasks. Leap-safe numbering check:")
        print(pd.DataFrame({"date": dates.dt.date, "today": dates.dt.dayofyear, "leap_safe": leap_safe_day_of_year(dates)}).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker49, "series49")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
