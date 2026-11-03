"""
Series 44 - does the final model need all six month and day-of-year columns?

The final model reads `month`, `month_sin`, `month_cos`, `day_of_year`, `day_of_year_sin` and `day_of_year_cos`: the same two
calendar facts written three ways each. Series 1's leave-one-out removed one column at a time and found that removing most of them
helped, but every column still had two siblings carrying its signal, so that test could not say whether the columns are useless or
merely redundant. This series removes whole encodings instead. Each of the two facts gets one of four options:

  none   the fact is left out altogether
  raw    the plain number (`month`, `day_of_year`)
  pair   the sine and cosine columns only
  both   the plain number plus the pair (what the final model has today)

4 options for the month x 4 for the day of year = 16 variants, `month-both__doy-both` being the final model. Every other column keeps its
place in the final model's order, so only the six columns move.

PRE-REGISTERED (written before any result was seen)
  H44        some variant with fewer of the six columns scores a lower MAE than all six
  protocol   the final model's own: log1p target, 300 iterations, depth 4, the 950 rows, 10 rolling-origin folds, 585 test days
  seeds      25 seeds (0-24) per variant, repeated on 25 different seeds (100-124) as a check
  reading    pooled MAE over the 585 days, the change against `month-both__doy-both` with a 95% moving-block bootstrap interval,
             the same over the December test days, and the main effect of each fact (mean MAE of the variants sharing one option)
  decision   a smaller set replaces the six only if its change is below zero with an interval that excludes zero on BOTH seed sets;
             otherwise the six stay. A variant that ties the six within noise is reported as a tie, not a win: fewer columns
             would be simpler but would not be shown to be better.
  not tested a leap-year-safe day of year (day 366 and the one-day shift after 29 February); that is the next series, built on
             whichever of these encodings comes out best

Run:  python -m src.experiments.run_series44_month_doy_encodings list
      python -m src.experiments.run_series44_month_doy_encodings run [--workers 10]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series44_month_doy_encodings report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.series19_21_common import (
    _STATE, FINAL_FEATURES, FINAL_PARAMS, bagged_log1p_predictions, init_worker, load_protocol, read_done, run_parallel, test_days,
)

SEED_SETS = {"seeds0_24": list(range(25)), "seeds100_124": list(range(100, 125))}
SEED_CHUNK = 5
MONTH_OPTIONS = {"none": [], "raw": ["month"], "pair": ["month_sin", "month_cos"], "both": ["month", "month_sin", "month_cos"]}
DOY_OPTIONS = {
    "none": [], "raw": ["day_of_year"], "pair": ["day_of_year_sin", "day_of_year_cos"],
    "both": ["day_of_year", "day_of_year_sin", "day_of_year_cos"],
}
SIX = MONTH_OPTIONS["both"] + DOY_OPTIONS["both"]
BASELINE = "month-both__doy-both"
OUT_DIR = CONFIG.path("experiments_dir") / "series44"
RESULTS = OUT_DIR / "series44_month_doy_encodings.jsonl"


def variant_name(month: str, doy: str) -> str:
    return f"month-{month}__doy-{doy}"


def variant_features(month: str, doy: str) -> list[str]:
    """The final model's columns, in its order, with the six calendar columns cut down to the chosen options."""
    keep = set(MONTH_OPTIONS[month]) | set(DOY_OPTIONS[doy])
    return [f for f in FINAL_FEATURES if f not in SIX or f in keep]


VARIANTS = {variant_name(m, d): (m, d) for m in MONTH_OPTIONS for d in DOY_OPTIONS}
assert set(SIX) <= set(FINAL_FEATURES), "the six calendar columns must all be in the final model's features"
assert variant_features("both", "both") == list(FINAL_FEATURES)


# --------------------------------------------------------------------------- fitting


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    per_seed = bagged_log1p_predictions(df, folds, task["features"], FINAL_PARAMS, task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "set": task["set"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for set_name, seeds in SEED_SETS.items():
        for variant, (m, d) in VARIANTS.items():
            for i in range(0, len(seeds), SEED_CHUNK):
                chunk = seeds[i:i + SEED_CHUNK]
                out.append({"id": f"{variant}_{set_name}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "set": set_name,
                            "seeds": chunk, "features": variant_features(m, d)})
    return out


def load_bags() -> dict[str, dict[str, np.ndarray]]:
    """seed set -> variant -> the 25-seed bag in pounds, shape (585,)."""
    done = read_done(RESULTS)
    out: dict[str, dict[str, np.ndarray]] = {s: {} for s in SEED_SETS}
    for set_name, seeds in SEED_SETS.items():
        for variant in VARIANTS:
            rows = [np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant and r["set"] == set_name]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(seeds), (set_name, variant, per_seed.shape)
            out[set_name][variant] = per_seed.mean(axis=0)
    return out


# --------------------------------------------------------------------------- reading


def report() -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    a = days["actual"].to_numpy()
    december = (pd.to_datetime(days["date"]).dt.month == 12).to_numpy()
    bags = load_bags()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test days {len(a)}, of which December {int(december.sum())}")

    summary = []
    for set_name in SEED_SETS:
        base = bags[set_name][BASELINE]
        for variant, (m, d) in VARIANTS.items():
            p = bags[set_name][variant]
            diff = np.abs(p - a) - np.abs(base - a)
            change, lo, hi = masked_block_bootstrap(diff, np.ones(len(a)))
            dec_change = float(diff[december].mean())
            summary.append({"seeds": set_name, "variant": variant, "month": m, "doy": d, "n_features": len(variant_features(m, d)),
                            "MAE": float(np.abs(p - a).mean()), "change": change, "lo": lo, "hi": hi,
                            "dec_MAE": float(np.abs(p - a)[december].mean()), "dec_change": dec_change})
    table = pd.DataFrame(summary)
    table.to_csv(OUT_DIR / "variants.csv", index=False)

    for set_name in SEED_SETS:
        sub = table[table["seeds"] == set_name].sort_values("MAE")
        print(f"\n===== {set_name}: all 16 variants, best first (change is against {BASELINE}; negative = better) =====")
        print(sub.drop(columns=["seeds"]).round(2).to_string(index=False))
        print(f"\n{set_name}: main effect of each fact (mean MAE over the variants that share one option)")
        for fact in ("month", "doy"):
            print(sub.groupby(fact)["MAE"].mean().round(2).rename(f"{fact} option").to_string(), "\n")

    wins = table.groupby("variant").apply(lambda g: bool((g["hi"] < 0).all()), include_groups=False)
    winners = sorted(wins[wins].index)
    print("\nDECISION RULE (change below zero, interval excluding zero, on both seed sets):",
          ", ".join(winners) if winners else "no variant qualifies - the six columns stay")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(SEED_SETS)} seed sets = {len(tasks)} tasks of {SEED_CHUNK} seeds each")
        for name, (m, d) in VARIANTS.items():
            print(f"  {name:24s} {len(variant_features(m, d))} features")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker, "series44")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
