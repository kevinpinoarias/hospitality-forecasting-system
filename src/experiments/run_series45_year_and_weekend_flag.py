"""
Series 45 - does the final model need `year`, and is `is_weekend` the right weekend flag?

Two questions about the final model's 38 columns, tested together so each can be read separately:

  year        `year` (2024, 2025, 2026) can only be cut into the years the model has seen, so in 2027 it behaves as 2026. kept / dropped.
  is_weekend  1 on Friday, Saturday and Sunday (day_of_week >= 4), 0 otherwise. `day_of_week` and its sin/cos stay in every variant.
                as_today              the final model's `is_weekend`
                removed               no weekend flag at all
                single_days           `is_friday`, `is_saturday`, `is_sunday` (three separate 0/1 columns)
                fri_sat_plus_sunday   `is_fri_sat` (Friday and Saturday together) and `is_sunday`

2 x 4 = 8 variants; year kept + is_weekend as_today is the final model. The replacement flags sit where `is_weekend` sits in the
final model's column order (Series 19: CatBoost's result moves by a few pounds with column order), and every other column keeps its place.
The new flags are built here from `day_of_week`; the production pipeline is untouched.

PRE-REGISTERED (written before any result was seen)
  H45        dropping `year`, or changing the weekend flag, lowers the MAE of the final model
  protocol   the final model's own, as in Series 44: log1p target, 300 iterations, depth 4, 950+ rows, 10 rolling-origin folds, 585 test days
  seeds      25 seeds (0-24) per variant, repeated on 25 different seeds (100-124) as a check
  reading    pooled MAE over the 585 days and the change against the final model with a 95% moving-block bootstrap interval; the same
             split into Monday-Thursday, Friday, Saturday and Sunday; the main effect of each question
  decision   a variant replaces the final model only if its change is below zero with an interval that excludes zero on BOTH seed sets.
             A tie is reported as a tie. Whether to drop `year` or `is_weekend` on a tie (fewer columns, and `year` cannot extend into
             2027) is Kevin's call after seeing the result; it is not made here.

Run:  python -m src.experiments.run_series45_year_and_weekend_flag list
      python -m src.experiments.run_series45_year_and_weekend_flag run [--workers 10]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series45_year_and_weekend_flag report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.run_series44_month_doy_encodings import SEED_CHUNK, SEED_SETS, fit_task
from src.experiments.series19_21_common import _STATE, FINAL_FEATURES, init_worker, load_protocol, read_done, run_parallel, test_days

YEAR_OPTIONS = ("kept", "dropped")
WEEKEND_OPTIONS = {
    "as_today": ["is_weekend"],
    "removed": [],
    "single_days": ["is_friday", "is_saturday", "is_sunday"],
    "fri_sat_plus_sunday": ["is_fri_sat", "is_sunday"],
}
BASELINE = ("kept", "as_today")
OUT_DIR = CONFIG.path("experiments_dir") / "series45"
RESULTS = OUT_DIR / "series45_year_and_weekend_flag.jsonl"
WEEKDAY_GROUPS = {"Mon-Thu": [0, 1, 2, 3], "Fri": [4], "Sat": [5], "Sun": [6]}


def variant_name(year: str, weekend: str) -> str:
    return f"year_{year}__weekend_{weekend}"


def variant_features(year: str, weekend: str) -> list[str]:
    out: list[str] = []
    for f in FINAL_FEATURES:
        if f == "year" and year == "dropped":
            continue
        if f == "is_weekend":
            out += WEEKEND_OPTIONS[weekend]
        else:
            out.append(f)
    return out


VARIANTS = {variant_name(y, w): (y, w) for y in YEAR_OPTIONS for w in WEEKEND_OPTIONS}
assert {"year", "is_weekend"} <= set(FINAL_FEATURES)
assert variant_features(*BASELINE) == list(FINAL_FEATURES)


def add_weekday_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_friday"] = (out["day_of_week"] == 4).astype(int)
    out["is_saturday"] = (out["day_of_week"] == 5).astype(int)
    out["is_sunday"] = (out["day_of_week"] == 6).astype(int)
    out["is_fri_sat"] = out["day_of_week"].isin([4, 5]).astype(int)
    return out


def init_worker45() -> None:
    init_worker()
    _STATE["df"] = add_weekday_flags(_STATE["df"])


def make_tasks() -> list[dict]:
    out = []
    for set_name, seeds in SEED_SETS.items():
        for variant, (y, w) in VARIANTS.items():
            for i in range(0, len(seeds), SEED_CHUNK):
                chunk = seeds[i:i + SEED_CHUNK]
                out.append({"id": f"{variant}_{set_name}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "set": set_name,
                            "seeds": chunk, "features": variant_features(y, w)})
    return out


def load_bags() -> dict[str, dict[str, np.ndarray]]:
    done = read_done(RESULTS)
    out: dict[str, dict[str, np.ndarray]] = {s: {} for s in SEED_SETS}
    for set_name, seeds in SEED_SETS.items():
        for variant in VARIANTS:
            rows = [np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant and r["set"] == set_name]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(seeds), (set_name, variant, per_seed.shape)
            out[set_name][variant] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    a = days["actual"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    bags = load_bags()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = variant_name(*BASELINE)

    rows = []
    for set_name in SEED_SETS:
        base = bags[set_name][base_name]
        for variant, (y, w) in VARIANTS.items():
            p = bags[set_name][variant]
            diff = np.abs(p - a) - np.abs(base - a)
            change, lo, hi = masked_block_bootstrap(diff, np.ones(len(a)))
            row = {"seeds": set_name, "year": y, "is_weekend": w, "n_features": len(variant_features(y, w)),
                   "MAE": float(np.abs(p - a).mean()), "change": change, "lo": lo, "hi": hi}
            for group, days_in in WEEKDAY_GROUPS.items():
                mask = np.isin(dow, days_in)
                row[f"{group}_change"] = float(diff[mask].mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "variants.csv", index=False)

    for set_name in SEED_SETS:
        sub = table[table["seeds"] == set_name].sort_values("MAE")
        print(f"\n===== {set_name}: 8 variants, best first (change is against year kept + is_weekend as_today; negative = better) =====")
        print(sub.drop(columns=["seeds"]).round(2).to_string(index=False))
        print(f"\n{set_name}: main effect (mean MAE over the variants sharing one option)")
        print(sub.groupby("year")["MAE"].mean().round(2).to_string(), "\n")
        print(sub.groupby("is_weekend")["MAE"].mean().round(2).to_string(), "\n")

    wins = table.groupby(["year", "is_weekend"]).apply(lambda g: bool((g["hi"] < 0).all()), include_groups=False)
    winners = [f"year {y} / is_weekend {w}" for (y, w), ok in wins.items() if ok]
    print("\nDECISION RULE (change below zero, interval excluding zero, on both seed sets):",
          ", ".join(winners) if winners else "no variant qualifies - the final model's columns stay")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(SEED_SETS)} seed sets = {len(tasks)} tasks of {SEED_CHUNK} seeds each")
        for name, (y, w) in VARIANTS.items():
            print(f"  {name:48s} {len(variant_features(y, w))} features")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker45, "series45")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
