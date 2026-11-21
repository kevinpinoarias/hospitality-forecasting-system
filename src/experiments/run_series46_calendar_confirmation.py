"""
Series 46 - confirming the calendar-column candidates from Series 44 and 45 on fresh seeds and on days no earlier series scored.

Series 44 and 45 found nothing that met their rule, but three things looked worth a second look:
  drop `month`              Series 44: the plain month number adds nothing once `month_sin`, `month_cos` and the day-of-year columns are there
  single-day flags          Series 45: `is_friday`, `is_saturday`, `is_sunday` in place of `is_weekend` (-7.6 and -12.2 on the two seed sets)
  drop `year`               Series 45: -3.8 and -3.7 on the main effect, but it moved error between weekdays

Two kinds of noise stand between those numbers and a decision, and this series removes them one at a time:

  part A, backtest      the same 585 test days (15 January 2025 to 6 September 2026) with 100 NEW seeds (200-299) per variant. More seeds
                        shrink the noise from CatBoost's own randomness (two 25-seed bags of the identical model differed by about 2.8 pounds
                        in Series 44). They cannot shrink the noise from WHICH days were scored.
  part B, new days      every model fitted on rows before 7 September 2026 and scored on the days from 7 September on (the days after the
                        585, Series 35's holdout days plus what has arrived since). No earlier series chose any calendar column using
                        them; they were looked at for other questions (Series 35, 36) and are few, so they can contradict a candidate
                        but cannot confirm one.

6 variants, each a list of the final model's 38 columns with the changes below; every other column keeps its place.

PRE-REGISTERED (written before any result was seen)
  protocol   the final model's: log1p target, 300 iterations, depth 4, 10 rolling-origin folds for part A; 100 seeds (200-299) for both parts
  reading    part A: pooled MAE over the 585 days, change against the final model with a 95% moving-block bootstrap interval, and the
             change by Mon-Thu, Fri, Sat, Sun. Part B: MAE over the new days and the change against the final model (7-day blocks).
  decision   a variant is SUPPORTED only if (a) part A's change is below zero with an interval that excludes zero AND (b) part B's change
             is below zero. Part B cannot reach significance with this few days; it is a veto, not a confirmation. A variant that is
             not supported leaves the final model's columns as they are. Supported or not, nothing in the production pipeline or the
             served model changes in this series: adopting a candidate and retraining are Kevin's decisions.
  caution    the candidates were picked after seeing Series 44 and 45 on the same 585 days, so part A can fall short of "real" even when
             it passes; that is why part B exists.

Run:  python -m src.experiments.run_series46_calendar_confirmation list
      python -m src.experiments.run_series46_calendar_confirmation run [--workers 10]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series46_calendar_confirmation report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.run_series44_month_doy_encodings import fit_task as backtest_fit_task
from src.experiments.run_series45_year_and_weekend_flag import WEEKDAY_GROUPS, add_weekday_flags, variant_features as year_weekend_features
from src.experiments.series19_21_common import (
    _STATE, DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL, init_worker, load_protocol, read_done, run_parallel, test_days,
)

SEEDS = list(range(200, 300))
BACKTEST_CHUNK, NEW_DAYS_CHUNK = 5, 10
CUTOFF = pd.Timestamp("2026-09-07")  # the first day after the 585 backtest days
OUT_DIR = CONFIG.path("experiments_dir") / "series46"
RESULTS = OUT_DIR / "series46_calendar_confirmation.jsonl"

# name -> (drop the plain `month` number, weekend flag option, year option), described in plain words in LABELS
CANDIDATES: dict[str, tuple[bool, str, str]] = {
    "final_model": (False, "as_today", "kept"),
    "drop_month": (True, "as_today", "kept"),
    "single_day_flags": (False, "single_days", "kept"),
    "drop_month_and_single_day_flags": (True, "single_days", "kept"),
    "single_day_flags_and_drop_year": (False, "single_days", "dropped"),
    "drop_month_and_single_day_flags_and_drop_year": (True, "single_days", "dropped"),
}
LABELS = {
    "final_model": "Today's 38 columns",
    "drop_month": "Without the plain month number",
    "single_day_flags": "is_weekend replaced by is_friday / is_saturday / is_sunday",
    "drop_month_and_single_day_flags": "Both of the above",
    "single_day_flags_and_drop_year": "Single-day flags, and without year",
    "drop_month_and_single_day_flags_and_drop_year": "Single-day flags, without the month number and without year",
}
BASELINE = "final_model"


def features_for_candidate(name: str) -> list[str]:
    drop_month, weekend, year = CANDIDATES[name]
    return [f for f in year_weekend_features(year, weekend) if not (drop_month and f == "month")]


assert features_for_candidate(BASELINE) == list(FINAL_FEATURES)


# --------------------------------------------------------------------------- fitting


def init_worker46() -> None:
    init_worker()
    _STATE["df"] = add_weekday_flags(_STATE["df"])


def new_days_task(task: dict) -> dict:
    """Fit on the rows before CUTOFF, one model per seed, and predict every row from CUTOFF on."""
    from catboost import CatBoostRegressor

    df = _STATE["df"]
    train, test = df[df[DATE_COL] < CUTOFF], df[df[DATE_COL] >= CUTOFF]
    started, preds = time.time(), []
    for seed in task["seeds"]:
        model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
        model.fit(train[task["features"]], np.log1p(train[TARGET_COL]))
        preds.append(np.expm1(model.predict(test[task["features"]])))
    return {"id": task["id"], "variant": task["variant"], "part": "new_days", "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in preds], "seconds": round(time.time() - started, 1)}


def fit_task(task: dict) -> dict:
    if task["part"] == "new_days":
        return new_days_task(task)
    rec = backtest_fit_task(task)
    rec["part"] = "backtest"
    return rec


def make_tasks() -> list[dict]:
    out = []
    for variant in CANDIDATES:
        features = features_for_candidate(variant)
        for i in range(0, len(SEEDS), BACKTEST_CHUNK):
            chunk = SEEDS[i:i + BACKTEST_CHUNK]
            out.append({"id": f"backtest_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "variant": variant,
                        "set": "fresh200_299", "seeds": chunk, "features": features})
        for i in range(0, len(SEEDS), NEW_DAYS_CHUNK):
            chunk = SEEDS[i:i + NEW_DAYS_CHUNK]
            out.append({"id": f"new_days_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "new_days", "variant": variant,
                        "seeds": chunk, "features": features})
    return out


def load_bags(part: str) -> dict[str, np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for variant in CANDIDATES:
        rows = [np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant and r["part"] == part]
        per_seed = np.concatenate(rows, axis=0)
        assert per_seed.shape[0] == len(SEEDS), (part, variant, per_seed.shape)
        out[variant] = per_seed.mean(axis=0)
    return out


# --------------------------------------------------------------------------- reading


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, f"the protocol now has {len(folds)} folds; Series 46 is defined on the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    assert days["date"].iloc[-1] == "2026-09-06" and len(days) == 585, "the 585 backtest days have moved"
    a = days["actual"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new = new[TARGET_COL].to_numpy()
    backtest, fresh = load_bags("backtest"), load_bags("new_days")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in CANDIDATES:
        diff = np.abs(backtest[variant] - a) - np.abs(backtest[BASELINE] - a)
        change, lo, hi = masked_block_bootstrap(diff, np.ones(len(a)))
        diff_new = np.abs(fresh[variant] - a_new) - np.abs(fresh[BASELINE] - a_new)
        change_new, lo_new, hi_new = masked_block_bootstrap(diff_new, np.ones(len(a_new)), block=7)
        row = {"what": LABELS[variant], "features": len(features_for_candidate(variant)),
               "A_MAE": float(np.abs(backtest[variant] - a).mean()), "A_change": change, "A_lo": lo, "A_hi": hi}
        for group, days_in in WEEKDAY_GROUPS.items():
            row[f"A_{group}"] = float(diff[np.isin(dow, days_in)].mean())
        row.update({"B_MAE": float(np.abs(fresh[variant] - a_new).mean()), "B_change": change_new, "B_lo": lo_new, "B_hi": hi_new,
                    "supported": bool(hi < 0 and change_new < 0)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "variants.csv", index=False)

    print(f"Part A: {len(a)} backtest days, 100 seeds (200-299). Part B: {len(a_new)} new days "
          f"({new[DATE_COL].min().date()} to {new[DATE_COL].max().date()}), same seeds.\n")
    print("Part A (change against today's 38 columns; negative = better)")
    print(table[["what", "features", "A_MAE", "A_change", "A_lo", "A_hi", "A_Mon-Thu", "A_Fri", "A_Sat", "A_Sun"]].round(2).to_string(index=False))
    print("\nPart B (new days only; interval is wide because there are few days)")
    print(table[["what", "B_MAE", "B_change", "B_lo", "B_hi"]].round(2).to_string(index=False))
    print("\nDECISION RULE (part A below zero with an interval excluding zero, and part B below zero):")
    supported = table[table["supported"] & (table["what"] != LABELS[BASELINE])]["what"].tolist()
    print("  " + ("; ".join(supported) if supported else "no candidate is supported - the final model's columns stay"))
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(CANDIDATES)} variants, {len(tasks)} tasks (backtest chunks of {BACKTEST_CHUNK} seeds, new-days chunks of {NEW_DAYS_CHUNK})")
        for name in CANDIDATES:
            print(f"  {LABELS[name]:62s} {len(features_for_candidate(name))} features")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker46, "series46")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
