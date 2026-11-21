"""
Series 53 - the Friday question as the weekly log really asks it.

The weekly log issues next week's forecast on a Monday, from the data known on the Sunday before. A Friday of that week is 12 days after the last known
day, so the Friday piece (A_all, which needs the sales 7, 14, 21 and 28 days earlier) cannot run and the log uses the plain log model (`route` in
`run_series43_post_model_horizon.py`). On that model, with every input available, replacing `is_weekend` by `is_friday` / `is_saturday` / `is_sunday`
lowered the Friday error by about 27 and 36 pounds on two seed sets (Series 46, 48, 52). This series asks whether it still does in the situation the
log is in: a replay of the weekly log over the backtest.

For each Friday in the 585 backtest days, the Monday 11 days before is the issue day and the Sunday before it the last known day. From the data up to
that Sunday the rows for that week are built the way the live API builds a date beyond its data (`served_rows` in `run_series35_holdout.py`): the days
after it get weekday-and-season medians for sales, the manager's forecast and the weather. Two plain log models, fitted on the history before the
backtest block, forecast the week:

  today        the final model's 38 columns, with `is_weekend`
  flags        the same, with `is_friday`, `is_saturday`, `is_sunday` in its place (same position, same seeds)

and Monday to Friday of that week (the days the log gets from the plain model) are scored against what sold. 25 seeds (0-24) and 25 fresh seeds
(100-124), 10 folds as the backtest, the model refitted once per fold on all history before the fold (the deployed log is fitted once; the replay
refits per fold only so that no forecast sees its own future).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  look at    the Friday change (flags minus today, with a 95% moving-block bootstrap interval) on both seed sets; Monday to Thursday, since the log
             also takes those from this model; Monday to Friday together; the number of folds in which Fridays improve; the plain model's Friday
             error here against its error with every input available (Series 46), to confirm the replay really is harder
  limits     about 75 Fridays, so the interval is wide; the weather is the median fallback for every day, harsher than the log's real forecast for the
             first 16 days; the model is refitted per fold, unlike the fitted-once log. No pass/fail gate. A gain would be a candidate for a new model
             version (the deployed v1 pieces are never refitted); that decision is Kevin's.

Run:  python -m src.experiments.run_series53_friday_as_the_log_serves_it list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series35_holdout as h
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.run_series45_year_and_weekend_flag import add_weekday_flags, variant_features as flag_features
from src.experiments.series19_21_common import _STATE, DATE_COL, FINAL_FEATURES, TARGET_COL, load_protocol, read_done, run_parallel

SEED_SETS = {"seeds0_24": list(range(25)), "seeds100_124": list(range(100, 125))}
VERSIONS = {"today": list(FINAL_FEATURES), "flags": flag_features("kept", "single_days")}
OUT_DIR = CONFIG.path("experiments_dir") / "series53"
RESULTS = OUT_DIR / "series53_friday_as_the_log_serves_it.jsonl"
ISSUE_TO_FRIDAY = 11  # days from the issue Monday to the Friday of the following week
MON_FRI = [0, 1, 2, 3, 4]


def init_worker53() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = add_weekday_flags(df), folds
    _STATE["full"] = h.load_engineered(h.frozen_features_path())


def fridays_in_fold(df: pd.DataFrame, fold) -> list[pd.Timestamp]:
    test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
    return [d for d in test[DATE_COL] if d.dayofweek == 4]


def replay_fold(task: dict) -> dict:
    df, folds, full = _STATE["df"], _STATE["folds"], _STATE["full"]
    fold = folds[task["fold"]]
    started = time.time()
    train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
    test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].set_index(DATE_COL)
    bags = {(v, s): h.Bag.fit(train, VERSIONS[v], "log1p", seeds, thread_count=1) for v in VERSIONS for s, seeds in SEED_SETS.items()}
    rows = []
    for friday in fridays_in_fold(df, fold):
        issue = friday - pd.Timedelta(days=ISSUE_TO_FRIDAY)
        last_known = issue - pd.Timedelta(days=1)
        week = [issue + pd.Timedelta(days=7 + i) for i in range(7)]
        served = add_weekday_flags(h.served_rows(full, last_known, week, "v2", None, False))
        preds = {key: bag.predict(served) for key, bag in bags.items()}
        for i, day in enumerate(week):
            if day.dayofweek in MON_FRI and day in test.index:
                rows.append({"fold": task["fold"], "issue": str(issue.date()), "date": str(day.date()), "weekday": int(day.dayofweek),
                             "actual": float(test.loc[day, TARGET_COL]),
                             **{f"{v}__{s}": round(float(preds[(v, s)][i]), 2) for (v, s) in preds}})
    return {"id": task["id"], "fold": task["fold"], "rows": rows, "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    return [{"id": f"fold{k}", "fold": k} for k in range(10)]


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    done = read_done(RESULTS)
    table = pd.DataFrame([r for rec in done.values() for r in rec["rows"]]).sort_values("date").reset_index(drop=True)
    table = table.drop_duplicates(subset="date")  # a week cut by two blocks is scored once, by the block holding the Friday
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "replay_days.csv", index=False)
    a, dow, fold = table["actual"].to_numpy(), table["weekday"].to_numpy(), table["fold"].to_numpy()
    fridays = dow == 4
    print(f"{len(table)} days scored, of which {int(fridays.sum())} Fridays, from {table['date'].min()} to {table['date'].max()}\n")

    groups = {"Friday": fridays, "Monday-Thursday": dow < 4, "Monday-Friday": np.ones(len(a), bool)}
    rows = []
    for s in SEED_SETS:
        today, flags = table[f"today__{s}"].to_numpy(), table[f"flags__{s}"].to_numpy()
        diff = np.abs(flags - a) - np.abs(today - a)
        for name, mask in groups.items():
            change, lo, hi = masked_block_bootstrap(diff, mask.astype(float), block=10)
            rows.append({"seeds": s, "days": name, "n": int(mask.sum()), "MAE_today": float(np.abs(today - a)[mask].mean()),
                         "MAE_flags": float(np.abs(flags - a)[mask].mean()), "change": change, "lo": lo, "hi": hi,
                         "Fridays_better_in_folds": int(sum(diff[mask & (fold == k)].mean() < 0 for k in range(10) if (mask & (fold == k)).any()))
                         if name == "Friday" else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "friday_replay.csv", index=False)
    print("The plain log model as the weekly log serves it: flags minus today (negative = the flags help)")
    print(out.round(2).to_string(index=False))
    print("\nBy weekday (change, pounds):")
    for s in SEED_SETS:
        diff = np.abs(table[f"flags__{s}"].to_numpy() - a) - np.abs(table[f"today__{s}"].to_numpy() - a)
        print(" ", s, {d: round(float(diff[dow == k].mean()), 1) for k, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])})
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        df, folds = load_protocol()
        counts = [len(fridays_in_fold(df, f)) for f in folds]
        print(f"{len(folds)} folds, Fridays per fold {counts}, {sum(counts)} replayed weeks; features: today {len(VERSIONS['today'])}, flags {len(VERSIONS['flags'])}")
    elif step == "run":
        run_parallel(make_tasks(), replay_fold, RESULTS, workers, init_worker53, "series53")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
