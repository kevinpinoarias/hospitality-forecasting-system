"""
Series 50 - what does dropping `year` cost or save inside the real post-model pieces?

Series 45 and 46 dropped `year` from the plain log model: Saturday got much better, Monday-Thursday a little worse, and nearly all of the
average gain came from one fold (10 January to 10 March 2026), whose training data held only nine 2026 rows, all low-sales early-January days,
so `year = 2026` acted as a "quiet day" marker. Series 47 showed the plain log model is not what the post model uses on Friday, Saturday and
Sunday. This series drops `year` from each piece the post model really uses, everything else as deployed (`is_weekend` stays):

  Monday-Thursday   the plain log model, log1p target                          (src/forecast_log/v1.py: "log")
  Friday            log1p target + the Friday inputs (A_all)
  Saturday          raw-pound target + the Saturday inputs (A_lags)
  Sunday            log1p target + the Sunday inputs (B)

The "year kept" side is not refitted: it is the 100-seed bags already produced with identical settings and seeds (Series 47 for the three weekend
pieces, Series 46's `final_model` for Monday-Thursday). Those fits reproduced one another exactly in Series 48, so the comparison is like for like.

PRE-REGISTERED (written before any result was seen)
  protocol   as Series 47: 585 backtest days, 10 folds, 100 seeds (200-299), the new days from 7 September 2026 as a veto, the package put
             through the Series 24 correction (half life 8 weeks, strength 0.5, Saturday and Sunday)
  reading    per piece on its own weekday: change from dropping `year` with a 95% moving-block bootstrap interval, folds (of 10) where it is
             better, the change with the Jan-Mar 2026 fold (fold 6) left out, and the new-days change; the package before and after the
             correction, with and without fold 6
  decision   AMENDED at Kevin's request after the run finished and BEFORE any result was read. The first version made three pass/fail gates
             (interval below zero, still below zero without fold 6, below zero on the new days); the printed DECISION RULE line still shows
             them and is a reference only. The reading is by judgment on: (1) the package's net change in pounds with its interval, over
             the weekday-by-weekday picture, since a loss on one weekday against a larger gain on another is a gain; (2) whether the change
             is spread over many folds or sits in one (fold 6); (3) whether there is a plausible mechanism; (4) the cost in complexity or
             risk. Every weekday is still reported. Dropping `year` on the rollover-risk grounds of Series 46 is Kevin's call, made with
             this result in hand; the deployed v1 models are never refitted.

Run:  python -m src.experiments.run_series50_drop_year_in_pieces list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS, load_bags as load_log_model_bags
from src.experiments.series19_21_common import (
    _STATE, DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)

BACKTEST_CHUNK, NEW_DAYS_CHUNK = 5, 10
OUT_DIR = CONFIG.path("experiments_dir") / "series50"
RESULTS = OUT_DIR / "series50_drop_year_in_pieces.jsonl"
FOLD_6 = 6  # 10 January to 10 March 2026
# piece -> (extra inputs, target, weekdays it covers)
PIECES = {"Monday-Thursday": ([], "log1p", [0, 1, 2, 3]), **{p: (extra, target, [day]) for p, (extra, target, day) in s47.PIECES.items()}}


def piece_features(piece: str) -> list[str]:
    return [f for f in FINAL_FEATURES if f != "year"] + PIECES[piece][0]


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], PIECES[task["piece"]][1]
    if task["part"] == "backtest":
        if target == "log1p":
            per_seed = bagged_log1p_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
        else:
            per_seed = s23.raw_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
    else:
        from catboost import CatBoostRegressor

        train, test = df[df[DATE_COL] < CUTOFF], df[df[DATE_COL] >= CUTOFF]
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        rows = []
        for seed in task["seeds"]:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y)
            p = model.predict(test[features])
            rows.append(np.expm1(p) if target == "log1p" else p)
        per_seed = np.array(rows)
    return {"id": task["id"], "part": task["part"], "piece": task["piece"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        features = piece_features(piece)
        for part, size in (("backtest", BACKTEST_CHUNK), ("new_days", NEW_DAYS_CHUNK)):
            for i in range(0, len(SEEDS), size):
                chunk = SEEDS[i:i + size]
                out.append({"id": f"{part}_{piece}_dropped_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece,
                            "seeds": chunk, "features": features})
    return out


def load_dropped(part: str) -> dict[str, np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for piece in PIECES:
        per_seed = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["part"] == part and r["piece"] == piece], axis=0)
        assert per_seed.shape[0] == len(SEEDS), (part, piece, per_seed.shape)
        out[piece] = per_seed.mean(axis=0)
    return out


def load_kept(part: str) -> dict[str, np.ndarray]:
    """The year-kept side: Series 47's as-deployed weekend pieces and Series 46's final model for Monday-Thursday."""
    weekend = s47.load_bags(part)
    return {"Monday-Thursday": load_log_model_bags(part)["final_model"], **{p: weekend[(p, "as_today")] for p in s47.PIECES}}


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    kept, dropped = load_kept("backtest"), load_dropped("backtest")
    kept_new, dropped_new = load_kept("new_days"), load_dropped("new_days")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def interval(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        return s23.masked_block_bootstrap(diff, mask.astype(float))

    rows, supported = [], []
    for piece, (_, _, weekdays) in PIECES.items():
        own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
        diff = np.abs(dropped[piece] - a) - np.abs(kept[piece] - a)
        change, lo, hi = interval(diff, own)
        change_no6, lo6, hi6 = interval(diff, own & (fold != FOLD_6))
        better = sum(diff[own & (fold == k)].mean() < 0 for k in range(10))
        diff_new = np.abs(dropped_new[piece] - a_new) - np.abs(kept_new[piece] - a_new)
        change_new = float(diff_new[own_new].mean())
        ok = bool(hi < 0 and hi6 < 0 and change_new < 0)
        supported.append(piece) if ok else None
        rows.append({"piece": piece, "days": int(own.sum()), "MAE_year_kept": float(np.abs(kept[piece] - a)[own].mean()),
                     "MAE_year_dropped": float(np.abs(dropped[piece] - a)[own].mean()), "change": change, "lo": lo, "hi": hi,
                     "folds_better": int(better), "change_without_fold6": change_no6, "lo_no6": lo6, "hi_no6": hi6,
                     "new_days_n": int(own_new.sum()), "new_days_change": change_new, "supported": ok})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "pieces.csv", index=False)
    print("Each piece on its own weekdays, 585 backtest days, 100 seeds (change = year dropped minus year kept; negative = dropping helps)")
    print(table.round(2).to_string(index=False))

    def package(version: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pred = np.select([dow == 4, dow == 5, dow == 6], [version["Friday"], version["Saturday"], version["Sunday"]], default=version["Monday-Thursday"])
        return pred, s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    pk_kept, pk_dropped = package(kept), package(dropped)
    out = []
    for step, idx in (("before the correction", 0), ("after the Saturday and Sunday correction", 1)):
        diff = np.abs(pk_dropped[idx] - a) - np.abs(pk_kept[idx] - a)
        change, lo, hi = interval(diff, np.ones(len(a), bool))
        change6, lo6, hi6 = interval(diff, fold != FOLD_6)
        out.append({"step": step, "MAE_year_kept": float(np.abs(pk_kept[idx] - a).mean()), "MAE_year_dropped": float(np.abs(pk_dropped[idx] - a).mean()),
                    "change": change, "lo": lo, "hi": hi, "change_without_fold6": change6, "lo_no6": lo6, "hi_no6": hi6,
                    "Mon-Thu": float(diff[dow < 4].mean()), "Fri": float(diff[dow == 4].mean()),
                    "Sat": float(diff[dow == 5].mean()), "Sun": float(diff[dow == 6].mean())})
    pkg = pd.DataFrame(out)
    pkg.to_csv(OUT_DIR / "package.csv", index=False)
    print("\nThe package, all 585 days")
    print(pkg.round(2).to_string(index=False))
    diff = np.abs(pk_dropped[1] - a) - np.abs(pk_kept[1] - a)
    by_fold = pd.DataFrame({"fold": range(10), "from": [days.loc[fold == k, "date"].iloc[0] for k in range(10)],
                            "change_after_correction": [diff[fold == k].mean() for k in range(10)]})
    print("\nPackage after the correction, change by fold")
    print(by_fold.round(1).to_string(index=False))

    package_ok = bool(pkg.iloc[1]["hi"] < 0 and pkg.iloc[1]["hi_no6"] < 0)
    print("\nDECISION RULE: pieces where dropping year is supported as a gain:", ", ".join(supported) if supported else "none",
          "| package after the correction supported:", "yes" if package_ok else "no")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces, year dropped, {len(tasks)} tasks")
        for p in PIECES:
            print(f"  {p:16s} {len(piece_features(p))} features, target {PIECES[p][1]}")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, s47.init_worker47, "series50")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
