"""
Series 47 - do the single-day flags still help inside the pieces the post model really uses on Friday, Saturday and Sunday?

Series 45 and 46 tested `is_friday` / `is_saturday` / `is_sunday` in place of `is_weekend` on the plain log model. The deployed post model
(src/forecast_log/v1.py) does not use that model on those days. It uses one piece per weekend day, then a Saturday and Sunday correction:

  Friday     log1p target, the final model's 38 columns + same-weekday memory (A_all: lag_14, lag_21, mean and max of the last 4)
  Saturday   raw-pound target, the 38 + the sales of 2 and 3 weeks ago (A_lags)
  Sunday     log1p target, the 38 + the weekend interactions (B)
  Mon-Thu    the plain log model (already tested in Series 46)

Each piece is refitted twice, once as the post model has it and once with `is_weekend` replaced by the three single-day flags in the same
place in the column order, and scored on its own weekday over the 585 backtest days (100 fresh seeds, 200-299, as in Series 46) and on its
weekday among the 56 days from 7 September to 1 November 2026. The package (Monday-Thursday from the log model, the three pieces on their
days) is then assembled both ways and put through the Series 24 correction (half life 8 weeks, strength 0.5, Saturday and Sunday), exactly
as the validated package does.

PRE-REGISTERED (written before any result was seen)
  H47        the flags lower the MAE of the piece on its own weekday, and of the package after the correction
  reading    per piece: MAE on its weekday, change with a 95% moving-block bootstrap interval, folds (of 10) where it is better;
             the package before and after the correction, with the same interval; the new-days change per piece
  decision   a piece is SUPPORTED only if its backtest change is below zero with an interval that excludes zero AND its change on the new
             days is below zero (the new days are a veto: 8 days a weekday cannot confirm anything). The package is supported only if the
             change after the correction is below zero with an interval that excludes zero. Not supported means the post model's pieces
             stay as they are. The deployed forecast-log models (v1) are fitted once and never refitted; adopting the flags would be a
             new version, and that is Kevin's decision, not this series'.
  caveat     the flags were chosen after seeing Series 45 and 46 on the same 585 days. The new-days score uses the lag inputs as actually
             observed, as the operational mode of Series 35 does, not as the API would fill them weeks ahead.

Run:  python -m src.experiments.run_series47_weekend_flags_in_pieces list
      python -m src.experiments.run_series47_weekend_flags_in_pieces run [--workers 10]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series47_weekend_flags_in_pieces report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series26_weekend_features as s26
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series45_year_and_weekend_flag import add_weekday_flags, variant_features as flag_features
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS, load_bags as load_log_model_bags
from src.experiments.series19_21_common import (
    _STATE, DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)

BACKTEST_CHUNK, NEW_DAYS_CHUNK = 5, 10
OUT_DIR = CONFIG.path("experiments_dir") / "series47"
RESULTS = OUT_DIR / "series47_weekend_flags_in_pieces.jsonl"
# piece -> (extra inputs on top of the 38, target, weekday number): the pieces of src/forecast_log/v1.py without the Hydro model
PIECES = {
    "Friday": (s26.VARIANTS["A_all"]["extra"], "log1p", 4),
    "Saturday": (s26.VARIANTS["A_lags"]["extra"], "raw", 5),
    "Sunday": (s26.VARIANTS["B"]["extra"], "log1p", 6),
}
VERSIONS = ("as_today", "single_day_flags")


def piece_features(piece: str, version: str) -> list[str]:
    return flag_features("kept", "as_today" if version == "as_today" else "single_days") + PIECES[piece][0]


assert flag_features("kept", "as_today") == list(FINAL_FEATURES)


# --------------------------------------------------------------------------- fitting


def init_worker47() -> None:
    df, folds = s26.load_frame()
    _STATE["df"], _STATE["folds"] = add_weekday_flags(df), folds


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
    return {"id": task["id"], "part": task["part"], "piece": task["piece"], "version": task["version"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        for version in VERSIONS:
            features = piece_features(piece, version)
            for part, chunk_size in (("backtest", BACKTEST_CHUNK), ("new_days", NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), chunk_size):
                    chunk = SEEDS[i:i + chunk_size]
                    out.append({"id": f"{part}_{piece}_{version}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece,
                                "version": version, "seeds": chunk, "features": features})
    return out


def load_bags(part: str) -> dict[tuple[str, str], np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for piece in PIECES:
        for version in VERSIONS:
            rows = [np.array(r["per_seed"]) for r in done.values() if r["part"] == part and r["piece"] == piece and r["version"] == version]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS), (part, piece, version, per_seed.shape)
            out[(piece, version)] = per_seed.mean(axis=0)
    return out


# --------------------------------------------------------------------------- reading


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    bt, nd = load_bags("backtest"), load_bags("new_days")
    log_bt, log_nd = load_log_model_bags("backtest"), load_log_model_bags("new_days")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, supported = [], []
    for piece, (_, _, day) in PIECES.items():
        own, own_new = dow == day, dow_new == day
        b, f = bt[(piece, "as_today")], bt[(piece, "single_day_flags")]
        diff = np.abs(f - a) - np.abs(b - a)
        change, lo, hi = s23.masked_block_bootstrap(diff, own.astype(float))
        better = sum(diff[own & (fold == k)].mean() < 0 for k in range(10))
        diff_new = np.abs(nd[(piece, "single_day_flags")] - a_new) - np.abs(nd[(piece, "as_today")] - a_new)
        change_new = float(diff_new[own_new].mean())
        ok = bool(hi < 0 and change_new < 0)
        supported.append(piece) if ok else None
        rows.append({"piece": piece, "days": int(own.sum()), "MAE_as_today": float(np.abs(b - a)[own].mean()),
                     "MAE_flags": float(np.abs(f - a)[own].mean()), "change": change, "lo": lo, "hi": hi, "folds_better": int(better),
                     "new_days_n": int(own_new.sum()), "new_days_change": change_new, "supported": ok})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "pieces.csv", index=False)
    print("Pieces on their own weekday, 585 backtest days, 100 seeds (change = flags minus as-today; negative = better)")
    print(table.round(2).to_string(index=False))

    wd_names = days["weekday"].to_numpy()
    packages, steps = {}, {}
    for version in VERSIONS:
        log = log_bt["final_model" if version == "as_today" else "single_day_flags"]
        pred = np.select([dow == 4, dow == 5, dow == 6],
                         [bt[("Friday", version)], bt[("Saturday", version)], bt[("Sunday", version)]], default=log)
        packages[version] = (pred, s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"]))
    print("\nThe package (Monday-Thursday from the log model, the three pieces on their days), all 585 days")
    out = []
    for step, idx in (("before the correction", 0), ("after the Saturday and Sunday correction", 1)):
        base, flags = packages["as_today"][idx], packages["single_day_flags"][idx]
        diff = np.abs(flags - a) - np.abs(base - a)
        change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
        out.append({"step": step, "MAE_as_today": float(np.abs(base - a).mean()), "MAE_flags": float(np.abs(flags - a).mean()),
                    "change": change, "lo": lo, "hi": hi, "Fri": float(diff[dow == 4].mean()), "Sat": float(diff[dow == 5].mean()),
                    "Sun": float(diff[dow == 6].mean()), "Mon-Thu": float(diff[dow < 4].mean())})
    package = pd.DataFrame(out)
    package.to_csv(OUT_DIR / "package.csv", index=False)
    print(package.round(2).to_string(index=False))

    package_ok = bool(package.iloc[1]["hi"] < 0)
    print("\nDECISION RULE: pieces supported:", ", ".join(supported) if supported else "none",
          "| package after the correction supported:", "yes" if package_ok else "no")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces x {len(VERSIONS)} versions, {len(tasks)} tasks")
        for piece in PIECES:
            for version in VERSIONS:
                print(f"  {piece:9s} {version:17s} {len(piece_features(piece, version))} features, target {PIECES[piece][1]}")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker47, "series47")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
