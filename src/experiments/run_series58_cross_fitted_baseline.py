"""
Series 58 - Series 57's seasonal baseline again, with the baseline cross-fitted so CatBoost sees leftovers as large as on new days.

Series 57 fitted the seasonal baseline on the training days and trained CatBoost on the leftovers of those same days. A baseline always fits the
days it was fitted on better than days it has not seen, so CatBoost learned to correct leftovers that were too small, while on the test days the
baseline's real error (about £1,200 a day on its own) went straight into the forecast. That is a plausible cause of the £99 to £246 a day loss,
not a proven one. Cross-fitting is the standard repair:

  the training days are cut into blocks of 56 days (eight whole weeks, so each block holds every weekday equally), in date order; each block's
  baseline comes from a ridge regression fitted on all the OTHER blocks of that fold's training days, so a block's leftovers are what the baseline
  gets wrong on days it has not seen; the test days' baseline is fitted on all training days, as in Series 57

Everything else is Series 57's (its docstring has the baseline, the four variants and the pieces): wave pairs K = 2 or 4 per weekday, ridge 1.0,
CatBoost with every column or without the six seasonal columns, every piece of the weekly log, post and hydro models, 100 seeds (200-299),
585 days and the 56 new days. Baselines are the same `none` fits.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval, against the deployed model AND against Series 57's in-sample version (does cross-fitting recover any of the loss?);
             where it comes from by weekday and piece; in how many folds it is better; the new days
  expected   it recovers some of the loss but stays worse than the deployed model: the baseline alone is about £1,200 a day, and the trees
             already draw the season better than waves do
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version

Run:  python -m src.experiments.run_series58_cross_fitted_baseline list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series57_seasonal_baseline as s57
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, TARGET_COL, load_protocol, run_parallel

OUT_DIR = CONFIG.path("experiments_dir") / "series58"
RESULTS = OUT_DIR / "series58_cross_fitted_baseline.jsonl"
BLOCK_DAYS = 56
VARIANTS, VARIANT_COLUMNS, piece_features = s57.VARIANTS, s57.VARIANT_COLUMNS, s57.piece_features


def cross_fitted_baseline(train: pd.DataFrame, test: pd.DataFrame, pairs: int, target: str) -> tuple[np.ndarray, np.ndarray]:
    """The baseline for the training days (each block from a fit that never saw it) and for the test days (a fit on every training day)."""
    from sklearn.linear_model import Ridge

    y = np.log1p(train[TARGET_COL].to_numpy()) if target == "log1p" else train[TARGET_COL].to_numpy().astype(float)
    open_days = train[TARGET_COL].to_numpy() > 0
    x_train = s57.design(train, pairs)
    block = ((train[DATE_COL] - train[DATE_COL].min()).dt.days.to_numpy()) // BLOCK_DAYS
    base_train = np.empty(len(train))
    for b in np.unique(block):
        held = block == b
        fit = ~held & open_days
        base_train[held] = Ridge(alpha=s57.RIDGE_ALPHA).fit(x_train[fit], y[fit]).predict(x_train[held])
    base_test = Ridge(alpha=s57.RIDGE_ALPHA).fit(x_train[open_days], y[open_days]).predict(s57.design(test, pairs))
    return base_train, base_test


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    pairs = VARIANTS[task["variant"]][0]
    per_seed = s57.residual_predictions(df, folds, task["features"], s54.PIECES[task["piece"]][1], pairs, task["seeds"], split=task["part"],
                                        baseline_fn=cross_fitted_baseline)
    return {"id": task["id"], "part": task["part"], "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def report() -> None:
    s54.report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=s54.RESULTS)


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = s57.make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS) - 1} variants, {len(tasks)} tasks")
        df, folds = load_protocol()
        for pairs in (2, 4):
            for fold in (folds[0], folds[4], folds[-1]):
                train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
                test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
                a_train = train[TARGET_COL].to_numpy()
                open_days = a_train > 0
                insample, _ = s57.seasonal_baseline(train, test, pairs, "log1p")
                cross, base_test = cross_fitted_baseline(train, test, pairs, "log1p")
                err = lambda base, actual, mask=None: float(np.abs(np.expm1(base) - actual)[mask if mask is not None else slice(None)].mean())  # noqa: E731
                print(f"  k{pairs} fold {fold.fold_id}: {len(train)} train days in {len(np.unique(((train[DATE_COL] - train[DATE_COL].min()).dt.days) // BLOCK_DAYS))} blocks; "
                      f"baseline error on the training days in-sample {err(insample, a_train, open_days):.0f}, cross-fitted {err(cross, a_train, open_days):.0f}; "
                      f"on the test days {err(base_test, test[TARGET_COL].to_numpy()):.0f}")
    elif step == "run":
        run_parallel(s57.make_tasks(), fit_task, RESULTS, workers, s57.init_worker57, "series58")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
