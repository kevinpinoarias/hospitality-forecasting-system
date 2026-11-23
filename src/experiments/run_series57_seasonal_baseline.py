"""
Series 57 - give the model a smooth seasonal baseline and let CatBoost learn only what is left over.

CatBoost draws the year's seasonal curve from `day_of_year` (and its sine and cosine) with decision trees: yes/no cuts, each tree four questions deep,
300 trees added together. That builds a smooth curve as a staircase. Series 44-56 found nothing to add to the calendar columns, so this tests the
representation instead: fit the season by ordinary arithmetic first, and give CatBoost the leftovers.

The baseline, refitted on each fold's training days only (so nothing from the test days leaks in):

  a ridge regression of the piece's target (log1p of sales for the log pieces, pounds for the raw Saturday pieces) on
  a level for each weekday, plus, for each weekday separately, K pairs of sine and cosine waves of the day of the year (periods 365.25 / k days).
  "A typical Friday in week 30 sells about this much", with its own curve for each of the seven weekdays. The four closure days with zero sales are
  left out of the baseline's fit. Ridge strength 1.0 (mild: the sine and cosine columns have a sum of squares of about half the rows of a weekday).

CatBoost is then trained on the target minus the baseline, with the piece's own columns, and the forecast is the baseline plus CatBoost's output
(the log pieces invert with expm1 as before). Four variants, written beforehand, none chosen after a result:

  none                    today's piece
  residual_k2             baseline with 2 wave pairs per weekday; CatBoost keeps every column
  residual_k4             baseline with 4 wave pairs per weekday; CatBoost keeps every column
  residual_k2_no_season   the k2 baseline, and CatBoost loses `day_of_year`, its sine and cosine, `month` and its sine and cosine
  residual_k4_no_season   the same with the k4 baseline

The last two are the pure form of the idea: the season is handled outside the trees, so the trees need no seasonal columns. Every variant goes into
every piece of the weekly log, exactly as in Series 54-56 (Series 54's docstring has the pieces, the post and hydro models, the correction and the
baselines; the hydro and Saturday-fallback baselines are Series 54's `none` fits).

Known limits of the design: the baseline is fitted in-sample on the training days, so its training residuals are a little smaller than its errors on
new days; the first fold has one year of training days, so the k4 baseline (9 coefficients per weekday on about 52 rows) is the loosest.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece; in how many folds it is better; the new days
  expected   little or nothing: the 300 trees together already approximate a smooth curve, and Series 44-56 found the calendar columns saturated.
             If anything, the no-season variants should be level (they test whether the trees need the columns at all)
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version

Run:  python -m src.experiments.run_series57_seasonal_baseline list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, FINAL_PARAMS, TARGET_COL, run_parallel
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series57"
RESULTS = OUT_DIR / "series57_seasonal_baseline.jsonl"
SEASON_COLUMNS = ["day_of_year", "day_of_year_sin", "day_of_year_cos", "month", "month_sin", "month_cos"]
RIDGE_ALPHA = 1.0
# variant -> (wave pairs per weekday, drop the seasonal columns from CatBoost)
VARIANTS: dict[str, tuple[int, bool] | None] = {
    "none": None,
    "residual_k2": (2, False),
    "residual_k4": (4, False),
    "residual_k2_no_season": (2, True),
    "residual_k4_no_season": (4, True),
}
VARIANT_COLUMNS = {name: [] for name in VARIANTS}  # the shared report only needs the names


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    if VARIANTS[variant] and VARIANTS[variant][1]:
        cols = [c for c in cols if c not in SEASON_COLUMNS]
    return cols


def design(frame: pd.DataFrame, pairs: int) -> np.ndarray:
    """Weekday levels, and for each weekday its own sine and cosine waves of the day of the year."""
    weekday = frame[DATE_COL].dt.dayofweek.to_numpy()
    angle = 2 * np.pi * frame[DATE_COL].dt.dayofyear.to_numpy() / 365.25
    levels = np.eye(7)[weekday]
    waves = [f(k * angle)[:, None] * levels for k in range(1, pairs + 1) for f in (np.sin, np.cos)]
    return np.hstack([levels] + waves)


def seasonal_baseline(train: pd.DataFrame, test: pd.DataFrame, pairs: int, target: str) -> tuple[np.ndarray, np.ndarray]:
    """The baseline for the training days and the test days, in the target's scale (log1p of sales, or pounds)."""
    from sklearn.linear_model import Ridge

    y = np.log1p(train[TARGET_COL].to_numpy()) if target == "log1p" else train[TARGET_COL].to_numpy().astype(float)
    open_days = train[TARGET_COL].to_numpy() > 0
    model = Ridge(alpha=RIDGE_ALPHA).fit(design(train, pairs)[open_days], y[open_days])
    return model.predict(design(train, pairs)), model.predict(design(test, pairs))


def residual_predictions(df: pd.DataFrame, folds: list, features: list[str], target: str, pairs: int, seeds: list[int],
                         split: str | None = None, baseline_fn=seasonal_baseline) -> np.ndarray:
    """One row per seed, test days in protocol order (backtest folds), or the days from the cutoff (new_days). `baseline_fn(train, test, pairs,
    target)` returns the baseline for the training and the test days (Series 58 swaps in a cross-fitted one)."""
    from catboost import CatBoostRegressor

    if split == "new_days":
        splits = [(df[df[DATE_COL] < CUTOFF], df[df[DATE_COL] >= CUTOFF])]
    else:
        splits = [(df[(df[DATE_COL] >= f.train_start) & (df[DATE_COL] < f.train_end)],
                   df[(df[DATE_COL] >= f.test_start) & (df[DATE_COL] < f.test_end)]) for f in folds]
    out = []
    for train, test in splits:
        base_train, base_test = baseline_fn(train, test, pairs, target)
        y = np.log1p(train[TARGET_COL].to_numpy()) if target == "log1p" else train[TARGET_COL].to_numpy().astype(float)
        rows = []
        for seed in seeds:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y - base_train)
            p = base_test + model.predict(test[features])
            rows.append(np.expm1(p) if target == "log1p" else p)
        out.append(np.array(rows))
    return np.concatenate(out, axis=1)


def init_worker57() -> None:
    s47.init_worker47()
    _STATE["df"] = add_hydro_thu(_STATE["df"])


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    pairs = VARIANTS[task["variant"]][0]
    per_seed = residual_predictions(df, folds, task["features"], s54.PIECES[task["piece"]][1], pairs, task["seeds"], split=task["part"])
    return {"id": task["id"], "part": task["part"], "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


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
            print(f"  {piece:26s} baseline {len(piece_features(piece, 'none'))} columns, no-season {len(piece_features(piece, 'residual_k2_no_season'))}")
        df, folds = _load()
        for pairs in (2, 4):
            show = []
            for fold in (folds[0], folds[-1]):
                train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
                test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
                bt, bs = seasonal_baseline(train, test, pairs, "log1p")
                open_days = train[TARGET_COL].to_numpy() > 0
                show.append(f"fold {fold.fold_id}: {len(train)} train days, baseline alone MAE on its own training days "
                            f"{np.abs(np.expm1(bt) - train[TARGET_COL].to_numpy())[open_days].mean():.0f}, on the test days "
                            f"{np.abs(np.expm1(bs) - test[TARGET_COL].to_numpy()).mean():.0f}")
            print(f"  k{pairs}: " + "; ".join(show))
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker57, "series57")
        report()
    else:
        report()


def _load():
    from src.experiments.series19_21_common import load_protocol

    return load_protocol()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
