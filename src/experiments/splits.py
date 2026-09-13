"""
Rolling-origin (walk-forward) train/test split generation for the
systematic model-comparison sweep (src/experiments/run_sweep.py).

Why rolling-origin, not one fixed split
----------------------------------------
The existing pipeline (src/models/train_xgboost.py etc.) evaluates each
model on a single fixed train/test cutoff. With ~2.5 years of data now
available, one cutoff is only one snapshot of how a model performs - and
this project has already found that performance on one period doesn't
guarantee performance on another (see README's "Validation Strategy and
Generalisation" section on 2026 drift). Rolling-origin backtesting trains
on a window of history, tests on the following block, then rolls the
whole thing forward repeatedly - producing many independent (train, test)
snapshots across the full date range rather than one.

Each fold also supports a `train_window_days` option: None keeps every
day of history before the fold's test period (expanding window, more
data every fold); an integer restricts training to only the most recent
N days before the test period (sliding window, fixed-size). Comparing
these directly tests the project's own open question from its README -
"does more historical data actually help, or does an older, possibly
stale period hurt more than it contributes?" - rather than assuming one
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp  # == train_end
    test_end: pd.Timestamp  # exclusive


def generate_rolling_origin_folds(
    dates: pd.Series,
    initial_train_days: int,
    test_days: int,
    step_days: int | None = None,
    max_folds: int | None = None,
) -> list[Fold]:
    """
    Generate rolling-origin fold boundaries covering the full date range.

    Fold 0 trains on [min_date, min_date + initial_train_days) and tests
    on the following `test_days`. Each subsequent fold's test window
    starts `step_days` after the previous one's (default: back-to-back,
    non-overlapping test windows). A fold is only included if its full
    test window fits within the available data - a partial trailing
    window is dropped rather than silently evaluated on fewer days than
    every other fold.
    """
    if step_days is None:
        step_days = test_days

    dates = pd.to_datetime(pd.Series(dates)).sort_values()
    min_date = dates.min().normalize()
    max_date_exclusive = dates.max().normalize() + pd.Timedelta(days=1)

    folds: list[Fold] = []
    test_start = min_date + pd.Timedelta(days=initial_train_days)

    fold_id = 0
    while test_start + pd.Timedelta(days=test_days) <= max_date_exclusive:
        test_end = test_start + pd.Timedelta(days=test_days)
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=min_date,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        test_start = test_start + pd.Timedelta(days=step_days)

        if max_folds is not None and len(folds) >= max_folds:
            break

    return folds


def slice_train_window(
    df: pd.DataFrame,
    date_col: str,
    fold: Fold,
    train_window_days: int | None,
) -> pd.DataFrame:
    """
    Return the training rows for one fold: every day of history before
    the test period if `train_window_days` is None (expanding window), or
    only the most recent `train_window_days` days before it otherwise
    (sliding window).
    """
    train_start = fold.train_start
    if train_window_days is not None:
        candidate_start = fold.train_end - pd.Timedelta(days=train_window_days)
        if candidate_start > train_start:
            train_start = candidate_start

    mask = (df[date_col] >= train_start) & (df[date_col] < fold.train_end)
    return df.loc[mask]


def slice_test_window(df: pd.DataFrame, date_col: str, fold: Fold) -> pd.DataFrame:
    """Return the test rows for one fold."""
    mask = (df[date_col] >= fold.test_start) & (df[date_col] < fold.test_end)
    return df.loc[mask]
