"""
Hyperparameter grids and train-window options for the systematic sweep
(src/experiments/run_sweep.py).

These are deliberately modest, not exhaustive - a full grid search is
already multiplied by the number of rolling-origin folds and by the
number of train-window options (see TRAIN_WINDOW_OPTIONS below), so each
per-family grid is kept small enough that the whole sweep finishes in a
practical amount of time. Widen any grid here directly if you want a more
thorough (and slower) search later; run_sweep.py's pilot mode reports a
fit-count/time estimate before committing to a full run either way.
"""

from __future__ import annotations

from itertools import product

# Expanding (None = all available history) vs a 1-year sliding window -
# directly tests this project's own open question (see README's
# "Validation Strategy and Generalisation"): does more historical data
# actually help, or does an older, possibly-stale period hurt more than
# it contributes? Applied uniformly on top of every model family's own
# grid below.
TRAIN_WINDOW_OPTIONS: list[int | None] = [None, 365]


def _grid(**kwargs) -> list[dict]:
    """Cartesian product of named option lists -> list of param dicts."""
    keys = list(kwargs)
    return [dict(zip(keys, combo)) for combo in product(*kwargs.values())]


# Baselines have no real hyperparameters to search - "rolling_mean" is the
# one exception, generalising the existing roll7/roll14/roll28 variants
# into a single family with a `window` parameter.
NAIVE_GRID = [{}]
SEASONAL_NAIVE_GRID = [{"lag_days": 7}]
ROLLING_MEAN_GRID = _grid(window=[7, 14, 21, 28, 45])
WEEKDAY_AVERAGE_GRID = [{}]

SARIMAX_GRID = _grid(
    order=[(1, 1, 1), (2, 1, 1), (0, 1, 1)],
    seasonal_order=[(0, 1, 1, 7), (1, 1, 1, 7)],
)

XGBOOST_GRID = _grid(
    n_estimators=[100, 300],
    learning_rate=[0.03, 0.05, 0.1],
    max_depth=[3, 4, 6],
    subsample=[0.8, 1.0],
    colsample_bytree=[0.8, 1.0],
)

LIGHTGBM_GRID = _grid(
    n_estimators=[100, 300],
    learning_rate=[0.03, 0.05, 0.1],
    max_depth=[3, 4, 6],
    subsample=[0.8, 1.0],
    colsample_bytree=[0.8, 1.0],
)

CATBOOST_GRID = _grid(
    iterations=[100, 300],
    learning_rate=[0.03, 0.05, 0.1],
    depth=[4, 6],
)

PROPHET_GRID = _grid(
    changepoint_prior_scale=[0.01, 0.05, 0.5],
    seasonality_prior_scale=[1.0, 10.0],
    seasonality_mode=["additive", "multiplicative"],
)

# Reduced epochs vs. the production LSTM/Transformer scripts (100) to keep
# the sweep tractable - this trades a little per-fit accuracy for being
# able to search architecture/window choices at all across every fold.
# Whatever wins should be retrained at full epochs before being trusted
# as a final result.
LSTM_GRID = _grid(
    hidden_size=[16, 32],
    num_layers=[1, 2],
    window=[14, 30],
    epochs=[40],
)

TRANSFORMER_GRID = _grid(
    d_model=[16, 32],
    nhead=[2],
    window=[14, 30],
    epochs=[40],
)

MODEL_GRIDS: dict[str, list[dict]] = {
    "naive": NAIVE_GRID,
    "seasonal_naive": SEASONAL_NAIVE_GRID,
    "rolling_mean": ROLLING_MEAN_GRID,
    "weekday_average": WEEKDAY_AVERAGE_GRID,
    "sarimax": SARIMAX_GRID,
    "xgboost": XGBOOST_GRID,
    "lightgbm": LIGHTGBM_GRID,
    "catboost": CATBOOST_GRID,
    "prophet": PROPHET_GRID,
    "lstm": LSTM_GRID,
    "transformer": TRANSFORMER_GRID,
}


def total_planned_fits(n_folds: int, model_names: list[str] | None = None) -> dict[str, int]:
    """Fit count per family for a sweep of `n_folds` folds, given the
    current grids and TRAIN_WINDOW_OPTIONS - used to report an estimate
    before running the full sweep."""
    names = model_names or list(MODEL_GRIDS)
    return {
        name: len(MODEL_GRIDS[name]) * len(TRAIN_WINDOW_OPTIONS) * n_folds
        for name in names
    }
