"""
Per-algorithm-family fit/predict adapters for the systematic sweep
(src/experiments/run_sweep.py).

Every adapter has the same signature and return shape - a DataFrame with
columns [date, actual, pred] covering whatever it actually predicted for
one fold - regardless of how differently each underlying library wants
its data shaped. This uniform return type (rather than assuming
predictions align positionally with the test set) matters most for the
neural adapters: gap-aware windowing (see build_windows in
src/models/neural_common.py) can legitimately drop a day or two right at
a fold boundary, so scoring is always done by joining on date, never by
position.

Baseline families (naive/seasonal_naive/rolling_mean) deliberately
compute their shift/rolling statistic over the *combined* train+test
frame, not train alone - matching src/baselines/run_baselines.py's
existing behaviour and the realistic assumption that a naive forecast at
any point in production knows the real outcome of every prior day,
including earlier days within the current test window.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Harmless per-fold noise: boolean-masking a DatetimeIndex to drop a rare
# gap day (see fit_predict_sarimax) correctly clears its `freq` label
# since it's no longer perfectly regular in general, and MLE convergence
# warnings are expected background noise across hundreds of grid points,
# not something worth surfacing per-fit in a sweep this size.
warnings.filterwarnings("ignore", module="statsmodels")

DATE_COL = "date"
TARGET_COL = "total_sales"

# Same final feature set the production XGBoost/LSTM/Transformer models
# use (see src/features/build_features.py's FINAL_MODEL_FEATURES) -
# redefined locally, matching this codebase's existing convention of each
# script owning its own copy rather than a shared config import (a known,
# already-documented gap - see README's Future Work).
FEATURES = [
    "forecast_sales",
    "month_sin",
    "day_of_week",
    "day_of_year_cos",
    "day_of_year_sin",
    "day_of_year",
    "month_cos",
    "month",
    "day_of_week_sin",
    "is_bank_holiday",
    "days_to_bank_holiday",
    "days_since_payday",
    "is_payday_window_pm3",
    "is_long_weekend",
    "is_heavy_rain",
]

PROPHET_REGRESSORS = [
    "forecast_sales",
    "is_bank_holiday",
    "is_heavy_rain",
    "is_long_weekend",
    "is_payday_window_pm3",
]

SARIMAX_EXOG_FEATURES = ["forecast_sales", "lag_7_sales"]


def _result(dates: pd.Series, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({DATE_COL: pd.to_datetime(dates).to_numpy(), "actual": actual, "pred": pred})


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def fit_predict_naive(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    combined = pd.concat([train_df, test_df]).sort_values(DATE_COL)
    combined["pred"] = combined[TARGET_COL].shift(1)
    out = combined[combined[DATE_COL].isin(test_df[DATE_COL])]
    return _result(out[DATE_COL], out[TARGET_COL].to_numpy(), out["pred"].to_numpy())


def fit_predict_seasonal_naive(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    lag_days = params.get("lag_days", 7)
    combined = pd.concat([train_df, test_df]).sort_values(DATE_COL)
    combined["pred"] = combined[TARGET_COL].shift(lag_days)
    out = combined[combined[DATE_COL].isin(test_df[DATE_COL])]
    return _result(out[DATE_COL], out[TARGET_COL].to_numpy(), out["pred"].to_numpy())


def fit_predict_rolling_mean(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    window = params["window"]
    combined = pd.concat([train_df, test_df]).sort_values(DATE_COL)
    combined["pred"] = combined[TARGET_COL].shift(1).rolling(window).mean()
    out = combined[combined[DATE_COL].isin(test_df[DATE_COL])]
    return _result(out[DATE_COL], out[TARGET_COL].to_numpy(), out["pred"].to_numpy())


def fit_predict_weekday_average(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    weekday_means = train_df.groupby("day_of_week")[TARGET_COL].mean()
    pred = test_df["day_of_week"].map(weekday_means).to_numpy()
    return _result(test_df[DATE_COL], test_df[TARGET_COL].to_numpy(), pred)


# ---------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------

def fit_predict_sarimax(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Resolve gaps ONCE on the full continuous train+test series before
    # splitting, exactly as src/models/train_sarimax.py's
    # prepare_sarimax_data does - dropping a NaN row and then calling
    # asfreq("D") can reintroduce that exact date as a fresh NaN row (the
    # reindex fills the now-missing calendar slot back in). Doing this per
    # train/test slice separately, as a naive implementation would, misses
    # that a mid-test-window gap re-appears as NaN inside the exog SARIMAX
    # is asked to forecast against. Resolving it once on the combined
    # series and then splitting keeps train and test both genuinely
    # gap-free, at the cost of dropping that single date from evaluation.
    cols = [DATE_COL, TARGET_COL] + SARIMAX_EXOG_FEATURES
    combined = pd.concat([train_df[cols], test_df[cols]]).dropna().sort_values(DATE_COL)
    combined = combined.set_index(DATE_COL).asfreq("D")
    combined = combined[combined.notna().all(axis=1)]

    test_start = test_df[DATE_COL].min()
    train = combined[combined.index < test_start]
    test = combined[combined.index >= test_start]

    if len(train) == 0 or len(test) == 0:
        raise ValueError("no usable rows remained after resolving gaps for this fold")

    model = SARIMAX(
        train[TARGET_COL],
        exog=train[SARIMAX_EXOG_FEATURES],
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    results = model.fit(disp=False)
    pred = results.forecast(steps=len(test), exog=test[SARIMAX_EXOG_FEATURES])
    return _result(test.index, test[TARGET_COL].to_numpy(), np.asarray(pred))


# ---------------------------------------------------------------------
# Gradient-boosted trees
# ---------------------------------------------------------------------

def fit_predict_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from xgboost import XGBRegressor

    model = XGBRegressor(objective="reg:squarederror", random_state=42, **params)
    model.fit(train_df[FEATURES], train_df[TARGET_COL])
    pred = model.predict(test_df[FEATURES])
    return _result(test_df[DATE_COL], test_df[TARGET_COL].to_numpy(), pred)


def fit_predict_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(random_state=42, verbose=-1, **params)
    model.fit(train_df[FEATURES], train_df[TARGET_COL])
    pred = model.predict(test_df[FEATURES])
    return _result(test_df[DATE_COL], test_df[TARGET_COL].to_numpy(), pred)


def fit_predict_catboost(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, features: list[str] | None = None
) -> pd.DataFrame:
    from catboost import CatBoostRegressor

    feats = features if features is not None else FEATURES
    model = CatBoostRegressor(random_state=42, verbose=False, allow_writing_files=False, **params)
    model.fit(train_df[feats], train_df[TARGET_COL])
    pred = model.predict(test_df[feats])
    return _result(test_df[DATE_COL], test_df[TARGET_COL].to_numpy(), pred)


# ---------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------

def fit_predict_prophet(
    train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, features: list[str] | None = None
) -> pd.DataFrame:
    from prophet import Prophet

    regressors = features if features is not None else PROPHET_REGRESSORS

    train = train_df.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})[["ds", "y"] + regressors]
    test = test_df.rename(columns={DATE_COL: "ds"})[["ds"] + regressors]

    model = Prophet(**params)
    for regressor in regressors:
        model.add_regressor(regressor)
    model.fit(train)

    forecast = model.predict(test)
    return _result(test["ds"], test_df[TARGET_COL].to_numpy(), forecast["yhat"].to_numpy())


# ---------------------------------------------------------------------
# Neural forecasters
# ---------------------------------------------------------------------

def _fit_predict_neural(model_cls_kwargs_fn, train_df, test_df, params):
    from src.models.neural_common import predict, prepare_datasets, set_seed, train_model

    window = params["window"]
    epochs = params["epochs"]

    combined = pd.concat([train_df, test_df]).sort_values(DATE_COL).reset_index(drop=True)
    split_date = str(test_df[DATE_COL].min().date())
    valid_end_date = str((test_df[DATE_COL].max() + pd.Timedelta(days=1)).date())

    set_seed(42)
    data = prepare_datasets(
        split_date=split_date,
        window=window,
        features=FEATURES,
        df=combined,
        train_window_days=None,  # train_df is already the correctly-windowed slice
        valid_end_date=valid_end_date,
    )

    model = model_cls_kwargs_fn(data["n_features"])
    train_model(model, data["train_dataset"], epochs=epochs, batch_size=16, lr=1e-3)
    preds = predict(model, data["valid_dataset"], data["scaler"])

    return _result(data["valid_dates"], data["y_valid_actual"], preds)


def fit_predict_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from src.models.train_lstm import LSTMForecaster

    def build(n_features: int):
        return LSTMForecaster(
            n_features=n_features,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
        )

    return _fit_predict_neural(build, train_df, test_df, params)


def fit_predict_transformer(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from src.models.train_transformer import TransformerForecaster

    def build(n_features: int):
        return TransformerForecaster(
            n_features=n_features,
            window=params["window"],
            d_model=params["d_model"],
            nhead=params["nhead"],
        )

    return _fit_predict_neural(build, train_df, test_df, params)


MODEL_ADAPTERS = {
    "naive": fit_predict_naive,
    "seasonal_naive": fit_predict_seasonal_naive,
    "rolling_mean": fit_predict_rolling_mean,
    "weekday_average": fit_predict_weekday_average,
    "sarimax": fit_predict_sarimax,
    "xgboost": fit_predict_xgboost,
    "lightgbm": fit_predict_lightgbm,
    "catboost": fit_predict_catboost,
    "prophet": fit_predict_prophet,
    "lstm": fit_predict_lstm,
    "transformer": fit_predict_transformer,
}
