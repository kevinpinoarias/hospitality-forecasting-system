"""
Builds features for a requested date and produces the forecast and the
dry/heavy-rain scenarios from the final model (25-seed log1p CatBoost, see
src/models/final_model.py).

How a date is handled depends on where it falls:

- Inside the historical data: every feature is real. If the date was one of
  the pipeline's backtest days and the caller didn't supply their own
  forecast_sales, the forecast returned is the backtest's out-of-sample
  prediction - what the model predicted before it saw that day - so a
  comparison with actual_sales is a fair test. Otherwise the served model,
  which was trained on data covering that date, makes the prediction.
- Beyond the historical data: the date's recent sales history is filled
  with historical weekday/season medians, and its weather with the live
  Open-Meteo figures where they exist and medians elsewhere - the approach
  Series 18 of docs/EXPERIMENT_LOG.md validated out to 180 days. The shared
  code lives in src/features/serving_features.py, which reuses the training
  pipeline's own feature functions so serving and training can't silently
  drift apart.
"""

from __future__ import annotations

import datetime as dt
import threading

import numpy as np
import pandas as pd

from src.api.day_context import build_day_context
from src.api.history import (
    HistoryCache,
    actual_sales_on,
    days_beyond_training_data,
    equivalent_weekday_last_year,
)
from src.api.schemas import (
    HistoricalComparison,
    PredictResult,
    ScenarioPredictions,
    WeatherDetails,
)
from src.api.weather import (
    DRY_MM,
    HEAVY_RAIN_MM,
    categorize_rain,
    fetch_live_weather,
    live_weather_end,
)
from src.features.build_features import add_weather_features
from src.features.serving_features import (
    WEATHER_COLS,
    build_feature_frame,
    extend_daily_frame,
    fill_future_sales_seasonally,
    history_features_at,
    typical_heavy_rain_mm,
)
from src.models.final_model import BACKTEST_PREDICTIONS_FILENAME, MODEL_DIR, BaggedCatBoost
from src.preprocessing.build_daily_sales import KNOWN_CLOSURE_MONTH_DAYS

DATE_COL = "date"
# How far past the end of the data the precomputed frame reaches; a request
# beyond it extends the frame by another EXTENSION_DAYS.
EXTENSION_DAYS = 730
WEATHER_DERIVED_COLS = ["is_hot_for_scotland", "is_heavy_rain", "is_dry_day", "temp_anomaly_14d", "warm_streak_len"]


def _with_rain(rows: pd.DataFrame, rain_mm: float) -> pd.DataFrame:
    """The same feature rows with a different amount of rain - only the
    rain-derived flags depend on it."""
    out = rows.copy()
    out["rain_mm"] = rain_mm
    out["is_heavy_rain"] = int(rain_mm > HEAVY_RAIN_MM)
    out["is_dry_day"] = int(rain_mm < DRY_MM)
    return out


class ModelService:
    """Loads the trained model once and serves predictions from it."""

    def __init__(self, history: HistoryCache, model_dir=MODEL_DIR) -> None:
        self.history = history
        self.model, self.metadata = BaggedCatBoost.load(model_dir)
        backtest = pd.read_csv(model_dir / BACKTEST_PREDICTIONS_FILENAME, parse_dates=[DATE_COL])
        self.backtest_predictions: dict[dt.date, float] = dict(
            zip(backtest[DATE_COL].dt.date, backtest["final_model_prediction"])
        )
        self._lock = threading.Lock()
        self._frame_built_for: tuple | None = None
        self._frame = pd.DataFrame()
        self._sales = np.array([])
        self._first_future = 0

    @property
    def model_name(self) -> str:
        return self.metadata["model"]

    @property
    def trained_through(self) -> str:
        return self.metadata["trained_on"]["end_date"]

    # -----------------------------------------------------------------
    # Precomputed frame: history plus seasonal-median future days
    # -----------------------------------------------------------------

    def _ensure_frame(self, end_date: dt.date) -> None:
        with self._lock:
            key = (self.history.last_refreshed,)
            covered = self._frame_built_for == key and self._frame[DATE_COL].iloc[-1].date() >= end_date
            if covered:
                return

            history = self.history.df
            reach = max(end_date, self.history.max_date) + dt.timedelta(days=EXTENSION_DAYS)
            daily = extend_daily_frame(history, reach)
            frame = build_feature_frame(daily)
            first_future = len(history)

            self._frame = frame
            self._first_future = first_future
            self._sales = fill_future_sales_seasonally(frame, first_future, history)
            self._frame_built_for = key

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------

    def _live_weather(self, target_date: dt.date, today: dt.date) -> tuple[pd.DataFrame, bool]:
        """Live weather for every day after the data that Open-Meteo covers,
        when any of those days feed the target date's features (the date
        itself or the 14 days before it)."""
        start = self.history.max_date + dt.timedelta(days=1)
        end = live_weather_end(today)
        if target_date - dt.timedelta(days=14) > end:
            return pd.DataFrame(columns=[DATE_COL] + WEATHER_COLS), False
        return fetch_live_weather(start, end)

    def _future_row(
        self, target_date: dt.date, today: dt.date,
    ) -> tuple[pd.DataFrame, bool, str]:
        """Feature row for a date beyond the data, whether its weather is a
        real live figure, and the weather source label."""
        live, errored = self._live_weather(target_date, today)

        idx = self._first_future + (target_date - self.history.max_date).days - 1
        window = self._frame.iloc[: idx + 1]
        weather = window[[DATE_COL] + WEATHER_COLS].copy()

        if not live.empty:
            live = live.assign(**{DATE_COL: pd.to_datetime(live[DATE_COL]).dt.normalize()}).set_index(DATE_COL)
            overlap = weather[DATE_COL].isin(live.index)
            weather.loc[overlap, WEATHER_COLS] = live.loc[weather.loc[overlap, DATE_COL], WEATHER_COLS].to_numpy()

        derived = add_weather_features(weather[[DATE_COL]], weather)
        row = window.iloc[[-1]].copy()
        for col in WEATHER_COLS + WEATHER_DERIVED_COLS:
            row[col] = derived[col].iloc[-1]
        sales_hist = history_features_at(self._sales, self._frame["forecast_sales"].to_numpy(dtype=float), idx)
        for name, value in sales_hist.items():
            row[name] = value

        target_ts = pd.Timestamp(target_date)
        if not live.empty and target_ts in live.index:
            return row, True, "observed" if target_date <= today else "forecast"
        within_live_range = target_date <= live_weather_end(today)
        return row, False, "historical_average_api_error" if (errored and within_live_range) else "historical_average"

    @staticmethod
    def _weather_details(features: pd.Series) -> WeatherDetails:
        """The real weather behind a forecast, and the weather features the
        model derived from it."""
        def number(col: str) -> float | None:
            value = float(features[col])
            return None if np.isnan(value) else round(value, 2)

        anomaly = number("temp_anomaly_14d")
        return WeatherDetails(
            expected_rain_mm=number("rain_mm"),
            expected_rain_description=categorize_rain(float(features["rain_mm"])),
            expected_max_temp_c=number("max_temp"),
            expected_sunshine_hours=number("sun_hours"),
            hot_for_scotland=bool(features["is_hot_for_scotland"]),
            temperature_vs_previous_fortnight_c=anomaly,
            warm_streak_days=int(features["warm_streak_len"]),
        )

    def predict_one(
        self,
        target_date: dt.date,
        forecast_sales: float | None,
        today: dt.date | None = None,
    ) -> PredictResult:
        today = today or dt.datetime.now(dt.timezone.utc).date()
        history = self.history.df
        first_date = history[DATE_COL].min().date()
        if target_date < first_date:
            raise ValueError(f"{target_date} is before the start of the historical data ({first_date}).")
        in_history = target_date <= self.history.max_date

        if in_history:
            row = history[history[DATE_COL].dt.date == target_date].copy()
            weather_is_real, rain_source = True, "observed"
            if forecast_sales is None:
                forecast_sales_used, forecast_sales_source = float(row["forecast_sales"].iloc[0]), "manager_forecast_on_record"
            else:
                forecast_sales_used, forecast_sales_source = forecast_sales, "user_provided"
        else:
            self._ensure_frame(target_date)
            row, weather_is_real, rain_source = self._future_row(target_date, today)
            if forecast_sales is None:
                forecast_sales_used = float(row["forecast_sales"].iloc[0])
                forecast_sales_source = "historical_weekday_seasonal_average"
            else:
                forecast_sales_used, forecast_sales_source = forecast_sales, "user_provided"

        row["forecast_sales"] = forecast_sales_used
        features = row.iloc[0]
        day_context = build_day_context(
            features, target_date,
            sales_history_is_real=target_date <= self.history.max_date + dt.timedelta(days=1),
        )
        best_rain = float(row["rain_mm"].iloc[0])
        heavy_rain = best_rain if best_rain > HEAVY_RAIN_MM else typical_heavy_rain_mm(history)

        scenario_rows = pd.concat([row, _with_rain(row, 0.0), _with_rain(row, heavy_rain)], ignore_index=True)
        best_pred, dry_pred, rain_pred = (float(p) for p in self.model.predict(scenario_rows))

        if not in_history:
            prediction_source = "forecast"
        elif forecast_sales is None and target_date in self.backtest_predictions:
            # The weather on a past date is already known, so there are no
            # alternative scenarios to offer - and the served model's own
            # figures for this date would be in-sample, so they are not
            # mixed in with the out-of-sample prediction.
            prediction_source = "made_before_the_day"
            best_pred = dry_pred = rain_pred = float(self.backtest_predictions[target_date])
        else:
            prediction_source = "model_had_seen_the_day"

        weather = self._weather_details(features) if weather_is_real else None

        last_week = actual_sales_on(history, target_date - dt.timedelta(days=7))
        last_year = actual_sales_on(history, equivalent_weekday_last_year(target_date))
        actual_sales = actual_sales_on(history, target_date)

        return PredictResult(
            date=target_date,
            day_of_week=target_date.strftime("%A"),
            is_known_closure_day=(target_date.month, target_date.day) in KNOWN_CLOSURE_MONTH_DAYS,
            forecast_sales_used=forecast_sales_used,
            forecast_sales_source=forecast_sales_source,
            rain_data_source=rain_source,
            days_beyond_training_data=days_beyond_training_data(history, target_date),
            prediction_source=prediction_source,
            predictions=ScenarioPredictions(
                dry_scenario=dry_pred,
                heavy_rain_scenario=rain_pred,
                best_estimate=best_pred,
            ),
            day_context=day_context,
            weather=weather,
            historical_comparison=HistoricalComparison(
                same_day_last_week=last_week,
                same_day_last_year=last_year,
            ),
            actual_sales=actual_sales,
        )
