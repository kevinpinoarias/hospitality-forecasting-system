"""
The log's three models and the benchmarks, as functions of the target days and what is known when the forecast is issued.

  portfolio   the frozen deployed model, on the original school calendar
  post        the weekend package: the log model Monday to Thursday; Friday, Saturday and Sunday with their own inputs where those inputs exist; the Saturday and Sunday correction
  hydro       the post model with the Thursday-Hydro flag in its Monday-to-Thursday log model (log_hydro_thu); identical to the post model on Friday to Sunday

Routing at any lead (it reduces to the pre-registered routing of the holdout test at a week's distance): Friday uses `A_all` only if the sales of the same weekday 1 to 4 weeks earlier are all known,
else the plain Monday-to-Thursday model; Saturday uses `A_lags` if the sales 2 and 3 weeks earlier are known, else the plain raw model; Sunday uses `B` (calendar and weather only, always available).

Features for days beyond the data are built the way the live API builds them (src/api/inference.py): real history up to the last known day, later days filled with weekday-and-season medians, and
the weather Open-Meteo has for the days after the last known one (recorded for past days, forecast for about 16 days ahead), the historical average beyond that or if the service fails. The
manager's number is never known in time, so its place is taken by the stand-in the API uses: the weekday-and-season median of the manager's past forecasts.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.weather import fetch_live_weather, live_weather_end
from src.config import CONFIG
from src.experiments import run_series35_holdout as h
from src.features import year_ago
from src.features.payday_intensity import add_payday_intensity, add_payday_intensity_fixed
from src.features.serving_features import WEATHER_COLS, extend_daily_frame, fill_future_sales_seasonally, seasonal_medians, with_history_features
from src.forecast_log import v1, v1_13
from src.models.final_model import DATE_COL, TARGET_COL, BaggedCatBoost

MODELS = ("portfolio", "post", "hydro")


@dataclass
class Models:
    version: str
    portfolio: h.Bag | None  # None for a version without a portfolio arm (v1.11 is a hydro arm only)
    pieces: dict[str, h.Bag]
    portfolio_fingerprint: str | None
    piece_fingerprints: dict[str, str]

    @property
    def combined_fingerprint(self) -> str:
        """One hash standing for the six fitted pieces (post and hydro rows); the portfolio row carries its own."""
        import hashlib

        return hashlib.sha256("".join(self.piece_fingerprints[k] for k in sorted(self.piece_fingerprints)).encode()).hexdigest()


def load_models(log_repo: Path, version: str = v1.VERSION, models_root: Path | None = None) -> Models:
    """Load the registered version and verify every file against the fingerprint recorded when it was fitted. Any difference stops the run."""
    registry = json.loads((log_repo / "models" / "versions.json").read_text(encoding="utf-8"))
    entry = next((v for v in registry["versions"] if v["version"] == version), None)
    if entry is None:
        raise RuntimeError(f"version {version} is not registered in {log_repo / 'models' / 'versions.json'}")
    root = (models_root or v1.MODELS_DIR.parent) / version
    pieces, prints = {}, {}
    for name, spec in entry["pieces"].items():
        got = h.fingerprint(root / name)
        if got != spec["fingerprint"]:
            raise RuntimeError(f"model piece {name} of {version} does not match its recorded fingerprint ({got[:16]}... against {spec['fingerprint'][:16]}...)")
        pieces[name] = h.Bag.load(root / name)
        prints[name] = got
    if "portfolio_model" not in entry:  # a version without a portfolio arm (v1.11): the pieces are all it has
        return Models(version, None, pieces, None, prints)
    portfolio_print = h.fingerprint(CONFIG.path("final_model_dir"))
    if portfolio_print != entry["portfolio_model"]["fingerprint"]:
        raise RuntimeError("the deployed model files do not match the fingerprint recorded for the portfolio model")
    b, _ = BaggedCatBoost.load(CONFIG.path("final_model_dir"))
    return Models(version, h.Bag(b.models, b.features, "log1p"), pieces, portfolio_print, prints)


# --------------------------------------------------------------------------- weather, as the live API takes it


def fetch_run_weather(last_known_day: pd.Timestamp, issue_date: dt.date) -> tuple[pd.DataFrame, bool]:
    """Open-Meteo's daily weather for every day after the last known one that it covers (recorded for past days, forecast up to about 15 days after the issue date),
    with the API's own function. Returns (weather, errored); on a failed fetch the weather is empty and every day keeps its historical average."""
    start = (last_known_day + pd.Timedelta(days=1)).date()
    weather, errored = fetch_live_weather(start, live_weather_end(issue_date))
    weather = weather.copy()
    weather[DATE_COL] = pd.to_datetime(weather[DATE_COL]).dt.normalize()
    return weather.reset_index(drop=True), errored


def weather_label(target: dt.date, weather: pd.DataFrame, errored: bool, issue_date: dt.date) -> str:
    """Where the target day's own weather came from, in the API's words (src/api/inference.py)."""
    if len(weather) and pd.Timestamp(target) in set(weather[DATE_COL]):
        return "observed" if target <= issue_date else "forecast"
    if errored and target <= live_weather_end(issue_date):
        return "historical_average_api_error"
    return "historical_average"


def served_rows(full: pd.DataFrame, last_known_day: pd.Timestamp, target_dates, calendar: str, weather: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for `target_dates`: real history up to `last_known_day`, the days after it filled with weekday-and-season medians (sales, the manager's number, weather), then the
    live weather laid over those days where Open-Meteo has a figure. Nothing dated after `last_known_day` is read from `full`."""
    history = full[full[DATE_COL] <= last_known_day]
    daily = extend_daily_frame(history, max(pd.Timestamp(d) for d in target_dates).date())
    if len(weather):
        live = weather.set_index(DATE_COL)
        later = (daily[DATE_COL] > last_known_day) & daily[DATE_COL].isin(live.index)
        for col in WEATHER_COLS:
            daily.loc[later, col] = live.loc[daily.loc[later, DATE_COL], col].to_numpy()
    frame = h.build_frame(daily, calendar)
    position = {d: i for i, d in enumerate(frame[DATE_COL])}
    sales = fill_future_sales_seasonally(frame, len(history), history)
    return with_history_features(frame, sales, [position[pd.Timestamp(d)] for d in target_dates])


# --------------------------------------------------------------------------- the models


def lags_known(sales_known: pd.Series, dates: pd.Series, lags: tuple[int, ...]) -> np.ndarray:
    """True for each date whose sales exactly `lag` days earlier are all known, for every lag given."""
    ok = pd.Series(True, index=sales_known.index)
    for k in lags:
        ok &= sales_known.shift(k).notna()
    return ok.reindex(pd.DatetimeIndex(dates)).fillna(False).to_numpy()


def route_post(models: Models, rows: pd.DataFrame, sales_known: pd.Series, hydro: bool, prefix: str = "", masked: tuple[str, ...] | list[str] = ()) -> np.ndarray:
    """The post (or hydro) model's raw forecasts for `rows` (before the weekend correction). `rows` must carry the `hydro_thu` column when hydro is True.
    `prefix` picks a specialist set of the same pieces (v1.13's `comp_` and `mask_`); `masked` lists inputs set to missing before the pieces predict (the routing itself is decided on the unmasked inputs)."""
    w = h.with_weekend(rows, sales_known)
    wd = h.weekday_names(rows[DATE_COL])
    friday_ok = lags_known(sales_known, rows[DATE_COL], (7, 14, 21, 28))
    sat_ok = w[h.EXTRA["Saturday"]].notna().all(axis=1).to_numpy()
    if masked:
        rows, w = rows.copy(), w.copy()
        for frame in (rows, w):
            for c in masked:
                if c in frame.columns:
                    frame[c] = np.nan
    piece = lambda name: models.pieces[prefix + name]  # noqa: E731
    base = piece("log_hydro_thu" if hydro else "log").predict(rows)
    out = np.where(wd == "Friday", np.where(friday_ok, piece("fri_A_all").predict(w), base), base)
    saturday = np.where(sat_ok, piece("sat_A_lags").predict(w), piece("raw").predict(rows))
    out = np.where(wd == "Saturday", saturday, out)
    return np.where(wd == "Sunday", piece("sun_B").predict(w), out)


def with_year_end_specialists(models: Models, rows: pd.DataFrame, sales_known: pd.Series, raw: np.ndarray) -> np.ndarray:
    """v1.13's rule: on 26 to 31 December the companion pieces (no recent-sales inputs) forecast, on 1 to 8 January the masked pieces with the ten inputs set to missing; every other day keeps `raw`.
    A version without the specialist pieces is returned unchanged."""
    if v1_13.COMPANION_PREFIX + "log_hydro_thu" not in models.pieces:
        return raw
    out = np.array(raw, dtype=float)
    for prefix, window, masked in ((v1_13.COMPANION_PREFIX, v1_13.COMPANION_WINDOW, ()), (v1_13.MASK_PREFIX, v1_13.MASK_WINDOW, v1_13.INPUTS)):
        selected = v1_13.in_window(rows[DATE_COL], *window)
        if selected.any():
            out[selected] = route_post(models, rows[selected].reset_index(drop=True), sales_known, hydro=True, prefix=prefix, masked=masked)
    return out


def extra_arm_name(version: str) -> str:
    """The model name of an extra version's hydro arm in the log (`hydro_v1.11`): the log's de-duplication key is (type, day, model, variant), with no version."""
    return f"hydro_{version}"


def forecast_days(models: Models, full: pd.DataFrame, last_known_day: pd.Timestamp, targets: list[pd.Timestamp], history: tuple[pd.DataFrame, np.ndarray],
                  weather: pd.DataFrame, extras: list[tuple[Models, tuple[pd.DataFrame, np.ndarray]]] | None = None) -> pd.DataFrame:
    """Portfolio, post and hydro forecasts for `targets`, without the manager's number (the stand-in in its place), with the live weather where it exists. `extras` are further versions,
    each with its own correction history, whose hydro arm is added as a column named `hydro_<version>` (the same rows, the same weather, the payday intensity added for the pieces that use it)."""
    hist_days, hist_pred = history
    sales_known = h.sales_known_series(full, last_known_day, max(targets))
    rows_v2 = v1.add_hydro_thu(served_rows(full, last_known_day, targets, "v2", weather))
    rows_v1 = served_rows(full, last_known_day, targets, "v1", weather)
    post = h.correct_new_block(hist_days, hist_pred, targets, route_post(models, rows_v2, sales_known, hydro=False))
    hydro = h.correct_new_block(hist_days, hist_pred, targets, route_post(models, rows_v2, sales_known, hydro=True))
    out = pd.DataFrame({"target_date": [pd.Timestamp(t).date() for t in targets], "portfolio": models.portfolio.predict(rows_v1), "post": post, "hydro": hydro,
                        "forecast_sales_input": rows_v2["forecast_sales"].to_numpy()})
    for extra, (extra_days, extra_pred) in extras or []:
        rows_extra = add_payday_intensity_fixed(add_payday_intensity(rows_v2, DATE_COL), DATE_COL)  # both columns computed from the date alone; a piece uses the one it was fitted with and ignores the other
        rows_extra[year_ago.COLUMN] = year_ago.year_ago_sales(rows_extra[DATE_COL], sales_known).to_numpy()  # v1.14's input, the actual sales 364 days earlier; a piece without it ignores the column
        raw = with_year_end_specialists(extra, rows_extra, sales_known, route_post(extra, rows_extra, sales_known, hydro=True))
        out[extra_arm_name(extra.version)] = h.correct_new_block(extra_days, extra_pred, targets, raw)
    return out


def seasonal_baseline(full: pd.DataFrame, last_known_day: pd.Timestamp, targets: list[pd.Timestamp]) -> np.ndarray:
    """The benchmark that needs no model: the weekday-and-season median of actual sales over the days known when the forecast is issued."""
    return seasonal_medians(full[full[DATE_COL] <= last_known_day], list(targets), TARGET_COL)
