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
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.weather import fetch_live_weather, live_weather_end
from src.config import CONFIG
from src.forecast_core.routing import Models, route_post, with_year_end_specialists  # noqa: F401  (moved there unchanged)
from src.forecast_core.rows import lags_known, served_rows  # noqa: F401
from src.experiments import run_series35_holdout as h
from src.features import year_ago
from src.features.payday_intensity import add_payday_intensity, add_payday_intensity_fixed
from src.features.serving_features import seasonal_medians
from src.forecast_log import standin as standin_module
from src.forecast_log import v1
from src.models.final_model import DATE_COL, TARGET_COL, BaggedCatBoost

MODELS = ("portfolio", "post", "hydro")
STANDIN_SUFFIX = "_manager_input"  # forecast_days adds one column `<arm>_manager_input` per further version


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
    standin = entry.get("standin", "S0")
    if standin not in ("S0", standin_module.NAME):
        raise RuntimeError(f"version {version} names the stand-in {standin}, which this code does not know")
    if "portfolio_model" not in entry:  # a version without a portfolio arm (v1.11): the pieces are all it has
        return Models(version, None, pieces, None, prints, standin)
    portfolio_print = h.fingerprint(CONFIG.path("final_model_dir"))
    if portfolio_print != entry["portfolio_model"]["fingerprint"]:
        raise RuntimeError("the deployed model files do not match the fingerprint recorded for the portfolio model")
    b, _ = BaggedCatBoost.load(CONFIG.path("final_model_dir"))
    return Models(version, h.Bag(b.models, b.features, "log1p"), pieces, portfolio_print, prints, standin)


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


# --------------------------------------------------------------------------- the models


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
        if extra.standin == standin_module.NAME:  # v1.131 and v1.141: the hybrid stand-in in place of the weekday-and-season median, in the manager's-number input of the target rows only
            rows_extra["forecast_sales"] = standin_module.hybrid_standin(full, last_known_day, rows_extra[DATE_COL])
        raw = with_year_end_specialists(extra, rows_extra, sales_known, route_post(extra, rows_extra, sales_known, hydro=True))
        out[extra_arm_name(extra.version)] = h.correct_new_block(extra_days, extra_pred, targets, raw)
        out[extra_arm_name(extra.version) + STANDIN_SUFFIX] = rows_extra["forecast_sales"].to_numpy()  # the manager's-number input this arm was given, for the log's `manager_forecast_used`
    return out


def seasonal_baseline(full: pd.DataFrame, last_known_day: pd.Timestamp, targets: list[pd.Timestamp]) -> np.ndarray:
    """The benchmark that needs no model: the weekday-and-season median of actual sales over the days known when the forecast is issued."""
    return seasonal_medians(full[full[DATE_COL] <= last_known_day], list(targets), TARGET_COL)
