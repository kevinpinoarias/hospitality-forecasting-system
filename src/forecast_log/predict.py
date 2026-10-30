"""
The log's three models and the benchmarks, as functions of the target days and what is known when the forecast is issued.

  portfolio   the frozen deployed model, on the original school calendar
  post        the weekend package: the log model Monday to Thursday; Friday, Saturday and Sunday with their own inputs where those inputs exist; the Saturday and Sunday correction
  hydro       the post model with the Thursday-Hydro flag in its Monday-to-Thursday log model (log_hydro_thu); identical to the post model on Friday to Sunday

Routing at any lead (it reduces to the pre-registered routing of the holdout test at a week's distance): Friday uses `A_all` only if the sales of the same weekday 1 to 4 weeks earlier are all known,
else the plain Monday-to-Thursday model; Saturday uses `A_lags` if the sales 2 and 3 weeks earlier are known, else the plain raw model; Sunday uses `B` (calendar and weather only, always available).

Features for days beyond the data are built the way the live API builds them (`served_rows`): real history up to the last known day, later days filled with weekday-and-season medians, weather from the
historical-average fallback. The manager's forecast is the supplied number, or (without-manager variant) the weekday-and-season median of the manager's past forecasts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series35_holdout as h
from src.features.serving_features import seasonal_medians
from src.forecast_log import v1
from src.models.final_model import DATE_COL, TARGET_COL, BaggedCatBoost

MODELS = ("portfolio", "post", "hydro")
WEATHER_SOURCE = "fallback_average"


@dataclass
class Models:
    version: str
    portfolio: h.Bag
    pieces: dict[str, h.Bag]
    portfolio_fingerprint: str
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
    portfolio_print = h.fingerprint(CONFIG.path("final_model_dir"))
    if portfolio_print != entry["portfolio_model"]["fingerprint"]:
        raise RuntimeError("the deployed model files do not match the fingerprint recorded for the portfolio model")
    b, _ = BaggedCatBoost.load(CONFIG.path("final_model_dir"))
    return Models(version, h.Bag(b.models, b.features, "log1p"), pieces, portfolio_print, prints)


def lags_known(sales_known: pd.Series, dates: pd.Series, lags: tuple[int, ...]) -> np.ndarray:
    """True for each date whose sales exactly `lag` days earlier are all known, for every lag given."""
    ok = pd.Series(True, index=sales_known.index)
    for k in lags:
        ok &= sales_known.shift(k).notna()
    return ok.reindex(pd.DatetimeIndex(dates)).fillna(False).to_numpy()


def route_post(models: Models, rows: pd.DataFrame, sales_known: pd.Series, hydro: bool) -> np.ndarray:
    """The post (or hydro) model's raw forecasts for `rows` (before the weekend correction). `rows` must carry the `hydro_thu` column when hydro is True."""
    w = h.with_weekend(rows, sales_known)
    wd = h.weekday_names(rows[DATE_COL])
    base = models.pieces["log_hydro_thu" if hydro else "log"].predict(rows)
    out = np.where(wd == "Friday", np.where(lags_known(sales_known, rows[DATE_COL], (7, 14, 21, 28)), models.pieces["fri_A_all"].predict(w), base), base)
    sat_ok = w[h.EXTRA["Saturday"]].notna().all(axis=1).to_numpy()
    saturday = np.where(sat_ok, models.pieces["sat_A_lags"].predict(w), models.pieces["raw"].predict(rows))
    out = np.where(wd == "Saturday", saturday, out)
    return np.where(wd == "Sunday", models.pieces["sun_B"].predict(w), out)


def forecast_days(models: Models, full: pd.DataFrame, last_known_day: pd.Timestamp, targets: list[pd.Timestamp], manager: np.ndarray | None,
                  history: tuple[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """Portfolio, post and hydro forecasts for `targets`. `manager` is the manager's numbers (one per target) or None for the without-manager variant."""
    hist_days, hist_pred = history
    sales_known = h.sales_known_series(full, last_known_day, max(targets))
    rows_v2 = v1.add_hydro_thu(h.served_rows(full, last_known_day, targets, "v2", manager))
    rows_v1 = h.served_rows(full, last_known_day, targets, "v1", manager)
    post = h.correct_new_block(hist_days, hist_pred, targets, route_post(models, rows_v2, sales_known, hydro=False))
    hydro = h.correct_new_block(hist_days, hist_pred, targets, route_post(models, rows_v2, sales_known, hydro=True))
    return pd.DataFrame({"target_date": [pd.Timestamp(t).date() for t in targets], "portfolio": models.portfolio.predict(rows_v1), "post": post, "hydro": hydro,
                         "forecast_sales_input": rows_v2["forecast_sales"].to_numpy()})


def seasonal_baseline(full: pd.DataFrame, last_known_day: pd.Timestamp, targets: list[pd.Timestamp]) -> np.ndarray:
    """The benchmark that needs no model: the weekday-and-season median of actual sales over the days known when the forecast is issued."""
    return seasonal_medians(full[full[DATE_COL] <= last_known_day], list(targets), TARGET_COL)
