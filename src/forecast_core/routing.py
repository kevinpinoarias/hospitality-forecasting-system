"""
Which piece of a version forecasts which day, and v1.13's year-end specialist windows. Moved unchanged from
src/forecast_log/predict.py (Models, route_post, with_year_end_specialists) and src/forecast_log/v1_13.py
(the window constants and in_window).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecast_core.bags import Bag
from src.forecast_core.rows import EXTRA, lags_known, weekday_names, with_weekend
from src.models.final_model import DATE_COL

# v1.13's specialist windows
COMPANION_PREFIX, MASK_PREFIX = "comp_", "mask_"
COMPANION_WINDOW = ((12, 26), (12, 31))  # (month, day) first and last, inclusive
MASK_WINDOW = ((1, 1), (1, 8))
INPUTS = ["lag_7_sales", "rolling_7_sales", "rolling_14_sales", "lag_1_fe", "lag_7_fe", "rolling_7_fe", "lag_14_sales", "lag_21_sales", "same_wd_mean4", "same_wd_max4"]


def in_window(dates, first: tuple[int, int], last: tuple[int, int]) -> np.ndarray:
    """True for the dates inside a window given as (month, day) first and last, inclusive; a window that starts in December wraps over the new year."""
    d = pd.to_datetime(pd.Series(list(dates)))
    md = (d.dt.month * 100 + d.dt.day).to_numpy()
    a, b = first[0] * 100 + first[1], last[0] * 100 + last[1]
    return (md >= a) | (md <= b) if a > b else (md >= a) & (md <= b)


@dataclass
class Models:
    version: str
    portfolio: Bag | None  # None for a version without a portfolio arm (v1.11 is a hydro arm only)
    pieces: dict[str, Bag]
    portfolio_fingerprint: str | None
    piece_fingerprints: dict[str, str]
    standin: str = "S0"  # the stand-in for the manager's number the version is served with: S0 (the weekday-and-season median, every version up to v1.14) or S6 (the hybrid, v1.131 and v1.141)

    @property
    def combined_fingerprint(self) -> str:
        """One hash standing for the six fitted pieces (post and hydro rows); the portfolio row carries its own."""
        return hashlib.sha256("".join(self.piece_fingerprints[k] for k in sorted(self.piece_fingerprints)).encode()).hexdigest()


def route_post(models: Models, rows: pd.DataFrame, sales_known: pd.Series, hydro: bool, prefix: str = "", masked: tuple[str, ...] | list[str] = ()) -> np.ndarray:
    """The post (or hydro) model's raw forecasts for `rows` (before the weekend correction). `rows` must carry the `hydro_thu` column when hydro is True.
    `prefix` picks a specialist set of the same pieces (v1.13's `comp_` and `mask_`); `masked` lists inputs set to missing before the pieces predict (the routing itself is decided on the unmasked inputs)."""
    w = with_weekend(rows, sales_known)
    wd = weekday_names(rows[DATE_COL])
    friday_ok = lags_known(sales_known, rows[DATE_COL], (7, 14, 21, 28))
    sat_ok = w[EXTRA["Saturday"]].notna().all(axis=1).to_numpy()
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
    if COMPANION_PREFIX + "log_hydro_thu" not in models.pieces:
        return raw
    out = np.array(raw, dtype=float)
    for prefix, window, masked in ((COMPANION_PREFIX, COMPANION_WINDOW, ()), (MASK_PREFIX, MASK_WINDOW, INPUTS)):
        selected = in_window(rows[DATE_COL], *window)
        if selected.any():
            out[selected] = route_post(models, rows[selected].reset_index(drop=True), sales_known, hydro=True, prefix=prefix, masked=masked)
    return out
