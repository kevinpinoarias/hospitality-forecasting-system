"""
Ranks the days a forecast missed by the most - behind the chat assistant's
"worst days / biggest over- and under-forecasts" tools.

Only days that have a forecast made before the day was known (the
pipeline's rolling-origin backtest period) are ranked, for BOTH the model
and the manager's own forecast, so the two are judged on the same days and
the period is the same whichever is asked about.

Left out: known closure days (a closed venue's £0 is not a forecasting
miss - the assistant once presented one as a shortfall) and days with no
recorded sales figure.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import pandas as pd

from src.api.schemas import ErrorRankingResponse, ExcludedDays, RankedErrorDay
from src.evaluation.error_analysis import (
    top_best_days,
    top_overpredictions,
    top_underpredictions,
    top_worst_days,
)
from src.preprocessing.build_daily_sales import KNOWN_CLOSURE_MONTH_DAYS

DATE_COL = "date"
ACTUAL_COL = "total_sales"
MANAGER_COL = "forecast_sales"
MODEL_COL = "model_forecast"

DEFAULT_RANKING_DAYS = 10
MAX_RANKING_DAYS = 20

Kind = Literal["worst", "over", "under", "best"]
Forecast = Literal["model", "manager"]

_RANKERS = {
    "worst": top_worst_days,
    "over": top_overpredictions,
    "under": top_underpredictions,
    "best": top_best_days,
}
_JUDGED_COL = {"model": MODEL_COL, "manager": MANAGER_COL}
_SUBJECT = {"model": "the model's forecast", "manager": "the manager's own forecast"}
_WHAT = {
    "worst": "was furthest from what was actually sold, too high or too low",
    "over": "was too high by the most (actual sales came in below it)",
    "under": "was too low by the most (actual sales came in above it)",
    "best": "came closest to what was actually sold (the smallest gaps)",
}


def rank_errors(
    history: pd.DataFrame,
    model_forecasts: dict[dt.date, float],
    kind: Kind,
    forecast: Forecast,
    limit: int = DEFAULT_RANKING_DAYS,
) -> ErrorRankingResponse:
    """
    `history` needs date, total_sales and forecast_sales (the manager's
    forecast on record) columns; `model_forecasts` maps each date that has a
    before-the-day model forecast to that forecast.
    """
    if kind not in _RANKERS:
        raise ValueError(f"kind must be one of {sorted(_RANKERS)}, not {kind!r}.")
    if forecast not in _JUDGED_COL:
        raise ValueError(f"forecast must be one of {sorted(_JUDGED_COL)}, not {forecast!r}.")
    if not 1 <= limit <= MAX_RANKING_DAYS:
        raise ValueError(f"limit must be between 1 and {MAX_RANKING_DAYS}, not {limit}.")

    days = history[[DATE_COL, ACTUAL_COL, MANAGER_COL]].copy()
    days[DATE_COL] = pd.to_datetime(days[DATE_COL])
    days[MODEL_COL] = days[DATE_COL].dt.date.map(model_forecasts)
    days = days[days[MODEL_COL].notna()].sort_values(DATE_COL).reset_index(drop=True)
    if days.empty:
        raise ValueError("There are no days with a forecast made before the day to rank.")

    judged_col = _JUDGED_COL[forecast]
    closure = pd.Series(
        [(d.month, d.day) in KNOWN_CLOSURE_MONTH_DAYS for d in days[DATE_COL]], index=days.index
    )
    missing = days[ACTUAL_COL].isna() | days[judged_col].isna()
    ranked = days[~closure & ~missing]

    top = _RANKERS[kind](ranked, ACTUAL_COL, judged_col, n=limit)
    # top_over/under_predictions sort by signed error and would pad the list
    # with the wrong direction if fewer than `limit` days qualify.
    if kind == "over":
        top = top[top["signed_error"] > 0]
    elif kind == "under":
        top = top[top["signed_error"] < 0]

    items = []
    for row in top.itertuples():
        actual = float(getattr(row, ACTUAL_COL))
        predicted = float(getattr(row, judged_col))
        gap = abs(float(row.signed_error))
        if round(gap, 2) == 0:
            direction = "exact"
        else:
            direction = "over_forecast" if row.signed_error > 0 else "under_forecast"
        items.append(RankedErrorDay(
            date=row.date.date(),
            day_of_week=row.date.strftime("%A"),
            actual_sales=round(actual, 2),
            forecast=round(predicted, 2),
            gap_gbp=round(gap, 2),
            direction=direction,
            gap_percent_of_actual=round(gap / actual * 100, 1) if actual > 0 else None,
        ))

    period_start, period_end = days[DATE_COL].min().date(), days[DATE_COL].max().date()
    closure_dates = [d.date() for d in days.loc[closure, DATE_COL]]
    return ErrorRankingResponse(
        ranking=f"Days {_SUBJECT[forecast]} {_WHAT[kind]}.",
        forecast_judged=forecast,
        period_start=period_start,
        period_end=period_end,
        days_ranked=len(ranked),
        excluded=ExcludedDays(
            known_closure_days=len(closure_dates),
            known_closure_dates=closure_dates,
            missing_figures=int((missing & ~closure).sum()),
        ),
        limit=limit,
        days=items,
        note=(
            f"Ranked over {len(ranked)} days from {period_start} to {period_end} - the days "
            "that have a forecast made before the day was known, so the comparison is fair. "
            "Days outside this period are not covered. Christmas Day and New Year's Day are "
            "left out because the venue is closed."
            + (
                " The smallest gaps are in pounds, so a quiet day can rank high simply because "
                "there was less to miss - gap_percent_of_actual shows each gap relative to sales."
                if kind == "best"
                else ""
            )
        ),
    )
