"""
The assistant's eight tools, on their own: what each one does, with nothing
about which AI calls it.

Moved here from query_forecast.py (November 2026) so the tools have no
Gemini import. Two programs use them: the Gemini assistant (query_forecast.py
and agent_forecast.py, which hand them to the model as Python functions) and
the MCP server (src/mcp_server/, which offers them to MCP clients such as
Claude Desktop). Neither knows about the other.

Six of the tools ask the forecasting API over HTTP - the numbers are always
computed there, never here. The other two return already-computed results
held in this package.
"""

from __future__ import annotations

import datetime as dt
import os

import requests
from dotenv import load_dotenv

from src.assistant.model_comparison_data import get_model_comparison
from src.assistant.sales_patterns_data import get_sales_and_forecast_patterns

load_dotenv()

# Defaults to the live API on Cloud Run. Set FORECAST_API_BASE_URL to use
# another instance, e.g. http://localhost:8000 for a local
# `uvicorn src.api.main:app`.
API_BASE_URL = (
    os.environ.get("FORECAST_API_BASE_URL")
    or "https://hospitality-forecasting-api-56220375160.us-central1.run.app"
)


def get_forecast(date: str, forecast_sales: float | None = None, model_version: str | None = None) -> dict:
    """Get the hospitality sales forecast for one specific date.

    Args:
        date: the date to forecast, in YYYY-MM-DD format.
        forecast_sales: the manager's own sales estimate for that day, only
            if the user asking the question actually provided one. Omit
            this argument entirely if they didn't - the forecasting system
            will use a sensible historical estimate automatically.
        model_version: which model answers, only if the person asked for a
            specific one by name (for example "v1.14" or "the weekend
            package"); the choices and their names come from the API's
            /versions list. Omit it otherwise - the standard model, or the
            model chosen in the app, is used automatically.
    """
    payload: dict = {"date": date}
    if forecast_sales is not None:
        payload["forecast_sales"] = forecast_sales
    if model_version:
        payload["model_version"] = model_version

    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"requests": [payload]},
        # Generous timeout: Cloud Run scales to zero when idle, and the
        # first request after a while has to cold-start a fresh container
        # (load the model, etc.) before it can even begin answering.
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["results"][0]


MAX_RANGE_DAYS = 31


def get_forecast_range(start_date: str, end_date: str, model_version: str | None = None) -> dict:
    """Get the hospitality sales forecast for every day in a period, in one
    call - use this instead of repeated get_forecast calls when a question
    covers several dates (a weekend, a week, a holiday period).

    Args:
        start_date: the first date, in YYYY-MM-DD format.
        end_date: the last date (inclusive), in YYYY-MM-DD format, at most
            31 days after start_date.
        model_version: which model answers, only if the person asked for a
            specific one by name (for example "v1.14" or "the weekend
            package"); the choices and their names come from the API's
            /versions list. Omit it otherwise - the standard model, or the
            model chosen in the app, is used automatically.
    """
    start, end = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    days = (end - start).days + 1
    if days < 1 or days > MAX_RANGE_DAYS:
        return {"error": f"The range must cover 1 to {MAX_RANGE_DAYS} days; this one covers {days}."}

    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(days)]
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"requests": [{"date": d, **({"model_version": model_version} if model_version else {})} for d in dates]},
        timeout=90,  # same cold-start allowance as get_forecast, for a larger batch
    )
    response.raise_for_status()
    return {"results": response.json()["results"]}


# Mirrors MAX_RANKING_DAYS in src/api/error_ranking.py (this module can't
# import it - the assistant's image contains only src/assistant/);
# tests/test_assistant_tools.py checks the two stay equal.
MAX_RANKING_DAYS = 20


def _get_error_ranking(kind: str, forecast: str, n: int) -> dict:
    forecast = str(forecast).strip().lower()
    if forecast not in {"model", "manager"}:
        return {"error": 'forecast must be "model" or "manager".'}

    days = max(1, min(int(n), MAX_RANKING_DAYS))
    response = requests.get(
        f"{API_BASE_URL}/errors/ranking",
        params={"kind": kind, "forecast": forecast, "limit": days},
        timeout=45,  # same cold-start allowance as get_forecast
    )
    response.raise_for_status()
    result = response.json()
    if days != int(n):
        result["limit_note"] = (
            f"{int(n)} days were asked for, but at most {MAX_RANKING_DAYS} can be shown, "
            f"so this is the top {days}."
        )
    return result


def get_top_worst_days(forecast: str = "model", n: int = 10) -> dict:
    """The days a forecast was furthest from what was actually sold, in either
    direction - the biggest misses overall, largest first.

    Args:
        forecast: whose forecast to judge - "model" (this system's own
            forecast, the default) or "manager" (the venue manager's own
            manual sales estimate).
        n: how many days to return, 1 to 20 (default 10).
    """
    return _get_error_ranking("worst", forecast, n)


def get_top_overpredictions(forecast: str = "model", n: int = 10) -> dict:
    """The days a forecast was too HIGH by the most - actual sales came in
    below it - largest first.

    Args:
        forecast: whose forecast to judge - "model" (this system's own
            forecast, the default) or "manager" (the venue manager's own
            manual sales estimate).
        n: how many days to return, 1 to 20 (default 10).
    """
    return _get_error_ranking("over", forecast, n)


def get_top_underpredictions(forecast: str = "model", n: int = 10) -> dict:
    """The days a forecast was too LOW by the most - actual sales came in
    above it - largest first.

    Args:
        forecast: whose forecast to judge - "model" (this system's own
            forecast, the default) or "manager" (the venue manager's own
            manual sales estimate).
        n: how many days to return, 1 to 20 (default 10).
    """
    return _get_error_ranking("under", forecast, n)


def get_top_best_days(forecast: str = "model", n: int = 10) -> dict:
    """The days a forecast came CLOSEST to what was actually sold - the best
    forecasts, smallest gap first.

    Args:
        forecast: whose forecast to judge - "model" (this system's own
            forecast, the default) or "manager" (the venue manager's own
            manual sales estimate).
        n: how many days to return, 1 to 20 (default 10).
    """
    return _get_error_ranking("best", forecast, n)


TOOL_FUNCTIONS = [
    get_forecast,
    get_forecast_range,
    get_model_comparison,
    get_sales_and_forecast_patterns,
    get_top_worst_days,
    get_top_overpredictions,
    get_top_underpredictions,
    get_top_best_days,
]
TOOLS = {fn.__name__: fn for fn in TOOL_FUNCTIONS}
