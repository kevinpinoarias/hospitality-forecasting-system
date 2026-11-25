"""
Input checks for the MCP tools, done before anything is sent to the API, so
a mistake comes back as a message that says what is wrong and how to fix it
instead of a bare HTTP error.

Only the shape of the input is checked here (a real date, an end not before
the start, a positive sales figure). What the data allows - for example a
date before the historical data starts - is still decided by the API, and
its own message is passed on (see server.py).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from src.assistant.tools import MAX_RANGE_DAYS

MODEL_CHOICES_FILE = Path(__file__).resolve().parents[2] / "config" / "model_versions.json"


def model_choice_ids() -> list[str]:
    """The model picker's choices (the standard model and the forecast log's versions), as the API lists them."""
    return [c["id"] for c in json.loads(MODEL_CHOICES_FILE.read_text(encoding="utf-8"))["choices"]]


def check_model_version(value: str | None) -> None:
    if value is None:
        return
    choices = model_choice_ids()
    if value not in choices:
        raise ToolError(f"model_version must be one of {', '.join(choices)}; got {value!r}. Leave it out for the standard (live) model.")


def check_date(value: str, field: str = "date") -> dt.date:
    """A real calendar date written as YYYY-MM-DD."""
    text = str(value).strip()
    try:
        if len(text) != 10:
            raise ValueError
        return dt.date.fromisoformat(text)
    except ValueError:
        raise ToolError(
            f"{field} must be a real date written as YYYY-MM-DD (for example 2026-12-24); got {value!r}."
        ) from None


def check_range(start_date: str, end_date: str) -> tuple[dt.date, dt.date]:
    start = check_date(start_date, "start_date")
    end = check_date(end_date, "end_date")
    if end < start:
        raise ToolError(f"end_date ({end}) is before start_date ({start}); swap them or pick a later end_date.")
    days = (end - start).days + 1
    if days > MAX_RANGE_DAYS:
        raise ToolError(
            f"The range {start} to {end} covers {days} days; at most {MAX_RANGE_DAYS} can be asked for in one call. "
            "Split it into shorter periods."
        )
    return start, end


def check_sales_estimate(value: float | None) -> None:
    if value is None:
        return
    if not value > 0:
        raise ToolError(f"forecast_sales must be a sales figure above 0 in pounds; got {value!r}. Leave it out if there is none.")
