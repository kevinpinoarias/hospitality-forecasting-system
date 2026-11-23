"""
The MCP server: the assistant's eight tools, offered to MCP clients.

Each MCP tool below is a thin wrapper. It checks the input (validation.py),
calls the SAME function the Gemini assistant uses (src/assistant/tools.py)
and returns that function's result unchanged - no forecasting logic is
copied or rewritten here. tests/test_mcp_server.py checks the results match.

The tool names are the assistant's own, so the Gradio agent can later load
these tools through MCP (Phase 2) without any change to its system prompt.

What the client's AI is told comes from each tool's docstring (its
description) and the parameter descriptions. An MCP client never sees the
Gemini assistant's system prompt, so the rules that keep answers honest
(closure days, one figure or two depending on the weather, never inventing
a reason) are written into the descriptions here.
"""

from __future__ import annotations

from typing import Annotated, Literal

import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from src.assistant import tools
from src.mcp_server.call_log import configure_stderr_logging, logged_tool
from src.mcp_server.validation import check_date, check_range, check_sales_estimate

SERVER_NAME = "hospitality-forecasting"

# Sent once when a client connects. A chat that knows nothing about the project
# reads "what's the forecast for Christmas Eve?" as a weather question (seen in
# the first Claude Desktop test, 21 November 2026), so this says what the whole
# server is for. Not every client passes it on, so the forecast tools'
# descriptions say it too (VENUE_SALES below).
SERVER_INSTRUCTIONS = (
    "These tools give the SALES forecasts, in pounds (GBP), for one hospitality venue (a bar-restaurant), "
    "and how accurate past forecasts were. When the person asks about a forecast for a date or period, how busy "
    "a day will be or how the venue will do, they mean sales: use these tools, not a weather forecast. Weather "
    "only appears as one of the inputs to a sales forecast. Only state figures that came from a tool result, and "
    "never invent a reason why sales were higher or lower."
)

mcp = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, log_level="WARNING")

# Every tool only reads; the forecast tools reach out to the forecasting API.
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)
READ_ONLY_LOCAL = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

DateText = Annotated[str, Field(description="A date written as YYYY-MM-DD, e.g. 2026-12-24.")]
Forecast = Annotated[
    Literal["model", "manager"],
    Field(description='Whose forecast to judge: "model" (this system\'s forecast) or "manager" (the venue manager\'s own manual estimate).'),
]
Days = Annotated[int, Field(description="How many days to return, 1 to 20. More than 20 is capped at 20 and the result says so.")]

FORECAST_READING_RULES = """
How to read the result:
- day_of_week: always use this to name the weekday; never work it out from the date.
- is_known_closure_day true (Christmas Day, New Year's Day): the venue is closed. The predictions are not a real
  forecast and actual_sales of 0 is the closure, not a miss. Say so and do not quote the figures.
- weather present: the weather outlook is known - give ONE figure, predictions.best_estimate, and describe the
  weather with expected_rain_description and expected_max_temp_c.
- weather null: too far ahead for an outlook - give both predictions.dry_scenario and predictions.heavy_rain_scenario
  as two plain possibilities.
- actual_sales is the only source for what really sold. If it is null, there is no figure on record: never estimate one.
- prediction_source "made_before_the_day": comparing with actual_sales is a fair accuracy check.
  "model_had_seen_the_day": the model had already learned from that day, so it is not a fair check - say so.
- day_context lists what the model took into account (payday, bank and school holidays, weekend trading, recent
  sales). These are inputs, not measured causes: never claim one caused a figure or a gap.
- Never give a reason why sales were higher or lower than forecast unless the result itself states it.
- Amounts are in pounds (GBP)."""

RANKING_READING_RULES = """
How to read the result: each day has actual_sales, forecast (the forecast being judged), gap_gbp (always positive)
and direction: "over_forecast" = the forecast was too HIGH, "under_forecast" = too LOW, "exact" = matched. Always say
whose forecast was judged (forecast_judged) and the period covered (period_start to period_end); only those days were
ranked. Christmas Day and New Year's Day are left out because the venue is closed. If limit_note is present, mention
it. Never invent a reason a day was missed. Amounts are in pounds (GBP)."""


VENUE_SALES = "Sales forecast in pounds for the hospitality venue - not a weather forecast. "


def _describe(fn, rules: str = "", lead: str = "") -> str:
    """The tool's description for MCP clients: an optional lead-in, the
    assistant's own, tested wording (the part of the docstring before "Args:" -
    the parameters are described separately in the schema), and the reading
    rules. The lead-in exists only here, so the live Gemini assistant's tool
    descriptions stay exactly as evaluated."""
    summary = " ".join(fn.__doc__.split("Args:")[0].split())
    return lead + summary + ("\n" + rules if rules else "")


def _call_api(fn, *args, **kwargs) -> dict:
    """Calls one of the assistant's tool functions and turns API failures into
    messages a person can act on."""
    try:
        return fn(*args, **kwargs)
    except requests.HTTPError as exc:
        response = exc.response
        detail = None
        if response is not None:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
        if response is not None and response.status_code == 422 and detail:
            raise ToolError(f"The forecasting API could not answer for that input: {detail}") from None
        status = response.status_code if response is not None else "unknown"
        raise ToolError(f"The forecasting API returned an error (HTTP {status}). Try again in a minute.") from None
    except (requests.ConnectionError, requests.Timeout):
        raise ToolError(
            "The forecasting API could not be reached. It scales to zero when idle, so the first call can take up "
            "to a minute while it starts - try again shortly."
        ) from None


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_forecast, FORECAST_READING_RULES, lead=VENUE_SALES))
@logged_tool
def get_forecast(
    date: DateText,
    forecast_sales: Annotated[
        float | None,
        Field(description="The manager's own sales estimate for that day in pounds, ONLY if the person asking gave one. Leave it out otherwise."),
    ] = None,
) -> dict:
    check_date(date)
    check_sales_estimate(forecast_sales)
    return _call_api(tools.get_forecast, date, forecast_sales)


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_forecast_range, FORECAST_READING_RULES, lead=VENUE_SALES))
@logged_tool
def get_forecast_range(
    start_date: Annotated[str, Field(description="The first date, YYYY-MM-DD.")],
    end_date: Annotated[str, Field(description="The last date (inclusive), YYYY-MM-DD; at most 31 days from start_date.")],
) -> dict:
    check_range(start_date, end_date)
    return _call_api(tools.get_forecast_range, start_date, end_date)


@mcp.tool(annotations=READ_ONLY_LOCAL, description=_describe(tools.get_model_comparison))
@logged_tool
def get_model_comparison() -> dict:
    return tools.get_model_comparison()


@mcp.tool(annotations=READ_ONLY_LOCAL, description=_describe(tools.get_sales_and_forecast_patterns))
@logged_tool
def get_sales_and_forecast_patterns() -> dict:
    return tools.get_sales_and_forecast_patterns()


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_top_worst_days, RANKING_READING_RULES))
@logged_tool
def get_top_worst_days(forecast: Forecast = "model", n: Days = 10) -> dict:
    return _call_api(tools.get_top_worst_days, forecast, n)


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_top_overpredictions, RANKING_READING_RULES))
@logged_tool
def get_top_overpredictions(forecast: Forecast = "model", n: Days = 10) -> dict:
    return _call_api(tools.get_top_overpredictions, forecast, n)


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_top_underpredictions, RANKING_READING_RULES))
@logged_tool
def get_top_underpredictions(forecast: Forecast = "model", n: Days = 10) -> dict:
    return _call_api(tools.get_top_underpredictions, forecast, n)


@mcp.tool(annotations=READ_ONLY, description=_describe(tools.get_top_best_days, RANKING_READING_RULES))
@logged_tool
def get_top_best_days(forecast: Forecast = "model", n: Days = 10) -> dict:
    return _call_api(tools.get_top_best_days, forecast, n)


def main() -> None:
    configure_stderr_logging()
    mcp.run(transport="stdio")
