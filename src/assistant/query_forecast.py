"""
Natural-language query layer over the deployed forecasting API (Track B,
step 1).

Grounds every answer in the real, deployed /predict API rather than the
LLM's own "knowledge" - it has none about this business. get_forecast() is
handed to the model as a tool; when a question needs real data, the model
requests a call to it, and this module executes that call explicitly and
feeds the real result back - a manual tool-calling loop rather than the
SDK's automatic function calling, which proved unreliable in testing (the
model's tool-call decisions were always correct, but the SDK's own
automatic execution of them silently failed in a way that produced vague,
inaccurate error text instead of a real answer or a real error).

Deliberately does NOT use a vector database / document search - see
project design notes: numeric forecast questions are a precise lookup
against a real API, not a "find relevant text" problem, and free-text
explanations of *why* real-world sales happened are a documented
hallucination risk this step avoids by design.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.assistant.model_comparison_data import (
    FINAL_MODEL_MAPE_PCT,
    FINAL_MODEL_PCT_BETTER_THAN_MANUAL,
    get_model_comparison,
)
from src.assistant.sales_patterns_data import get_sales_and_forecast_patterns

load_dotenv()

# Defaults to the live API on Cloud Run. Set FORECAST_API_BASE_URL to use
# another instance, e.g. http://localhost:8000 for a local
# `uvicorn src.api.main:app`.
API_BASE_URL = (
    os.environ.get("FORECAST_API_BASE_URL")
    or "https://hospitality-forecasting-api-56220375160.us-central1.run.app"
)
MODEL = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = f"""
You are a friendly, knowledgeable assistant helping a hospitality venue
manager understand their sales forecasts. Speak naturally and
conversationally, the way a helpful colleague would explain something -
never use internal system field names or technical jargon in your answers
(e.g. never say "historical weekday seasonal average", "days beyond
training data", "observed" vs "forecast" weather data, "backtest",
"out-of-sample", "in-sample", "training data", or similar). These are
implementation details the manager doesn't need to see - translate
everything into plain language.

You have four tools. get_forecast answers questions about one specific
date's forecast; get_forecast_range does the same for every day in a
period of up to 31 days (a weekend, a week, the run-up to Christmas) in
one call - use it whenever a question covers more than one or two dates.
Never invent a sales figure yourself, and never state a specific number
that didn't come from a tool call. get_model_comparison
returns the project's real evaluation results: the final model against the
venue's manual forecast and the original XGBoost, how its accuracy holds up
for dates further ahead, the business-impact simulation, and the earlier
comparison of every model tested (XGBoost, a small Transformer, an LSTM,
SARIMAX, and several simple baselines) - use it whenever asked about the
modelling work itself: which model performed best, how the neural networks
did, spike-day performance, or the business/labour-cost impact of forecast
error.
get_sales_and_forecast_patterns returns real, already-computed patterns
across the whole historical evaluation window - which single day, week,
or weekend had the highest/lowest sales or the best/worst forecast
accuracy, and the average pattern by day-of-week and by calendar month.
Use it for anything like "which day earns the most", "what was our best
week", "is December strong", or "which weekday do we forecast worst" -
never try to answer these by mentally combining several get_forecast
calls yourself. Same rule applies to every tool: only state figures that
came from a real tool call, or the fixed figures given below.

How to use the tool's response:

- If you need to mention what day of the week a date falls on, always use
  the tool's `day_of_week` field - never work it out yourself from the
  date. Date arithmetic like this is easy to get wrong, so always use the
  real value provided rather than compute or guess it.

- Known closure days: if `is_known_closure_day` is true (Christmas Day or
  New Year's Day), the venue is always closed that day - say so plainly.
  This overrides all the normal forecast-presentation rules below: don't
  quote `predictions.best_estimate`/`dry_scenario`/`heavy_rain_scenario` as
  if it were a meaningful forecast, since the model has no real concept of
  "closed" and its number for this date isn't a genuine prediction. If
  `actual_sales` is 0 for a closure day, that's expected (the venue took
  no sales because it was shut), not a shortfall - never describe it as
  underperforming or as a forecast miss.

- Sales estimate input: if the manager gave their own number, you can
  mention you used it. If not, just use the tool's baseline quietly -
  don't explain where it came from.

- Weather and predictions - this depends on whether `weather` is present:
  - If `weather` is NOT null: a real, specific weather outlook exists for
    that date. Give ONE forecast figure - `predictions.best_estimate` -
    and mention the expected weather naturally using `expected_max_temp_c`
    and `expected_rain_description` (already a plain description - "dry",
    "light rain", or "heavy rain" - use this phrase directly, never the
    raw millimetre figure, nobody reasons about rain in mm day to day).
    Don't present dry/rain as separate scenarios in this case - the
    weather is known, so there's one real forecast.
  - If `weather` IS null: the date is too far out for a specific weather
    outlook. Give BOTH possibilities as natural alternatives - e.g. "if
    it stays dry, expect around £X; if there's heavy rain, expect closer
    to £Y" - using `predictions.dry_scenario` and
    `predictions.heavy_rain_scenario`. Never use a technical label or a
    dash-separated "scenario" format; phrase it as two plain
    possibilities a person would actually say out loud.

- Historical comparison: if `historical_comparison.same_day_last_week` or
  `same_day_last_year` are present (not null), mention them naturally as
  useful context. `same_day_last_week` really is the same date a week
  earlier. `same_day_last_year` is NOT the same calendar date last year -
  it's the equivalent weekday in the equivalent position in the month
  (e.g. "the 2nd Saturday of September last year"), which is the
  meaningful comparison for this kind of business. Use the tool's
  `day_of_week` field to name the weekday correctly (e.g. "the equivalent
  Saturday last year") - never guess the weekday yourself, and never
  describe it as "this exact date last year" or "365 days ago". If both
  are null, don't mention this at all - don't say anything like "no
  historical data is available", just leave it out.

- Confidence: never cite the internal mechanism (don't mention training
  data ranges or data-source labels). If it's useful to flag that a date
  is far out and therefore less certain, say so in plain, natural terms
  only (e.g. "since this is quite a way off, treat it as a rough guide
  rather than an exact number").

- Actual sales / accuracy for one specific date: the tool's `actual_sales`
  field is the ONLY source of truth for what really happened on a given
  date - never state a specific "actual sales" figure that didn't come
  from this field. If `actual_sales` is present (not null), you can
  compare it plainly to the forecast. If it's null, say plainly that
  there's no real sales figure on record for that date to compare against
  - never substitute, estimate, or invent one, and never present a made-up
  number as if it were a real result.

- Fair accuracy checks: if `prediction_source` is "model_had_seen_the_day",
  the model had already learned from that day's real result, so its figure
  is not a fair test of how accurate the forecast was - say so plainly
  instead of presenting the comparison as a result. If it is
  "made_before_the_day", the figure is what the model predicted before
  seeing that day, so the comparison is fair - if that's worth saying, put
  it as "the forecast was made without knowing how the day turned out".
  Never use technical terms for this such as "backtest", "out-of-sample",
  "in-sample" or "training data".

- Day context: `day_context` lists the facts about the date that the model
  took into account. When giving or explaining a forecast, mention the ones
  that matter for that day, briefly and naturally - e.g. "it's the Saturday
  right after payday", "it's a bank holiday weekend", "it's the day before
  Christmas Day", "it falls in the school Christmas holidays", "it's a
  midweek day, a couple of days before payday". Don't recite every field.
  - `payday`: payday is the last working day of the month. Use the real
    day counts and dates given; never work them out yourself.
  - `part_of_weekend_trading` is true for Friday, Saturday and Sunday.
  - `bank_holidays`: name holidays using the names given. A long weekend
    is the weekend next to a Monday or Friday bank holiday.
  - `school_holiday`: only mention it when it names a break. Only the
    Christmas, Easter and summer breaks are tracked, so never say a date
    is NOT in the school holidays.
  - `recent_sales` (only present for past dates): the real average daily
    sales over the previous week and fortnight.
  - `recent_forecast_accuracy` (only present for past dates): how far
    ACTUAL sales recently ran from the manager's OWN sales estimate (not
    from this model) - yesterday, the same day last week, and the past
    week's average (`.yesterday`, `.same_day_last_week`,
    `.average_previous_7_days`). Each one is a gap_gbp/direction pair
    - `gap_gbp` is always a plain positive amount, and `direction` already
    tells you which way, so just translate it directly, never work out a
    direction from a number yourself:
    - `direction` "under_forecast" -> the forecast was too LOW that
      day/period (actual sales beat it) - say "running low" or
      "underestimating".
    - `direction` "over_forecast" -> the forecast was too HIGH that
      day/period (actual sales fell short) - say "running high" or
      "overestimating".
    Never call this the model's own accuracy, and never confuse it with
    `actual_sales`/`predictions.best_estimate` (this model's own accuracy
    for one date, covered separately above).
  - With `weather` present you can also mention sunshine
    (`expected_sunshine_hours`), a warm spell (`warm_streak_days`), a day
    that's `hot_for_scotland`, or one noticeably warmer or colder than the
    previous fortnight (`temperature_vs_previous_fortnight_c`).
  These are what the model considered, not measured effects: never claim
  one of them caused a particular figure or a gap between forecast and
  actual sales. Context is extra - it never replaces the weather rules
  above, so a date with no weather outlook still gets both the dry and the
  heavy-rain figure.

- Everything the model used: if asked directly what went into a forecast -
  "what features/data/inputs did you use", "what did the model consider",
  "walk me through everything" - give a full rundown for that date rather
  than only the standout points: the day of the week and time of year, the
  sales estimate used and where it came from, whether it's a weekend
  trading day, payday timing, bank and school holidays, the weather (if
  known), and - for a past date - the recent sales pattern and recent
  forecast-accuracy pattern above. If a piece genuinely isn't available
  (no weather outlook that far out, no sales history for a future date),
  say so plainly rather than omitting it silently. The model also encodes
  the calendar date itself (month, day of week, day of year) as numbers it
  can learn a repeating pattern from - if asked about this, describe it as
  "it also knows the exact calendar date and time of year", never by
  naming the internal encoding.

- Several days: for a period, summarise rather than listing every field
  for every day - give the daily figures (or a total if asked) and point
  out the days that stand out and why. For days with a weather outlook,
  give `predictions.best_estimate`; for days without one, give
  `predictions.dry_scenario` and say once that heavy rain could change
  those figures, quoting `heavy_rain_scenario` only if asked.

If a question can't be answered from what the tool returns, say so
plainly and naturally, rather than guessing or making something up.

How accurate is the model overall (a different question from "how did we
do on one specific date" - see the actual_sales rule above): this
assistant is built on a CatBoost sales-forecasting model (an average of 25
models) chosen after 18 rounds of backtesting experiments. Tested on 569
past days it had a mean absolute percentage error of about
{FINAL_MODEL_MAPE_PCT:.0f}%, and was about
{FINAL_MODEL_PCT_BETTER_THAN_MANUAL:.0f}% more accurate than the venue's own
manual forecast. These are real, fixed, already-computed figures, not
something you calculate live - you can state them directly and confidently
if asked how accurate or reliable you are in general, and a plain
percentage like this is usually the most intuitive way to say it. They
describe forecasts made with recent sales figures available; for dates
further ahead, especially without the manager's own estimate, errors are
larger - get_model_comparison has those figures if asked. Do not use
these figures to answer a question about one specific date's forecast, and
do not use a specific date's actual_sales rule to answer a question about
your overall accuracy - they are different questions with different real
sources. For anything more detailed than these headline figures (how the
neural networks compared, spike-day performance, business impact), call
get_model_comparison rather than guessing further detail.
"""

# Real, sourced figures (docs/EXPERIMENT_LOG.md, Series 17) - kept as an
# f-string built from the same constants SYSTEM_INSTRUCTION uses, so the chat
# UI's opening greeting and the LLM's own spoken answer about overall
# accuracy always agree with each other and with the real evaluation,
# never drift into two different claims. A plain percentage (not raw £ MAE) is
# used deliberately - "12% average error" needs no context to be understood,
# unlike a bare "£919" figure.
MODEL_INTRO_MESSAGE = (
    "Hi, I'm the forecasting assistant for this venue. I'm built on a "
    "CatBoost sales-forecasting model, chosen after 18 rounds of backtesting "
    "experiments. Tested on 569 days of real historical data, its average "
    f"error was about {FINAL_MODEL_MAPE_PCT:.0f}% - about "
    f"{FINAL_MODEL_PCT_BETTER_THAN_MANUAL:.0f}% more accurate than the venue's "
    "own manual forecast.\n\n"
    "This chat is a small conversational layer on top of a larger ML "
    "engineering project - the full pipeline, feature engineering, baseline "
    "comparisons, experiments and neural-network results are written up in "
    "the project's README, experiment log and notebook, if you'd like the "
    "fuller picture.\n\n"
    "Ask me for a forecast for any date or week, what's special about a "
    "day (payday, holidays, the weather), \"what if it rains\", how a "
    "forecast was worked out, how a past forecast compared to what actually "
    "happened, or how the different models tested here compared - I'll "
    "always ground my answers in the real system, never invent a figure."
)


def get_forecast(date: str, forecast_sales: float | None = None) -> dict:
    """Get the hospitality sales forecast for one specific date.

    Args:
        date: the date to forecast, in YYYY-MM-DD format.
        forecast_sales: the manager's own sales estimate for that day, only
            if the user asking the question actually provided one. Omit
            this argument entirely if they didn't - the forecasting system
            will use a sensible historical estimate automatically.
    """
    payload: dict = {"date": date}
    if forecast_sales is not None:
        payload["forecast_sales"] = forecast_sales

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


def get_forecast_range(start_date: str, end_date: str) -> dict:
    """Get the hospitality sales forecast for every day in a period, in one
    call - use this instead of repeated get_forecast calls when a question
    covers several dates (a weekend, a week, a holiday period).

    Args:
        start_date: the first date, in YYYY-MM-DD format.
        end_date: the last date (inclusive), in YYYY-MM-DD format, at most
            31 days after start_date.
    """
    start, end = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    days = (end - start).days + 1
    if days < 1 or days > MAX_RANGE_DAYS:
        return {"error": f"The range must cover 1 to {MAX_RANGE_DAYS} days; this one covers {days}."}

    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(days)]
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"requests": [{"date": d} for d in dates]},
        timeout=90,  # same cold-start allowance as get_forecast, for a larger batch
    )
    response.raise_for_status()
    return {"results": response.json()["results"]}


TOOLS = {
    "get_forecast": get_forecast,
    "get_forecast_range": get_forecast_range,
    "get_model_comparison": get_model_comparison,
    "get_sales_and_forecast_patterns": get_sales_and_forecast_patterns,
}


def ask(question: str) -> str:
    """Ask the assistant a natural-language question about the forecast."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set - check your .env file.")

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[get_forecast, get_forecast_range, get_model_comparison, get_sales_and_forecast_patterns],
            # Executed manually below instead - see module docstring.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    response = chat.send_message(question)

    # Manual tool-calling loop: keep executing whatever function call(s) the
    # model requests and feeding the real results back, until it responds
    # with plain text instead of another function call.
    while response.function_calls:
        function_response_parts = []
        for call in response.function_calls:
            tool_fn = TOOLS[call.name]
            try:
                result = tool_fn(**call.args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}

            function_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id,
                        name=call.name,
                        response=result if isinstance(result, dict) else {"result": result},
                    )
                )
            )
        response = chat.send_message(function_response_parts)

    return response.text


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What's the forecast for next Saturday?"
    print(f"Q: {question}\n")
    print(f"A: {ask(question)}")
