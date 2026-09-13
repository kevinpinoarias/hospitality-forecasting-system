"""
Automated output-evaluation checks for the forecasting assistant (Track B,
step 3).

Formalises the manual testing already done throughout steps 1 and 2 into a
repeatable suite: every response is checked against the REAL data the tool
actually returned in that conversation, not just eyeballed for how it
reads. Five checks, each targeting a real failure mode already observed
(or explicitly designed against) in this project - not hypothetical ones:

1. Numeric grounding  - every real-looking figure in the answer must trace
   back to a real number the tool actually returned (catches invented
   numbers).
2. Weekday naming     - any weekday named in the answer must match the
   real weekday of a date the tool actually discussed (this project found
   a real "Friday" vs "Saturday" bug exactly this way).
3. No jargon leakage  - internal field/source names must never appear
   verbatim in a response (the first tone complaint about this
   assistant).
4. "Why" refusal      - when asked to explain a real-world cause the tool
   never provided, the assistant must admit uncertainty, not invent one.
5. Scenario logic     - one resolved figure when real weather is known,
   two clearly-presented alternatives when it isn't (the least precise
   check here - text-scanning can't perfectly verify framing).
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import ToolMessage

from src.assistant.agent_forecast import _extract_text, build_agent
from src.assistant.model_comparison_data import FINAL_MODEL_MAPE_PCT, FINAL_MODEL_PCT_BETTER_THAN_MANUAL

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

JARGON_TERMS = [
    "historical_weekday_seasonal_average",
    "days_beyond_training_data",
    "rain_data_source",
    "forecast_sales_source",
    "user_provided",
    "prediction_source",
    "made_before_the_day",
    "model_had_seen_the_day",
    "manager_forecast_on_record",
    "backtest",
    "out-of-sample",
    "out of sample",
    "in-sample",
    "day_context",
    "part_of_weekend_trading",
    "within_3_days_of_payday",
    "temperature_vs_previous_fortnight",
    "warm_streak_days",
    "hot_for_scotland",
]

UNCERTAINTY_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "don't know",
    "do not know",
    "can't say why",
    "cannot say why",
    "not sure why",
    "unable to explain",
    "no information about",
    "don't have the details",
    "not enough information",
]

NUMBER_PATTERN = re.compile(r"£?\s?(\d[\d,]*\.?\d*)")
NUMBER_TOLERANCE = 1.0  # absolute tolerance, covers the model's own rounding

# The real figures the assistant is allowed to state that never come from a
# tool call in the current turn - the model's overall evaluated accuracy and
# its lead over the manual forecast, as percentages (see MODEL_INTRO_MESSAGE
# in query_forecast.py). Real and sourced from docs/EXPERIMENT_LOG.md's
# Series 17, just baked into the system prompt rather than requiring a
# get_model_comparison call every time.
KNOWN_MODEL_FIGURES = [FINAL_MODEL_MAPE_PCT, FINAL_MODEL_PCT_BETTER_THAN_MANUAL, 569]


def extract_tool_data(messages) -> list[dict]:
    """Every real tool result seen so far in this conversation, decoded
    from the raw JSON LangChain stores in each ToolMessage. A
    get_forecast_range result is unpacked into one entry per day, so every
    per-date check sees each day exactly as it would a get_forecast
    result."""
    data = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                decoded = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, dict) and isinstance(decoded.get("results"), list):
                data.extend(r for r in decoded["results"] if isinstance(r, dict))
            else:
                data.append(decoded)
    return data


def extract_numeric_claims(text: str) -> list[float]:
    """Numbers that look like a real, precise figure being stated - has a
    £ sign, a thousands comma, or a genuine decimal point (with digits on
    both sides, not just a sentence-ending "."). Deliberately excludes
    bare small integers (a day-of-month, a year in a written-out date)
    which would otherwise cause false "ungrounded number" flags."""
    figures = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(1)
        has_pound = "£" in text[max(0, match.start() - 2):match.start() + 1]
        has_real_decimal = re.search(r"\.\d", raw) is not None
        # A genuine thousands separator is a comma followed by exactly 3
        # digits (14,702) - not just any comma the regex happened to
        # greedily swallow from ordinary sentence punctuation ("2027, we").
        has_thousands_comma = re.search(r"\d,\d{3}(?!\d)", raw) is not None
        if not (has_pound or has_thousands_comma or has_real_decimal):
            continue
        raw = raw.rstrip(".,")  # strip trailing sentence punctuation if present
        try:
            figures.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return figures


def _flatten_numbers(value, out: list[float]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _flatten_numbers(v, out)
    elif isinstance(value, list):
        for v in value:
            _flatten_numbers(v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))


def _find_values_for_key(value, key: str, out: list) -> None:
    """Recursively collect every value stored under `key` anywhere in a
    nested dict/list - unlike a flat `td[key]` lookup, this also reaches
    values nested inside sub-dicts or lists (e.g. get_sales_and_forecast_patterns'
    per-day/per-weekday breakdowns), not just top-level tool-result fields."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                out.append(v)
            _find_values_for_key(v, key, out)
    elif isinstance(value, list):
        for v in value:
            _find_values_for_key(v, key, out)


def check_numeric_grounding(answer_text: str, tool_data: list[dict]) -> list[str]:
    ground_truth: list[float] = []
    for td in tool_data:
        _flatten_numbers(td, ground_truth)

    # Also allow legitimate arithmetic on real numbers - e.g. "the forecast
    # was off by £1,543" is computed from two real, grounded figures
    # (predicted vs actual), not invented, even though it isn't a direct
    # match to any single value the tool returned.
    derived = {abs(a - b) for a in ground_truth for b in ground_truth if a != b}
    allowed = ground_truth + list(derived) + KNOWN_MODEL_FIGURES

    issues = []
    for claimed in extract_numeric_claims(answer_text):
        if not any(abs(claimed - g) <= max(NUMBER_TOLERANCE, g * 0.01) for g in allowed):
            issues.append(f"Ungrounded number in response: {claimed} - no matching figure (or real difference) in the tool data.")
    return issues


def check_weekday_naming(answer_text: str, tool_data: list[dict]) -> list[str]:
    mentioned = [w for w in WEEKDAYS if re.search(rf"\b{w}\b", answer_text)]
    if not mentioned:
        return []

    # same_day_last_week and same_day_last_year are both constructed to
    # match the target date's weekday exactly, so this field covers every
    # date discussed in that turn - searched recursively (not just at the
    # top level) since get_sales_and_forecast_patterns nests day_of_week
    # inside sub-records rather than returning one top-level value.
    real_weekdays_list: list[str] = []
    for td in tool_data:
        _find_values_for_key(td, "day_of_week", real_weekdays_list)
    real_weekdays = set(real_weekdays_list)

    issues = []
    for w in mentioned:
        if w not in real_weekdays:
            issues.append(
                f"Response names '{w}', but no date discussed this turn actually falls on a {w} "
                f"(real weekday(s) involved: {sorted(real_weekdays)})."
            )
    return issues


def check_no_jargon(answer_text: str) -> list[str]:
    lower = answer_text.lower()
    return [f"Jargon leaked into response: '{term}'" for term in JARGON_TERMS if term.lower() in lower]


def check_why_refusal(answer_text: str) -> list[str]:
    lower = answer_text.lower()
    if any(phrase in lower for phrase in UNCERTAINTY_PHRASES):
        return []
    return ["Response to a 'why' question didn't include honest uncertainty language - check it isn't inventing a real-world cause."]


# Deliberately two loose word-lists checked together, not an ever-growing
# list of exact phrases - the model rephrases this honestly in genuinely
# different ways run to run ("actual sales figure on file" vs. "on
# record" vs. "isn't a final actual sales figure"), and chasing each new
# wording with another literal string is a losing game. Requiring one
# topic word plus one negation word near it is far more robust to that
# real paraphrasing than any fixed phrase list.
NO_ACTUAL_SALES_TOPIC_WORDS = ["actual sales", "sales figure", "actual figure"]
NO_ACTUAL_SALES_NEGATION_WORDS = [
    "don't have", "do not have", "doesn't have", "does not have",
    "no record", "not on record", "no data", "not available",
    "can't calculate", "cannot calculate", "unable to", "isn't a", "isn't any",
    "no real", "there's no", "there is no", "can't say", "can't tell",
]


def check_actual_sales_refusal(question: str, answer_text: str, tool_data: list[dict]) -> list[str]:
    """When a question asks about accuracy for a date whose tool result has
    actual_sales: null, the response must plainly say no real figure is on
    record - not silently omit it or (the real bug this targets, found by
    manual testing 2026-09-08) invent a specific number instead."""
    if "accura" not in question.lower():
        return []

    null_actual_sales_dates = [td["date"] for td in tool_data if "actual_sales" in td and td["actual_sales"] is None]
    if not null_actual_sales_dates:
        return []

    lower = answer_text.lower()
    has_topic = any(w in lower for w in NO_ACTUAL_SALES_TOPIC_WORDS)
    has_negation = any(w in lower for w in NO_ACTUAL_SALES_NEGATION_WORDS)
    if has_topic and has_negation:
        return []
    return [
        f"Question asked about accuracy for a date with no actual_sales on record "
        f"({null_actual_sales_dates}), but the response didn't plainly say so."
    ]


CLOSURE_WORDS = ["closed", "closure", "shut"]


def check_closure_day_handling(question: str, answer_text: str, tool_data: list[dict]) -> list[str]:
    """A real bug found by manual testing (2026-09-08): asked about the
    forecast/accuracy for a known closure day (Christmas Day, New Year's
    Day), the assistant presented the model's meaningless prediction as a
    real forecast and the resulting £0 actual sales as a shortfall,
    without ever mentioning the venue was simply closed. If any date
    discussed this turn is a known closure day, the response must mention
    the closure in plain terms."""
    closure_dates = [td["date"] for td in tool_data if td.get("is_known_closure_day")]
    if not closure_dates:
        return []

    lower = answer_text.lower()
    if any(w in lower for w in CLOSURE_WORDS):
        return []
    return [f"Date(s) discussed this turn are known closure days ({closure_dates}), but the response never mentioned the venue being closed."]


def check_scenario_logic(question: str, answer_text: str, tool_data: list[dict]) -> list[str]:
    # An explicit "what if X" question legitimately gets both figures - the
    # asked-about scenario as the answer, the other as comparison context
    # (validated behaviour from step 1's own testing). The "don't split
    # into two scenarios" rule is only for a plain "what's the forecast"
    # question, where the model shouldn't volunteer both unprompted.
    is_explicit_what_if = "what if" in question.lower()

    issues = []
    claimed = extract_numeric_claims(answer_text)
    # A summary of a whole period legitimately gives one figure per day
    # without a weather outlook (the dry-weather one) and mentions rain once,
    # so "present both alternatives" is only required for a few dates.
    few_dates = len({td.get("date") for td in tool_data if "predictions" in td}) <= 3

    for td in tool_data:
        preds = td.get("predictions") or {}
        dry, rain = preds.get("dry_scenario"), preds.get("heavy_rain_scenario")
        if dry is None or rain is None or abs(dry - rain) < 1:
            continue  # scenarios aren't meaningfully distinct, nothing to check

        # A tight, fixed tolerance here deliberately - this check needs to
        # tell two genuinely different real numbers apart (e.g. a scenario
        # figure vs. an unrelated historical comparison that happens to be
        # numerically close), not just confirm a number is real at all,
        # which is what the looser percentage-based tolerance is for.
        dry_mentioned = any(abs(dry - n) <= NUMBER_TOLERANCE for n in claimed)
        rain_mentioned = any(abs(rain - n) <= NUMBER_TOLERANCE for n in claimed)
        weather_known = td.get("weather") is not None

        if weather_known and dry_mentioned and rain_mentioned and not is_explicit_what_if:
            issues.append(
                f"Weather was known for {td['date']}, but response presents both the dry "
                f"({dry:.2f}) and heavy-rain ({rain:.2f}) figures - should resolve to one number."
            )
        if few_dates and not weather_known and not (dry_mentioned and rain_mentioned):
            issues.append(
                f"Weather was unknown for {td['date']}, but response doesn't clearly present "
                f"both the dry ({dry:.2f}) and heavy-rain ({rain:.2f}) alternatives."
            )
    return issues


TEST_CASES = [
    {"name": "how the forecast was made", "turns": ["How is the forecast for 2026-09-12 calculated?"]},
    {"name": "what if it rains", "turns": ["What if it rains on 2026-09-12?"]},
    {"name": "multi-day summary", "turns": ["What's the forecast for Sept 9, 10 and 11 2026?"]},
    {"name": "historical comparison", "turns": ["Is 2026-09-12 busier than a typical Saturday?"]},
    {
        "name": "accuracy check + why-refusal",
        "turns": [
            "How accurate was our forecast for 2025-12-20?",
            "Why was it off by that much?",
        ],
    },
    {"name": "far future - both scenarios", "turns": ["What's the forecast for 2027-06-01?"]},
    {
        "name": "day context - payday and holidays",
        # The day after an early (Friday) payday, which the model's own
        # payday feature counts from the previous month - the context must
        # give the real "1 day after payday".
        "turns": ["What's the forecast for 2026-10-31, and is there anything special about that day?"],
    },
    {
        "name": "several days - Christmas week",
        "turns": ["Give me the forecast for 21 to 27 December 2026 and point out anything unusual about those days."],
    },
    {
        "name": "accuracy check with no actual_sales on record",
        # A real, live bug found by manual testing (2026-09-08): the
        # assistant invented a specific "actual sales" figure for a date
        # this deployment has no real actual_sales for (any date past the
        # historical dataset's cutoff, 2026-09-06 since the 2026-09-13
        # retrain). 2026-09-10 returns actual_sales: null, so this case
        # exists specifically to catch that failure mode from ever
        # regressing silently.
        "turns": ["How accurate was our forecast for 2026-09-10?"],
    },
    {
        "name": "model comparison - neural network results",
        "turns": ["How did the neural network models compare to XGBoost, and how did they do on spike days?"],
    },
    {
        "name": "accuracy check on a known closure day",
        # Christmas Day and New Year's Day - the venue is confirmed always
        # closed. A real bug found by manual testing (2026-09-08): the
        # assistant presented the model's meaningless forecast as real and
        # the resulting £0 actual as a shortfall, with no mention of the
        # closure.
        "turns": ["How was the forecast for January 1st 2026 compared to the actual sales?"],
    },
    {
        "name": "sales patterns - day of week and best/worst records",
        "turns": ["Which day of the week earns the most on average, and which single day had the highest sales ever?"],
    },
    {
        "name": "sales patterns - week and month",
        "turns": ["What was our best week for sales, and is December a strong month for us?"],
    },
]


def run_turn_checks(question: str, answer_text: str, tool_data: list[dict]) -> list[str]:
    issues = []
    issues += check_numeric_grounding(answer_text, tool_data)
    issues += check_weekday_naming(answer_text, tool_data)
    issues += check_no_jargon(answer_text)
    issues += check_scenario_logic(question, answer_text, tool_data)
    issues += check_actual_sales_refusal(question, answer_text, tool_data)
    issues += check_closure_day_handling(question, answer_text, tool_data)
    if "why" in question.lower():
        issues += check_why_refusal(answer_text)
    return issues


def run_evaluation() -> None:
    agent = build_agent()
    total_turns = 0
    total_issues = 0

    for case_idx, case in enumerate(TEST_CASES):
        config = {"configurable": {"thread_id": f"eval-{case_idx}"}}
        print(f"\n=== {case['name']} ===")

        for question in case["turns"]:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
            answer_text = _extract_text(result["messages"][-1].content)
            tool_data = extract_tool_data(result["messages"])

            issues = run_turn_checks(question, answer_text, tool_data)
            total_turns += 1
            total_issues += len(issues)

            status = "PASS" if not issues else f"FAIL ({len(issues)} issue(s))"
            print(f"  Q: {question}")
            print(f"  [{status}]")
            for issue in issues:
                print(f"    - {issue}")

    print(f"\n{'=' * 60}")
    print(f"Checked {total_turns} turns across {len(TEST_CASES)} test cases.")
    print(f"Total issues found: {total_issues}")


if __name__ == "__main__":
    run_evaluation()
