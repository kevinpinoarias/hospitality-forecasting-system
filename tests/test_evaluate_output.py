"""
Tests for check_scenario_logic in src/assistant/evaluate_output.py - the
check that an answer gives one figure when the weather is known and both
the dry and heavy-rain figures when it isn't.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from src.assistant.evaluate_output import check_scenario_logic

WEATHER = {"expected_rain_mm": 8.0, "expected_rain_description": "heavy rain"}


def day(weather, last_week=11000.0, dry=14144.70, rain=13775.97):
    return {
        "date": "2026-09-12",
        "weather": weather,
        "predictions": {"dry_scenario": dry, "heavy_rain_scenario": rain, "best_estimate": rain},
        "historical_comparison": {"same_day_last_week": last_week, "same_day_last_year": 15277.0},
    }


def test_a_known_weather_day_should_give_one_figure():
    answer = "Dry it would be about £14,145, but with the heavy rain expect £13,776."
    issues = check_scenario_logic("What's the forecast?", answer, [day(WEATHER)])
    assert len(issues) == 1 and "should resolve to one number" in issues[0]


def test_one_figure_for_a_known_weather_day_is_fine():
    answer = "Expect around £13,776 with the heavy rain."
    assert check_scenario_logic("What's the forecast?", answer, [day(WEATHER)]) == []


def test_last_weeks_sales_close_to_the_dry_figure_are_not_mistaken_for_it():
    # A real run: last week's sales (£14,142.71) were within £2 of the dry
    # scenario (£14,144.70), so quoting them looked like quoting the dry forecast.
    answer = "The forecast is £13,776, a little under last Saturday's £14,143."
    td = day(WEATHER, last_week=14142.71)
    assert check_scenario_logic("Is it busier than a typical Saturday?", answer, [td]) == []


def test_the_dry_figure_still_counts_when_it_is_the_closer_match():
    answer = "Dry it would be £14,145, but with the heavy rain £13,776."
    td = day(WEATHER, last_week=14142.71)
    assert len(check_scenario_logic("What's the forecast?", answer, [td])) == 1


def test_an_explicit_what_if_may_give_both_figures():
    answer = "If it stays dry about £14,145; with heavy rain about £13,776."
    assert check_scenario_logic("What if it rains?", answer, [day(WEATHER)]) == []


def test_an_unknown_weather_day_needs_both_figures():
    only_dry = "Expect about £14,145."
    issues = check_scenario_logic("What's the forecast?", only_dry, [day(None)])
    assert len(issues) == 1 and "doesn't clearly present both" in issues[0]
    both = "About £14,145 if dry, or about £13,776 with heavy rain."
    assert check_scenario_logic("What's the forecast?", both, [day(None)]) == []
