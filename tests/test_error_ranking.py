"""
Tests for src/api/error_ranking.py on small hand-built data, where the right
ranking can be stated exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.error_ranking import MAX_RANKING_DAYS, rank_errors

# Ten days from Saturday 1 March 2025, with these signed errors for the model
# (forecast - actual):  +100  -500  0  +400  -1000  +300  0  -800  +900  -600
ACTUAL = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
MODEL = [1100, 1500, 3000, 4400, 4000, 6300, 7000, 7200, 9900, 9400]
# The manager's forecast: very wrong on day 1 (+2000), exact elsewhere.
MANAGER = [3000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


def make_inputs():
    dates = pd.date_range("2025-03-01", periods=10)
    history = pd.DataFrame({"date": dates, "total_sales": ACTUAL, "forecast_sales": MANAGER})
    model_forecasts = {d.date(): float(p) for d, p in zip(dates, MODEL)}
    return history, model_forecasts


def dates_of(response) -> list[str]:
    return [d.date.isoformat() for d in response.days]


def test_worst_ranks_by_size_of_miss_in_either_direction():
    history, model = make_inputs()
    result = rank_errors(history, model, "worst", "model", limit=4)
    assert dates_of(result) == ["2025-03-05", "2025-03-09", "2025-03-08", "2025-03-10"]
    assert [d.gap_gbp for d in result.days] == [1000, 900, 800, 600]


def test_over_lists_only_days_the_forecast_was_too_high():
    history, model = make_inputs()
    result = rank_errors(history, model, "over", "model", limit=10)
    # Only 4 days qualify - the list is not padded with days that were too low or exact.
    assert dates_of(result) == ["2025-03-09", "2025-03-04", "2025-03-06", "2025-03-01"]
    assert {d.direction for d in result.days} == {"over_forecast"}
    assert all(d.forecast > d.actual_sales for d in result.days)


def test_under_lists_only_days_the_forecast_was_too_low():
    history, model = make_inputs()
    result = rank_errors(history, model, "under", "model", limit=10)
    assert dates_of(result) == ["2025-03-05", "2025-03-08", "2025-03-10", "2025-03-02"]
    assert {d.direction for d in result.days} == {"under_forecast"}
    assert all(d.forecast < d.actual_sales for d in result.days)


def test_the_manager_forecast_is_judged_on_its_own_numbers():
    history, model = make_inputs()
    result = rank_errors(history, model, "worst", "manager", limit=3)
    assert result.forecast_judged == "manager"
    assert dates_of(result)[0] == "2025-03-01"  # +2000, the manager's one big miss
    assert result.days[0].forecast == 3000
    assert result.days[0].direction == "over_forecast"
    # Every other day is exact, so the rest of the list is zero-gap days.
    assert [d.gap_gbp for d in result.days[1:]] == [0, 0]
    assert rank_errors(history, model, "under", "manager").days == []


def test_gap_percent_and_weekday_are_provided():
    history, model = make_inputs()
    top = rank_errors(history, model, "worst", "model", limit=1).days[0]
    assert top.gap_percent_of_actual == 20.0  # 1000 off on 5000
    assert top.day_of_week == top.date.strftime("%A") == "Wednesday"
    assert top.actual_sales == 5000 and top.forecast == 4000


def test_known_closure_days_are_left_out_and_reported():
    history, model = make_inputs()
    closure = pd.DataFrame({"date": [pd.Timestamp("2025-12-25")], "total_sales": [0.0], "forecast_sales": [8000.0]})
    history = pd.concat([history, closure], ignore_index=True)
    model = {**model, pd.Timestamp("2025-12-25").date(): 8000.0}

    for kind in ("worst", "over", "under"):
        result = rank_errors(history, model, kind, "model", limit=MAX_RANKING_DAYS)
        assert "2025-12-25" not in dates_of(result)
    assert result.excluded.known_closure_days == 1
    assert [d.isoformat() for d in result.excluded.known_closure_dates] == ["2025-12-25"]
    assert result.days_ranked == 10


def test_days_with_no_sales_figure_are_left_out_and_counted():
    history, model = make_inputs()
    history.loc[4, "total_sales"] = np.nan  # the model's worst day, now with no recorded sales
    result = rank_errors(history, model, "worst", "model", limit=3)
    assert "2025-03-05" not in dates_of(result)
    assert result.excluded.missing_figures == 1
    assert result.days_ranked == 9


def test_only_days_with_a_forecast_made_beforehand_are_ranked():
    history, model = make_inputs()
    earlier = pd.DataFrame({"date": [pd.Timestamp("2025-02-01")], "total_sales": [100.0], "forecast_sales": [9000.0]})
    history = pd.concat([earlier, history], ignore_index=True)  # no model forecast for this one
    for forecast in ("model", "manager"):
        result = rank_errors(history, model, "worst", forecast)
        assert "2025-02-01" not in dates_of(result)
        assert result.period_start.isoformat() == "2025-03-01"
        assert result.period_end.isoformat() == "2025-03-10"


def test_the_limit_is_respected_and_bounded():
    history, model = make_inputs()
    assert len(rank_errors(history, model, "worst", "model", limit=2).days) == 2
    for bad in (0, -1, MAX_RANKING_DAYS + 1):
        with pytest.raises(ValueError):
            rank_errors(history, model, "worst", "model", limit=bad)


def test_unknown_options_are_rejected():
    history, model = make_inputs()
    with pytest.raises(ValueError):
        rank_errors(history, model, "best", "model")
    with pytest.raises(ValueError):
        rank_errors(history, model, "worst", "robot")


def test_the_response_describes_what_was_ranked_in_plain_words():
    history, model = make_inputs()
    result = rank_errors(history, model, "over", "manager")
    assert "manager's own forecast" in result.ranking and "too high" in result.ranking
    assert "2025-03-01" in result.note and "2025-03-10" in result.note
    lowered = (result.ranking + result.note).lower()
    assert "backtest" not in lowered and "sample" not in lowered  # no jargon for the assistant to repeat
