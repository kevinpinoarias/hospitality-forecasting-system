"""
Tests for the labour business-impact metrics in
src/models/evaluate_human_forecast.py, particularly
calculate_implied_labour_metrics - the forecast-side-only methodology
that avoids the "actual wages/hours already reflect operational
correction" confound in calculate_labour_business_metrics.
"""

from __future__ import annotations

import pandas as pd

from src.models.evaluate_human_forecast import (
    calculate_implied_labour_metrics,
    calculate_labour_business_metrics,
)


def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_implied_wage_ratio_is_forecast_only_not_actual():
    """Two days with identical forecasts but very different actuals (as if
    staff were sent home on one of them) must produce the same implied
    ratio - it must never be influenced by actual wages/hours."""
    df = make_df([
        {"forecast_sales": 1000, "total_sales": 1000, "forecast_wages": 300, "total_wages": 300,
         "forecast_hours": 20, "total_hours": 20},
        {"forecast_sales": 1000, "total_sales": 600, "forecast_wages": 300, "total_wages": 150,
         "forecast_hours": 20, "total_hours": 10},
    ])
    result = calculate_implied_labour_metrics(df)
    assert result["Forecast wage £ needed per £1 sale"] == 0.3
    assert result["Forecast hours needed per £1 sale"] == 0.02


def test_implied_overforecast_wages_applies_ratio_to_sales_gap_only():
    df = make_df([
        {"forecast_sales": 1000, "total_sales": 600, "forecast_wages": 300, "total_wages": 150,
         "forecast_hours": 20, "total_hours": 10},
    ])
    result = calculate_implied_labour_metrics(df)
    # ratio = 300/1000 = 0.3; overforecast_sales = 1000-600 = 400; implied = 400*0.3 = 120
    assert result["Implied wages allocated to over-forecasted sales (£)"] == 120.0
    # hours ratio = 20/1000 = 0.02; implied hours = 400*0.02 = 8
    assert result["Implied hours allocated to over-forecasted sales"] == 8.0


def test_zero_forecast_sales_days_excluded_from_ratio():
    """A closure day (forecast_sales=0) must not blow up the ratio via
    division by zero, and must not distort it either."""
    df = make_df([
        {"forecast_sales": 0, "total_sales": 0, "forecast_wages": 0, "total_wages": 0,
         "forecast_hours": 0, "total_hours": 0},
        {"forecast_sales": 1000, "total_sales": 1000, "forecast_wages": 300, "total_wages": 300,
         "forecast_hours": 20, "total_hours": 20},
    ])
    result = calculate_implied_labour_metrics(df)
    assert result["Forecast wage £ needed per £1 sale"] == 0.3


def test_days_over_plan_counts_actual_exceeding_forecast_hours():
    df = make_df([
        {"forecast_sales": 1000, "total_sales": 1000, "forecast_wages": 300, "total_wages": 300,
         "forecast_hours": 20, "total_hours": 25},  # over plan
        {"forecast_sales": 1000, "total_sales": 1000, "forecast_wages": 300, "total_wages": 300,
         "forecast_hours": 20, "total_hours": 15},  # under plan
    ])
    result = calculate_implied_labour_metrics(df)
    assert result["Days actual hours exceeded the planned rota"] == 1
    assert result["% of days actual hours exceeded the planned rota"] == 50.0
    assert result["Total hours worked beyond the planned rota"] == 5.0


def test_implied_method_can_exceed_net_of_correction_method():
    """The whole point of the implied method: when a venue corrects
    aggressively (actual wages far below forecast), the net-of-correction
    method's 'waste' figure balloons, while the implied method stays tied
    to the real, uncorrectable sales miss - regression-guarding the
    exact effect this method was built to fix."""
    df = make_df([
        {"forecast_sales": 1000, "total_sales": 900, "forecast_wages": 300, "total_wages": 100,
         "forecast_hours": 20, "total_hours": 8},
    ])
    net_of_correction = calculate_labour_business_metrics(df)
    implied = calculate_implied_labour_metrics(df)

    # Net-of-correction: forecast_wages(300) - actual_wages(100) = 200 "waste"
    assert net_of_correction["Total overforecasted wages, net of correction (£)"] == 200.0
    # Implied: ratio 0.3 * overforecast_sales(100) = 30 - far smaller, and
    # correctly reflects that the sales miss itself was small.
    assert implied["Implied wages allocated to over-forecasted sales (£)"] == 30.0
