"""
Tests for the Series 26 weekend features: lags are matched by calendar date and use only earlier days, the
same-weekday summaries ignore missing weeks, the interactions are Friday-to-Sunday products, and each variant
appends its columns without touching the final model's 38.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series26_weekend_features as s26
from src.models.final_model import DATE_COL, FINAL_FEATURES


def _frame(start: str = "2025-03-03", n: int = 14) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n)
    return pd.DataFrame({
        DATE_COL: dates, "is_payday_window_pm3": np.resize([1, 0], n), "is_dry_day": np.resize([1, 1, 0, 0, 1, 0, 1], n),
        "max_temp": np.arange(n, dtype=float), "days_to_bank_holiday": [2] * n, "days_since_bank_holiday": [10] * n,
    })


def _sales(start: str = "2025-01-01", n: int = 120) -> pd.Series:
    idx = pd.date_range(start, periods=n)
    return pd.Series(np.arange(n, dtype=float) * 10 + 1000, index=idx, name="total_sales")


def test_lags_are_the_sales_on_that_calendar_date_and_use_only_earlier_days():
    sales = _sales()
    out = s26.add_weekend_features(_frame(), sales)
    for _, r in out.iterrows():
        d = r[DATE_COL]
        assert r["lag_14_sales"] == sales[d - pd.Timedelta(days=14)]
        assert r["lag_21_sales"] == sales[d - pd.Timedelta(days=21)]
        assert r["same_wd_mean4"] == np.mean([sales[d - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)])
        assert r["same_wd_max4"] == max(sales[d - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4))
    changed = sales.copy()
    changed[changed.index >= pd.Timestamp("2025-03-03")] += 99999.0  # the row's own day and the future
    after = s26.add_weekend_features(_frame(n=7), changed)
    before = s26.add_weekend_features(_frame(n=7), sales)
    assert before[s26.FAMILY_A].equals(after[s26.FAMILY_A])


def test_a_missing_week_is_ignored_by_the_summaries_and_gives_nan_for_that_lag():
    sales = _sales()
    sales = sales.drop(pd.Timestamp("2025-02-24"))  # one missing day
    out = s26.add_weekend_features(_frame(start="2025-03-03", n=1), sales)  # a Monday: lag_7 = 24 Feb is missing
    r = out.iloc[0]
    expected = np.mean([sales[pd.Timestamp("2025-03-03") - pd.Timedelta(days=7 * k)] for k in (2, 3, 4)])
    assert r["same_wd_mean4"] == expected
    assert r["lag_14_sales"] == sales[pd.Timestamp("2025-02-17")]


def test_interactions_are_friday_to_sunday_products_and_zero_on_weekdays():
    out = s26.add_weekend_features(_frame(start="2025-03-03", n=14), _sales())
    fri_sun = out[DATE_COL].dt.dayofweek >= 4
    assert (out.loc[~fri_sun, s26.FAMILY_B] == 0).all().all()
    assert (out.loc[fri_sun, "wk_temp"] == out.loc[fri_sun, "max_temp"]).all()
    assert (out.loc[fri_sun, "wk_payday"] == out.loc[fri_sun, "is_payday_window_pm3"]).all()
    assert (out.loc[fri_sun, "wk_dry"] == out.loc[fri_sun, "is_dry_day"]).all()
    assert (out.loc[fri_sun, "wk_bh_near"] == 1).all()  # a bank holiday is 2 days away in the test frame


def test_variants_append_their_columns_and_keep_the_final_models_38_first():
    assert len(FINAL_FEATURES) == 38
    assert {v: len(spec["extra"]) for v, spec in s26.VARIANTS.items()} == {"A_all": 4, "A_summary": 2, "A_lags": 2, "B": 4, "AB": 8}
    assert s26.VARIANTS["AB"]["extra"] == s26.FAMILY_A + s26.FAMILY_B
    assert not set(s26.FAMILY_A + s26.FAMILY_B) & set(FINAL_FEATURES)
    assert len(s26.all_variant_targets()) == 10


def test_variant_hybrids_take_the_raw_forecast_on_saturdays():
    days = pd.DataFrame({"weekday": ["Friday", "Saturday", "Sunday"]})
    bags = {(v, "log1p"): np.array([1.0, 2.0, 3.0]) for v in s26.VARIANTS}
    bags.update({(v, "raw"): np.array([10.0, 20.0, 30.0]) for v in s26.VARIANTS})
    out = s26.variant_hybrids(days, bags)
    assert all(h.tolist() == [1.0, 20.0, 3.0] for h in out.values()) and set(out) == set(s26.VARIANTS)
