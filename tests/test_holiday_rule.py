"""
Tests for the holiday rule: 0 on the closed days, the earlier Boxing Days' mean on Boxing Day, nothing on any other day; the level updates from the sales history; the run adds the rule's rows
only when the setting is on and only on those days; the scores add Q5 only once rule rows exist; and the tracked configuration leaves the rule off.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

pytest.importorskip("catboost")

from src.forecast_log import holiday_rule as hr  # noqa: E402
from src.forecast_log import run_forecasts, score_forecasts  # noqa: E402


def _history() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2024-12-25", "2024-12-26", "2025-12-25", "2025-12-26", "2025-12-27", "2026-12-26"]), "total_sales": [0.0, 4321.0, 0.0, 2919.0, 14358.0, None]})


def test_the_level_is_the_mean_of_the_earlier_boxing_days_with_known_sales():
    assert hr.boxing_day_level(_history(), pd.Timestamp("2026-12-26")) == pytest.approx((4321.0 + 2919.0) / 2)
    assert hr.boxing_day_level(_history(), pd.Timestamp("2025-12-26")) == pytest.approx(4321.0)  # only the earlier ones: a day never averages itself in
    assert hr.boxing_day_level(_history(), pd.Timestamp("2024-12-26")) is None  # no earlier Boxing Day: no rule row rather than a guess


def test_the_rule_forecasts_zero_on_closed_days_the_level_on_boxing_day_and_nothing_else():
    level = 3620.0
    assert hr.rule_forecast(pd.Timestamp("2026-12-25"), level) == 0.0 and hr.rule_forecast(pd.Timestamp("2027-01-01"), level) == 0.0
    assert hr.rule_forecast(pd.Timestamp("2026-12-26"), level) == level  # a Saturday Boxing Day: the level does not look at the weekday
    assert hr.rule_forecast(pd.Timestamp("2026-12-26"), None) is None
    for other in ("2026-12-24", "2026-12-27", "2026-12-31", "2027-01-02", "2026-11-02"):
        assert hr.rule_forecast(pd.Timestamp(other), level) is None


def test_the_fingerprint_is_stable_and_names_the_rule():
    assert hr.fingerprint() == hr.fingerprint() and len(hr.fingerprint()) == 32
    assert hr.RULE_MODEL == "holiday_rule" and hr.RULE_VERSION in hr.DESCRIPTION


def test_the_tracked_configuration_leaves_the_rule_off():
    settings = yaml.safe_load((run_forecasts.PROJECT_ROOT / "config" / "forecast_log.yaml").read_text(encoding="utf-8"))
    assert not settings.get("holiday_rule")


def _scored(with_rule: bool) -> pd.DataFrame:
    days = pd.to_datetime(["2026-12-24", "2026-12-25", "2026-12-26", "2026-12-27"])
    actual = {"2026-12-24": 7500.0, "2026-12-25": 0.0, "2026-12-26": 3000.0, "2026-12-27": 14000.0}
    rows = []
    for d in days:
        for model, variant, forecast in (("portfolio", "no_manager", 9000.0), ("post", "no_manager", 9000.0), ("hydro", "no_manager", 9000.0), ("manager", "", 5000.0)):
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": model, "variant": variant, "on_time": "on_time", "forecast": forecast, "actual_sales": actual[str(d.date())]})
        if with_rule and d.day in (25, 26):
            rows.append({"forecast_type": "TW", "target_date": str(d.date()), "model": hr.RULE_MODEL, "variant": "no_manager", "on_time": "on_time", "forecast": 0.0 if d.day == 25 else 3620.0,
                         "actual_sales": actual[str(d.date())]})
    frame = pd.DataFrame(rows)
    frame["miss"] = frame["forecast"] - frame["actual_sales"]
    return frame


def test_q5_appears_only_once_rule_rows_exist_and_scores_the_rule_on_its_days_only():
    without, _ = score_forecasts.build_report(_scored(False))
    assert "Q5 holiday rule" not in without
    report, _ = score_forecasts.build_report(_scored(True))
    assert "Q5 holiday rule vs portfolio, closed days and Boxing Day" in report and "Q5 holiday rule vs the manager, closed days and Boxing Day" in report
    frame = _scored(True)
    q5 = score_forecasts.paired_difference(frame, (hr.RULE_MODEL, "no_manager"), ("portfolio", "no_manager"), "TW", {"2026-12-25", "2026-12-26"})
    assert q5["days"] == 2 and q5["better_days"] == 2
    assert q5["change"] == pytest.approx((0.0 + 620.0) / 2 - (9000.0 + 6000.0) / 2)  # |rule miss| - |model miss| on the closed day and on Boxing Day
