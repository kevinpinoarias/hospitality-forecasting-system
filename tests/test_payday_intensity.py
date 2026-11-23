"""
Tests for the triangle payday intensity (src/features/payday_intensity.py): hand-checked values, the bank-holiday rolling, agreement with the experiments' own copy
(Series 70-72) on every day, and that the column is appended last without touching the input.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.features import payday_intensity as pi


def test_hand_checked_values_from_the_experiment_log():
    dates = pd.to_datetime(["2025-09-19", "2025-09-24", "2025-09-26", "2025-09-27", "2025-09-30"])
    got = pi.payday_intensity(dates)
    # 24 Sep: 25th rule 1 day before its payday (0.30 x 0.75) + 28th rule's payday (Fri 26th) 2 days away (0.30 x 0.5); 26 Sep: the 28th rule pays (0.30) + the 25th rule 1 day after (0.225);
    # 27 Sep: 28th rule +1 (0.225), 25th rule +2 (0.15), last working day (Tue 30th) 3 days away (0.34 x 0.25); 30 Sep: the last working day pays (0.34); 19 Sep: no payday within 3 days
    assert np.allclose(got, [0.0, 0.375, 0.525, 0.46, 0.34])


def test_a_payday_on_a_weekend_or_bank_holiday_is_paid_the_working_day_before():
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    assert pi.month_payday(pd.Period("2025-09", "M"), 28, calendar) == pd.Timestamp("2025-09-26")  # the 28th was a Sunday
    assert pi.month_payday(pd.Period("2025-12", "M"), 25, calendar) == pd.Timestamp("2025-12-24")  # Christmas Day, a Thursday and a bank holiday
    assert pi.month_payday(pd.Period("2025-12", "M"), "last_working_day", calendar) == pd.Timestamp("2025-12-31")
    assert pi.month_payday(pd.Period("2025-08", "M"), "last_working_day", calendar) == pd.Timestamp("2025-08-29")  # the 31st was a Sunday


def test_it_agrees_with_the_experiments_copy_on_every_day():
    from src.experiments import run_series71_amplify_intensity as s71

    days = pd.date_range("2024-01-01", "2026-12-31", freq="D")
    assert np.allclose(pi.payday_intensity(days), s71.triangle_intensity(pd.Series(days), pi.WIDTH))


def test_range_order_and_no_missing_values():
    days = pd.date_range("2024-01-01", "2026-12-31", freq="D")
    values = pi.payday_intensity(days)
    assert not np.isnan(values).any()
    assert values.min() >= 0.0 and values.max() <= sum(pi.SHARES.values()) + 1e-9
    shuffled = days[np.random.default_rng(0).permutation(len(days))]
    assert np.allclose(pi.payday_intensity(shuffled), pd.Series(values, index=days).reindex(shuffled).to_numpy())  # each date gets its own value whatever the order


def test_the_column_is_appended_last_and_the_input_is_untouched():
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-09-26", "2025-09-27"]), "x": [1, 2]})
    out = pi.add_payday_intensity(frame)
    assert list(out.columns) == ["date", "x", pi.COLUMN] and list(frame.columns) == ["date", "x"]
    assert np.allclose(out[pi.COLUMN], [0.525, 0.46])
