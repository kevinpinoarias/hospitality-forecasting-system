"""
Tests for Series 29: the calendar swap replaces only the four school columns, flags the verified dates, and the task
list covers two seed sets for the two targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series29_school_calendar as s29
from src.features.build_features import SCHOOL_BREAKS_V1


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-03-25", "2024-04-20")
    return pd.DataFrame({"date": dates, "is_school_holiday": 0, "is_christmas_break": 0, "is_summer_break": 0, "is_easter_break": 0,
                         "max_temp": np.arange(len(dates), dtype=float), "month": dates.month})


def test_with_calendar_replaces_only_the_four_school_columns():
    frame = _frame()
    out = s29.with_calendar(frame)
    other = [c for c in frame.columns if c not in s29.SCHOOL_COLUMNS]
    assert out[other].equals(frame[other])
    easter = out.set_index("date")
    assert easter.loc["2024-03-28", "is_school_holiday"] == 0 and easter.loc["2024-03-29", "is_easter_break"] == 1
    assert easter.loc["2024-04-14", "is_school_holiday"] == 1 and easter.loc["2024-04-15", "is_school_holiday"] == 0


def test_the_original_calendar_misses_easter_2024_and_the_verified_one_does_not():
    frame = _frame()
    assert s29.with_calendar(frame, SCHOOL_BREAKS_V1)["is_school_holiday"].sum() == 0
    assert s29.with_calendar(frame)["is_school_holiday"].sum() == 17  # 29 Mar to 14 Apr 2024


def test_tasks_cover_both_seed_sets_for_both_targets():
    tasks = s29.make_tasks()
    assert len(tasks) == 20 and len({t["id"] for t in tasks}) == 20
    for set_name, seeds in s29.SEEDS.items():
        for variant in s29.VARIANT_TARGETS:
            got = sorted(s for t in tasks if t["set"] == set_name and t["variant"] == variant for s in t["seeds"])
            assert got == seeds
    assert s29.results_path("stage1").name == "series29_school_calendar_stage1.jsonl"
