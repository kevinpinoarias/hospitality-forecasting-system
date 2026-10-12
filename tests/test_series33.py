"""
Tests for Series 33: the median ignores a single record day while the mean does not, pooled history uses every earlier break and
per-type history only earlier breaks of the same kind, a correction never learns from the break it corrects or from later ones, and the
summary counts the breaks that improve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments import run_series33_first_week_by_type as s33


def _first(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """rows: (break name, start date, forecast, actual); seven days each."""
    out = []
    for name, start, pred, actual in rows:
        for k in range(7):
            d = pd.Timestamp(start) + pd.Timedelta(days=k)
            out.append({"break": name, "start": pd.Timestamp(start), "date": d, "pred": pred, "actual": actual})
    return pd.DataFrame(out)


def test_the_median_ignores_one_record_day_and_the_mean_does_not():
    pred = np.array([100.0] * 7)
    actual = np.array([110.0] * 6 + [1100.0])  # one record day
    assert s33.learned_shift(pred, actual, "mean") < -100.0
    assert s33.learned_shift(pred, actual, "median") == -10.0


def test_pooled_history_uses_every_earlier_break_and_per_type_only_the_same_kind():
    first = _first([("Easter 2024", "2024-03-29", 9000.0, 10000.0), ("Summer 2024", "2024-06-27", 9000.0, 12000.0),
                    ("Easter 2025", "2025-04-05", 9000.0, 10500.0)])
    pooled = s33.forward_chain(first, "pooled", "mean").set_index("break")
    by_type = s33.forward_chain(first, "type", "mean").set_index("break")
    assert list(pooled.index) == ["Summer 2024", "Easter 2025"] and list(by_type.index) == ["Easter 2025"]
    assert pooled.loc["Summer 2024", "earlier_breaks"] == 1 and pooled.loc["Easter 2025", "earlier_breaks"] == 2
    assert np.isclose(pooled.loc["Easter 2025", "learned_shift"], -2000.0)   # mean of -1000 and -3000
    assert np.isclose(by_type.loc["Easter 2025", "learned_shift"], -1000.0)  # Easter 2024 only
    assert np.isclose(by_type.loc["Easter 2025", "after_MAE"], 1000.0)  # 9000 + 0.5 x 1000 = 9500 against an actual of 10500


def test_a_correction_never_learns_from_the_break_it_corrects_or_from_later_ones():
    first = _first([("Easter 2024", "2024-03-29", 9000.0, 10000.0), ("Summer 2024", "2024-06-27", 9000.0, 12000.0),
                    ("Easter 2025", "2025-04-05", 9000.0, 10500.0)])
    before = s33.forward_chain(first, "pooled", "mean").set_index("break")
    changed = first.copy()
    changed.loc[changed["break"] == "Easter 2025", "actual"] = 99999.0  # the last break changes completely
    after = s33.forward_chain(changed, "pooled", "mean").set_index("break")
    assert np.isclose(before.loc["Summer 2024", "learned_shift"], after.loc["Summer 2024", "learned_shift"])
    assert np.isclose(before.loc["Easter 2025", "learned_shift"], after.loc["Easter 2025", "learned_shift"])  # its own actuals are not in its history


def test_summary_counts_the_improving_breaks_and_weights_by_days():
    table = pd.DataFrame({"break": ["a", "b", "c", "d"], "days": [7, 7, 7, 7], "before_MAE": [100.0] * 4, "after_MAE": [80.0, 90.0, 120.0, 100.0],
                          "change": [-20.0, -10.0, 20.0, 0.0]})
    s = s33.summary(table, ["a", "b", "c", "d"])
    assert s["breaks_improved"] == 2 and s["days"] == 28 and np.isclose(s["change"], -2.5) and s["lo"] <= s["change"] <= s["hi"] + 1e-9
    assert s33.summary(table, ["a", "b"])["breaks_scored"] == 2


def test_the_plan_matches_the_header():
    assert s33.STRENGTH == 0.5 and set(s33.VARIANTS) == {"pooled mean", "per type mean", "pooled median"}
    assert s33.SCORED == ["Easter 2025", "Summer 2025", "Easter 2026", "Summer 2026"]
    assert s33.kind("Summer 2026") == "summer"
