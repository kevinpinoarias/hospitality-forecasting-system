"""
Tests for the package validation: bags are the mean over seeds, the weekday choice uses only the first test blocks, the package
steps compose as documented, and a results-folder suffix keeps a re-run apart from the original.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments import run_package_validation as pv
from src.experiments import run_series26_weekend_features as s26


def _days(n: int = 140) -> pd.DataFrame:
    dates = pd.date_range("2025-03-03", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "fold_id": np.arange(n) // 14, "actual": rng.uniform(5000, 15000, n),
                         "weekday": dates.day_name()})


def test_bag_dict_averages_over_seeds_and_keys_by_the_given_fields(tmp_path):
    path = tmp_path / "r.jsonl"
    rows = [{"id": "a", "variant": "A_lags", "target": "raw", "per_seed": [[1.0, 2.0], [3.0, 4.0]]},
            {"id": "b", "variant": "A_lags", "target": "raw", "per_seed": [[5.0, 6.0]]},
            {"id": "c", "variant": "B", "target": "log1p", "per_seed": [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]]}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    bags = pv.bag_dict(path, ("variant", "target"), 3)
    assert set(bags) == {("A_lags", "raw"), ("B", "log1p")}
    assert np.allclose(bags[("A_lags", "raw")], [3.0, 4.0])
    with pytest.raises(AssertionError):
        pv.bag_dict(path, ("variant", "target"), 5)


def test_the_weekday_choice_uses_only_the_first_blocks():
    days = _days()
    a = days["actual"].to_numpy()
    log_raw = {"log1p_final": a + 300.0, "raw_same_params": a + 300.0}
    features = {(v, t): a + 300.0 for v in s26.VARIANTS for t in s26.TARGETS}
    late = days["fold_id"].to_numpy() >= pv.s27.SELECTION_BLOCKS
    features[("A_lags", "raw")] = np.where(late, a, a + 5000.0)  # looks perfect only on the judging blocks
    assert pv.choose_on_first_blocks(days, log_raw, features)["Saturday"] == "none"
    features[("A_lags", "raw")] = np.where(late, a + 5000.0, a)  # looks perfect on the selection blocks
    assert pv.choose_on_first_blocks(days, log_raw, features)["Saturday"] == "A_lags"


def test_package_steps_compose_in_the_documented_order():
    days = _days()
    a = days["actual"].to_numpy()
    log_raw = {"log1p_final": a - 400.0, "raw_same_params": a - 100.0}
    features = {("A_lags", "raw"): a - 50.0}
    steps = pv.package_steps(days, log_raw, features, {"Friday": "none", "Saturday": "A_lags", "Sunday": "none"})
    assert list(steps) == pv.STEPS
    wd = days["weekday"].to_numpy()
    assert np.array_equal(steps[pv.STEPS[0]], log_raw["log1p_final"])
    assert np.allclose(steps[pv.STEPS[1]][wd == "Saturday"], a[wd == "Saturday"] - 100.0)
    assert np.allclose(steps[pv.STEPS[2]][wd == "Saturday"], a[wd == "Saturday"] - 50.0)
    mon = wd == "Monday"
    assert np.array_equal(steps[pv.STEPS[3]][mon], steps[pv.STEPS[2]][mon])  # Monday to Thursday are never corrected


def test_scores_report_the_three_windows():
    days = _days()
    t = pv.scores(days, {"x": days["actual"].to_numpy() + 100.0})
    assert t.loc[0, "all 585 days"] == 100.0 and t.loc[0, "blocks 1-9 (540)"] == 100.0 and t.loc[0, "blocks 5-9 (300)"] == 100.0


def test_a_results_suffix_names_its_own_folder(monkeypatch):
    assert pv.results_dir("_cal2").name == "series19_21_cal2" and pv.results_dir("").name == "series19_21"
