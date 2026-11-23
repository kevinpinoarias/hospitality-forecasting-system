"""Tests for the Series 42 machinery (src/experiments/run_series42_no_manager.py): the arms, the weeks, and the four rules on made-up numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments import run_series42_no_manager as s42
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL


def toy_training(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2025-01-06", periods=n, freq="D")
    frame = pd.DataFrame({DATE_COL: dates, TARGET_COL: np.linspace(5000, 7000, n), s42.MANAGER_COL: np.linspace(4800, 6800, n)})
    for column in FINAL_FEATURES:
        if column not in frame:
            frame[column] = 1.0
    return frame


def test_whole_weeks_only_counts_weeks_wholly_inside_the_block():
    # a block from Wednesday 15 Jan 2025 for 60 days: the first Monday is 20 Jan and a week must end before the block does
    mondays = s42.whole_weeks(pd.Timestamp("2025-01-15"), pd.Timestamp("2025-03-16"))
    assert mondays[0] == pd.Timestamp("2025-01-20")
    assert all(m.weekday() == 0 for m in mondays)
    assert mondays[-1] + pd.Timedelta(days=7) <= pd.Timestamp("2025-03-16")
    assert mondays[-1] + pd.Timedelta(days=14) > pd.Timestamp("2025-03-16")


def test_n0_trains_on_the_real_number():
    train = toy_training()
    frame, features = s42.arm_frame("N0", train)
    assert frame[s42.MANAGER_COL].equals(train[s42.MANAGER_COL]) and features == list(FINAL_FEATURES)


def test_n1_replaces_only_the_manager_column():
    train = toy_training()
    frame, features = s42.arm_frame("N1", train)
    assert not np.allclose(frame[s42.MANAGER_COL], train[s42.MANAGER_COL])
    assert features == list(FINAL_FEATURES)
    others = [c for c in train.columns if c != s42.MANAGER_COL]
    assert frame[others].equals(train[others])


def test_n2_drops_the_manager_input_and_keeps_the_order():
    _, features = s42.arm_frame("N2", toy_training())
    assert s42.MANAGER_COL not in features
    assert features == [f for f in FINAL_FEATURES if f != s42.MANAGER_COL]
    assert len(features) == len(FINAL_FEATURES) - 1


def test_the_stand_in_leaves_the_row_itself_out():
    train = toy_training()
    base = s42.stand_in_leave_one_out(train)
    changed = train.copy()
    changed.loc[10, s42.MANAGER_COL] = 1e9  # an extreme value on row 10 must not move row 10's own stand-in
    assert s42.stand_in_leave_one_out(changed)[10] == pytest.approx(base[10])


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        s42.arm_frame("N9", toy_training())


def fake_frame(gain: float, n_days: int = 140, seed: int = 0) -> pd.DataFrame:
    """Absolute errors for N0, N1 and N2: N1 is `gain` pounds better than N0 on every day (negative gain = worse), N2 equals N0."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-02-03", periods=n_days, freq="D")
    actual = rng.uniform(4000, 9000, n_days)
    rows = []
    for kind in s42.KINDS:
        for s in s42.SEED_SETS:
            base = rng.uniform(500, 1500, n_days)
            for arm, shift in (("N0", 0.0), ("N1", -gain), ("N2", 0.0)):
                for i, d in enumerate(dates):
                    rows.append({"arm": arm, "seedset": s, "fold": i * 10 // n_days, "date": d.strftime("%Y-%m-%d"), "kind": kind,
                                 "actual": actual[i], "abs_err": base[i] + shift})
    return pd.DataFrame(rows)


def test_a_clear_gain_in_every_fold_is_a_candidate():
    verdict = s42.judge(fake_frame(gain=40.0), "N1")
    assert verdict["candidate"] and all(verdict["rules"].values())
    assert verdict["nw_pooled"][0] == pytest.approx(-40.0)


def test_no_change_is_not_a_candidate():
    verdict = s42.judge(fake_frame(gain=0.0), "N2")
    assert not verdict["candidate"]
    assert not verdict["rules"]["1 gain, upper end below zero on both seed sets"]


def test_a_gain_below_the_size_threshold_fails_rule_2_only():
    verdict = s42.judge(fake_frame(gain=8.0), "N1")
    rules = verdict["rules"]
    assert rules["1 gain, upper end below zero on both seed sets"] and not rules[f"2 pooled gain of at least {s42.MIN_GAIN:.0f}"]
    assert not verdict["candidate"]


def test_tasks_cover_every_fold_arm_and_seed_set():
    tasks = s42.build_tasks([0, 1], seeds_per_set=3)
    assert len(tasks) == 2 * len(s42.ARMS) * len(s42.SEED_SETS)
    assert len({t["id"] for t in tasks}) == len(tasks) and all(len(t["seeds"]) == 3 for t in tasks)
