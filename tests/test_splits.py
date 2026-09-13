"""
Tests for src/experiments/splits.py.

The single most important property of the whole sweep is that no fold
ever leaks test-period data into training - these tests exist to pin
that down directly, not just check fold counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.splits import (
    generate_rolling_origin_folds,
    slice_test_window,
    slice_train_window,
)


@pytest.fixture
def two_year_dates():
    return pd.Series(pd.date_range("2024-01-01", "2025-12-31", freq="D"))


def test_fold_test_windows_are_contiguous_and_non_overlapping(two_year_dates):
    folds = generate_rolling_origin_folds(
        two_year_dates, initial_train_days=365, test_days=60
    )
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start


def test_first_fold_train_window_matches_initial_train_days(two_year_dates):
    folds = generate_rolling_origin_folds(
        two_year_dates, initial_train_days=365, test_days=60
    )
    first = folds[0]
    assert (first.train_end - first.train_start).days == 365


def test_no_fold_test_window_exceeds_available_data(two_year_dates):
    max_date = pd.to_datetime(two_year_dates).max()
    folds = generate_rolling_origin_folds(
        two_year_dates, initial_train_days=365, test_days=60
    )
    for fold in folds:
        assert fold.test_end <= max_date + pd.Timedelta(days=1)


def test_max_folds_caps_the_count(two_year_dates):
    folds = generate_rolling_origin_folds(
        two_year_dates, initial_train_days=365, test_days=60, max_folds=3
    )
    assert len(folds) == 3


def test_step_days_controls_overlap(two_year_dates):
    folds = generate_rolling_origin_folds(
        two_year_dates, initial_train_days=365, test_days=60, step_days=30
    )
    # With step < test_days, consecutive test windows overlap by design.
    assert folds[1].test_start == folds[0].test_start + pd.Timedelta(days=30)
    assert folds[1].test_start < folds[0].test_end


def test_expanding_window_uses_all_prior_history(two_year_dates):
    df = pd.DataFrame({"date": two_year_dates})
    folds = generate_rolling_origin_folds(two_year_dates, initial_train_days=365, test_days=60)
    fold = folds[2]  # some later fold with plenty of accumulated history

    train = slice_train_window(df, "date", fold, train_window_days=None)
    assert train["date"].min() == fold.train_start
    assert train["date"].max() < fold.test_start


def test_sliding_window_restricts_to_recent_days_only(two_year_dates):
    df = pd.DataFrame({"date": two_year_dates})
    folds = generate_rolling_origin_folds(two_year_dates, initial_train_days=365, test_days=60)
    fold = folds[2]

    train = slice_train_window(df, "date", fold, train_window_days=90)
    assert train["date"].min() == fold.train_end - pd.Timedelta(days=90)
    assert len(train) == 90


def test_sliding_window_never_extends_before_available_history(two_year_dates):
    """The very first fold has exactly initial_train_days of history - a
    sliding window request longer than that must not reach before the
    start of the dataset (there's nothing there to leak, but it should
    clamp rather than silently return a shorter-than-requested window
    without explanation)."""
    df = pd.DataFrame({"date": two_year_dates})
    folds = generate_rolling_origin_folds(two_year_dates, initial_train_days=365, test_days=60)
    fold = folds[0]

    train = slice_train_window(df, "date", fold, train_window_days=9999)
    assert train["date"].min() == fold.train_start


def test_train_and_test_slices_never_overlap(two_year_dates):
    df = pd.DataFrame({"date": two_year_dates})
    folds = generate_rolling_origin_folds(two_year_dates, initial_train_days=365, test_days=60)

    for fold in folds:
        for train_window_days in (None, 90, 365):
            train = slice_train_window(df, "date", fold, train_window_days)
            test = slice_test_window(df, "date", fold)
            assert train["date"].max() < test["date"].min()
            assert set(train["date"]).isdisjoint(set(test["date"]))


def test_test_slice_matches_fold_boundaries_exactly(two_year_dates):
    df = pd.DataFrame({"date": two_year_dates})
    folds = generate_rolling_origin_folds(two_year_dates, initial_train_days=365, test_days=60)
    fold = folds[0]

    test = slice_test_window(df, "date", fold)
    assert len(test) == 60
    assert test["date"].min() == fold.test_start
    assert test["date"].max() == fold.test_end - pd.Timedelta(days=1)
