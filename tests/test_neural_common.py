"""
Tests for the sweep-oriented additions to src/models/neural_common.py's
prepare_datasets (df injection, sliding train window, capped validation
end). These exist to guard against exactly the kind of change that would
silently leak a later rolling-origin fold's data into an earlier one's
training or validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.neural_common import FEATURES, prepare_datasets

WINDOW = 5


def make_df(n_days: int, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    data = {"date": dates, "total_sales": rng.random(n_days) * 1000}
    for feat in FEATURES:
        data[feat] = rng.random(n_days)
    return pd.DataFrame(data)


def test_default_behaviour_unchanged_train_valid_partition_the_data():
    df = make_df(120)
    split_date = df["date"].iloc[80]
    result = prepare_datasets(split_date=str(split_date.date()), window=WINDOW, df=df)

    assert result["n_train"] + result["n_valid"] == len(df) - WINDOW


def test_train_window_days_restricts_training_only():
    df = make_df(120)
    split_date = df["date"].iloc[100]

    full = prepare_datasets(split_date=str(split_date.date()), window=WINDOW, df=df)
    restricted = prepare_datasets(
        split_date=str(split_date.date()), window=WINDOW, df=df, train_window_days=20
    )

    assert restricted["n_train"] < full["n_train"]
    assert restricted["n_valid"] == full["n_valid"]


def test_valid_end_date_caps_validation_window():
    df = make_df(120)
    split_date = df["date"].iloc[80]
    valid_end = df["date"].iloc[90]

    capped = prepare_datasets(
        split_date=str(split_date.date()),
        window=WINDOW,
        df=df,
        valid_end_date=str(valid_end.date()),
    )
    uncapped = prepare_datasets(split_date=str(split_date.date()), window=WINDOW, df=df)

    assert capped["n_valid"] < uncapped["n_valid"]
    assert capped["valid_dates"].max() < valid_end


def test_train_and_valid_dates_never_overlap_with_new_options():
    df = make_df(150)
    split_date = df["date"].iloc[100]
    valid_end = df["date"].iloc[120]

    result = prepare_datasets(
        split_date=str(split_date.date()),
        window=WINDOW,
        df=df,
        train_window_days=30,
        valid_end_date=str(valid_end.date()),
    )
    assert result["valid_dates"].min() >= split_date
    assert result["valid_dates"].max() < valid_end


def test_empty_valid_window_raises():
    df = make_df(120)
    split_date = df["date"].iloc[80]
    with pytest.raises(ValueError):
        # valid_end_date before split_date -> empty validation set
        prepare_datasets(
            split_date=str(split_date.date()),
            window=WINDOW,
            df=df,
            valid_end_date=str(df["date"].iloc[79].date()),
        )
