"""
Tests for the definition of forecast-log version v1.11 (src/forecast_log/v1_11.py): v1 with the triangle payday intensity in the two Saturday pieces only, on the same
training data, nothing registered. (The fit itself is a dry run into a folder and is not exercised here: it needs the frozen features file.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("catboost")

from src.forecast_log import v1, v1_11  # noqa: E402


def test_the_version_and_its_parent():
    assert v1_11.VERSION == "v1.11" and v1_11.PARENT == v1.VERSION == "v1"


def test_only_the_two_saturday_pieces_differ_from_v1_and_the_new_column_is_last():
    assert set(v1_11.PIECES) == {"log_hydro_thu", "fri_A_all", "sat_A_lags", "raw", "sun_B"}  # the hydro arm's pieces; the post arm's `log` is not part of it
    for name, (extra, target) in v1_11.PIECES.items():
        assert target == v1.PIECES[name][1]
        if name in v1_11.SATURDAY_PIECES:
            assert extra == v1.PIECES[name][0] + ["intensity_sym"]
        else:
            assert extra == v1.PIECES[name][0]
    assert set(v1_11.SATURDAY_PIECES) == {"sat_A_lags", "raw"}


def test_the_hydro_flag_stays_in_the_monday_to_thursday_piece():
    assert v1_11.PIECES["log_hydro_thu"][0] == ["hydro_thu"]


def test_it_uses_the_same_training_cut_off_and_frozen_file_as_v1():
    assert v1_11.v1.FIRST_EXCLUDED_DAY == v1.FIRST_EXCLUDED_DAY
    assert v1_11.v1.TRAINING_FEATURES_SHA == v1.TRAINING_FEATURES_SHA


def test_a_features_file_that_is_not_the_frozen_one_is_refused(tmp_path):
    other = tmp_path / "features.csv"
    other.write_text("date,total_sales\n2026-01-01,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not the frozen features file"):
        v1_11.training_frames(other)


def test_a_dry_run_never_writes_into_a_folder_that_holds_models(tmp_path):
    (tmp_path / "already_there").mkdir()
    (tmp_path / "already_there" / "seed_00.cbm").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="never overwritten"):
        v1_11.fit_pieces(tmp_path / "features.csv", tmp_path / "already_there")
