"""
Tests for src/config.py and the feature sets it selects.

The real config must load and be consistent with the code that reads it;
broken configs must be refused loudly, not half-applied.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from src.config import CONFIG, DEFAULT_CONFIG_PATH, FEATURE_USERS, PROJECT_ROOT, REQUIRED_PATHS, load_config
from src.features.build_features import (
    CANDIDATE_FEATURES,
    FEATURE_SETS,
    V1_FEATURES,
    V2_FEATURES,
    features_for,
    load_model_frame,
    model_frame,
)


@pytest.fixture
def raw() -> dict:
    """The real config as a plain dict, to break one thing at a time."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def write(tmp_path, data: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- the real config ------------------------------------------------------------


def test_the_real_config_loads_and_resolves_every_path_under_the_project():
    for name in REQUIRED_PATHS:
        path = CONFIG.path(name)
        assert PROJECT_ROOT in path.parents or path == PROJECT_ROOT


def test_every_model_uses_a_feature_set_that_exists():
    assert set(CONFIG.features.use) == set(FEATURE_USERS)
    for model in FEATURE_USERS:
        assert features_for(model) == FEATURE_SETS[CONFIG.features.use[model]]


def test_the_served_model_uses_all_38_and_the_baselines_the_original_15():
    assert len(V1_FEATURES) == 15 and len(V2_FEATURES) == 38
    assert features_for("catboost") == V2_FEATURES
    for model in ("xgboost", "neural", "experiments_baseline"):
        assert features_for(model) == V1_FEATURES


def test_v2_is_v1_followed_by_the_candidates_with_no_duplicates():
    assert V2_FEATURES[:15] == V1_FEATURES  # order matters: it changes CatBoost's seeded fits
    assert V2_FEATURES[15:] == CANDIDATE_FEATURES
    assert len(set(V2_FEATURES)) == len(V2_FEATURES)


def test_the_feature_sets_match_the_experiment_framework():
    from src.experiments.feature_sets import ALL_CANDIDATES
    from src.experiments.series_common import ADD_ALL_CANDIDATES

    assert CANDIDATE_FEATURES == list(ALL_CANDIDATES)
    assert V2_FEATURES == ADD_ALL_CANDIDATES


def test_features_for_returns_a_copy_so_callers_cannot_change_the_set():
    features = features_for("catboost")
    features.append("junk")
    assert "junk" not in features_for("catboost")


def test_the_model_settings_build_the_models_they_describe():
    from src.models.final_model import FINAL_PARAMS, FINAL_SEEDS
    from src.models.train_xgboost import build_model

    assert build_model().get_params()["n_estimators"] == CONFIG.xgboost.n_estimators
    assert FINAL_PARAMS == {"iterations": 300, "learning_rate": 0.05, "depth": 4}
    assert FINAL_SEEDS == list(range(CONFIG.catboost.n_seeds))


# --- model_frame / load_model_frame -----------------------------------------------


def frame(rows: int = 6) -> pd.DataFrame:
    data = {"date": pd.date_range("2025-01-01", periods=rows), "total_sales": range(rows), "junk": range(rows)}
    for i, name in enumerate(V1_FEATURES):
        data[name] = [float(i + r) for r in range(rows)]
    return pd.DataFrame(data)


def test_model_frame_keeps_the_date_target_and_features_in_order_and_drops_incomplete_rows():
    df = frame()
    df.loc[0, "lag_x"] = 1.0  # an unrelated column with gaps must not cost any rows
    df.loc[2, V1_FEATURES[3]] = float("nan")
    out = model_frame(df)
    assert list(out.columns) == ["date", "total_sales"] + V1_FEATURES
    assert len(out) == 5 and list(out.index) == list(range(5))
    assert "junk" not in out.columns


def test_model_frame_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required model columns"):
        model_frame(frame().drop(columns=[V1_FEATURES[0]]))


def test_load_model_frame_reads_the_engineered_csv_and_parses_dates(tmp_path):
    path = tmp_path / "engineered.csv"
    frame().to_csv(path, index=False)
    out = load_model_frame(path)
    assert pd.api.types.is_datetime64_any_dtype(out["date"]) and len(out) == 6
    with pytest.raises(FileNotFoundError):
        load_model_frame(tmp_path / "missing.csv")


# --- broken configs are refused ----------------------------------------------------


def test_a_copy_of_the_real_config_loads(raw, tmp_path):
    assert load_config(write(tmp_path, raw)).project.council == CONFIG.project.council


def test_an_unknown_key_is_an_error_not_silently_ignored(raw, tmp_path):
    raw["weather"]["heavy_rain_mmm"] = 5.0  # a typo
    with pytest.raises(ValidationError, match="heavy_rain_mmm"):
        load_config(write(tmp_path, raw))


def test_an_unknown_feature_set_name_is_refused(raw, tmp_path):
    raw["features"]["use"]["catboost"] = "v3"
    with pytest.raises(ValidationError, match="v3"):
        load_config(write(tmp_path, raw))


def test_every_model_must_pick_a_feature_set(raw, tmp_path):
    del raw["features"]["use"]["neural"]
    with pytest.raises(ValidationError, match="neural"):
        load_config(write(tmp_path, raw))


def test_a_bad_date_is_refused(raw, tmp_path):
    raw["splits"]["xgboost_split_date"] = "31/12/2024"
    with pytest.raises(ValidationError, match="not a YYYY-MM-DD"):
        load_config(write(tmp_path, raw))


def test_a_wrong_type_is_refused(raw, tmp_path):
    raw["xgboost"]["n_estimators"] = "lots"
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, raw))


def test_a_missing_or_absolute_path_is_refused(raw, tmp_path):
    missing = copy.deepcopy(raw)
    del missing["paths"]["figures_dir"]
    with pytest.raises(ValidationError, match="figures_dir"):
        load_config(write(tmp_path, missing))

    absolute = copy.deepcopy(raw)
    absolute["paths"]["models_dir"] = "C:/somewhere/models"
    with pytest.raises(ValidationError, match="relative"):
        load_config(write(tmp_path, absolute))

    escaping = copy.deepcopy(raw)
    escaping["paths"]["models_dir"] = "../models"
    with pytest.raises(ValidationError, match="relative"):
        load_config(write(tmp_path, escaping))


def test_the_config_is_read_only(raw):
    with pytest.raises(ValidationError):
        CONFIG.weather.heavy_rain_mm = 99.0
