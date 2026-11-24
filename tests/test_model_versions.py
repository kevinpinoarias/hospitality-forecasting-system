"""
Tests for the model picker's back end (src/api/model_versions.py) and the API's model_version option.

The most important one is parity: a version's forecast through the API must equal what the forecast log
computes for the same day with the same inputs. Tests that need the registered model files skip where they
are not installed (a fresh clone); no test reaches the network (the live weather is replaced).
"""

from __future__ import annotations

import datetime as dt
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from src.api import model_versions as mv  # noqa: E402
from src.api.history import ENGINEERED_FEATURES_PATH, HistoryCache  # noqa: E402
from src.models.final_model import MODEL_DIR  # noqa: E402

CHOICES = mv.load_choices()
VERSION_IDS = [c["id"] for c in CHOICES if c["id"] != mv.STANDARD]
TODAY = dt.date(2026, 11, 23)
DAYS = [dt.date(2026, 12, d) for d in (17, 18, 19, 20)] + [dt.date(2026, 12, 24), dt.date(2026, 12, 26), dt.date(2026, 12, 31), dt.date(2027, 1, 2), dt.date(2027, 1, 5)]


def installed(choice_id: str) -> bool:
    return mv.VersionService(history=None).is_available(choice_id)


needs_data = pytest.mark.skipif(not ENGINEERED_FEATURES_PATH.exists() or not (MODEL_DIR / "metadata.json").exists(), reason="needs the rebuilt data and model")
needs_versions = pytest.mark.skipif(not all(installed(v) for v in VERSION_IDS), reason="needs the registered version files (models/<version>, data/training)")


@pytest.fixture(scope="module")
def history():
    h = HistoryCache()
    h.refresh()
    return h


@pytest.fixture(scope="module")
def service(history):
    s = mv.VersionService(history)
    s._weather = lambda last_known, today: pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"])  # no network: historical averages
    return s


# --- the menu ---------------------------------------------------------------

def test_the_menu_is_the_standard_model_and_the_eight_versions():
    assert [c["id"] for c in CHOICES] == ["standard", "v1-post", "v1-hydro", "v1.11", "v1.12", "v1.13", "v1.14", "v1.131", "v1.141"]
    for c in CHOICES[1:]:
        assert c["status"].startswith("Under test in the weekly forecast log since") and c["legend"]
        assert len(c["pieces"]) in (5, 6, 15) and len(c["correction_history_sha256"]) == 64


def test_the_standard_model_is_always_available_and_an_unknown_id_never(tmp_path):
    s = mv.VersionService(history=None, models_root=tmp_path, correction_dir=tmp_path)
    assert s.is_available("standard") and not s.is_known("v9") and not s.is_available("v9")
    assert not s.is_available("v1.14")  # nothing installed in an empty folder
    assert {c["id"]: c["available"] for c in s.menu()}["v1.14"] is False


@needs_versions
def test_a_version_with_a_changed_file_is_refused(tmp_path):
    c = next(c for c in CHOICES if c["id"] == "v1.11")
    for name in c["pieces"]:
        shutil.copytree(mv.MODELS_ROOT / c["version"] / name, tmp_path / "models" / c["version"] / name)
    (tmp_path / "data").mkdir()
    shutil.copy(mv.CORRECTION_DIR / f"{c['version']}_correction_history.csv", tmp_path / "data")
    meta = tmp_path / "models" / c["version"] / "raw" / "meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + " ", encoding="utf-8")
    s = mv.VersionService(history=None, models_root=tmp_path / "models", correction_dir=tmp_path / "data")
    assert s.is_available("v1.11")  # the files are there...
    with pytest.raises(mv.VersionUnavailable, match="fingerprint"):
        s._load("v1.11")  # ...but one does not match what the log registered


# --- parity with the forecast log ----------------------------------------------

@needs_data
@needs_versions
@pytest.mark.parametrize("choice_id, column", [("v1-post", "post"), ("v1-hydro", "hydro"), ("v1.12", "hydro_v1.12"), ("v1.13", "hydro_v1.13"),
                                               ("v1.14", "hydro_v1.14"), ("v1.141", "hydro_v1.141")])
def test_a_version_answers_exactly_as_the_forecast_log_does(service, history, choice_id, column):
    from src.forecast_core.bags import Bag
    from src.forecast_log import predict
    from src.models.final_model import BaggedCatBoost

    weather = pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"])
    v1_models, v1_history = service._load("v1-post")
    final, _ = BaggedCatBoost.load(MODEL_DIR)
    v1_models = predict.Models("v1", Bag(final.models, final.features, "log1p"), v1_models.pieces, None, v1_models.piece_fingerprints)
    extras = [service._load(choice_id)] if column.startswith("hydro_") else []
    last_known = pd.Timestamp(history.max_date)
    logged = predict.forecast_days(v1_models, history.df, last_known, [pd.Timestamp(d) for d in DAYS], v1_history, weather, extras)

    ours = service.forecast(choice_id, DAYS, today=TODAY)
    expected = dict(zip(logged["target_date"], logged[column]))
    for d in DAYS:
        assert ours[d].best == pytest.approx(expected[d], abs=1e-6), d


# --- what the picker adds ------------------------------------------------------

@needs_data
@needs_versions
def test_rain_scenarios_and_a_given_manager_estimate(service):
    plain = service.forecast("v1.14", DAYS, today=TODAY)
    assert all(x.dry is not None and x.heavy is not None for x in plain.values())
    assert any(abs(x.dry - x.heavy) > 1 for x in plain.values())  # rain moves the forecast
    given = service.forecast("v1.14", [DAYS[0]], today=TODAY, manager_estimates={DAYS[0]: 20000.0})
    assert given[DAYS[0]].best != pytest.approx(plain[DAYS[0]].best)


@needs_data
@needs_versions
def test_past_dates_and_uncovered_hydro_thursdays_fall_back_with_a_reason(service, history, monkeypatch):
    past = history.max_date - dt.timedelta(days=10)
    out = service.forecast("v1.14", [past], today=TODAY)
    assert out[past].best is None and "no forecast made before this day" in out[past].note  # after the backtest's last day

    monkeypatch.setattr(mv, "hydro_list_through", lambda: dt.date(2026, 12, 20))
    thursday = dt.date(2026, 12, 24)
    assert service.forecast("v1.14", [thursday], today=TODAY)[thursday].best is None  # a Hydro model, unknown Thursday
    assert "Hydro show list" in service.forecast("v1.11", [thursday], today=TODAY)[thursday].note
    assert service.forecast("v1-post", [thursday], today=TODAY)[thursday].best is not None  # no Hydro flag: unaffected


@needs_data
@needs_versions
def test_shared_files_are_loaded_once(service):
    a, _ = service._load("v1.13")
    b, _ = service._load("v1.131")
    assert all(a.pieces[name] is b.pieces[name] for name in a.pieces)
    assert b.standin == "S6" and a.standin == "S0"


# --- through the API -------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api import main

    with TestClient(main.app) as c:
        main.version_service._weather = lambda last_known, today: pd.DataFrame(columns=["date", "max_temp", "rain_mm", "sun_hours"])
        yield c


@needs_data
def test_versions_page_lists_the_menu(client):
    choices = client.get("/versions").json()["choices"]
    assert [c["id"] for c in choices][:2] == ["standard", "v1-post"] and choices[0]["available"] is True


@needs_data
def test_an_unknown_version_is_refused(client):
    r = client.post("/predict", json={"requests": [{"date": "2026-12-24", "model_version": "v9"}]})
    assert r.status_code == 422 and "/versions" in r.json()["detail"]


@needs_data
@needs_versions
def test_predict_with_a_version_says_which_model_answered(client):
    results = client.post("/predict", json={"requests": [{"date": "2026-12-24", "model_version": "v1.14"}, {"date": "2026-12-24"}]}).json()["results"]
    picked, standard = results
    assert picked["model_version"] == "v1.14" and standard["model_version"] == "standard"
    assert picked["model_version_note"] is None and picked["predictions"]["best_estimate"] != standard["predictions"]["best_estimate"]
    assert picked["day_context"] == standard["day_context"]  # the same day, only the model differs


needs_backtest = pytest.mark.skipif(not (mv.MODELS_ROOT / mv.BACKTEST_FILE).exists(), reason="needs the versions' backtest file")


@needs_data
@needs_versions
@needs_backtest
def test_a_past_date_comes_from_the_versions_backtest(client):
    backtest = pd.read_csv(mv.MODELS_ROOT / mv.BACKTEST_FILE, index_col="date")
    result = client.post("/predict", json={"requests": [{"date": "2025-12-20", "model_version": "v1.13"}]}).json()["results"][0]
    assert result["model_version"] == "v1.13" and result["prediction_source"] == "made_before_the_day"
    assert result["predictions"]["best_estimate"] == pytest.approx(backtest.loc["2025-12-20", "v1.13"])
    assert result["predictions"]["dry_scenario"] == result["predictions"]["heavy_rain_scenario"] == result["predictions"]["best_estimate"]


@needs_data
@needs_versions
@needs_backtest
def test_an_estimate_in_one_request_never_reaches_another(client):
    plain, own = client.post("/predict", json={"requests": [
        {"date": "2025-12-20", "model_version": "v1.13"},
        {"date": "2025-12-20", "model_version": "v1.13", "forecast_sales": 9000},
    ]}).json()["results"]
    assert plain["model_version"] == "v1.13" and plain["model_version_note"] is None
    assert own["model_version"] == "standard" and "manager's number on record" in own["model_version_note"]


@needs_data
@needs_versions
def test_a_past_date_outside_the_backtest_falls_back(client):
    result = client.post("/predict", json={"requests": [{"date": "2026-10-10", "model_version": "v1.13"}]}).json()["results"][0]
    assert result["model_version"] == "standard" and "backtest covers" in result["model_version_note"]


@needs_backtest
def test_a_changed_backtest_file_is_refused(tmp_path):
    original = (mv.MODELS_ROOT / mv.BACKTEST_FILE).read_bytes()
    (tmp_path / mv.BACKTEST_FILE).write_bytes(original + b" ")  # one extra byte
    service = mv.VersionService(history=None, models_root=tmp_path)
    with pytest.raises(mv.VersionUnavailable, match="backtest"):
        service.backtest()


def test_the_picker_uses_no_training_or_plotting_stack():
    import subprocess
    import sys

    code = "import sys, src.api.model_versions; bad = [m for m in ('xgboost', 'sklearn', 'wandb', 'matplotlib') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=mv.PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_rain_switch_changes_only_the_rain_inputs():
    rows = pd.DataFrame({"rain_mm": [0.0, 12.0], "is_heavy_rain": [0, 1], "is_dry_day": [1, 0], "max_temp": [10.0, 11.0]})
    heavy = mv._with_rain(rows, np.array([12.0, 12.0]))
    assert heavy["is_heavy_rain"].tolist() == [1, 1] and heavy["is_dry_day"].tolist() == [0, 0]
    assert heavy["max_temp"].tolist() == [10.0, 11.0]
