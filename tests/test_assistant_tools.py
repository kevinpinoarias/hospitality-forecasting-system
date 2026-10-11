"""
Tests for the chat assistant's three error-ranking tools in
src/assistant/query_forecast.py. No LLM and no network: the HTTP call is
replaced, and one test routes it to the real API in-process.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
pytest.importorskip("dotenv")

from src.api.history import ENGINEERED_FEATURES_PATH
from src.assistant import query_forecast as q
from src.models.final_model import MODEL_DIR

TOOLS_UNDER_TEST = {
    "get_top_worst_days": "worst",
    "get_top_overpredictions": "over",
    "get_top_underpredictions": "under",
    "get_top_best_days": "best",
}


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return dict(self.payload)


@pytest.fixture
def requests_made(monkeypatch):
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append({"url": url, "params": params})
        return FakeResponse({"days": [], "limit": params["limit"]})

    monkeypatch.setattr(q.requests, "get", fake_get)
    return seen


@pytest.mark.parametrize("tool_name, kind", TOOLS_UNDER_TEST.items())
def test_each_tool_asks_the_api_for_its_own_kind_of_ranking(requests_made, tool_name, kind):
    getattr(q, tool_name)()
    assert requests_made[0]["url"].endswith("/errors/ranking")
    assert requests_made[0]["params"] == {"kind": kind, "forecast": "model", "limit": 10}


def test_the_manager_forecast_can_be_judged_in_any_letter_case(requests_made):
    q.get_top_overpredictions(forecast=" Manager ", n=5)
    assert requests_made[0]["params"] == {"kind": "over", "forecast": "manager", "limit": 5}


def test_a_request_for_too_many_days_is_capped_and_says_so(requests_made):
    result = q.get_top_worst_days(n=50)
    assert requests_made[0]["params"]["limit"] == q.MAX_RANKING_DAYS == 20
    assert "50 days were asked for" in result["limit_note"]
    assert "20" in result["limit_note"]


def test_zero_or_negative_days_is_raised_to_one(requests_made):
    result = q.get_top_underpredictions(n=0)
    assert requests_made[0]["params"]["limit"] == 1
    assert "limit_note" in result


def test_an_ordinary_request_has_no_limit_note(requests_made):
    assert "limit_note" not in q.get_top_worst_days(n=7)


def test_an_unknown_forecast_is_refused_without_calling_the_api(requests_made):
    result = q.get_top_worst_days(forecast="robot")
    assert "error" in result
    assert requests_made == []


def test_the_cap_matches_the_apis():
    from src.api.error_ranking import MAX_RANKING_DAYS

    assert q.MAX_RANKING_DAYS == MAX_RANKING_DAYS


def test_the_tools_are_registered_and_described_to_the_model():
    names = [fn.__name__ for fn in q.TOOL_FUNCTIONS]
    for tool_name in TOOLS_UNDER_TEST:
        assert tool_name in names and tool_name in q.TOOLS
        assert tool_name in q.SYSTEM_INSTRUCTION
        assert "Args:" in getattr(q, tool_name).__doc__  # the schema the LLM sees is built from this
    assert q.TOOLS.keys() == set(names)
    assert len(names) == 8 and "You have eight tools" in q.SYSTEM_INSTRUCTION


@pytest.mark.skipif(
    not (MODEL_DIR / "metadata.json").exists() or not ENGINEERED_FEATURES_PATH.exists(),
    reason="needs a trained final model - run `python main.py` first",
)
def test_a_tool_returns_real_ranked_days_from_the_api(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        def route_to_the_app(url, params=None, timeout=None):
            response = client.get(url.replace(q.API_BASE_URL, ""), params=params)
            return SimpleNamespace(raise_for_status=lambda: None, json=response.json)

        monkeypatch.setattr(q.requests, "get", route_to_the_app)
        result = q.get_top_overpredictions(forecast="manager", n=3)

    assert result["forecast_judged"] == "manager"
    assert len(result["days"]) == 3
    assert {d["direction"] for d in result["days"]} == {"over_forecast"}
