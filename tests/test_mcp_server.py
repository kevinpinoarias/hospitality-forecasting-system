"""
Tests for the MCP server (src/mcp_server/). No LLM and no network: the HTTP
calls inside src/assistant/tools.py are replaced, and two tests route them
to the real API in-process.

What is checked:
- the server offers exactly the assistant's eight tools, all read-only, with
  the reading rules in their descriptions;
- parity: each MCP tool returns exactly what calling the assistant's tool
  function directly returns;
- bad input is refused with a helpful message before any API call, and API
  failures come back as messages a person can act on;
- one log line per call, holding no arguments or results;
- the API never imports the MCP server (the CI deploy filter relies on it).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
pytest.importorskip("dotenv")

import requests
from mcp.server.fastmcp.exceptions import ToolError

from src.api.history import ENGINEERED_FEATURES_PATH
from src.assistant import tools
from src.mcp_server import call_log
from src.mcp_server.server import mcp
from src.models.final_model import MODEL_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_TOOLS = ["get_forecast", "get_forecast_range"]
RANKING_TOOLS = ["get_top_worst_days", "get_top_overpredictions", "get_top_underpredictions", "get_top_best_days"]
STATIC_TOOLS = ["get_model_comparison", "get_sales_and_forecast_patterns"]


def call(name: str, arguments: dict | None = None):
    """Calls a tool the way an MCP client does and returns the decoded result."""
    result = asyncio.run(mcp.call_tool(name, arguments or {}))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def listed_tools() -> dict:
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload, self.status_code = payload, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self) -> dict:
        return json.loads(json.dumps(self.payload))  # a fresh copy each time


def fake_forecast(date: str) -> dict:
    return {"date": date, "day_of_week": "Thursday", "predictions": {"best_estimate": 6150.5}, "actual_sales": None}


@pytest.fixture
def fake_api(monkeypatch):
    """Replaces the API with canned answers and records every request."""
    seen = []

    def fake_post(url, json=None, timeout=None):
        seen.append({"url": url, "json": json})
        return FakeResponse({"results": [fake_forecast(r["date"]) for r in json["requests"]]})

    def fake_get(url, params=None, timeout=None):
        seen.append({"url": url, "params": params})
        return FakeResponse({"ranking": params["kind"], "forecast_judged": params["forecast"], "days": [], "limit": params["limit"]})

    monkeypatch.setattr(tools.requests, "post", fake_post)
    monkeypatch.setattr(tools.requests, "get", fake_get)
    return seen


# --- what the server offers -------------------------------------------------

def test_the_server_offers_exactly_the_assistants_eight_tools():
    assert set(listed_tools()) == set(tools.TOOLS)


def test_every_tool_is_marked_read_only():
    for tool in listed_tools().values():
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False


def test_descriptions_keep_the_assistants_wording_and_add_the_reading_rules():
    offered = listed_tools()
    for name, fn in tools.TOOLS.items():
        first_sentence = " ".join(fn.__doc__.split())[:40]
        assert offered[name].description.startswith(first_sentence)
    for name in FORECAST_TOOLS:
        text = offered[name].description
        assert "is_known_closure_day" in text and "heavy_rain_scenario" in text and "never estimate" in text
    for name in RANKING_TOOLS:
        assert "over_forecast" in offered[name].description


def test_parameters_are_typed_for_the_client():
    offered = listed_tools()
    assert offered["get_top_worst_days"].inputSchema["properties"]["forecast"]["enum"] == ["model", "manager"]
    assert offered["get_forecast"].inputSchema["required"] == ["date"]
    assert set(offered["get_forecast_range"].inputSchema["required"]) == {"start_date", "end_date"}


# --- parity: the MCP tool returns exactly what the function returns ---------

@pytest.mark.parametrize("name, arguments", [
    ("get_forecast", {"date": "2026-12-24"}),
    ("get_forecast", {"date": "2026-12-24", "forecast_sales": 7000.0}),
    ("get_forecast_range", {"start_date": "2026-12-19", "end_date": "2026-12-27"}),
    ("get_top_worst_days", {}),
    ("get_top_overpredictions", {"forecast": "manager", "n": 5}),
    ("get_top_underpredictions", {"n": 20}),
    ("get_top_best_days", {"forecast": "model", "n": 50}),
    ("get_model_comparison", {}),
    ("get_sales_and_forecast_patterns", {}),
])
def test_each_tool_returns_exactly_what_the_assistants_function_returns(fake_api, name, arguments):
    assert call(name, arguments) == tools.TOOLS[name](**arguments)


def test_the_forecast_tool_sends_the_same_request_as_the_function(fake_api):
    call("get_forecast", {"date": "2026-12-24", "forecast_sales": 7000.0})
    tools.get_forecast("2026-12-24", 7000.0)
    assert fake_api[0] == fake_api[1]
    assert fake_api[0]["json"] == {"requests": [{"date": "2026-12-24", "forecast_sales": 7000.0}]}


needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "metadata.json").exists() or not ENGINEERED_FEATURES_PATH.exists(),
    reason="needs a trained final model - run `python main.py` first",
)


@pytest.fixture
def real_api(monkeypatch):
    """Routes the tools' HTTP calls to the real API, in-process."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        def post(url, json=None, timeout=None):
            r = client.post(url.replace(tools.API_BASE_URL, ""), json=json)
            return SimpleNamespace(raise_for_status=lambda: None, json=r.json)

        def get(url, params=None, timeout=None):
            r = client.get(url.replace(tools.API_BASE_URL, ""), params=params)
            return SimpleNamespace(raise_for_status=lambda: None, json=r.json)

        monkeypatch.setattr(tools.requests, "post", post)
        monkeypatch.setattr(tools.requests, "get", get)
        yield


@needs_model
def test_parity_holds_against_the_real_api_for_a_past_date(real_api):
    via_mcp = call("get_forecast", {"date": "2025-06-14"})
    assert via_mcp == tools.get_forecast("2025-06-14")
    assert via_mcp["day_of_week"] == "Saturday"


@needs_model
def test_parity_holds_against_the_real_api_for_a_ranking(real_api):
    via_mcp = call("get_top_overpredictions", {"forecast": "manager", "n": 3})
    assert via_mcp == tools.get_top_overpredictions("manager", 3)
    assert len(via_mcp["days"]) == 3


# --- bad input --------------------------------------------------------------

@pytest.mark.parametrize("bad_date", ["24/12/2026", "2026-13-01", "2026-02-30", "tomorrow", "2026-1-5", ""])
def test_a_badly_written_or_impossible_date_is_refused_before_any_api_call(fake_api, bad_date):
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        call("get_forecast", {"date": bad_date})
    assert fake_api == []


def test_a_range_that_ends_before_it_starts_is_refused(fake_api):
    with pytest.raises(ToolError, match="before start_date"):
        call("get_forecast_range", {"start_date": "2026-12-10", "end_date": "2026-12-01"})
    assert fake_api == []


def test_a_range_longer_than_31_days_is_refused(fake_api):
    with pytest.raises(ToolError, match="covers 32 days; at most 31"):
        call("get_forecast_range", {"start_date": "2026-12-01", "end_date": "2027-01-01"})
    assert fake_api == []


def test_a_bad_date_in_a_range_names_the_field(fake_api):
    with pytest.raises(ToolError, match="end_date must be a real date"):
        call("get_forecast_range", {"start_date": "2026-12-01", "end_date": "31-12-2026"})


@pytest.mark.parametrize("bad_estimate", [0, -500.0])
def test_a_sales_estimate_of_zero_or_less_is_refused(fake_api, bad_estimate):
    with pytest.raises(ToolError, match="above 0"):
        call("get_forecast", {"date": "2026-12-24", "forecast_sales": bad_estimate})
    assert fake_api == []


def test_an_unknown_forecast_owner_is_refused_by_the_schema(fake_api):
    with pytest.raises(ToolError, match="forecast"):
        call("get_top_worst_days", {"forecast": "robot"})
    assert fake_api == []


def test_a_non_number_day_count_is_refused(fake_api):
    with pytest.raises(ToolError):
        call("get_top_best_days", {"n": "lots"})
    assert fake_api == []


def test_the_apis_own_refusal_is_passed_on_in_plain_words(monkeypatch):
    detail = "2020-01-01 is before the start of the historical data (2024-01-02)."
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse({"detail": detail}, status=422))
    with pytest.raises(ToolError, match=re.escape(detail)):
        call("get_forecast", {"date": "2020-01-01"})


def test_an_unreachable_api_gives_an_actionable_message(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(tools.requests, "get", down)
    with pytest.raises(ToolError, match="could not be reached"):
        call("get_top_worst_days", {})


def test_a_server_error_does_not_leak_internals(monkeypatch):
    monkeypatch.setattr(tools.requests, "post", lambda *a, **k: FakeResponse({"detail": "Traceback ..."}, status=500))
    with pytest.raises(ToolError, match="HTTP 500") as err:
        call("get_forecast", {"date": "2026-12-24"})
    assert "Traceback" not in str(err.value)


# --- logging ----------------------------------------------------------------

def _call_lines(caplog) -> list[dict]:
    formatter = call_log.JsonLineFormatter()
    return [json.loads(formatter.format(r)) for r in caplog.records if r.name == call_log.LOGGER_NAME]


def test_one_log_line_per_call_with_tool_duration_and_outcome(fake_api, caplog):
    caplog.set_level(logging.INFO, logger=call_log.LOGGER_NAME)
    call("get_forecast", {"date": "2026-12-24"})
    with pytest.raises(ToolError):
        call("get_forecast", {"date": "not-a-date"})

    first, second = _call_lines(caplog)
    assert first["tool"] == "get_forecast" and first["outcome"] == "ok" and first["duration_ms"] >= 0
    assert second["outcome"] == "error" and second["error_type"] == "ToolError"
    assert set(first) == {"time", "level", "event", "tool", "duration_ms", "outcome"}


def test_log_lines_hold_no_arguments_or_results(fake_api, caplog):
    caplog.set_level(logging.INFO)
    marker_date, marker_sales = "2026-12-24", 9876.54
    call("get_forecast", {"date": marker_date, "forecast_sales": marker_sales})
    with pytest.raises(ToolError):
        call("get_forecast", {"date": "private note: call Jane on 07700 900123"})

    everything_logged = caplog.text + json.dumps(_call_lines(caplog))
    for leaked in [marker_date, "9876", "Jane", "07700", "6150", "Thursday"]:
        assert leaked not in everything_logged


# --- separation from the deployed API ---------------------------------------

def test_nothing_outside_the_mcp_server_and_tests_imports_it():
    """The CI deploy filter skips src/mcp_server/ when deciding whether to
    redeploy the API, which is only safe while no other code imports it."""
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if "mcp_server" in path.parts:
            continue
        if re.search(r"^\s*(from|import)\s+(src\.)?mcp_server\b", path.read_text(encoding="utf-8"), re.MULTILINE):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_the_assistant_tools_need_no_gemini_import():
    text = (PROJECT_ROOT / "src" / "assistant" / "tools.py").read_text(encoding="utf-8")
    assert "genai" not in text and "langchain" not in text
