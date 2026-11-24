"""
Tests for how a chosen model reaches the forecast: the tools' model_version input, the chat's dropdown
choice travelling as a hidden per-message setting (never handled by the LLM), and the MCP tools' option.
No LLM and no network: the HTTP call is replaced.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("google.genai")

from langchain_core.tools import tool as make_tool  # noqa: E402

from src.assistant import agent_forecast, tools  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return json.loads(json.dumps(self.payload))


@pytest.fixture
def sent(monkeypatch):
    requests_sent = []

    def fake_post(url, json=None, timeout=None):
        requests_sent.append(json)
        return FakeResponse({"results": [{"date": r["date"], "model_version": r.get("model_version", "standard")} for r in json["requests"]]})

    monkeypatch.setattr(tools.requests, "post", fake_post)
    return requests_sent


def test_the_tools_send_model_version_only_when_given(sent):
    tools.get_forecast("2026-12-24")
    tools.get_forecast("2026-12-24", model_version="v1.14")
    tools.get_forecast_range("2026-12-24", "2026-12-25", model_version="v1.13")
    assert sent[0] == {"requests": [{"date": "2026-12-24"}]}
    assert sent[1] == {"requests": [{"date": "2026-12-24", "model_version": "v1.14"}]}
    assert sent[2] == {"requests": [{"date": "2026-12-24", "model_version": "v1.13"}, {"date": "2026-12-25", "model_version": "v1.13"}]}


def test_the_llm_sees_model_version_but_never_the_dropdowns_setting():
    for fn in (agent_forecast.get_forecast, agent_forecast.get_forecast_range):
        args = list(make_tool(fn).args)
        assert "model_version" in args and "config" not in args
        assert fn.__doc__ == getattr(tools, fn.__name__).__doc__  # the same description as before


def test_the_dropdown_choice_reaches_the_forecast(sent):
    tool = make_tool(agent_forecast.get_forecast)
    tool.invoke({"date": "2026-12-24"}, config={"configurable": {"model_version": "v1.141"}})
    assert sent[-1]["requests"][0]["model_version"] == "v1.141"


def test_a_model_named_in_the_chat_wins_over_the_dropdown(sent):
    tool = make_tool(agent_forecast.get_forecast_range)
    tool.invoke({"start_date": "2026-12-24", "end_date": "2026-12-24", "model_version": "v1.12"}, config={"configurable": {"model_version": "v1.141"}})
    assert sent[-1]["requests"][0]["model_version"] == "v1.12"


@pytest.mark.parametrize("setting", [None, "", "standard"])
def test_the_standard_model_sends_nothing_extra(sent, setting):
    make_tool(agent_forecast.get_forecast).invoke({"date": "2026-12-24"}, config={"configurable": {"model_version": setting}})
    assert "model_version" not in sent[-1]["requests"][0]


def test_the_agent_gets_the_wrapped_forecast_tools_and_the_rest_unchanged():
    names = [fn.__name__ for fn in agent_forecast.DIRECT_TOOLS]
    assert names == [fn.__name__ for fn in tools.TOOL_FUNCTIONS]
    assert agent_forecast.get_forecast in agent_forecast.DIRECT_TOOLS and tools.get_top_worst_days in agent_forecast.DIRECT_TOOLS


def test_invoke_agent_passes_the_choice_with_the_message():
    seen = {}

    class FakeAgent:
        def invoke(self, payload, config=None):
            seen.update(config)
            return {"messages": []}

    agent_forecast.invoke_agent(FakeAgent(), "hello", "thread-1", "v1.14")
    assert seen["configurable"] == {"thread_id": "thread-1", "model_version": "v1.14"}
    agent_forecast.invoke_agent(FakeAgent(), "hello", "thread-1")
    assert seen["configurable"]["model_version"] == "standard"


def test_the_instructions_tell_the_llm_how_to_present_a_version():
    from src.assistant.query_forecast import SYSTEM_INSTRUCTION

    text = " ".join(SYSTEM_INSTRUCTION.split())  # the prompt wraps its lines
    assert "model_version_note" in text and "still being tested" in text
    assert "never choose one yourself" in text


# --- the MCP tools -------------------------------------------------------------

mcp = pytest.importorskip("mcp")


def call(name: str, arguments: dict):
    from src.mcp_server.server import mcp as server

    result = asyncio.run(server.call_tool(name, arguments))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def test_the_mcp_forecast_tools_pass_model_version(sent):
    call("get_forecast", {"date": "2026-12-24", "model_version": "v1.14"})
    call("get_forecast_range", {"start_date": "2026-12-24", "end_date": "2026-12-24", "model_version": "v1-post"})
    assert sent[-2]["requests"][0]["model_version"] == "v1.14" and sent[-1]["requests"][0]["model_version"] == "v1-post"


def test_an_unknown_model_is_refused_by_the_mcp_tools_before_any_call(sent):
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="must be one of standard, v1-post"):
        call("get_forecast", {"date": "2026-12-24", "model_version": "v9"})
    assert sent == []
