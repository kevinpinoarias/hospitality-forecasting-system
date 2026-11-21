"""
Tests for the assistant's MCP tool source (src/assistant/mcp_tools.py) and
the ASSISTANT_TOOLS switch in agent_forecast.py. No LLM and no network: the
MCP server is started for real over stdio, and only the two tools that need
no API (the model comparison and the sales patterns) are called through it.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

pytest.importorskip("langchain_mcp_adapters")
pytest.importorskip("langchain_google_genai")
pytest.importorskip("mcp")

from langchain_core.messages import ToolMessage

from src.assistant import agent_forecast, tools
from src.assistant.evaluate_output import extract_tool_data
from src.assistant.mcp_tools import MCPAgent, MCPToolSession


def text_of(content) -> str:
    return "".join(block["text"] for block in content if block.get("type") == "text")


@pytest.fixture(scope="module")
def session():
    s = MCPToolSession()
    yield s
    s.close()


# --- the switch --------------------------------------------------------------

@pytest.mark.parametrize("value, mode", [(None, "direct"), ("", "direct"), ("direct", "direct"), ("mcp", "mcp"), (" MCP ", "mcp"), ("other", "direct")])
def test_the_switch_defaults_to_the_direct_tools(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv(agent_forecast.TOOLS_ENV, raising=False)
    else:
        monkeypatch.setenv(agent_forecast.TOOLS_ENV, value)
    assert agent_forecast.tools_mode() == mode


def test_the_direct_agent_is_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    agent = agent_forecast.build_agent("direct")
    assert not isinstance(agent, MCPAgent)
    assert hasattr(agent, "invoke")


def test_the_mcp_agent_gets_the_same_eight_tools(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    agent = agent_forecast.build_agent("mcp")
    assert isinstance(agent, MCPAgent)
    from src.assistant.mcp_tools import shared_session

    assert sorted(t.name for t in shared_session().tools) == sorted(tools.TOOLS)


# --- the shared session -------------------------------------------------------

def test_the_session_offers_the_assistants_eight_tools_by_name(session):
    assert sorted(t.name for t in session.tools) == sorted(tools.TOOLS)


@pytest.mark.parametrize("name", ["get_model_comparison", "get_sales_and_forecast_patterns"])
def test_a_tool_through_mcp_returns_what_the_function_returns(session, name):
    tool = {t.name: t for t in session.tools}[name]
    content = session.run(tool.ainvoke({}))
    assert json.loads(text_of(content)) == tools.TOOLS[name]()


def test_the_session_is_reused_across_calls(session):
    tool = {t.name: t for t in session.tools}["get_model_comparison"]
    first = session.run(tool.ainvoke({}))
    second = session.run(tool.ainvoke({}))
    assert first[0]["text"] == second[0]["text"]
    assert not session._holder.done()  # still the same open session


def test_bad_input_reaches_the_agent_as_a_readable_message(session):
    tool = {t.name: t for t in session.tools}["get_forecast"]
    content = session.run(tool.ainvoke({"date": "24/12/2026"}))
    assert "YYYY-MM-DD" in text_of(content)


# --- the agent wrapper --------------------------------------------------------

class FakeGraph:
    name = "fake-graph"

    async def ainvoke(self, input, config=None):
        return {"thread": threading.current_thread().name, "input": input, "config": config}


def test_the_mcp_agent_runs_each_call_on_the_sessions_loop(session):
    agent = MCPAgent(FakeGraph(), session)
    result = agent.invoke({"messages": ["hi"]}, config={"configurable": {"thread_id": "t"}})
    assert result["thread"] == "mcp-session"
    assert result["input"] == {"messages": ["hi"]} and result["config"]["configurable"]["thread_id"] == "t"
    assert agent.name == "fake-graph"  # everything else passes through


def test_a_closed_session_says_so(session):
    other = MCPToolSession()
    other.close()
    with pytest.raises(RuntimeError, match="closed"):
        other.run(asyncio.sleep(0))


# --- the evaluation reads both kinds of result --------------------------------

def test_the_evaluator_reads_mcp_content_blocks_like_json_strings():
    payload = {"date": "2026-12-24", "day_of_week": "Thursday"}
    direct = ToolMessage(content=json.dumps(payload), tool_call_id="1")
    via_mcp = ToolMessage(content=[{"type": "text", "text": json.dumps(payload)}], tool_call_id="2")
    assert extract_tool_data([direct]) == extract_tool_data([via_mcp]) == [payload]
