"""
The assistant's tools loaded through the MCP server instead of imported
directly (Phase 2, November 2026). Used only when ASSISTANT_TOOLS=mcp - the
deployed app keeps the direct tools (see agent_forecast.build_agent).

How it works: the MCP server (src/mcp_server/) is started once, as a child
process over stdio, and ONE session to it is kept open for the life of the
app. langchain-mcp-adapters turns the server's tools into LangChain tools
with the same names, so the agent and its system prompt are unchanged.

Two details the library forces:
- Its tools only run asynchronously. The session therefore lives on its own
  background thread with its own event loop, and every agent call is run on
  that loop (MCPAgent.invoke below). Callers - Gradio's chat function, the
  evaluation - stay ordinary blocking code, as before.
- Its default (client.get_tools()) opens a NEW session, i.e. starts a new
  server process, for every tool call. One shared session avoids that; the
  difference is measured in scripts/measure_mcp_latency.py.

If the server process dies, the session is not reopened automatically:
calls fail with an error until the app is restarted. Acceptable for a
switch that is only used locally; to revisit before any deployment.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import threading
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# Not imported from src.config: the assistant's image holds only src/assistant/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_NAME = "hospitality-forecasting"
START_TIMEOUT_SECONDS = 60
CALL_TIMEOUT_SECONDS = 300


def server_connection() -> dict:
    """How to start the MCP server: the same Python, from the project folder,
    with this process's environment (so FORECAST_API_BASE_URL carries over)."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "src.mcp_server"],
        "cwd": str(PROJECT_ROOT),
        "env": dict(os.environ),
    }


def mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({SERVER_NAME: server_connection()})


class MCPToolSession:
    """One open MCP session on a background event loop, and the LangChain
    tools bound to it."""

    def __init__(self) -> None:
        self.tools: list = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-session", daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._holder = asyncio.run_coroutine_threadsafe(self._hold_session(), self._loop)
        if not self._ready.wait(START_TIMEOUT_SECONDS):
            raise RuntimeError(f"The MCP server did not start within {START_TIMEOUT_SECONDS} seconds.")
        if self._error is not None:
            raise RuntimeError(f"The MCP server could not be started: {self._error}") from self._error

    async def _hold_session(self) -> None:
        # The session is opened and closed inside this one task, as the
        # library's async context managers require.
        self._stop = asyncio.Event()
        try:
            async with mcp_client().session(SERVER_NAME) as session:
                self.tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # reported to the starting thread
            self._error = exc
            self._ready.set()
            raise

    def run(self, coroutine, timeout: float = CALL_TIMEOUT_SECONDS) -> Any:
        """Runs a coroutine (an agent call, a tool call) on the session's loop
        and waits for its result."""
        if self._holder.done():
            coroutine.close()
            raise RuntimeError("The MCP session has closed; restart the assistant.")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def close(self) -> None:
        if self._stop is not None and not self._holder.done():
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._holder.result(10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


_shared: MCPToolSession | None = None
_shared_lock = threading.Lock()


def shared_session() -> MCPToolSession:
    """The app's one session, opened on first use and closed at exit."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = MCPToolSession()
            atexit.register(_shared.close)
        return _shared


class MCPAgent:
    """The agent built with MCP tools, with the same blocking `invoke` as the
    direct one: each call runs as `ainvoke` on the session's loop. Everything
    else is passed through to the agent unchanged."""

    def __init__(self, graph, session: MCPToolSession) -> None:
        self._graph = graph
        self._session = session

    def invoke(self, input, config=None, **kwargs):
        return self._session.run(self._graph.ainvoke(input, config=config, **kwargs))

    def __getattr__(self, name: str):
        return getattr(self._graph, name)
