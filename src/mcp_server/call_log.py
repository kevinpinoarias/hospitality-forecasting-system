"""
One structured log line per tool call: which tool, how long it took, and
whether it succeeded - nothing else. The arguments (dates, sales figures) and
the results are never logged, and the question the person typed never
reaches this server at all.

The lines go to stderr. With the stdio transport, stdout carries the MCP
protocol itself, so anything else printed there would break the connection.
Claude Desktop keeps a server's stderr in its own log file.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

LOGGER_NAME = "src.mcp_server.calls"
logger = logging.getLogger(LOGGER_NAME)


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname}
        line.update(getattr(record, "fields", {}))
        return json.dumps(line)


def configure_stderr_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def logged_tool(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Wraps a (blocking) tool function: runs it on a worker thread, so a slow
    API call does not freeze the server, and logs one line about the call.
    functools.wraps keeps the name, docstring and signature, which FastMCP
    reads to build the tool's description and input schema."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        outcome, error_type = "ok", None
        try:
            return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
        except Exception as exc:
            outcome, error_type = "error", type(exc).__name__
            raise
        finally:
            fields = {
                "event": "tool_call",
                "tool": fn.__name__,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "outcome": outcome,
            }
            if error_type:
                fields["error_type"] = error_type
            logger.info("tool_call", extra={"fields": fields})

    return wrapper
