"""
How much slower is a tool call through MCP than calling the function directly?

Three ways of calling the same tool, interleaved so they see the same API
conditions:
- direct:      the Python function in src/assistant/tools.py
- mcp_shared:  through the MCP server over ONE open stdio session (what
               ASSISTANT_TOOLS=mcp uses)
- mcp_percall: through the MCP server with langchain-mcp-adapters' default,
               a new session - a new server process - for every call

Two tools: get_model_comparison (no network, so the difference is pure MCP
overhead) and get_forecast for one date (a real call to the forecasting API,
the realistic case). The API is woken up first so a cold start is not counted.

    python scripts/measure_mcp_latency.py [--repeats 15] [--api-repeats 8]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant import tools  # noqa: E402
from src.assistant.mcp_tools import SERVER_NAME, MCPToolSession, mcp_client  # noqa: E402

DATE = "2026-12-24"


def timed(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000


def summary(ms: list[float]) -> str:
    ms = sorted(ms)
    p90 = ms[min(len(ms) - 1, int(round(0.9 * (len(ms) - 1))))]
    return f"median {statistics.median(ms):8.1f} ms   p90 {p90:8.1f} ms   (n={len(ms)})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--api-repeats", type=int, default=8)
    args = parser.parse_args()

    session = MCPToolSession()
    shared = {t.name: t for t in session.tools}

    # Tool objects from get_tools() open a new session on every call, so load
    # them once outside the timing and time only the calls.
    percall_tools = session.run(mcp_client().get_tools(server_name=SERVER_NAME))
    percall = {t.name: t for t in percall_tools}

    print("Waking the API (not timed)...")
    tools.get_forecast(DATE)

    cases = [
        ("get_model_comparison (no network)", "get_model_comparison", {}, args.repeats),
        (f"get_forecast {DATE} (live API)", "get_forecast", {"date": DATE}, args.api_repeats),
    ]
    for label, name, arguments, n in cases:
        times = {"direct": [], "mcp_shared": [], "mcp_percall": []}
        for _ in range(n):
            times["direct"].append(timed(lambda: tools.TOOLS[name](**arguments)))
            times["mcp_shared"].append(timed(lambda: session.run(shared[name].ainvoke(arguments))))
            times["mcp_percall"].append(timed(lambda: session.run(percall[name].ainvoke(arguments))))
        print(f"\n{label}")
        for way, ms in times.items():
            print(f"  {way:12s} {summary(ms)}")
        base = statistics.median(times["direct"])
        print(f"  overhead vs direct: shared session {statistics.median(times['mcp_shared']) - base:+.1f} ms, "
              f"new session per call {statistics.median(times['mcp_percall']) - base:+.1f} ms")

    session.close()


if __name__ == "__main__":
    main()
