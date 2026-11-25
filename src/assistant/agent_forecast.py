"""
LangChain version of the natural-language forecasting assistant (Track B,
step 2 - orchestration).

Built specifically for portfolio purposes - demonstrates working with a
recognised agent framework (LangChain's create_agent, which runs on
LangGraph's execution engine as of the 1.0 release). The hand-rolled
version in query_forecast.py is deliberately kept as-is alongside this one:
it's the version this project would actually build on for a real product,
where full control and transparency matter more than framework recognition.

Adds two real capabilities beyond step 1:
1. Conversation memory - this runs as an interactive loop, not one
   question per process, so follow-up questions ("was that accurate?")
   actually work.
2. A grounded accuracy check - get_forecast() now also returns
   `actual_sales` for past dates (a real API extension, not a prompt
   trick), so the assistant can honestly compare a past prediction to
   what really happened. It never invents a real-world reason for any
   gap - only reports the real numbers, per the same grounding principle
   used throughout this project.
"""

from __future__ import annotations

import datetime as dt
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from src.assistant import tools as tool_functions
from src.assistant.query_forecast import MODEL, SYSTEM_INSTRUCTION, TOOL_FUNCTIONS

# The model chosen in the chat's dropdown travels with each message as a per-call setting
# (config["configurable"]["model_version"]); the two forecast tools read it from there, so the
# LLM never has to remember or pass it. A model the person names in the chat (the tools'
# model_version input) wins over the dropdown.
MODEL_VERSION_KEY = "model_version"


def selected_model(config: RunnableConfig | None) -> str | None:
    value = ((config or {}).get("configurable") or {}).get(MODEL_VERSION_KEY)
    return None if value in (None, "", "standard") else value


def get_forecast(date: str, forecast_sales: float | None = None, model_version: str | None = None, config: RunnableConfig = None) -> dict:
    return tool_functions.get_forecast(date, forecast_sales, model_version or selected_model(config))


def get_forecast_range(start_date: str, end_date: str, model_version: str | None = None, config: RunnableConfig = None) -> dict:
    return tool_functions.get_forecast_range(start_date, end_date, model_version or selected_model(config))


# The same descriptions the LLM has always seen (the config parameter is hidden from it by LangChain).
get_forecast.__doc__ = tool_functions.get_forecast.__doc__
get_forecast_range.__doc__ = tool_functions.get_forecast_range.__doc__
_WITH_SELECTION = {tool_functions.get_forecast: get_forecast, tool_functions.get_forecast_range: get_forecast_range}
DIRECT_TOOLS = [_WITH_SELECTION.get(fn, fn) for fn in TOOL_FUNCTIONS]


def _extract_text(content) -> str:
    """The final message's content can be a plain string or a list of
    content blocks (e.g. [{"type": "text", "text": "..."}]) depending on
    the model - normalise either shape into plain text for display."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "".join(parts) if parts else str(content)

load_dotenv()

ACCURACY_CHECK_INSTRUCTION = """

You can also check how accurate a past forecast turned out to be, subject
to the actual_sales rule already given above and the known-closure-day
rule below.

If `is_known_closure_day` is true for that date, do NOT run an accuracy
comparison at all - say plainly that the venue is always closed that day
(Christmas Day or New Year's Day), so `predictions.best_estimate` isn't a
real forecast to judge and `actual_sales` being 0 reflects the closure,
not a miss. Stop there; this isn't a case of the model being right or
wrong.

Otherwise: only when a past date's tool result includes `actual_sales`
that is NOT null, compare it to `predictions.best_estimate` and describe
the gap in plain terms (e.g. "that was about £X higher/lower than
actually came in") - unless `prediction_source` is
"model_had_seen_the_day", in which case the model had already learned from
that day's result: give both figures if asked, but say plainly it isn't a
fair test of accuracy. Never describe this with technical terms such as
"backtest", "out-of-sample" or "in-sample". If
`actual_sales` is null for that date, say plainly that no real sales
figure is on record for it, and stop there - do not go on to estimate,
guess, or otherwise imply a comparison happened. Only ever
describe the size and direction of a REAL gap using these real numbers -
never guess or invent a real-world reason for why the gap happened
(weather, events, etc.) unless the tool itself told you that reason. If
asked "why" and you don't have a real explanation from the tool's data,
say plainly that you don't have enough information to explain the cause,
only the size of the difference.
"""


# Where the agent's tools come from: "direct" (the default, and what the
# deployed app uses) hands it the Python functions in tools.py; "mcp" loads the
# same eight tools through the MCP server (src/assistant/mcp_tools.py). Set
# ASSISTANT_TOOLS=mcp to try it; the system prompt and the agent are the same.
TOOLS_ENV = "ASSISTANT_TOOLS"


def tools_mode() -> str:
    return "mcp" if os.environ.get(TOOLS_ENV, "").strip().lower() == "mcp" else "direct"


def build_agent(mode: str | None = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set - check your .env file.")

    llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key)
    mode = mode or tools_mode()

    if mode == "mcp":
        # Imported only here, so the deployed (direct) app needs neither
        # langchain-mcp-adapters nor the MCP server.
        from src.assistant.mcp_tools import MCPAgent, shared_session

        session = shared_session()
        graph = create_agent(
            model=llm,
            tools=session.tools,
            system_prompt=SYSTEM_INSTRUCTION + ACCURACY_CHECK_INSTRUCTION,
            checkpointer=InMemorySaver(),
        )
        return MCPAgent(graph, session)

    return create_agent(
        model=llm,
        tools=DIRECT_TOOLS,
        system_prompt=SYSTEM_INSTRUCTION + ACCURACY_CHECK_INSTRUCTION,
        checkpointer=InMemorySaver(),
    )


def invoke_agent(agent, question: str, thread_id: str, model_version: str | None = None):
    """
    Runs one question through the agent, always with a fresh statement of
    today's real date attached.

    The LLM has no innate, reliable sense of the actual current date - left
    ungrounded, it will guess at what "next Saturday" or "this weekend"
    means, and can guess wrong (confirmed: it once resolved "next Saturday"
    to a date that had already passed). Computed fresh on every call rather
    than baked into the system prompt once at startup, since this runs as a
    long-lived deployed service - a "today" fixed at startup would silently
    go stale after the first day.

    `model_version` is the model chosen in the app (None or "standard" for
    the live model). With ASSISTANT_TOOLS=mcp the tools come from the MCP
    server and only a model named in the chat is used.
    """
    today = dt.datetime.now(dt.timezone.utc).date()
    date_context = f"For reference, today's real date is {today.isoformat()} ({today.strftime('%A')})."

    return agent.invoke(
        {
            "messages": [
                {"role": "system", "content": date_context},
                {"role": "user", "content": question},
            ]
        },
        config={"configurable": {"thread_id": thread_id, MODEL_VERSION_KEY: model_version or "standard"}},
    )


def run_chat() -> None:
    agent = build_agent()
    thread_id = "cli-session"  # same thread for the whole run = shared memory

    print("Hospitality forecast assistant (LangChain). Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = invoke_agent(agent, question, thread_id)
        reply = _extract_text(result["messages"][-1].content)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    run_chat()
