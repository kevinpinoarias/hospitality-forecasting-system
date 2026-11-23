# MCP server for the forecasting tools (design record)

Status: **accepted**. Phase 1 built on 21 November 2026 (branch `mcp-server`), Phase 2 the same day (branch `mcp-phase2`), both from `Product-branch`. Decisions are Kevin's; the reasoning is recorded here so it can be picked up again.

## 1. Problem

The chat assistant's eight tools (`get_forecast`, `get_forecast_range`, the four error rankings, `get_model_comparison`, `get_sales_and_forecast_patterns`) lived in the same file as the Gemini system prompt and the CLI's tool loop (`src/assistant/query_forecast.py`), and were handed to the agent as plain Python functions. Three consequences:

1. **The tools could not be used, tested or changed apart from the agent.** Their tests skip when the Gemini library is missing, because importing the tools imported Gemini. Swapping the model or the agent meant editing the file that held the tools. Any other program wanting them had to import the whole assistant package.
2. **A second kind of user had no safe way in.** The maintainer can already ask anything through Claude Code, which reads the code and runs scripts - but it improvises a new way to get each number, untested, and it has full access to the repository and a shell. The finance team should be able to ask forecast questions from an AI client they already use, with access to read forecasts and nothing else.
3. **Manager event flags** (a manager flags a local event the model cannot know about, with human approval and an audit trail) need a write path that no AI can approve by itself.

What was already separate, and is not part of this problem: the forecasting itself (model, features, weather, rankings) runs in the forecasting API, its own Cloud Run service. Six of the eight tools are thin HTTP wrappers around it. MCP does not separate the forecasting from the agent - that was done in Phase 0 - it gives the **thin tool layer** its own interface, tests and users.

## 2. Options

| Option | For | Against |
|---|---|---|
| **Keep the tools inside the agent** | Nothing to build. | The coupling in point 1 stays; a second user means a second app; no place for an approval boundary. |
| **A plain REST API for the second user** | The forecasting API already is one; finance could call `/predict` directly or a small client could be built on it. | Every question needs someone to build a report, script or dashboard for it. Answers only the questions designed in advance. Nothing for the agent's coupling. |
| **An MCP server over the existing tools** (chosen) | One tested contract for the tools, usable by any MCP client (Claude Desktop, Claude Code, and the Gradio agent in Phase 2). No new app for finance. A natural place for a propose-only write tool. | A new dependency (`mcp`) and a new process. Grounding rules must travel with the tools, because an MCP client never sees the Gemini system prompt. Local stdio means each user needs the repository and Python. |

## 3. Decision

An MCP server, `src/mcp_server/`, built with the official Python SDK (`mcp`, FastMCP), with these choices (all Kevin's, 21 November 2026):

- **The MCP tools call the assistant's existing tool functions, which call the API** (option A of the plan) - the common shape for MCP servers (an adapter over an existing API) and consistent with Phase 0/Track B: one place computes the numbers, so the CLI, the Gradio app and an MCP client cannot disagree. Rejected: loading the model inside the server (offline, but 25 CatBoost models loaded at every client start, a second copy of the serving path that could drift, and Phase 2 would need the model in the Gradio image).
- **The tools moved to `src/assistant/tools.py`**, with no Gemini import. `query_forecast.py` re-exports them, so every existing import still works; the move changed no behaviour (the existing tests pass unchanged apart from where they patch HTTP).
- **All eight tools, same names**, so the Gradio agent can load them through MCP in Phase 2 with no change to its system prompt (which names them) and the 19-case evaluation stays a fair comparison.
- **Each MCP tool is a thin wrapper:** it checks the input, calls the same function, and returns its result unchanged. Tool descriptions are the assistant's own tested docstrings plus the reading rules (closure days, one figure or two depending on the weather, `actual_sales` as the only source, never inventing a reason).
- **Telling the client what the server is for** (added 21 November 2026 after the first Claude Desktop test, where a chat with no project context read "what's the forecast for Christmas Eve?" as a weather question and never called the tools, while the ranking question was answered correctly by the rules): the server sends short `instructions` (sales forecasts in pounds for one venue; "forecast" questions about dates mean sales), and the two forecast tools' descriptions start "Sales forecast in pounds for the hospitality venue - not a weather forecast". The lead-in is added only in the MCP layer, so the Gemini assistant's tested descriptions are unchanged.
- **No horizon limit:** the MCP serves any date the assistant does. Input checks cover only the shape of the input (a real YYYY-MM-DD date, an end not before the start, at most 31 days in a range, a sales estimate above 0); what the data allows is still the API's call, and its message is passed on.
- **Local stdio only.** No network port, no deployment, no authentication needed. All eight tools are marked read-only.
- **`mcp>=1.24,<2`**: version 2 renamed FastMCP to MCPServer, and `langchain-mcp-adapters` (Phase 2) requires `mcp` below 2. Moving to 2.x later is a short rename.
- **Logging:** one JSON line per tool call on stderr - tool name, duration, outcome, error type - and nothing else: no arguments, no results. The question a person types never reaches the server.
- **CI:** the deploy job's change filter skips `src/mcp_server/`, so MCP changes never redeploy the API; `.dockerignore` keeps it out of the API image. A test checks that nothing outside the server imports it, which is what makes the skip safe.

## 4. Consequences

- The tools have their own tests (`tests/test_mcp_server.py`): the tool list, typed schemas, parity with the direct functions (with faked HTTP and, when a model is present, against the real API in-process), refused bad input, API failures turned into actionable messages, and log lines that hold no arguments or results.
- The maintainer and the finance team can connect Claude Desktop (README: "MCP server"). Answers are then written by Anthropic's Claude, not Google's Gemini: the forecast figures go to a different provider.
- Each user runs the server on their own computer, so each needs the repository, Python and `requirements-mcp.txt`. For a team this is the point where a remote MCP server (HTTP, sign-in, shared data) would make sense - which would be a deployment, and is not part of this work.
- The forecasting API has no authentication (a Phase 0 decision); the server reads only what that public API already serves.

## 5. Phase 2: the Gradio agent as an MCP client (21 November 2026)

Built on Kevin's go, with his choices:

- **A switch, off by default.** `ASSISTANT_TOOLS=mcp` makes the agent load its eight tools through the MCP server (`langchain-mcp-adapters`); anything else keeps the direct Python functions. The deployed app is unchanged: it does not set the switch, and its image needs neither the adapter nor the server. Turning it on in production belongs to the "publish later" step (the assistant image would then need `src/mcp_server/` and `mcp`).
- **One shared session** (`src/assistant/mcp_tools.py`). The adapter's default opens a new session - a new server process - for every tool call. Instead the server is started once and one session stays open on a background thread with its own event loop. The adapter's tools only run asynchronously, so each agent call runs as `ainvoke` on that loop behind the same blocking `invoke` the app and the evaluation already use. The agent, its tool loop (LangChain's `create_agent`) and the system prompt are unchanged; the tool names are the same. Not handled yet: if the server process dies, calls fail until the app is restarted (no automatic reconnect).
- **What Gemini sees differs slightly:** through MCP each tool's description is the MCP one (the assistant's docstring plus the reading rules and, for the two forecast tools, the "sales, not weather" lead-in), and a tool result arrives as a text block rather than a JSON string. The evaluator now reads both shapes.

**Evaluation (the 19 cases, 20 turns), each mode run twice, same day, same questions:**

| Run | Turns | Issues flagged | Tool choices |
|---|---|---|---|
| Direct, run 1 | 20 | 1 (turn 1: names "Monday") | baseline |
| Direct, run 2 | 20 | 0 | identical to run 1 except turn 11 (run 1 also called `get_top_best_days`) |
| MCP, run 1 | 11 of 20 | 1 (turn 1: names "Monday") | stopped by Gemini's rate limit (429), not by MCP |
| MCP, run 2 | 20 | 0 | identical to direct run 2 |
| MCP, run 3 (repeat of run 1) | 20 | 2 (turn 1: "Monday"; turn 10: "Friday") | identical to direct run 2 |

No difference that can be put down to MCP: every MCP run chose exactly the tools the direct runs chose, and the only flags are one kind (a weekday named that is not the asked date's own) that appears in both modes. For a Saturday such as 28 November 2026 the tool result itself holds the Friday payday before it and the Monday St Andrew's Day bank holiday after it, so these flags are most likely the weekday check being too strict rather than wrong answers; the evaluation prints only the flag, not the answer, so this is not confirmed. Gemini's answers vary between runs either way.

**Latency** (`scripts/measure_mcp_latency.py`, medians, 21 November 2026): a tool call through MCP over the shared session adds about **3 ms** with no network (get_model_comparison: 0.0 against 2.7 ms) and about **13 ms** on a live API call (get_forecast: 682 against 694 ms) - negligible next to the API and Gemini; the adapter's default of a new session per call adds about **2.7 s** to every call (starting a server process), which is why the session is shared.

## 6. Where MCP is not needed

- **The customer-facing Gradio app.** Its users chat with the Gemini agent; MCP adds nothing they can see. Phase 2 would make the agent an MCP client for the architectural gain (one tool layer), not for them.
- **The maintainer's own questions.** Claude Code already answers them; the MCP adds consistency (tested tools instead of improvised scripts), not new reach.
- **Multi-site support.** If all sites share one database, a site is a parameter of the existing tools and the API, not a reason for MCP.
- **The forecasting itself.** Already a separate service; MCP changes nothing about how numbers are computed.

## 7. Parked and later

- **Manager event flags (reason 3): parked by Kevin on 21 November 2026** so Phase 1 is read-only. The design when it comes back: a `propose_event_flag` MCP tool that stores a pending flag only; approval and rejection only through a CLI outside any AI client's reach; an append-only, hash-chained store; approved flags shown beside the unchanged model figure with a clamped, clearly labelled manager adjustment. Open: where the store lives (one machine or shared), and no Approve button on the public Gradio app (anonymous visitors; Cloud Run's disk is wiped on scale-to-zero).
- **Going public:** in the November 2026 release the whole private branch (this work included) was merged into the public repository in one go, which also retrained the live model on the newer data. The MCP server is published but not switched on in the live assistant (`ASSISTANT_TOOLS` stays unset there). The MCP server is published but not switched on in the live assistant (`ASSISTANT_TOOLS` stays unset there).
