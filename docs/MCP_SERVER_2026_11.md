# MCP server for the forecasting tools (design record)

Status: **accepted**, Phase 1 built on 21 November 2026 (branch `mcp-server`, from `Product-branch`). Decisions are Kevin's; the reasoning is recorded here so it can be picked up again.

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

## 5. Where MCP is not needed

- **The customer-facing Gradio app.** Its users chat with the Gemini agent; MCP adds nothing they can see. Phase 2 would make the agent an MCP client for the architectural gain (one tool layer), not for them.
- **The maintainer's own questions.** Claude Code already answers them; the MCP adds consistency (tested tools instead of improvised scripts), not new reach.
- **Multi-site support.** If all sites share one database, a site is a parameter of the existing tools and the API, not a reason for MCP.
- **The forecasting itself.** Already a separate service; MCP changes nothing about how numbers are computed.

## 6. Parked and later

- **Manager event flags (reason 3): parked by Kevin on 21 November 2026** so Phase 1 is read-only. The design when it comes back: a `propose_event_flag` MCP tool that stores a pending flag only; approval and rejection only through a CLI outside any AI client's reach; an append-only, hash-chained store; approved flags shown beside the unchanged model figure with a clamped, clearly labelled manager adjustment. Open: where the store lives (one machine or shared), and no Approve button on the public Gradio app (anonymous visitors; Cloud Run's disk is wiped on scale-to-zero).
- **Phase 2** (only on Kevin's go): the Gradio agent loads the tools through `langchain-mcp-adapters`; re-run the 19-case evaluation against the direct-tool baseline; measure the per-call latency of MCP against direct calls and add it here.
- **Going public:** `Product-branch` (and this branch) are published to the public repository later, as their own planned step, because merging into `main` also retrains and redeploys the live model.
