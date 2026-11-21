"""
MCP server for the forecasting tools (Phase 1, November 2026).

Offers the assistant's eight tools (src/assistant/tools.py) to MCP clients
such as Claude Desktop, over stdio only: the client starts this server as a
local process and talks to it through stdin/stdout. Nothing listens on a
network port. Every tool is read-only.

Run it with `python -m src.mcp_server`; see docs/MCP_SERVER_2026_11.md for
why it exists and README.md for connecting Claude Desktop.
"""
