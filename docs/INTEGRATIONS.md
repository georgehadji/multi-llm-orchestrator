# Integrations: MCP Bridge & External Tool Interop

## CodeWhale Integration

[CodeWhale](https://github.com/Hmbown/CodeWhale) is a local-first agent harness
that can consume the orchestrator's MCP server as an external tool provider.

### Quick Start

Add the orchestrator's MCP server to CodeWhale's MCP config:

```json
// ~/.codewhale/mcp.json
{
  "mcpServers": {
    "orchestrator": {
      "command": "python",
      "args": ["-m", "orchestrator.mcp_server"]
    }
  }
}
```

### Available MCP Tools

| Tool | Description | Route |
|------|-------------|-------|
| `orch_search` | Fast keyword/BM25 search over orchestrator knowledge | `orchestrator.mcp_server` |
| `orch_query` | Hybrid search with re-ranking (best quality) | `orchestrator.mcp_server` |
| `orch_get` | Retrieve document/memory by ID | `orchestrator.mcp_server` |
| `orch_status` | System health and statistics | `orchestrator.mcp_server` |
| `orch_memory` | Store/retrieve memories | `orchestrator.mcp_server` |
| `orch_persona` | Get/set persona settings | `orchestrator.mcp_server` |
| `orch_session` | Manage conversation sessions | `orchestrator.mcp_server` |

### Use Case: Planning Backend + Execution Frontend

The orchestrator serves as a planning/optimization backend while CodeWhale
handles interactive execution:

1. Orchestrator decomposes a project into tasks, optimizes model selection,
   runs the critique cycle with LSP validation
2. CodeWhale agent queries `orch_status` / `orch_query` to retrieve
   orchestrator's analysis
3. CodeWhale handles interactive file editing, shell commands, git operations
4. Orchestrator's snapshot store provides rollback if CodeWhale's changes
   need to be reverted

---

## Running the MCP Server

### As stdio server (subprocess, suitable for Claude Desktop / Cursor)

```bash
python -m orchestrator.mcp_server
```

### As HTTP server (shared, long-lived)

```bash
python -m orchestrator.mcp_server --http --port 8181
```

### As daemon

```bash
python -m orchestrator.mcp_server --http --port 8181 --daemon
```

### Config for Claude Desktop

```json
{
  "mcpServers": {
    "orchestrator": {
      "command": "python",
      "args": ["-m", "orchestrator.mcp_server"],
      "env": {
        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}"
      }
    }
  }
}
```
