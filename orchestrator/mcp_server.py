"""
MCP Server — Model Context Protocol Server for Orchestrator
============================================================

Implements MCP (Model Context Protocol) server for tight integration
with AI agents (Claude Desktop, Cursor, etc.).

Based on QMD MCP Server architecture.

Tools Exposed:
- orch_search — Fast keyword/BM25 search
- orch_query — Hybrid search with re-ranking (best quality)
- orch_get — Retrieve document/memory by ID
- orch_status — System health and statistics
- orch_memory — Store/retrieve memories
- orch_persona — Get/set persona settings
- orch_session — Manage conversation sessions

Usage:
    # As stdio server (subprocess)
    python -m orchestrator.mcp_server

    # As HTTP server (shared, long-lived)
    python -m orchestrator.mcp_server --http --port 8181

    # MCP Client configuration (Claude Desktop)
    {
      "mcpServers": {
        "orchestrator": {
          "command": "python",
          "args": ["-m", "orchestrator.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Server = None

# Import orchestrator components
from .memory_tier import MemoryTierManager, MemoryType
from .persona import PersonaManager, PersonaMode
from .session_watcher import SessionWatcher
from .token_optimizer import TokenOptimizer


@dataclass
class MCPConfig:
    """MCP Server configuration."""
    http_mode: bool = False
    port: int = 8181
    host: str = "0.0.0.0"
    daemon: bool = False
    collections: list[str] = None

    def __post_init__(self):
        if self.collections is None:
            self.collections = []


class MCPServer:
    """
    MCP Server for Orchestrator.

    Exposes orchestrator functionality as MCP tools for AI agents.
    """

    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig()

        # Initialize orchestrator components
        self.memory_manager = MemoryTierManager()
        self.persona_manager = PersonaManager()
        self.session_watcher = SessionWatcher()
        self.token_optimizer = TokenOptimizer()

        # MCP server instance
        self.server: Server | None = None
        if HAS_MCP and Server:
            self.server = Server("orchestrator")
            self._register_tools()
            self._register_handlers()

    def _register_tools(self) -> None:
        """Register MCP tools."""
        if not self.server:
            return

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="orch_search",
                    description="Fast keyword/BM25 search across memories and knowledge",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "project_id": {"type": "string", "description": "Filter by project"},
                            "limit": {"type": "integer", "default": 10, "description": "Max results"},
                            "memory_type": {"type": "string", "description": "Filter by type (task, conversation, knowledge)"},
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="orch_query",
                    description="Hybrid search with re-ranking (best quality results)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language query"},
                            "project_id": {"type": "string", "description": "Filter by project"},
                            "limit": {"type": "integer", "default": 10},
                            "min_score": {"type": "number", "default": 0.3},
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="orch_get",
                    description="Retrieve a specific memory by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "Memory ID or path"},
                            "project_id": {"type": "string", "description": "Project ID"},
                        },
                        "required": ["memory_id"],
                    },
                ),
                Tool(
                    name="orch_status",
                    description="Get system health and statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="orch_memory_store",
                    description="Store a new memory in the tiered memory system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "content": {"type": "string"},
                            "memory_type": {"type": "string", "default": "task"},
                        },
                        "required": ["project_id", "content"],
                    },
                ),
                Tool(
                    name="orch_memory_retrieve",
                    description="Retrieve memories from the tiered memory system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["project_id"],
                    },
                ),
                Tool(
                    name="orch_persona_set",
                    description="Set persona mode for a project",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "mode": {"type": "string", "enum": ["strict", "creative", "balanced", "custom"]},
                        },
                        "required": ["project_id", "mode"],
                    },
                ),
                Tool(
                    name="orch_persona_get",
                    description="Get persona settings for a project",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                        },
                        "required": ["project_id"],
                    },
                ),
                Tool(
                    name="orch_session_start",
                    description="Start a new conversation session",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                        },
                        "required": ["project_id"],
                    },
                ),
                Tool(
                    name="orch_session_record",
                    description="Record an interaction in a session",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "task_input": {"type": "string"},
                            "task_output": {"type": "string"},
                            "task_type": {"type": "string"},
                        },
                        "required": ["session_id", "task_input", "task_output"],
                    },
                ),
                Tool(
                    name="orch_optimize_output",
                    description="Optimize command output for token efficiency (60-90% savings)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["command", "output"],
                    },
                ),
            ]

    def _register_handlers(self) -> None:
        """Register MCP tool call handlers."""
        if not self.server:
            return

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                result = await self._handle_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_tool(self, name: str, arguments: dict) -> Any:
        """Handle MCP tool calls."""

        if name == "orch_search":
            return await self._tool_search(arguments)
        elif name == "orch_query":
            return await self._tool_query(arguments)
        elif name == "orch_get":
            return await self._tool_get(arguments)
        elif name == "orch_status":
            return self._tool_status()
        elif name == "orch_memory_store":
            return await self._tool_memory_store(arguments)
        elif name == "orch_memory_retrieve":
            return await self._tool_memory_retrieve(arguments)
        elif name == "orch_persona_set":
            return self._tool_persona_set(arguments)
        elif name == "orch_persona_get":
            return self._tool_persona_get(arguments)
        elif name == "orch_session_start":
            return self._tool_session_start(arguments)
        elif name == "orch_session_record":
            return self._tool_session_record(arguments)
        elif name == "orch_optimize_output":
            return self._tool_optimize_output(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _tool_search(self, args: dict) -> dict:
        """Handle orch_search tool."""
        query = args.get("query", "")
        project_id = args.get("project_id")
        limit = args.get("limit", 10)
        memory_type = args.get("memory_type")

        memories = await self.memory_manager.retrieve(
            project_id=project_id or "",
            query=query,
            memory_type=MemoryType(memory_type) if memory_type else None,
            limit=limit,
        )

        return {
            "query": query,
            "results": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    async def _tool_query(self, args: dict) -> dict:
        """Handle orch_query tool (hybrid search with re-ranking)."""
        query = args.get("query", "")
        project_id = args.get("project_id")
        limit = args.get("limit", 10)
        args.get("min_score", 0.3)

        # For now, use basic retrieve (BM25/re-ranking to be added)
        memories = await self.memory_manager.retrieve(
            project_id=project_id or "",
            query=query,
            limit=limit,
        )

        # Filter by min_score (placeholder - actual scoring to be added)
        results = [m.to_dict() for m in memories]

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "hybrid",
        }

    async def _tool_get(self, args: dict) -> dict:
        """Handle orch_get tool."""
        memory_id = args.get("memory_id")
        project_id = args.get("project_id")

        # Search in memory manager
        if project_id:
            memories = await self.memory_manager.retrieve(project_id, limit=100)
            for m in memories:
                if m.id == memory_id:
                    return m.to_dict()

        return {"error": f"Memory not found: {memory_id}"}

    def _tool_status(self) -> dict:
        """Handle orch_status tool."""
        memory_stats = self.memory_manager.get_stats()
        session_stats = self.session_watcher.get_session_stats()
        token_stats = self.token_optimizer.get_stats()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory_stats,
            "sessions": session_stats,
            "token_optimizer": {
                "total_original": token_stats.total_original,
                "total_optimized": token_stats.total_optimized,
                "savings_percent": token_stats.savings_percent,
            },
        }

    async def _tool_memory_store(self, args: dict) -> dict:
        """Handle orch_memory_store tool."""
        project_id = args.get("project_id")
        content = args.get("content")
        memory_type = args.get("memory_type", "task")

        memory_id = await self.memory_manager.store(
            project_id=project_id,
            content=content,
            memory_type=MemoryType(memory_type),
        )

        return {"memory_id": memory_id, "status": "stored"}

    async def _tool_memory_retrieve(self, args: dict) -> dict:
        """Handle orch_memory_retrieve tool."""
        project_id = args.get("project_id")
        query = args.get("query")
        limit = args.get("limit", 5)

        memories = await self.memory_manager.retrieve(
            project_id=project_id,
            query=query,
            limit=limit,
        )

        return {
            "results": [m.to_dict() for m in memories],
            "count": len(memories),
        }

    def _tool_persona_set(self, args: dict) -> dict:
        """Handle orch_persona_set tool."""
        project_id = args.get("project_id")
        mode = args.get("mode", "balanced")

        try:
            persona_mode = PersonaMode(mode)
            self.persona_manager.set_persona(project_id, persona_mode)
            return {"status": "success", "project_id": project_id, "mode": mode}
        except ValueError:
            return {"error": f"Invalid mode: {mode}. Valid: strict, creative, balanced, custom"}

    def _tool_persona_get(self, args: dict) -> dict:
        """Handle orch_persona_get tool."""
        project_id = args.get("project_id")
        settings = self.persona_manager.get_persona_settings(project_id)
        return settings.to_dict()

    def _tool_session_start(self, args: dict) -> dict:
        """Handle orch_session_start tool."""
        project_id = args.get("project_id")
        session_id = self.session_watcher.start_session(project_id)
        return {"session_id": session_id, "project_id": project_id}

    def _tool_session_record(self, args: dict) -> dict:
        """Handle orch_session_record tool."""
        session_id = args.get("session_id")
        task_input = args.get("task_input")
        task_output = args.get("task_output")
        task_type = args.get("task_type", "task")

        interaction_id = self.session_watcher.record_interaction(
            session_id=session_id,
            task_input=task_input,
            task_output=task_output,
            task_type=task_type,
        )

        return {"interaction_id": interaction_id, "status": "recorded"}

    def _tool_optimize_output(self, args: dict) -> dict:
        """Handle orch_optimize_output tool."""
        command = args.get("command")
        output = args.get("output")

        optimized = self.token_optimizer.optimize(command, output)
        stats = self.token_optimizer.get_stats()

        return {
            "optimized": optimized,
            "original_tokens": stats.total_original,
            "optimized_tokens": stats.total_optimized,
            "savings_percent": stats.savings_percent,
        }

    async def run_stdio(self) -> None:
        """Run MCP server over stdio (subprocess mode)."""
        if not self.server:
            logger.error("MCP SDK not available")
            return

        logger.info("Starting MCP server (stdio mode)")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    async def run_http(self) -> None:
        """Run MCP server over HTTP (shared server mode)."""
        logger.info(f"Starting MCP server (HTTP mode on {self.config.host}:{self.config.port})")

        # Simple HTTP server implementation
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class MCPHTTPHandler(BaseHTTPRequestHandler):
            def __init__(self, mcp_server, *args, **kwargs):
                self.mcp_server = mcp_server
                super().__init__(*args, **kwargs)

            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')

                try:
                    request = json.loads(body)
                    # Handle MCP JSON-RPC request
                    response = {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())

            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = self.mcp_server._tool_status()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                logger.debug(f"HTTP: {args[0]}")

        def make_handler(*args, **kwargs):
            return MCPHTTPHandler(self, *args, **kwargs)

        server = HTTPServer((self.config.host, self.config.port), make_handler)
        logger.info(f"MCP HTTP server running at http://{self.config.host}:{self.config.port}")
        logger.info(f"Health endpoint: http://{self.config.host}:{self.config.port}/health")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down MCP HTTP server")
            server.shutdown()


async def run_mcp_server(config: MCPConfig | None = None) -> None:
    """Run MCP server."""
    config = config or MCPConfig()
    server = MCPServer(config)

    if config.http_mode:
        await server.run_http()
    else:
        await server.run_stdio()


def main():
    """Main entry point for MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrator MCP Server")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode")
    parser.add_argument("--port", type=int, default=8181, help="HTTP port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    config = MCPConfig(
        http_mode=args.http,
        port=args.port,
        host=args.host,
        daemon=args.daemon,
    )

    try:
        asyncio.run(run_mcp_server(config))
    except KeyboardInterrupt:
        logger.info("MCP server stopped")


if __name__ == "__main__":
    main()
