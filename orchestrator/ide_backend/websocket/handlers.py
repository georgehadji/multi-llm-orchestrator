"""
WebSocket Event Handlers
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

from fastapi import WebSocket

from ..log_config import get_logger
from ..websocket_manager import ConnectionManager
from ..session_manager import SessionManager, ChatMessage, FileNode, TaskProgress

logger = get_logger(__name__)


def setup_websocket_handlers(
    connection_manager: ConnectionManager,
    session_manager: SessionManager,
):
    """Set up WebSocket event handlers."""
    
    @connection_manager.register_handler("chat_message")
    async def handle_chat_message(data: Dict[str, Any], websocket: WebSocket):
        """Handle incoming chat message."""
        message_text = data.get("message", "")
        session_id = data.get("session_id")
        
        if not session_id:
            # Get session from connection
            session_id = connection_manager._connection_sessions.get(websocket)
        
        if not session_id:
            await connection_manager.send_to_client(
                websocket, "error", {"message": "No session found"}
            )
            return
        
        # Add user message
        user_message = ChatMessage(
            role="user",
            content=message_text,
            timestamp=datetime.now().strftime("%H:%M"),
        )
        await session_manager.add_message(session_id, user_message)
        
        # Broadcast updated messages
        session = await session_manager.get_session(session_id)
        if session:
            await connection_manager.send_to_session(
                session_id, "messages_update", {"messages": [m.to_dict() for m in session.messages[-10:]]}
            )
        
        # Simulate AI response (in production, this would call the orchestrator)
        await asyncio.sleep(0.5)
        
        # Add thinking message
        thinking_message = ChatMessage(
            role="assistant",
            content=None,
            thinking=True,
            steps=[
                {"label": "Analyzing request...", "done": True},
                {"label": "Planning approach...", "done": False},
            ],
        )
        await session_manager.add_message(session_id, thinking_message)
        
        # Broadcast thinking state
        await connection_manager.send_to_session(
            session_id, "messages_update",
            {"messages": [thinking_message.to_dict()]}
        )
        
        # Simulate processing
        await asyncio.sleep(1.5)
        
        # Update thinking message with completion
        completed_message = ChatMessage(
            role="assistant",
            content=f"I've analyzed your request: \"{message_text[:100]}...\"\n\nI'll start working on this now. I'll break it down into tasks and begin implementation.",
            thinking=False,
            steps=[
                {"label": "Analyzing request...", "done": True},
                {"label": "Planning approach...", "done": True},
                {"label": "Creating tasks...", "done": True},
            ],
        )
        # Replace last message
        session = await session_manager.get_session(session_id)
        if session and session.messages:
            session.messages[-1] = completed_message
            await session_manager._save_session(session)
            
            await connection_manager.send_to_session(
                session_id, "messages_update",
                {"messages": [completed_message.to_dict()]}
            )
    
    @connection_manager.register_handler("session_update")
    async def handle_session_update(data: Dict[str, Any], websocket: WebSocket):
        """Handle session settings update."""
        session_id = data.get("session_id")
        if not session_id:
            session_id = connection_manager._connection_sessions.get(websocket)
        
        if not session_id:
            await connection_manager.send_to_client(
                websocket, "error", {"message": "No session found"}
            )
            return
        
        updates = {}
        if "mode" in data:
            updates["mode"] = data["mode"]
        if "autonomy" in data:
            updates["autonomy"] = data["autonomy"]
        if "model" in data:
            updates["model"] = data["model"]
        if "budget" in data:
            updates["budget"] = float(data["budget"])
        
        if updates:
            await session_manager.update_session(session_id, **updates)
            session = await session_manager.get_session(session_id)
            if session:
                await connection_manager.send_to_session(
                    session_id, "session_state", session.to_dict()
                )
    
    @connection_manager.register_handler("file_request")
    async def handle_file_request(data: Dict[str, Any], websocket: WebSocket):
        """Handle file content request."""
        session_id = data.get("session_id")
        file_path = data.get("path")
        
        if not session_id or not file_path:
            return
        
        session = await session_manager.get_session(session_id)
        if not session:
            return
        
        # Find file in tree
        def find_file(nodes, parts, depth=0):
            for node in nodes:
                if node.name == parts[depth]:
                    if depth == len(parts) - 1:
                        return node if node.type == "file" else None
                    if node.type == "folder":
                        return find_file(node.children, parts, depth + 1)
            return None
        
        path_parts = file_path.split("/")
        file_node = find_file(session.files, path_parts)
        
        if file_node:
            await connection_manager.send_to_client(
                websocket, "file_content",
                {"path": file_path, "content": file_node.content or ""}
            )
    
    @connection_manager.register_handler("file_update")
    async def handle_file_update(data: Dict[str, Any], websocket: WebSocket):
        """Handle file content update."""
        session_id = data.get("session_id")
        file_path = data.get("path")
        content = data.get("content")
        
        if not session_id or not file_path or content is None:
            return
        
        await session_manager.update_file(session_id, file_path, content)
        
        # Broadcast update
        await connection_manager.send_to_session(
            session_id, "file_updated", {"path": file_path}
        )
    
    @connection_manager.register_handler("terminal_command")
    async def handle_terminal_command(data: Dict[str, Any], websocket: WebSocket):
        """Handle terminal command execution."""
        session_id = data.get("session_id")
        command = data.get("command")
        
        if not session_id or not command:
            return
        
        # Echo command
        await session_manager.add_terminal_line(session_id, "cmd", f"$ {command}")
        
        # Simulate command output
        await asyncio.sleep(0.3)
        await session_manager.add_terminal_line(
            session_id, "out", f"Executing: {command}"
        )
        
        # Broadcast terminal update
        session = await session_manager.get_session(session_id)
        if session:
            await connection_manager.send_to_session(
                session_id, "terminal_update",
                {"lines": session.terminal_lines[-20:]}
            )
    
    @connection_manager.register_handler("ping")
    async def handle_ping(data: Dict[str, Any], websocket: WebSocket):
        """Handle ping for connection testing."""
        await connection_manager.send_to_client(
            websocket, "pong", {"timestamp": datetime.now().isoformat()}
        )
    
    logger.info("WebSocket handlers registered")
