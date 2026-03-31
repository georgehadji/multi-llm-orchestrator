"""
WebSocket Manager - Handles real-time connections
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import WebSocket

logger = get_logger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""

    event: str
    data: dict[str, Any]
    timestamp: str = None

    def __init__(self, **kwargs):
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.utcnow().isoformat()
        super().__init__(**kwargs)


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        # Active connections: session_id -> set of websockets
        self._connections: dict[str, set[WebSocket]] = {}
        # Connection metadata: websocket -> session_id
        self._connection_sessions: dict[WebSocket, str] = {}
        # Event handlers: event_name -> list of callbacks
        self._event_handlers: dict[str, list[Callable]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Accept WebSocket connection and register it."""
        try:
            await websocket.accept()
            async with self._lock:
                if session_id not in self._connections:
                    self._connections[session_id] = set()
                self._connections[session_id].add(websocket)
                self._connection_sessions[websocket] = session_id
            logger.info(
                f"WebSocket connected: session={session_id}, total={self.get_connection_count(session_id)}"
            )
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        async with self._lock:
            session_id = self._connection_sessions.pop(websocket, None)
            if session_id and websocket in self._connections.get(session_id, set()):
                self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]
                logger.info(f"WebSocket disconnected: session={session_id}")

    async def send_to_session(self, session_id: str, event: str, data: dict[str, Any]):
        """Send event to all connections in a session."""
        message = WebSocketMessage(event=event, data=data).dict()
        async with self._lock:
            connections = self._connections.get(session_id, set()).copy()

        disconnected = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)

    async def broadcast(self, event: str, data: dict[str, Any], exclude_session: str | None = None):
        """Broadcast event to all connected sessions."""
        async with self._lock:
            session_ids = list(self._connections.keys())

        for session_id in session_ids:
            if session_id != exclude_session:
                await self.send_to_session(session_id, event, data)

    async def send_to_client(self, websocket: WebSocket, event: str, data: dict[str, Any]):
        """Send event to a specific WebSocket client."""
        try:
            message = WebSocketMessage(event=event, data=data).dict()
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to specific client: {e}")

    def get_connection_count(self, session_id: str) -> int:
        """Get number of connections for a session."""
        return len(self._connections.get(session_id, set()))

    def get_all_sessions(self) -> list[str]:
        """Get all active session IDs."""
        return list(self._connections.keys())

    def register_handler(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def handle_event(self, event: str, data: dict[str, Any], websocket: WebSocket):
        """Handle incoming event from client."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data, websocket)
                else:
                    handler(data, websocket)
            except Exception as e:
                logger.error(f"Event handler error for {event}: {e}")


# Global connection manager instance
_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global connection manager."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager


def reset_connection_manager():
    """Reset the connection manager (for testing)."""
    global _manager
    _manager = None
