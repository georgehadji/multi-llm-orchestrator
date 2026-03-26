"""
IDE Backend v1.0 - Real-time Dashboard for AI Orchestrator
===========================================================
Provides FastAPI server with WebSocket support for the React IDE dashboard.
"""

__version__ = "1.0.0"

from .server import create_app, run_ide_server
from .session_manager import SessionManager, SessionState
from .websocket_manager import ConnectionManager, get_connection_manager

__all__ = [
    "create_app",
    "run_ide_server",
    "SessionManager",
    "SessionState",
    "ConnectionManager",
    "get_connection_manager",
]
