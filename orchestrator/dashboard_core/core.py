"""
Unified Dashboard Core — Plugin-Based Dashboard System
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Single dashboard core with pluggable view components.
Eliminates 6+ redundant server implementations.

This module consolidates:
- dashboard.py
- dashboard_antd.py
- dashboard_enhanced.py
- dashboard_live.py
- dashboard_mission_control.py
- dashboard_optimized.py
- dashboard_real.py

Into a single core with pluggable views.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable, Awaitable, Dict, List

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = type("FastAPI", (), {})
    WebSocket = type("WebSocket", (), {})

from .events_unified import get_event_bus, DomainEvent, EventType
from .models import ProjectState, Model, TaskStatus

logger = logging.getLogger("orchestrator.dashboard_core")


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard View Interface (Plugin Base)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ViewContext:
    """Context passed to views for rendering."""
    project_id: str = ""
    project_state: Optional[ProjectState] = None
    active_tasks: List[Dict[str, Any]] = field(default_factory=list)
    model_status: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    architecture_info: Optional[Dict[str, Any]] = None
    budget: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_state": self.project_state.to_dict() if self.project_state else None,
            "active_tasks": self.active_tasks,
            "model_status": self.model_status,
            "metrics": self.metrics,
            "architecture_info": self.architecture_info,
            "budget": self.budget,
            "events": self.events[-100:],  # Last 100 events
        }


class DashboardView(ABC):
    """
    Abstract base class for dashboard views.
    
    Each view (MissionControl, AntDesign, etc.) implements this interface.
    Views are plugins - they can be registered/unregistered dynamically.
    """
    
    name: str = "base"
    display_name: str = "Base View"
    version: str = "1.0.0"
    
    @abstractmethod
    async def render(self, context: ViewContext) -> str:
        """Render the view as HTML string."""
        pass
    
    @abstractmethod
    def get_assets(self) -> Dict[str, List[str]]:
        """Return CSS and JS assets required by this view."""
        return {"css": [], "js": []}
    
    def handle_event(self, event: DomainEvent) -> Optional[Dict[str, Any]]:
        """
        Process an event and return optional WebSocket message.
        Override to provide custom event handling.
        """
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# View Registry
# ═══════════════════════════════════════════════════════════════════════════════

class ViewRegistry:
    """Registry for dashboard views."""
    
    def __init__(self):
        self._views: Dict[str, DashboardView] = {}
        self._default_view: str = "mission-control"
    
    def register(self, view: DashboardView, make_default: bool = False) -> None:
        """Register a view."""
        self._views[view.name] = view
        logger.info(f"Registered dashboard view: {view.name}")
        if make_default or len(self._views) == 1:
            self._default_view = view.name
    
    def unregister(self, name: str) -> None:
        """Unregister a view."""
        if name in self._views:
            del self._views[name]
            logger.info(f"Unregistered dashboard view: {name}")
    
    def get(self, name: str) -> Optional[DashboardView]:
        """Get a view by name."""
        return self._views.get(name)
    
    def list_views(self) -> List[Dict[str, str]]:
        """List all registered views."""
        return [
            {
                "name": v.name,
                "display_name": v.display_name,
                "version": v.version,
                "is_default": v.name == self._default_view,
            }
            for v in self._views.values()
        ]
    
    def get_default(self) -> Optional[DashboardView]:
        """Get the default view."""
        return self._views.get(self._default_view)


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Core
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardCore:
    """
    Unified dashboard core with plugin-based views.
    
    Single server handles all views - eliminates 6+ redundant implementations.
    """
    
    _instance: Optional[DashboardCore] = None
    _lock = asyncio.Lock()
    
    def __init__(self, event_bus=None):
        self.registry = ViewRegistry()
        self.event_bus = event_bus or get_event_bus()
        self.context = ViewContext()
        self._websocket_clients: List[WebSocket] = []
        self._event_task: Optional[asyncio.Task] = None
        self._app: Optional[FastAPI] = None
        
    @classmethod
    async def get_instance(cls) -> DashboardCore:
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register_view(self, view: DashboardView, make_default: bool = False) -> None:
        """Register a view plugin."""
        self.registry.register(view, make_default)
    
    async def render(self, view_name: Optional[str] = None, 
                     context: Optional[ViewContext] = None) -> str:
        """
        Render a specific view.
        
        Args:
            view_name: Name of view to render (uses default if None)
            context: Optional context override
        """
        ctx = context or self.context
        view = self.registry.get(view_name) if view_name else self.registry.get_default()
        
        if view is None:
            return self._render_error(f"View not found: {view_name}")
        
        try:
            return await view.render(ctx)
        except Exception as e:
            logger.exception(f"Error rendering view {view.name}")
            return self._render_error(str(e))
    
    def _render_error(self, message: str) -> str:
        """Render error page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Dashboard Error</title></head>
        <body>
            <h1>Dashboard Error</h1>
            <p>{message}</p>
            <p>Available views: {self.registry.list_views()}</p>
        </body>
        </html>
        """
    
    async def start_event_stream(self) -> None:
        """Start streaming events to WebSocket clients."""
        if self._event_task is not None:
            return
        
        async def _stream():
            async for event in self.event_bus.subscribe():
                await self._broadcast_event(event)
        
        self._event_task = asyncio.create_task(_stream())
    
    async def stop_event_stream(self) -> None:
        """Stop the event stream."""
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
    
    async def _broadcast_event(self, event: DomainEvent) -> None:
        """Broadcast event to all connected WebSocket clients."""
        if not self._websocket_clients:
            return
        
        message = {
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.to_dict(),
        }
        
        # Let views transform the event
        for view in self.registry._views.values():
            transformed = view.handle_event(event)
            if transformed:
                message["view_data"] = transformed
                break
        
        disconnected = []
        for ws in self._websocket_clients:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self._websocket_clients.remove(ws)
    
    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection."""
        await websocket.accept()
        self._websocket_clients.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self._websocket_clients)}")
        
        try:
            while True:
                # Keep connection alive, handle ping/pong
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "subscribe":
                    await websocket.send_json({
                        "type": "connected",
                        "views": self.registry.list_views(),
                    })
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in self._websocket_clients:
                self._websocket_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self._websocket_clients)}")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application with all routes."""
        if not HAS_FASTAPI:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
        
        if self._app is not None:
            return self._app
        
        app = FastAPI(title="Orchestrator Dashboard")
        
        @app.get("/")
        async def index(view: str = ""):
            """Main dashboard page."""
            html = await self.render(view if view else None)
            return HTMLResponse(content=html)
        
        @app.get("/api/views")
        async def list_views():
            """List available views."""
            return {"views": self.registry.list_views()}
        
        @app.get("/api/context")
        async def get_context():
            """Get current context."""
            return self.context.to_dict()
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)
        
        self._app = app
        return app
    
    async def run(self, host: str = "0.0.0.0", port: int = 8888) -> None:
        """Run the dashboard server."""
        import uvicorn
        
        app = self.create_app()
        await self.start_event_stream()
        
        logger.info(f"Starting unified dashboard on http://{host}:{port}")
        logger.info(f"Available views: {[v['name'] for v in self.registry.list_views()]}")
        
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

async def get_dashboard_core() -> DashboardCore:
    """Get the global dashboard core instance."""
    return await DashboardCore.get_instance()


def run_dashboard(view: str = "", host: str = "0.0.0.0", port: int = 8888) -> None:
    """
    Run the unified dashboard.
    
    Usage:
        from orchestrator.dashboard_core_core import run_dashboard
        run_dashboard(view="mission-control")
    """
    async def _run():
        core = await get_dashboard_core()
        await core.run(host, port)
    
    asyncio.run(_run())
