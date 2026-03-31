"""
FastAPI Server - IDE Backend
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Simple logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("ide_backend")

from typing import TYPE_CHECKING

from .api.routes import router as api_router
from .integration.orchestrator_bridge import get_orchestrator_bridge
from .session_manager import get_session_manager
from .websocket.handlers import setup_websocket_handlers
from .websocket_manager import get_connection_manager

if TYPE_CHECKING:
    from pathlib import Path


def create_app(
    orchestrator: Any | None = None,
    frontend_path: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AI Orchestrator IDE",
        description="Real-time IDE dashboard for AI Orchestrator",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize managers
    connection_manager = get_connection_manager()
    session_manager = get_session_manager()

    # Initialize orchestrator bridge if orchestrator provided
    if orchestrator:
        get_orchestrator_bridge(orchestrator, connection_manager)
        logger.info("Orchestrator bridge initialized")

    # Include API routes
    app.include_router(api_router, prefix="/api")

    # WebSocket endpoint
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time updates."""
        await connection_manager.connect(websocket, session_id)

        # Set up event handlers for this connection
        setup_websocket_handlers(connection_manager, session_manager)

        # Send initial state
        session = await session_manager.get_session(session_id)
        if session:
            await connection_manager.send_to_client(websocket, "session_state", session.to_dict())
        else:
            # Create default session
            session = await session_manager.create_session()
            await connection_manager.send_to_client(websocket, "session_state", session.to_dict())

        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    event = data.get("event")
                    payload = data.get("data", {})

                    await connection_manager.handle_event(event, payload, websocket)
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"WebSocket message error: {e}")
                    await connection_manager.send_to_client(websocket, "error", {"message": str(e)})
        except WebSocketDisconnect:
            await connection_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await connection_manager.disconnect(websocket)

    # Serve frontend if path provided
    if frontend_path and frontend_path.exists():
        app.mount("/ide", StaticFiles(directory=str(frontend_path), html=True), name="ide")

        @app.get("/")
        async def root_redirect():
            return FileResponse(str(frontend_path / "index.html"))

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "sessions": len(session_manager._sessions),
            "connections": sum(len(conns) for conns in connection_manager._connections.values()),
        }

    logger.info("FastAPI app created successfully")
    return app


def run_ide_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    orchestrator: Any | None = None,
    frontend_path: Path | None = None,
    reload: bool = False,
):
    """Run the IDE server."""
    import uvicorn

    app = create_app(orchestrator=orchestrator, frontend_path=frontend_path)

    logger.info(f"Starting IDE server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    # Run with reload for development
    run_ide_server(reload=True)
