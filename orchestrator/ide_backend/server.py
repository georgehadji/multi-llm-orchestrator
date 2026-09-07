"""
FastAPI Server - IDE Backend
"""

from __future__ import annotations

import json
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
from .auth import AuthError, authenticate, owns
from .integration.orchestrator_bridge import get_orchestrator_bridge
from .security import allowed_origins, remote_allowed, validate_bind_target
from .session_manager import get_session_manager
from .websocket.handlers import setup_websocket_handlers
from .websocket_manager import get_connection_manager

if TYPE_CHECKING:
    from pathlib import Path

#: Cap on a single inbound WebSocket frame. Generous for editor payloads,
#: small enough that one client cannot queue unbounded work.
MAX_WS_MESSAGE_BYTES = 1024 * 1024


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

    # CORS middleware. An explicit allowlist, never "*": the app sends
    # credentials, and wildcard-plus-credentials makes every site the user
    # visits a same-origin client of this API (SEC-001).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
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
        """WebSocket endpoint for real-time updates.

        Authenticated and ownership-checked *before* the accept, through the
        same `auth` module the REST routes use — a socket that authenticates
        differently from the routes is a second policy waiting to drift.
        """
        try:
            principal = authenticate(websocket.headers)
        except AuthError:
            # 1008 = policy violation. Closed before accept, so an
            # unauthenticated client never reaches the event loop.
            await websocket.close(code=1008)
            return

        session = await session_manager.get_session(session_id)
        if session is None:
            session = await session_manager.create_session(owner_id=principal.id)
            session_id = session.id
        elif not owns(principal, session.owner_id):
            await websocket.close(code=1008)
            return

        await connection_manager.connect(websocket, session_id)

        # Set up event handlers for this connection
        setup_websocket_handlers(connection_manager, session_manager)

        await connection_manager.send_to_client(websocket, "session_state", session.to_dict())

        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                    if len(raw) > MAX_WS_MESSAGE_BYTES:
                        # Bound the work a single client can queue up.
                        await connection_manager.send_to_client(
                            websocket, "error", {"code": "message_too_large"}
                        )
                        continue

                    data = json.loads(raw)
                    event = data.get("event")
                    payload = data.get("data", {})

                    await connection_manager.handle_event(event, payload, websocket)
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    # Log the detail, return a code. Raw exception text on the
                    # wire hands a caller internal paths and state.
                    logger.error(f"WebSocket message error: {e}")
                    await connection_manager.send_to_client(
                        websocket, "error", {"code": "message_failed"}
                    )
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
        # Liveness only. Session and connection counts used to be returned here
        # unauthenticated, which told an anonymous caller whether the instance
        # was in use and how heavily — reconnaissance for SEC-001. Expose those
        # through an authenticated metrics route instead.
        return {"status": "healthy"}

    logger.info("FastAPI app created successfully")
    return app


def run_ide_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    orchestrator: Any | None = None,
    frontend_path: Path | None = None,
    reload: bool = False,
    auth_required: bool = False,
):
    """Run the IDE server.

    Binds loopback by default. Exposing the IDE on another interface requires
    both ``ORCHESTRATOR_IDE_ALLOW_REMOTE=true`` and authentication; anything
    else raises `InsecureBindError` before the socket is opened (SEC-001).
    """
    import uvicorn

    validate_bind_target(
        host,
        allow_remote=remote_allowed(),
        auth_required=auth_required,
    )

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
