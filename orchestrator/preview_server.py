"""
Preview Server — Live preview with hot reload
==============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Live preview server for generated applications with hot reload
on file changes. Provides immediate visual feedback.

Features:
- Live preview server
- Hot reload on file changes
- WebSocket for real-time updates
- Screenshot capture
- Multi-project support

USAGE:
    from orchestrator.preview_server import PreviewServer

    server = PreviewServer()

    # Start preview
    url = await server.start("./my-app", port=3000)
    print(f"Preview at: {url}")

    # Hot reload on changes
    await server.hot_reload([change1, change2])

    # Take screenshot
    screenshot = await server.take_screenshot()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.preview_server")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────


@dataclass
class PreviewConfig:
    """Preview server configuration."""

    port: int = 3000
    host: str = "localhost"
    hot_reload: bool = True
    open_browser: bool = False
    build_command: str | None = None
    watch_patterns: list[str] = field(
        default_factory=lambda: [
            "*.tsx",
            "*.ts",
            "*.jsx",
            "*.js",
            "*.css",
            "*.html",
        ]
    )
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "node_modules/",
            ".git/",
            "*.log",
            "dist/",
            "build/",
        ]
    )


@dataclass
class PreviewSession:
    """Active preview session."""

    project_path: str
    url: str
    port: int
    pid: int | None = None
    connected_clients: int = 0
    last_reload: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_path": self.project_path,
            "url": self.url,
            "port": self.port,
            "pid": self.pid,
            "connected_clients": self.connected_clients,
            "last_reload": self.last_reload,
            "errors": self.errors,
        }


# ─────────────────────────────────────────────
# Preview Server
# ─────────────────────────────────────────────


class PreviewServer:
    """
    Live preview server with hot reload.

    Provides instant visual feedback on generated code
    with automatic browser refresh on changes.
    """

    _instance: PreviewServer | None = None

    def __new__(cls) -> PreviewServer:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize preview server."""
        # Only initialize once
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._sessions: dict[str, PreviewSession] = {}
        self._watch_tasks: dict[str, asyncio.Task] = {}
        self._ws_connections: set[Any] = set()

        self._total_previews = 0
        self._total_reloads = 0

        self._initialized = True

        logger.info("PreviewServer initialized")

    async def start(
        self,
        project_path: str,
        config: PreviewConfig | None = None,
    ) -> str:
        """
        Start preview server.

        Args:
            project_path: Path to project
            config: Preview configuration

        Returns:
            Preview URL
        """
        config = config or PreviewConfig()

        # Check if project exists
        path = Path(project_path)
        if not path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Check for existing session
        if project_path in self._sessions:
            session = self._sessions[project_path]
            logger.info(f"Preview already running at {session.url}")
            return session.url

        # Start development server
        pid = await self._start_dev_server(path, config)

        # Create session
        url = f"http://{config.host}:{config.port}"
        session = PreviewSession(
            project_path=project_path,
            url=url,
            port=config.port,
            pid=pid,
        )

        self._sessions[project_path] = session
        self._total_previews += 1

        # Start file watcher
        if config.hot_reload:
            await self._start_file_watcher(project_path, config)

        # Open browser if requested
        if config.open_browser:
            await self._open_browser(url)

        logger.info(f"Preview started at {url}")
        return url

    async def stop(self, project_path: str) -> bool:
        """
        Stop preview server.

        Args:
            project_path: Project path

        Returns:
            True if stopped
        """
        if project_path not in self._sessions:
            return False

        session = self._sessions[project_path]

        # Stop file watcher
        if project_path in self._watch_tasks:
            self._watch_tasks[project_path].cancel()
            del self._watch_tasks[project_path]

        # Stop dev server
        if session.pid:
            try:
                os.kill(session.pid, 9)
            except ProcessLookupError:
                pass

        del self._sessions[project_path]

        logger.info(f"Preview stopped for {project_path}")
        return True

    async def hot_reload(
        self,
        project_path: str,
        changes: list[Any] | None = None,
    ) -> bool:
        """
        Trigger hot reload.

        Args:
            project_path: Project path
            changes: Optional list of changes

        Returns:
            True if reload triggered
        """
        if project_path not in self._sessions:
            return False

        session = self._sessions[project_path]
        session.last_reload = asyncio.get_event_loop().time()
        self._total_reloads += 1

        # Notify WebSocket clients
        await self._notify_clients(
            "reload",
            {
                "project": project_path,
                "changes": [str(c) for c in changes] if changes else [],
            },
        )

        logger.debug(f"Hot reload triggered for {project_path}")
        return True

    async def take_screenshot(
        self,
        project_path: str,
    ) -> bytes | None:
        """
        Capture preview screenshot.

        Args:
            project_path: Project path

        Returns:
            Screenshot bytes or None
        """
        if project_path not in self._sessions:
            return None

        self._sessions[project_path]

        try:
            # Use playwright or similar for screenshot
            # For now, return None
            logger.warning("Screenshot not implemented")
            return None

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def get_session(self, project_path: str) -> PreviewSession | None:
        """Get preview session."""
        return self._sessions.get(project_path)

    def get_all_sessions(self) -> list[PreviewSession]:
        """Get all active sessions."""
        return list(self._sessions.values())

    async def _start_dev_server(
        self,
        path: Path,
        config: PreviewConfig,
    ) -> int | None:
        """Start development server."""
        # Detect framework and start appropriate server
        if (path / "package.json").exists():
            return await self._start_node_server(path, config)
        elif (path / "requirements.txt").exists():
            return await self._start_python_server(path, config)
        else:
            return await self._start_static_server(path, config)

    async def _start_node_server(
        self,
        path: Path,
        config: PreviewConfig,
    ) -> int | None:
        """Start Node.js dev server."""
        # Check for Vite
        if (path / "vite.config.js").exists() or (path / "vite.config.ts").exists():
            cmd = ["npx", "vite", "--port", str(config.port), "--host", config.host]
        # Check for Next.js
        elif (path / "next.config.js").exists():
            cmd = ["npm", "run", "dev", "--", "-p", str(config.port)]
        # Check for Create React App
        elif (path / "package.json").exists():
            cmd = ["npm", "start"]
        else:
            cmd = ["npx", "serve", "-p", str(config.port)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(f"Started Node server: {' '.join(cmd)}")
            return process.pid

        except Exception as e:
            logger.error(f"Failed to start Node server: {e}")
            return None

    async def _start_python_server(
        self,
        path: Path,
        config: PreviewConfig,
    ) -> int | None:
        """Start Python dev server."""
        # Check for Flask
        if (path / "app.py").exists() or (path / "main.py").exists():
            cmd = ["python", "-m", "flask", "run", "--port", str(config.port)]
        # Check for Streamlit
        elif any(f.name.endswith(".py") for f in path.iterdir()):
            cmd = ["streamlit", "run", str(path / "*.py"), "--server.port", str(config.port)]
        else:
            cmd = ["python", "-m", "http.server", str(config.port)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(f"Started Python server: {' '.join(cmd)}")
            return process.pid

        except Exception as e:
            logger.error(f"Failed to start Python server: {e}")
            return None

    async def _start_static_server(
        self,
        path: Path,
        config: PreviewConfig,
    ) -> int | None:
        """Start static file server."""
        cmd = ["python", "-m", "http.server", str(config.port)]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(f"Started static server: {' '.join(cmd)}")
            return process.pid

        except Exception as e:
            logger.error(f"Failed to start static server: {e}")
            return None

    async def _start_file_watcher(
        self,
        project_path: str,
        config: PreviewConfig,
    ) -> None:
        """Start file watcher for hot reload."""

        async def watch_loop():
            from pathlib import Path

            path = Path(project_path)
            last_modified: dict[str, float] = {}

            while True:
                try:
                    await asyncio.sleep(1.0)  # Check every second

                    # Scan files
                    for pattern in config.watch_patterns:
                        for file in path.rglob(pattern):
                            # Check ignore patterns
                            if any(ignore in str(file) for ignore in config.ignore_patterns):
                                continue

                            # Check modification time
                            try:
                                mtime = file.stat().st_mtime
                                if file not in last_modified:
                                    last_modified[file] = mtime
                                elif mtime > last_modified[file]:
                                    last_modified[file] = mtime
                                    logger.debug(f"File changed: {file}")

                                    # Trigger reload
                                    await self.hot_reload(project_path, [file])

                            except FileNotFoundError:
                                pass

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"File watcher error: {e}")

        task = asyncio.create_task(watch_loop())
        self._watch_tasks[project_path] = task

    async def _open_browser(self, url: str) -> None:
        """Open browser to preview URL."""
        import webbrowser

        webbrowser.open(url)

    async def _notify_clients(self, event: str, data: dict) -> None:
        """Notify WebSocket clients of event."""
        message = {"event": event, "data": data}

        # Send to all connected clients
        for ws in list(self._ws_connections):
            try:
                await ws.send_json(message)
            except Exception:
                self._ws_connections.discard(ws)

    def get_stats(self) -> dict[str, Any]:
        """Get preview server statistics."""
        return {
            "total_previews": self._total_previews,
            "total_reloads": self._total_reloads,
            "active_sessions": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_server: PreviewServer | None = None


def get_preview_server() -> PreviewServer:
    """Get or create default preview server."""
    global _default_server
    if _default_server is None:
        _default_server = PreviewServer()
    return _default_server


def reset_preview_server() -> None:
    """Reset default server (for testing)."""
    global _default_server
    _default_server = None


async def start_preview(
    project_path: str,
    port: int = 3000,
    hot_reload: bool = True,
) -> str:
    """
    Start preview with default settings.

    Args:
        project_path: Project path
        port: Server port
        hot_reload: Enable hot reload

    Returns:
        Preview URL
    """
    server = get_preview_server()
    config = PreviewConfig(
        port=port,
        hot_reload=hot_reload,
    )
    return await server.start(project_path, config)


async def stop_preview(project_path: str) -> bool:
    """Stop preview."""
    server = get_preview_server()
    return await server.stop(project_path)
