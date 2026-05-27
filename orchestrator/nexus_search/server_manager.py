"""
Nexus Search — Server Manager
==============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Automatic server management for Nexus Search.
Starts SearXNG server automatically when AI Orchestrator starts.

Features:
- Docker Compose auto-start
- Health check monitoring
- Graceful shutdown
- Automatic restart on failure
- Status reporting

Usage:
    from orchestrator.nexus_search import NexusServerManager

    manager = NexusServerManager()
    await manager.start()  # Auto-starts with orchestrator
    await manager.stop()   # Graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger("orchestrator.nexus_server")


class NexusServerManager:
    """
    Automatic server manager for Nexus Search.

    Manages SearXNG server lifecycle:
    - Auto-start on orchestrator initialization
    - Health check monitoring
    - Graceful shutdown
    - Automatic restart on failure

    Usage:
        manager = NexusServerManager()
        await manager.start()
    """

    def __init__(
        self,
        compose_file: str | None = None,
        port: int = 8080,
        auto_start: bool = True,
        health_check_interval: int = 30,
        max_restarts: int = 3,
    ):
        """
        Initialize server manager.

        Args:
            compose_file: Path to docker-compose.yml (default: nexus-search.docker-compose.yml)
            port: Nexus Search port (default: 8080)
            auto_start: Auto-start server on init (default: True)
            health_check_interval: Health check interval in seconds (default: 30)
            max_restarts: Maximum restart attempts (default: 3)
        """
        # Find compose file
        if compose_file:
            self.compose_file = Path(compose_file)
        else:
            # Search for compose file in project root
            self.compose_file = self._find_compose_file()

        self.port = port
        self.auto_start = auto_start
        self.health_check_interval = health_check_interval
        self.max_restarts = max_restarts

        # State tracking
        self._server_started = False
        self._restart_count = 0
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Docker Compose command
        self._docker_compose_cmd = self._find_docker_compose()

    def _find_compose_file(self) -> Path:
        """Find docker-compose file in project root."""
        possible_paths = [
            Path(__file__).parent.parent / "nexus-search.docker-compose.yml",
            Path(__file__).parent.parent / "docker-compose.nexus.yml",
            Path.cwd() / "nexus-search.docker-compose.yml",
            Path.cwd() / "docker-compose.nexus.yml",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found docker-compose file: {path}")
                return path

        # Create default compose file if not found
        logger.warning("docker-compose file not found, creating default...")
        return self._create_default_compose_file()

    def _create_default_compose_file(self) -> Path:
        """Create default docker-compose file."""
        compose_path = Path(__file__).parent.parent / "nexus-search.docker-compose.yml"

        compose_content = """
version: '3.8'
services:
  nexus-search:
    image: searxng/searxng:latest
    container_name: nexus-search
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
      - SEARXNG_PORT=8080
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

        compose_path.write_text(compose_content.strip())
        logger.info(f"Created default docker-compose file: {compose_path}")
        return compose_path

    def _find_docker_compose(self) -> str:
        """Find docker-compose command."""
        # Try docker compose (v2)
        try:
            subprocess.run(["docker", "compose", "version"], capture_output=True, check=True)
            logger.info("Using 'docker compose' (v2)")
            return "docker compose"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try docker-compose (v1)
        try:
            subprocess.run(["docker-compose", "version"], capture_output=True, check=True)
            logger.info("Using 'docker-compose' (v1)")
            return "docker-compose"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        logger.error("Neither 'docker compose' nor 'docker-compose' found")
        return ""

    async def start(self) -> bool:
        """
        Start Nexus Search server.

        Returns:
            True if server started successfully, False otherwise
        """
        if not self._docker_compose_cmd:
            logger.error("Docker Compose not available, cannot start server")
            return False

        if self._server_started:
            logger.info("Server already started")
            return True

        logger.info("Starting Nexus Search server...")

        try:
            # Start with docker-compose
            cmd = f"{self._docker_compose_cmd} -f {self.compose_file} up -d"
            logger.info(f"Running: {cmd}")

            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to start server: {stderr.decode()}")
                return False

            logger.info("Docker compose started successfully")

            # Wait for server to be healthy
            if await self._wait_for_healthy():
                self._server_started = True
                logger.info(f"Nexus Search server started on port {self.port}")

                # Start health check monitoring
                if self.health_check_interval > 0:
                    self._health_check_task = asyncio.create_task(self._health_check_loop())

                return True
            else:
                logger.error("Server failed health check")
                await self.stop()
                return False

        except Exception as e:
            logger.error(f"Error starting server: {e}")
            return False

    async def _wait_for_healthy(self, timeout: int = 60) -> bool:
        """
        Wait for server to become healthy.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if healthy, False if timeout
        """
        logger.info(f"Waiting for server to become healthy (timeout: {timeout}s)...")

        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if await self.health_check():
                logger.info("Server is healthy")
                return True

            await asyncio.sleep(5)

        logger.error(f"Server health check timed out after {timeout}s")
        return False

    async def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            import aiohttp

            url = f"http://localhost:{self.port}/healthz"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200

        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def _health_check_loop(self):
        """Background health check loop."""
        logger.info(f"Starting health check loop (interval: {self.health_check_interval}s)")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)

                if not await self.health_check():
                    logger.warning("Health check failed, attempting restart...")
                    await self._restart()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _restart(self):
        """Restart server with backoff."""
        if self._restart_count >= self.max_restarts:
            logger.error(f"Max restarts ({self.max_restarts}) exceeded, giving up")
            return

        self._restart_count += 1
        backoff = min(30, 2**self._restart_count)  # Exponential backoff, max 30s

        logger.info(
            f"Restarting server (attempt {self._restart_count}/{self.max_restarts}, backoff: {backoff}s)"
        )

        await asyncio.sleep(backoff)

        if await self.start():
            logger.info("Server restarted successfully")
            self._restart_count = 0
        else:
            logger.error("Server restart failed")

    async def stop(self):
        """Stop Nexus Search server."""
        if not self._server_started:
            return

        logger.info("Stopping Nexus Search server...")

        # Stop health check loop
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Signal shutdown
        self._shutdown_event.set()

        # Stop with docker-compose
        if self._docker_compose_cmd:
            try:
                cmd = f"{self._docker_compose_cmd} -f {self.compose_file} down"
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()
                logger.info("Server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

        self._server_started = False
        self._restart_count = 0

    def get_status(self) -> dict:
        """Get server status."""
        return {
            "started": self._server_started,
            "port": self.port,
            "compose_file": str(self.compose_file),
            "restart_count": self._restart_count,
            "max_restarts": self.max_restarts,
            "health_check_interval": self.health_check_interval,
            "docker_compose_cmd": self._docker_compose_cmd,
        }


# Global manager instance
_manager: Optional[NexusServerManager] = None


def get_server_manager(auto_start: bool = True) -> NexusServerManager:
    """
    Get or create NexusServerManager instance.

    Args:
        auto_start: Auto-start server on creation

    Returns:
        NexusServerManager instance
    """
    global _manager
    if _manager is None:
        _manager = NexusServerManager(auto_start=auto_start)
    return _manager


async def start_server() -> bool:
    """Start Nexus Search server."""
    manager = get_server_manager(auto_start=True)
    return await manager.start()


async def stop_server():
    """Stop Nexus Search server."""
    if _manager:
        await _manager.stop()


async def check_server_health() -> bool:
    """Check server health."""
    if _manager:
        return await _manager.health_check()
    return False
