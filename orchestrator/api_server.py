"""
APIServer — REST API server
==========================
Module for providing a REST API server for the orchestrator.

Pattern: Facade (Driving Adapter in hexagonal architecture)
Async: Yes — all I/O is async via aiohttp
Layer: L4 Supervisor / Driving Adapter

Imports from infrastructure are prohibited by the root-modules-no-infra
import-linter contract. All infrastructure dependencies are injected via
the constructor (Orchestrator, Supervisor) or wrapped through domain ports.

SECURITY FIXES:
- Configurable CORS allowlist (not wildcard)
- Rate limiting on all endpoints
- Secure API key authentication
- Request size limits

Usage:
    from orchestrator.api_server import APIServer
    from orchestrator.engine import Orchestrator
    server = APIServer(port=8000, orchestrator=Orchestrator(...), cors_origins=["https://trusted-domain.com"])
    await server.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

    from orchestrator.engine import Orchestrator
    from orchestrator.supervisor.service import Supervisor
else:
    import aiohttp.web as web

from .crosscutting.config import flags

logger = logging.getLogger("orchestrator.api_server")


# ─────────────────────────────────────────────
# Rate Limiter (Token Bucket)
# ─────────────────────────────────────────────


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API endpoints.

    SECURITY: Prevents DoS and brute force attacks.
    """

    def __init__(
        self,
        rate: int = 100,  # requests per window
        window_seconds: int = 60,
    ):
        self.rate = rate
        self.window_seconds = window_seconds
        self._buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"tokens": rate, "last_update": time.time()}
        )
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.

        Args:
            client_id: Client identifier (IP, API key, etc.)

        Returns:
            True if allowed
        """
        async with self._lock:
            bucket = self._buckets[client_id]
            now = time.time()

            # Refill tokens
            elapsed = now - bucket["last_update"]
            tokens_to_add = (elapsed / self.window_seconds) * self.rate
            bucket["tokens"] = min(self.rate, bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = now

            # Check if allowed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            else:
                return False

    def get_retry_after(self, client_id: str) -> float:
        """Get seconds until next request is allowed."""
        bucket = self._buckets.get(client_id)
        if not bucket:
            return 0.0

        if bucket["tokens"] >= 1:
            return 0.0

        tokens_needed = 1 - bucket["tokens"]
        return (tokens_needed / self.rate) * self.window_seconds


# ─────────────────────────────────────────────
# API Server
# ─────────────────────────────────────────────


class APIServer:
    """REST API server for the orchestrator."""

    def __init__(
        self,
        port: int = 8000,
        host: str = "localhost",
        cors_origins: list[str] | None = None,
        auth_required: bool = True,
        rate_limit: int = 100,
        rate_window: int = 60,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        supervisor: Supervisor | None = None,
        orchestrator: Orchestrator | None = None,
    ):
        """
        Initialize the API server.

        SECURITY:
        - cors_origins: Configurable allowlist (not wildcard)
        - rate_limit: Rate limiting to prevent DoS
        - max_request_size: Limit request size
        """
        self.port = port
        self.host = host
        self.cors_origins = cors_origins or []  # Empty = same origin only
        self.auth_required = auth_required
        if not auth_required:
            logger.warning(
                "APIServer: auth_required=False — all endpoints are unauthenticated. "
                "Only use this in development."
            )
        self.max_request_size = max_request_size
        self.supervisor = supervisor
        self._orchestrator = orchestrator

        self.app = web.Application(client_max_size=self.max_request_size)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # API keys: hashed_key -> {user_id, permissions, created_at}
        self.api_keys: dict[str, dict[str, Any]] = {}

        # Rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            rate=rate_limit,
            window_seconds=rate_window,
        )

        # Request stats
        self.request_stats: dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "start_time": datetime.now(),
        }

        # Active project tracking
        self._active_projects: dict[str, asyncio.Task] = {}
        self._active_project_state: dict[str, dict[str, Any]] = {}
        self._active_project_lock: asyncio.Lock = asyncio.Lock()

        # Register routes
        self._setup_routes()

        # Enable CORS with allowlist (SECURITY FIX)
        if self.cors_origins:
            self._enable_cors()

        # Add rate limiting middleware
        self.app.middlewares.append(self._rate_limit_middleware)

        # Add request size limit
        self.app.middlewares.append(self._request_size_middleware)

    # ─────────────────────────────────────────────
    # Route Setup
    # ─────────────────────────────────────────────

    def _setup_routes(self):
        """Setup API routes."""
        # Health
        self.app.router.add_get("/", self.health_check)
        self.app.router.add_get("/health", self.health_check)

        # Execute (legacy backward-compat auto-detect)
        self.app.router.add_post("/execute", self.execute_task)

        # Execute endpoints
        self.app.router.add_post("/execute/project", self.execute_project)
        self.app.router.add_post("/execute/tasks", self.execute_tasks)
        self.app.router.add_post("/execute/from-speckit", self.execute_from_speckit)

        # Project status & streaming
        self.app.router.add_get("/projects/{project_id}", self.get_project_status)
        self.app.router.add_get("/projects/{project_id}/stream", self.project_stream)
        self.app.router.add_delete("/projects/{project_id}", self.cancel_project)

        # Backward compat status
        self.app.router.add_get("/status/{task_id}", self.get_task_status)

        # Info
        self.app.router.add_get("/models", self.list_models)
        self.app.router.add_get("/stats", self.get_stats)

        # Auth
        self.app.router.add_post("/register_key", self.register_api_key)

        # Supervisor
        if self.supervisor is not None:
            self.app.router.add_post("/supervisor/directive", self.supervisor_directive)
            self.app.router.add_get("/supervisor/sessions", self.list_supervisor_sessions)
            self.app.router.add_get("/supervisor/sessions/{id}", self.get_supervisor_session)
            self.app.router.add_get(
                "/supervisor/sessions/{id}/lessons", self.get_supervisor_lessons
            )

    def _enable_cors(self):
        """
        Enable CORS with configurable allowlist.

        SECURITY FIX: No longer uses wildcard '*' which allows any origin.
        Now uses configurable allowlist for trusted domains only.
        """

        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                response = await handler(request)

                # Check origin against allowlist
                origin = request.headers.get("Origin", "")

                if origin in self.cors_origins:
                    response.headers["Access-Control-Allow-Origin"] = origin
                elif "*" in self.cors_origins:
                    # Only allow wildcard if explicitly configured
                    response.headers["Access-Control-Allow-Origin"] = "*"

                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours

                return response

            return middleware_handler

        self.app.middlewares.append(cors_middleware)

    async def _rate_limit_middleware(self, app, handler):
        """
        Rate limiting middleware.

        SECURITY: Prevents DoS and brute force attacks.
        """

        async def middleware_handler(request):
            # Get client identifier (IP address or API key)
            client_id = request.headers.get("X-API-Key", request.remote or "unknown")

            # Check rate limit
            if not await self.rate_limiter.is_allowed(client_id):
                self.request_stats["rate_limited_requests"] += 1
                retry_after = self.rate_limiter.get_retry_after(client_id)

                return web.json_response(
                    {
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                    },
                    status=429,
                    headers={"Retry-After": str(int(retry_after))},
                )

            return await handler(request)

        return middleware_handler

    async def _request_size_middleware(self, app, handler):
        """
        Request size limit middleware.

        SECURITY: Prevents large payload DoS attacks.
        """

        async def middleware_handler(request):
            content_length = request.content_length

            if content_length and content_length > self.max_request_size:
                return web.json_response(
                    {
                        "error": f"Request too large. Max size: {self.max_request_size} bytes",
                    },
                    status=413,
                )

            return await handler(request)

        return middleware_handler

    # ─────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        self._update_request_stats(success=True)

        return web.json_response(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.request_stats["start_time"]),
            }
        )

    # ─────────────────────────────────────────────
    # Execute Endpoints (Real Orchestrator)
    # ─────────────────────────────────────────────

    async def execute_task(self, request: web.Request) -> web.Response:
        """Legacy backward-compat endpoint. Auto-detects format and routes."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        try:
            data = await request.json()
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        # Auto-detect: if "task" field present, route to project execute
        if "task" in data or "project_description" in data:
            return await self._dispatch_execute_project(request, data)
        elif "tasks" in data:
            return await self._dispatch_execute_tasks(request, data)
        elif "spec_dir" in data:
            return await self._dispatch_execute_speckit(request, data)
        else:
            self._update_request_stats(success=False)
            return web.json_response(
                {
                    "error": "Unrecognized request format. Use /execute/project, /execute/tasks, or /execute/from-speckit"
                },
                status=400,
            )

    async def execute_project(self, request: web.Request) -> web.Response:
        """Execute a full project (decompose → execute pipeline).

        POST /execute/project
        Body: {project_description, success_criteria, budget, ...}
        Returns: 202 Accepted with project_id
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        try:
            data = await request.json()
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        return await self._dispatch_execute_project(request, data)

    async def execute_tasks(self, request: web.Request) -> web.Response:
        """Execute pre-composed tasks (skip LLM decomposition).

        POST /execute/tasks
        Body: {tasks: [{id, type, prompt, ...}], budget, ...}
        Returns: 202 Accepted with project_id
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        if not flags.http_ingest_enabled:
            self._update_request_stats(success=False)
            return web.json_response({"error": "HTTP ingest endpoints disabled"}, status=501)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        if "tasks" not in data:
            self._update_request_stats(success=False)
            return web.json_response({"error": "tasks array required"}, status=400)

        return await self._dispatch_execute_tasks(request, data)

    async def execute_from_speckit(self, request: web.Request) -> web.Response:
        """Execute a Spec-Kit directory.

        POST /execute/from-speckit
        Body: {spec_dir, budget, max_concurrency, output_dir}
        Returns: 202 Accepted with project_id and task_count
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        if not flags.http_ingest_enabled:
            self._update_request_stats(success=False)
            return web.json_response({"error": "HTTP ingest endpoints disabled"}, status=501)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        if "spec_dir" not in data:
            self._update_request_stats(success=False)
            return web.json_response({"error": "spec_dir path required"}, status=400)

        return await self._dispatch_execute_speckit(request, data)

    # ─────────────────────────────────────────────
    # Internal Dispatch Methods
    # ─────────────────────────────────────────────

    async def _dispatch_execute_project(self, request: web.Request, data: dict) -> web.Response:
        """Create and dispatch a project execution in the background."""
        from orchestrator.budget import Budget
        from orchestrator.engine import Orchestrator

        project_description = data.get("project_description") or data.get("task", "")
        success_criteria = data.get("success_criteria") or data.get("criteria", "")
        budget_usd = _safe_float(data.get("budget"), 8.0)
        max_time = _safe_int(data.get("max_time_seconds"), 5400)
        concurrency = _safe_int(data.get("concurrency"), 3)

        # Request validation
        err = self._validate_execute_request(data, check_description=True, check_budget=True)
        if err is not None:
            self._update_request_stats(success=False)
            return err

        if not project_description:
            self._update_request_stats(success=False)
            return web.json_response({"error": "project_description required"}, status=400)

        # Generate project ID
        project_id = hashlib.sha256(
            f"{project_description}{datetime.now()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        budget = Budget(max_usd=budget_usd, max_time_seconds=max_time)

        # Reuse the long-running orchestrator if configured, else build one
        # scoped to this request. Either way, `budget` is passed straight
        # into run_project() rather than mutated onto orch._run_ctx here:
        # orch may be shared across concurrently in-flight requests, and
        # writing orch._run_ctx.budget would be unsynchronized shared state
        # between them (see engine.py::run_project's `budget` parameter).
        if self._orchestrator is not None:
            orch = self._orchestrator
        else:
            orch = Orchestrator(budget=budget, max_concurrency=concurrency)

        # Store initial state
        async with self._active_project_lock:
            self._active_project_state[project_id] = {
                "project_id": project_id,
                "status": "accepted",
                "description": project_description[:100],
                "created_at": datetime.now().isoformat(),
                "tasks_total": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "cost_spent_usd": 0.0,
                "elapsed_seconds": 0.0,
            }

        # Launch background execution
        await self._launch_background_project(
            project_id=project_id,
            orch=orch,
            run_fn=lambda: orch.run_project(
                project_description=project_description,
                success_criteria=success_criteria,
                project_id=project_id,
                budget=budget,
            ),
        )
        self._update_request_stats(success=True)
        return web.json_response(
            {
                "project_id": project_id,
                "status": "accepted",
                "status_url": f"/projects/{project_id}",
                "stream_url": f"/projects/{project_id}/stream",
            },
            status=202,
        )

    async def _dispatch_execute_tasks(self, request: web.Request, data: dict) -> web.Response:
        """Create and dispatch a pre-composed task execution in the background."""
        from orchestrator.budget import Budget
        from orchestrator.engine import Orchestrator
        from orchestrator.models import Task, TaskType

        project_description = data.get("project_description", "Pre-composed tasks")
        success_criteria = data.get("success_criteria", "All tasks complete")
        budget_usd = _safe_float(data.get("budget"), 8.0)
        max_time = _safe_int(data.get("max_time_seconds"), 5400)
        concurrency = _safe_int(data.get("concurrency"), 3)
        tasks_data = data.get("tasks", [])

        # Request validation
        err = self._validate_execute_request(
            data,
            required_fields=["tasks"],
            check_description=False,
            check_budget=True,
        )
        if err is not None:
            self._update_request_stats(success=False)
            return err

        if not tasks_data:
            self._update_request_stats(success=False)
            return web.json_response({"error": "tasks array is empty"}, status=400)

        # Parse tasks from JSON
        tasks: dict[str, Task] = {}
        for t in tasks_data:
            tid = t.get("id", f"T{len(tasks) + 1:03d}")
            task = Task(
                id=tid,
                type=TaskType(t.get("type", "code_generation")),
                prompt=t.get("prompt", ""),
                context=t.get("context", ""),
                target_path=t.get("target_path", ""),
                dependencies=t.get("dependencies", []),
                hard_validators=t.get("hard_validators", []),
            )
            tasks[tid] = task

        # Generate project ID
        project_id = hashlib.sha256(
            f"{project_description}{datetime.now()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        budget = Budget(max_usd=budget_usd, max_time_seconds=max_time)
        # See _dispatch_execute_project: pass budget into the call rather
        # than mutating a possibly-shared orch._run_ctx.
        if self._orchestrator is not None:
            orch = self._orchestrator
        else:
            orch = Orchestrator(budget=budget, max_concurrency=concurrency)

        # Store initial state
        async with self._active_project_lock:
            self._active_project_state[project_id] = {
                "project_id": project_id,
                "status": "accepted",
                "description": project_description[:100],
                "created_at": datetime.now().isoformat(),
                "tasks_total": len(tasks),
                "tasks_completed": 0,
                "tasks_failed": 0,
                "cost_spent_usd": 0.0,
                "elapsed_seconds": 0.0,
            }

        # Launch background execution
        await self._launch_background_project(
            project_id=project_id,
            orch=orch,
            run_fn=lambda: orch.run_project_with_tasks(
                project_description=project_description,
                success_criteria=success_criteria,
                tasks=tasks,
                project_id=project_id,
                budget=budget,
            ),
        )

        self._update_request_stats(success=True)
        return web.json_response(
            {
                "project_id": project_id,
                "status": "accepted",
                "task_count": len(tasks),
                "status_url": f"/projects/{project_id}",
                "stream_url": f"/projects/{project_id}/stream",
            },
            status=202,
        )

    async def _dispatch_execute_speckit(self, request: web.Request, data: dict) -> web.Response:
        """Load Spec-Kit artifacts and dispatch execution."""
        from orchestrator.budget import Budget
        from orchestrator.engine import Orchestrator
        from orchestrator.ingest import SpecKitAdapter

        spec_dir = data["spec_dir"]
        budget_usd = _safe_float(data.get("budget"), 8.0)
        max_concurrency = _safe_int(data.get("max_concurrency"), 3)

        # Validate spec_dir
        spec_path = Path(spec_dir)
        if not spec_path.is_dir():
            self._update_request_stats(success=False)
            return web.json_response(
                {"error": f"spec_dir not found or not a directory: {spec_dir}"},
                status=400,
            )

        # Load Spec-Kit artifacts (inline file reader avoids infra import)
        from pathlib import Path as _Path

        class _LocalFileReader:
            async def read_text(self, path: str) -> str:
                return await asyncio.to_thread(_Path(path).read_text, encoding="utf-8")

        adapter = SpecKitAdapter(file_reader=_LocalFileReader())
        try:
            artifacts = await adapter.load(spec_dir)
        except FileNotFoundError as exc:
            self._update_request_stats(success=False)
            return web.json_response({"error": str(exc)}, status=400)
        except ValueError as exc:
            self._update_request_stats(success=False)
            return web.json_response({"error": str(exc)}, status=400)

        if not artifacts.tasks:
            self._update_request_stats(success=False)
            return web.json_response({"error": "No tasks found in tasks.md"}, status=400)

        project_description = f"Spec-Kit project from {spec_dir}"
        success_criteria = (
            " ".join(artifacts.raw_spec_criteria)
            if artifacts.raw_spec_criteria
            else "All tasks complete"
        )

        project_id = hashlib.sha256(
            f"{spec_dir}{datetime.now()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        budget = Budget(max_usd=budget_usd)
        # See _dispatch_execute_project: pass budget into the call rather
        # than mutating a possibly-shared orch._run_ctx.
        if self._orchestrator is not None:
            orch = self._orchestrator
        else:
            orch = Orchestrator(budget=budget, max_concurrency=max_concurrency)

        async with self._active_project_lock:
            self._active_project_state[project_id] = {
                "project_id": project_id,
                "status": "accepted",
                "description": f"Spec-Kit: {spec_dir}",
                "created_at": datetime.now().isoformat(),
                "tasks_total": len(artifacts.tasks),
                "tasks_completed": 0,
                "tasks_failed": 0,
                "cost_spent_usd": 0.0,
                "elapsed_seconds": 0.0,
            }

        # Launch background execution
        await self._launch_background_project(
            project_id=project_id,
            orch=orch,
            run_fn=lambda: orch.run_project_with_tasks(
                project_description=project_description,
                success_criteria=success_criteria,
                tasks=artifacts.tasks,
                project_id=project_id,
                constitution=artifacts.constitution,
                budget=budget,
            ),
        )

        self._update_request_stats(success=True)
        return web.json_response(
            {
                "project_id": project_id,
                "status": "accepted",
                "task_count": len(artifacts.tasks),
                "status_url": f"/projects/{project_id}",
                "stream_url": f"/projects/{project_id}/stream",
            },
            status=202,
        )

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # Background Execution & Validation
    # ─────────────────────────────────────────────

    async def _launch_background_project(
        self,
        project_id: str,
        orch: Any,
        run_fn: Any,
    ) -> asyncio.Task:
        """Create and track a background project execution.

        Wraps a run coroutine with state tracking, error handling,
        and cleanup. Replaces three duplicated ``_run_and_store`` closures.

        Args:
            project_id: Unique project identifier.
            orch: Orchestrator instance (shut down in finally).
            run_fn: Zero-argument async callable returning a ProjectState-like object.

        Returns:
            The background asyncio.Task (also stored in ``self._active_projects``).
        """

        async def _run_and_store() -> None:
            try:
                state = await run_fn()
                async with self._active_project_lock:
                    ps = self._active_project_state.get(project_id, {})
                    ps["status"] = (
                        (getattr(state, "status", None) or "completed").value
                        if hasattr(getattr(state, "status", None), "value")
                        else "completed"
                    )
                    ps["tasks_total"] = len(getattr(state, "tasks", {}))
                    ps["tasks_completed"] = sum(
                        1
                        for r in getattr(state, "results", {}).values()
                        if getattr(r, "status", None) and r.status.name == "COMPLETED"
                    )
                    ps["tasks_failed"] = sum(
                        1
                        for r in getattr(state, "results", {}).values()
                        if getattr(r, "status", None) and r.status.name == "FAILED"
                    )
                    ps["cost_spent_usd"] = getattr(orch.budget, "spent_usd", 0.0)
                    ps["elapsed_seconds"] = getattr(orch.budget, "elapsed_seconds", 0.0)
            except Exception as exc:
                logger.exception("Project %s failed: %s", project_id, exc)
                async with self._active_project_lock:
                    ps = self._active_project_state.get(project_id, {})
                    ps["status"] = "failed"
                    ps["error"] = str(exc)[:500]
            finally:
                async with self._active_project_lock:
                    self._active_projects.pop(project_id, None)
                await _safe_shutdown_async(orch)

        task = asyncio.create_task(_run_and_store())
        async with self._active_project_lock:
            self._active_projects[project_id] = task
        return task

    def _validate_execute_request(
        self,
        data: dict,
        required_fields: list[str] | None = None,
        *,
        check_budget: bool = True,
        check_description: bool = True,
    ) -> web.Response | None:
        """Validate common execute request fields.

        Args:
            data: Parsed JSON body.
            required_fields: Additional field names that must be non-empty.
            check_budget: If True, validate budget_usd > 0 when explicitly set.
            check_description: If True, validate project_description is non-empty.

        Returns:
            A 400 error ``web.Response`` if validation fails, or ``None`` if valid.
        """
        if check_description:
            desc = data.get("project_description") or data.get("task", "")
            if not desc or not desc.strip():
                return web.json_response(
                    {"error": "project_description is required and must be non-empty"},
                    status=400,
                )

        if check_budget:
            budget_raw = data.get("budget")
            if budget_raw is not None:
                budget_val = _safe_float(budget_raw, -1.0)
                if budget_val <= 0:
                    return web.json_response(
                        {"error": "budget must be a positive number"},
                        status=400,
                    )

        if required_fields:
            for field in required_fields:
                val = data.get(field)
                if not val:
                    return web.json_response(
                        {"error": f"{field} is required"},
                        status=400,
                    )

        return None

    # Project Status & Streaming
    # ─────────────────────────────────────────────

    async def get_project_status(self, request: web.Request) -> web.Response:
        """Get status of a project by ID.

        GET /projects/{project_id}
        Returns current ProjectState from the background execution or StateManager.
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        project_id = request.match_info["project_id"]

        # Check active (in-memory) state first
        async with self._active_project_lock:
            state = self._active_project_state.get(project_id)

        # Try StateManager for completed/persisted projects
        if state is None:
            try:
                from orchestrator.state import StateManager

                sm = StateManager()
                try:
                    persisted = await sm.load_project(project_id)
                    if persisted:
                        results = {}
                        for task_id, result in getattr(persisted, "results", {}).items():
                            results[task_id] = {
                                "status": (
                                    getattr(result, "status", None).name
                                    if hasattr(getattr(result, "status", None), "name")
                                    else "unknown"
                                ),
                                "score": getattr(result, "score", 0.0),
                                "cost_usd": getattr(result, "cost_usd", 0.0),
                                "output_preview": (getattr(result, "output", "") or "")[:200],
                            }
                        state = {
                            "project_id": project_id,
                            "status": (
                                getattr(persisted, "status", None).value
                                if hasattr(getattr(persisted, "status", None), "value")
                                else "unknown"
                            ),
                            "tasks_total": len(getattr(persisted, "tasks", {})),
                            "tasks_completed": sum(
                                1
                                for r in getattr(persisted, "results", {}).values()
                                if getattr(r, "status", None) and r.status.name == "COMPLETED"
                            ),
                            "tasks_failed": sum(
                                1
                                for r in getattr(persisted, "results", {}).values()
                                if getattr(r, "status", None) and r.status.name == "FAILED"
                            ),
                            "results": results,
                        }
                finally:
                    await sm.close()
            except Exception as exc:
                logger.error(
                    "Failed to load project %s from StateManager: %s",
                    project_id,
                    exc,
                )
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "Internal error loading project state"},
                    status=500,
                )

        if state is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Project not found"}, status=404)

        # Check if task is still running
        async with self._active_project_lock:
            is_running = project_id in self._active_projects

        now = datetime.now()

        response = {
            "project_id": project_id,
            "status": state.get("status", "unknown"),
            "tasks_total": state.get("tasks_total", 0),
            "tasks_completed": state.get("tasks_completed", 0),
            "tasks_failed": state.get("tasks_failed", 0),
            "cost_spent_usd": state.get("cost_spent_usd", 0.0),
            "is_running": is_running,
            "results": state.get("results", {}),
            "error": state.get("error"),
        }

        self._update_request_stats(success=True)
        return web.json_response(response)

    async def project_stream(self, request: web.Request) -> web.Response:
        """SSE endpoint for real-time project execution events.

        GET /projects/{project_id}/stream
        Returns text/event-stream with PipelineEvent JSON.
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        if not flags.http_stream_enabled:
            self._update_request_stats(success=False)
            return web.json_response({"error": "HTTP streaming endpoints disabled"}, status=501)

        project_id = request.match_info["project_id"]

        # Check project exists
        async with self._active_project_lock:
            if (
                project_id not in self._active_projects
                and project_id not in self._active_project_state
            ):
                self._update_request_stats(success=False)
                return web.json_response({"error": "Project not found"}, status=404)

        # Create SSE response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            # SSE streaming: event-bus subscription when available, polling fallback
            if self._orchestrator is not None and not flags.http_stream_polling:
                await self._stream_via_event_bus(response, request, project_id)
                return response

            # Poll for project state changes and emit SSE events
            last_status = None
            last_task_count = 0
            while True:
                async with self._active_project_lock:
                    state = self._active_project_state.get(project_id)
                    is_running = project_id in self._active_projects

                if state is None:
                    break

                status = state.get("status", "unknown")

                # Emit status events on change
                if status != last_status:
                    event_data = json.dumps(
                        {
                            "type": "STATUS_CHANGE",
                            "project_id": project_id,
                            "status": status,
                            "tasks_total": state.get("tasks_total", 0),
                            "tasks_completed": state.get("tasks_completed", 0),
                            "tasks_failed": state.get("tasks_failed", 0),
                            "cost_spent_usd": state.get("cost_spent_usd", 0.0),
                        }
                    )
                    payload = f"event: {status}\ndata: {event_data}\n\n"
                    await response.write(payload.encode("utf-8"))
                    last_status = status

                # Emit on task progress changes
                tc = state.get("tasks_completed", 0)
                if tc != last_task_count:
                    event_data = json.dumps(
                        {
                            "type": "TASK_PROGRESS",
                            "project_id": project_id,
                            "tasks_completed": tc,
                            "tasks_failed": state.get("tasks_failed", 0),
                            "tasks_total": state.get("tasks_total", 0),
                        }
                    )
                    payload = f"event: task_progress\ndata: {event_data}\n\n"
                    await response.write(payload.encode("utf-8"))
                    last_task_count = tc

                # Completed or failed — send final event and exit
                if status in ("completed", "failed", "error", "cancelled"):
                    final_data = json.dumps(
                        {
                            "type": (
                                "PROJECT_COMPLETE" if status == "completed" else "PROJECT_FAILED"
                            ),
                            "project_id": project_id,
                            "status": status,
                            "cost_spent_usd": state.get("cost_spent_usd", 0.0),
                        }
                    )
                    payload = f"event: complete\ndata: {final_data}\n\n"
                    await response.write(payload.encode("utf-8"))
                    break

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("SSE stream error for %s: %s", project_id, exc)
        finally:
            await response.write_eof()

        return response

    async def cancel_project(self, request: web.Request) -> web.Response:
        """Cancel a running project.

        DELETE /projects/{project_id}
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        project_id = request.match_info["project_id"]

        async with self._active_project_lock:
            task = self._active_projects.get(project_id)
            if task is None:
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "Project not found or already completed"}, status=404
                )

            task.cancel()
            del self._active_projects[project_id]
            if project_id in self._active_project_state:
                self._active_project_state[project_id]["status"] = "cancelled"

        self._update_request_stats(success=True)
        return web.json_response({"project_id": project_id, "status": "cancelled"})

    # ─────────────────────────────────────────────
    # Legacy Status (backward compat)
    # ─────────────────────────────────────────────

    async def get_task_status(self, request: web.Request) -> web.Response:
        """Get the status of a task (legacy).

        GET /status/{task_id} — backward compat, wraps project status.
        """
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        task_id = request.match_info["task_id"]

        # Check if this is actually a project_id
        async with self._active_project_lock:
            state = self._active_project_state.get(task_id)

        if state is None:
            self._update_request_stats(success=False)
            return web.json_response(
                {
                    "task_id": task_id,
                    "status": "unknown",
                    "message": "No project found with this ID",
                }
            )

        async with self._active_project_lock:
            is_running = task_id in self._active_projects

        self._update_request_stats(success=True)
        return web.json_response(
            {
                "task_id": task_id,
                "status": state.get("status", "unknown"),
                "progress": (
                    state.get("tasks_completed", 0) / max(state.get("tasks_total", 1), 1) * 100
                    if state.get("tasks_total", 0) > 0
                    else 0
                ),
                "result_available": state.get("status") in ("completed", "failed"),
                "is_running": is_running,
            }
        )

    # ─────────────────────────────────────────────
    # Models (de-stubbed)
    # ─────────────────────────────────────────────

    async def list_models(self, request: web.Request) -> web.Response:
        """List available models from the routing table."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        try:
            from orchestrator.models import COST_TABLE, Model, ROUTING_TABLE

            models = []
            for model in Model:
                cost = COST_TABLE.get(model, {})
                route_map = {}
                for task_type_str, preferred_model in ROUTING_TABLE.items():
                    if preferred_model == model:
                        route_map[task_type_str] = "preferred"

                models.append(
                    {
                        "id": model.value,
                        "name": model.name,
                        "input_cost_per_mil": (
                            cost.get("input", 0) if isinstance(cost, dict) else (cost or 0)
                        ),
                        "output_cost_per_mil": (
                            cost.get("output", 0) if isinstance(cost, dict) else 0
                        ),
                        "routed_for": list(route_map.keys()),
                    }
                )

            models.sort(key=lambda m: m["id"])
            self._update_request_stats(success=True)
            return web.json_response(models)

        except ImportError:
            # Fallback to hardcoded list
            self._update_request_stats(success=True)
            return web.json_response(
                [
                    {
                        "id": "gpt-4o",
                        "name": "GPT-4o",
                        "input_cost_per_mil": 2.5,
                        "output_cost_per_mil": 10.0,
                    },
                    {
                        "id": "claude-sonnet-4-20250514",
                        "name": "Claude Sonnet 4",
                        "input_cost_per_mil": 3.0,
                        "output_cost_per_mil": 15.0,
                    },
                    {
                        "id": "deepseek-chat",
                        "name": "DeepSeek Chat V3",
                        "input_cost_per_mil": 0.5,
                        "output_cost_per_mil": 2.0,
                    },
                ]
            )

    # ─────────────────────────────────────────────
    # Auth
    # ─────────────────────────────────────────────

    async def register_api_key(self, request: web.Request) -> web.Response:
        """Register a new API key. Requires ORCHESTRATOR_ADMIN_SECRET header."""
        self._update_request_stats()

        denied = self._require_admin(request)
        if denied:
            self._update_request_stats(success=False)
            return denied

        try:
            data = await request.json()

            if "user_id" not in data:
                self._update_request_stats(success=False)
                return web.json_response({"error": "user_id required"}, status=400)

            user_id = str(data["user_id"])
            if not user_id.replace("-", "").replace("_", "").isalnum():
                self._update_request_stats(success=False)
                return web.json_response({"error": "user_id must be alphanumeric"}, status=400)

            permissions = data.get("permissions", ["read", "execute"])

            # Generate a cryptographically random API key
            import secrets as _secrets

            raw_key = f"orchestrator_{_secrets.token_urlsafe(32)}"
            hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

            # Store the API key
            self.api_keys[hashed_key] = {
                "user_id": user_id,
                "permissions": permissions,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
            }

            self._update_request_stats(success=True)
            return web.json_response(
                {"api_key": raw_key, "message": "API key registered successfully"}
            )

        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)
        except Exception as e:
            logger.error(f"Error registering API key: {e}")
            self._update_request_stats(success=False)
            return web.json_response({"error": "Internal server error"}, status=500)

    # ─────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────

    async def get_stats(self, request: web.Request) -> web.Response:
        """Get server statistics."""
        self._update_request_stats()

        denied = self._require_auth(request)
        if denied:
            self._update_request_stats(success=False)
            return denied

        uptime = datetime.now() - self.request_stats["start_time"]

        stats = {
            "total_requests": self.request_stats["total_requests"],
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "success_rate": (
                self.request_stats["successful_requests"] / self.request_stats["total_requests"]
                if self.request_stats["total_requests"] > 0
                else 0
            ),
            "uptime": str(uptime),
            "server_time": datetime.now().isoformat(),
            "registered_api_keys": len(self.api_keys),
            "active_projects": len(self._active_projects),
        }

        return web.json_response(stats)

    # ─────────────────────────────────────────────
    # Supervisor endpoints (Phase 2)
    # ─────────────────────────────────────────────

    async def supervisor_directive(self, request: web.Request) -> web.Response:
        """Accept an inbound agent directive and hand it to the Supervisor."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error

        if self.supervisor is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Supervisor not configured"}, status=503)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        text = data.get("text", "").strip()
        if not text:
            self._update_request_stats(success=False)
            return web.json_response({"error": "text required"}, status=400)

        def _to_str(value: Any, max_len: int = 1000) -> str:
            if not isinstance(value, str):
                return ""
            return value[:max_len]

        def _to_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        budget = _to_float(data.get("budget"))
        if data.get("budget") is not None and budget is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "budget must be a number"}, status=400)

        from orchestrator.supervisor.models import Directive

        directive = Directive(
            source="agent",
            text=text[:4000],
            project_id=_to_str(data.get("project_id", ""), 256),
            criteria=_to_str(data.get("criteria", ""), 4000),
            budget=budget,
        )

        try:
            result = await self.supervisor.handle(directive)
            self._update_request_stats(success=True)
            return web.json_response(
                {
                    "session_id": result.session_id,
                    "status": result.project_status,
                    "lessons_recorded": result.lessons_recorded,
                }
            )
        except Exception as exc:
            logger.exception("Supervisor directive failed")
            self._update_request_stats(success=False)
            return web.json_response(
                {
                    "error": "Supervisor directive failed",
                    "detail": "See server logs for more information",
                },
                status=500,
            )

    async def list_supervisor_sessions(self, request: web.Request) -> web.Response:
        """List supervisor sessions."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error
        if self.supervisor is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Supervisor not configured"}, status=503)
        self._update_request_stats(success=True)
        sessions = await self.supervisor._store.list_sessions()
        return web.json_response(
            [
                {
                    "id": s.id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "status": s.status,
                    "summary": s.summary,
                    "directive_count": s.directive_count,
                }
                for s in sessions
            ]
        )

    async def get_supervisor_session(self, request: web.Request) -> web.Response:
        """Get a single supervisor session."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error
        if self.supervisor is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Supervisor not configured"}, status=503)
        self._update_request_stats(success=True)
        session_id = request.match_info["id"]
        session = await self.supervisor._store.get_session(session_id)
        if session is None:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response(
            {
                "id": session.id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "status": session.status,
                "summary": session.summary,
                "directive_count": session.directive_count,
            }
        )

    async def get_supervisor_lessons(self, request: web.Request) -> web.Response:
        """Get lessons for a supervisor session."""
        auth_error = self._require_auth(request)
        if auth_error is not None:
            return auth_error
        if self.supervisor is None:
            self._update_request_stats(success=False)
            return web.json_response({"error": "Supervisor not configured"}, status=503)
        self._update_request_stats(success=True)
        session_id = request.match_info["id"]
        lessons = await self.supervisor._store.search_lessons(session_id, limit=100)
        return web.json_response(
            [
                {
                    "id": lesson.id,
                    "created_at": lesson.created_at,
                    "updated_at": lesson.updated_at,
                    "session_id": lesson.session_id,
                    "project_id": lesson.project_id,
                    "task_type": lesson.task_type,
                    "kind": lesson.kind,
                    "signal": lesson.signal,
                    "detail": lesson.detail,
                }
                for lesson in lessons
            ]
        )

    # ─────────────────────────────────────────────
    # Auth helpers
    # ─────────────────────────────────────────────

    async def _stream_via_event_bus(
        self, response: web.StreamResponse, request: web.Request, project_id: str
    ) -> None:
        """Stream SSE events via ProjectEventBus subscription."""
        try:
            from orchestrator.streaming import ProjectEventBus
            import asyncio as _asyncio

            bus = ProjectEventBus()
            subscription = bus.subscribe()
            async for event in subscription:
                event_type = getattr(event, "event_type", "message")
                event_data = json.dumps(
                    {"type": event_type, "project_id": project_id},
                    default=str,
                )
                payload = f"event: {event_type}\ndata: {event_data}\n\n"
                await response.write(payload.encode("utf-8"))
                if event_type in ("project_completed", "project_failed", "task_failed"):
                    is_terminal = getattr(event, "project_id", "") == project_id
                    if is_terminal:
                        break
        except ImportError:
            logger.info("ProjectEventBus not available — falling back to polling")
            return  # fall through to polling
        except Exception as exc:
            logger.error("Event-bus SSE error for %s: %s", project_id, exc)

    def _require_auth(self, request: web.Request) -> web.Response | None:
        """Enforce Bearer auth when ``auth_required``.

        Returns a 401 response if the header is missing/invalid, else ``None``.
        Shared by all endpoints so read endpoints can't bypass auth.
        """
        if not self.auth_required:
            return None
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            self._update_request_stats(success=False)
            return web.json_response({"error": "Authorization header required"}, status=401)
        if not self._verify_api_key(auth_header[7:]):
            self._update_request_stats(success=False)
            return web.json_response({"error": "Invalid API key"}, status=401)
        return None

    def _verify_api_key(self, api_key: str) -> bool:
        """Verify an API key using constant-time comparison to prevent timing attacks."""
        import hmac as _hmac

        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        for stored_key, meta in self.api_keys.items():
            if _hmac.compare_digest(stored_key, hashed_key):
                meta["last_used"] = datetime.now().isoformat()
                return True
        return False

    def _require_admin(self, request: web.Request) -> web.Response | None:
        """Return 403 Response if not admin, None if OK. Use for key registration."""
        import os as _os

        admin_secret = _os.environ.get("ORCHESTRATOR_ADMIN_SECRET")
        if not admin_secret:
            return web.json_response(
                {"error": "Admin key registration disabled (ORCHESTRATOR_ADMIN_SECRET not set)"},
                status=503,
            )
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return web.json_response({"error": "Admin authorization required"}, status=401)
        import hmac as _hmac

        if not _hmac.compare_digest(auth_header[7:], admin_secret):
            return web.json_response({"error": "Invalid admin secret"}, status=403)
        return None

    def _update_request_stats(self, success: bool = True):
        """Update request statistics."""
        self.request_stats["total_requests"] += 1
        if success:
            self.request_stats["successful_requests"] += 1
        else:
            self.request_stats["failed_requests"] += 1

    # ─────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────

    async def start(self):
        """Start the API server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        logger.info(f"API Server started at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the API server and cancel active projects."""
        # Cancel all active projects
        for pid, task in list(self._active_projects.items()):
            task.cancel()
            logger.info("Cancelled project %s", pid)
        self._active_projects.clear()

        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        logger.info("API Server stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.site is not None and self.runner is not None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def _safe_shutdown_async(orch: Any) -> None:
    """Safely shut down an orchestrator instance without raising."""
    try:
        if hasattr(orch, "state_mgr") and hasattr(orch.state_mgr, "close"):
            await orch.state_mgr.close()
        if hasattr(orch, "cache") and hasattr(orch.cache, "close"):
            await orch.cache.close()
        if hasattr(orch, "close"):
            await orch.close()
    except Exception:
        pass


# Global server instance for convenience
_global_server: APIServer | None = None


async def get_server_instance(port: int = 8000, host: str = "localhost") -> APIServer:
    """
    Get the global server instance, creating it if it doesn't exist.

    Args:
        port: Port to run the server on
        host: Host to bind to

    Returns:
        APIServer instance
    """
    global _global_server
    if _global_server is None:
        _global_server = APIServer(port=port, host=host)
    return _global_server


async def start_server(port: int = 8000, host: str = "localhost"):
    """
    Start the API server.

    Args:
        port: Port to run the server on
        host: Host to bind to
    """
    server = await get_server_instance(port, host)
    await server.start()


async def stop_server():
    """Stop the API server."""
    global _global_server
    if _global_server:
        await _global_server.stop()
        _global_server = None
