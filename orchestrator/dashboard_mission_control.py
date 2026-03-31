"""
LLM Orchestrator Dashboard v6.5.9 - Project Specification Support
================================================================
✅ New Project - Create from scratch
✅ Improve Codebase - Refactor existing code
✅ YAML Spec - Upload project specification file
✅ HTTP Polling - Reliable real-time updates
"""

from __future__ import annotations

import asyncio
import json

# Load environment variables from .env file FIRST (before other imports)
import os
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    print("[LLM Orchestrator] [OK] Loaded .env file")
except ImportError:
    print("[LLM Orchestrator] [WARN] python-dotenv not installed, .env file not loaded")
except Exception as e:
    print(f"[LLM Orchestrator] [WARN] Failed to load .env: {e}")

from .analyzer import CodebaseAnalyzer
from .budget import Budget
from .engine import Orchestrator
from .hooks import EventType
from .improvement_suggester import ImprovementSuggester
from .log_config import get_logger
from .models import ProjectStatus, Task, TaskResult

logger = get_logger(__name__)


class ProjectMode(str, Enum):
    NEW = "new"
    IMPROVE = "improve"
    UPLOAD = "upload"


@dataclass
class ApiStatus:
    """API connection status with credit info."""

    provider: str
    status: str = "checking"  # checking, connected, error, no_key, invalid_key, no_credits
    models: list[str] = field(default_factory=list)
    error: str = ""
    response_time_ms: float = 0.0
    credits_info: str | None = None  # e.g., "Credits: $5.23 remaining"
    validated: bool = False  # True if actually tested (not just key presence)

    def to_dict(self):
        return {
            "provider": self.provider,
            "status": self.status,
            "models": self.models,
            "error": self.error,
            "response_time_ms": round(self.response_time_ms, 2),
            "credits_info": self.credits_info,
            "validated": self.validated,
        }


@dataclass
class TaskInfo:
    """Task information for dashboard."""

    task_id: str
    task_type: str
    status: str = "pending"
    model: str = ""
    progress: int = 0
    score: float = 0.0
    start_time: float = 0.0

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "model": self.model,
            "progress": self.progress,
            "score": self.score,
        }


@dataclass
class ActiveProject:
    """Active project state."""

    id: str
    name: str
    mode: str
    prompt: str = ""
    criteria: str = ""
    project_type: str = ""
    codebase_path: str = ""
    file_path: str = ""
    file_name: str = ""
    instructions: str = ""
    budget: float = 5.0
    status: str = "starting"
    progress: int = 0
    current_task: str = "Initializing..."
    tasks: list[TaskInfo] = field(default_factory=list)
    completed_tasks: int = 0
    total_tasks: int = 5
    cost: float = 0.0
    start_time: float = 0.0
    logs: list[dict] = field(default_factory=list)
    orchestrator: Orchestrator | None = None
    task_handle: asyncio.Task | None = None

    # Advanced settings
    time_seconds: int = 4500
    concurrency: int = 3
    project_id: str = ""
    output_dir: str = "./outputs"
    quality_targets: dict = field(default_factory=dict)
    policies: list[dict] = field(default_factory=list)

    def add_log(self, level: str, message: str):
        self.logs.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )

    def to_dict(self):
        # Calculate actual task counts from tasks list if available
        actual_completed = (
            len([t for t in self.tasks if t.status == "completed"])
            if self.tasks
            else self.completed_tasks
        )
        actual_total = len(self.tasks) if self.tasks else self.total_tasks

        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "prompt": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "type": self.project_type,
            "status": self.status,
            "progress": self.progress,
            "current_task": self.current_task,
            "tasks_completed": actual_completed,
            "tasks_total": max(actual_total, 1),  # Ensure at least 1 to avoid division by zero
            "tasks": [t.to_dict() for t in self.tasks],  # Include full tasks list for frontend
            "cost": round(self.cost, 4),
            "budget": self.budget,
            "elapsed": int(time.time() - self.start_time) if self.start_time else 0,
            # Advanced settings (for display)
            "project_id": self.project_id,
            "output_dir": self.output_dir,
            "time_seconds": self.time_seconds,
            "concurrency": self.concurrency,
        }


@dataclass
class MissionState:
    """Dashboard state."""

    version: str = "6.5.22"
    server_status: str = "starting"  # starting, connected, error
    active_projects: list[ActiveProject] = field(default_factory=list)
    api_status: list[ApiStatus] = field(default_factory=list)
    system_logs: list[dict] = field(default_factory=list)

    def get_running_count(self) -> int:
        return sum(1 for p in self.active_projects if p.status == "running")

    def get_completed_count(self) -> int:
        return sum(1 for p in self.active_projects if p.status == "completed")

    def add_system_log(self, level: str, message: str):
        """Add a system log entry."""
        import time

        self.system_logs.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        # Keep only last 50 logs
        if len(self.system_logs) > 50:
            self.system_logs = self.system_logs[-50:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "server_status": self.server_status,
            "projects_running": self.get_running_count(),
            "projects_completed": self.get_completed_count(),
            "total_projects": len(self.active_projects),
            "active_projects": [p.to_dict() for p in self.active_projects],
            "api_status": [a.to_dict() for a in self.api_status],
            "system_logs": self.system_logs,
        }


class MissionControlServer:
    """Full-featured mission control server with concurrent project support."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.state = MissionState()
        self.app = None
        self.active_connections: list[Any] = []

    async def _check_api_connections(self):
        """Check all API connections on startup with actual validation."""

        print("[_check_api_connections] Starting...")
        logger.info("🔌 Checking API connections...")

        providers = [
            ("openai", "OPENAI_API_KEY", ["gpt-4o", "gpt-4o-mini"]),
            ("deepseek", "DEEPSEEK_API_KEY", ["deepseek-chat", "deepseek-reasoner"]),
            ("google", "GOOGLE_API_KEY", ["gemini-2.5-pro", "gemini-2.5-flash"]),
            ("anthropic", "ANTHROPIC_API_KEY", ["claude-3-5-sonnet", "claude-3-haiku"]),
            ("minimax", "MINIMAX_API_KEY", ["minimax-3"]),
        ]

        for provider, env_var, models in providers:
            print(f"[_check_api_connections] Checking {provider}...")
            status = ApiStatus(provider=provider, models=models)
            api_key = os.getenv(env_var)

            if not api_key:
                status.status = "no_key"
                status.error = f"{env_var} not set"
                logger.warning(f"⚠️ {provider}: No API key")
                print(f"[_check_api_connections] {provider}: No API key")
            else:
                # Actually test the API connection
                await self._validate_api_provider(provider, api_key, status)

            self.state.api_status.append(status)

        validated = sum(1 for a in self.state.api_status if a.validated)
        connected = sum(1 for a in self.state.api_status if a.status == "connected")
        failed = sum(
            1 for a in self.state.api_status if a.status in ("error", "invalid_key", "no_credits")
        )

        logger.info(f"🔌 API Check Complete: {connected}/{validated} validated, {failed} failed")
        print(
            f"[_check_api_connections] Complete: {connected}/{validated} validated, {failed} failed"
        )

        print("[_check_api_connections] Broadcasting state...")
        await self._broadcast_state()
        print("[_check_api_connections] Done!")

    async def _validate_api_provider(self, provider: str, api_key: str, status: ApiStatus):
        """Validate API key by making a test call and checking credits."""
        import time as time_module

        start_time = time_module.time()

        try:
            if provider == "openai":
                await self._test_openai(api_key, status)
            elif provider == "deepseek":
                await self._test_deepseek(api_key, status)
            elif provider == "google":
                await self._test_google(api_key, status)
            elif provider == "anthropic":
                await self._test_anthropic(api_key, status)
            elif provider == "minimax":
                await self._test_minimax(api_key, status)
            else:
                status.status = "error"
                status.error = "Unknown provider"

        except Exception as e:
            error_str = str(e).lower()
            status.response_time_ms = (time_module.time() - start_time) * 1000
            status.validated = True

            # Classify the error
            if any(
                x in error_str for x in ["invalid api key", "authentication", "unauthorized", "401"]
            ):
                status.status = "invalid_key"
                status.error = "Invalid API key"
                logger.warning(f"❌ {provider}: Invalid API key")
                # Add to state for system log
                self.state.add_system_log(
                    "error", f"{provider}: Invalid API key - check your {provider.upper()}_API_KEY"
                )
            elif any(
                x in error_str
                for x in ["quota", "credit", "billing", "insufficient", "exceeded", "429", "limit"]
            ):
                status.status = "no_credits"
                status.error = "Insufficient credits or quota exceeded"
                status.credits_info = "No credits available"
                logger.warning(f"⚠️ {provider}: No credits")
                self.state.add_system_log(
                    "error", f"{provider}: No credits - please check your billing/account"
                )
            elif any(
                x in error_str
                for x in ["timeout", "timed out", "connection", "network", "unreachable"]
            ):
                status.status = "error"
                status.error = f"Connection timeout: {str(e)[:50]}"
                logger.warning(f"⏱️ {provider}: Timeout - {e}")
                self.state.add_system_log(
                    "warning", f"{provider}: Connection timeout - API may be slow/down"
                )
            else:
                status.status = "error"
                status.error = str(e)[:100]
                logger.warning(f"❌ {provider}: Error - {e}")
                self.state.add_system_log("error", f"{provider}: API error - {str(e)[:80]}")

    async def _test_openai(self, api_key: str, status: ApiStatus):
        """Test OpenAI API and get credit info."""
        import asyncio
        import time as time_module

        start_time = time_module.time()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, max_retries=0)

            # Test with minimal completion (costs almost nothing)
            await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                ),
                timeout=10,
            )

            status.status = "connected"
            status.response_time_ms = (time_module.time() - start_time) * 1000
            status.validated = True
            status.credits_info = "API key valid"
            logger.info(f"✅ OpenAI: Connected ({status.response_time_ms:.0f}ms)")

        except Exception:
            raise

    async def _test_deepseek(self, api_key: str, status: ApiStatus):
        """Test DeepSeek API."""
        import asyncio
        import time as time_module

        start_time = time_module.time()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=api_key, base_url="https://api.deepseek.com/v1", max_retries=0
            )

            # Test with minimal completion
            await asyncio.wait_for(
                client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                ),
                timeout=10,
            )

            status.status = "connected"
            status.response_time_ms = (time_module.time() - start_time) * 1000
            status.validated = True
            status.credits_info = "API key valid"
            logger.info(f"✅ DeepSeek: Connected ({status.response_time_ms:.0f}ms)")

        except Exception:
            raise

    async def _test_google(self, api_key: str, status: ApiStatus):
        """Test Google Gemini API via REST."""
        import time as time_module

        start_time = time_module.time()

        try:
            import aiohttp

            # Use REST API directly to avoid SDK issues
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("name", "") for m in data.get("models", [])]

                        status.status = "connected"
                        status.response_time_ms = (time_module.time() - start_time) * 1000
                        status.validated = True
                        status.credits_info = f"{len(models)} models available"
                        logger.info(f"✅ Google: Connected ({status.response_time_ms:.0f}ms)")
                    elif response.status == 400:
                        text = await response.text()
                        if "API key not valid" in text:
                            raise Exception("Invalid API key")
                        else:
                            raise Exception(f"Bad request: {text[:100]}")
                    else:
                        text = await response.text()
                        raise Exception(f"HTTP {response.status}: {text[:100]}")

        except ImportError:
            # aiohttp not available, try with httpx
            try:
                import httpx

                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        models = [m.get("name", "") for m in data.get("models", [])]

                        status.status = "connected"
                        status.response_time_ms = (time_module.time() - start_time) * 1000
                        status.validated = True
                        status.credits_info = f"{len(models)} models available"
                        logger.info(f"✅ Google: Connected ({status.response_time_ms:.0f}ms)")
                    else:
                        raise Exception(f"HTTP {response.status_code}")
            except Exception:
                raise
        except Exception:
            raise

    async def _test_anthropic(self, api_key: str, status: ApiStatus):
        """Test Anthropic Claude API."""
        import asyncio
        import time as time_module

        start_time = time_module.time()

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key)

            # Try a simple message with minimal tokens
            await asyncio.wait_for(
                client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "Hi"}],
                ),
                timeout=10,
            )

            status.status = "connected"
            status.response_time_ms = (time_module.time() - start_time) * 1000
            status.validated = True
            status.credits_info = "API key valid"
            logger.info(f"✅ Anthropic: Connected ({status.response_time_ms:.0f}ms)")

        except Exception:
            raise

    async def _test_minimax(self, api_key: str, status: ApiStatus):
        """Test Minimax API."""
        import asyncio
        import time as time_module

        start_time = time_module.time()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=api_key, base_url="https://api.minimaxi.chat/v1", max_retries=0
            )

            # Try a simple chat completion with minimal tokens
            await asyncio.wait_for(
                client.chat.completions.create(
                    model="MiniMax-Text-01",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                ),
                timeout=10,
            )

            status.status = "connected"
            status.response_time_ms = (time_module.time() - start_time) * 1000
            status.validated = True
            status.credits_info = "API key valid"
            logger.info(f"✅ Minimax: Connected ({status.response_time_ms:.0f}ms)")

        except Exception:
            raise

    async def _setup_fastapi(self):
        from fastapi import FastAPI, File, Form
        from fastapi.responses import HTMLResponse, JSONResponse

        self.app = FastAPI(title="LLM Orchestrator v6.5.22")

        # Note: CORS disabled - may cause issues with some browsers
        # self.app.add_middleware(
        #     CORSMiddleware,
        #     allow_origins=["*"],
        #     allow_credentials=True,
        #     allow_methods=["*"],
        #     allow_headers=["*"],
        # )

        @self.app.get("/")
        async def get_dashboard():
            return HTMLResponse(content=self._get_html())

        @self.app.get("/api/state")
        async def get_state():
            try:
                return JSONResponse(content=self.state.to_dict())
            except Exception as e:
                logger.exception("Failed to get state")
                return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

        @self.app.get("/api/ping")
        async def ping():
            """Simple ping endpoint for testing."""
            return {
                "status": "ok",
                "time": time.time(),
                "projects": len(self.state.active_projects),
            }

        @self.app.get("/api/debug")
        async def get_debug_info():
            """Debug endpoint with detailed internal state."""

            try:
                # Build debug info step by step with error checking
                debug_info = {"server_time": time.time()}

                # API status
                try:
                    debug_info["api_status"] = [
                        {
                            "provider": a.provider,
                            "status": a.status,
                            "models": a.models,
                            "error": a.error,
                            "response_time_ms": a.response_time_ms,
                        }
                        for a in self.state.api_status
                    ]
                except Exception as api_e:
                    debug_info["api_status_error"] = str(api_e)

                # Active projects
                debug_info["active_projects"] = []
                for p in self.state.active_projects:
                    try:
                        project_debug = {
                            "id": str(p.id),
                            "name": str(p.name),
                            "status": str(p.status),
                            "progress": int(p.progress),
                            "current_task": str(p.current_task) if p.current_task else None,
                            "mode": str(p.mode),
                            "budget": float(p.budget),
                            "cost": float(p.cost),
                            "tasks_count": len(p.tasks),
                            "tasks_completed": int(p.completed_tasks),
                            "logs_count": len(p.logs),
                            "has_orchestrator": p.orchestrator is not None,
                            "task_handle": (
                                {
                                    "done": p.task_handle.done() if p.task_handle else None,
                                    "cancelled": (
                                        p.task_handle.cancelled() if p.task_handle else None
                                    ),
                                }
                                if p.task_handle
                                else None
                            ),
                        }
                        debug_info["active_projects"].append(project_debug)
                    except Exception as p_e:
                        debug_info["active_projects"].append(
                            {"error": str(p_e), "id": str(getattr(p, "id", "unknown"))}
                        )

                # Serialize with default=str as fallback
                return JSONResponse(content=json.loads(json.dumps(debug_info, default=str)))

            except Exception as e:
                logger.exception("Failed to get debug info")
                return JSONResponse(
                    {"status": "error", "message": str(e), "traceback": traceback.format_exc()},
                    status_code=500,
                )

        @self.app.get("/api/models")
        async def get_models():
            all_models = []
            for api in self.state.api_status:
                for model in api.models:
                    all_models.append(
                        {
                            "name": model,
                            "provider": api.provider,
                            "status": api.status,
                            "available": api.status == "connected",
                        }
                    )
            return JSONResponse(
                {"models": all_models, "apis": [a.to_dict() for a in self.state.api_status]}
            )

        @self.app.get("/api/debug/env")
        async def debug_env():
            """Debug endpoint to check which API keys are set (without exposing values)."""
            env_vars = [
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GOOGLE_API_KEY",
                "ANTHROPIC_API_KEY",
                "MINIMAX_API_KEY",
            ]
            result = {}
            for var in env_vars:
                value = os.getenv(var)
                if value:
                    # Show first 10 chars and last 4 chars only
                    masked = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
                    result[var] = {"status": "set", "preview": masked, "length": len(value)}
                else:
                    result[var] = {"status": "not_set"}
            return JSONResponse(result)

        @self.app.post("/api/reconnect")
        async def reconnect_apis():
            """Manually trigger API reconnection."""
            self.state.api_status = []
            await self._check_api_connections()
            return {"status": "ok", "message": "Reconnection complete"}

        # Global exception handler
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.exception("Unhandled API error")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Internal error: {str(exc)}"},
            )

        # ===== NEW PROJECT =====
        @self.app.post("/api/project/start")
        async def start_project(data: dict):
            print(f"[API] START PROJECT endpoint called: {data.get('name', 'Untitled')}")
            logger.info(f"🚀 START PROJECT: {data.get('name', 'Untitled')}")

            try:
                print("[API] Validating prompt...")
                prompt = data.get("prompt", "").strip()
                if not prompt:
                    return {"status": "error", "message": "Prompt is required"}

                print("[API] Creating project object...")

                # Generate project ID if not provided
                project_id = data.get("project_id", "")
                if not project_id:
                    project_id = data.get("name", "project").lower().replace(" ", "-")[:30]

                # Generate output dir if not provided
                output_dir = data.get("output_dir", "")
                if not output_dir:
                    output_dir = f"./outputs/{project_id}"

                project = ActiveProject(
                    id=str(uuid.uuid4())[:8],
                    name=data.get("name", "Untitled Project"),
                    mode="new",
                    prompt=prompt,
                    criteria=data.get("criteria", ""),
                    project_type=data.get("project_type", "custom"),
                    budget=float(data.get("budget", 5.0)),
                    status="starting",
                    start_time=time.time(),
                    # Advanced settings
                    time_seconds=int(data.get("time_seconds", 4500)),
                    concurrency=int(data.get("concurrency", 3)),
                    project_id=project_id,
                    output_dir=output_dir,
                    quality_targets=data.get(
                        "quality_targets",
                        {
                            "code_generation": 0.88,
                            "code_review": 0.75,
                            "complex_reasoning": 0.88,
                            "evaluation": 0.80,
                        },
                    ),
                    policies=data.get(
                        "policies", [{"name": "no_training", "allow_training_on_output": False}]
                    ),
                )
                project.add_log("info", f"🚀 Starting: {project.name}")
                project.add_log("info", f"📁 Project ID: {project.project_id}")
                project.add_log("info", f"📂 Output: {project.output_dir}")

                print("[API] Adding to active projects...")
                self.state.active_projects.append(project)

                print("[API] Broadcasting state...")
                await self._broadcast_state()

                print("[API] Creating task...")
                project.task_handle = asyncio.create_task(self._execute_new_project(project))

                print("[API] Returning response")
                return {"status": "started", "project_id": project.id}

            except Exception as e:
                logger.exception("Failed to start project")
                return {"status": "error", "message": str(e)}

        # ===== IMPROVE CODEBASE =====
        @self.app.post("/api/project/improve")
        async def improve_project(data: dict):
            logger.info(f"🔧 IMPROVE: {data.get('codebase_path', '')}")

            try:
                path = data.get("codebase_path", "").strip()
                instructions = data.get("instructions", "").strip()

                if not path:
                    return {"status": "error", "message": "Codebase path is required"}
                if not instructions:
                    return {"status": "error", "message": "Instructions are required"}

                project = ActiveProject(
                    id=str(uuid.uuid4())[:8],
                    name=f"Improve: {Path(path).name}",
                    mode="improve",
                    codebase_path=path,
                    instructions=instructions,
                    prompt=instructions,
                    budget=float(data.get("budget", 3.0)),
                    status="starting",
                    start_time=time.time(),
                )
                project.add_log("info", f"🔧 Analyzing: {path}")

                self.state.active_projects.append(project)
                await self._broadcast_state()

                project.task_handle = asyncio.create_task(self._execute_improvement(project))

                return {"status": "started", "project_id": project.id}

            except Exception as e:
                logger.exception("Failed to start improvement")
                return {"status": "error", "message": str(e)}

        # ===== UPLOAD PROJECT =====
        @self.app.post("/api/project/upload")
        async def upload_project(
            file=File(...),
            instructions: str = Form(""),
            budget: float = Form(5.0),
        ):
            logger.info(f"📤 UPLOAD: {file.filename}")

            try:
                content = await file.read()
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

                with open(file_path, "wb") as f:
                    f.write(content)

                project = ActiveProject(
                    id=str(uuid.uuid4())[:8],
                    name=f"Upload: {file.filename}",
                    mode="upload",
                    file_path=str(file_path),
                    file_name=file.filename,
                    instructions=instructions,
                    prompt=instructions,
                    budget=budget,
                    status="starting",
                    start_time=time.time(),
                )
                project.add_log("info", f"📤 Uploaded: {file.filename}")

                self.state.active_projects.append(project)
                await self._broadcast_state()

                project.task_handle = asyncio.create_task(self._execute_upload(project))

                return {"status": "started", "project_id": project.id}

            except Exception as e:
                logger.exception("Upload failed")
                return {"status": "error", "message": str(e)}

        # ===== STOP PROJECT =====
        @self.app.post("/api/project/{project_id}/stop")
        async def stop_project(project_id: str):
            logger.info(f"⏹️ STOP PROJECT: {project_id}")

            project = next((p for p in self.state.active_projects if p.id == project_id), None)
            if not project:
                return {"status": "error", "message": "Project not found"}

            if project.task_handle and not project.task_handle.done():
                project.task_handle.cancel()

            project.status = "stopped"
            project.current_task = "Stopped"
            project.add_log("warning", "⏹️ Stopped by user")

            await self._broadcast_state()
            return {"status": "stopped"}

        # ===== REMOVE PROJECT =====
        @self.app.post("/api/project/{project_id}/remove")
        async def remove_project(project_id: str):
            logger.info(f"🗑️ REMOVE PROJECT: {project_id}")

            project = next((p for p in self.state.active_projects if p.id == project_id), None)
            if not project:
                return {"status": "error", "message": "Project not found"}

            # Only allow removing finished projects
            if project.status == "running":
                return {
                    "status": "error",
                    "message": "Cannot remove running project. Stop it first.",
                }

            self.state.active_projects.remove(project)
            await self._broadcast_state()
            return {"status": "removed"}

        # ===== CLEAR FINISHED =====
        @self.app.post("/api/projects/clear-finished")
        async def clear_finished():
            logger.info("🧹 CLEAR FINISHED PROJECTS")

            finished = [p for p in self.state.active_projects if p.status != "running"]
            for p in finished:
                self.state.active_projects.remove(p)

            await self._broadcast_state()
            return {"status": "ok", "removed": len(finished)}

        # ===== GET PROJECT LOGS =====
        @self.app.get("/api/project/{project_id}/logs")
        async def get_project_logs(project_id: str):
            project = next((p for p in self.state.active_projects if p.id == project_id), None)
            if not project:
                return JSONResponse({"error": "Project not found"}, status_code=404)
            return JSONResponse({"logs": project.logs})

        # ===== WEBSOCKET (DISABLED - Using HTTP polling instead) =====
        # WebSocket is disabled due to 403 Forbidden errors.
        # The dashboard uses HTTP polling as fallback.
        # @self.app.websocket("/ws")
        # async def websocket_endpoint(websocket: WebSocket):
        #     ...

    async def _execute_new_project(self, project: ActiveProject):
        """Execute new project with advanced settings."""
        try:
            project.status = "running"
            project.add_log("info", "🤖 Initializing orchestrator...")
            project.add_log("info", f"   Output dir: {project.output_dir}")
            project.add_log(
                "info",
                f"   Budget: ${project.budget}, Time: {project.time_seconds}s, Concurrency: {project.concurrency}",
            )
            await self._broadcast_state()

            # Create output directory
            import os

            os.makedirs(project.output_dir, exist_ok=True)
            project.add_log("info", f"📁 Output dir: {project.output_dir}")

            # Create budget with project settings
            budget = Budget(max_usd=project.budget, max_time_seconds=project.time_seconds)

            # Create orchestrator with concurrency setting
            project.orchestrator = Orchestrator(
                budget=budget,
                max_concurrency=project.concurrency,
            )
            self._setup_hooks(project)

            # Log quality targets
            qt = project.quality_targets
            project.add_log(
                "info",
                f"🎯 Quality targets: code_gen={qt.get('code_generation', 0.88)}, review={qt.get('code_review', 0.75)}",
            )

            await asyncio.sleep(0.5)

            project.add_log("info", "🎯 Executing project...")
            project.add_log("info", f"   Prompt: {project.prompt[:100]}...")
            project.current_task = "Decomposing project..."
            await self._broadcast_state()

            # Add heartbeat task to show orchestrator is alive
            heartbeat_task = asyncio.create_task(self._heartbeat(project))

            # Add timeout detection
            start_time = time.time()
            timeout_seconds = project.time_seconds

            try:
                project.add_log("info", "🚀 Calling orchestrator.run_project()...")
                project.add_log("info", f"   Prompt length: {len(project.prompt)} chars")
                project.add_log(
                    "info", f"   Criteria: {project.criteria or 'Complete implementation'}"
                )

                # Run with timeout
                import asyncio

                try:
                    project_state = await asyncio.wait_for(
                        project.orchestrator.run_project(
                            project_description=project.prompt,
                            success_criteria=project.criteria or "Complete implementation",
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    project.add_log("error", f"⏱️ Project timed out after {timeout_seconds}s")
                    project.status = "failed"
                    project.current_task = "Timed out"
                    await self._broadcast_state()
                    return

                elapsed = time.time() - start_time
                project.add_log(
                    "info", f"✅ Orchestrator returned in {elapsed:.1f}s: {project_state.status}"
                )
                project.add_log(
                    "info",
                    f"   Tasks completed: {len([t for t in project.tasks if t.status == 'completed'])}",
                )

            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            if project_state.status == ProjectStatus.SUCCESS:
                project.status = "completed"
                project.progress = 100
                project.current_task = "Completed"
                project.add_log("success", "✅ Project completed!")
            elif project_state.status == ProjectStatus.PARTIAL_SUCCESS:
                project.status = "completed"
                project.progress = 100
                project.current_task = "Partial success"
                # Calculate which tasks succeeded/failed
                success_count = sum(
                    1 for t in project.tasks if t.status == "completed" and t.score > 0
                )
                failed_count = len(project.tasks) - success_count
                avg_score = sum(t.score for t in project.tasks if t.score > 0) / max(
                    success_count, 1
                )
                project.add_log(
                    "warning",
                    f"⚠️ Partial success: {success_count}/{len(project.tasks)} tasks OK (avg score: {avg_score:.2f})",
                )
                if failed_count > 0:
                    project.add_log(
                        "info",
                        f"💡 {failed_count} task(s) failed or scored 0 - check output for details",
                    )
            else:
                project.status = "failed"
                project.current_task = "Failed"
                project.add_log("error", f"❌ Failed: {project_state.status}")

        except asyncio.CancelledError:
            project.status = "stopped"
            project.current_task = "Stopped"
            project.add_log("warning", "⏹️ Cancelled")
        except Exception as e:
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()
            project.status = "failed"
            project.current_task = "Error"
            project.add_log("error", f"❌ Error: {error_msg}")
            project.add_log("error", f"Traceback: {tb[:500]}...")  # First 500 chars
            logger.exception(f"Project execution failed: {error_msg}")
        finally:
            await self._broadcast_state()

    async def _execute_improvement(self, project: ActiveProject):
        """Execute codebase improvement."""
        try:
            project.status = "running"
            project.add_log("info", "🔍 Analyzing codebase...")
            project.current_task = "Analyzing structure..."
            await self._broadcast_state()

            analyzer = CodebaseAnalyzer()
            analysis = await analyzer.analyze(Path(project.codebase_path))

            project.add_log("info", f"📊 {analysis.total_files} files found")
            project.add_log("info", f"💻 Languages: {', '.join(analysis.files_by_language.keys())}")

            project.current_task = "Generating improvements..."
            await self._broadcast_state()

            suggester = ImprovementSuggester()
            improvements = suggester.suggest(analysis)

            for imp in improvements[:5]:
                project.add_log("info", f"💡 {imp.title} ({imp.priority})")

            project.current_task = "Applying improvements..."
            await self._broadcast_state()

            budget = Budget(max_usd=project.budget, max_time_seconds=1800)
            project.orchestrator = Orchestrator(
                budget=budget,
                max_concurrency=project.concurrency,
            )
            self._setup_hooks(project)

            prompt = f"""Improve this codebase:

Path: {project.codebase_path}
Files: {analysis.total_files}
Languages: {', '.join(analysis.files_by_language.keys())}

Improvements:
{chr(10).join([f'- {i.title}' for i in improvements[:5]])}

Instructions: {project.instructions}
"""

            # Add heartbeat task to show orchestrator is alive
            heartbeat_task = asyncio.create_task(self._heartbeat(project))

            # Add timeout detection
            start_time = time.time()
            timeout_seconds = project.time_seconds

            try:
                project.add_log("info", "🚀 Calling orchestrator.run_project()...")

                try:
                    project_state = await asyncio.wait_for(
                        project.orchestrator.run_project(
                            project_description=prompt,
                            success_criteria="All improvements applied, tests pass",
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    project.add_log("error", f"⏱️ Project timed out after {timeout_seconds}s")
                    project.status = "failed"
                    project.current_task = "Timed out"
                    await self._broadcast_state()
                    return

                elapsed = time.time() - start_time
                project.add_log(
                    "info", f"✅ Orchestrator returned in {elapsed:.1f}s: {project_state.status}"
                )
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            if project_state.status == ProjectStatus.SUCCESS:
                project.status = "completed"
                project.progress = 100
                project.add_log("success", "✅ Improvements applied!")
            elif project_state.status == ProjectStatus.PARTIAL_SUCCESS:
                project.status = "completed"
                project.progress = 100
                success_count = sum(
                    1 for t in project.tasks if t.status == "completed" and t.score > 0
                )
                project.add_log(
                    "warning",
                    f"⚠️ Partial improvements: {success_count}/{len(project.tasks)} tasks OK",
                )
            else:
                project.status = "failed"
                project.add_log("error", "❌ Failed")

        except asyncio.CancelledError:
            project.status = "stopped"
            project.current_task = "Stopped"
            project.add_log("warning", "⏹️ Cancelled")
        except Exception as e:
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()
            project.status = "failed"
            project.current_task = "Error"
            project.add_log("error", f"❌ Error: {error_msg}")
            project.add_log("error", f"Traceback: {tb[:500]}...")  # First 500 chars
            logger.exception(f"Project execution failed: {error_msg}")
        finally:
            await self._broadcast_state()

    async def _execute_upload(self, project: ActiveProject):
        """Execute uploaded project specification file (YAML/JSON/MD/TXT)."""
        try:
            project.status = "running"
            path = Path(project.file_path)

            # Read the spec file
            content = path.read_text(encoding="utf-8", errors="ignore")
            project.add_log("info", f"📄 Read {len(content)} characters from {project.file_name}")

            # Parse YAML if possible
            spec_data = None
            if path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml

                    spec_data = yaml.safe_load(content)
                    project.add_log("info", "✅ Parsed YAML specification")
                except ImportError:
                    project.add_log("warning", "⚠️ PyYAML not installed, treating as text")
                except Exception as e:
                    project.add_log("warning", f"⚠️ Failed to parse YAML: {e}")
            elif path.suffix == ".json":
                try:
                    import json

                    spec_data = json.loads(content)
                    project.add_log("info", "✅ Parsed JSON specification")
                except Exception as e:
                    project.add_log("warning", f"⚠️ Failed to parse JSON: {e}")

            # Build prompt from spec
            if spec_data and isinstance(spec_data, dict):
                # Extract fields from YAML/JSON spec
                project_name = spec_data.get("name", spec_data.get("project_name", "Untitled"))
                description = spec_data.get("description", spec_data.get("prompt", ""))
                requirements = spec_data.get("requirements", spec_data.get("specs", ""))
                tech_stack = spec_data.get("tech_stack", spec_data.get("technologies", ""))
                spec_data.get("success_criteria", spec_data.get("criteria", ""))

                prompt_parts = [f"# Project: {project_name}", ""]

                if description:
                    prompt_parts.extend(["## Description", description, ""])
                if requirements:
                    prompt_parts.extend(["## Requirements", str(requirements), ""])
                if tech_stack:
                    prompt_parts.extend(["## Tech Stack", str(tech_stack), ""])
                if spec_data.get("features"):
                    prompt_parts.extend(["## Features", str(spec_data["features"]), ""])
                if spec_data.get("architecture"):
                    prompt_parts.extend(["## Architecture", str(spec_data["architecture"]), ""])
                if project.instructions:
                    prompt_parts.extend(["## Additional Instructions", project.instructions, ""])

                prompt = "\n".join(prompt_parts)
                project.add_log("info", f"📝 Built prompt from spec: {project_name}")
            else:
                # Treat as plain text/markdown
                prompt = f"""# Project Specification

File: {project.file_name}

## Specification Content
{content[:8000]}{'...' if len(content) > 8000 else ''}

## Additional Instructions
{project.instructions}
"""
                project.add_log("info", "📝 Using raw file content as prompt")

            project.current_task = "Processing specification..."
            await self._broadcast_state()

            # Extract budget and config from YAML spec with fallbacks to project/UI values
            if spec_data and isinstance(spec_data, dict):
                # Budget USD: try budget_usd, budget.max_usd, or project.budget (from UI)
                budget_usd = spec_data.get("budget_usd")
                if budget_usd is None and isinstance(spec_data.get("budget"), dict):
                    budget_usd = spec_data["budget"].get("max_usd")
                if budget_usd is None:
                    budget_usd = project.budget  # Fallback to UI value (default 5.0)

                # Time seconds: try time_seconds, budget.max_time_seconds, or default 3600
                time_seconds = spec_data.get("time_seconds")
                if time_seconds is None and isinstance(spec_data.get("budget"), dict):
                    time_seconds = spec_data["budget"].get("max_time_seconds")
                if time_seconds is None:
                    time_seconds = 3600  # Default 1 hour

                # Concurrency: try concurrency, max_concurrency, or default 3
                concurrency = spec_data.get("concurrency")
                if concurrency is None:
                    concurrency = spec_data.get("max_concurrency", project.concurrency)

                # Project ID: try project_id or generate one
                yaml_project_id = spec_data.get("project_id", "")
                if yaml_project_id:
                    project.project_id = yaml_project_id

                project.add_log(
                    "info",
                    f"⚙️ Config from spec: budget=${budget_usd}, time={time_seconds}s, concurrency={concurrency}",
                )
            else:
                # Fallback to project values from UI
                budget_usd = project.budget
                time_seconds = project.time_seconds
                concurrency = project.concurrency

            budget = Budget(max_usd=budget_usd, max_time_seconds=time_seconds)
            project.orchestrator = Orchestrator(budget=budget, max_concurrency=concurrency)
            self._setup_hooks(project)

            # Use success_criteria from spec if available
            final_success_criteria = "Complete implementation according to specification"
            if spec_data and isinstance(spec_data, dict):
                final_success_criteria = spec_data.get(
                    "success_criteria", spec_data.get("criteria", final_success_criteria)
                )

            # Add heartbeat task to show orchestrator is alive
            heartbeat_task = asyncio.create_task(self._heartbeat(project))

            # Add timeout detection
            start_time = time.time()
            timeout_seconds = time_seconds  # Use the value from YAML or default

            try:
                project.add_log("info", "🚀 Calling orchestrator.run_project()...")
                project.add_log("info", f"   Prompt length: {len(prompt)} chars")

                # Run with timeout from spec
                try:
                    project_state = await asyncio.wait_for(
                        project.orchestrator.run_project(
                            project_description=prompt,
                            success_criteria=final_success_criteria,
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    project.add_log("error", f"⏱️ Project timed out after {timeout_seconds}s")
                    project.status = "failed"
                    project.current_task = "Timed out"
                    await self._broadcast_state()
                    return

                elapsed = time.time() - start_time
                project.add_log(
                    "info", f"✅ Orchestrator returned in {elapsed:.1f}s: {project_state.status}"
                )

                if project_state.status == ProjectStatus.SUCCESS:
                    project.status = "completed"
                    project.progress = 100
                    project.add_log("success", "✅ Project completed from spec!")
                elif project_state.status == ProjectStatus.PARTIAL_SUCCESS:
                    project.status = "completed"
                    project.progress = 100
                    success_count = sum(
                        1 for t in project.tasks if t.status == "completed" and t.score > 0
                    )
                    failed_count = len(project.tasks) - success_count
                    avg_score = sum(t.score for t in project.tasks if t.score > 0) / max(
                        success_count, 1
                    )
                    project.add_log(
                        "warning",
                        f"⚠️ Partial success: {success_count}/{len(project.tasks)} tasks OK (avg: {avg_score:.2f})",
                    )
                    if failed_count > 0:
                        project.add_log(
                            "info", f"💡 {failed_count} task(s) failed - check output for details"
                        )
                else:
                    project.status = "failed"
                    project.add_log("error", f"❌ Failed: {project_state.status}")
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            project.status = "stopped"
            project.current_task = "Stopped"
            project.add_log("warning", "⏹️ Cancelled")
        except Exception as e:
            import traceback

            error_msg = str(e)
            tb = traceback.format_exc()
            project.status = "failed"
            project.current_task = "Error"
            project.add_log("error", f"❌ Error: {error_msg}")
            project.add_log("error", f"Traceback: {tb[:500]}...")  # First 500 chars
            logger.exception(f"Project execution failed: {error_msg}")
        finally:
            await self._broadcast_state()

    def _setup_hooks(self, project: ActiveProject):
        """Setup orchestrator hooks for a project."""
        if not project.orchestrator:
            logger.warning("No orchestrator to setup hooks")
            return

        try:
            orchestrator = project.orchestrator

            # Check if add_hook method exists
            if not hasattr(orchestrator, "add_hook"):
                logger.warning(f"Orchestrator missing add_hook method. Type: {type(orchestrator)}")
                project.add_log("warning", "⚠️ Real-time updates not available")
                return

            def on_task_started(task_id: str, task: Task, **kwargs):
                try:
                    task_type = str(task.task_type) if hasattr(task, "task_type") else "task"
                    project.current_task = f"Running {task_type}..."
                    project.add_log("info", f"🎯 {task_id[:8]}: {task_type}")
                    task_info = TaskInfo(
                        task_id=task_id,
                        task_type=task_type,
                        status="running",
                        start_time=time.time(),
                    )
                    project.tasks.append(task_info)
                    total = max(project.total_tasks, len(project.tasks))
                    project.progress = int((len(project.tasks) - 1) / total * 100)
                    asyncio.create_task(self._broadcast_state())
                except Exception as e:
                    logger.warning(f"Hook error (task_started): {e}")

            orchestrator.add_hook(EventType.TASK_STARTED, on_task_started)

            def on_task_completed(task_id: str, result: TaskResult, **kwargs):
                try:
                    for task in project.tasks:
                        if task.task_id == task_id:
                            task.status = "completed"
                            task.progress = 100
                            if hasattr(result, "score"):
                                task.score = result.score
                            if hasattr(result, "model_used"):
                                task.model = str(result.model_used)
                            break
                    project.completed_tasks += 1
                    if hasattr(result, "cost"):
                        project.cost += result.cost
                    score_str = f" ({result.score:.2f})" if hasattr(result, "score") else ""
                    project.add_log("success", f"✅ {task_id[:8]}{score_str}")
                    total = max(project.total_tasks, len(project.tasks))
                    project.progress = min(int(project.completed_tasks / total * 100), 99)
                    asyncio.create_task(self._broadcast_state())
                except Exception as e:
                    logger.warning(f"Hook error (task_completed): {e}")

            orchestrator.add_hook(EventType.TASK_COMPLETED, on_task_completed)

            def on_model_selected(task_id: str, model: str, **kwargs):
                try:
                    for task in project.tasks:
                        if task.task_id == task_id:
                            task.model = model
                            break
                    project.add_log("info", f"🤖 {task_id[:8]} -> {model}")
                    asyncio.create_task(self._broadcast_state())
                except Exception as e:
                    logger.warning(f"Hook error (model_selected): {e}")

            orchestrator.add_hook(EventType.MODEL_SELECTED, on_model_selected)
            logger.info(f"✅ Hooks setup for project {project.id}")

        except Exception as e:
            logger.warning(f"Failed to setup hooks: {e}")
            project.add_log("warning", "⚠️ Real-time updates disabled")

    async def _heartbeat(self, project: ActiveProject, interval: int = 30):
        """Send periodic heartbeat logs to show orchestrator is alive."""
        last_task_count = 0
        stuck_counter = 0

        try:
            while True:
                await asyncio.sleep(interval)
                if project.status == "running":
                    elapsed = int(time.time() - project.start_time)
                    current_tasks = len(project.tasks)

                    # Detect if orchestrator might be stuck
                    if current_tasks == last_task_count and current_tasks > 0:
                        stuck_counter += 1
                        if stuck_counter >= 2:  # 60 seconds without new tasks
                            project.add_log(
                                "warning",
                                f"⚠️ No new tasks for {stuck_counter * interval}s - orchestrator may be stuck",
                            )
                    else:
                        stuck_counter = 0

                    last_task_count = current_tasks

                    project.add_log(
                        "info",
                        f"💓 Orchestrator running... ({elapsed}s elapsed, {project.completed_tasks}/{current_tasks} tasks, current: {project.current_task})",
                    )
                    await self._broadcast_state()
        except asyncio.CancelledError:
            project.add_log("info", "🛑 Heartbeat stopped")
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")

    async def _broadcast_state(self):
        """Broadcast to all clients."""
        if not self.active_connections:
            return

        message = {"type": "state", "data": self.state.to_dict()}
        disconnected = []

        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            if ws in self.active_connections:
                self.active_connections.remove(ws)

    async def run(self):
        from uvicorn import Config, Server

        print("[Server.run] Setting up FastAPI...")
        await self._setup_fastapi()
        print("[Server.run] FastAPI setup complete")

        print("[Server.run] Checking API connections...")
        # Check API connections before starting server
        await self._check_api_connections()
        print("[Server.run] API check complete")

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)

        logger.info(f"🚀 LLM Orchestrator v6.5.22 on http://{self.host}:{self.port}")
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🚀 LLM Orchestrator v6.5.22 - Project Dashboard                   ║
╠══════════════════════════════════════════════════════════════════╣
║  🌐 URL: http://{self.host}:{self.port:<45}║
╠══════════════════════════════════════════════════════════════════╣
║  ✨ Features:                                                    ║
║     • ✅ Shows "Connected" when online                          ║
║     • ✅ Multiple projects can run simultaneously               ║
║     • ✅ New + Improve + Upload all at once!                    ║
║     • ✅ Real-time API Status Panel                             ║
║     • ✅ Live Console per project                               ║
╚══════════════════════════════════════════════════════════════════╝
        """)

        await server.serve()

    def _get_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Orchestrator v6.5.22</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        /* ═══════════════════════════════════════════════════════════════════════
           TECHNICAL DARK - REFINED DEPTH SYSTEM
           Base: #0B1220 | Cards: #131C2E | Elevated: #182235 | Inputs: #0F1728
           ═══════════════════════════════════════════════════════════════════════ */

        /* ─── Base Layer ─── */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0B1220;
            color: #E5ECF6;
            font-size: 14px;
            line-height: 1.5;
        }

        /* ─── Header: Solid Deep Navy with Accent Strip ─── */
        .gradient-header {
            background: #111827;
            position: relative;
            padding: 24px 24px 20px;
        }
        .gradient-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        }
        .gradient-header h1 {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .gradient-header .header-icon {
            font-size: 24px;
            margin-right: 8px;
            opacity: 0.9;
        }

        /* ─── Cards & Surfaces ─── */
        .panel {
            background: #131C2E;
            border-radius: 12px;
            border: 1px solid #23314A;
            transition: border-color 150ms ease, box-shadow 150ms ease;
        }
        .panel:hover {
            border-color: #2A3A56;
        }

        /* Right panel - elevated surface */
        .right-panel {
            background: #182235;
            border: 1px solid #23314A;
            border-top: 1px solid #2A3A56;
        }
        .right-panel .panel-header {
            border-bottom: 1px solid #23314A;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }

        /* ─── Typography Scale ─── */
        h1 { font-size: 28px; font-weight: 700; }
        h2 { font-size: 18px; font-weight: 600; }
        h3 { font-size: 16px; font-weight: 600; }
        .section-title {
            font-size: 13px;
            font-weight: 500;
            color: #8FA3C8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .field-label {
            font-size: 13px;
            font-weight: 500;
            color: #A0B0CC;
        }

        /* ─── Form Inputs ─── */
        .input-field {
            background: #0C1424;
            border: 1px solid #24324A;
            color: #E5ECF6;
            border-radius: 8px;
            padding: 11px 14px;
            width: 100%;
            height: 44px;
            font-size: 14px;
            transition: border-color 150ms ease, background 150ms ease;
        }
        .input-field::placeholder { color: #6B7A95; }
        .input-field:hover { border-color: #2F4560; }
        .input-field:focus {
            outline: none;
            border-color: #4F8CFF;
            background: #0F1728;
        }
        textarea.input-field { height: auto; min-height: 88px; padding: 12px 14px; }
        select.input-field { cursor: pointer; }

        /* ─── Primary Accent: ONE color only ─── */
        .btn-primary {
            background: #4F8CFF;
            color: white;
            padding: 11px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            font-size: 14px;
            transition: background 150ms ease, transform 50ms ease;
        }
        .btn-primary:hover:not(:disabled) {
            background: #3C7BEA;
            transform: translateY(-1px);
        }
        .btn-primary:active:not(:disabled) { background: #2F66D8; }
        .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

        /* ─── Secondary Buttons ─── */
        .btn-secondary {
            background: #1E2A44;
            color: #B8C5DC;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid #2A3A56;
            font-size: 13px;
            transition: all 150ms ease;
        }
        .btn-secondary:hover {
            background: #253350;
            border-color: #344866;
            color: #E5ECF6;
        }

        .btn-danger {
            background: #EF4444;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            font-size: 13px;
            opacity: 0.85;
            transition: opacity 150ms ease;
        }
        .btn-danger:hover { opacity: 1; }

        /* ─── Navigation Tabs ─── */
        .tab-btn {
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            background: transparent;
            color: #8FA3C8;
            transition: all 150ms ease;
            position: relative;
            font-size: 14px;
        }
        .tab-btn:hover:not(.active) {
            color: #B8C5DC;
            background: rgba(143, 163, 200, 0.08);
        }
        .tab-btn.active {
            background: #1E2A44;
            color: #FFFFFF;
        }
        .tab-btn.active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 8px;
            bottom: 8px;
            width: 3px;
            background: #4F8CFF;
            border-radius: 0 2px 2px 0;
        }
        .tab-btn i {
            color: #6B7A95;
            transition: color 150ms ease;
        }
        .tab-btn.active i,
        .tab-btn:hover i {
            color: #8FA3C8;
        }

        /* ─── Progress Bars ─── */
        .progress-bar {
            background: #1E2A44;
            border-radius: 999px;
            height: 8px;
            min-height: 8px;
            overflow: hidden;
            display: block;
            width: 100%;
            border: 1px solid #2A3A56;
        }
        .progress-fill {
            background: linear-gradient(90deg, #4F8CFF, #22C55E);
            height: 100%;
            min-width: 0px;
            transition: width 0.5s ease;
            border-radius: 999px;
            display: block;
        }
        /* Always show at least a tiny bit of fill for visibility */
        .progress-fill[style*="width: 0%"],
        .progress-fill:not([style]) {
            min-width: 2px;
            background: #3A4A60;
        }

        /* ─── Console ─── */
        .console {
            background: #0A0F1C;
            border: 1px solid #1E2A44;
            border-radius: 8px;
            padding: 12px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 12px;
            height: 200px;
            overflow-y: auto;
            line-height: 1.6;
        }
        .console-line { margin: 2px 0; }
        .console-info { color: #6EE7FF; }
        .console-success { color: #86EFAC; }
        .console-warning { color: #FCD34D; }
        .console-error { color: #FCA5A5; }

        /* ─── Status Badges ─── */
        .status-badge {
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-connected { background: rgba(34, 197, 94, 0.15); color: #22C55E; }
        .status-available { background: rgba(79, 140, 255, 0.15); color: #4F8CFF; }
        .status-running { background: rgba(79, 140, 255, 0.2); color: #4F8CFF; animation: pulse 2s infinite; }
        .status-completed { background: rgba(34, 197, 94, 0.15); color: #22C55E; }
        .status-failed { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
        .status-starting { background: rgba(245, 158, 11, 0.15); color: #F59E0B; }
        .status-stopped { background: rgba(148, 163, 184, 0.15); color: #94A3B8; }
        .status-partial { background: rgba(245, 158, 11, 0.15); color: #F59E0B; }

        /* ─── API Status Icons ─── */
        .api-status-connected { color: #22C55E; }
        .api-status-available { color: #60A5FA; }
        .api-status-error { color: #EF4444; }
        .api-status-no_key { color: #F59E0B; }
        .api-status-invalid_key { color: #DC2626; }
        .api-status-no_credits { color: #EA580C; }

        /* ─── Provider Badges ─── */
        .provider-badge {
            background: #0C1424;
            border: 1px solid #23314A;
            border-radius: 8px;
            padding: 8px 12px;
            transition: all 150ms ease;
        }
        .provider-badge:hover {
            background: #131C2E;
            border-color: #2A3A56;
        }

        /* ─── Animations ─── */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in { animation: fade-in 0.25s ease-out; }

        /* ─── Drop Zone ─── */
        .drop-zone {
            border: 2px dashed #2A3A56;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: all 150ms ease;
            cursor: pointer;
            background: #0C1424;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #4F8CFF;
            background: rgba(79, 140, 255, 0.05);
        }
        .drop-zone i {
            color: #4F8CFF;
            opacity: 0.8;
        }

        /* ─── Project Cards ─── */
        .project-card {
            background: #0C1424;
            border: 1px solid #23314A;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 150ms ease;
        }
        .project-card:hover {
            border-color: #2A3A56;
            background: #0F1728;
        }
        .project-card.active {
            border-color: #4F8CFF;
            background: rgba(79, 140, 255, 0.08);
            box-shadow: 0 0 0 1px rgba(79, 140, 255, 0.3);
        }
        .project-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .indicator-running { background: #4F8CFF; animation: pulse 1.5s infinite; }
        .indicator-completed { background: #22C55E; }
        .indicator-failed { background: #EF4444; }
        .indicator-starting { background: #F59E0B; animation: pulse 1.5s infinite; }
        .indicator-stopped { background: #64748B; }

        /* ─── Form Overlays ─── */
        .form-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(11, 18, 32, 0.85);
            backdrop-filter: blur(4px);
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
        }
        .form-overlay.hidden { display: none; }
        .form-disabled { position: relative; pointer-events: none; opacity: 0.5; }

        .running-indicator {
            background: #182235;
            border: 1px solid #2A3A56;
            border-radius: 12px;
            padding: 24px 32px;
            text-align: center;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        }

        /* ─── Section Dividers ─── */
        .section-divider {
            border-top: 1px solid #23314A;
            margin: 20px 0;
            padding-top: 16px;
        }

        /* ─── Advanced Settings ─── */
        #advanced-settings {
            background: #0C1424;
            border: 1px solid #23314A;
            border-radius: 10px;
        }
        #advanced-settings .border-t {
            border-color: #23314A;
        }

        /* ─── Checkbox Styling ─── */
        input[type="checkbox"] {
            background: #0C1424;
            border: 1px solid #2A3A56;
            border-radius: 4px;
            width: 16px;
            height: 16px;
            cursor: pointer;
        }
        input[type="checkbox"]:checked {
            background: #4F8CFF;
            border-color: #4F8CFF;
        }

        /* ─── Scrollbars ─── */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0B1220;
        }
        ::-webkit-scrollbar-thumb {
            background: #2A3A56;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #344866;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="gradient-header text-center">
        <h1 class="font-bold"><span class="header-icon">🚀</span>LLM Orchestrator <span class="text-blue-400">v6.5.22</span></h1>
        <p class="mt-3 text-sm" style="color: #8FA3C8;">
            <span id="ws-status"><span style="color: #64748B;">●</span> Disconnected</span>
            <span class="mx-2" style="color: #2A3A56;">|</span>
            <span id="server-status-badge" class="status-badge status-starting">Starting...</span>
            <span id="projects-count" class="ml-2"></span>
        </p>
    </div>

    <div class="p-6 max-w-6xl mx-auto space-y-6">

        <!-- API STATUS PANEL -->
        <div class="panel p-4">
            <div class="flex justify-between items-center mb-3">
                <h2 class="text-lg font-bold"><i class="fas fa-plug mr-2"></i>API Connections</h2>
                <button onclick="reconnectAPIs()" class="btn-secondary text-sm">
                    <i class="fas fa-sync mr-1"></i>Reconnect
                </button>
            </div>
            <p id="models-panel" class="section-title text-xs mb-2">Models</p>
            <div id="api-status-list" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
                <p class="col-span-full text-center" style="color: #6B7A95;">Checking API connections...</p>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

            <!-- LEFT COLUMN: Forms -->
            <div class="lg:col-span-2 space-y-5">

                <!-- TABS -->
                <div class="flex gap-1 justify-center flex-wrap p-1" style="background: #0C1424; border-radius: 10px; border: 1px solid #1E2A44;">
                    <button onclick="setMode('new')" id="tab-new" class="tab-btn active">
                        <i class="fas fa-plus mr-2"></i>New Project
                    </button>
                    <button onclick="setMode('improve')" id="tab-improve" class="tab-btn">
                        <i class="fas fa-wrench mr-2"></i>Improve
                    </button>
                    <button onclick="setMode('upload')" id="tab-upload" class="tab-btn">
                        <i class="fas fa-file-code mr-2"></i>Upload
                    </button>
                </div>

                <!-- NEW PROJECT -->
                <div id="form-new" class="panel p-6" style="position: relative;">
                    <!-- Running Overlay -->
                    <div id="form-new-overlay" class="form-overlay hidden">
                        <div class="running-indicator">
                            <i class="fas fa-spinner fa-spin text-2xl text-blue-400 mb-2"></i>
                            <p class="font-semibold">Project Running...</p>
                            <p style="font-size: 13px; color: #6B7A95;">Please wait for completion</p>
                        </div>
                    </div>
                    <h2 style="font-size: 18px; font-weight: 600; margin-bottom: 20px;"><i class="fas fa-rocket mr-2" style="color: #6B7A95;"></i>Start New Project</h2>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div>
                            <label class="field-label block mb-2">Project Name</label>
                            <input type="text" id="new-name" class="input-field" placeholder="My Project" value="My Project">
                        </div>
                        <div>
                            <label id="project-type" class="field-label block mb-2">Project Type</label>
                            <select id="new-type" class="input-field">
                                <option value="custom" selected>Custom (Generic)</option>
                                <option value="python">Python Project</option>
                                <option value="fullstack">Full-Stack Application</option>
                                <option value="frontend">Front-End Only</option>
                                <option value="backend">Back-End API</option>
                                <option value="mobile">Mobile App</option>
                                <option value="wordpress">WordPress Plugin</option>
                                <option value="ai_agent">AI Agent</option>
                                <option value="cli">CLI Tool</option>
                            </select>
                        </div>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Prompt <span style="color: #EF4444;">*</span></label>
                        <textarea id="new-prompt" class="input-field" rows="4" placeholder="Describe what you want to build..."></textarea>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Success Criteria</label>
                        <textarea id="new-criteria" class="input-field" rows="2" placeholder="What defines success?"></textarea>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div>
                            <label class="field-label block mb-2">Budget (USD)</label>
                            <input type="number" id="new-budget" class="input-field" value="2.0" min="0.5" max="20" step="0.5">
                        </div>
                        <div>
                            <label class="field-label block mb-2">Time Limit (sec)</label>
                            <input type="number" id="new-time-seconds" class="input-field" value="4500" min="300" max="36000" step="300">
                        </div>
                        <div>
                            <label class="field-label block mb-2">Concurrency</label>
                            <input type="number" id="new-concurrency" class="input-field" value="3" min="1" max="10" step="1">
                        </div>
                    </div>

                    <!-- Advanced Settings Toggle -->
                    <div class="mb-4">
                        <button type="button" onclick="toggleAdvancedSettings()" class="btn-secondary text-sm" style="background: transparent; border: none; padding: 0;">
                            <i class="fas fa-cog mr-1" style="color: #4F8CFF;"></i>
                            <span id="advanced-toggle-text" style="color: #8FA3C8;">Show Advanced Settings</span>
                            <i class="fas fa-chevron-down ml-1" id="advanced-toggle-icon" style="color: #6B7A95;"></i>
                        </button>
                    </div>

                    <!-- Advanced Settings Panel -->
                    <div id="advanced-settings" class="hidden p-4 mb-4 space-y-4" style="background: #0C1424; border: 1px solid #23314A; border-radius: 10px;">
                        <h4 class="section-title mb-3">Advanced Configuration</h4>

                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label class="field-label block mb-2">Project ID</label>
                                <input type="text" id="new-project-id" class="input-field" placeholder="auto-generated">
                            </div>
                            <div>
                                <label class="field-label block mb-2">Output Directory</label>
                                <input type="text" id="new-output-dir" class="input-field" value="./outputs">
                            </div>
                        </div>

                        <div style="border-top: 1px solid #23314A; padding-top: 16px; margin-top: 16px;">
                            <label class="field-label block mb-2">Quality Targets (0.0 - 1.0)</label>
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div>
                                    <label style="font-size: 11px; color: #6B7A95;">Code Gen</label>
                                    <input type="number" id="qt-code-gen" class="input-field" value="0.88" min="0" max="1" step="0.01">
                                </div>
                                <div>
                                    <label style="font-size: 11px; color: #6B7A95;">Code Review</label>
                                    <input type="number" id="qt-code-review" class="input-field" value="0.75" min="0" max="1" step="0.01">
                                </div>
                                <div>
                                    <label style="font-size: 11px; color: #6B7A95;">Reasoning</label>
                                    <input type="number" id="qt-reasoning" class="input-field" value="0.88" min="0" max="1" step="0.01">
                                </div>
                                <div>
                                    <label style="font-size: 11px; color: #6B7A95;">Evaluation</label>
                                    <input type="number" id="qt-evaluation" class="input-field" value="0.80" min="0" max="1" step="0.01">
                                </div>
                            </div>
                        </div>

                        <div style="border-top: 1px solid #23314A; padding-top: 16px; margin-top: 16px;">
                            <label class="field-label block mb-2">Policies</label>
                            <div class="flex items-center gap-3">
                                <input type="checkbox" id="policy-no-training" class="w-4 h-4 rounded" style="border: 1px solid #2A3A56; background: #0C1424;" checked>
                                <label for="policy-no-training" style="font-size: 13px; color: #A0B0CC;">No training on output</label>
                            </div>
                        </div>
                    </div>

                    <button onclick="startNewProject()" id="btn-start-new" class="btn-primary">
                        <i class="fas fa-play mr-2"></i>Start Project
                    </button>
                </div>

                <!-- IMPROVE CODEBASE -->
                <div id="form-improve" class="panel p-6 hidden" style="position: relative;">
                    <!-- Running Overlay -->
                    <div id="form-improve-overlay" class="form-overlay hidden">
                        <div class="running-indicator">
                            <i class="fas fa-spinner fa-spin text-2xl text-blue-400 mb-2"></i>
                            <p class="font-semibold">Improvement Running...</p>
                            <p style="font-size: 13px; color: #6B7A95;">Please wait for completion</p>
                        </div>
                    </div>
                    <h2 style="font-size: 18px; font-weight: 600; margin-bottom: 20px;"><i class="fas fa-wrench mr-2" style="color: #6B7A95;"></i>Improve Existing Codebase</h2>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Codebase Path <span style="color: #EF4444;">*</span></label>
                        <div class="flex gap-2">
                            <input type="text" id="improve-path" class="input-field flex-1" placeholder="C:\\Projects\\my-app or /home/user/projects/my-app">
                            <button type="button" onclick="browseFolder()" class="btn-secondary whitespace-nowrap">
                                <i class="fas fa-folder-open mr-2"></i>Browse
                            </button>
                        </div>
                        <input type="file" id="folder-picker" class="hidden" webkitdirectory directory onchange="handleFolderSelect(event)">
                        <p style="font-size: 11px; color: #6B7A95; margin-top: 6px;">Select project folder using Windows Explorer</p>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Improvement Instructions <span style="color: #EF4444;">*</span></label>
                        <textarea id="improve-instructions" class="input-field" rows="4" placeholder="What improvements to make? e.g., Add tests, refactor to TypeScript, optimize performance..."></textarea>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Budget (USD)</label>
                        <input type="number" id="improve-budget" class="input-field" value="3.0" min="0.5" max="20" step="0.5">
                    </div>

                    <button onclick="startImprovement()" id="btn-start-improve" class="btn-primary">
                        <i class="fas fa-magic mr-2"></i>Analyze & Improve
                    </button>
                </div>

                <!-- UPLOAD PROJECT SPEC -->
                <div id="form-upload" class="panel p-6 hidden" style="position: relative;">
                    <!-- Running Overlay -->
                    <div id="form-upload-overlay" class="form-overlay hidden">
                        <div class="running-indicator">
                            <i class="fas fa-spinner fa-spin text-2xl text-blue-400 mb-2"></i>
                            <p class="font-semibold">Processing Spec...</p>
                            <p style="font-size: 13px; color: #6B7A95;">Please wait for completion</p>
                        </div>
                    </div>
                    <h2 style="font-size: 18px; font-weight: 600; margin-bottom: 12px;"><i class="fas fa-file-code mr-2" style="color: #6B7A95;"></i>Upload Project Specification</h2>
                    <p style="color: #8FA3C8; margin-bottom: 20px; font-size: 13px;">Upload a YAML file with complete project instructions, requirements, and specifications.</p>

                    <!-- YAML Format Help -->
                    <div class="mb-4 text-xs" style="background: #0C1424; border: 1px solid #23314A; border-radius: 8px; padding: 12px;">
                        <div class="flex items-center gap-2 cursor-pointer" onclick="document.getElementById('yaml-help').classList.toggle('hidden')" style="color: #A0B0CC;">
                            <i class="fas fa-info-circle" style="color: #4F8CFF;"></i>
                            <span style="font-weight: 500;">YAML Format Help (click to expand)</span>
                            <i class="fas fa-chevron-down ml-auto" style="color: #6B7A95;"></i>
                        </div>
                        <div id="yaml-help" class="hidden font-mono space-y-1" style="color: #8FA3C8;">
                            <p>name: "Project Name"</p>
                            <p>description: "What to build"</p>
                            <p>tech_stack:</p>
                            <p class="pl-4">- "Python/FastAPI"</p>
                            <p>requirements:</p>
                            <p class="pl-4">- "Feature 1"</p>
                            <p class="pl-4">- "Feature 2"</p>
                            <p>success_criteria: "Tests pass"</p>
                        </div>
                    </div>

                    <div id="drop-zone" class="drop-zone mb-4" onclick="document.getElementById('upload-file').click()">
                        <i class="fas fa-file-alt text-4xl text-blue-400 mb-2"></i>
                        <p style="color: #8FA3C8;">Drop your YAML spec file here or click to browse</p>
                        <p style="font-size: 11px; color: #6B7A95; margin-top: 4px;">Supports: .yaml, .yml, .json, .md, .txt</p>
                        <input type="file" id="upload-file" class="hidden" accept=".yaml,.yml,.json,.md,.txt" onchange="handleFileSelect(event)">
                    </div>

                    <div id="upload-preview" class="hidden mb-4 p-4 flex items-center gap-3" style="background: #0C1424; border: 1px solid #23314A; border-radius: 8px;">
                        <i class="fas fa-file text-2xl text-blue-400"></i>
                        <div class="flex-1">
                            <p id="upload-filename" class="font-medium"></p>
                            <p id="upload-size" style="font-size: 13px; color: #6B7A95;"></p>
                        </div>
                        <button onclick="clearUpload()" class="text-red-400 hover:text-red-300">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Additional Instructions (Optional)</label>
                        <textarea id="upload-instructions" class="input-field" rows="3" placeholder="Any additional instructions not covered in the YAML file..."></textarea>
                    </div>

                    <div class="mb-4">
                        <label class="field-label block mb-2">Budget (USD)</label>
                        <input type="number" id="upload-budget" class="input-field" value="5.0" min="0.5" max="20" step="0.5">
                    </div>

                    <button onclick="uploadAndRun()" id="btn-start-upload" class="btn-primary" disabled>
                        <i class="fas fa-rocket mr-2"></i>Start Project from Spec
                    </button>
                </div>

            </div>

            <!-- RIGHT COLUMN: Active Projects -->
            <div class="lg:col-span-1">
                <div class="panel p-4 right-panel">
                    <div class="flex justify-between items-center panel-header">
                        <h2><i class="fas fa-tasks mr-2" style="color: #6B7A95;"></i>Active Projects</h2>
                        <button onclick="clearFinished()" id="btn-clear-all" class="btn-secondary text-xs hidden" style="padding: 4px 10px;">
                            <i class="fas fa-broom mr-1"></i>Clear
                        </button>
                    </div>

                    <!-- Project List -->
                    <div id="projects-list" class="mb-4">
                        <p class="text-center text-sm" style="color: #6B7A95;">No active projects</p>
                    </div>

                    <!-- Selected Project Details -->
                    <div id="selected-project-panel" class="hidden" style="border-top: 1px solid #23314A; padding-top: 16px;">
                        <h3 style="font-weight: 600; margin-bottom: 12px; font-size: 15px;" id="selected-project-name">Project</h3>

                        <!-- Advanced Settings Info -->
                        <div id="project-settings-info" class="mb-3 p-3 text-xs hidden" style="background: #0C1424; border: 1px solid #23314A; border-radius: 8px;">
                            <div class="grid grid-cols-2 gap-2" style="color: #8FA3C8;">
                                <span>ID: <span id="info-project-id" style="color: #A0B0CC;">auto-generated</span></span>
                                <span>Output: <span id="info-output-dir" style="color: #A0B0CC;">./outputs/</span></span>
                                <span>Time: <span id="info-time" style="color: #A0B0CC;">4500</span>s</span>
                                <span>Concurrency: <span id="info-concurrency" style="color: #A0B0CC;">3</span></span>
                            </div>
                            <p style="font-size: 10px; color: #6B7A95; margin-top: 8px; font-style: italic;">* Default values - can be changed in form before starting</p>
                        </div>

                        <!-- Task Progress -->
                        <div class="mb-3" style="background: #0C1424; border: 1px solid #23314A; border-radius: 8px; padding: 12px;">
                            <div class="flex justify-between text-xs mb-2">
                                <span id="selected-current-task" style="color: #A0B0CC; font-weight: 500;">Initializing...</span>
                                <span id="selected-progress-text" style="color: #4F8CFF; font-weight: 600;">0%</span>
                            </div>
                            <div class="progress-bar" style="margin-bottom: 8px;">
                                <div id="selected-progress-fill" class="progress-fill" style="width: 0%"></div>
                            </div>
                            <div class="flex justify-between text-xs" style="color: #6B7A95;">
                                <span id="selected-tasks-stats">0 / 0 tasks</span>
                                <span id="selected-cost-stats">$0.00 / $0.00</span>
                            </div>
                        </div>

                        <div id="project-action-buttons">
                            <button onclick="stopSelectedProject()" id="btn-stop-project" class="btn-danger w-full">
                                <i class="fas fa-stop mr-2"></i>Stop Project
                            </button>
                            <button onclick="removeProject(selectedProjectId)" id="btn-remove-project" class="btn-danger w-full hidden" style="background: #64748b;">
                                <i class="fas fa-trash mr-2"></i>Remove Project
                            </button>
                        </div>

                        <div class="mt-4">
                            <h4 class="section-title mb-2" id="architecture-panel-title">Architecture</h4>
                            <p style="color: #A0B0CC; font-size: 12px; margin-bottom: 0;">Structural view: <span id="architecture-summary">Awaiting project data...</span></p>
                        </div>

                        <!-- Mini Console -->
                        <div class="mt-4">
                            <h4 class="section-title mb-2">CONSOLE</h4>
                            <div id="mini-console" class="console" style="height: 150px;">
                                <p style="color: #6B7A95;">Waiting...</p>
                            </div>
                        </div>
                    </div>

                    <!-- System Error Log -->
                    <div class="mt-4 pt-4" style="border-top: 1px solid #23314A;">
                        <div class="flex justify-between items-center mb-2">
                            <h4 class="section-title"><i class="fas fa-bug mr-1" style="color: #6B7A95;"></i>System Log</h4>
                            <button onclick="clearSystemLog()" style="font-size: 11px; color: #6B7A95;" onmouseover="this.style.color='#8FA3C8'" onmouseout="this.style.color='#6B7A95'">
                                <i class="fas fa-eraser"></i>
                            </button>
                        </div>
                        <div id="system-log" class="console" style="height: 100px;">
                            <p style="color: #4A5568; font-size: 11px;">System messages and API errors will appear here...</p>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        let ws = null;
        let state = null;
        let currentMode = 'new';
        let selectedFile = null;
        let selectedProjectId = null;
        let systemLogs = [];
        let lastUpdateTime = Date.now();

        // WebSocket (DISABLED - Using HTTP polling)
        // WebSocket causes 403 Forbidden errors, so we use HTTP polling only
        function connectWS() {
            // Mark as using HTTP mode
            document.getElementById('ws-status').innerHTML = '🌐 HTTP Mode';
            logSystem('info', 'Using HTTP polling mode');
            // Don't try to connect WebSocket
        }

        // Update UI
        function updateUI(data) {
            if (!data) return;
            state = data;
            lastUpdateTime = Date.now();

            // Server status badge
            const serverBadge = document.getElementById('server-status-badge');
            const wsStatus = document.getElementById('ws-status');
            const runningCount = data.projects_running || 0;

            if (data.server_status === 'connected') {
                if (runningCount > 0) {
                    serverBadge.textContent = `⚡ Running (${runningCount})`;
                    serverBadge.className = 'status-badge status-running';
                } else {
                    serverBadge.textContent = '🟢 Connected';
                    serverBadge.className = 'status-badge status-connected';
                }
            } else if (data.server_status === 'starting') {
                // If we have data but server says starting, we're in HTTP mode
                if (data.total_projects !== undefined) {
                    if (runningCount > 0) {
                        serverBadge.textContent = `⚡ Running (${runningCount})`;
                        serverBadge.className = 'status-badge status-running';
                    } else {
                        serverBadge.textContent = '🟢 Ready';
                        serverBadge.className = 'status-badge status-connected';
                    }
                } else {
                    serverBadge.textContent = '🟡 Starting...';
                    serverBadge.className = 'status-badge status-starting';
                }
            } else {
                serverBadge.textContent = '🔴 Error';
                serverBadge.className = 'status-badge status-failed';
            }

            // Projects count
            const countEl = document.getElementById('projects-count');
            if (data.total_projects > 0) {
                countEl.textContent = `| ${data.total_projects} projects (${data.projects_running} running)`;
            } else {
                countEl.textContent = '';
            }

            // Update API status
            if (data.api_status) {
                updateApiStatus(data.api_status);
            }

            // Update system logs
            if (data.system_logs && data.system_logs.length > 0) {
                updateSystemLogsFromServer(data.system_logs);
            }

            // Update projects list
            updateProjectsList(data.active_projects || []);

            // Manage form overlays based on running projects
            const activeProjects = data.active_projects || [];
            const newRunning = activeProjects.some(p => p.mode === 'new' && p.status === 'running');
            const improveRunning = activeProjects.some(p => p.mode === 'improve' && p.status === 'running');
            const uploadRunning = activeProjects.some(p => p.mode === 'upload' && p.status === 'running');

            if (newRunning) showFormOverlay('form-new');
            else hideFormOverlay('form-new');

            if (improveRunning) showFormOverlay('form-improve');
            else hideFormOverlay('form-improve');

            if (uploadRunning) showFormOverlay('form-upload');
            else hideFormOverlay('form-upload');
        }

        function updateApiStatus(apiStatus) {
            const container = document.getElementById('api-status-list');
            if (!apiStatus || apiStatus.length === 0) return;

            container.innerHTML = apiStatus.map(api => {
                // Determine icon and status styling
                let iconClass, statusText, statusColor;
                switch(api.status) {
                    case 'connected':
                        iconClass = 'fa-check-circle api-status-connected';
                        statusText = 'Connected';
                        break;
                    case 'available':
                        iconClass = 'fa-plug api-status-available';
                        statusText = 'Available';
                        break;
                    case 'no_key':
                        iconClass = 'fa-exclamation-circle api-status-no_key';
                        statusText = 'No API Key';
                        break;
                    case 'invalid_key':
                        iconClass = 'fa-key api-status-invalid_key';
                        statusText = 'Invalid Key';
                        break;
                    case 'no_credits':
                        iconClass = 'fa-coins api-status-no_credits';
                        statusText = 'No Credits';
                        break;
                    case 'checking':
                        iconClass = 'fa-spinner fa-spin';
                        statusText = 'Checking...';
                        break;
                    default:
                        iconClass = 'fa-times-circle api-status-error';
                        statusText = 'Error';
                }

                // Build tooltip with error/credit info
                let tooltip = api.provider;
                if (api.error) tooltip += ` - ${api.error}`;
                if (api.credits_info) tooltip += ` (${api.credits_info})`;
                if (api.response_time_ms > 0) tooltip += ` [${Math.round(api.response_time_ms)}ms]`;

                // Determine border color based on status
                let borderColor = '#23314A';
                if (api.status === 'connected') borderColor = '#22C55E40';
                else if (api.status === 'invalid_key' || api.status === 'no_credits') borderColor = '#DC262640';
                else if (api.status === 'error') borderColor = '#EF444440';

                return `
                    <div class="provider-badge flex items-center gap-2" style="border-color: ${borderColor};" title="${tooltip}">
                        <i class="fas ${iconClass}"></i>
                        <div>
                            <p style="font-weight: 500; font-size: 13px; text-transform: capitalize; color: #A0B0CC;">${api.provider}</p>
                            <p style="font-size: 11px; color: #6B7A95;">${statusText}</p>
                            ${api.credits_info ? `<p style="font-size: 10px; color: #22C55E;">${api.credits_info}</p>` : ''}
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Track previous project states for notifications
        let previousProjectStates = {};

        function updateProjectsList(projects) {
            console.log('updateProjectsList called with', projects.length, 'projects');
            if (projects.length > 0) {
                console.log('First project:', projects[0].id, 'progress:', projects[0].progress, 'status:', projects[0].status);
            }
            const container = document.getElementById('projects-list');
            const clearBtn = document.getElementById('btn-clear-all');

            // Check for completed/failed projects and send notifications
            projects.forEach(p => {
                const prev = previousProjectStates[p.id];
                if (prev && prev.status === 'running' && p.status !== 'running') {
                    if (p.status === 'completed') {
                        sendNotification('✅ Project Complete', `${p.name} finished successfully!`);
                        showSuccess('Project Complete', `${p.name} finished successfully!`);
                    } else if (p.status === 'failed') {
                        sendNotification('❌ Project Failed', `${p.name} encountered an error`);
                        showError('Project Failed', `${p.name} encountered an error`);
                    } else if (p.status === 'stopped') {
                        sendNotification('⏹️ Project Stopped', `${p.name} was stopped`);
                    }
                }
            });

            // Update previous states
            previousProjectStates = {};
            projects.forEach(p => {
                previousProjectStates[p.id] = { status: p.status, name: p.name };
            });

            // Check if there are any finished (non-running) projects
            const hasFinished = projects.some(p => p.status !== 'running');
            if (hasFinished) {
                clearBtn.classList.remove('hidden');
            } else {
                clearBtn.classList.add('hidden');
            }

            if (projects.length === 0) {
                container.innerHTML = '<p class="text-center text-sm" style="color: #6B7A95;">No active projects</p>';
                document.getElementById('selected-project-panel').classList.add('hidden');
                selectedProjectId = null;
                return;
            }

            // Sort by start time (newest first)
            const sorted = [...projects].sort((a, b) => b.elapsed - a.elapsed);

            container.innerHTML = sorted.map(p => {
                const indicatorClass = p.status === 'running' ? 'indicator-running' :
                                      p.status === 'completed' ? 'indicator-completed' :
                                      p.status === 'failed' ? 'indicator-failed' : 'indicator-starting';
                const isSelected = p.id === selectedProjectId;
                const isRunning = p.status === 'running';

                // Calculate task stats
                const tasksCompleted = p.tasks_completed || 0;
                const tasksTotal = p.tasks_total || 5;

                return `
                    <div class="project-card ${isSelected ? 'active' : ''}" onclick="selectProject('${p.id}')">
                        <div class="flex justify-between items-center">
                            <div class="flex items-center">
                                <span class="project-indicator ${indicatorClass}"></span>
                                <span style="font-weight: 500; font-size: 13px; color: #E5ECF6;" class="truncate" style="max-width: 120px;">${p.name}</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span style="font-size: 11px; color: #6B7A95;">${p.progress || 0}%</span>
                                ${!isRunning ? `<button onclick="event.stopPropagation(); removeProject('${p.id}')" style="color: #EF4444; opacity: 0.7; font-size: 11px;" onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.7'" title="Remove"><i class="fas fa-times"></i></button>` : ''}
                            </div>
                        </div>
                        <div style="margin-top: 8px;">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${p.progress || 0}%"></div>
                            </div>
                        </div>
                        <div style="margin-top: 6px; display: flex; justify-content: space-between; font-size: 11px; color: #6B7A95;">
                            <span>${tasksCompleted} / ${tasksTotal} tasks</span>
                            <span>$${(p.cost || 0).toFixed(2)}</span>
                        </div>
                    </div>
                `;
            }).join('');

            // Update selected project panel
            if (selectedProjectId) {
                const selected = projects.find(p => p.id === selectedProjectId);
                if (selected) {
                    updateSelectedProjectPanel(selected);
                } else {
                    // Selected project no longer exists
                    selectedProjectId = projects[0].id;
                    updateSelectedProjectPanel(projects[0]);
                }
            } else if (projects.length > 0) {
                selectedProjectId = projects[0].id;
                updateSelectedProjectPanel(projects[0]);
            }
        }

        function selectProject(projectId) {
            selectedProjectId = projectId;
            const project = state.active_projects.find(p => p.id === projectId);
            if (project) {
                updateSelectedProjectPanel(project);
            }
            // Re-render to update active state
            updateProjectsList(state.active_projects || []);
        }

        function updateSelectedProjectPanel(project) {
            console.log('updateSelectedProjectPanel:', project.id, 'progress:', project.progress, 'tasks:', project.tasks ? project.tasks.length : 0);
            document.getElementById('selected-project-panel').classList.remove('hidden');
            document.getElementById('selected-project-name').textContent = project.name;
            document.getElementById('selected-current-task').textContent = project.current_task || 'Working...';
            document.getElementById('selected-progress-text').textContent = (project.progress || 0) + '%';
            document.getElementById('selected-progress-fill').style.width = (project.progress || 0) + '%';
            // Force visibility
            document.getElementById('selected-progress-fill').style.display = 'block';
            // Calculate actual tasks count from project data dynamically
            const actualTotalTasks = project.tasks && project.tasks.length > 0 ? project.tasks.length : (project.tasks_total || 5);
            const completedCount = project.tasks ? project.tasks.filter(t => t.status === 'completed').length : (project.tasks_completed || 0);
            document.getElementById('selected-tasks-stats').textContent = `${completedCount} / ${actualTotalTasks} tasks`;
            document.getElementById('selected-cost-stats').textContent = `$${(project.cost || 0).toFixed(2)} / $${(project.budget || 0).toFixed(2)}`;

            // Show advanced settings - display actual values from project
            const settingsInfo = document.getElementById('project-settings-info');
            settingsInfo.classList.remove('hidden');

            // Show actual values or defaults with (editable) indicator
            const displayId = project.project_id && project.project_id !== '' ? project.project_id : 'auto-generated';
            const displayOutput = project.output_dir ? project.output_dir.replace('./outputs/', '') : 'project-name';
            const displayTime = project.time_seconds || 4500;
            const displayConcurrency = project.concurrency || 3;

            document.getElementById('info-project-id').textContent = displayId;
            document.getElementById('info-output-dir').textContent = './outputs/' + displayOutput;
            document.getElementById('info-time').textContent = displayTime + 's (editable)';
            document.getElementById('info-concurrency').textContent = displayConcurrency + ' (editable)';

            // Show note if using defaults
            if (!project.project_id && !project.time_seconds && !project.concurrency) {
                document.getElementById('info-project-id').innerHTML += ' <span class="text-yellow-500">*</span>';
            }

            // Show correct button based on status
            const isRunning = project.status === 'running';
            document.getElementById('btn-stop-project').classList.toggle('hidden', !isRunning);
            document.getElementById('btn-remove-project').classList.toggle('hidden', isRunning);

            // Update mini console
            fetchProjectLogs(project.id);
        }

        async function fetchProjectLogs(projectId) {
            if (!projectId) return;
            try {
                const response = await fetch(`/api/project/${projectId}/logs`);
                const data = await response.json();
                if (data.logs) {
                    updateMiniConsole(data.logs);
                }
            } catch (err) {
                console.error('Failed to fetch logs:', err);
            }
        }

        function updateMiniConsole(logs) {
            const consoleDiv = document.getElementById('mini-console');
            if (logs.length > 0) {
                consoleDiv.innerHTML = logs.slice(-50).map(log => {
                    const colorClass = 'console-' + log.level;
                    return `<div class="console-line ${colorClass}">[${log.time}] ${log.message}</div>`;
                }).join('');
                consoleDiv.scrollTop = consoleDiv.scrollHeight;
            }
        }

        // Form overlay control
        function showFormOverlay(formId) {
            const overlay = document.getElementById(formId + '-overlay');
            if (overlay) overlay.classList.remove('hidden');
            // Disable all inputs in this form
            const form = document.getElementById(formId);
            if (form) {
                form.querySelectorAll('input, textarea, button, select').forEach(el => {
                    el.disabled = true;
                });
            }
        }

        function hideFormOverlay(formId) {
            const overlay = document.getElementById(formId + '-overlay');
            if (overlay) overlay.classList.add('hidden');
            // Enable all inputs in this form
            const form = document.getElementById(formId);
            if (form) {
                form.querySelectorAll('input, textarea, button, select').forEach(el => {
                    el.disabled = false;
                });
            }
        }

        function hideAllOverlays() {
            ['form-new', 'form-improve', 'form-upload'].forEach(hideFormOverlay);
        }

        async function reconnectAPIs() {
            const btn = document.querySelector('button[onclick="reconnectAPIs()"]');
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-1"></i>Checking...';

            try {
                await fetch('/api/reconnect', {method: 'POST'});
            } catch (err) {
                console.error('Reconnect failed:', err);
            }

            setTimeout(() => {
                btn.innerHTML = '<i class="fas fa-sync mr-1"></i>Reconnect';
            }, 2000);
        }

        // Folder browser for Improve Codebase
        function browseFolder() {
            document.getElementById('folder-picker').click();
        }

        function handleFolderSelect(event) {
            const files = event.target.files;
            if (files.length > 0) {
                // Get the full path from webkitRelativePath or construct it
                const relativePath = files[0].webkitRelativePath || files[0].name;
                const folderName = relativePath.split('/')[0];
                // For Windows, we need to handle this differently
                // The webkitdirectory gives us files with webkitRelativePath
                // We'll use the path up to the folder name
                document.getElementById('improve-path').value = folderName;
                logSystem('info', 'Selected folder: ' + folderName);
            }
        }

        // Toggle advanced settings
        function toggleAdvancedSettings() {
            const panel = document.getElementById('advanced-settings');
            const text = document.getElementById('advanced-toggle-text');
            const icon = document.getElementById('advanced-toggle-icon');

            if (panel.classList.contains('hidden')) {
                panel.classList.remove('hidden');
                text.textContent = 'Hide Advanced Settings';
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            } else {
                panel.classList.add('hidden');
                text.textContent = 'Show Advanced Settings';
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            }
        }

        // Mode switching
        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('tab-' + mode).classList.add('active');
            showCurrentForm();
        }

        function showCurrentForm() {
            document.getElementById('form-new').classList.add('hidden');
            document.getElementById('form-improve').classList.add('hidden');
            document.getElementById('form-upload').classList.add('hidden');
            document.getElementById('form-' + currentMode).classList.remove('hidden');
        }

        // File handling
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('upload-filename').textContent = file.name;
                document.getElementById('upload-size').textContent = formatBytes(file.size);
                document.getElementById('upload-preview').classList.remove('hidden');
                document.getElementById('drop-zone').classList.add('hidden');
                document.getElementById('btn-start-upload').disabled = false;
            }
        }

        function clearUpload() {
            selectedFile = null;
            document.getElementById('upload-preview').classList.add('hidden');
            document.getElementById('drop-zone').classList.remove('hidden');
            document.getElementById('btn-start-upload').disabled = true;
            document.getElementById('upload-file').value = '';
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // Drag and drop
        const dropZone = document.getElementById('drop-zone');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) {
                document.getElementById('upload-file').files = files;
                handleFileSelect({target: {files: files}});
            }
        });

        // System logging
        function logSystem(level, message) {
            const time = new Date().toLocaleTimeString();
            systemLogs.push({time, level, message});
            if (systemLogs.length > 50) systemLogs.shift();

            const logDiv = document.getElementById('system-log');
            const colorClass = level === 'error' ? 'console-error' :
                              level === 'warn' ? 'console-warning' : 'console-info';

            logDiv.innerHTML = systemLogs.map(log =>
                `<div class="console-line ${log.level === 'error' ? 'console-error' : log.level === 'warn' ? 'console-warning' : 'console-info'}">[${log.time}] ${log.message}</div>`
            ).join('');
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function clearSystemLog() {
            systemLogs = [];
            document.getElementById('system-log').innerHTML = '<p style="color: #4A5568; font-size: 11px;">System messages and API errors will appear here...</p>';
        }

        // Update system logs from server state
        function updateSystemLogsFromServer(serverLogs) {
            // Merge server logs with local logs, avoiding duplicates
            const existingKeys = new Set(systemLogs.map(l => `${l.time}-${l.message}`));
            let newLogsAdded = false;

            for (const log of serverLogs) {
                const key = `${log.time}-${log.message}`;
                if (!existingKeys.has(key)) {
                    systemLogs.push(log);
                    newLogsAdded = true;
                }
            }

            if (newLogsAdded) {
                // Keep only last 50
                if (systemLogs.length > 50) {
                    systemLogs = systemLogs.slice(-50);
                }

                // Update display
                const logDiv = document.getElementById('system-log');
                if (systemLogs.length === 0) {
                    logDiv.innerHTML = '<p style="color: #4A5568; font-size: 11px;">System messages and API errors will appear here...</p>';
                } else {
                    logDiv.innerHTML = systemLogs.map(log =>
                        `<div class="console-line ${log.level === 'error' ? 'console-error' : log.level === 'warn' ? 'console-warning' : 'console-info'}">[${log.time}] ${log.message}</div>`
                    ).join('');
                    logDiv.scrollTop = logDiv.scrollHeight;
                }
            }
        }

        // API helper with error logging
        async function apiPost(url, body, isJson = true) {
            try {
                const options = {
                    method: 'POST',
                    headers: isJson ? {'Content-Type': 'application/json'} : {},
                    body: isJson ? JSON.stringify(body) : body
                };

                logSystem('info', `→ ${url}`);
                const response = await fetch(url, options);
                const text = await response.text();

                // Try to parse as JSON
                let result;
                try {
                    result = JSON.parse(text);
                } catch (e) {
                    // Not valid JSON - log the raw error
                    const errorMsg = 'Server error (not JSON): ' + text.substring(0, 100);
                    logSystem('error', errorMsg);
                    console.error('API Error (not JSON):', text);
                    throw new Error(errorMsg);
                }

                // Log API errors to console
                if (result.status === 'error') {
                    logSystem('error', `API Error: ${result.message}`);
                    console.error('API Error:', result.message);
                } else {
                    logSystem('info', `✓ ${url} success`);
                }

                return result;
            } catch (err) {
                logSystem('error', `Request failed: ${err.message}`);
                console.error('API Request Failed:', err);
                throw err;
            }
        }

        // Start new project
        async function startNewProject() {
            const prompt = document.getElementById('new-prompt').value.trim();
            if (!prompt) { alert('Please enter a prompt'); return; }

            const btn = document.getElementById('btn-start-new');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Starting...';

            try {
                // Build quality targets
                const qualityTargets = {
                    code_generation: parseFloat(document.getElementById('qt-code-gen').value),
                    code_review: parseFloat(document.getElementById('qt-code-review').value),
                    complex_reasoning: parseFloat(document.getElementById('qt-reasoning').value),
                    evaluation: parseFloat(document.getElementById('qt-evaluation').value)
                };

                // Build policies
                const policies = [];
                if (document.getElementById('policy-no-training').checked) {
                    policies.push({name: 'no_training', allow_training_on_output: false});
                }

                const result = await apiPost('/api/project/start', {
                    name: document.getElementById('new-name').value,
                    prompt: prompt,
                    project_type: document.getElementById('new-type').value,
                    criteria: document.getElementById('new-criteria').value,
                    budget: parseFloat(document.getElementById('new-budget').value),
                    time_seconds: parseInt(document.getElementById('new-time-seconds').value),
                    concurrency: parseInt(document.getElementById('new-concurrency').value),
                    project_id: document.getElementById('new-project-id').value,
                    output_dir: document.getElementById('new-output-dir').value,
                    quality_targets: qualityTargets,
                    policies: policies
                });

                if (result.status === 'error') {
                    alert('Error: ' + result.message);
                } else {
                    selectedProjectId = result.project_id;
                }
            } catch (err) {
                alert('Failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-play mr-2"></i>Start Project';
            }
        }

        // Start improvement
        async function startImprovement() {
            const path = document.getElementById('improve-path').value.trim();
            const instructions = document.getElementById('improve-instructions').value.trim();

            if (!path) { alert('Please enter codebase path'); return; }
            if (!instructions) { alert('Please enter instructions'); return; }

            const btn = document.getElementById('btn-start-improve');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';

            try {
                const result = await apiPost('/api/project/improve', {
                    codebase_path: path,
                    instructions: instructions,
                    budget: parseFloat(document.getElementById('improve-budget').value),
                });

                if (result.status === 'error') {
                    alert('Error: ' + result.message);
                } else {
                    selectedProjectId = result.project_id;
                }
            } catch (err) {
                alert('Failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-magic mr-2"></i>Analyze & Improve';
            }
        }

        // Upload and run
        async function uploadAndRun() {
            if (!selectedFile) { alert('Please select a file'); return; }

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('instructions', document.getElementById('upload-instructions').value);
            formData.append('budget', document.getElementById('upload-budget').value);

            const btn = document.getElementById('btn-start-upload');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';

            try {
                const result = await apiPost('/api/project/upload', formData, false);

                if (result.status === 'error') {
                    alert('Error: ' + result.message);
                } else {
                    selectedProjectId = result.project_id;
                    clearUpload();
                }
            } catch (err) {
                alert('Failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-play mr-2"></i>Process & Run';
            }
        }

        // Stop selected project
        async function stopSelectedProject() {
            if (!selectedProjectId) return;
            try {
                await fetch(`/api/project/${selectedProjectId}/stop`, {method: 'POST'});
            } catch (err) {
                console.error('Stop failed:', err);
            }
        }

        // Remove a finished project
        async function removeProject(projectId) {
            if (!projectId) return;
            try {
                const response = await fetch(`/api/project/${projectId}/remove`, {method: 'POST'});
                const result = await response.json();
                if (result.status === 'error') {
                    alert(result.message);
                } else {
                    // Clear selection if removed project was selected
                    if (selectedProjectId === projectId) {
                        selectedProjectId = null;
                    }
                }
            } catch (err) {
                console.error('Remove failed:', err);
            }
        }

        // Clear all finished projects
        async function clearFinished() {
            try {
                const response = await fetch('/api/projects/clear-finished', {method: 'POST'});
                const result = await response.json();
                if (result.status === 'ok') {
                    selectedProjectId = null;
                }
            } catch (err) {
                console.error('Clear failed:', err);
            }
        }

        // ===== AUTO-SAVE FORM FIELDS =====
        const FORM_FIELDS = [
            'new-name', 'new-type', 'new-prompt', 'new-criteria', 'new-budget',
            'new-time-seconds', 'new-concurrency', 'new-project-id', 'new-output-dir',
            'qt-code-gen', 'qt-code-review', 'qt-reasoning', 'qt-evaluation',
            'improve-path', 'improve-instructions', 'improve-budget',
            'upload-instructions', 'upload-budget'
        ];

        // Load saved values on startup
        function loadSavedForms() {
            FORM_FIELDS.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    const saved = localStorage.getItem('form_' + id);
                    if (saved !== null) {
                        if (el.type === 'checkbox') {
                            el.checked = saved === 'true';
                        } else {
                            el.value = saved;
                        }
                    }
                }
            });
            logSystem('info', 'Loaded saved form values');
        }

        // Save values on change
        FORM_FIELDS.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', () => {
                    const value = el.type === 'checkbox' ? el.checked : el.value;
                    localStorage.setItem('form_' + id, value);
                });
                // Also save on input for text fields (debounced)
                if (el.tagName === 'TEXTAREA' || el.type === 'text') {
                    let timeout;
                    el.addEventListener('input', () => {
                        clearTimeout(timeout);
                        timeout = setTimeout(() => {
                            localStorage.setItem('form_' + id, el.value);
                        }, 500);
                    });
                }
            }
        });

        // Clear saved forms function
        function clearSavedForms() {
            FORM_FIELDS.forEach(id => {
                localStorage.removeItem('form_' + id);
            });
            logSystem('info', 'Cleared saved form values');
        }

        // ===== BROWSER NOTIFICATIONS =====
        let notificationsEnabled = false;

        async function requestNotificationPermission() {
            if ('Notification' in window) {
                const permission = await Notification.requestPermission();
                notificationsEnabled = permission === 'granted';
                if (notificationsEnabled) {
                    logSystem('info', '🔔 Notifications enabled');
                }
            }
        }

        function sendNotification(title, body) {
            if (notificationsEnabled && document.visibilityState === 'hidden') {
                new Notification(title, {
                    body: body,
                    icon: '🚀',
                    tag: 'mission-control'
                });
            }
        }

        // Request permission on first user interaction
        document.addEventListener('click', () => {
            if (!notificationsEnabled && 'Notification' in window && Notification.permission === 'default') {
                requestNotificationPermission();
            }
        }, { once: true });

        // ===== BETTER ERROR HANDLING =====
        function showError(title, message, details = '') {
            logSystem('error', `${title}: ${message}`);

            // Create error toast
            const toast = document.createElement('div');
            toast.className = 'fixed bottom-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg max-w-md z-50 animate-fade-in';
            toast.innerHTML = `
                <div class="flex items-start gap-3">
                    <i class="fas fa-exclamation-circle mt-1"></i>
                    <div>
                        <h4 class="font-bold">${title}</h4>
                        <p class="text-sm mt-1">${message}</p>
                        ${details ? `<p class="text-xs mt-2 text-red-200">${details}</p>` : ''}
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-red-200 hover:text-white">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            document.body.appendChild(toast);

            // Auto-remove after 10 seconds
            setTimeout(() => {
                if (toast.parentElement) toast.remove();
            }, 10000);
        }

        function showSuccess(title, message) {
            logSystem('success', `${title}: ${message}`);

            const toast = document.createElement('div');
            toast.className = 'fixed bottom-4 right-4 bg-green-600 text-white p-4 rounded-lg shadow-lg max-w-md z-50 animate-fade-in';
            toast.innerHTML = `
                <div class="flex items-start gap-3">
                    <i class="fas fa-check-circle mt-1"></i>
                    <div>
                        <h4 class="font-bold">${title}</h4>
                        <p class="text-sm mt-1">${message}</p>
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-green-200 hover:text-white">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            document.body.appendChild(toast);

            setTimeout(() => {
                if (toast.parentElement) toast.remove();
            }, 5000);
        }

        // Global error handler
        window.addEventListener('error', (e) => {
            showError('JavaScript Error', e.message, `File: ${e.filename}`);
        });

        window.addEventListener('unhandledrejection', (e) => {
            showError('Async Error', e.reason?.message || 'Unknown error');
        });

        // Initialize
        loadSavedForms();
        connectWS();

        // Polling fallback + log updates
        setInterval(() => {
            fetch('/api/state')
                .then(r => {
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(data => {
                    updateUI(data);
                    // If WebSocket is disconnected but we got data, update status
                    if (ws && ws.readyState !== WebSocket.OPEN) {
                        document.getElementById('ws-status').innerHTML = '🟡 HTTP Mode';
                    }
                })
                .catch(err => {
                    console.error('Polling error:', err);
                    // Only show error if we haven't received data recently
                    if (!state || Date.now() - lastUpdateTime > 5000) {
                        document.getElementById('ws-status').innerHTML = '🔴 No Connection';
                        document.getElementById('server-status-badge').textContent = 'Disconnected';
                        document.getElementById('server-status-badge').className = 'status-badge status-failed';
                    }
                });

            // Refresh logs for selected project
            if (selectedProjectId) {
                fetchProjectLogs(selectedProjectId);
            }
        }, 1000);
    </script>
</body>
</html>
"""


def run_mission_control(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):
    import asyncio

    print("[run_mission_control] Creating server...")
    server = MissionControlServer(host=host, port=port)
    print(f"[run_mission_control] Server created for {host}:{port}")

    if open_browser:
        print("[run_mission_control] Opening browser...")
        webbrowser.open(f"http://{host}:{port}")

    print("[run_mission_control] Starting event loop...")
    asyncio.run(server.run())


if __name__ == "__main__":
    run_mission_control()
