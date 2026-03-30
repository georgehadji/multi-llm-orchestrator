"""
Enhanced Real-Time Dashboard v2.0
==================================
Dashboard with comprehensive project visibility:
- Architecture decisions (style, paradigm, stack)
- Active task progress (current/total)
- Project details (name, prompt, criteria)
- Model status with reasons for inactivity

Usage:
    from orchestrator.dashboard_enhanced import EnhancedDashboard
    dashboard = EnhancedDashboard()
    dashboard.start()
"""
from __future__ import annotations

import asyncio
import time
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .log_config import get_logger
from .models import COST_TABLE, ROUTING_TABLE, Model, TaskStatus, TaskType, get_provider
from .state import StateManager
from .telemetry_store import TelemetryStore

logger = get_logger(__name__)


@dataclass
class ActiveTaskInfo:
    """Information about the currently active task."""
    task_id: str = ""
    task_type: str = ""
    prompt: str = ""
    status: str = "idle"
    iteration: int = 0
    max_iterations: int = 0
    score: float = 0.0
    model_used: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class ProjectInfo:
    """Information about the current project."""
    project_id: str = ""
    description: str = ""
    success_criteria: str = ""
    status: str = "idle"
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    progress_percent: float = 0.0
    budget_used: float = 0.0
    budget_total: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class ArchitectureInfo:
    """Architecture decisions for the current project."""
    style: str = ""
    paradigm: str = ""
    api_style: str = ""
    database_type: str = ""
    primary_language: str = ""
    frameworks: list[str] = None
    libraries: list[str] = None
    databases: list[str] = None
    tools: list[str] = None
    constraints: list[str] = None
    patterns: list[str] = None
    rationale: str = ""

    def __post_init__(self):
        if self.frameworks is None:
            self.frameworks = []
        if self.libraries is None:
            self.libraries = []
        if self.databases is None:
            self.databases = []
        if self.tools is None:
            self.tools = []
        if self.constraints is None:
            self.constraints = []
        if self.patterns is None:
            self.patterns = []


@dataclass
class ModelStatusInfo:
    """Detailed model status with reason if inactive."""
    name: str
    provider: str
    available: bool
    reason: str = ""  # Why unavailable (if not available)
    health_status: str = "unknown"  # healthy, degraded, unhealthy
    success_rate: float = 0.95
    avg_latency: float = 100
    call_count: int = 0
    cost_input: float = 0.0
    cost_output: float = 0.0
    consecutive_failures: int = 0


class EnhancedDataProvider:
    """Provides enhanced real-time data from orchestrator."""

    def __init__(self):
        self.state_mgr = StateManager()
        self.telemetry = TelemetryStore()
        self._cache: dict[str, Any] = {}
        self._cache_time: float = 0
        self._cache_ttl = 3  # seconds - faster refresh

        # Current project tracking
        self._current_project_id: str | None = None
        self._current_state: Any | None = None
        self._project_start_time: float | None = None
        self._architecture_rules: ArchitectureInfo | None = None
        self._active_task: ActiveTaskInfo | None = None

        # Model health tracking
        self._model_health: dict[Model, dict[str, Any]] = {}
        self._consecutive_failures: dict[Model, int] = dict.fromkeys(Model, 0)

    def set_current_project(self, project_id: str, state: Any, architecture_rules: Any | None = None):
        """Set the current active project."""
        self._current_project_id = project_id
        self._current_state = state
        self._project_start_time = time.time()

        if architecture_rules:
            self._architecture_rules = self._convert_architecture_rules(architecture_rules)
        else:
            self._architecture_rules = None

    def set_active_task(self, task_id: str, task: Any, model: Model | None = None):
        """Set the currently active task."""
        self._active_task = ActiveTaskInfo(
            task_id=task_id,
            task_type=task.type.value if hasattr(task, 'type') else "unknown",
            prompt=task.prompt if hasattr(task, 'prompt') else "",
            status="running",
            iteration=1,
            max_iterations=task.max_iterations if hasattr(task, 'max_iterations') else 3,
            model_used=model.value if model else "",
            elapsed_seconds=0.0
        )

    def update_task_progress(self, iteration: int, score: float, status: str = "running"):
        """Update the active task progress."""
        if self._active_task:
            self._active_task.iteration = iteration
            self._active_task.score = score
            self._active_task.status = status
            if self._project_start_time:
                self._active_task.elapsed_seconds = time.time() - self._project_start_time

    def complete_task(self, task_id: str, status: str = "completed"):
        """Mark a task as complete."""
        if self._active_task and self._active_task.task_id == task_id:
            self._active_task.status = status
        self._active_task = None

    def _convert_architecture_rules(self, rules: Any) -> ArchitectureInfo:
        """Convert ProjectRules to ArchitectureInfo."""
        try:
            arch = rules.architecture if hasattr(rules, 'architecture') else rules
            stack = arch.stack if hasattr(arch, 'stack') else None

            return ArchitectureInfo(
                style=arch.style.value if hasattr(arch, 'style') else str(arch.get('style', '')),
                paradigm=arch.paradigm.value if hasattr(arch, 'paradigm') else str(arch.get('paradigm', '')),
                api_style=arch.api_style.value if hasattr(arch, 'api_style') else str(arch.get('api_style', '')),
                database_type=arch.database_type.value if hasattr(arch, 'database_type') else str(arch.get('database_type', '')),
                primary_language=stack.primary_language if stack else "",
                frameworks=list(stack.frameworks) if stack else [],
                libraries=list(stack.libraries) if stack else [],
                databases=list(stack.databases) if stack else [],
                tools=list(stack.tools) if stack else [],
                constraints=list(arch.constraints) if hasattr(arch, 'constraints') else [],
                patterns=list(arch.patterns) if hasattr(arch, 'patterns') else [],
                rationale=arch.rationale if hasattr(arch, 'rationale') else "",
            )
        except Exception as e:
            logger.warning(f"Could not convert architecture rules: {e}")
            return ArchitectureInfo()

    async def get_models_with_status(self) -> list[ModelStatusInfo]:
        """Get detailed model status with reasons for inactivity."""
        models = []

        # Import here to avoid circular dependency
        try:
            from .api_clients import UnifiedClient
            client = UnifiedClient()
        except Exception:
            client = None

        for model in Model:
            try:
                # Get telemetry
                telemetry = self.telemetry.get_model_snapshot(model)

                # Determine availability
                available = True
                reason = ""
                health_status = "healthy"

                # Check 1: API key / SDK availability
                if client and not client.is_available(model):
                    available = False
                    reason = "API key not configured or SDK not installed"
                    health_status = "unhealthy"
                # Check 2: Circuit breaker / consecutive failures
                elif self._consecutive_failures.get(model, 0) >= 3:
                    available = False
                    reason = f"Circuit breaker tripped ({self._consecutive_failures[model]} consecutive failures)"
                    health_status = "unhealthy"
                # Check 3: Adaptive router (if available)
                else:
                    try:
                        from .adaptive_router import AdaptiveRouter
                        router = AdaptiveRouter()
                        if not router.is_available(model):
                            available = False
                            state = router.get_model_state(model)
                            reason = f"Adaptive router: {state.value if hasattr(state, 'value') else str(state)}"
                            health_status = "degraded" if "degraded" in str(state).lower() else "unhealthy"
                    except Exception:
                        pass

                # Get metrics
                success_rate = 0.95
                avg_latency = 100
                call_count = 0

                if telemetry:
                    success_rate = telemetry.get("success_rate", 0.95)
                    avg_latency = telemetry.get("latency_avg_ms", 100)
                    call_count = telemetry.get("call_count", 0)

                models.append(ModelStatusInfo(
                    name=model.value,
                    provider=get_provider(model),
                    available=available,
                    reason=reason,
                    health_status=health_status,
                    success_rate=success_rate,
                    avg_latency=avg_latency,
                    call_count=call_count,
                    cost_input=COST_TABLE[model]["input"],
                    cost_output=COST_TABLE[model]["output"],
                    consecutive_failures=self._consecutive_failures.get(model, 0),
                ))
            except Exception as e:
                logger.debug(f"Could not get status for {model}: {e}")
                models.append(ModelStatusInfo(
                    name=model.value,
                    provider=get_provider(model),
                    available=False,
                    reason=f"Error checking status: {str(e)[:50]}",
                    health_status="unknown",
                    cost_input=COST_TABLE[model]["input"],
                    cost_output=COST_TABLE[model]["output"],
                ))

        return models

    async def get_architecture_info(self) -> dict[str, Any] | None:
        """Get current project architecture info."""
        if self._architecture_rules:
            return asdict(self._architecture_rules)
        return None

    async def get_project_info(self) -> ProjectInfo:
        """Get current project information."""
        info = ProjectInfo()

        if self._current_state:
            state = self._current_state
            info.project_id = self._current_project_id or ""
            info.description = state.project_description if hasattr(state, 'project_description') else ""
            info.success_criteria = state.success_criteria if hasattr(state, 'success_criteria') else ""
            info.status = state.status.value if hasattr(state, 'status') else "unknown"

            # Task counts
            tasks = state.tasks if hasattr(state, 'tasks') else {}
            results = state.results if hasattr(state, 'results') else {}
            info.total_tasks = len(tasks)
            info.completed_tasks = sum(1 for r in results.values()
                                       if hasattr(r, 'status') and r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED))
            info.failed_tasks = sum(1 for r in results.values()
                                    if hasattr(r, 'status') and r.status == TaskStatus.FAILED)

            if info.total_tasks > 0:
                info.progress_percent = (info.completed_tasks / info.total_tasks) * 100

            # Budget
            budget = state.budget if hasattr(state, 'budget') else None
            if budget:
                info.budget_used = budget.spent_usd if hasattr(budget, 'spent_usd') else 0.0
                info.budget_total = budget.max_usd if hasattr(budget, 'max_usd') else 0.0
                info.elapsed_seconds = budget.elapsed_seconds if hasattr(budget, 'elapsed_seconds') else 0.0

        return info

    async def get_active_task(self) -> dict[str, Any] | None:
        """Get currently active task info."""
        if self._active_task:
            return asdict(self._active_task)
        return None

    async def get_metrics(self) -> dict[str, Any]:
        """Get system-wide metrics."""
        try:
            total_calls = 0
            total_cost = 0.0
            total_latency = 0
            model_count = 0

            for model in Model:
                telemetry = self.telemetry.get_model_snapshot(model)
                if telemetry:
                    total_calls += telemetry.get("call_count", 0)
                    total_cost += telemetry.get("cost_total", 0)
                    total_latency += telemetry.get("latency_avg_ms", 100)
                    model_count += 1

            avg_latency = total_latency / model_count if model_count > 0 else 0

            return {
                "total_calls": total_calls,
                "total_cost": round(total_cost, 4),
                "avg_latency_ms": round(avg_latency, 1),
                "active_projects": 1 if self._current_project_id else 0,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Could not get metrics: {e}")
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0,
                "active_projects": 0,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_routing_table(self) -> dict[str, Any]:
        """Get current routing configuration."""
        routing = {}

        for task_type in TaskType:
            if task_type in ROUTING_TABLE:
                routing[task_type.value] = {
                    "preferred": [m.value for m in ROUTING_TABLE[task_type]],
                    "fallback": [m.value for m in ROUTING_TABLE[task_type][1:]] if len(ROUTING_TABLE[task_type]) > 1 else [],
                }

        return routing

    def record_model_failure(self, model: Model):
        """Record a model failure for circuit breaker tracking."""
        self._consecutive_failures[model] = self._consecutive_failures.get(model, 0) + 1

    def record_model_success(self, model: Model):
        """Reset failure counter on success."""
        self._consecutive_failures[model] = 0


class EnhancedDashboardServer:
    """Enhanced dashboard server with comprehensive project visibility."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.data_provider = EnhancedDataProvider()
        self._setup_app()

    def _setup_app(self):
        """Setup FastAPI app with enhanced endpoints."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import HTMLResponse, JSONResponse

            self.app = FastAPI(title="Mission Control Enhanced")

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )

            @self.app.get("/")
            async def dashboard():
                """Serve dashboard HTML."""
                return HTMLResponse(content=self._get_html())

            @self.app.get("/api/models")
            async def get_models():
                """Get detailed model status."""
                data = await self.data_provider.get_models_with_status()
                return JSONResponse(content=[asdict(m) for m in data])

            @self.app.get("/api/metrics")
            async def get_metrics():
                """Get system metrics."""
                data = await self.data_provider.get_metrics()
                return JSONResponse(content=data)

            @self.app.get("/api/project")
            async def get_project():
                """Get current project info."""
                info = await self.data_provider.get_project_info()
                return JSONResponse(content=asdict(info))

            @self.app.get("/api/architecture")
            async def get_architecture():
                """Get architecture decisions."""
                data = await self.data_provider.get_architecture_info()
                return JSONResponse(content=data or {})

            @self.app.get("/api/active-task")
            async def get_active_task():
                """Get currently active task."""
                data = await self.data_provider.get_active_task()
                return JSONResponse(content=data or {"status": "idle"})

            @self.app.get("/api/routing")
            async def get_routing():
                """Get routing table."""
                data = await self.data_provider.get_routing_table()
                return JSONResponse(content=data)

            @self.app.get("/api/status")
            async def get_full_status():
                """Get complete status in one call."""
                return JSONResponse(content={
                    "project": asdict(await self.data_provider.get_project_info()),
                    "architecture": await self.data_provider.get_architecture_info() or {},
                    "active_task": await self.data_provider.get_active_task() or {"status": "idle"},
                    "metrics": await self.data_provider.get_metrics(),
                    "models": [asdict(m) for m in await self.data_provider.get_models_with_status()],
                })

        except ImportError:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            raise

    def _get_html(self) -> str:
        """Generate enhanced dashboard HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control v2.0 | Multi-LLM Orchestrator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #111118;
            --bg-card: #1a1a24;
            --border-color: #3a3a4a;
            --text-primary: #ffffff;
            --text-secondary: #9090a0;
            --accent-cyan: #00d4ff;
            --accent-pink: #ff4db8;
            --accent-green: #00ff88;
            --accent-yellow: #ffcc00;
            --accent-red: #ff5577;
            --accent-orange: #ff8844;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 20px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Main Layout */
        .main-container {
            display: grid;
            grid-template-columns: 320px 1fr 380px;
            gap: 20px;
            padding: 20px;
            max-width: 1920px;
            margin: 0 auto;
        }

        /* Section Cards */
        .section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }

        .section-header {
            background: var(--bg-secondary);
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .section-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-icon {
            font-size: 16px;
        }

        .section-content {
            padding: 18px;
        }

        /* Project Info */
        .project-name {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
            line-height: 1.4;
        }

        .project-meta {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }

        .progress-container {
            margin: 16px 0;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 6px;
        }

        .progress-bar {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-pink));
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        .stat-box {
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 700;
            color: var(--accent-cyan);
        }

        .stat-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Architecture Section */
        .arch-item {
            margin-bottom: 14px;
        }

        .arch-label {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .arch-value {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-primary);
        }

        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 6px;
        }

        .tech-badge {
            background: rgba(0, 212, 255, 0.15);
            color: var(--accent-cyan);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }

        .tech-badge.secondary {
            background: rgba(255, 77, 184, 0.15);
            color: var(--accent-pink);
        }

        .tech-badge.tool {
            background: rgba(255, 204, 0, 0.15);
            color: var(--accent-yellow);
        }

        /* Active Task */
        .task-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .task-status.running {
            background: rgba(0, 212, 255, 0.15);
            color: var(--accent-cyan);
        }

        .task-status.idle {
            background: rgba(144, 144, 160, 0.15);
            color: var(--text-secondary);
        }

        .task-status.completed {
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-green);
        }

        .task-status.failed {
            background: rgba(255, 85, 119, 0.15);
            color: var(--accent-red);
        }

        .task-prompt {
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 12px;
            max-height: 120px;
            overflow-y: auto;
            line-height: 1.5;
        }

        .task-progress-detail {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }

        .iteration-counter {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .score-badge {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-pink));
            color: #000;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
        }

        /* Models Grid */
        .models-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .model-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            transition: all 0.2s;
        }

        .model-item:hover {
            border-color: var(--accent-cyan);
        }

        .model-item.unavailable {
            opacity: 0.6;
            border-color: var(--accent-red);
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .model-name {
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .model-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .model-indicator.online {
            background: var(--accent-green);
            box-shadow: 0 0 6px var(--accent-green);
        }

        .model-indicator.offline {
            background: var(--accent-red);
        }

        .model-indicator.degraded {
            background: var(--accent-orange);
            box-shadow: 0 0 6px var(--accent-orange);
        }

        .model-provider {
            font-size: 10px;
            color: var(--text-secondary);
            padding: 2px 6px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 3px;
        }

        .model-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            font-size: 11px;
        }

        .model-stat {
            text-align: center;
        }

        .model-stat-value {
            color: var(--text-primary);
            font-weight: 600;
        }

        .model-stat-label {
            color: var(--text-secondary);
            font-size: 9px;
            text-transform: uppercase;
        }

        .model-reason {
            font-size: 10px;
            color: var(--accent-red);
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid var(--border-color);
        }

        /* Success Criteria */
        .criteria-box {
            background: var(--bg-secondary);
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.5;
            border-left: 3px solid var(--accent-yellow);
        }

        /* Middle Column */
        .middle-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .constraints-list, .patterns-list {
            list-style: none;
            font-size: 12px;
        }

        .constraints-list li, .patterns-list li {
            padding: 6px 0;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }

        .constraints-list li:before {
            content: "✓";
            color: var(--accent-green);
            font-weight: bold;
        }

        .patterns-list li:before {
            content: "◈";
            color: var(--accent-cyan);
        }

        .constraints-list li:last-child, .patterns-list li:last-child {
            border-bottom: none;
        }

        /* Refresh Button */
        .refresh-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .last-updated {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .refresh-btn {
            background: linear-gradient(135deg, var(--accent-cyan), #0088ff);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .refresh-btn:hover {
            opacity: 0.9;
        }

        /* Responsive */
        @media (max-width: 1400px) {
            .main-container {
                grid-template-columns: 280px 1fr 340px;
            }
        }

        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
        }

        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 12px;
            opacity: 0.5;
        }

        .empty-state-text {
            font-size: 13px;
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>◈ MISSION CONTROL <span style="font-size: 14px; opacity: 0.7;">v2.0</span></h1>
        <div class="header-status">
            <span class="status-dot"></span>
            <span id="connection-status">Live</span>
        </div>
    </header>

    <main class="main-container">
        <!-- Left Column: Project & Architecture -->
        <div class="left-column">
            <!-- Project Info -->
            <div class="section">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">📋</span>
                        Project
                    </div>
                    <span id="project-status-badge" class="task-status idle">Idle</span>
                </div>
                <div class="section-content">
                    <div class="project-name" id="project-description">No active project</div>
                    <div class="project-meta" id="project-id">-</div>

                    <div class="progress-container">
                        <div class="progress-header">
                            <span>Progress</span>
                            <span id="progress-text">0/0 tasks</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-value" id="stat-completed">0</div>
                            <div class="stat-label">Completed</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="stat-failed">0</div>
                            <div class="stat-label">Failed</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="stat-budget">$0</div>
                            <div class="stat-label">Budget Used</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="stat-time">0s</div>
                            <div class="stat-label">Elapsed</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Architecture Decisions -->
            <div class="section" style="margin-top: 20px;">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">🏗️</span>
                        Architecture
                    </div>
                </div>
                <div class="section-content" id="architecture-content">
                    <div class="empty-state">
                        <div class="empty-state-icon">🏗️</div>
                        <div class="empty-state-text">No architecture decisions yet</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Column: Active Task & Criteria -->
        <div class="middle-column">
            <!-- Success Criteria -->
            <div class="section">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">🎯</span>
                        Success Criteria
                    </div>
                </div>
                <div class="section-content">
                    <div class="criteria-box" id="success-criteria">
                        No project running. Start a project to see success criteria.
                    </div>
                </div>
            </div>

            <!-- Active Task -->
            <div class="section">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">⚡</span>
                        Active Task
                    </div>
                    <span id="task-status-badge" class="task-status idle">Idle</span>
                </div>
                <div class="section-content">
                    <div id="active-task-content">
                        <div class="empty-state">
                            <div class="empty-state-icon">⏳</div>
                            <div class="empty-state-text">Waiting for tasks...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Constraints & Patterns -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="section">
                    <div class="section-header">
                        <div class="section-title">
                            <span class="section-icon">🔒</span>
                            Constraints
                        </div>
                    </div>
                    <div class="section-content">
                        <ul class="constraints-list" id="constraints-list">
                            <li style="opacity: 0.5; border: none;">No constraints defined</li>
                        </ul>
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">
                        <div class="section-title">
                            <span class="section-icon">📐</span>
                            Patterns
                        </div>
                    </div>
                    <div class="section-content">
                        <ul class="patterns-list" id="patterns-list">
                            <li style="opacity: 0.5; border: none;">No patterns defined</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Column: Models -->
        <div class="right-column">
            <div class="section">
                <div class="section-header">
                    <div class="section-title">
                        <span class="section-icon">🤖</span>
                        Models Status
                    </div>
                    <span id="models-count" style="font-size: 11px; color: var(--text-secondary);">-</span>
                </div>
                <div class="section-content" style="padding: 12px;">
                    <div class="models-list" id="models-list">
                        <div class="empty-state">
                            <div class="empty-state-icon">🤖</div>
                            <div class="empty-state-text">Loading models...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <div class="refresh-bar">
        <span class="last-updated">Last updated: <span id="last-updated">-</span></span>
        <button class="refresh-btn" onclick="refreshData()">
            🔄 Refresh
        </button>
    </div>

    <script>
        // State
        let currentProjectId = null;
        let currentTaskId = null;

        // Format helpers
        function formatCurrency(value) {
            return '$' + value.toFixed(4);
        }

        function formatDuration(seconds) {
            if (seconds < 60) return Math.round(seconds) + 's';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
            return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
        }

        function truncate(str, len) {
            if (!str) return '';
            return str.length > len ? str.substring(0, len) + '...' : str;
        }

        // Update functions
        function updateProject(project) {
            if (!project || !project.project_id) {
                document.getElementById('project-description').textContent = 'No active project';
                document.getElementById('project-id').textContent = '-';
                document.getElementById('project-status-badge').textContent = 'Idle';
                document.getElementById('project-status-badge').className = 'task-status idle';
                return;
            }

            currentProjectId = project.project_id;
            document.getElementById('project-description').textContent =
                truncate(project.description, 80) || 'Untitled Project';
            document.getElementById('project-id').textContent =
                `ID: ${project.project_id.substring(0, 12)}...`;

            // Status badge
            const statusBadge = document.getElementById('project-status-badge');
            statusBadge.textContent = project.status;
            statusBadge.className = 'task-status ' + (project.status === 'SUCCESS' ? 'completed' :
                                                      project.status === 'FAILED' ? 'failed' : 'running');

            // Progress
            const progressText = `${project.completed_tasks}/${project.total_tasks} tasks`;
            document.getElementById('progress-text').textContent = progressText;
            document.getElementById('progress-fill').style.width = project.progress_percent + '%';

            // Stats
            document.getElementById('stat-completed').textContent = project.completed_tasks;
            document.getElementById('stat-failed').textContent = project.failed_tasks;
            document.getElementById('stat-budget').textContent = formatCurrency(project.budget_used);
            document.getElementById('stat-time').textContent = formatDuration(project.elapsed_seconds);
        }

        function updateArchitecture(arch) {
            const container = document.getElementById('architecture-content');

            if (!arch || !arch.style) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">🏗️</div>
                        <div class="empty-state-text">No architecture decisions yet</div>
                    </div>
                `;
                return;
            }

            const frameworks = arch.frameworks?.map(f => `<span class="tech-badge">${f}</span>`).join('') || '';
            const libraries = arch.libraries?.slice(0, 5).map(l => `<span class="tech-badge secondary">${l}</span>`).join('') || '';
            const tools = arch.tools?.slice(0, 3).map(t => `<span class="tech-badge tool">${t}</span>`).join('') || '';

            container.innerHTML = `
                <div class="arch-item">
                    <div class="arch-label">Architectural Style</div>
                    <div class="arch-value">${arch.style.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">Programming Paradigm</div>
                    <div class="arch-value">${arch.paradigm.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">API Style</div>
                    <div class="arch-value">${arch.api_style?.toUpperCase() || 'REST'}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">Primary Language</div>
                    <div class="arch-value">${arch.primary_language?.replace(/\\b\\w/g, l => l.toUpperCase()) || 'Unknown'}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">Frameworks</div>
                    <div class="tech-stack">${frameworks || '<span style="color: var(--text-secondary); font-size: 12px;">None specified</span>'}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">Key Libraries</div>
                    <div class="tech-stack">${libraries || '<span style="color: var(--text-secondary); font-size: 12px;">None specified</span>'}</div>
                </div>
                <div class="arch-item">
                    <div class="arch-label">Tools</div>
                    <div class="tech-stack">${tools || '<span style="color: var(--text-secondary); font-size: 12px;">None specified</span>'}</div>
                </div>
            `;
        }

        function updateActiveTask(task) {
            const container = document.getElementById('active-task-content');
            const statusBadge = document.getElementById('task-status-badge');

            if (!task || task.status === 'idle') {
                statusBadge.textContent = 'Idle';
                statusBadge.className = 'task-status idle';
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">⏳</div>
                        <div class="empty-state-text">Waiting for tasks...</div>
                    </div>
                `;
                currentTaskId = null;
                return;
            }

            currentTaskId = task.task_id;
            statusBadge.textContent = task.status;
            statusBadge.className = 'task-status ' + task.status;

            container.innerHTML = `
                <div style="margin-bottom: 12px;">
                    <span style="font-size: 12px; color: var(--text-secondary);">Task ID:</span>
                    <span style="font-size: 13px; font-weight: 600; margin-left: 6px;">${task.task_id}</span>
                    <span style="margin-left: 12px; font-size: 11px; color: var(--accent-cyan); background: rgba(0,212,255,0.1); padding: 2px 6px; border-radius: 3px;">${task.task_type}</span>
                </div>
                <div class="task-prompt">${truncate(task.prompt, 400) || 'No prompt available'}</div>
                <div class="task-progress-detail">
                    <span class="iteration-counter">Iteration ${task.iteration || 0} / ${task.max_iterations || 3}</span>
                    <span class="score-badge">Score: ${(task.score || 0).toFixed(3)}</span>
                </div>
                ${task.model_used ? `<div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary);">Model: ${task.model_used}</div>` : ''}
            `;
        }

        function updateSuccessCriteria(criteria) {
            const el = document.getElementById('success-criteria');
            el.textContent = criteria || 'No success criteria defined';
        }

        function updateConstraintsAndPatterns(arch) {
            const constraintsList = document.getElementById('constraints-list');
            const patternsList = document.getElementById('patterns-list');

            if (arch && arch.constraints && arch.constraints.length > 0) {
                constraintsList.innerHTML = arch.constraints.map(c => `<li>${c}</li>`).join('');
            } else {
                constraintsList.innerHTML = '<li style="opacity: 0.5; border: none;">No constraints defined</li>';
            }

            if (arch && arch.patterns && arch.patterns.length > 0) {
                patternsList.innerHTML = arch.patterns.map(p => `<li>${p}</li>`).join('');
            } else {
                patternsList.innerHTML = '<li style="opacity: 0.5; border: none;">No patterns defined</li>';
            }
        }

        function updateModels(models) {
            const container = document.getElementById('models-list');
            const countEl = document.getElementById('models-count');

            if (!models || models.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No models</div></div>';
                countEl.textContent = '-';
                return;
            }

            const availableCount = models.filter(m => m.available).length;
            countEl.textContent = `${availableCount}/${models.length} available`;

            container.innerHTML = models.map(model => {
                const indicatorClass = model.available ? 'online' :
                                       model.health_status === 'degraded' ? 'degraded' : 'offline';
                const unavailableClass = model.available ? '' : 'unavailable';
                const reasonHtml = !model.available && model.reason ?
                    `<div class="model-reason">⚠ ${model.reason}</div>` : '';

                return `
                    <div class="model-item ${unavailableClass}">
                        <div class="model-header">
                            <div class="model-name">
                                <span class="model-indicator ${indicatorClass}"></span>
                                ${model.name}
                            </div>
                            <span class="model-provider">${model.provider}</span>
                        </div>
                        <div class="model-stats">
                            <div class="model-stat">
                                <div class="model-stat-value">${(model.success_rate * 100).toFixed(0)}%</div>
                                <div class="model-stat-label">Success</div>
                            </div>
                            <div class="model-stat">
                                <div class="model-stat-value">${Math.round(model.avg_latency)}ms</div>
                                <div class="model-stat-label">Latency</div>
                            </div>
                            <div class="model-stat">
                                <div class="model-stat-value">${model.call_count}</div>
                                <div class="model-stat-label">Calls</div>
                            </div>
                        </div>
                        ${reasonHtml}
                    </div>
                `;
            }).join('');
        }

        // Main data loading
        async function refreshData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                updateProject(data.project);
                updateArchitecture(data.architecture);
                updateActiveTask(data.active_task);
                updateSuccessCriteria(data.project?.success_criteria);
                updateConstraintsAndPatterns(data.architecture);
                updateModels(data.models);

                document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                document.getElementById('connection-status').textContent = 'Live';
                document.querySelector('.status-dot').style.background = 'var(--accent-green)';

            } catch (err) {
                console.error('Failed to load data:', err);
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.querySelector('.status-dot').style.background = 'var(--accent-red)';
            }
        }

        // Initial load
        refreshData();

        // Auto-refresh every 3 seconds
        setInterval(refreshData, 3000);
    </script>
</body>
</html>'''

    async def run(self):
        """Start the dashboard server."""
        from uvicorn import Config, Server

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)
        await server.serve()


def run_enhanced_dashboard(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):
    """Run the enhanced dashboard."""

    url = f"http://{host}:{port}"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     ◈ MISSION CONTROL v2.0 - ENHANCED ◈                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Dashboard URL: {url:<44} ║
║                                                                  ║
║  📊 Features:                                                    ║
║     • Architecture decisions (style, paradigm, stack)            ║
║     • Real-time task progress (current/total)                    ║
║     • Active project details & success criteria                  ║
║     • Current prompt being executed                              ║
║     • Model status with reasons for inactivity                   ║
║                                                                  ║
║  🔄 Auto-refresh: Every 3 seconds                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if open_browser:
        webbrowser.open(url)

    dashboard = EnhancedDashboardServer(host=host, port=port)
    asyncio.run(dashboard.run())


# Convenience class for integration with orchestrator
class DashboardIntegration:
    """Integrates dashboard with orchestrator for real-time updates."""

    def __init__(self, data_provider: EnhancedDataProvider):
        self.data_provider = data_provider

    def on_project_start(self, project_id: str, state: Any, architecture_rules: Any | None = None):
        """Call when project starts."""
        self.data_provider.set_current_project(project_id, state, architecture_rules)

    def on_task_start(self, task_id: str, task: Any, model: Model | None = None):
        """Call when a task starts."""
        self.data_provider.set_active_task(task_id, task, model)

    def on_task_progress(self, iteration: int, score: float):
        """Call during task execution."""
        self.data_provider.update_task_progress(iteration, score)

    def on_task_complete(self, task_id: str, status: str):
        """Call when task completes."""
        self.data_provider.complete_task(task_id, status)

    def on_model_failure(self, model: Model):
        """Call when a model fails."""
        self.data_provider.record_model_failure(model)

    def on_model_success(self, model: Model):
        """Call when a model succeeds."""
        self.data_provider.record_model_success(model)


if __name__ == "__main__":
    run_enhanced_dashboard()
