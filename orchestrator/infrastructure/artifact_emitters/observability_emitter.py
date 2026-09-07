"""Runtime operability artifact emitter (Phase 7, P-6).

Emits applications that an orchestrator can run and an engineer can
operate: liveness/readiness endpoints, graceful shutdown wired through
uvicorn's own SIGTERM handling (never re-implemented — that would risk
racing uvicorn's own drain logic), structured JSON logging with a
correlation id threaded through every record (wraps the previously-orphan
``generators/logging_generator.py`` rather than reimplementing it, mirroring
how P-4/P-5 wrap ``docker_generator.py``/``cicd_generator.py``), Prometheus
RED metrics, OpenTelemetry tracing gated on an OTLP endpoint being
configured, resilience defaults (request size cap, upstream timeout, a
fixed-window rate limiter — ported in spirit from this orchestrator's own
``resilience.py``/``rate_limiter.py``), and fail-fast config validation.

Archetype-scoped: ``PYTHON_SERVICE``/``FULLSTACK`` get the full stack;
``PYTHON_CLI`` gets structured logging and config validation only, with no
HTTP artifacts. ``LIBRARY``/``WEB_STATIC`` are not applicable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain.readiness import AppArchetype
from ...domain.testing_models import Workspace
from ...generators.logging_generator import LoggingConfigBuilder, LogFormat

_HTTP_ARCHETYPES = (AppArchetype.PYTHON_SERVICE, AppArchetype.FULLSTACK)
_ALL_ARCHETYPES = (*_HTTP_ARCHETYPES, AppArchetype.PYTHON_CLI)

_CONFIG_PY_SERVICE = '''"""Application configuration — fails fast at import time (Phase 7, P-6).

``Settings()`` is instantiated eagerly at module import, so a missing
required value raises ``pydantic.ValidationError`` at import/boot — never
silently deferred to whichever request first touches it.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "__SERVICE_NAME__"
    otel_service_name: str = "__SERVICE_NAME__"
    secret_key: str = Field(description="Required application secret — no default on purpose")
    max_request_bytes: int = 10 * 1024 * 1024
    request_timeout_s: float = 30.0
    rate_limit_requests_per_window: int = 100
    rate_limit_window_s: float = 60.0


settings = Settings()

__all__ = ["Settings", "settings"]
'''

_CONFIG_PY_CLI = '''"""Application configuration — fails fast at import time (Phase 7, P-6).

``Settings()`` is instantiated eagerly at module import, so a missing
required value raises ``pydantic.ValidationError`` at import/boot — never
silently deferred to first use.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "__SERVICE_NAME__"
    secret_key: str = Field(description="Required application secret — no default on purpose")


settings = Settings()

__all__ = ["Settings", "settings"]
'''

_HEALTH_PY = '''"""Health and readiness endpoints (Phase 7, P-6).

``/health`` is liveness only — no dependency calls, stays cheap by
construction. ``/ready`` reflects real startup/dependency state via
``ReadinessState`` and is never a copy of ``/health``.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import JSONResponse

from .observability import readiness_state

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    checks = readiness_state.run_checks()
    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not_ready", "checks": checks},
    )


__all__ = ["router"]
'''

_OBSERVABILITY_PY = '''"""Runtime operability middleware: correlation ids, resilience defaults,
Prometheus RED metrics, and OpenTelemetry tracing (Phase 7, P-6).

Ported in spirit from this orchestrator's own resilience.py/rate_limiter.py
— simplified, single-process, in-memory implementations appropriate for a
generated scaffold rather than the orchestrator's own multi-provider
concerns.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

from fastapi import APIRouter, FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


# ═══════════════════════════════════════════════════════════════
# Readiness state — distinct from liveness by construction
# ═══════════════════════════════════════════════════════════════


@dataclass
class ReadinessState:
    """Tracks startup completion plus optional named dependency checks.

    A future data-layer emitter (P-7) registers a migration-state check
    here; today startup completion alone is enough to make ``/ready``
    genuinely diverge from ``/health`` during boot and shutdown.
    """

    _checks: dict = field(default_factory=dict)
    _started: bool = False

    def mark_started(self) -> None:
        self._started = True

    def mark_stopped(self) -> None:
        self._started = False

    def register_check(self, name: str, fn: Callable[[], bool]) -> None:
        self._checks[name] = fn

    def run_checks(self) -> dict:
        result = {"startup_complete": self._started}
        for name, fn in self._checks.items():
            try:
                result[name] = bool(fn())
            except Exception:
                result[name] = False
        return result


readiness_state = ReadinessState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup marks readiness true; shutdown marks it false. uvicorn keeps
    in-flight connections open until this generator resumes past ``yield``,
    which is what makes it safe to rely on uvicorn's own SIGTERM handler
    here instead of re-implementing one.
    """

    readiness_state.mark_started()
    yield
    readiness_state.mark_stopped()


# ═══════════════════════════════════════════════════════════════
# Correlation id middleware
# ═══════════════════════════════════════════════════════════════


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = correlation_id_var.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers["X-Request-ID"] = correlation_id
        return response


# ═══════════════════════════════════════════════════════════════
# Resilience defaults
# ═══════════════════════════════════════════════════════════════


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int):
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_bytes:
            return PlainTextResponse("request entity too large", status_code=413)
        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout_s: float):
        super().__init__(app)
        self._timeout_s = timeout_s

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            return PlainTextResponse("upstream timeout", status_code=504)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window, in-memory, per-process limiter. Swap for a shared
    backend (Redis, etc.) before running more than one replica.
    """

    def __init__(self, app: ASGIApp, requests_per_window: int, window_s: float):
        super().__init__(app)
        self._limit = requests_per_window
        self._window_s = window_s
        self._hits: dict = {}

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start, count = self._hits.get(client, (now, 0))
        if now - window_start > self._window_s:
            window_start, count = now, 0
        count += 1
        self._hits[client] = (window_start, count)
        if count > self._limit:
            return PlainTextResponse("rate limit exceeded", status_code=429)
        return await call_next(request)


# ═══════════════════════════════════════════════════════════════
# Prometheus RED metrics
# ═══════════════════════════════════════════════════════════════

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ["method", "path"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
        REQUEST_DURATION.labels(request.method, path).observe(duration)
        return response


metrics_router = APIRouter()


@metrics_router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ═══════════════════════════════════════════════════════════════
# OpenTelemetry tracing — OTLP export configured by env
# ═══════════════════════════════════════════════════════════════


def configure_tracing(service_name: str) -> None:
    """No-op unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set — tracing stays
    off rather than exporting nowhere or crashing on a dependency the
    operator never opted into.
    """

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


__all__ = [
    "correlation_id_var",
    "ReadinessState",
    "readiness_state",
    "lifespan",
    "CorrelationIdMiddleware",
    "RequestSizeLimitMiddleware",
    "RequestTimeoutMiddleware",
    "RateLimitMiddleware",
    "MetricsMiddleware",
    "metrics_router",
    "configure_tracing",
]
'''

_MAIN_PY = '''"""Application entrypoint (Phase 7, P-6).

Wires health/ready, resilience defaults, correlation ids, metrics, and
tracing around the ASGI app. Single owner of this wiring layer: a
business-logic generator should ``include_router`` its own routes on
``app`` rather than redefining this file's middleware stack.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.health import router as health_router
from app.logging_config import setup_logger
from app.observability import (
    CorrelationIdMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    RequestTimeoutMiddleware,
    configure_tracing,
    lifespan,
    metrics_router,
)

setup_logger("__SERVICE_NAME__")
configure_tracing(settings.otel_service_name)

app = FastAPI(title=settings.service_name, lifespan=lifespan)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_window=settings.rate_limit_requests_per_window,
    window_s=settings.rate_limit_window_s,
)
app.add_middleware(RequestTimeoutMiddleware, timeout_s=settings.request_timeout_s)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
app.add_middleware(MetricsMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.include_router(health_router)
app.include_router(metrics_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
'''


def _build_logging_config() -> str:
    return (
        LoggingConfigBuilder()
        .for_python()
        .with_standard_logging()
        .with_format(LogFormat.JSON)
        .with_correlation_id()
        .build()
    )


class ObservabilityEmitter:
    """Emits the runtime-operability layer: config, logging, and — for
    HTTP archetypes — health/ready, middleware, and the entrypoint.
    """

    name = "observability"

    def applies_to(self, archetype: AppArchetype) -> bool:
        return archetype in _ALL_ARCHETYPES

    async def emit(self, workspace: Workspace, profile: Any) -> list[str]:
        archetype = getattr(profile, "archetype", None) or AppArchetype.PYTHON_SERVICE
        service_name = getattr(profile, "project_name", None) or workspace.root.name

        app_dir = workspace.root / "app"
        app_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = [self._write(app_dir / "__init__.py", "")]

        config_template = _CONFIG_PY_SERVICE if archetype in _HTTP_ARCHETYPES else _CONFIG_PY_CLI
        written.append(
            self._write(
                app_dir / "config.py", config_template.replace("__SERVICE_NAME__", service_name)
            )
        )
        written.append(self._write(app_dir / "logging_config.py", _build_logging_config()))

        if archetype in _HTTP_ARCHETYPES:
            written.append(self._write(app_dir / "observability.py", _OBSERVABILITY_PY))
            written.append(self._write(app_dir / "health.py", _HEALTH_PY))
            written.append(
                self._write(
                    workspace.root / "main.py", _MAIN_PY.replace("__SERVICE_NAME__", service_name)
                )
            )

        return [p.relative_to(workspace.root).as_posix() for p in written]

    def _write(self, path: Path, content: str) -> Path:
        path.write_text(content, encoding="utf-8")
        return path


__all__ = ["ObservabilityEmitter"]
