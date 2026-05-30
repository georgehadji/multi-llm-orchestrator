"""NexusScopeProfiler — core profiling API."""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

from .config import NexusScopeConfig
from .session import ProfileSession, SessionRingBuffer

logger = logging.getLogger("orchestrator.nexusscope")

_PROFILER_INSTANCE: NexusScopeProfiler | None = None


class NexusScopeProfiler:
    """Statistical profiler wrapping pyinstrument."""
    def __init__(self, config: NexusScopeConfig | None = None):
        self._config = config or NexusScopeConfig()
        self._buffer = SessionRingBuffer(capacity=self._config.buffer_size)
        self._pyinstrument = None
        self._import_attempted = False

    def _import_pyinstrument(self):
        if self._import_attempted:
            return
        self._import_attempted = True
        try:
            import pyinstrument
            self._pyinstrument = pyinstrument
        except ImportError:
            logger.warning("pyinstrument not installed")

    def _start_profiler(self):
        if not self._config.enabled:
            return None
        self._import_pyinstrument()
        if self._pyinstrument is None:
            return None
        p = self._pyinstrument.Profiler(
            interval=self._config.interval,
            async_mode="enabled" if self._config.async_mode else "disabled",
        )
        p.start()
        return p

    @contextmanager
    def session(self, name: str) -> Generator[ProfileSession, None, None]:
        session = ProfileSession(name=name)
        p = self._start_profiler()
        try:
            yield session
        finally:
            session.finished_at = time.time()
            if p:
                p.stop()
                session._profiler = p
            self._buffer.push(session)

    @asynccontextmanager
    async def async_session(self, name: str) -> AsyncGenerator[ProfileSession, None]:
        session = ProfileSession(name=name)
        p = self._start_profiler()
        try:
            yield session
        finally:
            session.finished_at = time.time()
            if p:
                p.stop()
                session._profiler = p
            self._buffer.push(session)

    def get_sessions(self, name=None, last_n=None):
        if name:
            sessions = self._buffer.by_name(name)
        else:
            sessions = self._buffer.all()
        if last_n:
            sessions = sessions[-last_n:]
        return sessions

    def render_last(self, name=None, fmt="text"):
        sessions = self.get_sessions(name=name, last_n=1)
        if not sessions:
            return "No sessions" if fmt == "text" else {}
        p = sessions[-1]._profiler
        if p is None:
            return "Profiling disabled" if fmt == "text" else {}
        if fmt == "text":
            return p.output_text(unicode=True, color=True)
        elif fmt == "html":
            return p.output_html()
        return str(p.output_text())


def get_profiler() -> NexusScopeProfiler:
    global _PROFILER_INSTANCE
    if _PROFILER_INSTANCE is None:
        _PROFILER_INSTANCE = NexusScopeProfiler()
    return _PROFILER_INSTANCE
