"""Profile session data types."""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class ProfileSession:
    """A single profiling session."""

    name: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _profiler: Any = None

    @property
    def duration_ms(self) -> float:
        end = self.finished_at or time.time()
        return (end - self.started_at) * 1000


class SessionRingBuffer:
    """Fixed-capacity buffer of ProfileSession objects."""

    def __init__(self, capacity: int = 100):
        self._capacity = capacity
        self._sessions: list[ProfileSession] = []

    def push(self, session: ProfileSession) -> None:
        self._sessions.append(session)
        if len(self._sessions) > self._capacity:
            self._sessions.pop(0)

    def all(self) -> list[ProfileSession]:
        return list(self._sessions)

    def by_name(self, name: str) -> list[ProfileSession]:
        return [s for s in self._sessions if s.name == name]

    def last(self, n: int = 1) -> list[ProfileSession]:
        return self._sessions[-n:]

    def __len__(self) -> int:
        return len(self._sessions)
