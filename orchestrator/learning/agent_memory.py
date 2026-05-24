"""Per-agent private memory tracking."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class AgentMemory:
    agent_id: str
    successes: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    tool_scores: dict[str, list[float]] = field(default_factory=dict)

    def record(self, task: str, success: bool, score: float):
        rec = {"task": task[:80], "score": score}
        if success:
            self.successes.append(rec)
            self.successes = self.successes[-50:]
        else:
            self.failures.append(rec)
            self.failures = self.failures[-50:]

    def success_rate(self) -> float:
        t = len(self.successes) + len(self.failures)
        return len(self.successes) / t if t else 0.0

    def lesson(self) -> str | None:
        if self.successes and len(self.successes) >= 3:
            avg = sum(s["score"] for s in self.successes[-10:]) / min(10, len(self.successes))
            return f"Agent has {len(self.successes)} successes, avg score {avg:.2f}"
        return None
