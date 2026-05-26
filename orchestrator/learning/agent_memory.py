"""Per-agent private memory tracking."""
from __future__ import annotations
import os
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


    def save(self, path: str | None = None) -> None:
        import json
        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "agent_memories", f"{self.agent_id}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = {
            "agent_id": self.agent_id,
            "successes": self.successes[-50:],
            "failures": self.failures[-50:],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, agent_id: str, path: str | None = None) -> "AgentMemory":
        import json, os
        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "agent_memories", f"{agent_id}.json")
        if not os.path.exists(p):
            return cls(agent_id=agent_id)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(agent_id=data.get("agent_id", agent_id),
                       successes=data.get("successes", []),
                       failures=data.get("failures", []))
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls(agent_id=agent_id)
