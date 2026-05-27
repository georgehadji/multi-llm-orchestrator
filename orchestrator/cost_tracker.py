"""
CostTracker — Per-call cost visibility.
==========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 11, Phase N7 (Newly-inspired): Tracks exact USD cost
per LLM call with token breakdown and cumulative project cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class CallCost:
    """Cost of a single LLM API call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }


class CostTracker:
    """Tracks per-call cost with cumulative totals and history."""

    def __init__(self, storage_dir: str | None = None):
        self._dir = Path(storage_dir or Path.home() / ".orchestrator_cache")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._history: list[CallCost] = []
        self._cumulative: dict[str, CallCost] = {}
        self._load()

    def _load(self) -> None:
        fp = self._dir / "cost_tracker.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._cumulative = {
                    k: CallCost(
                        model=v["model"],
                        input_tokens=v["input_tokens"],
                        output_tokens=v["output_tokens"],
                        cost_usd=v["cost_usd"],
                        latency_ms=v["latency_ms"],
                    )
                    for k, v in data.items()
                }
            except Exception:
                pass

    def _save(self) -> None:
        fp = self._dir / "cost_tracker.json"
        fp.write_text(
            json.dumps(
                {k: v.to_dict() for k, v in self._cumulative.items()},
                indent=2,
            ),
            encoding="utf-8",
        )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float = 0.0,
    ) -> CallCost:
        """Record a single call and update cumulatives."""
        cc = CallCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self._history.append(cc)

        if model not in self._cumulative:
            self._cumulative[model] = CallCost(model=model)
        self._cumulative[model].input_tokens += input_tokens
        self._cumulative[model].output_tokens += output_tokens
        self._cumulative[model].cost_usd += cost_usd
        self._save()
        return cc

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._cumulative.values())

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self._cumulative.values())

    def status_line(self, model: str = "") -> str:
        """Return a one-line cost status."""
        if model and model in self._cumulative:
            c = self._cumulative[model]
            return f"[{c.model}] ${c.cost_usd:.4f} | {c.input_tokens}+{c.output_tokens}tk | {c.latency_ms:.0f}ms"
        return f"Total: ${self.total_cost_usd:.4f} | {self.total_tokens:,} tokens | {len(self._cumulative)} models"

    def per_model(self) -> list[dict]:
        return sorted(
            [v.to_dict() for v in self._cumulative.values()],
            key=lambda x: x["cost_usd"],
            reverse=True,
        )

    def last_call(self) -> CallCost | None:
        return self._history[-1] if self._history else None
