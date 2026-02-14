"""
Multi-LLM Orchestrator — Core Models & Types
=============================================
All data structures, enums, routing tables, cost tables, budget logic.

Counterfactual: If using plain dicts instead of typed structures →
vulnerability Ψ: typo in model/task names causes silent routing failures.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskType(str, Enum):
    CODE_GEN = "code_generation"
    CODE_REVIEW = "code_review"
    REASONING = "complex_reasoning"
    WRITING = "creative_writing"
    DATA_EXTRACT = "data_extraction"
    SUMMARIZE = "summarization"
    EVALUATE = "evaluation"


class Model(str, Enum):
    CLAUDE_OPUS = "claude-opus-4-6"
    CLAUDE_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GEMINI_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"


class ProjectStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"


# ─────────────────────────────────────────────
# Provider detection
# ─────────────────────────────────────────────

def get_provider(model: Model) -> str:
    val = model.value
    if val.startswith("claude"):
        return "anthropic"
    elif val.startswith("gpt"):
        return "openai"
    elif val.startswith("gemini"):
        return "google"
    return "unknown"


# ─────────────────────────────────────────────
# Cost table (per 1M tokens, USD)
# ─────────────────────────────────────────────

COST_TABLE: dict[Model, dict[str, float]] = {
    Model.CLAUDE_OPUS:   {"input": 15.0,  "output": 75.0},
    Model.CLAUDE_SONNET: {"input": 3.0,   "output": 15.0},
    Model.CLAUDE_HAIKU:  {"input": 0.80,  "output": 4.0},
    Model.GPT_4O:        {"input": 2.50,  "output": 10.0},
    Model.GPT_4O_MINI:   {"input": 0.15,  "output": 0.60},
    Model.GEMINI_PRO:    {"input": 1.25,  "output": 10.0},
    Model.GEMINI_FLASH:  {"input": 0.15,  "output": 0.60},
}


# ─────────────────────────────────────────────
# Routing table (priority-ordered per task type)
# ─────────────────────────────────────────────

ROUTING_TABLE: dict[TaskType, list[Model]] = {
    TaskType.CODE_GEN:     [Model.CLAUDE_SONNET, Model.GPT_4O, Model.GEMINI_PRO],
    TaskType.CODE_REVIEW:  [Model.GPT_4O, Model.CLAUDE_OPUS, Model.GEMINI_PRO],
    TaskType.REASONING:    [Model.CLAUDE_OPUS, Model.GPT_4O, Model.GEMINI_PRO],
    TaskType.WRITING:      [Model.CLAUDE_OPUS, Model.GPT_4O, Model.GEMINI_PRO],
    TaskType.DATA_EXTRACT: [Model.GEMINI_FLASH, Model.GPT_4O_MINI, Model.CLAUDE_HAIKU],
    TaskType.SUMMARIZE:    [Model.GEMINI_FLASH, Model.CLAUDE_HAIKU, Model.GPT_4O_MINI],
    TaskType.EVALUATE:     [Model.CLAUDE_OPUS, Model.GPT_4O, Model.GEMINI_PRO],
}


# ─────────────────────────────────────────────
# Fallback chains (always cross-provider)
# ─────────────────────────────────────────────

FALLBACK_CHAIN: dict[Model, Model] = {
    Model.CLAUDE_OPUS:   Model.GPT_4O,
    Model.CLAUDE_SONNET: Model.GPT_4O,
    Model.CLAUDE_HAIKU:  Model.GPT_4O_MINI,
    Model.GPT_4O:        Model.CLAUDE_OPUS,
    Model.GPT_4O_MINI:   Model.GEMINI_FLASH,
    Model.GEMINI_PRO:    Model.CLAUDE_OPUS,
    Model.GEMINI_FLASH:  Model.GPT_4O_MINI,
}


# ─────────────────────────────────────────────
# Per-task thresholds & limits
# ─────────────────────────────────────────────

DEFAULT_THRESHOLDS: dict[TaskType, float] = {
    TaskType.DATA_EXTRACT: 0.90,
    TaskType.SUMMARIZE:    0.80,
    TaskType.CODE_GEN:     0.85,
    TaskType.CODE_REVIEW:  0.85,
    TaskType.REASONING:    0.90,
    TaskType.WRITING:      0.80,
    TaskType.EVALUATE:     0.95,
}

MAX_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     1500,
    TaskType.CODE_REVIEW:  1200,
    TaskType.REASONING:    1200,
    TaskType.WRITING:      1000,
    TaskType.DATA_EXTRACT: 800,
    TaskType.SUMMARIZE:    400,
    TaskType.EVALUATE:     600,
}


def get_max_iterations(task_type: TaskType) -> int:
    if task_type in (TaskType.REASONING, TaskType.CODE_GEN, TaskType.CODE_REVIEW):
        return 3  # heavy
    return 2  # light


# ─────────────────────────────────────────────
# Budget partitioning (soft caps)
# ─────────────────────────────────────────────

BUDGET_PARTITIONS: dict[str, float] = {
    "decomposition": 0.05,
    "generation":    0.45,
    "cross_review":  0.25,
    "evaluation":    0.15,
    "reserve":       0.10,
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Task:
    id: str
    type: TaskType
    prompt: str
    context: str = ""
    dependencies: list[str] = field(default_factory=list)
    acceptance_threshold: float = 0.85
    max_iterations: int = 3
    max_output_tokens: int = 1500
    status: TaskStatus = TaskStatus.PENDING
    hard_validators: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.acceptance_threshold = DEFAULT_THRESHOLDS.get(
            self.type, self.acceptance_threshold
        )
        self.max_iterations = get_max_iterations(self.type)
        self.max_output_tokens = MAX_OUTPUT_TOKENS.get(
            self.type, self.max_output_tokens
        )


@dataclass
class TaskResult:
    task_id: str
    output: str
    score: float
    model_used: Model
    reviewer_model: Optional[Model] = None
    tokens_used: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0}
    )
    iterations: int = 0
    cost_usd: float = 0.0
    status: TaskStatus = TaskStatus.COMPLETED
    critique: str = ""
    deterministic_check_passed: bool = True
    degraded_fallback_count: int = 0


@dataclass
class Budget:
    max_usd: float = 8.0
    max_time_seconds: float = 5400.0  # 90 min
    spent_usd: float = 0.0
    start_time: float = field(default_factory=time.time)
    phase_spent: dict[str, float] = field(default_factory=lambda: {
        "decomposition": 0.0,
        "generation": 0.0,
        "cross_review": 0.0,
        "evaluation": 0.0,
        "reserve": 0.0,
    })

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_time_seconds - self.elapsed_seconds)

    def can_afford(self, estimated_cost: float) -> bool:
        return self.remaining_usd >= estimated_cost

    def time_remaining(self) -> bool:
        return self.elapsed_seconds < self.max_time_seconds

    def phase_budget(self, phase: str) -> float:
        return self.max_usd * BUDGET_PARTITIONS.get(phase, 0.0)

    def phase_remaining(self, phase: str) -> float:
        return max(0.0, self.phase_budget(phase) - self.phase_spent.get(phase, 0.0))

    def charge(self, amount: float, phase: str = "generation"):
        self.spent_usd += amount
        if phase in self.phase_spent:
            self.phase_spent[phase] += amount

    def to_dict(self) -> dict:
        return {
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "phase_spent": {k: round(v, 4) for k, v in self.phase_spent.items()},
        }


@dataclass
class ProjectState:
    """Full serializable state for resume capability."""
    project_description: str
    success_criteria: str
    budget: Budget
    tasks: dict[str, Task] = field(default_factory=dict)
    results: dict[str, TaskResult] = field(default_factory=dict)
    api_health: dict[str, bool] = field(default_factory=dict)
    status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS
    execution_order: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def prompt_hash(model: str, prompt: str, max_tokens: int,
                system: str = "", temperature: float = 0.3) -> str:
    payload = f"{model}||{system}||{prompt}||{max_tokens}||{temperature}"
    return hashlib.sha256(payload.encode()).hexdigest()


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
