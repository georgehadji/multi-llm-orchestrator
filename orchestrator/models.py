"""
Multi-LLM Orchestrator — Core Models & Types
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis
All data structures, enums, routing tables, cost tables, budget logic.
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
    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O4_MINI = "o4-mini"                       # OpenAI o4-mini — cost-effective reasoning
    
    # Google Gemini models
    GEMINI_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"  # Cheapest Gemini for simple tasks
    
    # Moonshot Kimi models
    KIMI_K2_5 = "kimi-k2.5"
    
    # MiniMax models
    MINIMAX_TEXT_01 = "MiniMax-Text-01"       # MiniMax-Text-01 — efficient reasoning, cost-effective
    
    # Zhipu GLM models
    GLM_4_PLUS = "glm-4-plus"                 # Zhipu GLM-4-Plus — enhanced general tasks
    GLM_4_FLASH = "glm-4-flash"               # GLM-4-Flash — FREE tier model
    
    # DeepSeek models
    DEEPSEEK_CODER = "deepseek-coder"         # DeepSeek-Coder — code-specialized model
    DEEPSEEK_REASONER = "deepseek-reasoner"   # DeepSeek-R1 — o1-class reasoning model


class ProjectStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    COMPLETED_DEGRADED = "COMPLETED_DEGRADED"
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

from functools import lru_cache

@lru_cache(maxsize=128)
def get_provider(model: Model) -> str:
    """
    Get provider name for a model.
    
    Uses LRU cache for O(1) repeated lookups.
    Cache size 128 covers all current + future models.
    """
    val = model.value
    if val.startswith("gpt"):
        return "openai"
    elif val.startswith("gemini"):
        return "google"
    elif val.startswith("moonshot") or val.startswith("kimi"):
        return "kimi"
    elif val.startswith("minimax"):
        return "minimax"
    elif val.startswith("zai") or val.startswith("glm"):
        return "zhipu"
    elif val.startswith("deepseek"):
        return "deepseek"
    return "unknown"


# ─────────────────────────────────────────────
# Cost table (per 1M tokens, USD)
# ─────────────────────────────────────────────

COST_TABLE: dict[Model, dict[str, float]] = {
    # OpenAI models
    Model.GPT_4O:             {"input": 2.50,  "output": 10.0},
    Model.GPT_4O_MINI:        {"input": 0.15,  "output": 0.60},
    Model.O4_MINI:            {"input": 1.50,  "output": 6.00},  # OpenAI o4-mini
    
    # Google Gemini models
    Model.GEMINI_PRO:         {"input": 1.25,  "output": 10.0},
    Model.GEMINI_FLASH:       {"input": 0.15,  "output": 0.60},
    Model.GEMINI_FLASH_LITE:  {"input": 0.075, "output": 0.30},  # Cheapest Gemini
    
    # Moonshot Kimi models
    Model.KIMI_K2_5:          {"input": 0.14,  "output": 0.56},  # Cheapest overall
    
    # MiniMax models
    Model.MINIMAX_TEXT_01:    {"input": 0.50,  "output": 1.50},
    
    # Zhipu GLM models
    Model.GLM_4_PLUS:         {"input": 0.50,  "output": 2.00},
    Model.GLM_4_FLASH:        {"input": 0.00,  "output": 0.00},  # FREE tier
    
    # DeepSeek models (cache-miss rates)
    Model.DEEPSEEK_CODER:     {"input": 0.27,  "output": 1.10},  # Best cost/performance
    Model.DEEPSEEK_REASONER:  {"input": 0.55,  "output": 2.19},  # o1-class reasoning
}


# ─────────────────────────────────────────────
# Routing table (priority-ordered per task type)
# ─────────────────────────────────────────────

ROUTING_TABLE: dict[TaskType, list[Model]] = {
    # OPTIMIZED 2025-02: Based on latest pricing and benchmarks
    # 
    # CODE_GEN: DeepSeek #1 (best cost/quality), GPT-4o as premium fallback
    TaskType.CODE_GEN:     [
        Model.DEEPSEEK_CODER,      # $0.27/$1.10 — best cost/performance for code
        Model.GPT_4O,               # $2.50/$10 — premium quality fallback
        Model.GPT_4O_MINI,          # $0.15/$0.60 — budget option
        Model.GEMINI_FLASH,         # $0.15/$0.60 — 1M context
        Model.KIMI_K2_5,            # $0.14/$0.56 — cheapest but slow
    ],
    
    # CODE_REVIEW: Similar to CODE_GEN but without expensive fallbacks
    TaskType.CODE_REVIEW:  [
        Model.DEEPSEEK_CODER,      # Best for code understanding
        Model.GPT_4O,               # Premium quality
        Model.GPT_4O_MINI,          # Fast & cheap
        Model.GEMINI_FLASH,         # 1M context for large reviews
    ],
    
    # REASONING: DeepSeek-R1 competes with o1 at 1/10th cost
    TaskType.REASONING:    [
        Model.DEEPSEEK_REASONER,    # $0.55/$2.19 — o1-class reasoning
        Model.GPT_4O,               # $2.50/$10 — premium
        Model.O4_MINI,              # $1.50/$6.00 — OpenAI reasoning
        Model.GEMINI_PRO,           # $1.25/$10 — 1M context
        Model.MINIMAX_TEXT_01,      # $0.50/$1.50 — alternative
    ],
    
    # WRITING: GPT-4o leads, GLM-4-Flash as FREE fallback
    TaskType.WRITING:      [
        Model.GPT_4O,               # Best writing quality
        Model.GEMINI_PRO,           # 1M context for long docs
        Model.DEEPSEEK_CODER,       # Good & cheap
        Model.GLM_4_FLASH,          # FREE — ultimate budget option
    ],
    
    # DATA_EXTRACT: Gemini Flash-Lite cheapest, GLM-4-Flash FREE
    TaskType.DATA_EXTRACT: [
        Model.GEMINI_FLASH_LITE,    # $0.075/$0.30 — cheapest option
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
        Model.GEMINI_FLASH,         # $0.15/$0.60 — 1M context
        Model.GLM_4_FLASH,          # FREE — budget fallback
        Model.DEEPSEEK_CODER,       # Accurate when needed
    ],
    
    # SUMMARIZE: Same optimization as DATA_EXTRACT
    TaskType.SUMMARIZE:    [
        Model.GEMINI_FLASH_LITE,    # Cheapest
        Model.GEMINI_FLASH,         # 1M context
        Model.GPT_4O_MINI,          # Reliable
        Model.GLM_4_FLASH,          # FREE
    ],
    
    # EVALUATE: GPT-4o most reliable for evaluation tasks
    TaskType.EVALUATE:     [
        Model.GPT_4O,               # Most reliable evaluator
        Model.DEEPSEEK_CODER,       # Good & cheap
        Model.O4_MINI,              # Reasoning capabilities
        Model.MINIMAX_TEXT_01,      # Alternative
    ],
}


# ─────────────────────────────────────────────
# Fallback chains (always cross-provider)
# ─────────────────────────────────────────────

FALLBACK_CHAIN: dict[Model, Model] = {
    # OPTIMIZED 2025-02: Cross-provider fallbacks for resilience
    # Each fallback goes to a different provider to maximize availability
    
    # OpenAI fallbacks → DeepSeek (cost-effective, same quality tier)
    Model.GPT_4O:              Model.DEEPSEEK_CODER,     # Premium → DeepSeek
    Model.GPT_4O_MINI:         Model.GEMINI_FLASH,       # Budget → Gemini
    Model.O4_MINI:             Model.DEEPSEEK_REASONER,  # Reasoning → DeepSeek-R1
    
    # Gemini fallbacks → OpenAI or DeepSeek
    Model.GEMINI_PRO:          Model.GPT_4O,             # Pro → GPT-4o
    Model.GEMINI_FLASH:        Model.GPT_4O_MINI,        # Flash → GPT-4o-mini
    Model.GEMINI_FLASH_LITE:   Model.GLM_4_FLASH,        # Lite → FREE GLM-4-Flash
    
    # Kimi fallback → DeepSeek (both Chinese providers, but DeepSeek better)
    Model.KIMI_K2_5:           Model.DEEPSEEK_CODER,     # Kimi → DeepSeek
    
    # MiniMax fallback → OpenAI
    Model.MINIMAX_TEXT_01:     Model.GPT_4O,             # Minimax → GPT-4o
    
    # GLM fallbacks → OpenAI or Gemini
    Model.GLM_4_PLUS:          Model.GEMINI_PRO,         # GLM+ → Gemini Pro
    Model.GLM_4_FLASH:         Model.GEMINI_FLASH_LITE,  # FREE GLM → Cheapest Gemini
    
    # DeepSeek fallbacks → OpenAI (premium escalation)
    Model.DEEPSEEK_CODER:      Model.GPT_4O,             # DeepSeek → GPT-4o
    Model.DEEPSEEK_REASONER:   Model.O4_MINI,            # R1 → o4-mini (both reasoning)
}


# ─────────────────────────────────────────────
# Per-task thresholds & limits
# ─────────────────────────────────────────────

DEFAULT_THRESHOLDS: dict[TaskType, float] = {
    TaskType.DATA_EXTRACT: 0.90,
    TaskType.SUMMARIZE:    0.80,
    TaskType.CODE_GEN:     0.85,
    # CODE_REVIEW: lowered from 0.85 — review quality depends on how much
    # source context the LLM received, which is often partial due to truncation.
    # 0.75 is a realistic target; scores above this reflect genuine analysis.
    TaskType.CODE_REVIEW:  0.75,
    TaskType.REASONING:    0.90,
    TaskType.WRITING:      0.80,
    # EVALUATE: lowered from 0.95 — evaluation outputs are open-ended prose;
    # scoring ≥ 0.95 requires near-perfect structured responses which LLMs
    # rarely produce without a domain-specific rubric.
    TaskType.EVALUATE:     0.80,
}

MAX_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     8192,  # raised: avoid unterminated strings mid-class
    TaskType.CODE_REVIEW:  4096,  # raised: full analysis without truncation
    TaskType.REASONING:    4096,
    TaskType.WRITING:      4096,
    TaskType.DATA_EXTRACT: 2048,
    TaskType.SUMMARIZE:    1024,
    TaskType.EVALUATE:     2048,  # raised: evaluation tasks need more room
}


def get_max_iterations(task_type: TaskType) -> int:
    if task_type == TaskType.CODE_GEN:
        return 3  # heavy — needs generate + critique + revise
    if task_type == TaskType.CODE_REVIEW:
        return 4  # extra iteration: reviews depend on context quality and
                  # often need a full second pass after critique improves framing
    if task_type == TaskType.REASONING:
        return 3
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
    # App Builder fields (Improvement 8)
    target_path: str = ""     # e.g. "src/routes/auth.py"
    module_name: str = ""     # e.g. "src.routes.auth"
    tech_context: str = ""    # brief note on tech stack for this file

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
    attempt_history: list["AttemptRecord"] = field(default_factory=list)


@dataclass
class AttemptRecord:
    """Records one failed iteration attempt so the next retry has failure context."""
    attempt_num: int          # 1-based
    model_used: str           # Model.value — str for easy serialization
    output_snippet: str       # first 200 chars of output
    failure_reason: str       # human-readable description
    validators_failed: list[str] = field(default_factory=list)


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
# JobSpec — App Builder job specification
# ─────────────────────────────────────────────

@dataclass
class JobSpec:
    """
    App Builder job specification.

    A lightweight spec for the App Builder pipeline, separate from the
    policy-oriented JobSpec in policy.py.

    Fields
    ------
    description      : Human-readable description of the app to build.
    success_criteria : Acceptance criteria for the build.
    app_type         : Optional override for the app type (e.g. "fastapi",
                       "cli", "library").  Empty string means auto-detect.
    docker           : Whether to run Docker-based verification.
    output_dir       : Where to write the generated app.  Empty string means
                       auto-generate a temp directory.
    """
    description: str
    success_criteria: str
    app_type: str = ""
    docker: bool = False
    output_dir: str = ""


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


def build_default_profiles() -> "dict[Model, ModelProfile]":
    """
    Build a ModelProfile for every Model enum value using the static
    COST_TABLE and ROUTING_TABLE as the source of truth.

    Called once at Orchestrator construction time. Telemetry fields
    (quality_score, trust_factor, avg_latency_ms, …) start at their
    defaults and are updated at runtime by TelemetryCollector.

    Lazy-imports ModelProfile from policy to avoid a circular import
    (policy.py → models.py, models.py → policy.py would be circular).
    """
    # Lazy import to avoid circular dependency: policy.py imports models.py
    from .policy import ModelProfile  # noqa: PLC0415

    # Build capability map: {TaskType → priority_rank} for each model
    # Priority rank = index in ROUTING_TABLE list (0 = highest priority)
    capability_map: dict[Model, dict[TaskType, int]] = {m: {} for m in Model}
    for task_type, model_list in ROUTING_TABLE.items():
        for rank, model in enumerate(model_list):
            capability_map[model][task_type] = rank

    profiles: dict[Model, ModelProfile] = {}
    for model in Model:
        costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})
        profiles[model] = ModelProfile(
            model=model,
            provider=get_provider(model),
            cost_per_1m_input=costs["input"],
            cost_per_1m_output=costs["output"],
            capable_task_types=capability_map.get(model, {}),
        )
    return profiles
