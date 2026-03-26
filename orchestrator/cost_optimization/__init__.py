"""
Optimization Module
====================
Author: Georgios-Chrysovalantis Chatzivantsidis

Production cost & performance optimizations for AI Orchestrator.

Tiers:
- Tier 1: Provider-Level Cost Optimizations (80-90% input cost reduction)
- Tier 2: Architectural Optimizations (40-60% per-task reduction)
- Tier 3: Quality Optimizations (30-50% fewer repair cycles)
- Tier 4: DevOps Optimizations (Security + DX)

Usage:
    from orchestrator.optimization import PromptCacher, BatchClient, TokenBudget

    cacher = PromptCacher()
    batch = BatchClient()
    budget = TokenBudget()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..log_config import get_logger

logger = get_logger(__name__)


class OptimizationPhase(str, Enum):
    """Phases where optimizations apply."""
    DECOMPOSITION = "decomposition"
    GENERATION = "generation"
    CRITIQUE = "critique"
    EVALUATION = "evaluation"
    PROMPT_ENHANCEMENT = "prompt_enhancement"
    CONDENSING = "condensing"


@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization effectiveness."""
    # Prompt caching
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Batch API
    batch_calls: int = 0
    realtime_calls: int = 0
    batch_savings: float = 0.0

    # Token budget
    input_tokens_saved: int = 0
    output_tokens_saved: int = 0
    estimated_cost_savings: float = 0.0

    # Cascading
    cascade_attempts: int = 0
    cascade_exits_early: int = 0
    cascade_avg_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for telemetry."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "batch_calls": self.batch_calls,
            "realtime_calls": self.realtime_calls,
            "batch_savings": self.batch_savings,
            "input_tokens_saved": self.input_tokens_saved,
            "output_tokens_saved": self.output_tokens_saved,
            "estimated_cost_savings": self.estimated_cost_savings,
            "cascade_attempts": self.cascade_attempts,
            "cascade_exits_early": self.cascade_exits_early,
            "cascade_avg_score": self.cascade_avg_score,
        }


@dataclass
class OptimizationConfig:
    """Configuration for all optimizations."""
    # Prompt caching
    enable_prompt_caching: bool = True
    cache_warming_enabled: bool = True

    # Batch API
    enable_batch_api: bool = True
    batch_phases: List[OptimizationPhase] = field(default_factory=lambda: [
        OptimizationPhase.EVALUATION,
        OptimizationPhase.PROMPT_ENHANCEMENT,
        OptimizationPhase.CONDENSING,
    ])

    # Token budget
    enable_token_budget: bool = True
    output_token_limits: Dict[str, int] = field(default_factory=lambda: {
        "decomposition": 2000,
        "generation": 4000,
        "critique": 800,
        "evaluation": 500,
        "prompt_enhancement": 500,
        "condensing": 1000,
    })

    # Model cascading
    enable_cascading: bool = False
    cascade_chains: Dict[str, List[tuple]] = field(default_factory=lambda: {
        "code_generation": [
            ("deepseek-v3.2", 0.80),
            ("claude-sonnet-4.6", 0.75),
            ("claude-opus-4.6", 0.0),
        ],
        "code_review": [
            ("deepseek-v3.2", 0.75),
            ("claude-sonnet-4.6", 0.70),
            ("claude-opus-4.6", 0.0),
        ],
    })

    # Speculative generation
    enable_speculative: bool = False
    speculative_threshold: float = 0.85

    # Streaming validation
    enable_streaming_validation: bool = False

    # Adaptive temperature
    enable_adaptive_temperature: bool = True
    temperature_strategy: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "decomposition": {"initial": 0.0, "retry_1": 0.2, "retry_2": 0.4},
        "generation": {"initial": 0.0, "retry_1": 0.1, "retry_2": 0.3},
        "critique": {"initial": 0.3, "retry_1": 0.5, "retry_2": 0.7},
        "creative": {"initial": 0.7, "retry_1": 0.9, "retry_2": 1.0},
    })

    # Dependency context injection
    enable_dependency_context: bool = True

    # Auto eval dataset
    enable_auto_eval_dataset: bool = True
    
    # ═══════════════════════════════════════════════════════
    # Paradigm Shift Enhancements (Phase 1)
    # ═══════════════════════════════════════════════════════
    
    # TDD-First Generation
    enable_tdd_first: bool = False  # Opt-in until proven
    
    # Diff-Based Revisions
    enable_diff_revisions: bool = True  # Default on (60% token savings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enable_prompt_caching": self.enable_prompt_caching,
            "enable_batch_api": self.enable_batch_api,
            "enable_token_budget": self.enable_token_budget,
            "enable_cascading": self.enable_cascading,
            "enable_speculative": self.enable_speculative,
            "enable_streaming_validation": self.enable_streaming_validation,
            "enable_adaptive_temperature": self.enable_adaptive_temperature,
            "enable_dependency_context": self.enable_dependency_context,
            "enable_auto_eval_dataset": self.enable_auto_eval_dataset,
            "enable_tdd_first": self.enable_tdd_first,
            "enable_diff_revisions": self.enable_diff_revisions,
            "output_token_limits": self.output_token_limits,
        }


# Default configuration
DEFAULT_CONFIG = OptimizationConfig()


def get_optimization_config() -> OptimizationConfig:
    """Get current optimization configuration."""
    return DEFAULT_CONFIG


def update_config(config: OptimizationConfig) -> None:
    """Update optimization configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    logger.info(f"Optimization config updated: caching={config.enable_prompt_caching}, "
                f"batch={config.enable_batch_api}, budget={config.enable_token_budget}")


# Import optimization modules for convenience
from .prompt_cache import PromptCacher, CacheMetrics, warm_prompt_cache
from .batch_client import BatchClient, BatchStatus, BatchMetrics, batch_call
from .token_budget import TokenBudget, TokenUsage, TokenBudgetMetrics, get_token_limit
from .model_cascading import ModelCascader, CascadeMetrics, CascadeResult, cascading_generate
from .speculative_gen import SpeculativeGenerator, SpeculativeMetrics, SpeculativeResult, speculative_generate
from .streaming_validator import StreamingValidator, StreamingMetrics, StreamingResult, stream_and_validate
from .structured_output import (
    StructuredOutputEnforcer,
    TaskSpec,
    DecompositionOutput,
    CritiqueOutput,
    EvaluationOutput,
    CodeReviewOutput,
    PromptEnhancementOutput,
    CondensingOutput,
)
from .dependency_context import DependencyContextInjector, DependencyContext, ContextMetrics, inject_dependency_context
from .tier3_quality import (
    AdaptiveTemperatureController,
    TemperatureMetrics,
    EvalDatasetBuilder,
    EvalTestCase,
    DatasetMetrics,
    generate_with_adaptive_temp,
)
from .docker_sandbox import DockerSandbox, ExecutionResult, SandboxMetrics, execute_in_sandbox
from .github_push import GitHubIntegration, CommitMetadata, PushResult, GitHubMetrics, push_to_github

__all__ = [
    # Core
    "OptimizationPhase",
    "OptimizationMetrics",
    "OptimizationConfig",
    "get_optimization_config",
    "update_config",
    # Tier 1: Prompt caching
    "PromptCacher",
    "CacheMetrics",
    "warm_prompt_cache",
    # Tier 1: Batch processing
    "BatchClient",
    "BatchStatus",
    "BatchMetrics",
    "batch_call",
    # Tier 1: Token budget
    "TokenBudget",
    "TokenUsage",
    "TokenBudgetMetrics",
    "get_token_limit",
    # Tier 2: Model cascading
    "ModelCascader",
    "CascadeMetrics",
    "CascadeResult",
    "cascading_generate",
    # Tier 2: Speculative generation
    "SpeculativeGenerator",
    "SpeculativeMetrics",
    "SpeculativeResult",
    "speculative_generate",
    # Tier 2: Streaming validation
    "StreamingValidator",
    "StreamingMetrics",
    "StreamingResult",
    "stream_and_validate",
    # Tier 3: Structured output
    "StructuredOutputEnforcer",
    "TaskSpec",
    "DecompositionOutput",
    "CritiqueOutput",
    "EvaluationOutput",
    "CodeReviewOutput",
    "PromptEnhancementOutput",
    "CondensingOutput",
    # Tier 3: Dependency context
    "DependencyContextInjector",
    "DependencyContext",
    "ContextMetrics",
    "inject_dependency_context",
    # Tier 3: Quality
    "AdaptiveTemperatureController",
    "TemperatureMetrics",
    "EvalDatasetBuilder",
    "EvalTestCase",
    "DatasetMetrics",
    "generate_with_adaptive_temp",
    # Tier 4: Docker sandbox
    "DockerSandbox",
    "ExecutionResult",
    "SandboxMetrics",
    "execute_in_sandbox",
    # Tier 4: GitHub integration
    "GitHubIntegration",
    "CommitMetadata",
    "PushResult",
    "GitHubMetrics",
    "push_to_github",
]
