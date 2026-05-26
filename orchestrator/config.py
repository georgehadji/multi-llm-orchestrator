"""
RuntimeConfig — Single source of truth for all hard-coded values
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Consolidates all hard-coded values from across the codebase into one
centralized configuration. Every module that previously had inline magic
numbers now imports from here. Values are grouped by function.

This module uses zero external dependencies — importable by any module
without circular import risk.
"""

from __future__ import annotations

# ── Timeouts (seconds) ──────────────────────────────────────────────

class Timeout:
    """All timeout values in seconds."""

    # API calls
    API_CALL_SHORT: float = 30.0       # Basic LLM call
    API_CALL_MEDIUM: float = 60.0      # Critique / evaluation
    API_CALL_LONG: float = 120.0       # Decomposition / architecture
    API_CALL_EXTREME: float = 300.0    # Very large operations

    # Circuit breakers
    CB_RESET_TIMEOUT: float = 60.0     # Circuit breaker auto-reset
    CB_HALF_OPEN_RECOVERY: float = 30.0

    # Streaming
    STREAM_POLL: float = 0.01          # Streaming poll interval
    STREAM_QUEUE_GET: float = 0.1      # Queue.get() timeout
    STREAM_FINISH: float = 1.0         # Wait for stream to finish

    # Shutdown / lifecycle
    SHUTDOWN_WAIT: float = 5.0         # Graceful shutdown wait
    SESSION_CLEANUP: float = 3.0       # Session cleanup timeout

    # Generation phases
    GENERATE_STANDARD: int = 180       # Standard code generation
    GENERATE_EXTENDED: int = 240       # Extended code generation
    CRITIQUE: int = 240               # Critique phase
    EVALUATE: int = 60                 # Evaluation phase
    DECOMPOSE: int = 160               # Decomposition phase


# ── Token Limits ─────────────────────────────────────────────────────

class TokenLimits:
    """Standard token limits for LLM calls."""

    # Generation
    TINY: int = 150           # Simple queries
    SHORT: int = 200          # Brief responses
    COMPACT: int = 300        # Compact responses
    SMALL: int = 500          # Small prompts
    MEDIUM: int = 1000        # Medium complexity
    STANDARD: int = 2000      # Standard responses
    CODE_SHORT: int = 1500    # Short code generation
    CODE_STANDARD: int = 4096 # Standard code generation
    CODE_LONG: int = 8192     # Complex code / architecture
    VERY_LONG: int = 4000     # Long-form responses

    # Context
    CONTEXT_TRUNCATION: int = 8192   # Default context truncation
    DECOMPOSE_CONTEXT: int = 1200    # Decomposition context

    # System prompt
    SYSTEM_SHORT: int = 512
    SYSTEM_STANDARD: int = 2048


# ── Budget & Limits ─────────────────────────────────────────────────

class BudgetDefaults:
    """Default budget and concurrency values."""

    MAX_USD_DEFAULT: float = 8.0       # CLI default budget
    MAX_USD_DRY_RUN: float = 1.0       # Dry-run budget
    MAX_CONCURRENCY: int = 3           # Default API concurrency
    MAX_PARALLEL_TASKS: int = 3        # Max parallel task execution

    # Budget fractions
    ARA_FRACTION: float = 0.3          # Max budget for ARA methods
    COMPETITIVE_FRACTION: float = 0.3  # Max budget for competitive mode


# ── Quality Thresholds ───────────────────────────────────────────────

class QualityThresholds:
    """Quality decision thresholds (0.0-1.0)."""

    ACCEPT: float = 0.7            # Acceptable quality
    RETRY_TRIGGER: float = 0.7     # Below this → retry
    BLOCK_VERIFICATION: float = 0.5 # PersuasionDefense block threshold
    SEMANTIC_CACHE: float = 0.85   # Semantic cache similarity
    ARA_DECISION: float = 0.5      # Below → escalate to ARA
    PREFLIGHT_WARN: float = 0.6    # Preflight WARN threshold
    PREFLIGHT_BLOCK: float = 0.3   # Preflight BLOCK threshold


# ── Retry & Resilience ───────────────────────────────────────────────

class RetryDefaults:
    """Default retry behavior."""

    MAX_ATTEMPTS: int = 3          # Standard retry attempts
    MAX_ATTEMPTS_FAST: int = 1     # Fast fail
    MAX_ATTEMPTS_DEVELOPER: int = 3  # Developer self-correction
    MIN_WAIT: float = 0.1          # Min wait between retries
    MAX_WAIT: float = 30.0         # Max wait between retries
    EXPONENTIAL_BASE: float = 2.0  # Exponential backoff base

    # Circuit breaker
    CB_FAILURE_THRESHOLD: int = 3  # Failures before opening circuit


# ── Codebase Analysis ────────────────────────────────────────────────

class AnalysisDefaults:
    """Defaults for codebase analysis."""

    MAX_CONTEXT_TOKENS: int = 60000    # Max tokens for context
    MAX_SECTION_TOKENS: int = 4096     # Max tokens per analysis section
    MAX_CONCURRENCY: int = 2           # Max concurrent API calls


# ── Experience & Learning ────────────────────────────────────────────

class LearningDefaults:
    """Defaults for the learning system."""

    MAX_PATTERNS: int = 200            # Max patterns before eviction
    MAX_FAILURES: int = 50             # Max failure records
    COMPRESSION_THRESHOLD: int = 10    # Min patterns before compression
    PATTERN_WINDOW: int = 10           # Sliding window for averages
    SUCCESS_WINDOW: int = 50           # Recent successes to track
    FAILURE_WINDOW: int = 50           # Recent failures to track


# ── Agent Defaults ───────────────────────────────────────────────────

class AgentDefaults:
    """Defaults for agentic system."""

    MAX_ITERATIONS: int = 3            # Default agent iterations
    MAX_PARALLEL_AGENTS: int = 3       # Max concurrent agents
    MIN_CONFIDENCE: float = 0.5        # Minimum confidence to proceed


# ── Backward compatibility: OpenRouter feature flags ─────────────────
# NEW: see RuntimeConfig classes above for all other values.
# This is maintained for api_clients.py compatibility.

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class OpenRouterOptimizations:
    USE_JSON_SCHEMA_RESPONSES: bool = False
    USE_MODEL_VARIANTS: bool = False
    USE_NATIVE_FALLBACKS: bool = False
    USE_PROVIDER_SORTING: bool = False
    USE_STREAMING: bool = False
    USE_EMBEDDING_CACHE: bool = False

    @classmethod
    def from_env(cls):
        return cls(
            USE_JSON_SCHEMA_RESPONSES=os.getenv("USE_JSON_SCHEMA_RESPONSES", "false").lower() == "true",
            USE_MODEL_VARIANTS=os.getenv("USE_MODEL_VARIANTS", "false").lower() == "true",
            USE_NATIVE_FALLBACKS=os.getenv("USE_NATIVE_FALLBACKS", "false").lower() == "true",
            USE_PROVIDER_SORTING=os.getenv("USE_PROVIDER_SORTING", "false").lower() == "true",
            USE_STREAMING=os.getenv("USE_STREAMING", "false").lower() == "true",
            USE_EMBEDDING_CACHE=os.getenv("USE_EMBEDDING_CACHE", "false").lower() == "true",
        )

OPENROUTER_OPTS = OpenRouterOptimizations.from_env()


@dataclass(frozen=True)
class MemoryConfig:
    """Cross-session memory persistence configuration."""
    memory_dir: str = os.path.join(os.path.expanduser("~"), ".orchestrator") if hasattr(os, "path") else ".orchestrator"
    auto_save_interval: int = 50  # tasks between auto-saves
    max_patterns: int = 200
    cache_ttl_hours: int = 1
