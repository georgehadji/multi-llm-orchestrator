"""
Centralized Constants for AI Orchestrator
==========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

All magic numbers and configuration constants are defined here.
This module provides a single source of truth for all tunable parameters.

Usage:
    from orchestrator.constants import (
        SEMANTIC_CACHE_QUALITY,
        DEFAULT_TIMEOUT_S,
        EMA_DECAY,
    )
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# QUALITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

# Semantic cache quality threshold (0.0-1.0)
# Responses below this quality score are not cached
SEMANTIC_CACHE_QUALITY = 0.85

# Convergence threshold for iterative improvement (0.0-1.0)
# Stop iterating when quality score exceeds this
CONVERGENCE_THRESHOLD = 0.95

# Default power/confidence score (0.0-1.0)
DEFAULT_POWER = 0.80

# Minimum quality score for production-ready output
PRODUCTION_QUALITY_MIN = 0.90

# Plateau detection threshold (quality improvement below this is considered plateau)
PLATEAU_THRESHOLD = 0.02


# ═══════════════════════════════════════════════════════════════════
# TIMEOUTS (seconds)
# ═══════════════════════════════════════════════════════════════════

# Default timeout for standard API calls
DEFAULT_TIMEOUT_S = 60

# Timeout for reasoning models (longer context, more computation)
REASONING_TIMEOUT_S = 3600

# Timeout for decomposition tasks
DECOMPOSITION_TIMEOUT_S = 120

# Timeout for critique tasks
CRITIQUE_TIMEOUT_S = 90

# Timeout for revision tasks
REVISION_TIMEOUT_S = 120

# Cache connection timeout
CACHE_TIMEOUT_S = 10

# Cache connection TTL (time-to-live)
CACHE_TTL_S = 3600  # 1 hour


# ═══════════════════════════════════════════════════════════════════
# TOKEN LIMITS
# ═══════════════════════════════════════════════════════════════════

# Default max tokens for standard tasks
DEFAULT_MAX_TOKENS = 1500

# Max tokens for decomposition tasks
DECOMPOSITION_MAX_TOKENS = 4096

# Max tokens for critique tasks
CRITIQUE_MAX_TOKENS = 2048

# Max tokens for revision tasks
REVISION_MAX_TOKENS = 4096

# Max tokens for code generation
CODE_GEN_MAX_TOKENS = 8192

# Max tokens for code review
CODE_REVIEW_MAX_TOKENS = 4096

# Max tokens for reasoning tasks
REASONING_MAX_TOKENS = 4096

# Max tokens for writing tasks
WRITING_MAX_TOKENS = 4096

# Max tokens for data extraction
DATA_EXTRACT_MAX_TOKENS = 2048

# Max tokens for summarization
SUMMARIZE_MAX_TOKENS = 1024

# Max tokens for evaluation
EVALUATE_MAX_TOKENS = 2048

# Instructor max characters (for structured outputs)
INSTRUCTOR_MAX_CHARS = 8000  # ~2K tokens


# ═══════════════════════════════════════════════════════════════════
# TRUNCATION & SNIPPETS
# ═══════════════════════════════════════════════════════════════════

# Error snippet length for logs
ERROR_SNIPPET_CHARS = 200

# Log snippet length for debugging
LOG_SNIPPET_CHARS = 500

# Project context sample length
CONTEXT_SAMPLE_CHARS = 2000

# Maximum project description length before falling back to manual parsing
PROJECT_DESCRIPTION_MAX_CHARS = 8000


# ═══════════════════════════════════════════════════════════════════
# EMA (EXPONENTIAL MOVING AVERAGE) PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# EMA decay rate for telemetry updates
EMA_DECAY = 0.1

# EMA alpha for latency updates
EMA_ALPHA_LATENCY = 0.2

# EMA alpha for quality updates
EMA_ALPHA_QUALITY = 0.2

# EMA alpha for cost updates
EMA_ALPHA_COST = 0.2

# Trust factor penalty on failure
TRUST_PENALTY = 0.95

# Trust factor reward on success
TRUST_REWARD = 1.001

# Trust factor cap
TRUST_FACTOR_CAP = 1.0

# Success rate rolling window size
SUCCESS_RATE_WINDOW = 10

# P95 latency buffer size
P95_LATENCY_BUFFER = 50


# ═══════════════════════════════════════════════════════════════════
# BUDGET PARTITIONS
# ═══════════════════════════════════════════════════════════════════

# Default budget allocation by phase
BUDGET_DECOMPOSITION = 0.10  # 10% for decomposition
BUDGET_GENERATION = 0.60  # 60% for generation
BUDGET_CROSS_REVIEW = 0.15  # 15% for cross-review
BUDGET_EVALUATION = 0.10  # 10% for evaluation
BUDGET_RESERVE = 0.05  # 5% reserve

# Soft cap multiplier (warn when exceeded)
BUDGET_SOFT_CAP_MULTIPLIER = 2.0


# ═══════════════════════════════════════════════════════════════════
# RETRY & BACKOFF
# ═══════════════════════════════════════════════════════════════════

# Maximum retry attempts
MAX_RETRIES = 3

# Initial backoff time (seconds)
INITIAL_BACKOFF_S = 1.0

# Maximum backoff time (seconds)
MAX_BACKOFF_S = 60.0

# Backoff multiplier
BACKOFF_MULTIPLIER = 2.0

# Jitter factor (0.0-1.0)
BACKOFF_JITTER = 0.1


# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

# Consecutive failures before circuit opens
CIRCUIT_BREAKER_FAILURES = 3

# Circuit breaker timeout (seconds)
CIRCUIT_BREAKER_TIMEOUT_S = 60

# Health check interval (seconds)
HEALTH_CHECK_INTERVAL_S = 30


# ═══════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════

# Default requests per minute
DEFAULT_RPM = 60

# Default tokens per minute
DEFAULT_TPM = 100000

# Rate limit tier: budget
BUDGET_RPM = 30
BUDGET_TPM = 50000

# Rate limit tier: standard
STANDARD_RPM = 60
STANDARD_TPM = 100000

# Rate limit tier: premium
PREMIUM_RPM = 120
PREMIUM_TPM = 200000


# ═══════════════════════════════════════════════════════════════════
# CONCURRENCY
# ═══════════════════════════════════════════════════════════════════

# Default max concurrent tasks
DEFAULT_MAX_CONCURRENCY = 3

# Minimum max concurrent tasks
MIN_MAX_CONCURRENCY = 1

# Maximum max concurrent tasks
MAX_MAX_CONCURRENCY = 10


# ═══════════════════════════════════════════════════════════════════
# TDD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# TDD enabled by default
TDD_ENABLED = True

# TDD quality tier: budget | balanced | premium
TDD_DEFAULT_QUALITY_TIER = "balanced"

# Maximum TDD iterations
TDD_MAX_ITERATIONS = 3

# Minimum test coverage threshold
TDD_MIN_COVERAGE = 0.80


# ═══════════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════════

# Minimum security score for production readiness
SECURITY_MIN_SCORE = 85.0

# Block on critical security findings
SECURITY_BLOCK_CRITICAL = True


# ═══════════════════════════════════════════════════════════════════
# COST OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

# Enable prompt caching
ENABLE_PROMPT_CACHING = True

# Enable batch API
ENABLE_BATCH_API = True

# Enable token budget
ENABLE_TOKEN_BUDGET = True

# Enable model cascading
ENABLE_CASCADING = False

# Enable speculative generation
ENABLE_SPECULATIVE = False

# Enable streaming validation
ENABLE_STREAMING_VALIDATION = True

# Enable adaptive temperature
ENABLE_ADAPTIVE_TEMPERATURE = True

# Enable dependency context
ENABLE_DEPENDENCY_CONTEXT = True

# Enable auto-eval dataset
ENABLE_AUTO_EVAL = True

# Enable diff-based revisions
ENABLE_DIFF_REVISIONS = True

# Enable TDD-first
ENABLE_TDD_FIRST = True


# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Log date format
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ═══════════════════════════════════════════════════════════════════
# MISC
# ═══════════════════════════════════════════════════════════════════

# Database path
DEFAULT_DB_PATH = "~/.orchestrator_cache"

# Cache database name
CACHE_DB_NAME = "cache.db"

# State database name
STATE_DB_NAME = "state.db"

# Log directory
LOG_DIR = "logs"

# Output directory
DEFAULT_OUTPUT_DIR = "./results"

# Default project budget (USD)
DEFAULT_BUDGET_USD = 8.0

# Default project time limit (seconds)
DEFAULT_TIME_LIMIT_S = 5400  # 1.5 hours

# Resume detection window (seconds)
RESUME_WINDOW_S = 86400  # 24 hours
