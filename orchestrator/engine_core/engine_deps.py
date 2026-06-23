"""
engine_deps — All optional/try-except imports for the Orchestrator
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Every try/except import block and corresponding HAS_* flag lives here.
engine.py does `from .engine_deps import *` and gets everything.

Required imports (UnifiedClient, models, etc.) remain in engine.py.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import logging

logger = logging.getLogger("orchestrator.engine_deps")

try:
    from .config import OPENROUTER_OPTS
except ImportError:
    OPENROUTER_OPTS = None
try:
    from .task_schemas import generate_openrouter_schema, get_schema_for_task_type
except ImportError:
    generate_openrouter_schema = None
    get_schema_for_task_type = None

try:
    from .cache_optimizer import CacheConfig, CacheOptimizer

    HAS_CACHE_OPTIMIZER = True
except ImportError:
    HAS_CACHE_OPTIMIZER = False
    CacheOptimizer = None
    CacheConfig = None

# Test validation for reliable test generation
try:
    from .test_validator import TestValidator, validate_and_generate_test

    HAS_TEST_VALIDATOR = True
except ImportError as _e:
    HAS_TEST_VALIDATOR = False
    TestValidator = None
    validate_and_generate_test = None

# Code validation for clean code generation (no LLM commentary)
try:
    from .code_validator import validate_code, extract_code_from_llm_response

    HAS_CODE_VALIDATOR = True
except ImportError:
    HAS_CODE_VALIDATOR = False
    validate_code = None
    extract_code_from_llm_response = None

# Optional advanced features - wrapped in try/except to allow CLI to load even if any fail
# These modules may have external dependencies or circular import issues
HAS_ADVANCED_FEATURES = True

try:
    from .a2a_protocol import A2AManager, AgentCard
except (ImportError, TimeoutError):
    A2AManager = None
    AgentCard = None
    HAS_ADVANCED_FEATURES = False

try:
    from .accountability import AccountabilityTracker, ActionType, ActorType
except (ImportError, TimeoutError):
    AccountabilityTracker = None
    ActionType = None
    ActorType = None

try:
    from .agent_safety import AgentSafetyMonitor, SafetyEventType
except (ImportError, TimeoutError):
    AgentSafetyMonitor = None
    SafetyEventType = None

try:
    from .agents import TaskChannel
except (ImportError, TimeoutError):
    TaskChannel = None

try:
    from .audit import AuditLog
except (ImportError, TimeoutError):
    AuditLog = None

try:
    from .bm25_search import BM25Search, get_bm25_search
except (ImportError, TimeoutError):
    BM25Search = None
    get_bm25_search = None

# OPTIMIZATION: Cost & performance optimizations (Tiers 1-4)
try:
    from .cost_optimization import (
        AdaptiveTemperatureController,
        BatchClient,
        DependencyContextInjector,
        EvalDatasetBuilder,
        ModelCascader,
        OptimizationConfig,
        PromptCacher,
        SpeculativeGenerator,
        StreamingValidator,
        TokenBudget,
        cascading_generate,
        get_optimization_config,
        inject_dependency_context,
        speculative_generate,
        stream_and_validate,
        warm_prompt_cache,
    )
except (ImportError, TimeoutError):
    OptimizationConfig = None
    get_optimization_config = None
    # Set all others to None too
    AdaptiveTemperatureController = None
    BatchClient = None
    DependencyContextInjector = None
    EvalDatasetBuilder = None
    ModelCascader = None
    PromptCacher = None
    SpeculativeGenerator = None
    StreamingValidator = None
    TokenBudget = None
    cascading_generate = None
    inject_dependency_context = None
    speculative_generate = None
    stream_and_validate = None
    warm_prompt_cache = None

try:
    from .hooks import EventType, HookRegistry
except (ImportError, TimeoutError):
    EventType = None
    HookRegistry = None

try:
    from .memory_tier import MemoryTierManager
except (ImportError, TimeoutError):
    MemoryTierManager = None

try:
    from .persona import PersonaManager, PersonaMode
except (ImportError, TimeoutError):
    PersonaManager = None
    PersonaMode = None

try:
    from .planner import ConstraintPlanner
except (ImportError, TimeoutError):
    ConstraintPlanner = None

try:
    from .preflight import PreflightMode, PreflightValidator
except (ImportError, TimeoutError):
    PreflightMode = None
    PreflightValidator = None

try:
    from .rate_limiter import RateLimiter
except (ImportError, TimeoutError):
    RateLimiter = None

try:
    from .red_team import RedTeamFramework
except (ImportError, TimeoutError):
    RedTeamFramework = None

try:
    from .reranker import LLMReranker, get_reranker
except (ImportError, TimeoutError):
    LLMReranker = None
    get_reranker = None

try:
    from .session_lifecycle import SessionLifecycleManager
except (ImportError, TimeoutError):
    SessionLifecycleManager = None

try:
    from .session_watcher import SessionWatcher
except (ImportError, TimeoutError):
    SessionWatcher = None

# NEW: Security & Accountability modules from "Agents of Chaos" paper (arXiv:2602.20021)
try:
    from .task_verifier import TaskVerifier
except (ImportError, TimeoutError):
    TaskVerifier = None

try:
    from .telemetry import TelemetryCollector
except (ImportError, TimeoutError):
    TelemetryCollector = None

try:
    from .telemetry_store import TelemetryStore
except (ImportError, TimeoutError):
    TelemetryStore = None

# NEW: External Projects Integration (RTK, Mnemo Cortex, LiteLLM)
try:
    from .token_optimizer import TokenOptimizer
except (ImportError, TimeoutError):
    TokenOptimizer = None

try:
    from .tracing import TracingConfig, configure_tracing, get_tracer, traced_task
except (ImportError, TimeoutError):
    TracingConfig = None
    configure_tracing = None
    get_tracer = None
    traced_task = None

if TYPE_CHECKING:
    pass

# PARADIGM SHIFT: TDD-First and Diff-Based Generation
try:
    from .test_first_generator import TDDResult, TestFirstGenerator

    HAS_TDD = True
except ImportError:
    HAS_TDD = False
    TestFirstGenerator = None
    TDDResult = None

try:
    from .diff_generator import DiffGenerator, DiffResult, apply_unified_diff

    HAS_DIFF = True
except ImportError:
    HAS_DIFF = False
    DiffGenerator = None
    DiffResult = None
    apply_unified_diff = None

# Context management modules
try:
    from .context_condensing import ContextCondenser
    from .context_dedup import ContextDeduplicator
    from .context_truncator import SmartContextTruncator

    HAS_CONTEXT_MANAGEMENT = True
except ImportError:
    HAS_CONTEXT_MANAGEMENT = False
    ContextCondenser = None
    ContextDeduplicator = None
    SmartContextTruncator = None

# Phase 5: Cross-phase context accumulator

# Phase 6: Test infrastructure — automatically repair failing tests
try:
    from .test_fixer import FixResult, TestFixer

    HAS_TEST_FIXER = True
except ImportError:
    HAS_TEST_FIXER = False
    FixResult = None
    TestFixer = None

# Phase 6: Pre-submission testing gates
try:
    from .pre_submission_testing import PreSubmissionTester, SubmissionResult

    HAS_PRE_SUBMISSION = True
except ImportError:
    HAS_PRE_SUBMISSION = False
    PreSubmissionTester = None
    SubmissionResult = None


# Phase 6+: ARA reasoning pipeline integration
try:
    from .ara_integration import create_ara_integration, ARAPipelineIntegration

    HAS_ARA = True
except ImportError:
    HAS_ARA = False
    create_ara_integration = None
    ARAPipelineIntegration = None
