"""
Crosscutting Configuration — Feature Flags & Settings
=======================================================
Centralizes all environment-derived configuration into pydantic models.

Usage:
    from orchestrator.crosscutting.config import flags, settings

    if flags.context_compression:
        compressor.enable()

    max_concurrency = settings.max_concurrency

Environment variables use the ORCH_ prefix by convention.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Re-export static defaults from orchestrator/config.py (TASK 501).
# These names never existed there (real ones are namespaced under Timeout /
# TokenLimits / BudgetDefaults) so this always silently hit the except branch
# with a stale $10 fallback that didn't even match the real $8 default
# (hunt T16) — fixed to import the real attributes.
try:
    from ..config import Timeout, TokenLimits, BudgetDefaults

    TIMEOUT_SECONDS = Timeout.API_CALL_LONG
    MAX_TOKENS_OUTPUT = TokenLimits.CODE_STANDARD
    DEFAULT_BUDGET_USD = BudgetDefaults.MAX_USD_DEFAULT
except ImportError:
    TIMEOUT_SECONDS = 120
    MAX_TOKENS_OUTPUT = 4096
    DEFAULT_BUDGET_USD = 8.0


class FeatureFlags(BaseSettings):
    """All feature flags — single source of truth.

    Each flag is read from environment variables with the ORCH_ prefix.
    Example: ORCH_CONTEXT_COMPRESSION=true sets context_compression=True.

    Extra env vars (OPENROUTER_API_KEY, etc.) are ignored — only ORCH_* flags are read.
    """

    context_compression: bool = False
    pattern_injection: bool = False
    batch_parallelism: bool = False
    batch_concurrency: int = 3
    plugin_sandbox: bool = True
    audit_log: bool = True
    use_json_schema_responses: bool = False
    use_model_variants: bool = False
    use_native_fallbacks: bool = False
    use_provider_sorting: bool = False
    use_streaming: bool = False
    use_embedding_cache: bool = False

    # ── Optional advanced feature gates (P4-2) ────────────────────────────
    # Each flag controls whether engine.py attempts to import the corresponding
    # optional module at startup.  Default True preserves existing behaviour;
    # set to False (e.g. ORCH_A2A_ENABLED=false) to skip the import entirely.
    # This makes the feature surface explicit and testable.
    a2a_enabled: bool = True  # A2A multi-agent protocol
    accountability_enabled: bool = True  # Accountability / audit trail
    agent_safety_enabled: bool = True  # Agent safety monitor
    red_team_enabled: bool = True  # Red-team adversarial testing
    tdd_enabled: bool = True  # TDD-first generator
    diff_generation_enabled: bool = True  # Diff-based generation
    test_validation_enabled: bool = True  # Test validator (HAS_TEST_VALIDATOR)
    code_validation_enabled: bool = True  # Code output validator (HAS_CODE_VALIDATOR)
    cost_optimization_enabled: bool = True  # Cost-optimisation tier 1-4
    cache_optimizer_enabled: bool = True  # Cache optimiser (HAS_CACHE_OPTIMIZER)
    tracing_enabled: bool = False  # OpenTelemetry tracing (needs extra deps)
    skill_optimization_enabled: bool = False  # SkillOpt: self-improving per-TaskType skill docs

    # ── Bilevel Autoresearch gates ─────────────────────────────────────
    bilevel_autoresearch_enabled: bool = False  # Master switch for outer-loop self-improvement
    bilevel_tabu_enabled: bool = False  # Tabu Search mechanism for retry diversity
    bilevel_level15_enabled: bool = False  # Search-strategy tuner (Level 1.5)

    # ── HTTP server gates ──────────────────────────────────────────────────
    http_ingest_enabled: bool = True  # Enable /execute endpoints
    http_stream_enabled: bool = True  # Enable SSE streaming /projects/{id}/stream
    http_stream_polling: bool = True  # Use polling SSE fallback (True) instead of event-bus (False)

    # ── Secondary optional modules ────────────────────────────────────────────
    session_watcher_enabled: bool = True  # Session lifecycle watcher
    persona_enabled: bool = True  # Persona / role manager
    memory_tier_enabled: bool = True  # Multi-tier memory manager
    bm25_search_enabled: bool = True  # BM25 keyword search index
    reranker_enabled: bool = True  # LLM-based result reranker
    session_lifecycle_enabled: bool = True  # Session lifecycle manager
    task_verifier_enabled: bool = True  # Task output verifier
    token_optimizer_enabled: bool = True  # Token usage optimizer

    # ── Objective verifier gates (Router Enhancements Phase 1) ───────────
    use_objective_verifiers: bool = False  # E1: wire Verifier port into cascade + self-consistency

    # ── Verbalized Sampling gates (CodeWhale Phase 0) ────────────────────
    vs_map_elites_seeding: bool = False  # Phase 2: VS-tail seed MAP-Elites grid
    vs_retry_escape: bool = False  # Phase 3: VS-tail escape from stuck retries
    vs_test_generation: bool = False  # Phase 4: VS for test/synthetic data
    vs_generate: bool = False  # Phase 6: VS-first GenerateStage
    vs_architecture: bool = False  # Phase 7a: VS for architecture selection
    vs_code_review: bool = False  # Phase 7b: VS for multi-hypothesis code review
    vs_decomposition: bool = False  # Phase 7c: VS for task planning
    vs_bug_hunting: bool = False  # Phase 7d: VS for Bayesian bug hunting
    vs_k: int = 5  # Default candidates per VS call

    # ── taste-skill design quality gates ──────────────────────────────────
    taste_skill_enabled: bool = True  # Inject taste-skill anti-slop prefix for frontend tasks
    image_reference_pipeline: bool = False  # Pre-generate text visual context before code gen

    # ── Reranking gates ───────────────────────────────────────────────────
    vs_reranking_enabled: bool = False  # CandidateSelector for VS: score top-k candidates
    knowledge_rerank_enabled: bool = False  # Two-stage KB recall: cosine → LLM rerank

    # ── Adaptive Capability Router (ACR) — Phase 0 seam ───────────────────
    # off    = GreedyBackend (default; unchanged behaviour)
    # shadow = ACR computes + logs its pick; GreedyBackend's pick is still authoritative
    # on     = ACR is authoritative
    acr_backend: Literal["off", "shadow", "on"] = "off"

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OrchestratorSettings(BaseSettings):
    """Runtime settings for the orchestrator.

    Environment variables use the ORCH_ prefix.
    """

    max_concurrency: int = 3
    max_parallel_tasks: int = 3
    default_budget_usd: float = 10.0
    default_timeout_seconds: int = 10800
    context_truncation_limit: int = 40000
    rate_limit_per_minute: int = 60
    cache_ttl_hours: int = 48
    semantic_cache_threshold: float = 0.85
    dashboard_port: int = 8000
    dashboard_host: str = "127.0.0.1"
    mcp_port: int = 8181
    mcp_host: str = "0.0.0.0"
    mcp_http_mode: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    audit_log_path: str = "~/.orchestrator_cache/audit.log"

    # ── taste-skill design dials (1–10) ───────────────────────────────────
    design_variance: int = Field(default=5, ge=1, le=10)  # Layout experimentation
    motion_intensity: int = Field(default=5, ge=1, le=10)  # Animation depth
    visual_density: int = Field(default=5, ge=1, le=10)  # Info per viewport

    # ── Paths & models ────────────────────────────────────────────────────
    cache_home: str = ""  # Override ~/.orchestrator_cache (ORCH_CACHE_HOME)
    compression_model: str = ""  # Override context compression model

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def env_str(key: str, default: str = "") -> str:
        """Centralized env-var reader — prefer this over os.getenv."""
        import os

        return os.environ.get(f"ORCH_{key}", default)

    @staticmethod
    def env_int(key: str, default: int = 0) -> int:
        """Centralized env-var reader for integer values."""
        import os

        try:
            return int(os.environ.get(f"ORCH_{key}", str(default)))
        except (TypeError, ValueError):
            return default


# ── Singleton instances (created once at import time) ──
flags = FeatureFlags()
settings = OrchestratorSettings()
