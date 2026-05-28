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

from pydantic_settings import BaseSettings, SettingsConfigDict
# Re-export static defaults from orchestrator/config.py (TASK 501)
try:
    from ..config import (
        TIMEOUT_DEFAULT_SECONDS as TIMEOUT_SECONDS,
        TOKENS_MAX_OUTPUT as MAX_TOKENS_OUTPUT,
        BUDGET_DEFAULT_USD as DEFAULT_BUDGET_USD,
    )
except ImportError:
    TIMEOUT_SECONDS = 120
    MAX_TOKENS_OUTPUT = 4096
    DEFAULT_BUDGET_USD = 10.0



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
    a2a_enabled: bool = True                # A2A multi-agent protocol
    accountability_enabled: bool = True     # Accountability / audit trail
    agent_safety_enabled: bool = True       # Agent safety monitor
    red_team_enabled: bool = True           # Red-team adversarial testing
    tdd_enabled: bool = True                # TDD-first generator
    diff_generation_enabled: bool = True    # Diff-based generation
    test_validation_enabled: bool = True    # Test validator (HAS_TEST_VALIDATOR)
    code_validation_enabled: bool = True    # Code output validator (HAS_CODE_VALIDATOR)
    cost_optimization_enabled: bool = True  # Cost-optimisation tier 1-4
    cache_optimizer_enabled: bool = True    # Cache optimiser (HAS_CACHE_OPTIMIZER)
    tracing_enabled: bool = False           # OpenTelemetry tracing (needs extra deps)
    skill_optimization_enabled: bool = False  # SkillOpt: self-improving per-TaskType skill docs

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

    model_config = SettingsConfigDict(
        env_prefix="ORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ── Singleton instances (created once at import time) ──
flags = FeatureFlags()
settings = OrchestratorSettings()
