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
