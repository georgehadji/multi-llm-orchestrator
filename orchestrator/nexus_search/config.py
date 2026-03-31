"""
Nexus Search — Configuration Management
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Configuration management for Nexus Search.
"""

from __future__ import annotations

from .models import NexusConfig

# Global config instance
_config: NexusConfig | None = None


def get_config() -> NexusConfig:
    """
    Get Nexus Search configuration.

    Returns:
        NexusConfig instance
    """
    global _config
    if _config is None:
        _config = NexusConfig.from_env()
    return _config


def configure(
    enabled: bool | None = None,
    api_url: str | None = None,
    timeout: int | None = None,
    max_results: int | None = None,
    rate_limit: int | None = None,
    cache_enabled: bool | None = None,
    cache_ttl: int | None = None,
) -> NexusConfig:
    """
    Configure Nexus Search.

    Args:
        enabled: Enable/disable Nexus Search
        api_url: Nexus API URL
        timeout: Request timeout (seconds)
        max_results: Maximum results per query
        rate_limit: Queries per minute
        cache_enabled: Enable result caching
        cache_ttl: Cache TTL (seconds)

    Returns:
        Updated NexusConfig
    """
    global _config

    if _config is None:
        _config = NexusConfig.from_env()

    # Update configuration
    if enabled is not None:
        _config.enabled = enabled
    if api_url is not None:
        _config.api_url = api_url
    if timeout is not None:
        _config.timeout = timeout
    if max_results is not None:
        _config.max_results = max_results
    if rate_limit is not None:
        _config.rate_limit = rate_limit
    if cache_enabled is not None:
        _config.cache_enabled = cache_enabled
    if cache_ttl is not None:
        _config.cache_ttl = cache_ttl

    return _config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = None
