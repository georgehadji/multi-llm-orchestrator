"""
Configuration Management
=======================

Type-safe, validated configuration using Pydantic.

Usage:
    from orchestrator.config import OrchestratorConfig, get_config
    
    config = get_config()
    
    # Access settings
    print(config.default_budget)
    print(config.log_level)
    
    # Check feature flags
    if config.enable_feedback_loop:
        feedback_loop = FeedbackLoop()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from enum import Enum

# Try to import pydantic
try:
    from pydantic import BaseSettings, Field, validator, root_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback base class
    class BaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(*args, **kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Classes
# ═══════════════════════════════════════════════════════════════════════════════

class CacheBackend(Enum):
    """Cache backend options."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class EventBusBackend(Enum):
    """Event bus backend options."""
    MEMORY = "memory"
    SQLITE = "sqlite"
    REDIS = "redis"


if HAS_PYDANTIC:
    class OrchestratorConfig(BaseSettings):
        """
        Type-safe, validated configuration for the orchestrator.
        
        Settings can be loaded from:
        1. Environment variables (prefix: ORCHESTRATOR_)
        2. .env file
        3. Constructor arguments
        """
        
        # Core Settings
        default_budget: float = Field(
            default=5.0,
            gt=0,
            le=1000,
            description="Default budget for projects in USD",
        )
        
        max_concurrency: int = Field(
            default=3,
            ge=1,
            le=50,
            description="Maximum concurrent API calls",
        )
        
        max_parallel_tasks: int = Field(
            default=3,
            ge=1,
            le=20,
            description="Maximum parallel task execution",
        )
        
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO",
            description="Logging level",
        )
        
        # Feature Flags
        enable_feedback_loop: bool = Field(
            default=True,
            description="Enable production feedback loop",
        )
        
        enable_outcome_router: bool = Field(
            default=True,
            description="Enable outcome-weighted routing",
        )
        
        enable_plugin_isolation: bool = Field(
            default=False,
            description="Enable plugin process isolation (experimental)",
        )
        
        enable_streaming: bool = Field(
            default=True,
            description="Enable streaming for large projects",
        )
        
        enable_event_bus: bool = Field(
            default=True,
            description="Enable event-driven architecture",
        )
        
        # Cache Configuration
        cache_backend: CacheBackend = Field(
            default=CacheBackend.DISK,
            description="Cache backend to use",
        )
        
        cache_ttl_seconds: int = Field(
            default=3600,
            ge=60,
            description="Default cache TTL in seconds",
        )
        
        cache_memory_max_mb: int = Field(
            default=100,
            ge=10,
            description="Max memory for L1 cache in MB",
        )
        
        cache_disk_max_mb: int = Field(
            default=1000,
            ge=100,
            description="Max disk space for L3 cache in MB",
        )
        
        # Event Bus Configuration
        event_bus_backend: EventBusBackend = Field(
            default=EventBusBackend.SQLITE,
            description="Event bus backend",
        )
        
        event_store_path: str = Field(
            default=".events/event_store.db",
            description="Path to event store database",
        )
        
        # Plugin Security
        plugin_allow_network: bool = Field(
            default=True,
            description="Allow plugins to make network requests",
        )
        
        plugin_allow_filesystem: bool = Field(
            default=True,
            description="Allow plugins filesystem access",
        )
        
        plugin_timeout_seconds: float = Field(
            default=30.0,
            ge=1,
            le=300,
            description="Plugin execution timeout",
        )
        
        plugin_memory_limit_mb: int = Field(
            default=512,
            ge=64,
            description="Plugin memory limit in MB",
        )
        
        # Health Checks
        health_check_interval: int = Field(
            default=30,
            ge=5,
            description="Health check interval in seconds",
        )
        
        health_check_timeout: float = Field(
            default=5.0,
            ge=1,
            le=30,
            description="Health check timeout in seconds",
        )
        
        # Provider Settings
        deepseek_api_key: Optional[str] = Field(
            default=None,
            description="DeepSeek API key",
        )
        
        openai_api_key: Optional[str] = Field(
            default=None,
            description="OpenAI API key",
        )
        
        google_api_key: Optional[str] = Field(
            default=None,
            description="Google/Gemini API key",
        )
        
        anthropic_api_key: Optional[str] = Field(
            default=None,
            description="Anthropic API key",
        )
        
        # Paths
        data_dir: str = Field(
            default=".orchestrator",
            description="Directory for persistent data",
        )
        
        results_dir: str = Field(
            default="./results",
            description="Directory for project results",
        )
        
        # Telemetry
        telemetry_enabled: bool = Field(
            default=True,
            description="Enable telemetry collection",
        )
        
        telemetry_endpoint: Optional[str] = Field(
            default=None,
            description="Optional telemetry endpoint URL",
        )
        
        # Validation
        @validator('deepseek_api_key', 'openai_api_key', 'google_api_key', 'anthropic_api_key')
        def validate_key_format(cls, v):
            if v and not v.startswith(('sk-', 'sk-proj-', 'AIza')):
                raise ValueError("Invalid API key format")
            return v
        
        @validator('log_level')
        def validate_log_level(cls, v):
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v not in valid_levels:
                raise ValueError(f"Invalid log level: {v}")
            return v
        
        class Config:
            env_prefix = "ORCHESTRATOR_"
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            use_enum_values = True
    
else:
    # Fallback configuration without Pydantic
    class OrchestratorConfig:
        """Simple configuration class without validation."""
        
        def __init__(self, **kwargs):
            # Core Settings
            self.default_budget = kwargs.get('default_budget', 5.0)
            self.max_concurrency = kwargs.get('max_concurrency', 3)
            self.max_parallel_tasks = kwargs.get('max_parallel_tasks', 3)
            self.log_level = kwargs.get('log_level', 'INFO')
            
            # Feature Flags
            self.enable_feedback_loop = kwargs.get('enable_feedback_loop', True)
            self.enable_outcome_router = kwargs.get('enable_outcome_router', True)
            self.enable_plugin_isolation = kwargs.get('enable_plugin_isolation', False)
            self.enable_streaming = kwargs.get('enable_streaming', True)
            self.enable_event_bus = kwargs.get('enable_event_bus', True)
            
            # Cache
            self.cache_backend = kwargs.get('cache_backend', CacheBackend.DISK)
            self.cache_ttl_seconds = kwargs.get('cache_ttl_seconds', 3600)
            self.cache_memory_max_mb = kwargs.get('cache_memory_max_mb', 100)
            self.cache_disk_max_mb = kwargs.get('cache_disk_max_mb', 1000)
            
            # Event Bus
            self.event_bus_backend = kwargs.get('event_bus_backend', EventBusBackend.SQLITE)
            self.event_store_path = kwargs.get('event_store_path', '.events/event_store.db')
            
            # Plugin Security
            self.plugin_allow_network = kwargs.get('plugin_allow_network', True)
            self.plugin_allow_filesystem = kwargs.get('plugin_allow_filesystem', True)
            self.plugin_timeout_seconds = kwargs.get('plugin_timeout_seconds', 30.0)
            self.plugin_memory_limit_mb = kwargs.get('plugin_memory_limit_mb', 512)
            
            # Health Checks
            self.health_check_interval = kwargs.get('health_check_interval', 30)
            self.health_check_timeout = kwargs.get('health_check_timeout', 5.0)
            
            # Provider Settings
            self.deepseek_api_key = kwargs.get('deepseek_api_key')
            self.openai_api_key = kwargs.get('openai_api_key')
            self.google_api_key = kwargs.get('google_api_key')
            self.anthropic_api_key = kwargs.get('anthropic_api_key')
            
            # Paths
            self.data_dir = kwargs.get('data_dir', '.orchestrator')
            self.results_dir = kwargs.get('results_dir', './results')
            
            # Telemetry
            self.telemetry_enabled = kwargs.get('telemetry_enabled', True)
            self.telemetry_endpoint = kwargs.get('telemetry_endpoint')
        
        @classmethod
        def from_env(cls):
            """Load configuration from environment."""
            kwargs = {}
            
            # Map env vars to kwargs
            mappings = {
                'ORCHESTRATOR_DEFAULT_BUDGET': 'default_budget',
                'ORCHESTRATOR_MAX_CONCURRENCY': 'max_concurrency',
                'ORCHESTRATOR_LOG_LEVEL': 'log_level',
                'ORCHESTRATOR_DEEPSEEK_API_KEY': 'deepseek_api_key',
                'ORCHESTRATOR_OPENAI_API_KEY': 'openai_api_key',
                'ORCHESTRATOR_GOOGLE_API_KEY': 'google_api_key',
            }
            
            for env_var, key in mappings.items():
                value = os.environ.get(env_var)
                if value:
                    # Type conversion
                    if key in ['default_budget']:
                        value = float(value)
                    elif key in ['max_concurrency', 'max_parallel_tasks']:
                        value = int(value)
                    elif key in ['enable_feedback_loop', 'enable_outcome_router']:
                        value = value.lower() == 'true'
                    
                    kwargs[key] = value
            
            return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Global Configuration
# ═══════════════════════════════════════════════════════════════════════════════

_config: Optional[OrchestratorConfig] = None


def get_config() -> OrchestratorConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        if HAS_PYDANTIC:
            _config = OrchestratorConfig()
        else:
            _config = OrchestratorConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset global configuration."""
    global _config
    _config = None


def configure(**kwargs) -> OrchestratorConfig:
    """Configure with explicit values."""
    global _config
    _config = OrchestratorConfig(**kwargs)
    return _config


def load_from_file(path: str) -> OrchestratorConfig:
    """Load configuration from file."""
    global _config
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Parse based on extension
    if path.suffix == '.json':
        import json
        with open(path) as f:
            data = json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    _config = OrchestratorConfig(**data)
    return _config


def save_to_file(config: OrchestratorConfig, path: str) -> None:
    """Save configuration to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all settings as dict
    if HAS_PYDANTIC:
        data = config.dict()
    else:
        data = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_')
        }
    
    if path.suffix == '.json':
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_provider_config() -> Dict[str, Any]:
    """Get provider API configuration."""
    config = get_config()
    
    return {
        'deepseek': {'api_key': config.deepseek_api_key},
        'openai': {'api_key': config.openai_api_key},
        'google': {'api_key': config.google_api_key},
        'anthropic': {'api_key': config.anthropic_api_key},
    }


def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration."""
    config = get_config()
    
    return {
        'backend': config.cache_backend,
        'ttl_seconds': config.cache_ttl_seconds,
        'memory_max_mb': config.cache_memory_max_mb,
        'disk_max_mb': config.cache_disk_max_mb,
    }


def get_plugin_config() -> Dict[str, Any]:
    """Get plugin security configuration."""
    config = get_config()
    
    return {
        'allow_network': config.plugin_allow_network,
        'allow_filesystem': config.plugin_allow_filesystem,
        'timeout_seconds': config.plugin_timeout_seconds,
        'memory_limit_mb': config.plugin_memory_limit_mb,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Example
# ═══════════════════════════════════════════════════════════════════════════════

def example():
    """Example configuration usage."""
    # Get default config
    config = get_config()
    
    print(f"Default Budget: ${config.default_budget}")
    print(f"Log Level: {config.log_level}")
    print(f"Cache Backend: {config.cache_backend}")
    
    # Check feature flags
    if config.enable_feedback_loop:
        print("Feedback loop is enabled")
    
    # Configure explicitly
    new_config = configure(
        default_budget=10.0,
        log_level="DEBUG",
        enable_plugin_isolation=True,
    )
    
    print(f"New Budget: ${new_config.default_budget}")


if __name__ == "__main__":
    example()
