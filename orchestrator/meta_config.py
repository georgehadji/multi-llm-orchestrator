"""
Meta-Optimization Configuration Management
===========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Unified YAML configuration for all meta-optimization features:
- A/B Testing configuration
- HITL configuration
- Gradual Rollout configuration
- Transfer Learning configuration
- Performance configuration
- Monitoring configuration

USAGE:
    from orchestrator.meta_config import MetaOptimizationConfig

    # Load from YAML
    config = MetaOptimizationConfig.from_yaml("meta_config.yaml")

    # Or create programmatically
    config = MetaOptimizationConfig(
        ab_testing=ABTestingConfig(enabled=True, traffic_split=0.1),
        hitl=HITLConfig(auto_approve_low_risk=True),
    )

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid config: {errors}")

    # Save to YAML
    config.to_yaml("meta_config.yaml")
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.meta_config")


# ─────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────


@dataclass
class ConfigError:
    """A configuration error."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


# ─────────────────────────────────────────────
# A/B Testing Configuration
# ─────────────────────────────────────────────


@dataclass
class ABTestingConfig:
    """Configuration for A/B testing."""

    enabled: bool = True
    traffic_split: float = 0.1  # 10% to treatment
    min_samples: int = 30
    max_samples: int = 1000
    significance_level: float = 0.05  # p-value threshold
    early_stopping_enabled: bool = True
    early_stopping_threshold: float = 0.95  # Confidence for early stop
    cuped_enabled: bool = True  # Variance reduction

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if not 0 < self.traffic_split < 1:
            errors.append(
                ConfigError(
                    "ab_testing.traffic_split",
                    f"Must be between 0 and 1, got {self.traffic_split}",
                )
            )

        if self.min_samples < 2:
            errors.append(
                ConfigError(
                    "ab_testing.min_samples",
                    f"Must be at least 2, got {self.min_samples}",
                )
            )

        if self.max_samples < self.min_samples:
            errors.append(
                ConfigError(
                    "ab_testing.max_samples",
                    f"Must be >= min_samples ({self.min_samples}), got {self.max_samples}",
                )
            )

        if not 0 < self.significance_level < 1:
            errors.append(
                ConfigError(
                    "ab_testing.significance_level",
                    f"Must be between 0 and 1, got {self.significance_level}",
                )
            )

        return errors


# ─────────────────────────────────────────────
# HITL Configuration
# ─────────────────────────────────────────────


@dataclass
class HITLConfig:
    """Configuration for Human-in-the-Loop approval."""

    enabled: bool = True
    auto_approve_low_risk: bool = True
    auto_approve_confidence_threshold: float = 0.9
    auto_approve_max_impact: float = 0.05  # 5% max impact for auto-approve
    approval_timeout_hours: float = 72.0
    notification_channels: list[str] = field(default_factory=lambda: ["log"])

    # Email configuration (optional)
    email_enabled: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""  # Should use secrets manager in production
    email_from_address: str = ""
    email_recipients: list[str] = field(default_factory=list)

    # Webhook configuration (optional)
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if not 0 < self.auto_approve_confidence_threshold < 1:
            errors.append(
                ConfigError(
                    "hitl.auto_approve_confidence_threshold",
                    f"Must be between 0 and 1, got {self.auto_approve_confidence_threshold}",
                )
            )

        if self.approval_timeout_hours <= 0:
            errors.append(
                ConfigError(
                    "hitl.approval_timeout_hours",
                    f"Must be positive, got {self.approval_timeout_hours}",
                )
            )

        if self.email_enabled:
            if not self.email_smtp_host:
                errors.append(
                    ConfigError(
                        "hitl.email_smtp_host",
                        "Required when email_enabled is True",
                    )
                )
            if not self.email_from_address:
                errors.append(
                    ConfigError(
                        "hitl.email_from_address",
                        "Required when email_enabled is True",
                    )
                )

        if self.webhook_enabled and not self.webhook_url:
            errors.append(
                ConfigError(
                    "hitl.webhook_url",
                    "Required when webhook_enabled is True",
                )
            )

        return errors


# ─────────────────────────────────────────────
# Rollout Configuration
# ─────────────────────────────────────────────


@dataclass
class RolloutStage:
    """A stage in gradual rollout."""

    percentage: int
    min_successes: int
    max_failures: int
    timeout_hours: float = 0.0  # 0 = no timeout


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout."""

    enabled: bool = True
    auto_rollback: bool = True

    # Default stages
    stages: list[RolloutStage] = field(
        default_factory=lambda: [
            RolloutStage(percentage=5, min_successes=10, max_failures=3, timeout_hours=24),
            RolloutStage(percentage=25, min_successes=25, max_failures=5, timeout_hours=48),
            RolloutStage(percentage=50, min_successes=50, max_failures=10, timeout_hours=72),
            RolloutStage(percentage=100, min_successes=0, max_failures=0, timeout_hours=0),
        ]
    )

    # Canary analysis
    canary_enabled: bool = True
    canary_error_rate_threshold: float = 0.05  # 5% error rate triggers alert
    canary_latency_threshold: float = 2.0  # 2x latency triggers alert

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if not self.stages:
            errors.append(
                ConfigError(
                    "rollout.stages",
                    "Must have at least one stage",
                )
            )

        for i, stage in enumerate(self.stages):
            if not 0 <= stage.percentage <= 100:
                errors.append(
                    ConfigError(
                        f"rollout.stages[{i}].percentage",
                        f"Must be 0-100, got {stage.percentage}",
                    )
                )

            if stage.min_successes < 0:
                errors.append(
                    ConfigError(
                        f"rollout.stages[{i}].min_successes",
                        f"Must be non-negative, got {stage.min_successes}",
                    )
                )

            if stage.max_failures < 0:
                errors.append(
                    ConfigError(
                        f"rollout.stages[{i}].max_failures",
                        f"Must be non-negative, got {stage.max_failures}",
                    )
                )

        # Check stages are in ascending order by percentage
        for i in range(len(self.stages) - 1):
            if self.stages[i].percentage >= self.stages[i + 1].percentage:
                errors.append(
                    ConfigError(
                        f"rollout.stages[{i}].percentage",
                        f"Must be less than next stage ({self.stages[i+1].percentage}%)",
                    )
                )

        return errors


# ─────────────────────────────────────────────
# Transfer Learning Configuration
# ─────────────────────────────────────────────


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""

    enabled: bool = True
    min_similarity: float = 0.7  # Minimum similarity for transfer
    min_pattern_confidence: float = 0.8
    min_pattern_successes: int = 3
    max_patterns_per_project: int = 10

    # Embedding configuration
    embedding_cache_ttl: int = 7200  # 2 hours

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if not 0 < self.min_similarity < 1:
            errors.append(
                ConfigError(
                    "transfer.min_similarity",
                    f"Must be between 0 and 1, got {self.min_similarity}",
                )
            )

        if not 0 < self.min_pattern_confidence < 1:
            errors.append(
                ConfigError(
                    "transfer.min_pattern_confidence",
                    f"Must be between 0 and 1, got {self.min_pattern_confidence}",
                )
            )

        if self.min_pattern_successes < 1:
            errors.append(
                ConfigError(
                    "transfer.min_pattern_successes",
                    f"Must be at least 1, got {self.min_pattern_successes}",
                )
            )

        return errors


# ─────────────────────────────────────────────
# Performance Configuration
# ─────────────────────────────────────────────


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    # Batch processing
    batch_size: int = 100
    max_concurrency: int = 10
    batch_timeout: float = 30.0

    # Caching
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600
    max_memory_mb: float | None = None  # None = unlimited

    # Connection pooling
    pool_min_size: int = 1
    pool_max_size: int = 5
    connection_timeout: float = 10.0

    # Monitoring
    enable_stats: bool = True
    slow_threshold: float = 1.0  # Log operations slower than this

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if self.batch_size < 1:
            errors.append(
                ConfigError(
                    "performance.batch_size",
                    f"Must be at least 1, got {self.batch_size}",
                )
            )

        if self.max_concurrency < 1:
            errors.append(
                ConfigError(
                    "performance.max_concurrency",
                    f"Must be at least 1, got {self.max_concurrency}",
                )
            )

        if self.cache_max_size < 1:
            errors.append(
                ConfigError(
                    "performance.cache_max_size",
                    f"Must be at least 1, got {self.cache_max_size}",
                )
            )

        if self.pool_max_size < self.pool_min_size:
            errors.append(
                ConfigError(
                    "performance.pool_max_size",
                    f"Must be >= pool_min_size ({self.pool_min_size}), got {self.pool_max_size}",
                )
            )

        return errors


# ─────────────────────────────────────────────
# Monitoring Configuration
# ─────────────────────────────────────────────


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    severity: str = "warning"  # "info", "warning", "critical"
    cooldown_seconds: int = 300


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""

    enabled: bool = True
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Health checks
    health_check_interval: int = 60  # seconds

    # Alert rules
    alert_rules: list[AlertRule] = field(
        default_factory=lambda: [
            AlertRule(
                name="high_hitl_pending",
                metric_name="meta_optimization_hitl_pending",
                condition="gt",
                threshold=10,
                severity="warning",
                cooldown_seconds=600,
            ),
            AlertRule(
                name="critical_hitl_pending",
                metric_name="meta_optimization_hitl_pending",
                condition="gt",
                threshold=50,
                severity="critical",
                cooldown_seconds=300,
            ),
        ]
    )

    def validate(self) -> list[ConfigError]:
        """Validate configuration."""
        errors = []

        if self.prometheus_port < 1 or self.prometheus_port > 65535:
            errors.append(
                ConfigError(
                    "monitoring.prometheus_port",
                    f"Must be 1-65535, got {self.prometheus_port}",
                )
            )

        for i, rule in enumerate(self.alert_rules):
            if rule.condition not in ["gt", "lt", "eq", "gte", "lte"]:
                errors.append(
                    ConfigError(
                        f"monitoring.alert_rules[{i}].condition",
                        f"Must be gt/lt/eq/gte/lte, got {rule.condition}",
                    )
                )

            if rule.severity not in ["info", "warning", "critical"]:
                errors.append(
                    ConfigError(
                        f"monitoring.alert_rules[{i}].severity",
                        f"Must be info/warning/critical, got {rule.severity}",
                    )
                )

        return errors


# ─────────────────────────────────────────────
# Main Configuration
# ─────────────────────────────────────────────


@dataclass
class MetaOptimizationConfig:
    """
    Complete configuration for Meta-Optimization V2.

    All configuration options for A/B testing, HITL, rollout,
    transfer learning, performance, and monitoring.
    """

    # Component configurations
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    hitl: HITLConfig = field(default_factory=HITLConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # General settings
    storage_path: Path = field(
        default_factory=lambda: Path.home() / ".orchestrator_cache" / "meta_v2"
    )
    min_executions_for_optimization: int = 10
    max_proposals_per_cycle: int = 3

    # Feature flags
    enable_all: bool = False  # Enable all features

    def __post_init__(self):
        """Post-initialization processing."""
        if self.enable_all:
            self.ab_testing.enabled = True
            self.hitl.enabled = True
            self.rollout.enabled = True
            self.transfer.enabled = True
            self.monitoring.enabled = True

    def validate(self) -> list[ConfigError]:
        """
        Validate entire configuration.

        Returns:
            List of configuration errors (empty if valid)
        """
        errors = []

        # Validate each component
        errors.extend(self.ab_testing.validate())
        errors.extend(self.hitl.validate())
        errors.extend(self.rollout.validate())
        errors.extend(self.transfer.validate())
        errors.extend(self.performance.validate())
        errors.extend(self.monitoring.validate())

        # Validate storage path
        try:
            # Check if parent directory exists or can be created
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(
                ConfigError(
                    "storage_path",
                    f"Cannot create directory: {e}",
                )
            )

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ab_testing": asdict(self.ab_testing),
            "hitl": asdict(self.hitl),
            "rollout": {
                **asdict(self.rollout),
                "stages": [asdict(s) for s in self.rollout.stages],
            },
            "transfer": asdict(self.transfer),
            "performance": asdict(self.performance),
            "monitoring": {
                **asdict(self.monitoring),
                "alert_rules": [asdict(r) for r in self.monitoring.alert_rules],
            },
            "storage_path": str(self.storage_path),
            "min_executions_for_optimization": self.min_executions_for_optimization,
            "max_proposals_per_cycle": self.max_proposals_per_cycle,
            "enable_all": self.enable_all,
        }

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to YAML file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML support. Install with: pip install pyyaml")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaOptimizationConfig:
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            MetaOptimizationConfig instance
        """
        # Handle nested configs
        ab_testing = ABTestingConfig(**data.get("ab_testing", {}))
        hitl = HITLConfig(**data.get("hitl", {}))

        # Rollout stages need special handling
        rollout_data = data.get("rollout", {})
        stages = [RolloutStage(**s) for s in rollout_data.pop("stages", [])]
        rollout = RolloutConfig(stages=stages, **rollout_data)

        transfer = TransferConfig(**data.get("transfer", {}))
        performance = PerformanceConfig(**data.get("performance", {}))

        # Monitoring alert rules need special handling
        monitoring_data = data.get("monitoring", {})
        alert_rules = [AlertRule(**r) for r in monitoring_data.pop("alert_rules", [])]
        monitoring = MonitoringConfig(alert_rules=alert_rules, **monitoring_data)

        # Handle storage path
        storage_path_str = data.get("storage_path")
        storage_path = (
            Path(storage_path_str)
            if storage_path_str
            else Path.home() / ".orchestrator_cache" / "meta_v2"
        )

        return cls(
            ab_testing=ab_testing,
            hitl=hitl,
            rollout=rollout,
            transfer=transfer,
            performance=performance,
            monitoring=monitoring,
            storage_path=storage_path,
            min_executions_for_optimization=data.get("min_executions_for_optimization", 10),
            max_proposals_per_cycle=data.get("max_proposals_per_cycle", 3),
            enable_all=data.get("enable_all", False),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> MetaOptimizationConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            MetaOptimizationConfig instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML support. Install with: pip install pyyaml")

        if not path.exists():
            logger.warning(f"Configuration file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data or {})
        logger.info(f"Configuration loaded from {path}")

        return config

    @classmethod
    def from_env(cls) -> MetaOptimizationConfig:
        """
        Load configuration from environment variables.

        Environment variables:
        - META_OPTIMIZATION_ENABLED: Enable all features (true/false)
        - META_AB_TESTING_ENABLED: Enable A/B testing
        - META_HITL_ENABLED: Enable HITL
        - META_ROLLOUT_ENABLED: Enable rollout
        - META_TRANSFER_ENABLED: Enable transfer learning

        Returns:
            MetaOptimizationConfig instance
        """
        import os

        enable_all = os.environ.get("META_OPTIMIZATION_ENABLED", "").lower() == "true"

        config = cls(enable_all=enable_all)

        # Override individual settings
        ab_enabled = os.environ.get("META_AB_TESTING_ENABLED", "").lower() == "true"
        if ab_enabled:
            config.ab_testing.enabled = True

        hitl_enabled = os.environ.get("META_HITL_ENABLED", "").lower() == "true"
        if hitl_enabled:
            config.hitl.enabled = True

        rollout_enabled = os.environ.get("META_ROLLOUT_ENABLED", "").lower() == "true"
        if rollout_enabled:
            config.rollout.enabled = True

        transfer_enabled = os.environ.get("META_TRANSFER_ENABLED", "").lower() == "true"
        if transfer_enabled:
            config.transfer.enabled = True

        # Storage path
        storage_path = os.environ.get("META_STORAGE_PATH")
        if storage_path:
            config.storage_path = Path(storage_path)

        return config


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_config: MetaOptimizationConfig | None = None


def get_config(path: Path | None = None) -> MetaOptimizationConfig:
    """
    Get or create default configuration.

    Args:
        path: Optional path to YAML config file

    Returns:
        MetaOptimizationConfig instance
    """
    global _default_config

    if _default_config is None:
        if path:
            _default_config = MetaOptimizationConfig.from_yaml(path)
        else:
            _default_config = MetaOptimizationConfig.from_env()

    return _default_config


def reset_config() -> None:
    """Reset default configuration (for testing)."""
    global _default_config
    _default_config = None


def load_config(path: Path) -> MetaOptimizationConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        MetaOptimizationConfig instance
    """
    config = MetaOptimizationConfig.from_yaml(path)

    # Validate
    errors = config.validate()
    if errors:
        logger.warning("Configuration has validation errors:")
        for error in errors:
            logger.warning(f"  {error}")

    return config
