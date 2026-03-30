"""
Secrets Manager — Secure secrets management
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Secure secrets management with:
- Environment variable loading with validation
- Secrets masking in logs
- No hardcoded secrets
- Secure defaults

USAGE:
    from orchestrator.secrets_manager import SecretsManager, get_secrets

    secrets = SecretsManager()
    api_key = secrets.get("OPENAI_API_KEY")

    # Or use convenience function
    api_key = get_secrets().get("OPENAI_API_KEY")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger("orchestrator.secrets")


# ─────────────────────────────────────────────
# Secret Patterns for Masking
# ─────────────────────────────────────────────

SECRET_PATTERNS = [
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), 'API_KEY'),  # OpenAI/Anthropic style
    (re.compile(r'AIza[a-zA-Z0-9_-]{35}'), 'GOOGLE_KEY'),  # Google style
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), 'GITHUB_TOKEN'),  # GitHub PAT
    (re.compile(r'xox[baprs]-[a-zA-Z0-9-]+'), 'SLACK_TOKEN'),  # Slack
    (re.compile(r'Bearer\s+[a-zA-Z0-9_-]{20,}'), 'BEARER_TOKEN'),
]

# Known secret environment variable names
SECRET_ENV_NAMES = {
    'API_KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PASSWD', 'CREDENTIAL',
    'PRIVATE_KEY', 'AUTH', 'BEARER', 'JWT', 'APIKEY', 'API_SECRET'
}


@dataclass
class SecretsConfig:
    """Secrets manager configuration."""
    required_secrets: list[str] = field(default_factory=list)
    optional_secrets: list[str] = field(default_factory=list)
    mask_in_logs: bool = True
    validate_on_load: bool = True


class SecretsManager:
    """
    Secure secrets manager.

    Features:
    - Load secrets from environment variables
    - Mask secrets in logs
    - Validate required secrets are present
    - No hardcoded secrets
    """

    _instance: SecretsManager | None = None

    def __new__(cls, config: SecretsConfig | None = None) -> SecretsManager:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: SecretsConfig | None = None):
        """Initialize secrets manager."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or SecretsConfig()
        self._secrets: dict[str, str] = {}
        self._loaded = False
        self._validated = False
        self._initialized = True

        logger.info("SecretsManager initialized")

    def load_from_env(self, prefix: str = "") -> dict[str, str]:
        """
        Load secrets from environment variables.

        Args:
            prefix: Optional prefix for env vars (e.g., "APP_")

        Returns:
            Dictionary of loaded secrets
        """
        self._secrets.clear()

        # Load required secrets
        for secret_name in self.config.required_secrets:
            env_name = f"{prefix}{secret_name}"
            value = os.environ.get(env_name)

            if value:
                self._secrets[secret_name] = value
                logger.debug(f"Loaded secret: {secret_name}")
            elif self.config.validate_on_load:
                logger.warning(f"Required secret not found: {env_name}")

        # Load optional secrets
        for secret_name in self.config.optional_secrets:
            env_name = f"{prefix}{secret_name}"
            value = os.environ.get(env_name)

            if value:
                self._secrets[secret_name] = value

        self._loaded = True
        logger.info(f"Loaded {len(self._secrets)} secrets from environment")

        return self._secrets.copy()

    def get(self, name: str, default: str | None = None) -> str | None:
        """
        Get a secret by name.

        Args:
            name: Secret name
            default: Default value if not found

        Returns:
            Secret value or default
        """
        # First check loaded secrets
        if name in self._secrets:
            return self._secrets[name]

        # Then check environment
        value = os.environ.get(name)
        if value:
            return value

        return default

    def get_required(self, name: str) -> str:
        """
        Get a required secret.

        Args:
            name: Secret name

        Returns:
            Secret value

        Raises:
            ValueError: If secret not found
        """
        value = self.get(name)
        if not value:
            raise ValueError(f"Required secret not found: {name}")
        return value

    def has(self, name: str) -> bool:
        """Check if secret exists."""
        return name in self._secrets or os.environ.get(name) is not None

    def validate(self) -> list[str]:
        """
        Validate all required secrets are present.

        Returns:
            List of missing secret names
        """
        missing = []

        for secret_name in self.config.required_secrets:
            if not self.has(secret_name):
                missing.append(secret_name)

        self._validated = len(missing) == 0

        if missing:
            logger.error(f"Missing required secrets: {missing}")
        else:
            logger.info("All required secrets validated")

        return missing

    def mask_value(self, value: str) -> str:
        """
        Mask a secret value for logging.

        Args:
            value: Value to mask

        Returns:
            Masked value
        """
        if not self.config.mask_in_logs:
            return value

        if not value:
            return value

        # Show first 4 and last 4 characters
        if len(value) > 8:
            return f"{value[:4]}...{value[-4:]}"
        elif len(value) > 4:
            return f"{value[:2]}..."
        else:
            return "***"

    def mask_string(self, text: str) -> str:
        """
        Mask all secrets in a string.

        Args:
            text: Text to mask

        Returns:
            Text with secrets masked
        """
        if not self.config.mask_in_logs:
            return text

        masked = text

        # Mask known secret patterns
        for pattern, replacement in SECRET_PATTERNS:
            masked = pattern.sub(f'[REDACTED_{replacement}]', masked)

        # Mask loaded secrets
        for name, value in self._secrets.items():
            if value and len(value) > 4:
                masked = masked.replace(value, f'[REDACTED_{name}]')

        return masked

    def to_dict(self, mask: bool = True) -> dict[str, str]:
        """
        Get secrets as dictionary.

        Args:
            mask: Mask secret values

        Returns:
            Dictionary of secrets
        """
        if mask:
            return {name: self.mask_value(value) for name, value in self._secrets.items()}
        else:
            return self._secrets.copy()

    def clear(self) -> None:
        """Clear all loaded secrets."""
        self._secrets.clear()
        self._loaded = False
        self._validated = False
        logger.info("Secrets cleared")


# ─────────────────────────────────────────────
# Secure Logging Filter
# ─────────────────────────────────────────────

class SecretsFilter(logging.Filter):
    """
    Logging filter that masks secrets in log messages.

    Usage:
        logger.addFilter(SecretsFilter())
    """

    def __init__(self, secrets_manager: SecretsManager | None = None):
        super().__init__()
        self.secrets = secrets_manager or get_secrets()

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask secrets in log message."""
        if isinstance(record.msg, str):
            record.msg = self.secrets.mask_string(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self.secrets.mask_string(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self.secrets.mask_string(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        return True


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_secrets: SecretsManager | None = None


def get_secrets(config: SecretsConfig | None = None) -> SecretsManager:
    """Get or create default secrets manager."""
    global _default_secrets
    if _default_secrets is None:
        _default_secrets = SecretsManager(config)
    return _default_secrets


def reset_secrets() -> None:
    """Reset secrets manager (for testing)."""
    global _default_secrets
    if _default_secrets:
        _default_secrets.clear()
    _default_secrets = None


def mask_secrets(text: str) -> str:
    """Mask secrets in text."""
    return get_secrets().mask_string(text)


def setup_secure_logging(logger_obj: logging.Logger) -> None:
    """
    Setup secure logging with secret masking.

    Args:
        logger_obj: Logger to configure
    """
    secrets = get_secrets()
    logger_obj.addFilter(SecretsFilter(secrets))
    logger_obj.info("Secure logging enabled with secret masking")


# ─────────────────────────────────────────────
# Pre-configured Secret Sets
# ─────────────────────────────────────────────

LLM_SECRETS_CONFIG = SecretsConfig(
    required_secrets=[
        'DEEPSEEK_API_KEY',  # At least one LLM provider required
    ],
    optional_secrets=[
        'OPENAI_API_KEY',
        'GOOGLE_API_KEY',
        'ANTHROPIC_API_KEY',
        'MINIMAX_API_KEY',
    ],
    mask_in_logs=True,
    validate_on_load=False,  # Allow running without all providers
)

DATABASE_SECRETS_CONFIG = SecretsConfig(
    required_secrets=[],
    optional_secrets=[
        'DATABASE_URL',
        'DB_PASSWORD',
        'DB_USER',
    ],
    mask_in_logs=True,
    validate_on_load=False,
)
