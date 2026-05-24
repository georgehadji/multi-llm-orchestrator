"""
Secrets Generator — Factory Method + Builder Pattern
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Secure secret generation using Factory Method for different secret types
and Builder Pattern for constructing .env files.

Paradigm: OOP with Functional utilities
Patterns: Factory Method, Builder, Protocol

Usage:
    from orchestrator.secrets_generator import SecretsGenerator, EnvFileBuilder

    # Generate secrets
    jwt_secret = SecretsGenerator.create_jwt_secret()
    api_key = SecretsGenerator.create_api_key()

    # Build .env file
    env_content = (EnvFileBuilder()
        .add_comment("JWT Configuration")
        .add_secret("JWT_SECRET", jwt_secret)
        .add_secret("JWT_EXPIRY", "900")
        .add_blank()
        .build())
"""

from __future__ import annotations

import secrets
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# FACTORY METHOD — SECRET GENERATOR INTERFACE
# ═══════════════════════════════════════════════════════════════════


class SecretGenerator(ABC):
    """
    Factory Method for different secret types.

    Subclasses implement specific secret generation logic.
    """

    @abstractmethod
    def generate(self, length: int = 32) -> str:
        """
        Generate a secret.

        Args:
            length: Secret length in characters

        Returns:
            Generated secret string
        """
        pass

    @abstractmethod
    def validate(self, secret: str) -> bool:
        """
        Validate secret strength (pure function).

        Args:
            secret: Secret to validate

        Returns:
            True if secret is strong enough
        """
        pass


# ═══════════════════════════════════════════════════════════════════
# CONCRETE FACTORIES
# ═══════════════════════════════════════════════════════════════════


class JwtSecretFactory(SecretGenerator):
    """
    Factory for JWT secrets.

    Generates cryptographically secure secrets for JWT signing.
    """

    def generate(self, length: int = 64) -> str:
        """
        Generate JWT secret.

        Args:
            length: Secret length (default: 64 for strong security)

        Returns:
            Hex-encoded secret
        """
        return secrets.token_hex(length // 2)

    def validate(self, secret: str) -> bool:
        """
        Validate JWT secret strength.

        Requirements:
        - Minimum 32 characters
        - Alphanumeric or hex characters

        Args:
            secret: Secret to validate

        Returns:
            True if valid
        """
        if len(secret) < 32:
            return False

        # Check if hex-encoded (preferred) or alphanumeric
        try:
            bytes.fromhex(secret)
            return True
        except ValueError:
            return secret.isalnum()


class ApiKeyFactory(SecretGenerator):
    """
    Factory for API keys.

    Generates URL-safe API keys with prefix for identification.
    """

    def __init__(self, prefix: str = "sk"):
        """
        Initialize API key factory.

        Args:
            prefix: Key prefix for identification (e.g., "sk" for secret key)
        """
        self._prefix = prefix

    def generate(self, length: int = 32) -> str:
        """
        Generate API key.

        Args:
            length: Random part length

        Returns:
            API key with prefix (format: prefix_random)
        """
        # URL-safe alphabet
        alphabet = string.ascii_letters + string.digits
        random_part = "".join(secrets.choice(alphabet) for _ in range(length))
        return f"{self._prefix}_{random_part}"

    def validate(self, secret: str) -> bool:
        """
        Validate API key format.

        Requirements:
        - Has prefix
        - Minimum total length
        - URL-safe characters

        Args:
            secret: API key to validate

        Returns:
            True if valid
        """
        if not secret.startswith(f"{self._prefix}_"):
            return False

        if len(secret) < len(self._prefix) + 20:
            return False

        # Check URL-safe characters
        allowed = set(string.ascii_letters + string.digits + "_")
        return all(c in allowed for c in secret)


class DatabasePasswordFactory(SecretGenerator):
    """
    Factory for database passwords.

    Generates strong passwords meeting common requirements.
    """

    def generate(self, length: int = 32) -> str:
        """
        Generate database password.

        Requirements:
        - Uppercase letters
        - Lowercase letters
        - Digits
        - Special characters

        Args:
            length: Password length

        Returns:
            Strong password
        """
        # Ensure at least one of each required character type
        uppercase = secrets.choice(string.ascii_uppercase)
        lowercase = secrets.choice(string.ascii_lowercase)
        digit = secrets.choice(string.digits)
        special = secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?")

        # Fill rest with random characters from all types
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
        remaining_length = length - 4
        remaining = [secrets.choice(all_chars) for _ in range(remaining_length)]

        # Combine and shuffle
        password_chars = [uppercase, lowercase, digit, special] + remaining
        secrets.SystemRandom().shuffle(password_chars)

        return "".join(password_chars)

    def validate(self, secret: str) -> bool:
        """
        Validate password strength.

        Requirements:
        - Minimum 12 characters
        - Has uppercase
        - Has lowercase
        - Has digit
        - Has special character

        Args:
            secret: Password to validate

        Returns:
            True if strong enough
        """
        if len(secret) < 12:
            return False

        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in secret)

        return all([has_upper, has_lower, has_digit, has_special])


class GenericSecretFactory(SecretGenerator):
    """
    Factory for generic secrets.

    Generates random hex secrets for general use.
    """

    def generate(self, length: int = 32) -> str:
        """
        Generate generic secret.

        Args:
            length: Secret length

        Returns:
            Hex-encoded secret
        """
        return secrets.token_hex(length // 2)

    def validate(self, secret: str) -> bool:
        """
        Validate generic secret.

        Args:
            secret: Secret to validate

        Returns:
            True if valid hex string
        """
        if len(secret) < 16:
            return False

        try:
            bytes.fromhex(secret)
            return True
        except ValueError:
            return False


# ═══════════════════════════════════════════════════════════════════
# SECRETS GENERATOR — FACADE
# ═══════════════════════════════════════════════════════════════════


class SecretsGenerator:
    """
    Facade for secret generation.

    Provides simple interface for generating different secret types.

    Usage:
        jwt_secret = SecretsGenerator.create_jwt_secret()
        api_key = SecretsGenerator.create_api_key()
        password = SecretsGenerator.create_database_password()
    """

    _factories = {
        "jwt": JwtSecretFactory(),
        "api_key": ApiKeyFactory(),
        "database_password": DatabasePasswordFactory(),
        "generic": GenericSecretFactory(),
    }

    @classmethod
    def create_jwt_secret(cls, length: int = 64) -> str:
        """
        Create JWT secret.

        Args:
            length: Secret length

        Returns:
            JWT secret
        """
        return cls._factories["jwt"].generate(length)

    @classmethod
    def create_api_key(cls, prefix: str = "sk", length: int = 32) -> str:
        """
        Create API key.

        Args:
            prefix: Key prefix
            length: Random part length

        Returns:
            API key
        """
        factory = ApiKeyFactory(prefix)
        return factory.generate(length)

    @classmethod
    def create_database_password(cls, length: int = 32) -> str:
        """
        Create database password.

        Args:
            length: Password length

        Returns:
            Strong password
        """
        return cls._factories["database_password"].generate(length)

    @classmethod
    def create_generic_secret(cls, length: int = 32) -> str:
        """
        Create generic secret.

        Args:
            length: Secret length

        Returns:
            Generic secret
        """
        return cls._factories["generic"].generate(length)

    @classmethod
    def validate_secret(cls, secret_type: str, secret: str) -> bool:
        """
        Validate secret strength.

        Args:
            secret_type: Type of secret
            secret: Secret to validate

        Returns:
            True if valid
        """
        if secret_type not in cls._factories:
            raise ValueError(f"Unknown secret type: {secret_type}")

        return cls._factories[secret_type].validate(secret)


# ═══════════════════════════════════════════════════════════════════
# BUILDER PATTERN — ENV FILE BUILDER
# ═══════════════════════════════════════════════════════════════════


@dataclass
class EnvLine:
    """Immutable environment file line."""

    type: str  # "comment", "blank", "secret", "header"
    content: str
    order: int = 0


class EnvFileBuilder:
    """
    Builder Pattern for .env files.

    Fluent interface for constructing complex .env files.

    Usage:
        env_content = (EnvFileBuilder()
            .add_header("Application Configuration")
            .add_comment("JWT Settings")
            .add_secret("JWT_SECRET", SecretsGenerator.create_jwt_secret())
            .add_secret("JWT_EXPIRY", "900")
            .add_blank()
            .add_comment("Database")
            .add_secret("DB_PASSWORD", SecretsGenerator.create_database_password())
            .build())
    """

    def __init__(self):
        """Initialize builder."""
        self._lines: List[EnvLine] = []
        self._order = 0

    def _add_line(self, line_type: str, content: str) -> "EnvFileBuilder":
        """Add line to builder (internal)."""
        self._order += 1
        self._lines.append(EnvLine(type=line_type, content=content, order=self._order))
        return self

    def add_header(self, text: str) -> "EnvFileBuilder":
        """
        Add header comment.

        Args:
            text: Header text

        Returns:
            Self for fluent interface
        """
        self._add_line("header", f"# ═══════════════════════════════════════════════════════")
        self._add_line("header", f"# {text}")
        self._add_line("header", f"# ═══════════════════════════════════════════════════════")
        return self

    def add_comment(self, text: str) -> "EnvFileBuilder":
        """
        Add comment line.

        Args:
            text: Comment text

        Returns:
            Self for fluent interface
        """
        return self._add_line("comment", f"# {text}")

    def add_secret(self, key: str, value: str, comment: str = None) -> "EnvFileBuilder":
        """
        Add secret.

        Args:
            key: Environment variable name
            value: Secret value
            comment: Optional inline comment

        Returns:
            Self for fluent interface
        """
        line = f"{key}={value}"
        if comment:
            line += f"  # {comment}"
        return self._add_line("secret", line)

    def add_blank(self) -> "EnvFileBuilder":
        """
        Add blank line.

        Returns:
            Self for fluent interface
        """
        return self._add_line("blank", "")

    def add_generated_at(self) -> "EnvFileBuilder":
        """
        Add generation timestamp.

        Returns:
            Self for fluent interface
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        return self.add_comment(f"Generated: {timestamp}")

    def add_warning(self, warning: str) -> "EnvFileBuilder":
        """
        Add security warning.

        Args:
            warning: Warning text

        Returns:
            Self for fluent interface
        """
        self._add_line("comment", "# ⚠️  WARNING: Keep this file secure!")
        self._add_line("comment", f"# {warning}")
        self._add_line("comment", "# Never commit to version control!")
        return self

    def build(self) -> str:
        """
        Build final .env content.

        Returns:
            Complete .env file content
        """
        # Sort lines by order
        sorted_lines = sorted(self._lines, key=lambda l: l.order)

        # Build content
        content_lines = []
        for line in sorted_lines:
            content_lines.append(line.content)

        return "\n".join(content_lines)

    def build_to_file(self, filepath: str, secure_permissions: bool = True) -> str:
        """
        Build and write to file.

        Args:
            filepath: Output file path
            secure_permissions: Set restrictive file permissions

        Returns:
            File path
        """
        content = self.build()

        with open(filepath, "w") as f:
            f.write(content)

        # Set secure permissions (Unix only)
        if secure_permissions:
            import os

            try:
                os.chmod(filepath, 0o600)  # Owner read/write only
            except (OSError, AttributeError):
                pass  # Windows doesn't support Unix permissions

        return filepath


# ═══════════════════════════════════════════════════════════════════
# PRESET BUILDERS
# ═══════════════════════════════════════════════════════════════════


class JwtEnvBuilder(EnvFileBuilder):
    """Builder for JWT configuration .env file."""

    def __init__(self):
        super().__init__()
        self.add_header("JWT Configuration")
        self.add_warning("Do not share this file")
        self.add_blank()

    def with_defaults(self) -> "JwtEnvBuilder":
        """Add default JWT configuration."""
        self.add_comment("JWT Settings")
        self.add_secret("JWT_SECRET", SecretsGenerator.create_jwt_secret())
        self.add_secret("JWT_EXPIRY", "900", "15 minutes")
        self.add_secret("JWT_REFRESH_EXPIRY", "604800", "7 days")
        self.add_secret("JWT_ALGORITHM", "HS256")
        self.add_blank()
        return self


class DatabaseEnvBuilder(EnvFileBuilder):
    """Builder for database configuration .env file."""

    def __init__(self):
        super().__init__()
        self.add_header("Database Configuration")
        self.add_warning("Contains sensitive credentials")
        self.add_blank()

    def with_postgres_defaults(self) -> "DatabaseEnvBuilder":
        """Add PostgreSQL default configuration."""
        self.add_comment("PostgreSQL Configuration")
        self.add_secret("DB_HOST", "localhost")
        self.add_secret("DB_PORT", "5432")
        self.add_secret("DB_NAME", "myapp")
        self.add_secret("DB_USER", "postgres")
        self.add_secret("DB_PASSWORD", SecretsGenerator.create_database_password())
        self.add_blank()
        return self

    def with_mysql_defaults(self) -> "DatabaseEnvBuilder":
        """Add MySQL default configuration."""
        self.add_comment("MySQL Configuration")
        self.add_secret("DB_HOST", "localhost")
        self.add_secret("DB_PORT", "3306")
        self.add_secret("DB_NAME", "myapp")
        self.add_secret("DB_USER", "root")
        self.add_secret("DB_PASSWORD", SecretsGenerator.create_database_password())
        self.add_secret("DB_CHARSET", "utf8mb4")
        self.add_blank()
        return self


class RedisEnvBuilder(EnvFileBuilder):
    """Builder for Redis configuration .env file."""

    def __init__(self):
        super().__init__()
        self.add_header("Redis Configuration")
        self.add_blank()

    def with_defaults(self) -> "RedisEnvBuilder":
        """Add default Redis configuration."""
        self.add_comment("Redis Configuration")
        self.add_secret("REDIS_HOST", "localhost")
        self.add_secret("REDIS_PORT", "6379")
        self.add_secret("REDIS_PASSWORD", SecretsGenerator.create_generic_secret(32))
        self.add_secret("REDIS_DB", "0")
        self.add_blank()
        return self


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def generate_jwt_env(filepath: str = None) -> str:
    """
    Generate JWT configuration .env file.

    Args:
        filepath: Output file path (optional)

    Returns:
        File path or content
    """
    builder = JwtEnvBuilder().with_defaults()
    builder.add_generated_at()

    if filepath:
        return builder.build_to_file(filepath)
    return builder.build()


def generate_database_env(filepath: str = None, db_type: str = "postgres") -> str:
    """
    Generate database configuration .env file.

    Args:
        filepath: Output file path (optional)
        db_type: Database type ("postgres" or "mysql")

    Returns:
        File path or content
    """
    builder = DatabaseEnvBuilder()

    if db_type == "mysql":
        builder.with_mysql_defaults()
    else:
        builder.with_postgres_defaults()

    builder.add_generated_at()

    if filepath:
        return builder.build_to_file(filepath)
    return builder.build()


def generate_redis_env(filepath: str = None) -> str:
    """
    Generate Redis configuration .env file.

    Args:
        filepath: Output file path (optional)

    Returns:
        File path or content
    """
    builder = RedisEnvBuilder().with_defaults()
    builder.add_generated_at()

    if filepath:
        return builder.build_to_file(filepath)
    return builder.build()


def generate_complete_env(filepath: str = None) -> str:
    """
    Generate complete .env file with all configurations.

    Args:
        filepath: Output file path (optional)

    Returns:
        File path or content
    """
    builder = (
        EnvFileBuilder()
        .add_header("Application Configuration")
        .add_warning("Contains sensitive credentials - keep secure!")
        .add_blank()
    )

    # JWT
    builder.add_comment("JWT Authentication")
    builder.add_secret("JWT_SECRET", SecretsGenerator.create_jwt_secret())
    builder.add_secret("JWT_EXPIRY", "900")
    builder.add_secret("JWT_REFRESH_EXPIRY", "604800")
    builder.add_secret("JWT_ALGORITHM", "HS256")
    builder.add_blank()

    # Database
    builder.add_comment("Database Configuration")
    builder.add_secret("DB_HOST", "localhost")
    builder.add_secret("DB_PORT", "5432")
    builder.add_secret("DB_NAME", "myapp")
    builder.add_secret("DB_USER", "postgres")
    builder.add_secret("DB_PASSWORD", SecretsGenerator.create_database_password())
    builder.add_blank()

    # Redis
    builder.add_comment("Redis Configuration")
    builder.add_secret("REDIS_HOST", "localhost")
    builder.add_secret("REDIS_PORT", "6379")
    builder.add_secret("REDIS_PASSWORD", SecretsGenerator.create_generic_secret(32))
    builder.add_blank()

    # API Keys
    builder.add_comment("API Keys")
    builder.add_secret("API_KEY", SecretsGenerator.create_api_key("sk"))
    builder.add_secret("API_SECRET", SecretsGenerator.create_generic_secret(64))
    builder.add_blank()

    builder.add_generated_at()

    if filepath:
        return builder.build_to_file(filepath)
    return builder.build()
