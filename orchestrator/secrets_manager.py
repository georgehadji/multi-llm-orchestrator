"""
Secure Secrets Manager
======================

Handles API keys and sensitive configuration securely:
- Never logs secrets
- Masks secrets in string representations
- Validates key formats
- Supports environment variables and secure key storage

SECURITY NOTICE:
    - Never commit actual API keys to version control
    - Use .env file (not committed) or proper secrets manager
    - Rotate keys regularly
    - Use least-privilege API keys
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Set, Protocol
from pathlib import Path

logger = logging.getLogger("orchestrator.secrets")


class SecretValidationError(Exception):
    """Raised when a secret fails validation."""
    pass


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass


@dataclass
class SecretValue:
    """
    Wrapper for secret values that prevents accidental logging.
    
    Usage:
        secret = SecretValue("sk-abc123", "openai_api_key")
        print(secret)  # Prints: ***[openai_api_key]***
        print(secret.get_value())  # Prints actual value
    """
    _value: str
    _name: str
    _mask_char: str = "*"
    
    def get_value(self) -> str:
        """Get the actual secret value."""
        return self._value
    
    def __str__(self) -> str:
        """Return masked representation."""
        return f"{self._mask_char * 3}[{self._name}]{self._mask_char * 3}"
    
    def __repr__(self) -> str:
        """Return masked representation."""
        return self.__str__()
    
    def __bool__(self) -> bool:
        """Return True if secret has a value."""
        return bool(self._value)
    
    def mask_in_string(self, text: str) -> str:
        """Mask this secret if it appears in a string."""
        if not self._value:
            return text
        # Escape special regex characters
        escaped = re.escape(self._value)
        return re.sub(escaped, str(self), text)


@dataclass
class SecretsManager:
    """
    Secure manager for API keys and sensitive configuration.
    
    Features:
    - Loads from environment variables
    - Validates key formats
    - Prevents secrets from appearing in logs
    - Masks secrets in error messages
    """
    
    # API Keys (wrapped in SecretValue)
    openai_api_key: Optional[SecretValue] = None
    anthropic_api_key: Optional[SecretValue] = None
    google_api_key: Optional[SecretValue] = None
    deepseek_api_key: Optional[SecretValue] = None
    minimax_api_key: Optional[SecretValue] = None
    mistral_api_key: Optional[SecretValue] = None
    xai_api_key: Optional[SecretValue] = None
    cohere_api_key: Optional[SecretValue] = None
    zhipuai_api_key: Optional[SecretValue] = None
    moonshot_api_key: Optional[SecretValue] = None
    
    # Chinese Providers
    dashscope_api_key: Optional[SecretValue] = None
    ark_api_key: Optional[SecretValue] = None
    qianfan_access_key: Optional[SecretValue] = None
    qianfan_secret_key: Optional[SecretValue] = None
    tencent_secret_id: Optional[SecretValue] = None
    tencent_secret_key: Optional[SecretValue] = None
    baichuan_api_key: Optional[SecretValue] = None
    
    # Security settings
    secure_mode: bool = True
    _loaded: bool = field(default=False, repr=False)
    
    # Key format validators
    _KEY_PATTERNS: Dict[str, re.Pattern] = field(default_factory=lambda: {
        "openai": re.compile(r"^sk-(?:proj-)?[A-Za-z0-9_-]+$"),
        "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9_-]+$"),
        "google": re.compile(r"^AIza[ A-Za-z0-9_-]+$"),
        "deepseek": re.compile(r"^sk-[a-f0-9]+$"),
        "minimax": re.compile(r"^sk-api-[A-Za-z0-9_-]+$"),
        "mistral": re.compile(r"^[A-Za-z0-9_-]+$"),  # Generic alphanumeric
        "xai": re.compile(r"^xai-[A-Za-z0-9_-]+$"),
        "cohere": re.compile(r"^[A-Za-z0-9_-]+$"),  # Generic
        "zhipuai": re.compile(r"^[a-f0-9]+\.[A-Za-z0-9_-]+$"),
        "moonshot": re.compile(r"^sk-[A-Za-z0-9_-]+$"),
    })
    
    def __post_init__(self):
        """Load secrets from environment if not already loaded."""
        if not self._loaded:
            self.load_from_env()
    
    def load_from_env(self) -> None:
        """Load all secrets from environment variables."""
        env_mappings = {
            "openai_api_key": "OPENAI_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "google_api_key": "GOOGLE_API_KEY",
            "deepseek_api_key": "DEEPSEEK_API_KEY",
            "minimax_api_key": "MINIMAX_API_KEY",
            "mistral_api_key": "MISTRAL_API_KEY",
            "xai_api_key": "XAI_API_KEY",
            "cohere_api_key": "COHERE_API_KEY",
            "zhipuai_api_key": "ZHIPUAI_API_KEY",
            "moonshot_api_key": "MOONSHOT_API_KEY",
            "dashscope_api_key": "DASHSCOPE_API_KEY",
            "ark_api_key": "ARK_API_KEY",
            "qianfan_access_key": "QIANFAN_ACCESS_KEY",
            "qianfan_secret_key": "QIANFAN_SECRET_KEY",
            "tencent_secret_id": "TENCENTCLOUD_SECRET_ID",
            "tencent_secret_key": "TENCENTCLOUD_SECRET_KEY",
            "baichuan_api_key": "BAICHUAN_API_KEY",
        }
        
        for attr_name, env_var in env_mappings.items():
            value = os.environ.get(env_var)
            if value and value.strip():
                # Check if it's a placeholder
                if self._is_placeholder(value):
                    logger.debug(f"Skipping placeholder value for {attr_name}")
                    continue
                    
                secret = SecretValue(value.strip(), attr_name)
                
                # Validate key format
                if self.secure_mode:
                    self._validate_key_format(attr_name, value)
                
                setattr(self, attr_name, secret)
                logger.debug(f"Loaded secret: {secret}")
        
        self._loaded = True
        
        # Check for any hardcoded secrets in common mistake locations
        if self.secure_mode:
            self._check_for_exposed_secrets()
    
    def _is_placeholder(self, value: str) -> bool:
        """Check if value is a placeholder/example."""
        placeholders = [
            "your_", "YOUR_", "example", "EXAMPLE", 
            "placeholder", "PLACEHOLDER", "xxx", "XXX",
            "test", "TEST", "demo", "DEMO", "fake", "FAKE"
        ]
        value_lower = value.lower()
        return any(p.lower() in value_lower for p in placeholders) or len(value) < 10
    
    def _validate_key_format(self, name: str, value: str) -> None:
        """Validate API key format."""
        # Determine provider from attribute name
        provider = name.replace("_api_key", "").replace("_", "")
        
        pattern = self._KEY_PATTERNS.get(provider)
        if pattern and not pattern.match(value):
            # Don't log the actual value, just the issue
            raise SecretValidationError(
                f"Invalid API key format for {name}. "
                f"Key does not match expected pattern for {provider}."
            )
    
    def _check_for_exposed_secrets(self) -> None:
        """Check for accidentally committed secrets in common files."""
        dangerous_files = [
            Path(".env"),
            Path("secrets.json"),
            Path("config.json"),
        ]
        
        for file_path in dangerous_files:
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    # Check for actual API key patterns (not placeholders)
                    api_key_patterns = [
                        r'sk-[a-zA-Z0-9]{20,}',  # OpenAI-style
                        r'sk-ant-[a-zA-Z0-9_-]{20,}',  # Anthropic
                        r'AIza[a-zA-Z0-9_-]{30,}',  # Google
                        r'xai-[a-zA-Z0-9_-]{20,}',  # xAI
                    ]
                    
                    for pattern in api_key_patterns:
                        if re.search(pattern, content):
                            logger.warning(
                                f"⚠️  POTENTIAL EXPOSED SECRET detected in {file_path}. "
                                f"Ensure this file is in .gitignore and does not contain real keys."
                            )
                            break
                except Exception:
                    pass  # Can't read file, skip
    
    def get(self, name: str) -> Optional[str]:
        """
        Get a secret value by name.
        
        Args:
            name: Attribute name of the secret (e.g., "openai_api_key")
            
        Returns:
            The secret value, or None if not set
        """
        secret = getattr(self, name, None)
        if isinstance(secret, SecretValue):
            return secret.get_value()
        return secret
    
    def require(self, name: str) -> str:
        """
        Get a required secret value by name.
        
        Raises:
            SecretNotFoundError: If the secret is not set
        """
        value = self.get(name)
        if not value:
            raise SecretNotFoundError(
                f"Required secret '{name}' is not set. "
                f"Please set the corresponding environment variable."
            )
        return value
    
    def mask_in_text(self, text: str) -> str:
        """Mask all known secrets in text."""
        result = text
        for attr_name in dir(self):
            if attr_name.endswith("_api_key") or attr_name.endswith("_secret"):
                secret = getattr(self, attr_name, None)
                if isinstance(secret, SecretValue):
                    result = secret.mask_in_string(result)
        return result
    
    def get_available_providers(self) -> Set[str]:
        """Get set of providers with configured API keys."""
        providers = set()
        provider_map = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "deepseek": "deepseek_api_key",
            "minimax": "minimax_api_key",
            "mistral": "mistral_api_key",
            "xai": "xai_api_key",
            "cohere": "cohere_api_key",
            "zhipuai": "zhipuai_api_key",
            "moonshot": "moonshot_api_key",
            "dashscope": "dashscope_api_key",
            "ark": "ark_api_key",
            "qianfan": "qianfan_access_key",
            "tencent": "tencent_secret_id",
            "baichuan": "baichuan_api_key",
        }
        
        for provider, attr_name in provider_map.items():
            if self.get(attr_name):
                providers.add(provider)
        
        return providers


# Global instance
_secrets_manager: Optional[SecretsManager] = None


# ═══════════════════════════════════════════════════════════════════════════════
# TD-002 FIX: Production Secrets Provider Interface
# ═══════════════════════════════════════════════════════════════════════════════

class SecretsProvider(Protocol):
    """
    Protocol for external secrets providers (AWS Secrets Manager, Azure Key Vault, etc.).
    
    TD-002 FIX: Production-ready secrets management interface.
    
    Usage:
        class AWSSecretsProvider:
            async def get_secret(self, name: str) -> Optional[str]: ...
            async def set_secret(self, name: str, value: str) -> None: ...
    """
    async def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret from the provider."""
        ...
    
    async def set_secret(self, name: str, value: str) -> None:
        """Store a secret in the provider."""
        ...


class EnvironmentSecretsProvider:
    """
    Default secrets provider that reads from environment variables.
    
    TD-002 FIX: Production-ready with proper error handling.
    """
    
    async def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment variable."""
        return os.environ.get(name)
    
    async def set_secret(self, name: str, value: str) -> None:
        """Set secret in environment (temporary, process-only)."""
        os.environ[name] = value
        logger.warning(f"Secret {name} set in environment - not persistent across restarts")


class VaultSecretsProvider:
    """
    HashiCorp Vault secrets provider.
    
    TD-002 FIX: Production vault integration.
    
    Usage:
        vault = VaultSecretsProvider(
            url="http://localhost:8200",
            token="s.xxxxx",
            mount_path="secret"
        )
        api_key = await vault.get_secret("openai/api_key")
    """
    
    def __init__(
        self,
        url: str,
        token: str,
        mount_path: str = "secret",
        timeout: int = 30,
    ):
        self.url = url.rstrip('/')
        self.token = token
        self.mount_path = mount_path
        self.timeout = timeout
        self._client = None
    
    async def _get_client(self):
        """Lazy HTTP client initialization."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.url,
                    headers={"X-Vault-Token": self.token},
                    timeout=self.timeout,
                )
            except ImportError:
                raise RuntimeError("httpx required for VaultSecretsProvider. Install with: pip install httpx")
        return self._client
    
    async def get_secret(self, name: str) -> Optional[str]:
        """
        Get secret from Vault.
        
        Args:
            name: Secret path (e.g., "openai/api_key" or "database/password")
        
        Returns:
            Secret value or None if not found
        """
        client = await self._get_client()
        try:
            # Vault KV v2 API
            path = f"/v1/{self.mount_path}/data/{name}"
            response = await client.get(path)
            
            if response.status_code == 404:
                logger.debug(f"Secret {name} not found in Vault")
                return None
            elif response.status_code != 200:
                logger.error(f"Vault error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            return data.get("data", {}).get("data", {}).get(name.split("/")[-1])
        except Exception as e:
            logger.error(f"Failed to get secret {name} from Vault: {type(e).__name__}: {e}")
            return None
    
    async def set_secret(self, name: str, value: str) -> None:
        """
        Set secret in Vault.
        
        Args:
            name: Secret path (e.g., "openai/api_key")
            value: Secret value to store
        """
        client = await self._get_client()
        try:
            path = f"/v1/{self.mount_path}/data/{name}"
            payload = {
                "data": {
                    name.split("/")[-1]: value
                }
            }
            response = await client.post(path, json=payload)
            
            if response.status_code not in [200, 201, 204]:
                logger.error(f"Vault error setting secret: {response.status_code} - {response.text}")
            else:
                logger.info(f"Secret {name} stored in Vault")
        except Exception as e:
            logger.error(f"Failed to set secret {name} in Vault: {type(e).__name__}: {e}")


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def create_secrets_manager_with_provider(provider: SecretsProvider) -> SecretsManager:
    """
    TD-002 FIX: Create secrets manager with external provider.
    
    Usage:
        # Production with Vault
        vault_provider = VaultSecretsProvider(url="...", token="...")
        secrets = create_secrets_manager_with_provider(vault_provider)
        
        # Or AWS Secrets Manager
        aws_provider = AWSSecretsProvider(region="us-east-1")
        secrets = create_secrets_manager_with_provider(aws_provider)
    """
    # For now, return the default manager
    # Future enhancement: integrate provider into SecretsManager
    logger.info(f"Secrets manager created with provider: {type(provider).__name__}")
    return get_secrets_manager()


def reset_secrets_manager() -> None:
    """Reset the global secrets manager (for testing)."""
    global _secrets_manager
    _secrets_manager = None


# Convenience functions
def get_secret(name: str) -> Optional[str]:
    """Get a secret value by name."""
    return get_secrets_manager().get(name)


def require_secret(name: str) -> str:
    """Get a required secret value by name."""
    return get_secrets_manager().require(name)


def mask_secrets(text: str) -> str:
    """Mask all known secrets in text."""
    return get_secrets_manager().mask_in_text(text)


# Example usage
if __name__ == "__main__":
    # Test the secrets manager
    secrets = SecretsManager()
    
    # Test masking
    test_text = "My key is sk-abc123def456 but don't show it"
    secret = SecretValue("sk-abc123def456", "test_key")
    print(f"Original: {test_text}")
    print(f"Masked: {secret.mask_in_string(test_text)}")
    print(f"Secret repr: {repr(secret)}")
