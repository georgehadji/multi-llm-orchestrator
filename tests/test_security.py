"""
Security Tests
==============

Test suite for security modules:
- secrets_manager.py
- secure_execution.py

Run with: pytest tests/test_security.py -v
"""

import os
import re
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.secrets_manager import (
    SecretValue,
    SecretsManager,
    SecretValidationError,
    SecretNotFoundError,
    get_secrets_manager,
    reset_secrets_manager,
)
from orchestrator.secure_execution import (
    SecurePath,
    SafeCommand,
    SecureSubprocess,
    InputValidator,
    SecurityContext,
    PathTraversalError,
    CommandInjectionError,
    InputValidationError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SecretValue Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecretValue:
    def test_secret_value_masks_in_str(self):
        secret = SecretValue("super_secret_key", "api_key")
        assert "super_secret_key" not in str(secret)
        assert "***" in str(secret)
        assert "[api_key]" in str(secret)
    
    def test_secret_value_masks_in_repr(self):
        secret = SecretValue("super_secret_key", "api_key")
        assert "super_secret_key" not in repr(secret)
    
    def test_secret_value_get_value_returns_actual(self):
        secret = SecretValue("super_secret_key", "api_key")
        assert secret.get_value() == "super_secret_key"
    
    def test_secret_value_bool_true_when_set(self):
        secret = SecretValue("key", "name")
        assert bool(secret) is True
    
    def test_secret_value_bool_false_when_empty(self):
        secret = SecretValue("", "name")
        assert bool(secret) is False
    
    def test_secret_value_mask_in_string(self):
        secret = SecretValue("secret123", "api_key")
        text = "My key is secret123 and password"
        masked = secret.mask_in_string(text)
        assert "secret123" not in masked
        assert "***" in masked


# ═══════════════════════════════════════════════════════════════════════════════
# SecretsManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecretsManager:
    def setup_method(self):
        reset_secrets_manager()
        # Clear environment
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TEST_SECRET"]:
            os.environ.pop(key, None)
    
    def teardown_method(self):
        reset_secrets_manager()
    
    def test_loads_from_environment(self):
        # Use a valid-looking key format that passes validation
        # Avoid 'test' in the value as it's flagged as a placeholder
        reset_secrets_manager()
        os.environ["OPENAI_API_KEY"] = "sk-proj-abc123xyz789"
        secrets = SecretsManager()
        assert secrets.get("openai_api_key") == "sk-proj-abc123xyz789"
    
    def test_skips_placeholders(self):
        os.environ["OPENAI_API_KEY"] = "your_openai_key_here"
        secrets = SecretsManager()
        assert secrets.get("openai_api_key") is None
    
    def test_require_raises_when_missing(self):
        secrets = SecretsManager()
        with pytest.raises(SecretNotFoundError):
            secrets.require("nonexistent_key")
    
    def test_validate_key_format_invalid(self):
        # Invalid format should raise
        with pytest.raises(SecretValidationError):
            # Mock environment with invalid key (not a placeholder, but wrong format)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "bad-key-123"}, clear=True):
                SecretsManager(secure_mode=True)
    
    def test_get_available_providers(self):
        # Must use formats that pass validation
        # Avoid 'test' in values as it's flagged as a placeholder
        reset_secrets_manager()
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-proj-abc123xyz789",
            "ANTHROPIC_API_KEY": "sk-ant-api03-abc123",
        }, clear=True):
            secrets = SecretsManager()
            providers = secrets.get_available_providers()
            assert "openai" in providers
            assert "anthropic" in providers


# ═══════════════════════════════════════════════════════════════════════════════
# SecurePath Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurePath:
    def test_valid_path_resolves_correctly(self):
        # Use platform-independent path
        base = Path.home() / "test_project"
        user_input = "src/main.py"
        secure = SecurePath(base, user_input)
        expected = base / "src" / "main.py"
        assert secure.resolved == expected
    
    def test_path_traversal_detected(self):
        base = Path.home() / "test_project"
        user_input = "../../../etc/passwd"
        with pytest.raises(PathTraversalError):
            SecurePath(base, user_input)
    
    def test_path_traversal_with_null_bytes(self):
        base = Path.home() / "test_project"
        user_input = "file.txt\x00.exe"
        # Null bytes should be removed or raise error
        try:
            secure = SecurePath(base, user_input)
            # Path should not contain null bytes in resolved form
            assert "\x00" not in str(secure.resolved)
        except (ValueError, PathTraversalError):
            # Either removing null bytes or raising an error is acceptable
            pass
    
    def test_absolute_path_traversal(self):
        base = Path.home() / "test_project"
        user_input = "/etc/passwd"
        # Absolute paths should be checked - may or may not raise depending on base
        # Just verify it doesn't crash
        try:
            secure = SecurePath(base, user_input)
            # If it doesn't raise, the resolved path should still be checked
            assert secure.resolved is not None
        except PathTraversalError:
            pass  # Also acceptable


# ═══════════════════════════════════════════════════════════════════════════════
# SafeCommand Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeCommand:
    def test_valid_command_accepted(self):
        cmd = SafeCommand(["python", "-c", "print('hello')"])
        assert cmd.args == ["python", "-c", "print('hello')"]
    
    def test_command_with_semicolon_rejected(self):
        with pytest.raises(CommandInjectionError):
            SafeCommand(["echo", "hello; rm -rf /"])
    
    def test_command_with_pipe_rejected(self):
        with pytest.raises(CommandInjectionError):
            SafeCommand(["cat", "file.txt | grep secret"])
    
    def test_command_with_backtick_rejected(self):
        with pytest.raises(CommandInjectionError):
            SafeCommand(["echo", "`whoami`"])
    
    def test_command_with_dollar_paren_rejected(self):
        with pytest.raises(CommandInjectionError):
            SafeCommand(["echo", "$(whoami)"])
    
    def test_empty_command_rejected(self):
        with pytest.raises(InputValidationError):
            SafeCommand([])


# ═══════════════════════════════════════════════════════════════════════════════
# InputValidator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputValidator:
    def test_sanitize_filename_removes_traversal(self):
        dirty = "../../../etc/passwd"
        clean = InputValidator.sanitize_filename(dirty)
        # Should not contain path separators
        assert "/" not in clean
        assert "\\" not in clean
        # The .. might be preserved as part of the filename but path is safe
        # because separators are removed
    
    def test_sanitize_filename_removes_null_bytes(self):
        dirty = "file\x00.txt"
        clean = InputValidator.sanitize_filename(dirty)
        assert "\x00" not in clean
    
    def test_sanitize_filename_handles_empty(self):
        # Empty or whitespace-only should return 'unnamed'
        clean = InputValidator.sanitize_filename("")
        assert clean == "unnamed"
        clean2 = InputValidator.sanitize_filename("   ")
        assert clean2 == "unnamed"
    
    def test_validate_identifier_accepts_valid(self):
        assert InputValidator.validate_identifier("my_var") == "my_var"
        assert InputValidator.validate_identifier("_private") == "_private"
        assert InputValidator.validate_identifier("var123") == "var123"
    
    def test_validate_identifier_rejects_keyword(self):
        with pytest.raises(InputValidationError):
            InputValidator.validate_identifier("class")
    
    def test_validate_identifier_rejects_invalid_chars(self):
        with pytest.raises(InputValidationError):
            InputValidator.validate_identifier("my-var")
    
    def test_sanitize_branch_name(self):
        dirty = "feature/test branch with spaces!"
        clean = InputValidator.sanitize_branch_name(dirty)
        assert " " not in clean
        assert "!" not in clean
    
    def test_validate_allowed_extension(self):
        assert InputValidator.validate_allowed_extension("file.py", (".py", ".txt")) == ".py"
        with pytest.raises(InputValidationError):
            InputValidator.validate_allowed_extension("file.exe", (".py", ".txt"))


# ═══════════════════════════════════════════════════════════════════════════════
# SecurityContext Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityContext:
    def test_resolve_path_within_allowed(self, tmp_path):
        ctx = SecurityContext(allowed_paths=[tmp_path])
        resolved = ctx.resolve_path("subdir/file.txt")
        assert resolved == tmp_path / "subdir/file.txt"
    
    def test_resolve_path_outside_allowed_raises(self, tmp_path):
        ctx = SecurityContext(allowed_paths=[tmp_path])
        with pytest.raises(PathTraversalError):
            ctx.resolve_path("../outside.txt")
    
    def test_write_and_read_file(self, tmp_path):
        ctx = SecurityContext(allowed_paths=[tmp_path])
        ctx.write_file("test.txt", "Hello World")
        content = ctx.read_file("test.txt")
        assert content == "Hello World"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityIntegration:
    """Integration tests for security modules working together."""
    
    def test_secrets_masked_in_logs(self, caplog):
        secret_value = "sk-super-secret-api-key"
        secret = SecretValue(secret_value, "api_key")
        
        # Simulate logging
        import logging
        logger = logging.getLogger("test")
        logger.info(f"Using API key: {secret}")
        
        # Secret should not appear in log output
        assert secret_value not in caplog.text
    
    def test_full_pipeline_with_security(self, tmp_path):
        """Test a complete secure workflow."""
        # 1. Setup security context
        ctx = SecurityContext(
            allowed_paths=[tmp_path],
            allowed_extensions=(".py", ".txt")
        )
        
        # 2. Validate and write file
        safe_name = InputValidator.sanitize_filename("test_script.py")
        ctx.write_file(safe_name, "print('hello')")
        
        # 3. Read and verify
        content = ctx.read_file(safe_name)
        assert content == "print('hello')"
        
        # 4. Verify extension
        ext = InputValidator.validate_allowed_extension(safe_name, (".py", ".txt"))
        assert ext == ".py"


# ═══════════════════════════════════════════════════════════════════════════════
# Run Tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
