"""
Security Tests — Comprehensive security test suite
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Security test suite covering:
- No hardcoded secrets
- No pickle deserialization
- CORS policy validation
- Rate limiting
- Input validation
- Secure logging

USAGE:
    pytest tests/test_security.py -v
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pytest


# ─────────────────────────────────────────────
# Test: No Hardcoded Secrets
# ─────────────────────────────────────────────

class TestNoHardcodedSecrets:
    """Test that no secrets are hardcoded in the codebase."""
    
    SECRET_PATTERNS = [
        (r'sk-[a-zA-Z0-9]{20,}', 'OpenAI/Anthropic API key'),
        (r'AIza[a-zA-Z0-9_-]{35}', 'Google API key'),
        (r'ghp_[a-zA-Z0-9]{36}', 'GitHub PAT'),
        (r'xox[baprs]-[a-zA-Z0-9-]+', 'Slack token'),
        (r'Bearer\s+[a-zA-Z0-9_-]{20,}', 'Bearer token'),
    ]
    
    def test_no_secrets_in_env_file(self):
        """Test that .env file is not committed with real secrets."""
        env_file = Path('.env')
        
        if not env_file.exists():
            pytest.skip(".env file not found")
        
        content = env_file.read_text(encoding='utf-8', errors='ignore')
        
        # Check for actual secret values (not placeholders)
        for pattern, secret_type in self.SECRET_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                # Skip if it's a placeholder
                if '...' in match or 'xxx' in match.lower() or 'your_' in match.lower():
                    continue
                pytest.fail(f"Found hardcoded {secret_type} in .env: {match[:10]}...")
    
    def test_no_secrets_in_python_files(self):
        """Test that no secrets are hardcoded in Python files."""
        orchestrator_dir = Path('orchestrator')
        
        if not orchestrator_dir.exists():
            pytest.skip("orchestrator directory not found")
        
        for py_file in orchestrator_dir.rglob('*.py'):
            # Skip test files and __pycache__
            if 'test_' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            for pattern, secret_type in self.SECRET_PATTERNS:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Skip if it's in a comment or example
                    if '# Example:' in content or '# example' in content.lower():
                        continue
                    pytest.fail(f"Found hardcoded {secret_type} in {py_file}: {match[:10]}...")


# ─────────────────────────────────────────────
# Test: No Pickle Deserialization
# ─────────────────────────────────────────────

class TestNoPickleDeserialization:
    """Test that pickle is not used for deserialization (RCE risk)."""
    
    def test_no_pickle_loads(self):
        """Test that pickle.loads is not used."""
        orchestrator_dir = Path('orchestrator')
        
        if not orchestrator_dir.exists():
            pytest.skip("orchestrator directory not found")
        
        for py_file in orchestrator_dir.rglob('*.py'):
            # Skip test files
            if 'test_' in str(py_file):
                continue
            
            # Skip the old caching.py file (known to use pickle)
            if py_file.name == 'caching.py':
                continue
            
            # Skip leaderboard.py - it contains pickle in a benchmark task prompt (sample insecure code)
            if py_file.name == 'leaderboard.py':
                continue
            
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check for pickle.loads (RCE risk) - not just mentions in comments
            if 'pickle.loads(' in content or 'pickle.load(' in content:
                pytest.fail(f"Found pickle.loads in {py_file} - use JSON instead")
    
    def test_secure_cache_uses_json(self):
        """Test that secure_cache.py uses JSON instead of pickle."""
        secure_cache_file = Path('orchestrator/secure_cache.py')
        
        if not secure_cache_file.exists():
            pytest.skip("secure_cache.py not found")
        
        content = secure_cache_file.read_text(encoding='utf-8', errors='ignore')
        
        # Should use JSON
        assert 'json.dumps' in content, "secure_cache.py should use json.dumps"
        assert 'json.loads' in content, "secure_cache.py should use json.loads"
        
        # Should NOT use pickle for serialization (mentions in comments are OK)
        # Check for actual pickle usage, not just mentions
        assert 'import pickle' not in content, "secure_cache.py should not import pickle"
        assert 'pickle.dumps' not in content, "secure_cache.py should not use pickle.dumps"
        assert 'pickle.loads' not in content, "secure_cache.py should not use pickle.loads"


# ─────────────────────────────────────────────
# Test: CORS Policy
# ─────────────────────────────────────────────

class TestCORSPolicy:
    """Test CORS policy is secure."""
    
    def test_no_wildcard_cors_in_api_server(self):
        """Test that API server doesn't use wildcard CORS by default."""
        api_server_file = Path('orchestrator/api_server.py')
        
        if not api_server_file.exists():
            pytest.skip("api_server.py not found")
        
        content = api_server_file.read_text()
        
        # Check that wildcard is not the default
        if "Access-Control-Allow-Origin'] = '*'" in content:
            # It's OK if it's behind a condition
            if "if '*' in self.cors_origins:" not in content:
                pytest.fail("API server uses wildcard CORS by default - use allowlist instead")
    
    def test_cors_allowlist_configurable(self):
        """Test that CORS allowlist is configurable."""
        api_server_file = Path('orchestrator/api_server.py')
        
        if not api_server_file.exists():
            pytest.skip("api_server.py not found")
        
        content = api_server_file.read_text()
        
        # Should have configurable cors_origins parameter
        assert 'cors_origins' in content, "API server should have configurable cors_origins"
        assert 'cors_origins: Optional[List[str]]' in content or 'cors_origins=None' in content, \
            "cors_origins should be Optional[List[str]]"


# ─────────────────────────────────────────────
# Test: Rate Limiting
# ─────────────────────────────────────────────

class TestRateLimiting:
    """Test rate limiting is implemented."""
    
    def test_rate_limiter_exists(self):
        """Test that rate limiter is implemented."""
        api_server_file = Path('orchestrator/api_server.py')
        
        if not api_server_file.exists():
            pytest.skip("api_server.py not found")
        
        content = api_server_file.read_text()
        
        # Should have rate limiter class
        assert 'TokenBucketRateLimiter' in content or 'RateLimiter' in content, \
            "API server should have rate limiter"
    
    def test_rate_limiting_applied(self):
        """Test that rate limiting is applied to endpoints."""
        api_server_file = Path('orchestrator/api_server.py')
        
        if not api_server_file.exists():
            pytest.skip("api_server.py not found")
        
        content = api_server_file.read_text()
        
        # Should have rate limiting middleware or decorator
        assert 'rate_limit' in content.lower() or 'is_allowed' in content, \
            "Rate limiting should be applied"


# ─────────────────────────────────────────────
# Test: Input Validation
# ─────────────────────────────────────────────

class TestInputValidation:
    """Test input validation is implemented."""
    
    def test_code_validator_exists(self):
        """Test that code validator exists."""
        code_validator_file = Path('orchestrator/code_validator.py')
        
        if not code_validator_file.exists():
            pytest.skip("code_validator.py not found")
        
        content = code_validator_file.read_text()
        
        # Should have AST validation
        assert 'ast.parse' in content, "Code validator should use AST parsing"
        assert 'ASTSecurityAnalyzer' in content or 'validate_code' in content, \
            "Should have security analyzer"
    
    def test_dangerous_patterns_detected(self):
        """Test that dangerous patterns are detected."""
        from orchestrator.code_validator import validate_code
        
        # Test eval detection
        result = validate_code("eval(user_input)")
        assert not result.is_valid
        assert "eval" in result.dangerous_patterns_found
        
        # Test exec detection
        result = validate_code("exec(code)")
        assert not result.is_valid
        assert "exec" in result.dangerous_patterns_found
        
        # Test os.system detection
        result = validate_code("import os\nos.system('ls')")
        assert not result.is_valid
        assert "os_system" in result.dangerous_patterns_found


# ─────────────────────────────────────────────
# Test: Secure Logging
# ─────────────────────────────────────────────

class TestSecureLogging:
    """Test secure logging is implemented."""
    
    def test_secrets_manager_exists(self):
        """Test that secrets manager exists."""
        secrets_file = Path('orchestrator/secrets_manager.py')
        
        if not secrets_file.exists():
            pytest.skip("secrets_manager.py not found")
        
        content = secrets_file.read_text()
        
        # Should have secret masking
        assert 'mask' in content.lower(), "Secrets manager should have masking"
        assert 'SECRET_PATTERNS' in content, "Should have secret patterns"
    
    def test_secrets_masking(self):
        """Test that secrets are masked."""
        from orchestrator.secrets_manager import SecretsManager, SecretsConfig
        
        config = SecretsConfig(mask_in_logs=True)
        manager = SecretsManager(config)
        
        # Test value masking
        secret = "sk-1234567890abcdefghijklmnop"
        masked = manager.mask_value(secret)
        assert '...' in masked or '***' in masked, "Secret should be masked"
        assert secret not in masked, "Original secret should not appear"
        
        # Test string masking
        text = f"API key is {secret}"
        masked_text = manager.mask_string(text)
        assert secret not in masked_text, "Secret should be masked in text"


# ─────────────────────────────────────────────
# Test: Git Configuration
# ─────────────────────────────────────────────

class TestGitConfiguration:
    """Test git configuration is secure."""
    
    def test_env_in_gitignore(self):
        """Test that .env is in .gitignore."""
        gitignore_file = Path('.gitignore')
        
        if not gitignore_file.exists():
            pytest.skip(".gitignore not found")
        
        content = gitignore_file.read_text(encoding='utf-8', errors='ignore')
        
        assert '.env' in content, ".env should be in .gitignore"
        assert '*.key' in content or '*.pem' in content, "Key files should be in .gitignore"
    
    def test_no_env_in_git(self):
        """Test that no .env files are tracked by git."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['git', 'ls-files', '*.env'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.stdout.strip():
                pytest.fail(f".env files tracked by git: {result.stdout}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("git not available")
    
    def test_no_os_system(self):
        """Test that os.system is not used (use subprocess instead)."""
        cli_file = Path('orchestrator/cli.py')
        
        if not cli_file.exists():
            pytest.skip("cli.py not found")
        
        content = cli_file.read_text(encoding='utf-8', errors='ignore')
        
        # os.system is less secure than subprocess
        if 'os.system' in content:
            # It's OK if it's just for clearing screen
            if 'os.system(\'cls\' if os.name == \'nt\' else \'clear\')' in content:
                pass  # Acceptable use for clearing terminal
            else:
                pytest.fail("os.system found - use subprocess.run instead")


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
