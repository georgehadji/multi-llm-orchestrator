# AI Orchestrator - Security Guide

## 🔒 Security Overview

This document outlines the security features, best practices, and guidelines for securely using the AI Orchestrator.

---

## Critical Security Rules

### 1. API Key Management (CRITICAL)

**NEVER commit API keys to version control!**

#### ✅ Correct Approach
```bash
# 1. Copy the template
cp .env.example .env

# 2. Edit .env with your actual keys (NOT committed)
nano .env

# 3. Ensure .env is in .gitignore
grep "^\.env$" .gitignore
```

#### ❌ WRONG - Never Do This
```bash
# DON'T commit actual keys
git add .env
git commit -m "Add API keys"
```

#### Production Secrets Management
For production deployments, use proper secrets management:
- **AWS**: AWS Secrets Manager
- **Azure**: Azure Key Vault  
- **GCP**: Secret Manager
- **HashiCorp**: Vault
- **Docker**: Docker Secrets

### 2. Debug Mode (CRITICAL)

**Always disable debug mode in production:**

```bash
# .env - Production Settings
DEBUG=false
LOG_LEVEL=INFO
```

Debug mode (`DEBUG=true`) can leak:
- API keys in logs
- Internal system paths
- Stack traces with sensitive data

### 3. Shell Command Execution (HIGH)

The orchestrator has been hardened to prevent command injection:

- ✅ Uses `shell=False` in all subprocess calls
- ✅ Validates command arguments
- ✅ Sanitizes user inputs

**If you extend the code:**
```python
# ✅ CORRECT - Safe subprocess
from orchestrator.secure_execution import SecureSubprocess

result = SecureSubprocess.run(
    ["python", "-c", user_code],  # List, not string
    timeout=60
)

# ❌ WRONG - Never do this
import subprocess
subprocess.run(f"python -c {user_code}", shell=True)  # DANGEROUS!
```

### 4. Path Traversal Protection (HIGH)

All file paths are validated to prevent directory traversal:

```python
from orchestrator.secure_execution import SecurePath, PathTraversalError

try:
    safe_path = SecurePath(
        base_path=Path("/allowed/base"),
        user_input=user_provided_path
    )
    print(f"Safe path: {safe_path.resolved}")
except PathTraversalError:
    print("Path traversal attack detected!")
```

---

## Security Features

### Secrets Manager (`orchestrator.secrets_manager`)

```python
from orchestrator.secrets_manager import get_secrets_manager

# Automatic loading from environment
secrets = get_secrets_manager()

# Safe access - keys are masked in logs/repr
api_key = secrets.get("openai_api_key")
# Returns: "sk-..." or None

# Required keys
api_key = secrets.require("openai_api_key")  # Raises if missing

# Mask secrets in text
masked = secrets.mask_in_text(log_message)
```

### Input Validator (`orchestrator.secure_execution`)

```python
from orchestrator.secure_execution import InputValidator

# Filename sanitization
safe_name = InputValidator.sanitize_filename("../../../etc/passwd")
# Returns: "etc_passwd"

# Identifier validation
clean_id = InputValidator.validate_identifier("my_var_123")
# Returns: "my_var_123" or raises InputValidationError

# Branch name sanitization
safe_branch = InputValidator.sanitize_branch_name("feature/test branch!")
# Returns: "feature/test_branch_"
```

### Secure Subprocess (`orchestrator.secure_execution`)

```python
from orchestrator.secure_execution import SecureSubprocess

# Never uses shell=True
result = SecureSubprocess.run(
    ["git", "status"],  # List format required
    cwd="/path/to/repo",
    timeout=30
)

# Async version
returncode, stdout, stderr = await SecureSubprocess.run_async(
    ["python", "script.py"],
    timeout=60
)
```

---

## Security Checklist

### Before Committing Code

- [ ] No API keys in committed files
- [ ] `.env` is in `.gitignore`
- [ ] No `shell=True` in subprocess calls
- [ ] User inputs are validated/sanitized
- [ ] File paths use `SecurePath`
- [ ] Debug mode is disabled
- [ ] No hardcoded credentials

### Before Production Deployment

- [ ] `DEBUG=false` in environment
- [ ] `LOG_LEVEL=INFO` or higher
- [ ] Secrets managed via proper vault/service
- [ ] Network access restricted
- [ ] File system permissions set correctly
- [ ] Rate limiting enabled
- [ ] Audit logging configured

---

## Common Security Issues

### Issue 1: Accidentally Committed API Keys

**Detection:**
```bash
# Check for potential secrets
grep -r "sk-" . --include="*.py" --include="*.json" --include="*.yaml"
grep -r "AIza" . --include="*.py" --include="*.json" --include="*.yaml"
```

**Remediation:**
1. Immediately rotate the exposed key
2. Remove from git history (see below)
3. Update `.gitignore`

### Issue 2: Removing Secrets from Git History

```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove file from history
git filter-repo --path .env --invert-paths

# Or remove specific strings
git filter-repo --replace-text <(echo "SECRET_KEY==>REMOVED")

# Force push (careful!)
git push origin --force --all
```

### Issue 3: Path Traversal in User Input

**Vulnerable:**
```python
path = f"{base_dir}/{user_input}"
```

**Secure:**
```python
from orchestrator.secure_execution import SecurePath
path = SecurePath(base_dir, user_input).resolved
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input Layer                          │
│         (CLI args, file uploads, API requests)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Input Validation Layer                      │
│    (SecurePath, InputValidator, filename sanitization)      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Secrets Management Layer                    │
│         (SecretsManager, environment variables)             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Secure Execution Layer                       │
│      (SecureSubprocess - no shell=True, no injection)       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output/Logging Layer                       │
│           (masked secrets, safe file operations)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Email: security@example.com (placeholder)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

---

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [Bandit - Python Security Linter](https://bandit.readthedocs.io/)

---

## License

This security documentation is part of the AI Orchestrator project.
