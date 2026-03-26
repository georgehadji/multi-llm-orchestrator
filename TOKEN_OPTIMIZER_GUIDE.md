# Token Optimizer Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Reduce token usage by 60-90%** with domain-specific compression strategies for common command outputs.

---

## Quick Start

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Compress any command output
compressed = optimizer.compress_command_output(
    command="git log",
    output=long_git_log_output,
    target_ratio=0.3,  # Target 30% of original size
)

print(f"Original: {len(long_git_log_output)} chars")
print(f"Compressed: {len(compressed)} chars")
print(f"Saved: {(1 - len(compressed)/len(long_git_log_output))*100:.1f}%")
```

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Commands](#supported-commands)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Custom Strategies](#custom-strategies)
6. [Integration with Orchestrator](#integration-with-orchestrator)
7. [Best Practices](#best-practices)

---

## Overview

Token Optimizer applies intelligent, domain-specific compression strategies to reduce token usage for common command outputs. This is especially useful when:

- Sending command output to LLMs for analysis
- Storing command history in limited context
- Reducing API costs for large outputs

### Compression Strategies

| Strategy | Description | Typical Savings |
|----------|-------------|-----------------|
| **Summarization** | Extract key information only | 60-80% |
| **Filtering** | Remove irrelevant lines | 50-70% |
| **Extraction** | Keep only errors/failures | 70-90% |
| **Simplification** | Remove formatting/metadata | 40-60% |
| **Sampling** | Keep every Nth line | 50-90% |

---

## Supported Commands

### Git Commands

```python
optimizer = TokenOptimizer()

# Git log - summarize commits
git_log = """
commit abc123def456
Author: John Doe <john@example.com>
Date:   Mon Mar 25 10:00:00 2026 +0200

    Add authentication feature
    
    - Implement JWT token generation
    - Add login/logout endpoints
    - Write tests

commit def456abc789
Author: Jane Smith <jane@example.com>
Date:   Sun Mar 24 15:30:00 2026 +0200

    Fix bug in password reset flow
"""

compressed = optimizer.compress_command_output(
    command="git log",
    output=git_log,
    target_ratio=0.3,
)

print(compressed)
# Output:
# abc123: Add authentication feature (John Doe)
# def456: Fix bug in password reset flow (Jane Smith)
```

### Test Outputs

```python
# Pytest output - extract failures only
pytest_output = """
============================= test session starts =============================
platform linux -- Python 3.10.0
collected 50 items

test_auth.py::test_login PASSED
test_auth.py::test_logout PASSED
test_auth.py::test_register FAILED
test_auth.py::test_password_reset FAILED

================================== FAILURES ===================================
_____________________________ test_register ______________________________

    def test_register():
>       assert response.status_code == 201
E       assert 500 == 201

___________________________ test_password_reset __________________________

    def test_password_reset():
>       assert email_sent == True
E       assert False == True

=========================== short test summary info ============================
FAILED test_auth.py::test_register - assert 500 == 201
FAILED test_auth.py::test_password_reset - assert False == True
========================= 2 failed, 48 passed in 5.23s =========================
"""

compressed = optimizer.compress_command_output(
    command="pytest",
    output=pytest_output,
    target_ratio=0.2,
)

print(compressed)
# Output:
# FAILED: test_auth.py::test_register - assert 500 == 201
# FAILED: test_auth.py::test_password_reset - assert False == True
# 2 failed, 48 passed in 5.23s
```

### Linter Outputs

```python
# ESLint output - extract errors only
eslint_output = """

/src/components/Auth.tsx
  15:5  error  'useState' is defined but never used     no-unused-vars
  23:10 error  Missing return type on function          @typescript-eslint/typeref
  45:3  warning  Unexpected console statement           no-console

/src/utils/helpers.ts
  8:1  error  Expected indentation of 2 spaces           indent
  12:5  error  'config' is assigned a value but never used  no-unused-vars

✖ 5 problems (4 errors, 1 warning)
  2 errors and 1 warning potentially fixable with the `--fix` option.

"""

compressed = optimizer.compress_command_output(
    command="eslint",
    output=eslint_output,
    target_ratio=0.3,
)

print(compressed)
# Output:
# Errors:
#   Auth.tsx:15:5 - 'useState' is defined but never used
#   Auth.tsx:23:10 - Missing return type on function
#   helpers.ts:8:1 - Expected indentation of 2 spaces
#   helpers.ts:12:5 - 'config' is assigned a value but never used
# Total: 4 errors, 1 warning
```

### System Commands

```python
# Docker ps - table format
docker_output = """
CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS        PORTS                    NAMES
abc123def456   nginx:latest           "/docker-entrypoint.…"   2 hours ago    Up 2 hours    0.0.0.0:80->80/tcp       web-server
def456abc789   redis:alpine           "docker-entrypoint.s…"   3 hours ago    Up 3 hours    6379/tcp                 redis-cache
789abc123def   postgres:14            "docker-entrypoint.s…"   1 day ago      Up 1 day      5432/tcp                 db-postgres
"""

compressed = optimizer.compress_command_output(
    command="docker ps",
    output=docker_output,
    target_ratio=0.5,
)

print(compressed)
# Output:
# web-server (nginx:latest) - Up 2 hours - port 80
# redis-cache (redis:alpine) - Up 3 hours
# db-postgres (postgres:14) - Up 1 day - port 5432

# ps aux - top processes only
ps_output = """
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1  18504  3200 ?        Ss   Mar24   0:03 /sbin/init
john      1234  5.2  2.5 250000 50000 ?       Sl   10:00   2:30 python app.py
john      2345  2.1  1.2 150000 25000 ?       Sl   10:00   1:15 node server.js
"""

compressed = optimizer.compress_command_output(
    command="ps aux",
    output=ps_output,
    target_ratio=0.3,
)

print(compressed)
# Output:
# Top processes by CPU:
#   PID 1234 (python app.py) - CPU: 5.2%, MEM: 2.5%
#   PID 2345 (node server.js) - CPU: 2.1%, MEM: 1.2%
```

---

## API Reference

### TokenOptimizer Class

```python
class TokenOptimizer:
    """Domain-specific token compression for command outputs."""
    
    def __init__(self):
        """Initialize optimizer with built-in strategies."""
    
    def compress_command_output(
        self,
        command: str,
        output: str,
        target_ratio: float = 0.5,
    ) -> str:
        """
        Compress command output using appropriate strategy.
        
        Args:
            command: Command that generated output (e.g., "git log")
            output: The output to compress
            target_ratio: Target size ratio (0.1 = 10% of original)
            
        Returns:
            Compressed output string
        """
    
    def register_strategy(
        self,
        command: str,
        strategy: Callable[[str, float], str],
    ) -> None:
        """
        Register custom compression strategy.
        
        Args:
            command: Command name to associate with strategy
            strategy: Function that takes (output, target_ratio) and returns compressed string
        """
    
    def get_compression_stats(
        self,
        original: str,
        compressed: str,
    ) -> CompressionStats:
        """
        Get compression statistics.
        
        Returns:
            CompressionStats with original_size, compressed_size, ratio, savings_percent
        """
```

### CompressionStats

```python
@dataclass
class CompressionStats:
    """Statistics for compression operation."""
    
    original_size: int       # Original character count
    compressed_size: int     # Compressed character count
    ratio: float            # Compressed / Original ratio
    savings_percent: float  # Percentage saved
    
    def __str__(self) -> str:
        return (
            f"Compression: {self.original_size} → {self.compressed_size} chars "
            f"({self.savings_percent:.1f}% saved)"
        )
```

---

## Usage Examples

### Example 1: Git History Analysis

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Get full git log
import subprocess
git_log = subprocess.check_output(
    ["git", "log", "--oneline", "-50"],
    text=True,
)

# Compress for LLM context
compressed = optimizer.compress_command_output(
    command="git log",
    output=git_log,
    target_ratio=0.5,
)

# Send to LLM
response = await llm.generate(
    prompt=f"Analyze this git history:\n{compressed}",
    max_tokens=500,
)
```

### Example 2: CI/CD Pipeline Analysis

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Get CI pipeline output
ci_output = """
Running pipeline for commit abc123...

Stage: Build
  - npm install: SUCCESS (15s)
  - npm run build: SUCCESS (45s)

Stage: Test
  - npm run test:unit: SUCCESS (30s)
  - npm run test:integration: FAILED (120s)
    
    FAIL test/integration/auth.test.js
      ● Authentication › should login successfully
      
        expect(received).toBe(expected)
        
        Expected: 200
        Received: 500
        
        at login.test.js:25:15

Stage: Deploy
  - SKIPPED (tests failed)

Pipeline failed with exit code 1
"""

# Compress to failures only
compressed = optimizer.compress_command_output(
    command="ci_pipeline",
    output=ci_output,
    target_ratio=0.2,
)

print(compressed)
# Output:
# Pipeline FAILED
# Failed stage: Test
# Failed test: test/integration/auth.test.js
# Error: Expected 200, Received 500 at login.test.js:25
```

### Example 3: Performance Analysis

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Get benchmark output
benchmark_output = """
============================= benchmark session ==============================
platform linux -- Python 3.10.0
benchmark: 5 functions, 1000 iterations

test_fibonacci_recursive         1000 loops, best of 5: 234 usec per loop
test_fibonacci_iterative        10000 loops, best of 5: 45.2 usec per loop
test_fibonacci_memoized        100000 loops, best of 5: 2.34 usec per loop
test_fibonacci_matrix         1000000 loops, best of 5: 0.45 usec per loop
test_fibonacci_closed_form  10000000 loops, best of 5: 0.12 usec per loop

============================= summary ==============================
Name                      Min (usec)    Max (usec)    Mean (usec)    StdDev
test_fibonacci_recursive     230.5        245.2        234.1         5.2
test_fibonacci_iterative      44.1         48.5         45.2         1.8
test_fibonacci_memoized        2.1          2.8          2.34        0.2
test_fibonacci_matrix          0.4          0.5          0.45        0.05
test_fibonacci_closed_form     0.1          0.15         0.12        0.01
"""

# Extract key results
compressed = optimizer.compress_command_output(
    command="pytest-benchmark",
    output=benchmark_output,
    target_ratio=0.3,
)

print(compressed)
# Output:
# Benchmark Results (fastest to slowest):
#   fibonacci_closed_form: 0.12 usec (1950x faster than recursive)
#   fibonacci_matrix: 0.45 usec (520x faster)
#   fibonacci_memoized: 2.34 usec (100x faster)
#   fibonacci_iterative: 45.2 usec (5x faster)
#   fibonacci_recursive: 234 usec (baseline)
```

### Example 4: Error Log Analysis

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Get application logs
logs = """
2026-03-25 10:00:00 INFO  Starting application...
2026-03-25 10:00:01 INFO  Connected to database
2026-03-25 10:00:02 INFO  Server listening on port 8080
2026-03-25 10:05:23 ERROR Database connection lost
    Traceback (most recent call last):
      File "db.py", line 45, in connect
        connection = await pool.acquire()
      File "pool.py", line 123, in acquire
        raise ConnectionError("Pool exhausted")
    ConnectionError: Pool exhausted
2026-03-25 10:05:24 WARN  Retrying connection...
2026-03-25 10:05:25 INFO  Reconnected to database
2026-03-25 10:15:45 ERROR Request timeout for /api/users
    TimeoutError: Request took 30.5s (limit: 30s)
2026-03-25 10:30:00 INFO  Health check passed
"""

# Extract errors only
compressed = optimizer.compress_command_output(
    command="application_logs",
    output=logs,
    target_ratio=0.3,
)

print(compressed)
# Output:
# ERRORS:
# [10:05:23] Database connection lost - ConnectionError: Pool exhausted
# [10:15:45] Request timeout for /api/users - TimeoutError: 30.5s
```

---

## Custom Strategies

### Register Custom Strategy

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Register custom strategy for npm ls
@optimizer.register_strategy("npm_custom")
def compress_npm_custom(output: str, target_ratio: float) -> str:
    """Custom npm ls compression."""
    lines = output.split('\n')
    
    # Keep only top-level dependencies
    compressed = []
    for line in lines:
        if not line.startswith('│') and not line.startswith('└'):
            compressed.append(line)
        elif line.count('│') + line.count('└') < 2:
            compressed.append(line)
    
    # Limit to target ratio
    max_lines = int(len(lines) * target_ratio)
    return '\n'.join(compressed[:max_lines])

# Use custom strategy
compressed = optimizer.compress_command_output(
    command="npm_custom",
    output=npm_ls_output,
    target_ratio=0.3,
)
```

### Override Built-in Strategy

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Override git log strategy
@optimizer.register_strategy("git_log")
def compress_git_log_custom(output: str, target_ratio: float) -> str:
    """Custom git log compression with author info."""
    import re
    
    commits = []
    current_commit = {}
    
    for line in output.split('\n'):
        if line.startswith('commit '):
            if current_commit:
                commits.append(current_commit)
            current_commit = {'hash': line.split()[1]}
        elif line.startswith('Author:'):
            current_commit['author'] = line.split(':', 1)[1].strip()
        elif line.startswith('Date:'):
            current_commit['date'] = line.split(':', 1)[1].strip()
        elif line.strip():
            current_commit.setdefault('message', []).append(line.strip())
    
    if current_commit:
        commits.append(current_commit)
    
    # Format compressed output
    compressed = []
    for commit in commits[:int(len(commits) * target_ratio)]:
        author = commit.get('author', 'Unknown').split('<')[0].strip()
        message = commit.get('message', ['No message'])[0]
        compressed.append(f"{commit['hash'][:8]}: {message} ({author})")
    
    return '\n'.join(compressed)
```

---

## Integration with Orchestrator

### Automatic Token Optimization

```python
from orchestrator import Orchestrator
from orchestrator.token_optimizer import TokenOptimizer

orch = Orchestrator()
optimizer = TokenOptimizer()

async def run_with_token_optimization():
    # Run project
    state = await orch.run_project(
        project_description="Build REST API",
        success_criteria="All tests pass",
    )
    
    # Get test output
    test_output = state.test_results
    
    # Compress for LLM analysis
    compressed_tests = optimizer.compress_command_output(
        command="pytest",
        output=test_output,
        target_ratio=0.2,
    )
    
    # Use compressed output for analysis
    analysis = await orch.analyze(
        prompt=f"Analyze test failures:\n{compressed_tests}",
    )
    
    return analysis
```

### Context Management

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

class ContextManager:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.context = []
        self.optimizer = optimizer
    
    def add_command_output(self, command: str, output: str):
        """Add command output to context with compression."""
        # Estimate tokens
        estimated_tokens = len(output) / 4
        
        if estimated_tokens > self.max_tokens * 0.3:
            # Compress if output is too large
            target_ratio = (self.max_tokens * 0.2) / estimated_tokens
            output = self.optimizer.compress_command_output(
                command=command,
                output=output,
                target_ratio=max(0.1, target_ratio),
            )
        
        self.context.append({
            "command": command,
            "output": output,
            "tokens": len(output) / 4,
        })
    
    def get_context(self) -> str:
        """Get full context as string."""
        return '\n'.join([item["output"] for item in self.context])
    
    def get_total_tokens(self) -> int:
        """Get total token count."""
        return int(sum(item["tokens"] for item in self.context))
```

---

## Best Practices

### 1. Choose Appropriate Target Ratio

```python
# High compression (10-20%) - for quick summaries
compressed = optimizer.compress_command_output(
    command="pytest",
    output=output,
    target_ratio=0.15,
)

# Medium compression (30-50%) - for analysis
compressed = optimizer.compress_command_output(
    command="git log",
    output=output,
    target_ratio=0.4,
)

# Low compression (60-80%) - for detailed review
compressed = optimizer.compress_command_output(
    command="eslint",
    output=output,
    target_ratio=0.7,
)
```

### 2. Chain Compression Strategies

```python
# First filter errors, then summarize
errors_only = optimizer.compress_command_output(
    command="pytest",
    output=full_output,
    target_ratio=0.3,
)

final = optimizer.compress_command_output(
    command="summary",
    output=errors_only,
    target_ratio=0.5,
)
```

### 3. Monitor Compression Quality

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

compressed = optimizer.compress_command_output(
    command="git log",
    output=original,
    target_ratio=0.3,
)

stats = optimizer.get_compression_stats(original, compressed)
print(stats)
# Compression: 5000 → 1500 chars (70.0% saved)

# Verify key information preserved
assert "commit" in compressed or "fix" in compressed.lower()
```

---

## Configuration

### Environment Variables

```bash
# Default compression ratio
export TOKEN_OPTIMIZER_DEFAULT_RATIO=0.5

# Enable/disable specific strategies
export TOKEN_OPTIMIZER_GIT_ENABLED=true
export TOKEN_OPTIMIZER_PYTEST_ENABLED=true
export TOKEN_OPTIMIZER_ESLINT_ENABLED=true

# Minimum compression threshold
export TOKEN_OPTIMIZER_MIN_SIZE=1000  # Don't compress under 1000 chars
```

---

## Related Documentation

- [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md) — Preflight validation
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
