# Preflight, Session Watcher & Persona Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Mnemo Cortex Integration** — Quality control, memory management, and behavior customization for AI Orchestrator.

---

## Quick Start

### 1. Preflight Validation

```python
from orchestrator.preflight import PreflightValidator, PreflightMode

validator = PreflightValidator()

# Validate response before sending
result = validator.validate(
    response="Here's the code...",
    context={"task": "code_generation"},
    mode=PreflightMode.WARN
)

if result.action == "BLOCK":
    print(f"Blocked: {result.reason}")
```

### 2. Session Watcher

```python
from orchestrator.session_watcher import SessionWatcher

watcher = SessionWatcher()

# Start session
session_id = watcher.start_session("project_001")

# Record interaction
watcher.record_interaction(
    session_id=session_id,
    task_input="Write fibonacci function",
    task_output="def fibonacci(n): ...",
    task_type="code_generation",
)

# Get context
context = await watcher.get_context(session_id, limit=5)
```

### 3. Persona Modes

```python
from orchestrator.persona import PersonaManager, PersonaMode

manager = PersonaManager()

# Set persona
manager.set_persona("project_001", PersonaMode.STRICT)

# Get settings
settings = manager.get_persona_settings("project_001")
print(f"Temperature: {settings.temperature}")
```

---

## Table of Contents

1. [Preflight Validation](#1-preflight-validation)
2. [Session Watcher](#2-session-watcher)
3. [Persona Modes](#3-persona-modes)
4. [Token Optimizer](#4-token-optimizer)
5. [Integration Examples](#5-integration-examples)

---

## 1. Preflight Validation

Preflight validation checks AI responses before they're sent to the user, ensuring quality and safety.

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `PASS` | Allow all responses | Development, testing |
| `ENRICH` | Add missing context | Documentation, explanations |
| `WARN` | Flag potential issues | Code review, analysis |
| `BLOCK` | Prevent problematic responses | Production, safety-critical |
| `AUTO` | Auto-select based on task | General use |

### Check Types

| Check Type | Description | Severity |
|------------|-------------|----------|
| `SAFETY` | Harmful content detection | 8-10 |
| `ACCURACY` | Factual accuracy | 6-8 |
| `COMPLETENESS` | Response completeness | 4-6 |
| `TONALITY` | Appropriate tone | 3-5 |
| `FORMAT` | Output format validation | 4-6 |
| `PRIVACY` | PII/sensitive data detection | 9-10 |
| `SECURITY` | Security vulnerabilities | 8-10 |

### Basic Usage

```python
from orchestrator.preflight import (
    PreflightValidator,
    PreflightMode,
    PreflightAction,
    CheckType,
)

validator = PreflightValidator()

# Simple validation
result = validator.validate(
    response="def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    context={
        "task": "code_generation",
        "language": "python",
        "user_request": "Write fibonacci function"
    },
    mode=PreflightMode.WARN,
)

print(f"Action: {result.action.value}")
print(f"Passed: {result.passed}")
print(f"Warnings: {result.warnings}")

# Check individual results
for check in result.checks:
    print(f"{check.check_type.value}: {'✓' if check.passed else '✗'} (severity: {check.severity})")
```

### Advanced: Custom Checks

```python
from orchestrator.preflight import PreflightValidator, CheckResult, CheckType

validator = PreflightValidator()

# Register custom check
@validator.register_check(CheckType.SECURITY)
def check_sql_injection(response: str, context: dict) -> CheckResult:
    import re
    
    sql_patterns = [
        r"SELECT.*FROM.*WHERE",
        r"INSERT.*INTO.*VALUES",
        r"DROP.*TABLE",
        r"--.*$",  # SQL comments
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return CheckResult(
                check_type=CheckType.SECURITY,
                passed=False,
                severity=9,
                message="Potential SQL injection detected",
                details={"pattern": pattern}
            )
    
    return CheckResult(
        check_type=CheckType.SECURITY,
        passed=True,
        severity=0,
        message="No SQL injection detected"
    )

# Now validator will check for SQL injection
result = validator.validate(
    response="SELECT * FROM users WHERE id = ?",
    context={"task": "code_generation"},
    mode=PreflightMode.BLOCK,
)
```

### Integration with Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.preflight import PreflightValidator, PreflightMode

orch = Orchestrator()
validator = PreflightValidator()

async def run_project_with_preflight():
    state = await orch.run_project(
        project_description="Build authentication service",
        success_criteria="All tests pass",
    )
    
    # Validate generated code
    result = validator.validate(
        response=state.generated_code,
        context={
            "task": "code_generation",
            "project_type": "fastapi",
        },
        mode=PreflightMode.WARN,
    )
    
    if result.has_blocking_issues:
        print("Code has blocking issues, requesting revision...")
        # Trigger revision
    
    return state
```

---

## 2. Session Watcher

Session Watcher automatically captures conversations and stores them in a tiered memory system (HOT/WARM/COLD).

### Memory Tiers

| Tier | Age | Storage | Access |
|------|-----|---------|--------|
| `HOT` | Days 1-3 | Raw JSONL, instant keyword search | Instant |
| `WARM` | Days 4-30 | Summarized + embedded, semantic search | Fast |
| `COLD` | Day 30+ | Compressed archive, full scan | Slow |

### Basic Usage

```python
from orchestrator.session_watcher import SessionWatcher, MemoryTier

watcher = SessionWatcher()

# Start session
session_id = watcher.start_session(
    project_id="project_001",
    metadata={
        "user": "john_doe",
        "description": "Building FastAPI service"
    }
)

# Record interactions
watcher.record_interaction(
    session_id=session_id,
    task_input="Write authentication endpoint",
    task_output="async def login(...)",
    task_type="code_generation",
    metadata={
        "model": "gpt-4o",
        "tokens_used": 1500,
        "duration_ms": 2500,
    },
)

# Get recent context (HOT tier)
context = await watcher.get_context(session_id, limit=5)
print(f"Recent interactions: {len(context)}")

# Search sessions
sessions = await watcher.search_sessions(
    query="authentication",
    project_id="project_001",
)

# Archive session (move to COLD)
await watcher.archive_session(session_id)
```

### Session Lifecycle

```python
from orchestrator.session_watcher import SessionWatcher, SessionStatus

watcher = SessionWatcher()
session_id = watcher.start_session("project_001")

# Check status
status = watcher.get_session_status(session_id)
print(f"Status: {status.value}")  # "active"

# Get session info
info = watcher.get_session_info(session_id)
print(f"Created: {info.created_at}")
print(f"Interactions: {len(info.interactions)}")
print(f"Memory tier: {info.memory_tier.value}")

# Summarize session (move to WARM)
summary = await watcher.summarize_session(session_id)
print(f"Summary: {summary}")

# Archive session (move to COLD)
await watcher.archive_session(session_id)
status = watcher.get_session_status(session_id)
print(f"New status: {status.value}")  # "archived"
```

### Context Retrieval

```python
from orchestrator.session_watcher import SessionWatcher

watcher = SessionWatcher()

# Get HOT context (recent interactions)
hot_context = await watcher.get_context(
    session_id="proj_001",
    tier=MemoryTier.HOT,
    limit=10,
)

# Get WARM context (summarized)
warm_context = await watcher.get_context(
    session_id="proj_001",
    tier=MemoryTier.WARM,
)

# Get full history
full_history = await watcher.get_full_history(
    session_id="proj_001",
    include_summaries=True,
)

# Semantic search
results = await watcher.semantic_search(
    query="How did we implement authentication?",
    project_id="proj_001",
    limit=5,
)
```

### Multi-Project Tracking

```python
from orchestrator.session_watcher import SessionWatcher

watcher = SessionWatcher()

# Start multiple sessions
sessions = {}
for project_id in ["proj_1", "proj_2", "proj_3"]:
    sessions[project_id] = watcher.start_session(project_id)

# Record interactions for each
for project_id, session_id in sessions.items():
    watcher.record_interaction(
        session_id=session_id,
        task_input=f"Task for {project_id}",
        task_output=f"Result for {project_id}",
        task_type="code_generation",
    )

# Get all active sessions
active = watcher.get_active_sessions()
print(f"Active sessions: {len(active)}")

# Get sessions by project
proj_sessions = watcher.get_sessions_by_project("proj_1")
```

---

## 3. Persona Modes

Persona modes customize the AI's behavior for different use cases.

### Persona Modes

| Mode | Temperature | Validation | Best For |
|------|-------------|------------|----------|
| `STRICT` | 0.3 | Strict | Production code, APIs |
| `CREATIVE` | 0.9 | Flexible | Brainstorming, ideation |
| `BALANCED` | 0.7 | Default | General use |
| `CUSTOM` | Configurable | Custom | Specialized tasks |

### Basic Usage

```python
from orchestrator.persona import PersonaManager, PersonaMode

manager = PersonaManager()

# Set persona for project
manager.set_persona("project_001", PersonaMode.STRICT)

# Get settings
settings = manager.get_persona_settings("project_001")
print(f"Temperature: {settings.temperature}")
print(f"Strict validation: {settings.strict_validation}")
print(f"Require tests: {settings.require_tests}")

# Change persona
manager.set_persona("project_001", PersonaMode.CREATIVE)

# Remove persona
manager.remove_persona("project_001")
```

### Persona Settings

```python
from orchestrator.persona import PersonaSettings

# Default settings for each mode
strict_settings = PersonaSettings(
    temperature=0.3,
    top_p=0.5,
    strict_validation=True,
    require_tests=True,
    require_documentation=True,
    max_iterations=3,
    enable_preflight=True,
    preflight_mode="block",
)

creative_settings = PersonaSettings(
    temperature=0.9,
    top_p=1.0,
    strict_validation=False,
    require_tests=False,
    require_documentation=False,
    max_iterations=10,
    enable_preflight=False,
)

balanced_settings = PersonaSettings(
    temperature=0.7,
    top_p=0.9,
    strict_validation=True,
    require_tests=False,
    require_documentation=True,
)
```

### Custom Persona

```python
from orchestrator.persona import PersonaManager, PersonaSettings

manager = PersonaManager()

# Create custom persona
custom_settings = PersonaSettings(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    strict_validation=True,
    require_tests=True,
    require_documentation=False,
    max_iterations=5,
    include_reasoning=True,
    verbose_output=True,
    format_code=True,
    enable_preflight=True,
    preflight_mode="warn",
    max_tokens=8192,
    system_prompt_addition="You are an expert Python developer specializing in FastAPI.",
)

manager.set_custom_persona(
    project_id="project_001",
    settings=custom_settings,
    name="Python Expert",
)
```

### Integration with Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.persona import PersonaManager, PersonaMode

orch = Orchestrator()
manager = PersonaManager()

async def run_project_with_persona():
    # Set persona before running
    manager.set_persona("project_001", PersonaMode.STRICT)
    
    # Apply persona settings to orchestrator
    settings = manager.get_persona_settings("project_001")
    orch.configure(
        temperature=settings.temperature,
        max_iterations=settings.max_iterations,
        strict_validation=settings.strict_validation,
    )
    
    # Run project
    state = await orch.run_project(
        project_description="Production authentication service",
        success_criteria="All tests pass, security audit clean",
    )
    
    return state
```

---

## 4. Token Optimizer

Token Optimizer applies domain-specific compression strategies to reduce token usage for common command outputs.

### Supported Commands

| Command | Compression Strategy | Savings |
|---------|---------------------|---------|
| `git log` | Summarize commits, remove metadata | 60-80% |
| `pytest` | Extract failures only | 70-90% |
| `eslint` | Extract errors only | 60-80% |
| `docker ps` | Table format, remove columns | 50-70% |
| `npm ls` | Tree simplification | 60-80% |
| `ps aux` | Top processes only | 70-90% |
| `df -h` | Summary only | 50-60% |
| `free -m` | Key metrics only | 60-70% |
| `top` | Top 10 processes | 80-90% |
| `netstat` | Active connections only | 70-80% |

### Basic Usage

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Compress git log
git_log = """
commit abc123
Author: John Doe
Date:   Mon Mar 25 10:00:00 2026

    Add authentication feature
    
commit def456
Author: Jane Smith
Date:   Sun Mar 24 15:30:00 2026

    Fix bug in login flow
"""

compressed = optimizer.compress_command_output(
    command="git log",
    output=git_log,
    target_ratio=0.3,  # Target 30% of original size
)

print(f"Original: {len(git_log)} chars")
print(f"Compressed: {len(compressed)} chars")
print(f"Saved: {(1 - len(compressed)/len(git_log))*100:.1f}%")
```

### Command-Specific Examples

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

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

# Compress to failures only
compressed = optimizer.compress_command_output(
    command="pytest",
    output=pytest_output,
    target_ratio=0.2,
)

print(compressed)
# Output:
# FAILED: test_auth.py::test_register - assert 500 == 201
# FAILED: test_auth.py::test_password_reset - assert False == True
# 2 failed, 48 passed
```

### Custom Compression Strategy

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()

# Register custom strategy
@optimizer.register_strategy("custom_command")
def compress_custom_command(output: str, target_ratio: float) -> str:
    # Custom compression logic
    lines = output.split('\n')
    
    # Keep only lines with keywords
    keywords = ["ERROR", "WARNING", "FAILED"]
    filtered = [line for line in lines if any(k in line for k in keywords)]
    
    # Limit to target ratio
    max_lines = int(len(lines) * target_ratio)
    return '\n'.join(filtered[:max_lines])

# Use custom strategy
compressed = optimizer.compress_command_output(
    command="custom_command",
    output=large_output,
    target_ratio=0.3,
)
```

---

## 5. Integration Examples

### Example 1: Full Quality Pipeline

```python
from orchestrator import Orchestrator
from orchestrator.preflight import PreflightValidator, PreflightMode
from orchestrator.persona import PersonaManager, PersonaMode as PersonaModeType
from orchestrator.session_watcher import SessionWatcher

orch = Orchestrator()
validator = PreflightValidator()
persona_manager = PersonaManager()
watcher = SessionWatcher()

async def run_with_full_quality_control():
    project_id = "proj_001"
    
    # Set strict persona for production code
    persona_manager.set_persona(project_id, PersonaModeType.STRICT)
    
    # Start session tracking
    session_id = watcher.start_session(
        project_id=project_id,
        metadata={"type": "production"}
    )
    
    # Run project
    state = await orch.run_project(
        project_description="Production authentication API",
        success_criteria="All tests pass, security clean",
    )
    
    # Preflight validation
    preflight_result = validator.validate(
        response=state.generated_code,
        context={"task": "production_code"},
        mode=PreflightMode.BLOCK,
    )
    
    if preflight_result.has_blocking_issues:
        print(f"Blocked: {preflight_result.reason}")
        # Request revision
    
    # Record interaction
    watcher.record_interaction(
        session_id=session_id,
        task_input=state.project_description,
        task_output=state.generated_code,
        task_type="code_generation",
        metadata={
            "model": "gpt-4o",
            "tokens_used": state.tokens_used,
            "quality_score": state.overall_quality_score,
        },
    )
    
    return state
```

### Example 2: Creative Brainstorming Session

```python
from orchestrator.persona import PersonaManager, PersonaMode
from orchestrator.session_watcher import SessionWatcher

persona_manager = PersonaManager()
watcher = SessionWatcher()

async def brainstorm_features():
    session_id = watcher.start_session(
        project_id="feature_ideas",
        metadata={"type": "brainstorming"}
    )
    
    # Set creative persona
    persona_manager.set_persona("feature_ideas", PersonaMode.CREATIVE)
    
    ideas = []
    for i in range(5):
        # Generate ideas with creative persona
        idea = await generate_with_persona(
            prompt="New feature idea for project management app",
            persona=PersonaMode.CREATIVE,
        )
        ideas.append(idea)
        
        # Record each idea
        watcher.record_interaction(
            session_id=session_id,
            task_input=f"Idea {i+1}",
            task_output=idea,
            task_type="brainstorming",
        )
    
    # Get all ideas from session
    context = await watcher.get_context(session_id)
    return [r.task_output for r in context]
```

### Example 3: Production Code Review

```python
from orchestrator.preflight import PreflightValidator, PreflightMode, CheckType
from orchestrator.token_optimizer import TokenOptimizer

validator = PreflightValidator()
optimizer = TokenOptimizer()

async def review_code(code: str, test_output: str):
    # Compress test output
    compressed_tests = optimizer.compress_command_output(
        command="pytest",
        output=test_output,
        target_ratio=0.2,
    )
    
    # Validate with preflight
    result = validator.validate(
        response=code,
        context={
            "task": "code_review",
            "test_output": compressed_tests,
        },
        mode=PreflightMode.BLOCK,
    )
    
    # Check specific issues
    for check in result.checks:
        if not check.passed:
            print(f"❌ {check.check_type.value}: {check.message}")
    
    if result.passed:
        print("✅ Code passed all checks")
    else:
        print(f"⚠️ Code has issues: {result.summary}")
    
    return result
```

---

## Configuration

### Environment Variables

```bash
# Preflight
export PREFLIGHT_ENABLED=true
export PREFLIGHT_DEFAULT_MODE=warn  # pass, enrich, warn, block

# Session Watcher
export SESSION_WATCHER_ENABLED=true
export MEMORY_BASE_PATH=./sessions
export HOT_TIER_DAYS=3
export WARM_TIER_DAYS=30

# Persona
export DEFAULT_PERSONA=balanced  # strict, creative, balanced, custom

# Token Optimizer
export TOKEN_OPTIMIZER_ENABLED=true
export DEFAULT_COMPRESSION_RATIO=0.5
```

### Python Configuration

```python
from orchestrator.preflight import configure_preflight
from orchestrator.session_watcher import configure_session_watcher
from orchestrator.persona import configure_persona

# Configure all
configure_preflight(
    enabled=True,
    default_mode="warn",
)

configure_session_watcher(
    enabled=True,
    memory_base_path="./sessions",
    auto_archive_days=30,
)

configure_persona(
    default_mode="balanced",
    strict_validation=True,
)
```

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities
- [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) — Web search integration
- [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md) — External agent integration

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
