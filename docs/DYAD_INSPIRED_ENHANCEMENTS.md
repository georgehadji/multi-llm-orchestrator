# Enhancement Plan: Dyad-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Dyad.sh and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements Replit, Lovable, Newly, Base44, v0, Retool, and UI enhancement plans

---

## Overview

Dyad.sh is a **free, local, open-source AI app builder** — an alternative to v0, Lovable, and Bolt.new that runs entirely on the user's computer with no vendor lock-in. Its key differentiator is the **multiple-chat-per-project** model with shared version history, plus an **AI-powered security review** with structured severity levels and one-click fixes.

Six enhancements identified, focused on **context management**, **security automation**, and **execution isolation**.

```
Phase D1: Multiple Concurrent Execution Contexts per Project     (Highest ROI, 3-4 days)
Phase D2: AI Security Review with Structured Findings              (High ROI, 2-3 days)
Phase D3: Summarize into New Context (Context Compression UX)      (High ROI, 1-2 days)
Phase D4: Selective Undo + Model Switching for Retry              (Medium ROI, 2-3 days)
Phase D5: Copy Project for Safe Experimentation                    (Medium ROI, 1-2 days)
Phase D6: System Diagnostics Drawer (Build/Install/Error Status)  (Lower ROI, 2-3 days)
```

---

## What Dyad Has — Quick Reference

| Dyad Feature | Description | Translates? |
|-------------|-------------|:-----------:|
| **Multiple chats per project** | Separate AI contexts, shared codebase + versions | ✅ Phase D1 |
| **Security Review** | AI audit with severity levels + one-click fix | ✅ Phase D2 |
| **Summarize into new chat** | AI compresses context into new conversation | ✅ Phase D3 |
| **Undo + Retry with different model** | Revert bad change, switch model, retry | ✅ Phase D4 |
| **Copy app** | Duplicate project for safe experimentation | ✅ Phase D5 |
| **System Messages drawer** | Diagnostics panel for builds, installs, errors | ✅ Phase D6 |
| **Select UI to edit** | Click elements in preview to edit | Partial — Phase U4 (Element Picker) |
| **Undo-able database** | Database operations that can be undone | ✗ (Platform-specific) |
| **Stack templates** | User-created tech stack templates | Partial — Phase V7 (Templates) |
| **Mobile app via Capacitor** | Web app → hybrid mobile | ✗ (Platform-specific) |
| **Docker container support** | Run apps in Docker | Partial — Docker in scaffolds |
| **Import existing projects** | Bring external codebases in | ✗ (Different architecture model) |
| **Local/open-source** | Runs on user's computer, no lock-in | The orchestrator is already local/open-source |

---

## Phase D1: Multiple Concurrent Execution Contexts per Project

### Objective

Allow multiple independent execution contexts (chats/sessions) within the same project, each with its own AI context window but sharing the same codebase and version history. This is Dyad's most unique feature — one app, many chats.

### Current State

- Each `Orchestrator` instance manages one project with one execution path
- `Orchestrator._resume_project()` resumes from last state — no parallel sessions
- No concept of multiple concurrent contexts per project
- Phase 4 (Sandbox Tasks) isolates individual task execution but doesn't create separate contexts

### Implementation

#### D1.1 — Create `orchestrator/execution_context.py`

```python
"""
Multiple Execution Contexts per Project — isolated AI contexts, shared state.
==============================================================================

Each project can have multiple execution contexts (similar to Dyad's chats).
Each context has:
- Independent AI conversation history
- Independent context window pressure
- Shared codebase (writes propagate across contexts)
- Shared version history (versions are global across contexts)
- Context isolation (context A can't see context B's conversation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .bot import UnifiedClient


@dataclass
class ExecutionContext:
    """An isolated execution context within a project.

    Multiple contexts can be active simultaneously, each with its own
    AI conversation but sharing the same codebase and version history.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    created_at: float = field(default_factory=__import__('time').time)

    # AI conversation history (independent per context)
    conversation: list[dict] = field(default_factory=list)

    # Current task queue (what this context is working on)
    task_queue: list[str] = field(default_factory=list)

    # Context window pressure (0.0 = empty, 1.0 = full)
    context_pressure: float = 0.0

    # Metadata
    model_used: str = ""
    is_active: bool = False

    def estimate_context_size(self) -> int:
        """Estimate current context size in tokens."""
        return sum(len(str(msg.get("content", ""))) for msg in self.conversation) // 4

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "message_count": len(self.conversation),
            "context_pressure": self.context_pressure,
            "model_used": self.model_used,
            "is_active": self.is_active,
        }


class ContextManager:
    """Manages multiple execution contexts for a project.

    Features:
    - Create/destroy contexts without affecting the codebase
    - Switch between contexts (each has its own conversation)
    - Summarize context into a new context (context compression)
    - Isolate model selection per context (different context, different model)
    - Share version history across all contexts
    - Detect context pressure and suggest summarization
    """

    def __init__(
        self,
        project_id: str,
        codebase_path: Path,
        version_manager: VersionManager,
        client: UnifiedClient,
    ):
        self._project_id = project_id
        self._codebase_path = codebase_path
        self._versions = version_manager  # Shared across all contexts
        self._client = client

        self._contexts: dict[str, ExecutionContext] = {}
        self._active_context_id: str | None = None

        # Create default context
        default = ExecutionContext(name="main")
        self._contexts[default.id] = default
        self._active_context_id = default.id

    def create_context(self, name: str = "") -> ExecutionContext:
        """Create a new execution context.

        The new context starts with a clean conversation but shares
        the same codebase and version history.
        """
        ctx = ExecutionContext(name=name or f"context-{len(self._contexts) + 1}")
        self._contexts[ctx.id] = ctx
        return ctx

    def switch_context(self, context_id: str) -> ExecutionContext | None:
        """Switch the active execution context."""
        if context_id not in self._contexts:
            return None

        # Deactivate current
        if self._active_context_id:
            self._contexts[self._active_context_id].is_active = False

        # Activate new
        self._active_context_id = context_id
        self._contexts[context_id].is_active = True
        return self._contexts[context_id]

    def get_active_context(self) -> ExecutionContext | None:
        """Get the currently active context."""
        if self._active_context_id:
            return self._contexts.get(self._active_context_id)
        return None

    def list_contexts(self) -> list[dict]:
        """List all contexts with status."""
        return [ctx.to_dict() for ctx in self._contexts.values()]

    def delete_context(self, context_id: str) -> bool:
        """Delete a context (but not its generated code).

        Cannot delete the last context. Cannot delete if it's active.
        Switches to another context first if needed.
        """
        if len(self._contexts) <= 1:
            return False  # Must have at least one context

        if context_id == self._active_context_id:
            # Switch to another context first
            other = next(cid for cid in self._contexts if cid != context_id)
            self.switch_context(other)

        del self._contexts[context_id]
        return True

    async def summarize_into_new_context(
        self, source_context_id: str, new_name: str = ""
    ) -> ExecutionContext | None:
        """Summarize a context's conversation into a new clean context.

        Uses the LLM to produce a concise summary of what was discussed
        and accomplished, preserving context without the full history.

        Args:
            source_context_id: Context to summarize
            new_name: Name for the new context

        Returns:
            New context with summary injected
        """
        source = self._contexts.get(source_context_id)
        if not source:
            return None

        # Build conversation summary prompt
        summary = await self._summarize_conversation(source.conversation)

        # Create new context with summary as starting context
        new_ctx = ExecutionContext(name=new_name)
        new_ctx.conversation.append({
            "role": "system",
            "content": (
                f"This is a continuation of a previous session. "
                f"Summary of prior work:\n\n{summary}"
            ),
        })

        self._contexts[new_ctx.id] = new_ctx
        return new_ctx

    async def _summarize_conversation(
        self, conversation: list[dict]
    ) -> str:
        """Use the LLM to summarize a conversation.

        Focuses on key decisions, generated files, outstanding issues.
        """
        # Extract key parts of the conversation
        user_messages = [m for m in conversation if m.get("role") == "user"]
        ai_messages = [m for m in conversation if m.get("role") == "assistant"]

        prompt = (
            "Summarize this conversation in 3-5 bullet points. Include:\n"
            "1. What the user asked for\n"
            "2. What was generated/built\n"
            "3. Key decisions or architecture choices made\n"
            "4. Remaining issues or next steps\n"
            "5. Important context the next session needs\n\n"
            f"User messages ({len(user_messages)}):\n"
            f"{chr(10).join(m['content'][:200] for m in user_messages[-5:])}\n\n"
            f"Last AI response:\n"
            f"{ai_messages[-1]['content'][:500] if ai_messages else 'N/A'}"
        )

        response = await self._client.call(
            model=self._get_cheapest_model(),
            prompt=prompt,
            max_tokens=300,
            temperature=0.1,
        )

        return response.text

    def check_context_pressure(self) -> dict[str, float]:
        """Check context window pressure for all contexts.

        Returns dict of context_id → pressure (0.0-1.0).
        Suggests summarization when pressure > 0.8.
        """
        pressures = {}
        for ctx_id, ctx in self._contexts.items():
            estimated = ctx.estimate_context_size()
            # Assuming ~128k token context window
            ctx.context_pressure = min(estimated / 100_000, 1.0)
            pressures[ctx_id] = ctx.context_pressure
        return pressures
```

**UI integration:**
```
┌──────────────────────────────────────────────────────────────┐
│  📁 Project: Inventory App                                   │
│                                                              │
│  💬 Contexts:                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ● main (active)              3 messages  35% full    │   │
│  │ ○ auth-debugging            12 messages  72% full    │   │
│  │ ○ ui-redesign                7 messages  48% full    │   │
│  │ [+ New Context]                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ⚠️ auth-debugging is at 72% — [Summarize into new context] │
│                                                              │
│  Shared: 14 versions, 23 files, 1 module                    │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/execution_context.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — integrate `ContextManager`, support `--context` flag
- MODIFY: `orchestrator/cli.py` — add `context list|create|switch|delete|summarize` commands
- MODIFY: `orchestrator/state.py` — save/load contexts per project
- DEPENDS ON: Phase V6 (Versions — shared version history)

#### Verification Gate

```bash
# Create 3 contexts, switch between them
# Make a change in context A, verify it's visible in context B (shared codebase)
# Verify conversations are isolated (context A can't see B's chat)
# Summarize context A, verify new context has summary, not full history
```

---

## Phase D2: AI Security Review with Structured Findings

### Objective

Add an AI-powered security audit that analyzes generated code for vulnerabilities, produces structured findings with severity levels (Critical/High/Medium/Low), and offers one-click fixes for each issue. Results are cached in a `SECURITY_RULES.md` file for future use.

### Current State

- `orchestrator/validators.py` — `validate_tool_safety()` checks for unsafe patterns (eval, os.system, subprocess)
- `security/enhancer.py` — 22 OWASP rules injected as system prompts
- `ReviewerAgent` — reviews code for bugs, security, performance
- No structured security audit with severity levels
- No one-click fix from security findings
- No `SECURITY_RULES.md` for AI security knowledge

### Implementation

#### D2.1 — Create `orchestrator/security_review.py`

```python
"""
Security Review — AI-powered vulnerability audit with severity levels.
=======================================================================

Analyzes generated code for security vulnerabilities and produces
structured findings with severity levels and one-click fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bot import UnifiedClient


class Severity(str, Enum):
    CRITICAL = "critical"  # Must fix — potential data breach
    HIGH = "high"          # Should fix — significant risk
    MEDIUM = "medium"      # Consider fixing — moderate risk
    LOW = "low"            # Optional — best practice improvement
    INFO = "info"          # Informational — no immediate risk


@dataclass
class SecurityFinding:
    """A single security issue found during review."""

    id: str
    title: str
    description: str
    severity: Severity
    category: str  # "auth", "data", "injection", "crypto", "config", "dependency"
    file_path: str
    line_numbers: list[int]
    code_snippet: str
    remediation: str
    cwe_id: str | None = None  # Common Weakness Enumeration ID
    fixed: bool = False


@dataclass
class SecurityReview:
    """Complete security review result."""

    project_id: str
    timestamp: float
    findings: list[SecurityFinding]
    model_used: str
    cost: float
    total_files_scanned: int
    total_lines_scanned: int

    # Skip list — findings flagged as false positives
    skip_list: list[str] = field(default_factory=list)

    # Security rules learned from this review (for future runs)
    learned_rules: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def fixed_count(self) -> int:
        return sum(1 for f in self.findings if f.fixed)

    def to_summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Security review: {len(self.findings)} findings "
            f"({self.critical_count} critical, {self.high_count} high, "
            f"{self.fixed_count} fixed)"
        )


class SecurityReviewer:
    """AI-powered security audit for generated projects.

    Checks for:
    - OWASP Top 10 vulnerabilities
    - Authentication/authorization issues
    - Data exposure risks
    - Injection vulnerabilities (SQL, XSS, command)
    - Cryptographic weaknesses
    - Configuration security issues
    - Dependency vulnerabilities
    - Hardcoded secrets and credentials
    """

    # Categories and what to check
    SECURITY_CHECKS = {
        "auth": [
            "Missing authentication on protected routes",
            "Weak password policies",
            "Insecure session management",
            "Missing rate limiting on auth endpoints",
            "JWT token misconfigurations",
        ],
        "data": [
            "Exposed sensitive data in responses",
            "Missing input validation",
            "Missing output encoding",
            "Insecure direct object references (IDOR)",
            "Mass assignment vulnerabilities",
        ],
        "injection": [
            "SQL injection via string concatenation",
            "NoSQL injection",
            "Cross-site scripting (XSS)",
            "Command injection",
            "Server-side request forgery (SSRF)",
        ],
        "crypto": [
            "Weak hashing algorithms (MD5, SHA1)",
            "Insecure random number generation",
            "Hardcoded encryption keys",
            "Missing encryption for sensitive data at rest",
            "Insecure TLS configuration",
        ],
        "config": [
            "Debug mode enabled in production",
            "Exposed environment variables",
            "Missing security headers (CSP, HSTS, X-Frame-Options)",
            "CORS misconfigurations",
            "Default credentials or API keys",
        ],
        "dependency": [
            "Known vulnerable package versions",
            "Unpinned dependencies",
            "Unmaintained or deprecated packages",
        ],
    }

    def __init__(self, client: UnifiedClient):
        self._client = client
        self._security_rules: str = ""

    async def review(
        self,
        codebase_path: str,
        files: list[str] | None = None,
        skip_categories: list[str] | None = None,
    ) -> SecurityReview:
        """Run a full security audit on the project.

        Args:
            codebase_path: Root directory of the project
            files: Optional list of files to scan (all if None)
            skip_categories: Categories to skip

        Returns:
            SecurityReview with structured findings
        """

    async def review_file(
        self, file_path: str, content: str
    ) -> list[SecurityFinding]:
        """Audit a single file for security issues."""

    async def fix_finding(
        self, finding: SecurityFinding, file_content: str
    ) -> tuple[str, bool]:
        """Generate a fix for a specific security finding.

        Returns: (fixed_code, success)
        """

    async def generate_security_rules(
        self, review: SecurityReview
    ) -> str:
        """Generate SECURITY_RULES.md from review findings.

        These rules help the AI avoid introducing the same issues
        in future generations.
        """

    def load_security_rules(self, rules_path: str) -> None:
        """Load existing SECURITY_RULES.md for context-aware review."""
        try:
            with open(rules_path) as f:
                self._security_rules = f.read()
        except FileNotFoundError:
            self._security_rules = ""

    async def _categorize_finding(
        self, finding: SecurityFinding
    ) -> str:
        """Assign CWE ID and severity to a finding."""
```

**Example SECURITY_RULES.md:**
```markdown
# Security Rules — AI Knowledge Base

These rules were generated from security reviews. The AI should follow
them when generating code for this project.

## Authentication
- Always use bcrypt (cost 12+) for password hashing
- JWT tokens must have expiration (max 24h)
- All API endpoints behind auth middleware

## Data Protection
- Never return user passwords in API responses
- Always validate and sanitize user input
- Use parameterized queries — never string concatenation

## Configuration
- Debug mode must be disabled in production
- All environment variables stored in .env (never committed)
- Security headers: CSP, HSTS, X-Frame-Options, X-Content-Type-Options

## Known False Positives
- File: src/utils/debug.py, Lines 15-20 — debug function is safe in dev only
```

**UI display:**
```
┌──────────────────────────────────────────────────────────────┐
│  🛡️ Security Review — 8 findings                              │
│                                                              │
│  🔴 Critical (1)                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SQL Injection in src/api/orders.py:42                │   │
│  │ Query built with string concatenation                │   │
│  │ CWE-89 | [Fix Issue]                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  🟠 High (2)                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Debug mode enabled in src/config.py:8               │   │
│  │ DEBUG=True in production config                      │   │
│  │ [Fix Issue]                                          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Missing rate limiting in src/api/auth.py:15         │   │
│  │ No rate limiter on login endpoint                    │   │
│  │ [Fix Issue]                                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  🟡 Medium (3)  🔵 Low (1)  ℹ️ Info (1)                     │
│                                                              │
│  [Run Security Review]  [Edit Security Rules]               │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/security_review.py` (~500 lines)
- MODIFY: `orchestrator/engine.py` — add `security_review()` method, auto-run after project completion
- MODIFY: `orchestrator/cli.py` — add `security-review` command
- MODIFY: `orchestrator/ide_backend/websocket/handlers.py` — handle `security:fix` event
- DEPENDS ON: Phase N1 (Ask Mode — for cheaper security review queries)

#### Verification Gate

```bash
# Run security review on a generated project
# Verify: findings sorted by severity, CWE IDs assigned, fixes available
# Click "Fix Issue" on a finding, verify code is corrected
# Check SECURITY_RULES.md was generated/updated
```

---

## Phase D3: Summarize into New Context (Context Compression UX)

### Objective

When an execution context's conversation grows too large (context pressure > 0.8), offer a one-click "Summarize into new context" action. The LLM produces a concise summary of the conversation, and a new clean context is created with the summary injected as starting context.

### Current State

- `application/context_compressor.py` — LLM-powered summarization of dependency context
- Compressor is used for task dependencies, not for conversation context
- Phase U6 (Generation Progress) shows context pressure but doesn't offer summarization
- No UX for context compression

### Implementation

#### D3.1 — Integrate context summarization into ContextManager

```python
# Already implemented in Phase D1's ContextManager.summarize_into_new_context()

# UX integration:
class ContextPressureMonitor:
    """Monitors context window pressure and suggests summarization."""

    def __init__(self, context_manager: ContextManager, threshold: float = 0.8):
        self._ctx_mgr = context_manager
        self._threshold = threshold

    def check_and_suggest(self) -> str | None:
        """Check all contexts' pressure. Return suggestion message if any
        context exceeds the threshold.
        """
        pressures = self._ctx_mgr.check_context_pressure()
        for ctx_id, pressure in pressures.items():
            if pressure > self._threshold:
                ctx = self._ctx_mgr._contexts[ctx_id]
                return (
                    f"Context '{ctx.name}' is at {pressure*100:.0f}% capacity. "
                    f"Consider summarizing into a new context to free up space."
                )
        return None
```

**UI suggestion:**
```
┌──────────────────────────────────────────────────────────────┐
│  ⚠️ Context "auth-debugging" is at 85% capacity             │
│                                                              │
│  [Summarize into new context]  [Dismiss]                    │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/context_pressure.py` (~150 lines)
- MODIFY: `orchestrator/execution_context.py` — integrate pressure monitoring
- MODIFY: `orchestrator/engine.py` — auto-check pressure after each task
- DEPENDS ON: Phase D1 (Multiple Contexts — ContextManager)

#### Verification Gate

```bash
# Fill a context with 50+ messages, verify pressure warning appears
# Click "Summarize", verify new context created with summary
# Verify new context has low pressure and conversation is summarized
```

---

## Phase D4: Selective Undo + Model Switching for Retry

### Objective

After the AI makes a change, allow the user to undo that specific change, switch to a different (usually more capable) model, and retry the same prompt. This is Dyad's "Undo → Retry with different model" workflow.

### Current State

- Phase 1 (Checkpoints) — checkpoint-based restore
- Phase V6 (Versions) — version-based revert
- `CritiqueCycle` has iteration retries but with the same model
- `FallbackHandler` switches models on failure, not on user request
- No combined undo + model switch + retry workflow

### Implementation

#### D4.1 — Add `retry_with_model()` to VersionManager

```python
class VersionManager:
    async def retry_with_model(
        self,
        version_id: str,  # Version to revert to
        new_model: Model,  # Model to use for retry
        prompt: str,       # Original prompt to retry
    ) -> Version:
        """Revert to a version, switch model, and retry.

        Workflow:
        1. Revert code to the specified version
        2. Switch the active model to new_model
        3. Re-execute the prompt with the new model
        4. Create a new version with the result
        5. Report success/failure

        This is the "Undo → Retry with different model" workflow.
        """
        # 1. Revert
        await self.revert(version_id)

        # 2. Switch model
        previous_model = self._active_model
        self._active_model = new_model

        # 3. Retry
        try:
            result = await self._execute_prompt(prompt, model=new_model)

            # 4. Create version
            version = await self.create_version(
                description=f"Retry with {new_model.value}: {prompt[:80]}",
                metadata={
                    "retry_from_version": version_id,
                    "previous_model": previous_model.value,
                    "new_model": new_model.value,
                },
            )

            return version

        except Exception as e:
            # Restore previous model on failure
            self._active_model = previous_model
            raise

    def get_alternative_models(
        self, current_model: Model, task_type: TaskType
    ) -> list[Model]:
        """Get list of alternative (often more capable) models for retry.

        Prioritizes:
        1. More capable models from different providers
        2. Models with larger context windows
        3. Reasoning-specialist models
        """
```

**UI workflow:**
```
┌──────────────────────────────────────────────────────────────┐
│  ❌ Last change: "Add JWT authentication"                    │
│     Model: Qwen 2.5 Coder  |  Score: 0.45                   │
│                                                              │
│  [Undo & Retry with...]                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ○ Claude Sonnet 4.6 (stronger reasoning, $3.00/M)   │   │
│  │ ○ GPT-4.1 (broader knowledge, $1.50/M)              │   │
│  │ ○ DeepSeek V4 Pro (coding specialist, $1.50/M)      │   │
│  │ ○ Gemini 2.5 Pro (big context, $1.25/M)             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- MODIFY: `orchestrator/version_manager.py` — add `retry_with_model()`, `get_alternative_models()`
- MODIFY: `orchestrator/engine.py` — expose retry workflow
- MODIFY: `orchestrator/fallback_handler.py` — add `get_upgrade_models()` for model tier upgrades
- MODIFY: `ide_frontend/src/components/VersionTimeline.tsx` — add "Retry with..." button
- DEPENDS ON: Phase V6 (Versions — version infrastructure)
- DEPENDS ON: Phase D1 (Multiple Contexts — context isolation)

#### Verification Gate

```bash
# Make a change that produces low-quality output (score < 0.6)
# Click "Undo & Retry with Claude Sonnet 4.6"
# Verify: code reverted, model switched, retry executes, new version created
# Verify: higher score with better model
```

---

## Phase D5: Copy Project for Safe Experimentation

### Objective

Allow duplicating an entire project (codebase, state, history, contexts) into a new isolated project for safe experimentation. The copy can be merged back or discarded.

### Current State

- No project duplication feature
- `Orchestrator._resume_project()` resumes the same project
- `StateManager` stores one state per project
- Phase 4 (Sandbox Tasks) isolates individual tasks, not entire projects

### Implementation

#### D5.1 — Add `copy_project()` to Orchestrator

```python
class Orchestrator:
    async def copy_project(
        self,
        new_project_id: str | None = None,
        copy_state: bool = True,
        copy_versions: bool = True,
        copy_contexts: bool = False,  # Don't copy conversations by default
    ) -> str:
        """Duplicate the current project into a new isolated project.

        Args:
            new_project_id: Optional ID for the new project (auto-generated if None)
            copy_state: Copy task state and budget
            copy_versions: Copy version history
            copy_contexts: Copy execution contexts (default: False — clean slate)

        Returns:
            New project ID

        The copied project:
        - Has the same codebase (file-level copy)
        - Has its own state (independent budget tracking)
        - Has its own version history (if copy_versions=True)
        - Starts with a single clean context (if copy_contexts=False)
        - Can be modified without affecting the original
        - Can be merged back via Compare & Merge
        """
        new_id = new_project_id or f"{self._project_id}-copy-{uuid4().hex[:8]}"

        # Copy codebase
        shutil.copytree(self._output_dir, f"{self._output_dir}-{new_id}")

        # Copy state
        if copy_state:
            await self._state_mgr.copy_state(self._project_id, new_id)

        # Copy versions
        if copy_versions:
            await self._version_mgr.copy_versions(new_id)

        # Create fresh context
        new_ctx = ExecutionContext(name=f"Copy of {self._project_id}")

        logger.info(f"Copied project {self._project_id} → {new_id}")
        return new_id

    async def compare_and_merge(
        self,
        source_project_id: str,
        files: list[str] | None = None,
    ) -> dict:
        """Compare two projects and selectively merge changes.

        Shows a diff between the original and copy, allowing the user
        to pick which changes to bring back.
        """
```

**CLI commands:**
```bash
# Copy a project for experimentation
python -m orchestrator project copy --new-id my-app-experiment

# Compare original with copy
python -m orchestrator project diff my-app --with my-app-experiment

# Merge changes from experiment back to original
python -m orchestrator project merge my-app-experiment --files src/auth.py,src/api.py
```

**File changes:**
- MODIFY: `orchestrator/engine.py` — add `copy_project()`, `compare_and_merge()`
- MODIFY: `orchestrator/state.py` — add `copy_state()` method
- MODIFY: `orchestrator/version_manager.py` — add `copy_versions()` method
- MODIFY: `orchestrator/cli.py` — add `project copy|diff|merge` commands

#### Verification Gate

```bash
# Create a project, copy it, modify the copy
# Verify: original unchanged, copy has new changes
# Compare projects, merge specific files back
```

---

## Phase D6: System Diagnostics Drawer (Build/Install/Error Status)

### Objective

A diagnostics panel that shows what the system is doing — npm installs, builds, test runs, errors — in real time. Similar to Dyad's "System Messages" drawer at the bottom of the preview panel.

### Current State

- `orchestrator/app_verifier.py` — startup checks (npm build, Docker build)
- Console output is streamed via WebSockets (Phase U2 — Console Panel)
- Mission Control has an Event Log but it's event-based, not diagnostic
- No unified system diagnostics panel

### Implementation

#### D6.1 — Create `orchestrator/diagnostics_panel.py`

```python
"""
System Diagnostics Panel — real-time build/install/error status.
==================================================================

Centralized diagnostics for all system operations:
- Package installations (npm install, pip install)
- Build processes (next build, vite build, cargo build)
- Test runs (pytest, jest, vitest)
- Server starts (uvicorn, next dev, vite dev)
- Lint/format operations (ruff, eslint, prettier)
- Deployment steps (Docker build, Vercel deploy)
"""

@dataclass
class DiagnosticEvent:
    id: str
    operation: str  # "npm_install", "build", "test", "server_start", "lint", "deploy"
    status: str  # "running", "success", "failed", "warning"
    started_at: float
    completed_at: float | None = None
    output_lines: list[str] = field(default_factory=list)
    error_lines: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0-1.0

class DiagnosticsPanel:
    """Collects and broadcasts system diagnostics in real time.

    Events are broadcast via WebSocket to the frontend for display
    in a collapsible drawer at the bottom of the IDE.
    """

    def __init__(self, websocket_manager: ConnectionManager):
        self._ws = websocket_manager
        self._events: list[DiagnosticEvent] = []
        self._active_operations: dict[str, DiagnosticEvent] = {}

    def start_operation(self, operation: str) -> DiagnosticEvent:
        """Record the start of a system operation."""
        event = DiagnosticEvent(
            id=str(uuid.uuid4())[:8],
            operation=operation,
            status="running",
            started_at=time.time(),
        )
        self._active_operations[event.id] = event
        self._events.append(event)
        self._broadcast(event)
        return event

    def update_progress(self, event_id: str, progress: float, line: str = ""):
        """Update operation progress."""
        ...

    def complete_operation(self, event_id: str, success: bool, output: list[str] = None):
        """Mark an operation as complete."""
        ...

    def _broadcast(self, event: DiagnosticEvent):
        """Broadcast diagnostic event to WebSocket clients."""
        ...
```

**UI display (collapsible drawer at bottom of IDE):**
```
┌──────────────────────────────────────────────────────────────┐
│  🔧 System Messages                                    [−]   │
├──────────────────────────────────────────────────────────────┤
│  ✅ npm install (2.3s)                   12:34:02           │
│  ⚠️ Build warning: unused variable      12:34:05           │
│     src/components/Header.tsx:15: 'unused' is declared      │
│  ✅ Vite build (4.1s)                    12:34:06           │
│  ✅ pytest (1.2s) — 23 passed            12:34:08           │
│  🔄 npm install (new dependency)         12:34:10           │
└──────────────────────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/diagnostics_panel.py` (~250 lines)
- MODIFY: `orchestrator/engine.py` — wrap system operations with diagnostics
- MODIFY: `ide_frontend/src/components/SystemDrawer.tsx` — UI component
- DEPENDS ON: Phase U2 (Console Panel — real-time streaming infrastructure)

#### Verification Gate

```bash
# Run a project, verify diagnostics drawer shows npm install, build, test status
# Induce a build error, verify it appears in the drawer
# Verify drawer is collapsible and auto-opens on error
```

---

## Integration with Existing Plans

```
Phase D1 (Multiple Contexts) ──────────────────────────────────────────────────┐
    │  No dependencies — new concept                                           │
    │  Depends on Phase V6 (Versions — shared version history)                 │
Phase D2 (Security Review) ────────────────────────────────────────────────────┤
    │  No dependencies — new feature                                            │
    │  Depends on Phase N1 (Ask Mode — cheaper security queries)                │
Phase D3 (Summarize Context) ───────────────────────────────────────────────────┤
    │  Depends on Phase D1 (Multiple Contexts — ContextManager)                 │
Phase D4 (Undo + Retry) ────────────────────────────────────────────────────────┤
    │  Depends on Phase V6 (Versions — version infrastructure)                 │
    │  Depends on Phase D1 (Multiple Contexts — context isolation)              │
Phase D5 (Copy Project) ────────────────────────────────────────────────────────┤
    │  No dependencies — new feature                                            │
Phase D6 (Diagnostics Drawer) ──────────────────────────────────────────────────┘
    Depends on Phase U2 (Console Panel — real-time streaming)
```

## Dyad-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| D1 | Multiple Contexts | 1 | 3 | ~400 | 3-4 |
| D2 | Security Review | 1 | 3 | ~500 | 2-3 |
| D3 | Summarize Context | 1 | 2 | ~150 | 1-2 |
| D4 | Undo + Retry | 0 | 3 | ~200 | 2-3 |
| D5 | Copy Project | 0 | 3 | ~150 | 1-2 |
| D6 | Diagnostics Drawer | 1 | 3 | ~250 | 2-3 |
| **Dyad Subtotal** | **6 phases** | **4** | **17** | **~1,650** | **11-17** |

## Combined Grand Total (All Eight Sources)

| # | Source | Phases | New Files | Modified Files | Est. Days |
|---|--------|--------|-----------|---------------|-----------|
| 1 | Replit | 1-6 | 5 | 14 | 13-18 |
| 2 | Lovable | 7-10 | 4 | 10 | 9-13 |
| 3 | UI | U1-U7 | 20 | 15 | 17-23 |
| 4 | Newly | N1-N7 | 2 | 21 | 10-14 |
| 5 | Base44 | B1-B7 | 14 | 14 | 14-21 |
| 6 | v0 | V1-V7 | 12 | 19 | 18-26 |
| 7 | Retool | R1-R7 | 10 | 16 | 16-23 |
| 8 | Dyad | D1-D6 | 4 | 17 | 11-17 |
| **Grand Total** | **51** | **71** | **126** | **108-155** |
