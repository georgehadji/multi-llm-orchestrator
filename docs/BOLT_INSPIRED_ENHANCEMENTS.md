# Enhancement Plan: Bolt.new-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Bolt.new and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements all 8 prior enhancement plans

---

## Overview

Bolt.new (by StackBlitz) is a web-based AI app builder with built-in hosting, database, and domain management. Its key differentiator is the **dual-agent system** (Standard vs Max) where users choose an agent profile rather than a specific model, plus **Code View** with file-level AI scope control (Target/Lock files).

Five enhancements identified, focused on **agent profiles**, **file-level AI scoping**, and **explicit publish workflows**.

```
Phase W1: Agent Profiles (Standard vs Max) — model abstraction     (Highest ROI, 2-3 days)
Phase W2: Code View with Target/Lock Files (AI Scope Control)        (High ROI, 3-4 days)
Phase W3: Plan Mode with Quick Action Buttons                         (Medium ROI, 1-2 days)
Phase W4: Project vs Site Separation (Explicit Publish)               (Medium ROI, 2-3 days)
Phase W5: Team Templates for Repetitive Project Setup                  (Lower ROI, 2-3 days)
```

---

## What Bolt.new Has — Quick Reference

| Bolt.new Feature | Description | Translates? |
|-----------------|-------------|:-----------:|
| **Standard vs Max Agents** | Two agent profiles that auto-select models | ✅ Phase W1 |
| **Target/Lock Files** | Scope AI to specific files, lock others | ✅ Phase W2 |
| **Quick Action Buttons** | Context-aware actions after Plan Mode | ✅ Phase W3 |
| **Project vs Site** | Separate dev workspace from published version | ✅ Phase W4 |
| **Team Templates** | Skip repetitive project setup | ✅ Phase W5 |
| **Code View** | In-browser file editor with syntax highlighting | Partial — Phase U2 (Code Editor) |
| **Plan/Discussion Mode** | Chat without generating code | Partial — Phase N1 (Ask Mode), Phase N4 (Brainstorming) |
| **Design System Sync** | Design system components stay synced | Partial — Phase V1 (Design Registry) |
| **Bolt Cloud** | Built-in database, hosting, domains, analytics | ✗ (Platform-specific) |
| **Multiplayer collaboration** | Real-time co-editing with token accounting | Partial — Phase U5 (Collaboration sidebar) |
| **MCP Integration** | Connect external tools via MCP | Partial — orchestrator has MCP server |
| **Import from other platforms** | Import from Lovable, etc. | ✗ (Different architecture model) |
| **SEO Boost** | SEO optimization for published projects | ✗ (Platform-specific) |

---

## Phase W1: Agent Profiles (Standard vs Max) — Model Abstraction

### Objective

Introduce **agent profiles** — named configurations that bundle model selection, iteration strategy, and behavior parameters. Users select a profile (Standard, Max, Creative, etc.) rather than picking individual models. The orchestrator selects the best available model within that profile.

### Current State

- `AgentBase` uses `model_preferences: dict[TaskType, Model]` — direct model references
- `agent_model_registry.py` maps AgentRole → budget/premium model tiers
- `CritiqueCycle` has configurable `max_iterations`
- Users must know Model enum values to configure agents
- No abstraction layer between "what I want" and "which model to use"

### Implementation

#### W1.1 — Create `orchestrator/agent_profiles.py`

```python
"""
Agent Profiles — named configurations that abstract model selection.
=====================================================================

Users select a profile instead of picking individual models.
Profiles bundle: model tier, reasoning depth, iteration strategy,
temperature ranges, and behavior parameters.

Standard profiles:
- Standard — Fast, token-efficient, for well-defined tasks
- Max — Deep reasoning, complex tasks, large codebases
- Creative — High temperature, experimental, brainstorming
- Conservative — Low temperature, strict validation, production safety
- Research — Web search enabled, broad knowledge synthesis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentProfile(str, Enum):
    STANDARD = "standard"          # Fast, token-efficient
    MAX = "max"                    # Deep reasoning, complex tasks
    CREATIVE = "creative"          # High temperature, experimental
    CONSERVATIVE = "conservative"  # Low temperature, production safe
    RESEARCH = "research"          # Web search, knowledge synthesis


@dataclass
class ProfileConfig:
    """Configuration for an agent profile."""

    profile: AgentProfile
    display_name: str
    description: str

    # Model selection strategy
    model_tier: str  # "budget" or "premium"
    prefer_reasoning_models: bool = False
    prefer_large_context: bool = False

    # Behavior parameters
    max_iterations: int = 3
    temperature_range: tuple[float, float] = (0.2, 0.5)
    enable_web_search: bool = False
    enable_self_review: bool = False
    strict_validation: bool = False

    # Best use cases
    best_for: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "display_name": self.display_name,
            "description": self.description,
            "model_tier": self.model_tier,
            "max_iterations": self.max_iterations,
            "best_for": self.best_for,
        }


# Pre-configured profiles
BUILTIN_PROFILES: dict[AgentProfile, ProfileConfig] = {
    AgentProfile.STANDARD: ProfileConfig(
        profile=AgentProfile.STANDARD,
        display_name="Standard",
        description="Fast and token-efficient for everyday development",
        model_tier="budget",
        prefer_reasoning_models=False,
        prefer_large_context=False,
        max_iterations=3,
        temperature_range=(0.2, 0.4),
        enable_web_search=False,
        enable_self_review=False,
        strict_validation=False,
        best_for=[
            "Small to medium projects",
            "UI updates and styling changes",
            "Well-defined, clear tasks",
            "Quick iterations",
        ],
    ),
    AgentProfile.MAX: ProfileConfig(
        profile=AgentProfile.MAX,
        display_name="Max",
        description="Maximum reasoning power for complex tasks",
        model_tier="premium",
        prefer_reasoning_models=True,
        prefer_large_context=True,
        max_iterations=5,
        temperature_range=(0.1, 0.3),
        enable_web_search=False,
        enable_self_review=True,
        strict_validation=True,
        best_for=[
            "Large-scale applications",
            "Complex, interconnected features",
            "Refactoring existing code",
            "Open-ended or ambiguous tasks",
        ],
    ),
    AgentProfile.CREATIVE: ProfileConfig(
        profile=AgentProfile.CREATIVE,
        display_name="Creative",
        description="High-temperature, experimental brainstorming",
        model_tier="premium",
        prefer_reasoning_models=False,
        prefer_large_context=False,
        max_iterations=2,
        temperature_range=(0.7, 0.9),
        enable_web_search=True,
        enable_self_review=False,
        strict_validation=False,
        best_for=[
            "Brainstorming new features",
            "Alternative architectural approaches",
            "UI/UX exploration",
            "Creative problem-solving",
        ],
    ),
    AgentProfile.CONSERVATIVE: ProfileConfig(
        profile=AgentProfile.CONSERVATIVE,
        display_name="Conservative",
        description="Low-temperature, strict validation for production safety",
        model_tier="premium",
        prefer_reasoning_models=False,
        prefer_large_context=False,
        max_iterations=3,
        temperature_range=(0.0, 0.2),
        enable_web_search=False,
        enable_self_review=True,
        strict_validation=True,
        best_for=[
            "Production deployments",
            "Security-critical code",
            "API contracts and schemas",
            "Database migrations",
        ],
    ),
    AgentProfile.RESEARCH: ProfileConfig(
        profile=AgentProfile.RESEARCH,
        display_name="Research",
        description="Web search enabled for broad knowledge synthesis",
        model_tier="premium",
        prefer_reasoning_models=True,
        prefer_large_context=True,
        max_iterations=2,
        temperature_range=(0.2, 0.5),
        enable_web_search=True,
        enable_self_review=False,
        strict_validation=False,
        best_for=[
            "API documentation research",
            "Library/framework evaluation",
            "Best practice research",
            "Technology comparison",
        ],
    ),
}


class ProfileRouter:
    """Routes agent profiles to actual models and strategies.

    For each profile, selects the best available model from the
    agent_model_registry based on the profile's parameters.
    """

    def __init__(self, model_registry, fallback_handler, client):
        self._registry = model_registry
        self._fallback = fallback_handler
        self._client = client

        self._active_profile: AgentProfile = AgentProfile.STANDARD

    def set_profile(self, profile: AgentProfile) -> None:
        """Switch the active agent profile."""
        self._active_profile = profile

    def get_profile(self) -> AgentProfile:
        """Get the current agent profile."""
        return self._active_profile

    def get_config(self) -> ProfileConfig:
        """Get the configuration for the active profile."""
        return BUILTIN_PROFILES[self._active_profile]

    def select_model(self, task_type: TaskType) -> Model:
        """Select the best model for the current profile and task type.

        Strategy:
        1. Get agent role for this task type (via agent_model_registry)
        2. Get the model for the profile's tier (budget or premium)
        3. Fall through to next available model if unhealthy
        """
        config = self.get_config()
        role = self._task_type_to_agent_role(task_type)

        try:
            model = self._registry.get_model_for(role, tier=config.model_tier)
        except KeyError:
            # Fallback: use first available model for this task type
            model = self._fallback.select_model(task_type)

        # Check for reasoning model preference
        if config.prefer_reasoning_models and not self._is_reasoning(model):
            alt = self._get_reasoning_alternative(model, task_type)
            if alt:
                model = alt

        # Check for large context preference
        if config.prefer_large_context:
            alt = self._get_large_context_alternative(model, task_type)
            if alt:
                model = alt

        # Ensure model is healthy
        if not self._fallback.is_model_healthy(model):
            model = self._fallback.select_model(task_type)

        return model

    def list_profiles(self) -> list[dict]:
        """List all available profiles with descriptions."""
        return [config.to_dict() for config in BUILTIN_PROFILES.values()]

    def get_behavior_params(self) -> dict:
        """Get behavior parameters for the current profile."""
        config = self.get_config()
        return {
            "temperature_range": config.temperature_range,
            "max_iterations": config.max_iterations,
            "enable_web_search": config.enable_web_search,
            "enable_self_review": config.enable_self_review,
            "strict_validation": config.strict_validation,
        }

    def _task_type_to_agent_role(self, task_type: TaskType) -> AgentRole:
        """Map TaskType to AgentRole for model selection."""
        mapping = {
            TaskType.CODE_GEN: AgentRole.DEVELOPER,
            TaskType.CODE_REVIEW: AgentRole.REVIEWER,
            TaskType.ARCHITECT: AgentRole.ARCHITECT,
            TaskType.DATA_EXTRACT: AgentRole.RESEARCHER,
            TaskType.REASONING: AgentRole.ARCHITECT,
            TaskType.WRITING: AgentRole.PRODUCT_MANAGER,
        }
        return mapping.get(task_type, AgentRole.DEVELOPER)
```

**Profile selection UI:**
```
┌──────────────────────────────────────────────────────────────┐
│  🤖 Agent Profile                                            │
│                                                              │
│  ○ Standard — Fast, token-efficient, most tasks              │
│  ● Max — Deep reasoning, complex tasks (+$0.15/req)         │
│  ○ Creative — Experimental, brainstorming (+$0.10/req)      │
│  ○ Conservative — Strict validation, production (+$0.20/req)│
│  ○ Research — Web search, broad knowledge (+$0.05/req)      │
│                                                              │
│  Active model: Claude Sonnet 4.6 (Reasoning, 200K context)  │
│  Iterations: 5  |  Temperature: 0.1-0.3  |  Self-review: ON │
└──────────────────────────────────────────────────────────────┘
```

**CLI flags:**
```bash
# Use Max profile for complex tasks
python -m orchestrator run --project "..." --profile max

# Use Creative profile for brainstorming
python -m orchestrator brainstorm --project "..." --profile creative

# Use Conservative profile for production deployment
python -m orchestrator run --project "..." --profile conservative
```

**File changes:**
- NEW: `orchestrator/agent_profiles.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — integrate `ProfileRouter`, add `--profile` flag
- MODIFY: `orchestrator/cli.py` — add profile selection to all commands
- MODIFY: `orchestrator/agent_model_registry.py` — support profile-based lookups

#### Verification Gate

```bash
# Run with Standard profile, verify budget models used, 3 iterations
# Run with Max profile, verify premium models used, 5 iterations
# Switch profiles mid-project, verify model changes
```

---

## Phase W2: Code View with Target/Lock Files (AI Scope Control)

### Objective

Allow users to scope AI attention to specific files ("Target") and protect specific files from AI modification ("Lock"). This gives fine-grained control over what the AI can and cannot change in the generated codebase.

### Current State

- `Orchestrator._execute_all()` processes all tasks in dependency order
- No mechanism to restrict AI writes to specific files
- `PreflightValidator` checks safety but doesn't scope writes
- Phase V4 (Permission Modes) controls tool permission but at the tool level, not file level

### Implementation

#### W2.1 — Create `orchestrator/file_scope.py`

```python
"""
File Scope Manager — target and lock files for AI attention control.
=======================================================================

Targeted files: AI is asked to focus attention on these specific files.
Locked files: AI is prevented from modifying these files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileScope:
    """Controls which files the AI can see and modify."""

    project_dir: Path

    # Files the AI should focus on (others get less attention)
    targeted_files: set[str] = field(default_factory=set)

    # Files the AI must not modify
    locked_files: set[str] = field(default_factory=set)

    # Locked directories (all files within are protected)
    locked_dirs: set[str] = field(default_factory=set)

    def target(self, file_path: str) -> None:
        """Mark a file as AI focus target."""
        full_path = self.project_dir / file_path
        if full_path.exists():
            self.targeted_files.add(file_path)

    def untarget(self, file_path: str) -> None:
        """Remove a file from AI focus."""
        self.targeted_files.discard(file_path)

    def lock(self, file_path: str) -> None:
        """Prevent AI from modifying a file."""
        full_path = self.project_dir / file_path
        if full_path.exists():
            self.locked_files.add(file_path)

    def unlock(self, file_path: str) -> None:
        """Allow AI to modify a previously locked file."""
        self.locked_files.discard(file_path)

    def lock_dir(self, dir_path: str) -> None:
        """Prevent AI from modifying any file in a directory."""
        full_path = self.project_dir / dir_path
        if full_path.exists() and full_path.is_dir():
            self.locked_dirs.add(dir_path)

    def is_locked(self, file_path: str) -> bool:
        """Check if a file is locked from AI modification."""
        if file_path in self.locked_files:
            return True
        # Check if file is in a locked directory
        for locked_dir in self.locked_dirs:
            if file_path.startswith(locked_dir.rstrip("/") + "/"):
                return True
        return False

    def get_effective_files(self, all_files: list[str]) -> tuple[list[str], list[str]]:
        """Get the effective file lists for AI execution.

        Returns:
            (writable_files, readonly_context_files)

        writable_files: Files the AI is allowed to modify
        readonly_context_files: Files the AI can read for context but not modify
        """
        writable = []
        context_only = []

        for file in all_files:
            if self.is_locked(file):
                context_only.append(file)
            else:
                writable.append(file)

        return writable, context_only

    def build_scope_context(self) -> str:
        """Generate AI context about file scope restrictions.

        Injects into the AI's system prompt to inform it of
        which files are targeted (focus here) and locked (don't touch).
        """
        parts = []

        if self.targeted_files:
            parts.append("## Attention Focus")
            parts.append("Focus your attention on these files:\n")
            for f in sorted(self.targeted_files):
                parts.append(f"- `{f}`")

        if self.locked_files or self.locked_dirs:
            parts.append("\n## Protected Files (DO NOT MODIFY)")
            parts.append("These files are locked — do not modify them:\n")
            for f in sorted(self.locked_files):
                parts.append(f"- `{f}` (locked)")
            for d in sorted(self.locked_dirs):
                parts.append(f"- `{d}/` (directory locked)")

        return "\n".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "targeted_files": sorted(self.targeted_files),
            "locked_files": sorted(self.locked_files),
            "locked_dirs": sorted(self.locked_dirs),
        }


class ScopedExecutionMixin:
    """Mix-in for Orchestrator that respects file scope during execution.

    Before each task executes:
    1. Check if the task's output files include any locked files
    2. If so, skip those files (or warn the user)
    3. Inject scope context into the AI prompt
    """

    async def _execute_with_scope(
        self, task: Task, scope: FileScope
    ) -> TaskResult:
        """Execute a task with file scope restrictions.

        Args:
            task: The task to execute
            scope: File scope configuration

        Returns:
            TaskResult with scope-respecting output
        """
        # Inject scope context into the task prompt
        scope_context = scope.build_scope_context()
        if scope_context:
            task.prompt = f"{scope_context}\n\n{task.prompt}"

        # Check if any locked files would be modified
        locked_conflicts = []
        if hasattr(task, 'output_files'):
            for file_path in task.output_files:
                if scope.is_locked(file_path):
                    locked_conflicts.append(file_path)

        if locked_conflicts:
            logger.warning(
                f"Task {task.id} would modify locked files: {locked_conflicts}. "
                "Skipping those files."
            )

        # Execute with scope
        result = await self._execute_task(task)

        # Filter locked files from output
        if locked_conflicts:
            result.locked_files_skipped = locked_conflicts

        return result
```

**UI integration — Code View with context menu:**
```
┌──────────────────────────────────────────────────────────────┐
│  📁 Files                          [Target] [Lock] [Ask Bolt] │
│                                                              │
│  ├── src/                                                    │
│  │   ├── auth.py              🔒 Locked                     │
│  │   ├── api.py               🎯 Targeted                   │
│  │   ├── models.py                                         │
│  │   ├── utils/                                             │
│  │   │   └── debug.py         🔒 (dir locked)               │
│  │   └── config.py                                         │
│  ├── tests/                   🔒 Directory locked           │
│  └── requirements.txt                                       │
│                                                              │
│  Legend: 🔒 Locked (AI can't modify)   🎯 Targeted (focus)  │
└──────────────────────────────────────────────────────────────┘
```

**CLI commands:**
```bash
# Target specific files for AI focus
python -m orchestrator scope target src/auth.py src/api.py

# Lock sensitive files from AI modification
python -m orchestrator scope lock src/config.py tests/

# List current scope
python -m orchestrator scope list

# Clear all scope restrictions
python -m orchestrator scope clear
```

**File changes:**
- NEW: `orchestrator/file_scope.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — integrate `FileScope` into task execution
- MODIFY: `orchestrator/cli.py` — add `scope` command group
- MODIFY: `ide_frontend/src/components/CodeEditor.tsx` — add context menu actions
- DEPENDS ON: Phase U1 (Code Editor) for in-browser file explorer

#### Verification Gate

```bash
# Lock config.py, target auth.py, run a task
# Verify: auth.py focused, config.py not modified
# Lock entire tests/ directory, verify no test files changed
# Clear scope, verify restrictions removed
```

---

## Phase W3: Plan Mode with Quick Action Buttons

### Objective

Extend Plan Mode (Phase N4 — Brainstorming) with contextual **quick action buttons** at the end of AI responses — "Implement this plan", "Show an example", "Refine this idea". Currently the orchestrator's plan mode only generates text plans.

### Current State

- Phase N4 (Brainstorming Mode) — AI asks clarifying questions
- Phase 2 (Plan Workflow) — PlanOrchestrator generates task lists
- `dry_run()` returns a plan but doesn't offer interactive actions
- No quick actions on plan/discussion responses

### Implementation

#### W3.1 — Extend PlanOrchestrator with quick actions

```python
@dataclass
class QuickAction:
    """A contextual action the user can take after a plan/discussion response."""
    id: str
    label: str  # "Implement this plan", "Show an example", "Refine this idea"
    description: str
    action: str  # "implement", "example", "refine", "explain_more", "alternative"
    icon: str  # "⚡", "💡", "🔄", "❓", "🎨"

class PlanOrchestrator:
    """Extended with quick action detection."""

    async def plan(
        self, description: str, criteria: str
    ) -> PlanResponse:
        """Generate a plan with quick action buttons.

        After generating the plan, analyzes the response for
        contextual quick actions:
        - "Implement this plan" — if the plan is concrete and actionable
        - "Show an example" — if the plan describes a UI or API
        - "Refine this idea" — if the plan has ambiguity
        - "Suggest alternatives" — if the plan has trade-offs
        """

    def _detect_quick_actions(
        self, plan: str, response: str
    ) -> list[QuickAction]:
        """Detect which quick actions are appropriate for a response.

        Uses LLM to analyze the response and determine which
        actions make sense in context.
        """
        analysis_prompt = (
            "Given this plan/response, which of these actions make sense?\n"
            "Return a JSON array of action IDs.\n\n"
            "Available actions:\n"
            "- implement: The plan is concrete enough to execute immediately\n"
            "- example: The user might benefit from seeing an example\n"
            "- refine: The plan has ambiguity that could benefit from refinement\n"
            "- alternative: There are trade-offs worth exploring with alternatives\n"
            "- explain_more: Parts of the plan need more explanation\n\n"
            f"RESPONSE:\n{response[:2000]}"
        )

        # Call cheapest model for action detection
        actions_json = await self._client.call(
            model=self._get_cheapest_model(),
            prompt=analysis_prompt,
            max_tokens=100,
        )

        action_ids = json.loads(actions_json.text)
        return [self.QUICK_ACTIONS[a] for a in action_ids if a in self.QUICK_ACTIONS]

    QUICK_ACTIONS: dict[str, QuickAction] = {
        "implement": QuickAction(
            id="implement",
            label="⚡ Implement this plan",
            description="Switch to Build mode and execute the plan",
            action="implement",
            icon="⚡",
        ),
        "example": QuickAction(
            id="example",
            label="💡 Show an example",
            description="Show a concrete code example of the plan",
            action="example",
            icon="💡",
        ),
        "refine": QuickAction(
            id="refine",
            label="🔄 Refine this idea",
            description="Iterate on the plan for more detail",
            action="refine",
            icon="🔄",
        ),
        "alternative": QuickAction(
            id="alternative",
            label="🎨 Suggest alternatives",
            description="Explore different approaches to this plan",
            action="alternative",
            icon="🎨",
        ),
        "explain_more": QuickAction(
            id="explain_more",
            label="❓ Explain in more detail",
            description="Get deeper explanation of a specific part",
            action="explain_more",
            icon="❓",
        ),
    }
```

**UI display after Plan Mode:**
```
┌──────────────────────────────────────────────────────────────────┐
│  🎯 Plan: Build Inventory REST API                               │
│                                                                  │
│  Phase 1: PostgreSQL schema + FastAPI scaffold          $0.12   │
│  Phase 2: CRUD endpoints for inventory items            $0.45   │
│  Phase 3: Authentication with JWT                       $0.32   │
│  Phase 4: Tests and documentation                       $0.19   │
│                                                                  │
│  Estimated total: $1.08  |  4 phases  |  ~8 minutes             │
│                                                                  │
│  [⚡ Implement this plan]  [💡 Show an example]                  │
│  [🔄 Refine this idea]   [🎨 Suggest alternatives]              │
└──────────────────────────────────────────────────────────────────┘
```

**File changes:**
- MODIFY: `orchestrator/plan_orchestrator.py` — add `_detect_quick_actions()`, `QUICK_ACTIONS`
- MODIFY: `orchestrator/engine.py` — return quick actions from `plan()`
- MODIFY: `ide_frontend/src/components/PlanMode.tsx` — render quick action buttons
- DEPENDS ON: Phase N4 (Brainstorming — PlanOrchestrator base)
- DEPENDS ON: Phase U1 (Plan Review Panel — UI for plan display)

#### Verification Gate

```bash
# Generate a plan, verify quick actions appear based on context
# Click "Implement", verify switches to Build mode and executes
# Click "Refine", verify enters refinement loop
```

---

## Phase W4: Project vs Site Separation (Explicit Publish)

### Objective

Separate the "project" (development workspace) from the "site" (published version). Changes in the project don't automatically go live — the user must explicitly publish. This prevents accidental deployments of broken or unfinished code.

### Current State

- `Orchestrator._execute_all()` writes results to `self.results` and `self._output_dir`
- No concept of "published" vs "in development"
- `app_assembler.py` writes all task outputs to files immediately
- No staged deployment workflow

### Implementation

#### W4.1 — Add publish workflow to Orchestrator

```python
class Orchestrator:
    """Extended with explicit publish workflow.

    Development mode: All changes go to the working directory.
    Nothing is published without explicit user action.

    Publish: Creates a snapshot of the current working state
    as the "live" version that users interact with.
    """

    def __init__(self, ..., output_dir: Path):
        # Separate directories for development and published versions
        self._dev_dir = output_dir
        self._published_dir = output_dir / ".published"

        # Track what's published
        self._published_version: str | None = None
        self._unpublished_changes: list[str] = []

    @property
    def is_published(self) -> bool:
        """Whether the project has a published version."""
        return self._published_version is not None

    @property
    def has_unpublished_changes(self) -> bool:
        """Whether there are changes in dev not yet published."""
        return len(self._unpublished_changes) > 0

    @property
    def published_at(self) -> float | None:
        """When the project was last published (timestamp)."""
        return self._last_published_at

    async def publish(
        self,
        version_label: str | None = None,
        auto_test: bool = True,
    ) -> bool:
        """Publish the current development version to production.

        Workflow:
        1. Run validation suite (tests, lint, security)
        2. Create a version (checkpoint)
        3. Copy all files to the published directory
        4. Update published version tracking
        5. Clear unpublished changes tracking

        Args:
            version_label: Optional label for the published version
            auto_test: Run validation before publishing (default: True)

        Returns:
            True if publish succeeded
        """
        if auto_test:
            # Run validations before publishing
            test_result = await self._run_validation_suite()
            if not test_result.passed:
                logger.warning("Publish blocked: validation failed")
                return False

        # Create version
        version = await self._version_mgr.create_version(
            description=version_label or "Published version",
        )

        # Copy dev files to published directory
        if self._published_dir.exists():
            shutil.rmtree(self._published_dir)
        shutil.copytree(self._dev_dir, self._published_dir)

        # Update tracking
        self._published_version = version.id
        self._published_version_label = version_label
        self._last_published_at = time.time()
        self._unpublished_changes = []

        logger.info(f"Published project at version {version.id}")
        return True

    async def unpublish(self) -> bool:
        """Remove the published version — site goes offline."""
        if not self.is_published:
            return False

        shutil.rmtree(self._published_dir)
        self._published_version = None
        self._published_version_label = None
        self._last_published_at = None

        logger.info("Unpublished project")
        return True

    async def get_unpublished_changes(self) -> list[str]:
        """Get list of files changed since last publish.

        Returns:
            List of file paths with changes
        """
        if not self.is_published:
            return list(self._get_all_files(self._dev_dir))

        changes = []
        for file in self._get_all_files(self._dev_dir):
            dev_path = self._dev_dir / file
            pub_path = self._published_dir / file
            if not pub_path.exists():
                changes.append(f"{file} (new)")
            elif dev_path.read_text() != pub_path.read_text():
                changes.append(f"{file} (modified)")

        # Check for deleted files
        for pub_file in self._get_all_files(self._published_dir):
            if not (self._dev_dir / pub_file).exists():
                changes.append(f"{pub_file} (deleted in dev)")

        self._unpublished_changes = changes
        return changes
```

**UI status bar:**
```
┌──────────────────────────────────────────────────────────────────┐
│  📦 Project: Inventory App                  🟢 Published v1.2   │
│                                                   3 changes ▼    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Unpublished changes:                                      │   │
│  │ • src/auth.py (modified)                                  │   │
│  │ • src/api/search.py (new)                                 │   │
│  │ • tests/test_search.py (new)                              │   │
│  │                                                            │   │
│  │ [🚀 Publish]  [↩ Revert to published]  [📋 View diff]    │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**File changes:**
- MODIFY: `orchestrator/engine.py` — add `publish()`, `unpublish()`, `get_unpublished_changes()`
- MODIFY: `orchestrator/cli.py` — add `publish`, `unpublish`, `status` commands
- MODIFY: `ide_frontend/src/components/StatusBar.tsx` — show publish status
- DEPENDS ON: Phase V6 (Versions — version management)

#### Verification Gate

```bash
# Make changes, verify "3 unpublished changes" shown
# Publish, verify published version created, changes cleared
# Make more changes, verify diff between dev and published
# Unpublish, verify published directory removed
```

---

## Phase W5: Team Templates for Repetitive Project Setup

### Objective

Allow teams to create and share project templates — pre-configured project setups with entity schemas, auth config, design system, integrations, and module selections. Skip repetitive boilerplate setup when starting new projects.

### Current State

- Phase V7 (Templates) — template registry with categories but individual-focused
- `orchestrator/scaffold/` — static project templates (FastAPI, Next.js, React, CLI)
- No team-level template sharing
- Every new project starts from default scaffolds

### Implementation

#### W5.1 — Extend TemplateRegistry with team templates

```python
class TeamTemplate:
    """A shareable project template for teams.

    Templates capture:
    - Architecture decisions (from ProjectRules)
    - Entity schemas (from ConfigGenerator)
    - Auth configuration
    - Design system reference
    - Module selections
    - Integration setup
    - Environment variables template
    """

    name: str
    description: str
    team_id: str
    created_by: str
    created_at: float
    updated_at: float

    # What the template captures
    architecture: dict  # ProjectRules snapshot
    entities: list[dict]  # Entity schemas
    auth_config: dict  # Auth configuration
    design_system_url: str | None  # Registry URL
    modules: list[str]  # Selected module names
    integrations: list[str]  # Selected integration names
    env_template: dict[str, str]  # Environment variable template
    rules_file: str | None  # .orchestrator-rules.yml content

    def apply_to_new_project(self, project_name: str, customizations: dict = None) -> dict:
        """Apply this template to a new project.

        Returns the project configuration for the orchestrator to use.
        """
```

**CLI commands:**
```bash
# Save current project as a team template
python -m orchestrator template save --name "ecommerce-api" --team my-team

# List team templates
python -m orchestrator template list --team my-team

# Apply a team template to a new project
python -m orchestrator new --template team:ecommerce-api --project "New Store"

# Share template with another team
python -m orchestrator template share ecommerce-api --with other-team
```

**File changes:**
- MODIFY: `orchestrator/template_registry.py` — add `TeamTemplate` class and team support
- MODIFY: `orchestrator/engine.py` — support `--template team:` prefix
- MODIFY: `orchestrator/cli.py` — add `template save|share`
- DEPENDS ON: Phase V7 (Templates — TemplateRegistry base)

#### Verification Gate

```bash
# Create a project, save it as a team template
# Verify: template captures entities, auth, modules, integrations
# Apply the template to a new project, verify it matches the original
# Share the template with another team, verify it's available
```

---

## Integration with Existing Plans

```
Phase W1 (Agent Profiles) ───────────────────────────────────────────────────┐
    │  No dependencies — new abstraction layer                                 │
    │  Uses agent_model_registry for model lookups                             │
Phase W2 (File Scope Control) ─────────────────────────────────────────────────┤
    │  Depends on Phase U1 (Code Editor — file explorer context menu)          │
Phase W3 (Quick Action Buttons) ──────────────────────────────────────────────┤
    │  Depends on Phase N4 (Brainstorming — PlanOrchestrator base)             │
    │  Depends on Phase U1 (Plan Review Panel — UI for plan display)           │
Phase W4 (Project vs Site) ────────────────────────────────────────────────────┤
    │  Depends on Phase V6 (Versions — version management)                     │
Phase W5 (Team Templates) ──────────────────────────────────────────────────────┘
    Depends on Phase V7 (Templates — TemplateRegistry base)
```

## Bolt.new-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| W1 | Agent Profiles | 1 | 3 | ~400 | 2-3 |
| W2 | File Scope Control | 1 | 3 | ~300 | 3-4 |
| W3 | Quick Action Buttons | 0 | 3 | ~200 | 1-2 |
| W4 | Project vs Site | 0 | 3 | ~250 | 2-3 |
| W5 | Team Templates | 0 | 3 | ~200 | 2-3 |
| **Bolt.new Subtotal** | **5 phases** | **2** | **15** | **~1,350** | **10-15** |

## Combined Grand Total (All Nine Sources)

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
| 9 | Bolt.new | W1-W5 | 2 | 15 | 10-15 |
| **Grand Total** | **56** | **73** | **141** | **118-170** |
