# Enhancement Plan: Replit-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Replit Agent and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — pending prioritisation

---

## Overview

Six enhancements identified from Replit's Agent capabilities that would add significant value to the Multi-LLM Orchestrator. Ranked by ROI and implementation feasibility.

```
Phase 1: Checkpoints + AI Context Restoration        (Highest ROI, 2-3 days)
Phase 2: Plan-then-Build Interactive Workflow        (High ROI, 3-4 days)
Phase 3: Code Optimizations Self-Review Toggle        (Medium ROI, 1-2 days)
Phase 4: Isolated Sandbox Tasks with Review-before-Merge (Medium ROI, 3-4 days)
Phase 5: Browser-Based App Testing                    (Medium-High ROI, 4-5 days)
Phase 6: Design System Injection                       (Lower ROI, 1-2 days)
```

---

## Phase 1: Checkpoints with AI Context Restoration

### Objective

Create named snapshots of the full development state — project files, conversation context, database contents, environment config — with bidirectional rollback that restores the Agent's understanding of the project.

### Current State

- `orchestrator/state.py` — `StateManager` with SQLite persistence saves `ProjectState` (budget, tasks, results)
- `orchestrator/models.py` — `ProjectState` dataclass holds budget, tasks, results, api_health, execution_order
- Resume capability exists in `Orchestrator._resume_project()`
- No conversation context capture, no database snapshotting

### Implementation

#### 1.1 — Create `orchestrator/checkpoint_manager.py`

```python
@dataclass
class Checkpoint:
    id: str
    project_id: str
    timestamp: float
    label: str  # e.g. "After auth module", "Pre-refactor"
    state_snapshot: ProjectState  # Full task/budget/results state
    context_snapshot: list[dict]  # Conversation history
    file_manifest: dict[str, str]  # path → sha256 hash
    db_dump: bytes | None  # SQLite dump if applicable
    parent_checkpoint_id: str | None  # For chain navigation

class CheckpointManager:
    def __init__(self, state_mgr: StateManager, output_dir: Path):
        self._state_mgr = state_mgr
        self._output_dir = output_dir
        self._checkpoints: list[Checkpoint] = []
        self._current_index: int = -1

    async def create(self, label: str, context: list[dict]) -> Checkpoint:
        """Snapshot current state with label and conversation context."""

    async def rollback(self, checkpoint_id: str) -> Checkpoint:
        """Restore project to checkpoint — files, state, context."""

    async def roll_forward(self) -> Checkpoint | None:
        """Move forward in checkpoint chain if available."""

    def timeline(self) -> list[dict]:
        """Return checkpoint timeline for UI navigation."""

    async def diff(self, cp_a: str, cp_b: str) -> str:
        """Generate a diff between two checkpoints."""
```

**File changes:**
- NEW: `orchestrator/checkpoint_manager.py` (~250 lines)
- MODIFY: `orchestrator/engine.py` — wire `CheckpointManager` into `Orchestrator.__init__`
- MODIFY: `orchestrator/state.py` — add `save_checkpoint()` and `load_checkpoint()` methods
- MODIFY: `orchestrator/models.py` — add `Checkpoint` dataclass (or keep in checkpoint_manager)

**Integration points:**
- `Orchestrator._execute_all()` — auto-create checkpoint after each completed task
- `Orchestrator.run_project()` — create "project_started" and "project_completed" checkpoints
- `Orchestrator.__aexit__()` — flush all checkpoints
- `engine.py._execute_task()` — create checkpoint before risky operations (fallback, model escalation)

#### 1.2 — Auto-Checkpoint Triggers

| Trigger | Checkpoint Label |
|---------|-----------------|
| Task completed | `after:{task_id}` |
| Task failed (pre-retry) | `pre_retry:{task_id}` |
| Model fallback triggered | `fallback:{from_model}→{to_model}` |
| Budget 75% warning | `budget_75pct` |
| Project start | `project_started` |
| Project complete | `project_completed` |

#### Verification Gate

```bash
python -c "
from orchestrator.checkpoint_manager import CheckpointManager
# Create, rollback, verify state restored, verify context restored
"
```

---

## Phase 2: Plan-then-Build Interactive Workflow

### Objective

Extend the existing `dry_run()` into an interactive planning session where the user and Agent collaborate on task definition before execution begins.

### Current State

- `Orchestrator.dry_run()` — decomposes project into `ExecutionPlan` + `TaskPlan` objects, returns without executing
- `AgentOrchestrator._decompose_goal()` — keyword-based task decomposition
- No interactive plan review loop

### Implementation

#### 2.1 — Create `orchestrator/plan_orchestrator.py`

```python
@dataclass
class PlanTask:
    id: str
    title: str
    description: str
    priority: str  # P0, P1, P2, P3
    dependencies: list[str]
    estimated_model: Model
    estimated_cost: float
    status: str  # draft, approved, rejected, executed
    rationale: str

class PlanOrchestrator:
    def __init__(self, orchestrator: Orchestrator):
        self._orch = orchestrator
        self._plan: list[PlanTask] = []
        self._mode: str = "plan"  # "plan" or "build"

    async def plan(self, description: str, criteria: str) -> list[PlanTask]:
        """Generate a plan without executing. Returns tasks for review."""

    async def refine(self, feedback: str) -> list[PlanTask]:
        """Refine the plan based on user feedback — add, remove, reorder tasks."""

    async def approve(self, task_ids: list[str] | None = None) -> None:
        """Approve specific tasks (or all) for execution."""

    async def execute_plan(self) -> ProjectState:
        """Execute approved tasks in dependency order. Switches to build mode."""

    def to_dict(self) -> dict:
        """Serialize plan for display/storage."""
```

**Integration:**
- Wraps `Orchestrator.dry_run()` → displays tasks → collects user feedback → `Orchestrator.run_project()`
- Uses existing `DependencyResolver.topological_sort()` for task ordering
- Uses existing `COST_TABLE` for cost estimation

#### 2.2 — Add Plan Mode to CLI

```python
# orchestrator/cli.py — new command
@cli.command()
@click.option("--project", required=True)
@click.option("--criteria", default="Works correctly")
@click.option("--mode", type=click.Choice(["plan", "build"]), default="plan")
def plan(project, criteria, mode):
    """Plan a project before building."""
    orch = Orchestrator()
    planner = PlanOrchestrator(orch)
    tasks = await planner.plan(project, criteria)
    # Display tasks, prompt for approval
```

**File changes:**
- NEW: `orchestrator/plan_orchestrator.py` (~200 lines)
- MODIFY: `orchestrator/cli.py` — add `plan` command
- MODIFY: `orchestrator/engine.py` — expose `PlanOrchestrator` as `self.planner`

#### Verification Gate

```bash
python -m orchestrator plan --project "Build a REST API" --criteria "CRUD + auth"
# Expect: printed task list with priorities, estimated costs, dependencies
# Expect: prompt to approve/reject/refine
```

---

## Phase 3: Code Optimizations (Agent Self-Review Toggle)

### Objective

Add a configurable self-review pass where the same model reviews its own output before the expensive cross-provider critique runs. Togglable per task or per project.

### Current State

- `CritiqueCycle` in `application/critique_cycle.py` — always runs cross-provider review
- `DeveloperAgent` — 3-attempt self-correction loop (generate → validate → retry with error context)
- No configurable self-review pre-pass

### Implementation

#### 3.1 — Add Self-Review to CritiqueCycle

```python
class CritiqueCycle:
    def __init__(
        self,
        client: UnifiedClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_streaming: bool = False,
        self_review_enabled: bool = False,  # NEW
        self_review_threshold: float = 0.85,  # NEW: skip self-review if score >= this
    ):
        ...

    async def _self_review(
        self, model: Model, output: str, prompt: str
    ) -> tuple[str, float]:
        """Quick same-model review to catch obvious issues before cross-model critique.
        
        Uses a shorter prompt and lower max_tokens to minimise cost.
        Only runs if self_review_enabled=True and the initial generation
        score is below self_review_threshold.
        """
        review_prompt = (
            "Quick review: check for syntax errors, missing imports, "
            "obvious bugs, and incomplete implementations.\n\n"
            f"TASK: {prompt[:500]}\n\nCODE:\n{output[:2000]}"
        )
        response = await self.client.call(
            model=model,  # Same model — cheap self-check
            prompt=review_prompt,
            max_tokens=300,
            temperature=0.1,
        )
        score = self._extract_score(response.text)
        return response.text, score
```

**Integration into `run_cycle()`:**
```
generate → (if self_review_enabled) → self_review → apply suggestions
         → critique (different provider) → revise → evaluate → plateau check
```

#### 3.2 — Configuration Hook

```python
# orchestrator/crosscutting/config.py
class FeatureFlags(BaseSettings):
    ...
    self_review_enabled: bool = True  # NEW
    self_review_threshold: float = 0.85  # NEW
```

**File changes:**
- MODIFY: `orchestrator/application/critique_cycle.py` — add `_self_review()` method
- MODIFY: `orchestrator/crosscutting/config.py` — add feature flags
- MODIFY: `orchestrator/engine.py` — pass flags through at init

#### Verification Gate

```python
# Unit test: self_review catches a missing import that cross-model critique would catch
# Cost assertion: self_review costs < 20% of cross-model critique
```

---

## Phase 4: Isolated Sandbox Tasks with Review-before-Merge

### Objective

Execute tasks in isolated workspace copies, produce diffs, present results for human review, and merge only approved changes. Supports AI-assisted conflict resolution.

### Current State

- `Orchestrator._execute_all()` — sequential/parallel execution writing to shared `self.results`
- `orchestrator/git_integration.py` — branch→commit→push per milestone
- `orchestrator/app_assembler.py` — writes task outputs to files
- No sandbox isolation between parallel tasks

### Implementation

#### 4.1 — Create `orchestrator/isolated_task_runner.py`

```python
@dataclass
class SandboxResult:
    task_id: str
    diff: str  # Unified diff of changes
    test_results: str
    files_changed: list[str]
    cost: float
    status: str  # "ready_for_review", "failed", "needs_retry"

class IsolatedTaskRunner:
    def __init__(
        self,
        workspace: Path,
        executor: ExecutorService,
        git: GitIntegration | None = None,
    ):
        self._workspace = workspace
        self._executor = executor
        self._git = git

    async def run_isolated(
        self, task: Task, sandbox_dir: Path
    ) -> SandboxResult:
        """Copy workspace to sandbox, execute task, return diff + results."""

    async def review(self, result: SandboxResult) -> bool:
        """Present diff for human review. Returns True if approved."""

    async def merge(self, result: SandboxResult) -> None:
        """Apply approved changes back to main workspace. Handle conflicts."""

    async def run_batch(
        self, tasks: list[Task], max_concurrent: int = 3
    ) -> dict[str, SandboxResult]:
        """Run multiple tasks in parallel sandboxes."""
```

**Sandbox lifecycle:**
```
main_workspace/
    ├── .sandboxes/
    │   ├── task_auth/    ← isolated copy for auth task
    │   ├── task_api/     ← isolated copy for API task
    │   └── task_db/      ← isolated copy for DB task
    └── src/              ← main (untouched until merge)
```

**Conflict resolution:**
- Per-file merge: if two tasks modify the same file, AI-assisted merge using the LLM
- Automatic: if files don't overlap, merge automatically
- Manual: if AI confidence < 0.8, flag for human review

**File changes:**
- NEW: `orchestrator/isolated_task_runner.py` (~300 lines)
- MODIFY: `orchestrator/engine.py` — add `_isolated_runner` attr, use in `_execute_all()`
- MODIFY: `orchestrator/git_integration.py` — add `apply_patch()` method for merge

#### Verification Gate

```bash
# Run two tasks in parallel sandboxes that modify different files → auto-merge
# Run two tasks that modify the same file → flag conflict, AI resolve, human approve
```

---

## Phase 5: Browser-Based App Testing

### Objective

Use Playwright to spin up a browser, navigate generated web apps, validate UI functionality, capture video replays, and automatically fix issues found.

### Current State

- `validate_pytest()` — subprocess test runner
- `app_verifier.py` — startup checks (npm build, Docker build, process alive)
- `PreflightValidator` — static safety checks
- No browser interaction

### Implementation

#### 5.1 — Create `orchestrator/browser_tester.py`

```python
@dataclass
class BrowserTestResult:
    passed: bool
    test_name: str
    duration_ms: float
    video_path: str | None  # Path to recorded video
    screenshot_paths: list[str]
    errors: list[str]
    suggestions: list[str]

class BrowserTester:
    """Uses Playwright to test generated web apps in a real browser."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        record_video: bool = True,
    ):
        self._headless = headless
        self._timeout = timeout_ms
        self._record_video = record_video

    async def test_app(
        self,
        app_url: str,
        test_scenarios: list[dict] | None = None,
    ) -> list[BrowserTestResult]:
        """
        Test a deployed app. If no scenarios provided, generates them via
        LLM analysis of the app's structure (routes, forms, buttons).
        """

    async def generate_scenarios(
        self, app_url: str, client: UnifiedClient
    ) -> list[dict]:
        """Ask LLM to generate test scenarios based on app structure."""

    async def test_scenario(
        self, page: Page, scenario: dict
    ) -> BrowserTestResult:
        """Execute one test scenario — navigate, click, fill, assert."""

    async def auto_fix(
        self,
        results: list[BrowserTestResult],
        client: UnifiedClient,
        orchestrator: Orchestrator,
    ) -> list[str]:
        """Feed failing tests back into the critique cycle for automatic repair."""
```

**Test scenario generation:**
The LLM inspects the app's route structure and generates scenarios like:
```json
{
    "name": "User signup flow",
    "steps": [
        {"action": "navigate", "url": "/signup"},
        {"action": "fill", "selector": "#email", "value": "test@example.com"},
        {"action": "fill", "selector": "#password", "value": "Test123!"},
        {"action": "click", "selector": "button[type=submit]"},
        {"action": "assert_url", "value": "/dashboard"},
        {"action": "assert_text", "selector": "h1", "value": "Welcome"}
    ]
}
```

**Integration:**
- Wire into `Orchestrator.__init__` as `self._browser_tester`
- Call after `_execute_all()` completes — spin up app, run browser tests
- Feed failures into `CritiqueCycle` as revision tasks
- Capture video replays for debugging

**File changes:**
- NEW: `orchestrator/browser_tester.py` (~400 lines)
- MODIFY: `orchestrator/engine.py` — wire into post-execution phase
- MODIFY: `requirements.txt` — add `playwright`
- NEW: `tests/test_browser_tester.py` — mock Playwright tests

#### 5.2 — Requirements

```bash
pip install playwright
playwright install chromium
```

#### Verification Gate

```bash
# Generate a simple FastAPI app, deploy locally, run BrowserTester.test_app()
# Expect: video recording, passing scenarios, auto-fixed failures
```

---

## Phase 6: Design System Injection

### Objective

Read a `.design-system.yml` config file and inject brand-specific design tokens, components, and visual standards into every generation prompt.

### Current State

- `UXDesignEnhancer` (`orchestrator/ux/design_enhancer.py`) — 20 generic WCAG/UX standards
- `ux_system_prompt()` — injects standards into LLM prompts
- No brand or design system awareness

### Implementation

#### 6.1 — Create `orchestrator/design_system.py`

```python
@dataclass
class DesignSystemConfig:
    name: str
    colors: dict[str, str]  # primary, secondary, accent, background, text, error
    typography: dict[str, str]  # heading_font, body_font, base_size, scale_ratio
    spacing_scale: list[int]  # [4, 8, 12, 16, 24, 32, 48, 64]
    border_radius: str  # "4px", "8px", etc.
    component_library: str | None  # "tailwind", "mui", "shadcn", etc.
    dark_mode: bool
    brand_assets: dict[str, str]  # logo_url, favicon_url, etc.

    @classmethod
    def from_yaml(cls, path: Path) -> "DesignSystemConfig":
        """Load from .design-system.yml."""

    def to_system_prompt(self) -> str:
        """Generate injection text for LLM prompts."""

class DesignSystemInjector:
    """Injects design system constraints into generation prompts."""

    def __init__(self, config: DesignSystemConfig | None = None):
        self._config = config or DesignSystemConfig.defaults()

    def inject(self, prompt: str, task_type: TaskType) -> str:
        """Add design system constraints to a generation prompt."""
        if task_type != TaskType.CODE_GEN:
            return prompt
        system_context = self._config.to_system_prompt()
        return f"{prompt}\n\n## Design System Requirements\n{system_context}"

    def inject_system_prompt(self, existing: str) -> str:
        """Merge design system requirements into an existing system prompt."""
```

**.design-system.yml example:**
```yaml
name: "Acme Corp"
colors:
  primary: "#2563EB"
  secondary: "#7C3AED"
  accent: "#F59E0B"
  background: "#F8FAFC"
  text: "#1E293B"
  error: "#DC2626"
typography:
  heading_font: "Inter"
  body_font: "Inter"
  base_size: "16px"
  scale_ratio: 1.25
spacing_scale: [4, 8, 12, 16, 24, 32, 48, 64]
border_radius: "8px"
component_library: "tailwind"
dark_mode: true
brand_assets:
  logo_url: "https://acme.com/logo.svg"
  favicon_url: "https://acme.com/favicon.ico"
```

**File changes:**
- NEW: `orchestrator/design_system.py` (~200 lines)
- MODIFY: `orchestrator/ux/design_enhancer.py` — call `DesignSystemInjector.inject()` 
- MODIFY: `orchestrator/engine.py` — load `.design-system.yml` at project start

#### Verification Gate

```bash
# Create .design-system.yml, run project generation
# Expect: generated code uses configured colors, fonts, spacing
# Expect: Tailwind classes match design system tokens
```

---

## Implementation Order & Dependencies

```
Phase 1 (Checkpoints) ─────────────────────────────────────────────┐
    │  No dependencies — uses existing StateManager                │
    │                                                               │
Phase 2 (Plan Workflow) ───────────────────────────────────────────┤
    │  No dependencies — wraps dry_run()                           │
    │                                                               │
Phase 3 (Self-Review Toggle) ──────────────────────────────────────┤
    │  No dependencies — extends CritiqueCycle                     │
    │                                                               │
Phase 4 (Sandbox Tasks) ───────────────────────────────────────────┤
    │  Depends on Phase 1 (checkpoints for sandbox snapshots)      │
    │                                                               │
Phase 5 (Browser Testing) ─────────────────────────────────────────┤
    │  Depends on Phase 4 (needs deployed app to test)             │
    │                                                               │
Phase 6 (Design System) ───────────────────────────────────────────┘
    No dependencies — standalone enhancement
```

Phases 1-3 can be parallelised. Phases 4-5 are sequential.

## Total Effort Estimate

| Phase | New Files | Modified Files | Est. Lines | Est. Days |
|-------|-----------|---------------|------------|-----------|
| 1. Checkpoints | 1 | 3 | ~250 | 2-3 |
| 2. Plan Workflow | 1 | 2 | ~200 | 2-3 |
| 3. Self-Review | 0 | 3 | ~80 | 1 |
| 4. Sandbox Tasks | 1 | 2 | ~300 | 3-4 |
| 5. Browser Testing | 1 | 2 | ~400 | 4-5 |
| 6. Design System | 1 | 2 | ~200 | 1-2 |
| **Total** | **5** | **14** | **~1,430** | **13-18** |

## Success Metrics

- **Checkpoints**: 100% of task completions create a checkpoint; rollback restores state + context within 2 seconds
- **Plan Workflow**: User can review and refine tasks before any code is generated
- **Self-Review**: 30% reduction in cross-model critique calls (caught by cheaper same-model pass)
- **Sandbox Tasks**: Zero main-workspace corruption from parallel task execution
- **Browser Testing**: 90% of generated web apps pass browser tests on first publish
- **Design System**: Generated apps use configured colors/fonts/spacing with no manual corrections needed
