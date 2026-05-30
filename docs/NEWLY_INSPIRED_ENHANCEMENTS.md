# Enhancement Plan: Newly-Inspired Features for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Source:** Gap analysis between Newly.app and Multi-LLM Orchestrator v6.0  
> **Status:** Draft — complements Replit, Lovable, and UI enhancement plans

---

## Overview

Newly.app is a mobile-first AI app builder (React Native/Expo). While many Newly features are platform-specific (iOS/Android deployment, Expo preview), several core concepts translate directly to the AI Orchestrator's code-generation framework.

Seven enhancements identified. Ranked by applicability to the orchestrator.

```
Phase N1: Ask Mode — Query-Only (No Code Gen)                 (Highest ROI, 1-2 days)
Phase N2: Chat-Level Restore Points                            (High ROI, 1-2 days)
Phase N3: Screenshot-Based Debugging                           (Medium ROI, 2-3 days)
Phase N4: Brainstorming Mode for Plan                          (Medium ROI, 2-3 days)
Phase N5: Auto-Commit with Descriptive Messages                (Medium ROI, 1 day)
Phase N6: Two-Way Git Sync (Pull External Changes)             (Lower ROI, 2-3 days)
Phase N7: Per-Prompt Cost Visibility                           (Lower ROI, 1 day)
```

---

## What Newly Has — Quick Reference

| Newly Feature | Description | Translates? |
|--------------|-------------|:-----------:|
| **Ask Mode** | 10x fewer prompts — answers questions without generating code | ✅ Phase N1 |
| **Chat Restore** | Restore button next to each chat message to revert to that point | ✅ Phase N2 |
| **Screenshot Debugging** | Paste screenshots of errors, AI diagnoses and fixes | ✅ Phase N3 |
| **Brainstorming Mode** | Plan mode asks deeper clarifying questions | ✅ Phase N4 |
| **Auto Commit Messages** | Descriptive commit message per AI change | ✅ Phase N5 |
| **Two-Way GitHub Sync** | Pull external changes back into project | ✅ Phase N6 |
| **Per-Call Cost Display** | Cost per prompt visible in chat | ✅ Phase N7 |
| **Liquid Backend** | Auto-generated REST API + PostgreSQL from natural language | Partial — exists via FastAPI scaffold |
| **Marketing Site Generator** | Generate landing page from app | ✗ (Scaffold could add this) |
| **App Store Deployment** | Build APK/AAB/IPA directly | ✗ (Platform-specific) |
| **Expo QR Preview** | Device testing via QR code | ✗ (Platform-specific) |
| **RevenueCat Payments** | In-app purchase integration | ✗ (Platform-specific) |

---

## Phase N1: Ask Mode — Query-Only Without Code Generation

### Objective

A budget-saving mode that answers questions, explains concepts, or provides advice without triggering the full generate→critique→revise→evaluate pipeline. Uses 1/10th the budget of a build request.

### Current State

- `Orchestrator._execute_task()` always runs the full pipeline — generate + critique + revise + evaluate
- `ArchitectureAnalyzer` has a query-only LLM pass for architecture optimization, but it's not exposed as a user-facing mode
- `CodebaseAnalyzer.debug()` answers questions but requires a full codebase scan first
- `dry_run()` plans without executing but still runs LLM calls for decomposition
- No concept of "answer only" vs "build"

### Implementation

#### N1.1 — Add `query()` method to Orchestrator

```python
class Orchestrator:
    async def query(self, question: str, context: str = "") -> str:
        """Answer a question without generating or modifying any code.

        Uses the cheapest available model (skips critique cycle entirely).
        Charges at 1/10th the rate of a build request.
        Budget compartment: 'query' (separate from 'generation').

        Args:
            question: The question to answer
            context: Optional codebase context for domain-specific questions

        Returns:
            Plain text answer — no code, no file writes
        """
        # Select cheapest capable model
        model = self._get_cheapest_query_model()

        # Build minimal prompt — no system context, no project rules, no critique
        prompt = f"Answer concisely:\n\n{question}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        # Single call — no critique cycle, no validation
        response = await self._client.call(
            model=model,
            prompt=prompt,
            max_tokens=500,  # Short answers only
            temperature=0.3,
            timeout=30,
        )

        # Charge query compartment
        await self._budget.charge(response.cost_usd, compartment="query")

        return response.text
```

**Budget compartments:**
```python
BUDGET_PARTITIONS = {
    "decomposition": 0.05,
    "generation": 0.45,
    "cross_review": 0.25,
    "evaluation": 0.15,
    "query": 0.05,        # NEW: separate budget for ask-mode questions
    "reserve": 0.05,
}
```

#### N1.2 — Add to CLI

```python
@cli.command()
@click.option("--question", required=True)
@click.option("--context", default="")
def ask(question, context):
    """Ask a question without generating code (10x cheaper)."""
    orch = Orchestrator()
    answer = await orch.query(question, context)
    print(answer)
```

Usage:
```bash
python -m orchestrator ask --question "Should I use JWT or session-based auth?"
python -m orchestrator ask --question "What's the best index strategy for this query?" --context "$(cat schema.sql)"
```

**File changes:**
- MODIFY: `orchestrator/engine.py` — add `query()` method, `_get_cheapest_query_model()`
- MODIFY: `orchestrator/budget.py` — add `query` compartment
- MODIFY: `orchestrator/models.py` — update `BUDGET_PARTITIONS`
- MODIFY: `orchestrator/cli.py` — add `ask` command

#### Verification Gate

```bash
python -m orchestrator ask --question "Explain hexagonal architecture"
# Expect: concise answer, no code generated, cost < $0.01
```

---

## Phase N2: Chat-Level Restore Points

### Objective

Lightweight "restore to this point" associated with each user prompt. When clicked, reverts the project state (tasks, budget, files) to exactly what it was before that prompt was processed.

### Current State

- `StateManager` with SQLite persistence — saves `ProjectState` per project
- `Orchestrator._resume_project()` — resumes from last saved state
- No per-prompt checkpointing — only whole-project resume
- `GitIntegration` has per-milestone branches but no per-prompt granularity
- Phase 1 (Checkpoints) in the Replit plan adds bulk checkpoints but they're manually triggered

### Implementation

#### N2.1 — Create `orchestrator/prompt_history.py`

```python
@dataclass
class PromptState:
    """Snapshot of project state before a specific prompt was processed."""
    prompt_id: str
    prompt_text: str
    timestamp: float
    project_state: ProjectState  # Full state snapshot
    file_manifest: dict[str, str]  # path → sha256 hash
    budget_before: float

class PromptHistory:
    """Tracks state before each prompt for quick restore."""

    def __init__(self, state_mgr: StateManager, max_history: int = 50):
        self._state_mgr = state_mgr
        self._history: list[PromptState] = []
        self._max_history = max_history

    async def capture(self, prompt_text: str) -> PromptState:
        """Snapshot state before processing a prompt."""
        state = await self._state_mgr.get_current_state()
        manifest = self._build_file_manifest()
        budget = self._get_current_budget()

        ps = PromptState(
            prompt_id=str(len(self._history)),
            prompt_text=prompt_text[:200],  # Truncate for display
            timestamp=time.time(),
            project_state=deepcopy(state),
            file_manifest=manifest,
            budget_before=budget,
        )
        self._history.append(ps)

        # Trim old history
        while len(self._history) > self._max_history:
            self._history.pop(0)

        return ps

    async def restore(self, prompt_id: str) -> bool:
        """Revert project state to before a specific prompt."""
        ps = next((p for p in self._history if p.prompt_id == prompt_id), None)
        if not ps:
            return False

        # Restore state
        await self._state_mgr.save_state(ps.project_state)

        # Restore files
        for path, sha256 in ps.file_manifest.items():
            stored = self._state_mgr.get_file_backup(sha256)
            if stored:
                (self._output_dir / path).write_text(stored)

        return True

    def list_restore_points(self) -> list[dict]:
        """Return list of available restore points for UI display."""
        return [
            {
                "prompt_id": p.prompt_id,
                "prompt_text": p.prompt_text,
                "timestamp": p.timestamp,
                "budget_before": p.budget_before,
            }
            for p in self._history
        ]
```

**Integration into `Orchestrator.run_project()`:**
```python
async def run_project(self, description: str, criteria: str):
    # Capture state before processing
    self._prompt_history.capture(description)

    # Process normally
    plan = await self.plan(description, criteria)
    result = await self._execute_all()
    return result
```

**Chat-level restore in UI:**
```
┌────────────────────────────────────────────┐
│ 💬 "Build a login screen"          [Restore] │
│ 💬 "Add JWT authentication"        [Restore] │
│ 💬 "Fix the header layout"         [Restore] │  ← Click to restore to before this
│ 💬 "Add dark mode toggle"          [Restore] │
└────────────────────────────────────────────┘
```

**File changes:**
- NEW: `orchestrator/prompt_history.py` (~200 lines)
- MODIFY: `orchestrator/engine.py` — add `PromptHistory` capture in `run_project()`, add `restore()`
- MODIFY: `orchestrator/state.py` — add `get_file_backup()` for manifest-based restore
- DEPENDS ON: Phase 1 (Checkpoints) for file snapshot infrastructure

#### Verification Gate

```bash
# Make 3 changes, restore to before change 2
# Verify: code is reverted, state is restored, budget rolled back
```

---

## Phase N3: Screenshot-Based Debugging

### Objective

Accept screenshots of errors or visual bugs, have the AI analyze them, identify root causes with file locations, and produce code fixes.

### Current State

- `CodebaseAnalyzer.debug(path, issue)` — accepts text issue description, scans codebase, produces fix
- `ReviewerAgent` — reviews code for bugs, security, performance
- No image/screenshot input capability
- `UnifiedClient.call()` supports streaming text but no multimodal (vision) models

### Implementation

#### N3.1 — Create `orchestrator/screenshot_debugger.py`

```python
@dataclass
class ScreenshotDiagnosis:
    issue_description: str
    root_cause: str
    affected_files: list[str]
    fix_code: dict[str, str]  # file_path → patched content
    confidence: float  # 0.0-1.0
    requires_manual_review: bool

class ScreenshotDebugger:
    """Analyzes screenshots of errors/bugs and produces fixes.

    Supports:
    - Terminal error screenshots (stack traces, build errors)
    - Browser preview screenshots (visual bugs, layout issues)
    - Console output (log errors)
    """

    def __init__(self, client: UnifiedClient, codebase_analyzer: CodebaseAnalyzer):
        self._client = client
        self._analyzer = codebase_analyzer

    async def diagnose(
        self,
        screenshot_base64: str,
        project_path: str,
        additional_context: str = "",
    ) -> ScreenshotDiagnosis:
        """Analyze screenshot and produce diagnosis with fix.

        Pipeline:
        1. Vision model reads screenshot → text description
        2. Text description + codebase → CodebaseAnalyzer.debug()
        3. Structured diagnosis with file locations and fix code
        """

    async def _screenshot_to_text(
        self, screenshot_base64: str
    ) -> str:
        """Convert screenshot to text using a vision-capable model."""
        # Use a multimodal model (GPT-5, Claude, or Gemini)
        # that accepts image_url content blocks
        ...

    async def produce_fix(
        self, diagnosis: ScreenshotDiagnosis
    ) -> dict[str, str]:
        """Generate code fix from diagnosis."""
        ...
```

**Pipeline:**
```
1. User pastes screenshot (terminal error, build failure, UI bug)
2. Vision model analyzes: "TypeError: cannot read property 'map' of undefined at auth.ts:42"
3. Text extracted → CodebaseAnalyzer.debug(path, "auth.ts:42 TypeError: ...")
4. LLM diagnoses root cause, finds affected files, generates fix
5. Fix applied, tests run, preview updated
```

**Model requirements:**
- Need a multimodal model (vision-capable) for screenshot analysis
- Fallback: if no vision model available, prompt user to describe the screenshot in text
- Current cheapest vision models: Gemini 2.5 Flash (~$0.10/img), GPT-5 (~$1.25/1M input)

#### N3.2 — Integration with CritiqueCycle

When `ScreenshotDebugger.diagnose()` identifies a bug, it feeds the issue directly into the `CritiqueCycle` as a revision prompt:
```python
revise_prompt = f"The following bug was found via screenshot analysis:\n\n"
revise_prompt += f"Error: {diagnosis.issue_description}\n"
revise_prompt += f"Root cause: {diagnosis.root_cause}\n"
revise_prompt += f"Fix: {diagnosis.fix_code}\n\n"
revise_prompt += f"Apply this fix and re-run tests."
```

**File changes:**
- NEW: `orchestrator/screenshot_debugger.py` (~300 lines)
- MODIFY: `orchestrator/api_clients.py` — add `call_vision()` method for multimodal models
- MODIFY: `orchestrator/codebase_analyzer.py` — accept `line_number` and `error_text` parameters
- MODIFY: `orchestrator/cli.py` — add `debug-screenshot` command

#### Verification Gate

```bash
# Take a screenshot of a Python traceback
# Run: python -m orchestrator debug-screenshot --project . --image error.png
# Expect: root cause identified, fix produced, tests pass after fix
```

---

## Phase N4: Brainstorming Mode for Plan

### Objective

Before decomposing a project into tasks, run a brainstorming pass where the AI asks clarifying questions about requirements, constraints, and trade-offs. The answers refine the plan before any code is generated.

### Current State

- `Orchestrator.dry_run()` — decomposes project and returns plan, no interactive Q&A
- `AgentOrchestrator._decompose_goal()` — keyword-based task decomposition
- `ArchitectureAnalyzer._prompt_llm_optimization()` — LLM reviews architecture and returns optimization suggestions
- No interactive clarifying-question loop between user and AI

### Implementation

#### N4.1 — Add `brainstorm()` to PlanOrchestrator (from Phase 2)

```python
class PlanOrchestrator:
    async def brainstorm(self, description: str) -> list[str]:
        """Ask clarifying questions instead of jumping to a plan.

        Returns a list of questions the AI has about the project.
        User answers feed back into the plan generation.
        """
        questions_prompt = (
            "You are a technical product manager. Read this project "
            "description and generate 3-5 clarifying questions that "
            "would help you create a better development plan. "
            "Focus on: architecture decisions, trade-offs, priorities, "
            "constraints, and edge cases.\n\n"
            f"PROJECT: {description}\n\n"
            "Return a JSON array of questions."
        )

        response = await self._client.call(
            model=self._cheapest_model,
            prompt=questions_prompt,
            max_tokens=500,
            temperature=0.7,  # Creative questions
        )

        questions = json.loads(response.text)
        return questions

    async def plan_with_answers(
        self,
        description: str,
        qa_pairs: dict[str, str],  # question → answer
    ) -> list[PlanTask]:
        """Generate plan incorporating the user's answers."""
        enriched = f"{description}\n\nAnswers to clarifying questions:\n"
        for q, a in qa_pairs.items():
            enriched += f"- Q: {q}\n  A: {a}\n"

        return await self.plan(enriched, criteria)
```

**Workflow:**
```
User: "Build a task management app"

AI: "Before I create a plan, a few questions:
      1. Should this be a web app, mobile app, or both?
      2. Do users need real-time collaboration (WebSockets) or is polling sufficient?
      3. Should data be stored locally, in SQLite, or in PostgreSQL?
      4. Do you need user authentication? If so, social login or email/password?"

User answers → AI generates a refined plan with answers incorporated

User: "1. Web only 2. Polling is fine 3. PostgreSQL 4. Email/password"

AI: "Got it. Here's the plan:
      Phase 1: PostgreSQL schema + FastAPI scaffold
      Phase 2: Email/password auth with JWT
      Phase 3: CRUD API for tasks
      Phase 4: Web UI with polling
      Estimated: $1.20, 4 tasks, ~8 minutes"
```

**File changes:**
- MODIFY: `orchestrator/plan_orchestrator.py` — add `brainstorm()`, `plan_with_answers()`
- MODIFY: `orchestrator/engine.py` — expose `brainstorm()` on `Orchestrator`
- MODIFY: `orchestrator/cli.py` — add `brainstorm` command

#### Verification Gate

```bash
python -m orchestrator brainstorm --project "Build a social media app"
# Expect: 3-5 clarifying questions
# Answer them, get refined plan
```

---

## Phase N5: Auto-Commit with Descriptive Messages

### Objective

After each task completes, automatically create a git commit with a high-quality, descriptive message (not just "Task auth.py completed").

### Current State

- `orchestrator/git_integration.py` — `GitIntegration` has milestone-level commits
- `orchestrator/milestones.py` — `OrchestrateAdapter` creates milestones with commit→branch→push
- Milestones are grouped batches of tasks, not per-task commits
- Commit messages are generic: "Milestone: auth"

### Implementation

#### N5.1 — Generate descriptive commit messages

```python
class GitIntegration:
    async def commit_task(
        self,
        task: Task,
        result: TaskResult,
        diff: str,
    ) -> CommitResult:
        """Commit a single task's changes with a descriptive message.

        Two strategies for message generation:
        1. Fast (free): Template-based from task metadata
        2. Quality (LLM): Ask cheapest model to summarize the diff
        """

    def _generate_message_fast(self, task: Task, diff: str) -> str:
        """Template-based commit message."""
        # Count changed files and line changes
        files = self._parse_diff_files(diff)
        additions = diff.count("+") - diff.count("+++ ")
        deletions = diff.count("-") - diff.count("--- ")

        # Build structured message
        scope = self._extract_scope(task.id)  # e.g., "auth", "api", "ui"
        emoji = self._task_type_emoji(task.type)

        message = f"{emoji} {scope}: {task.description[:80]}"

        if files:
            message += f"\n\nChanged: {', '.join(files[:5])}"
        if additions or deletions:
            message += f" (+{additions}, -{deletions})"

        # Score if available
        if result.score:
            message += f"\nScore: {result.score:.2f}"

        return message

    async def _generate_message_llm(
        self, diff: str, task: Task
    ) -> str:
        """Use cheapest model to summarize the diff."""
        prompt = (
            "Write a concise, descriptive git commit message for this diff. "
            "Follow Conventional Commits format (feat:, fix:, refactor:, etc).\n\n"
            f"Task: {task.description}\n"
            f"Diff ({len(diff)} chars):\n{diff[:2000]}"
        )

        response = await self._client.call(
            model=self._cheapest_model,
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
        )

        return response.text.strip()
```

**Example commit messages:**
```
✨ feat(auth): Add JWT authentication with bcrypt password hashing

Changed: src/auth.py, src/models/user.py, src/utils/jwt.py (+87, -3)
Score: 0.94

🐛 fix(api): Fix race condition in concurrent inventory updates

Changed: src/api/inventory.py (+12, -8)
Score: 0.89

♻️ refactor(db): Extract query builder to shared utility module

Changed: src/utils/query_builder.py, src/models/base.py (+45, -22)
Score: 0.96
```

**File changes:**
- MODIFY: `orchestrator/git_integration.py` — add `commit_task()`, `_generate_message_fast()`, `_generate_message_llm()`
- MODIFY: `orchestrator/milestones.py` — call `commit_task()` instead of milestone-level batch commits
- MODIFY: `orchestrator/engine.py` — trigger commit after each `_execute_task()`

#### Verification Gate

```bash
# Run a project with 3 tasks
# Check git log: 3 descriptive commits, not 1 milestone commit
```

---

## Phase N6: Two-Way Git Sync (Pull External Changes)

### Objective

When code is pushed to the connected repository from outside the orchestrator (e.g., a human developer's PR), pull those changes back into the orchestrator's state so the AI is aware of them.

### Current State

- `orchestrator/git_integration.py` — push-only: milestone-based branch → commit → push
- No pull mechanism
- `Orchestrator._resume_project()` — resumes from SQLite state, not from git
- No awareness of external changes

### Implementation

#### N6.1 — Add `pull_and_sync()` to GitIntegration

```python
class GitIntegration:
    async def pull_and_sync(self) -> list[dict]:
        """Pull external changes and report what changed.

        Returns:
            List of change summaries for the AI to be aware of:
            [{file: 'src/auth.py', change_type: 'modified', diff: '...'}, ...]
        """
        # Fetch latest from origin
        await self._run_git("fetch", "origin")

        # Get the diff between local and remote
        local_hash = await self._run_git("rev-parse", "HEAD")
        remote_hash = await self._run_git("rev-parse", "origin/main")

        if local_hash.strip() == remote_hash.strip():
            return []  # No changes

        # Get diff
        diff = await self._run_git("diff", f"{local_hash.strip()}..{remote_hash.strip()}")

        # Parse changed files
        changes = self._parse_diff_files(diff)

        # Pull the changes
        await self._run_git("pull", "origin", "main")

        return changes

    async def sync_and_inform_ai(self, orchestrator: Orchestrator) -> None:
        """Pull changes and inject them into the AI's context."""
        changes = await self.pull_and_sync()
        if not changes:
            return

        # Build context for the AI
        context = "External changes detected:\n"
        for change in changes:
            context += f"- {change['file']}: {change['type']}\n"
            context += f"  {change['diff'][:200]}...\n"

        # Inject into the orchestrator's context for the next task
        orchestrator.inject_context(context)

        logger.info(f"Synced {len(changes)} external changes back to orchestrator")
```

**Integration:**
```python
class Orchestrator:
    async def run_project(self, description: str, criteria: str):
        # Check for external changes before starting
        if self._git:
            await self._git.sync_and_inform_ai(self)

        # Proceed with normal execution
        ...
```

**File changes:**
- MODIFY: `orchestrator/git_integration.py` — add `pull_and_sync()`, `sync_and_inform_ai()`
- MODIFY: `orchestrator/engine.py` — pull check at project start, add `inject_context()`

#### Verification Gate

```bash
# Run a project, push a manual change via git push
# Start a new run — verify AI mentions the external change in its response
```

---

## Phase N7: Per-Prompt Cost Visibility

### Objective

Display the exact USD cost of each LLM call (not just aggregate budget tracking). Useful for debugging expensive calls and optimizing prompts.

### Current State

- `orchestrator/budget.py` — aggregate tracking: `charge(amount, compartment)`
- `orchestrator/services/observability.py` — per-model metrics (calls, errors, avg latency, total cost)
- `orchestrator/telemetry.py` — per-model but not per-call
- No per-call cost logging
- `BudgetEnforcer` checks total budget, not per-call cost

### Implementation

#### N7.1 — Add per-call logging to UnifiedClient

```python
class UnifiedClient:
    def __init__(self, ...):
        self._call_log: list[CallRecord] = []

    async def call(self, model: Model, prompt: str, **kwargs) -> APIResponse:
        response = await self._raw_call(model, prompt, **kwargs)

        # Log the call
        record = CallRecord(
            model=model,
            timestamp=time.time(),
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            prompt_preview=prompt[:200],
            response_preview=response.text[:200],
            task_id=kwargs.get("task_id"),
            event_type=kwargs.get("event_type", "unknown"),
        )
        self._call_log.append(record)

        # Log if unusually expensive
        if response.cost_usd > self.EXPENSIVE_CALL_THRESHOLD:
            logger.warning(
                f"EXPENSIVE CALL: {model.value} - ${response.cost_usd:.4f} "
                f"({response.input_tokens}+{response.output_tokens} tokens)"
            )

        return response

    def get_call_log(self) -> list[CallRecord]:
        """Return per-call cost log for dashboard display."""
        return self._call_log

    def cost_summary(self) -> dict:
        """Summarize costs by model and event_type."""
        summary = {}
        for record in self._call_log:
            key = record.model.value
            if key not in summary:
                summary[key] = {"calls": 0, "total_cost": 0, "total_tokens": 0}
            summary[key]["calls"] += 1
            summary[key]["total_cost"] += record.cost_usd
            summary[key]["total_tokens"] += record.input_tokens + record.output_tokens
        return summary
```

**Dashboard display:**
```
Recent LLM Calls                        Total: $1.2345
┌──────────────────────────────────────────────────────┐
│ #1  🟢 Claude Sonnet 4.6   $0.3211   4.2k tokens     │
│     decompose: "Build inventory API..."              │
│                                                      │
│ #2  🟢 Qwen 2.5 Coder       $0.0045   1.1k tokens    │
│     generate: "Create models.py..."                  │
│                                                      │
│ #3  🔴 DeepSeek R1 (retry)   $0.0890   3.4k tokens   │
│     critique: "Review auth.py..."                   │
│                                                      │
│ #4  🟢 Qwen 2.5 Coder       $0.0032   0.8k tokens    │
│     generate: "Fix auth.py tests..."               │
└──────────────────────────────────────────────────────┘
```

**File changes:**
- MODIFY: `orchestrator/api_clients.py` — add `CallRecord` dataclass, per-call logging, warning threshold
- MODIFY: `orchestrator/ide_backend/api/routes.py` — add `GET /api/calls` endpoint
- NEW: `ide_frontend/src/components/CallLog.tsx` — display recent calls with cost

#### Verification Gate

```bash
# Run a project, check /api/calls
# Expect: every LLM call logged with cost, tokens, and event_type
```

---

## Integration with Existing Plans

```
PhaseN1 (Ask Mode) ─────────────────────────────────────────────────────┐
    │  No dependencies — pure budget optimization                        │
Phase N2 (Restore Points) ───────────────────────────────────────────────┤
    │  Depends on Phase 1 (Checkpoints — file snapshot infrastructure)   │
Phase N3 (Screenshot Debugger) ──────────────────────────────────────────┤
    │  Depends on Phase 5 (Browser Testing — for browser screenshots)    │
Phase N4 (Brainstorming) ────────────────────────────────────────────────┤
    │  Depends on Phase 2 (Plan Workflow — PlanOrchestrator class)       │
Phase N5 (Auto Commits) ─────────────────────────────────────────────────┤
    │  Depends on Phase 1 (Checkpoints) or Phase 4 (Sandbox Tasks)       │
Phase N6 (Two-Way Git Sync) ─────────────────────────────────────────────┤
    │  Depends on Phase 5 (Browser Testing) for deployed app             │
Phase N7 (Per-Call Cost) ────────────────────────────────────────────────┘
    No dependencies — pure telemetry enhancement
```

## Newly-Specific Effort Estimate

| Phase | Feature | New Files | Modified Files | Est. Lines | Est. Days |
|-------|---------|-----------|---------------|------------|-----------|
| N1 | Ask Mode | 0 | 4 | ~100 | 1 |
| N2 | Chat Restore Points | 1 | 3 | ~200 | 1-2 |
| N3 | Screenshot Debugger | 1 | 3 | ~300 | 2-3 |
| N4 | Brainstorming Mode | 0 | 3 | ~120 | 2-3 |
| N5 | Auto Commit Messages | 0 | 3 | ~150 | 1 |
| N6 | Two-Way Git Sync | 0 | 2 | ~120 | 2-3 |
| N7 | Per-Call Cost Visibility | 0 | 3 | ~100 | 1 |
| **Newly Subtotal** | **7 phases** | **2** | **21** | **~1,090** | **10-14** |

## Combined Grand Total (All Four Sources)

| Source | Phases | New Files | Modified Files | Est. Days |
|--------|--------|-----------|---------------|-----------|
| Replit | 1-6 | 5 | 14 | 13-18 |
| Lovable | 7-10 | 4 | 10 | 9-13 |
| UI | U1-U7 | 20 | 15 | 17-23 |
| Newly | N1-N7 | 2 | 21 | 10-14 |
| **Grand Total** | **24** | **31** | **60** | **49-68** |
