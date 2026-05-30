# AI Orchestrator — Complete Enhancement Implementation Plan

**Date:** 2026-05-25
**Based on:** Cross-project analysis: Hermes Agent (v0.14.0) → AI Orchestrator (v6.0.0)
**Total scope:** 10 improvement vectors, 6 implementation phases, ~45-65 new files

---

## Phase Dependency Graph

```
Phase 1: Foundation (Tool Guardrails + Dependency Policy + Command Registry)
    ↓
Phase 2: Plugin Surfaces (Memory Provider ABCs + Context Engine ABCs)
    ↓
Phase 3: Memory & Context (Cross-Session Memory + Context Compression)
    ↓
Phase 4: Parallelism (Subagent Delegation)
    ↓
Phase 5: Learning (Pattern Learner / Closed Loop)
    ↓
Phase 6: Delivery (Gateway + Kanban Work Queue)
```

Each phase produces independently-usable features. Phases 1-2 are prerequisites for 3+.

---

## Phase 1 — Foundation (Week 1-2)

**Vectors:** Tool Guardrails (#5), Dependency Pinning Policy (#7), Slash Command Registry (#9)
**New files:** 5-7 | **Modified files:** 2-4 | **Lines:** ~500-700

### 1.1 Tool Guardrails (`orchestrator/tool_guardrails.py`)

**Design:**

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import hashlib

class GuardrailDecision(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"

@dataclass
class GuardrailResult:
    decision: GuardrailDecision
    reason: str = ""
    synthetic_output: str | None = None  # injected when BLOCKed

@dataclass
class ToolCallGuardrailController:
    """Per-turn guardrail controller. Reset at each iteration start."""
    
    _generation_hashes: set[str] = field(default_factory=set)
    _consecutive_empty: int = 0
    _last_output_size: int = 0
    _halt_decision: GuardrailResult | None = None
    
    def reset_for_turn(self) -> None:
        self._generation_hashes.clear()
        self._consecutive_empty = 0
        self._last_output_size = 0
        self._halt_decision = None
    
    def check(self, output: str, task_type: str) -> GuardrailResult:
        """Check generated output before validators."""
        # 1. Repeat detection
        h = hashlib.sha256(output[:500].encode()).hexdigest()
        if h in self._generation_hashes:
            return GuardrailResult(
                GuardrailDecision.WARN,
                "Output identical to previous generation — possible loop"
            )
        self._generation_hashes.add(h)
        
        # 2. Destructive pattern detection (code_gen only)
        if task_type == "code_generation":
            destructive = _check_destructive(output)
            if destructive:
                return GuardrailResult(GuardrailDecision.BLOCK, destructive)
        
        # 3. Output size regression
        if self._last_output_size > 0 and len(output) < self._last_output_size * 0.1:
            return GuardrailResult(
                GuardrailDecision.WARN,
                f"Output ({len(output)} chars) < 10% of previous ({self._last_output_size})"
            )
        self._last_output_size = len(output)
        
        return GuardrailResult(GuardrailDecision.ALLOW)
```

**Integration point:** In `engine.py._execute_task()`, call `self._tool_guardrails.check(output, task.type.value)` before validators. If BLOCKed, use synthetic result. If WARNed, log but continue.

**Existing infrastructure used:**
- `validators.py` — guardrails run BEFORE validation (gate concept from Hermes)
- `engine.py` — `_execute_task()` is the single integration point

### 1.2 Dependency Policy Document (`docs/DEPENDENCY_POLICY.md`)

Documents existing posture; no code changes. Covers:
- Exact-pin rationale (supply-chain attacks: litellm, mistralai)
- `requirements.txt` pin strategy
- Optional-dependency lazy-loading (future; see Phase 2 plugin discovery)
- Audit checklist for dependency updates

### 1.3 Slash Command Registry (`orchestrator/command_registry.py`)

**Design — inspired by Hermes's COMMAND_REGISTRY pattern:**

```python
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class CommandDef:
    name: str           # canonical name without slash
    description: str    # human-readable
    category: str       # "Project", "Configuration", "Info", "Exit"
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    cli_only: bool = False
    handler: Callable[..., Awaitable[str]] | None = None

# Single source of truth — drives CLI help, autocomplete, future gateway
COMMAND_REGISTRY: list[CommandDef] = [
    CommandDef("new", "Start a new project", "Project", aliases=("project",)),
    CommandDef("resume", "Resume a previous project", "Project"),
    CommandDef("analyze", "Analyze a codebase", "Project"),
    CommandDef("build", "Build an app from description", "Project"),
    CommandDef("model", "Show or change the active model", "Configuration"),
    CommandDef("budget", "Show budget status", "Configuration"),
    CommandDef("help", "Show available commands", "Info", aliases=("h", "?")),
    CommandDef("quit", "Exit the orchestrator", "Exit", aliases=("exit", "q")),
    # ... all existing CLI subcommands mapped here
]

def resolve_command(name: str) -> CommandDef | None:
    """Resolve canonical name or alias to CommandDef."""
    name = name.lstrip("/").lower()
    for cmd in COMMAND_REGISTRY:
        if cmd.name == name or name in cmd.aliases:
            return cmd
    return None

def commands_by_category() -> dict[str, list[CommandDef]]:
    """Group commands by category for help display."""
    result: dict[str, list[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        result.setdefault(cmd.category, []).append(cmd)
    return result
```

**Integration point:** `cli.py` imports `COMMAND_REGISTRY` for help text and autocomplete. Does NOT replace argparse — augments it with a declarative registry.

---

## Phase 2 — Plugin Surfaces (Week 2-3)

**Vectors:** Plugin System Depth (#6)
**New files:** 6-8 | **Modified files:** 2 | **Lines:** ~500-800

### 2.1 Memory Provider ABC (`orchestrator/plugins/memory_provider.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import TaskResult

@dataclass
class MemoryProviderMetadata:
    name: str
    version: str
    description: str
    author: str = ""

class MemoryProvider(ABC):
    """Pluggable memory backend. Implementations live in
    ~/.orchestrator/plugins/memory/<name>/__init__.py
    """
    
    metadata: MemoryProviderMetadata
    
    @abstractmethod
    async def initialize(self, hermes_home: Path) -> None: ...
    
    @abstractmethod
    async def shutdown(self) -> None: ...
    
    @abstractmethod
    async def sync_turn(self, task_id: str, result: TaskResult) -> None:
        """Called after each task completes. Provider persists the
        result in whatever form it chooses."""
        ...
    
    @abstractmethod
    async def prefetch(self, query: str) -> str:
        """Return relevant context for an upcoming task.
        Empty string = no context available."""
        ...
    
    async def post_setup(self, config: dict) -> None:
        """Optional: post-setup-wizard integration."""
        pass

class BuiltinMemoryProvider(MemoryProvider):
    """Default provider: uses existing telemetry_store + bm25_search."""
    
    def __init__(self):
        self.metadata = MemoryProviderMetadata(
            name="builtin",
            version="1.0.0",
            description="Built-in memory using telemetry_store + BM25 search",
        )
    
    async def initialize(self, hermes_home: Path) -> None:
        from ..telemetry_store import TelemetryStore
        from ..bm25_search import get_bm25_search
        self._store = TelemetryStore()
        self._bm25 = get_bm25_search()
    
    async def sync_turn(self, task_id: str, result: TaskResult) -> None:
        await self._store.record_routing_event(
            project_id="", task_id=task_id,
            task_type=TaskType(result.task_type), result=result,
        )
    
    async def prefetch(self, query: str) -> str:
        results = self._bm25.search(query, limit=3)
        return "\n".join(r["content"][:500] for r in results) if results else ""
```

### 2.2 Context Engine ABC (`orchestrator/plugins/context_provider.py`)

```python
class ContextProvider(ABC):
    """Pluggable context enrichment. Implementations live in
    ~/.orchestrator/plugins/context/<name>/__init__.py
    """
    
    @abstractmethod
    async def enrich(self, prompt: str, task_type: str, 
                     project_context: str) -> str:
        """Enrich the generation prompt with additional context."""
        ...
```

### 2.3 Plugin Discovery (`orchestrator/plugins/discovery.py`)

```python
async def discover_plugins(plugin_kind: str) -> list:
    """Discover plugins of a given kind from user + bundled locations.
    
    Scan order:
    1. Bundled: <repo>/orchestrator/plugins/<kind>/
    2. User: ~/.orchestrator/plugins/<kind>/
    3. pip entry points: orchestrator_<kind>
    
    User plugins override bundled of the same name (last-writer-wins).
    """
```

**Existing infrastructure used:**
- `plugins/base.py` — existing `PluginRegistry` handles general lifecycle
- `telemetry_store.py` — provides data for `BuiltinMemoryProvider`
- `bm25_search.py` — provides retrieval for `BuiltinMemoryProvider`

---

## Phase 3 — Memory & Context (Week 3-5)

**Vectors:** Cross-Session Memory (#4), Context Compression (#3)
**New files:** 8-12 | **Modified files:** 3-5 | **Lines:** ~700-1,200

### 3.1 Memory Manager (`orchestrator/memory/memory_manager.py`)

Orchestrates multiple `MemoryProvider` implementations. Runs periodic consolidation.

```python
class MemoryManager:
    """Orchestrates memory providers and periodic consolidation."""
    
    def __init__(self, providers: list[MemoryProvider] | None = None):
        self._providers = providers or [BuiltinMemoryProvider()]
        self._nudge_interval = 5  # projects between consolidations
        self._projects_since_consolidation = 0
    
    async def sync_turn(self, task_id: str, result: TaskResult) -> None:
        """Fire-and-forget: sync to all providers after task completion."""
        tasks = [p.sync_turn(task_id, result) for p in self._providers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def prefetch_all(self, query: str) -> str:
        """Aggregate context from all providers."""
        results = await asyncio.gather(
            *(p.prefetch(query) for p in self._providers),
            return_exceptions=True,
        )
        return "\n\n".join(r for r in results if isinstance(r, str) and r)
    
    async def maybe_consolidate(self) -> None:
        """Trigger cross-project insight extraction after N projects."""
        self._projects_since_consolidation += 1
        if self._projects_since_consolidation >= self._nudge_interval:
            await self._consolidate()
            self._projects_since_consolidation = 0
    
    async def _consolidate(self) -> None:
        """Use a cheap model to extract cross-project insights from
        telemetry_store, producing a consolidation summary stored as
        a special 'insight' routing_event."""
```

**Integration:** `engine.py.run_project()` calls `self.memory_manager.maybe_consolidate()` on project completion (after `post_project` hooks).

### 3.2 Context Compressor (`orchestrator/context_compressor.py`)

LLM-powered summarization for dependency context that exceeds the truncation limit.

```python
from hashlib import sha256
from ..api_clients import UnifiedClient
from ..models import Model, TaskType

class ContextCompressor:
    """Compresses dependency context via LLM summarization when it exceeds
    the truncation limit. Falls back to hard truncation on failure."""
    
    def __init__(self, 
                 client: UnifiedClient,
                 enabled: bool = True,
                 cache_ttl_hours: int = 48):
        self._client = client
        self._enabled = enabled
        self._cache: dict[str, str] = {}  # content_hash → summary
    
    async def compress(self, text: str, max_chars: int,
                       preserve_signatures: bool = True) -> str:
        """Compress text to fit within max_chars. If within limit, return
        as-is. If compression disabled or fails, hard-truncate."""
        
        if len(text) <= max_chars:
            return text
        
        if not self._enabled:
            return text[:max_chars] + "\n\n[... truncated ...]"
        
        # Check cache
        h = sha256(text.encode()).hexdigest()
        if h in self._cache:
            return self._cache[h]
        
        try:
            summary = await self._summarize(text, max_chars, preserve_signatures)
            self._cache[h] = summary
            return summary
        except Exception:
            logger.warning("Context compression failed, falling back to truncation")
            return text[:max_chars] + "\n\n[... truncated ...]"
    
    async def _summarize(self, text: str, max_chars: int,
                         preserve_signatures: bool) -> str:
        """Use cheapest available model (Phi-4 / GLM-4.7-Flash) to
        summarise while preserving function/class signatures."""
        instruction = (
            "Summarise the following code while preserving ALL function "
            "signatures, class definitions, and import statements. "
            "Remove implementation bodies (replace with '# ...'). "
            "Keep variable names and type annotations. "
            f"Output must be under {max_chars} characters."
        ) if preserve_signatures else (
            f"Summarise the following text to under {max_chars} characters."
        )
        # Use cheapest model
        model = Model.ZHIPU_GLM_4_7_FLASH  # $0.06/1M input
        response = await self._client.call(
            model=model,
            system=instruction,
            prompt=text[:8000],  # don't send full text to summarizer
            max_tokens=2048,
        )
        return response.text[:max_chars]
```

**Integration:** Replace hard truncation in `dependency_resolver.py._format_dependency_context()` with `compressor.compress()` call. The DependencyResolver receives the compressor via constructor injection.

**Existing infrastructure used:**
- `UnifiedClient` — existing API client for LLM calls
- `DependencyResolver` — existing class with truncation logic
- `Model.ZHIPU_GLM_4_7_FLASH` — cheapest available model, already in routing tables

### 3.3 Memory Consolidation Loop (`orchestrator/memory/consolidation.py`)

```python
class ConsolidationLoop:
    """After N projects, uses LLM to extract cross-project insights."""
    
    async def run(self, store: TelemetryStore, 
                  min_projects: int = 5) -> list[ConsolidationInsight]:
        """Query telemetry_store for patterns across projects:
        - Which task types consistently fail on which models?
        - Which prompt patterns produce highest-quality outputs?
        - Any recurring validation failure patterns?
        
        Returns structured insights stored as routing_events
        with task_type="consolidation_insight".
        """
```

---

## Phase 4 — Parallelism (Week 5-6)

**Vectors:** Subagent Delegation (#2)
**New files:** 5-7 | **Modified files:** 3-4 | **Lines:** ~500-800

### 4.1 SubAgent (`orchestrator/delegation/subagent.py`)

Lightweight AIAgent wrapper with isolated context and iteration budget.

```python
from dataclasses import dataclass
from ..models import Task, TaskResult, TaskType
from ..budget import Budget

@dataclass
class SubAgentConfig:
    max_iterations: int = 90
    role: str = "leaf"  # "leaf" | "orchestrator"
    enabled_toolsets: list[str] | None = None
    inherit_memory: bool = True

class SubAgent:
    """Lightweight isolated agent for parallel task execution.
    
    Leaf agents cannot delegate tasks further. Orchestrator agents
    can (up to max_spawn_depth)."""
    
    def __init__(self, config: SubAgentConfig, budget_slice: Budget):
        self.config = config
        self.budget = budget_slice
        self._orch = None  # lazily created Orchestrator instance
    
    async def execute(self, task: Task, dependency_context: str = "") -> TaskResult:
        """Execute a single task in an isolated context."""
        # Create lightweight orchestrator with budget slice
        # (reuses existing Orchestrator.__init__ infrastructure)
        ...
    
    @property
    def is_leaf(self) -> bool:
        return self.config.role == "leaf"
```

### 4.2 Batch Runner (`orchestrator/delegation/batch_runner.py`)

```python
class BatchRunner:
    """Spawns N subagents for independent tasks, gathers results."""
    
    def __init__(self, max_concurrent: int = 3, max_depth: int = 2):
        self._max_concurrent = max_concurrent
        self._max_depth = max_depth
        self._current_depth = 0
    
    async def run_batch(self, tasks: list[Task],
                        dependency_contexts: dict[str, str],
                        parent_budget: Budget) -> dict[str, TaskResult]:
        """Execute multiple independent leaf tasks concurrently.
        Each gets a proportional budget slice from parent.
        Returns {task_id: TaskResult}."""
        budget_per_task = parent_budget.max_usd / len(tasks) * 0.8  # 80% split
        ...
```

### 4.3 Dependency Level Parallelism (`orchestrator/delegation/integration.py`)

**Integration:** Modify `engine.py.run_project()` to execute independent tasks within a dependency level in parallel using `BatchRunner`:

```python
# Current (serial):
for task_id in execution_order:
    result = await self._execute_task(tasks[task_id])

# New (parallel within levels):
for level in levels:  # tasks at same topological level
    if len(level) > 1 and self._batch_runner:
        # Parallel execution for independent tasks
        results = await self._batch_runner.run_batch(
            [tasks[tid] for tid in level], 
            dependency_contexts, self.budget
        )
    else:
        # Serial execution for single-task levels
        result = await self._execute_task(tasks[level[0]])
```

**Existing infrastructure used:**
- `concurrency_controller.py` — existing `TaskConcurrencyGuard`
- `engine_core/dependency_resolver.py` — topological sort provides levels
- `budget.py` — `reserve/commit/release` pattern for budget slicing

---

## Phase 5 — Learning (Week 6-8)

**Vectors:** Pattern Learner / Closed Learning Loop (#1)
**New files:** 10-14 | **Modified files:** 4-6 | **Lines:** ~800-1,200

### 5.1 Pattern Extractor (`orchestrator/pattern_learner/extractor.py`)

```python
from dataclasses import dataclass
from hashlib import sha256
from ..models import TaskResult, TaskType

@dataclass
class ExtractedPattern:
    pattern_id: str          # hash-based deterministic ID
    task_type: TaskType
    prompt_fingerprint: str  # hash of normalized prompt
    generated_code_hash: str # hash of output (for dedup)
    quality_score: float
    model_used: str
    validation_results: dict[str, bool]
    created_at: float
    provenance: str          # "agent" or "bundled"

class PatternExtractor:
    """Analyzes TaskResult history to extract successful patterns.
    
    A "successful pattern" is defined as:
    - Quality score ≥ 0.85 (above acceptance threshold)
    - All deterministic validators passed
    - At least 2 iterations (critique→revise cycle was productive)
    - Not a duplicate of an existing pattern (by code hash)
    """
    
    async def extract(self, task_type: TaskType,
                      result: TaskResult,
                      prompt: str) -> ExtractedPattern | None:
        """Extract a pattern from a successful task result.
        Returns None if the result doesn't meet quality thresholds."""
        if result.score < 0.85 or not result.deterministic_check_passed:
            return None
        if result.iterations < 2:
            return None
        
        # Check for duplicates via code hash
        code_hash = sha256(result.output.encode()).hexdigest()
        
        return ExtractedPattern(
            pattern_id=f"pat_{task_type.value}_{sha256(prompt.encode()).hexdigest()[:12]}",
            task_type=task_type,
            prompt_fingerprint=sha256(self._normalize_prompt(prompt).encode()).hexdigest(),
            generated_code_hash=code_hash,
            quality_score=result.score,
            model_used=result.model_used.value,
            validation_results={"syntax": True, "tests": True},  # from result
            created_at=time.time(),
            provenance="agent",
        )
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for fingerprinting: lowercase, strip whitespace,
        remove variable names that vary between runs."""
        ...
```

### 5.2 Pattern Store (`orchestrator/pattern_learner/pattern_store.py`)

SQLite-backed pattern library, modeled after `TelemetryStore` pattern.

```python
class PatternStore:
    """Persistent pattern library with usage tracking."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        pattern_id       TEXT PRIMARY KEY,
        task_type        TEXT NOT NULL,
        prompt_fingerprint TEXT NOT NULL,
        generated_code_hash TEXT NOT NULL,
        quality_score    REAL NOT NULL,
        model_used       TEXT NOT NULL,
        provenance       TEXT NOT NULL DEFAULT 'agent',
        status           TEXT NOT NULL DEFAULT 'active',
        reuse_count      INTEGER NOT NULL DEFAULT 0,
        last_reused_at   REAL,
        avg_score_on_reuse REAL,
        created_at       REAL NOT NULL,
        archived_at      REAL
    );
    
    CREATE TABLE IF NOT EXISTS pattern_artifacts (
        pattern_id       TEXT NOT NULL REFERENCES patterns(pattern_id),
        prompt_text      TEXT NOT NULL,
        generated_code   TEXT NOT NULL,
        critique_text    TEXT,
        revision_context TEXT
    );
    """
    
    async def insert(self, pattern: ExtractedPattern,
                     prompt: str, code: str, critique: str = "") -> None:
        """Insert new pattern. Ignored if code_hash already exists."""
        ...
    
    async def find_similar(self, task_type: TaskType,
                           prompt: str, limit: int = 3) -> list[dict]:
        """Find patterns with similar prompts (by fingerprint prefix match)."""
        ...
    
    async def record_reuse(self, pattern_id: str, score: float) -> None:
        """Update reuse_count, last_reused_at, avg_score_on_reuse."""
        ...
    
    async def archive_stale(self, days: int = 30) -> int:
        """Archive patterns unused for N days. Returns count archived.
        Never deletes — sets status='archived' + archived_at timestamp."""
        ...
```

### 5.3 Pattern Curator (`orchestrator/pattern_learner/curator.py`)

```python
class PatternCurator:
    """Periodic review of agent-created patterns. Follows Hermes's
    curator invariants:
    - Only touches provenance="agent" patterns
    - Never deletes — archives to status='archived'
    - Pinned patterns (status='pinned') are exempt from auto-transitions
    - Runs after every N projects (configurable)"""
    
    def __init__(self, store: PatternStore, review_model: Model = Model.PHI_4):
        self._store = store
        self._review_model = review_model
        self._interval = 10  # projects between reviews
    
    async def review(self) -> list[str]:
        """Review stale patterns, merge duplicates, archive unused.
        Returns list of actions taken (for logging)."""
        ...
```

### 5.4 Pattern Injector (`orchestrator/pattern_learner/injector.py`)

```python
class PatternInjector:
    """At task generation time, injects relevant past patterns into the
    prompt context. Ephemeral (not persisted to project state).
    Configurable via ORCH_PATTERN_INJECTION=true (default off)."""
    
    def __init__(self, store: PatternStore, enabled: bool = False):
        self._store = store
        self._enabled = enabled
    
    async def inject(self, task_type: TaskType, prompt: str,
                     max_patterns: int = 2) -> str:
        """Find similar successful patterns and inject them as
        few-shot examples into the prompt."""
        if not self._enabled:
            return prompt
        
        patterns = await self._store.find_similar(task_type, prompt, max_patterns)
        if not patterns:
            return prompt
        
        injection = "\n\n## Reference: Successful Patterns from Prior Runs\n\n"
        for i, pat in enumerate(patterns, 1):
            injection += (
                f"### Pattern {i} (score: {pat['quality_score']:.2f}, "
                f"reused {pat['reuse_count']}×)\n"
                f"```\n{pat['generated_code'][:1500]}\n```\n\n"
            )
        return prompt + injection
```

**Integration point:** `engine.py._execute_task()` calls `injector.inject()` before LLM generation, after model selection but before API call. Injected patterns are added to the task's `context` field (ephemeral, not persisted).

**Existing infrastructure used:**
- `telemetry_store.py` — existing pattern for SQLite-backed append-only store
- `state.py` — existing pattern for schema migration
- `TaskResult` — existing data model with score, attempt_history, validators
- `UnifiedClient` — for curator's LLM review calls

---

## Phase 6 — Delivery (Week 8-10)

**Vectors:** Multi-Platform Gateway (#8), Kanban Work Queue (#10)
**New files:** 15-20 | **Modified files:** 5-8 | **Lines:** ~1,200-2,000

### 6.1 Gateway Core (`orchestrator/gateway/run.py`)

```python
class OrchestratorGateway:
    """Multi-platform gateway for the AI Orchestrator.
    
    Accepts project specs via messaging platforms, reports progress,
    and allows approval/denial of expensive model choices."""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.platforms: dict[str, PlatformAdapter] = {}
        self._active_sessions: dict[str, Orchestrator] = {}
    
    async def start(self) -> None:
        """Start all configured platform adapters."""
        for name, adapter_config in self.config.platforms.items():
            adapter = PLATFORM_REGISTRY[name](adapter_config)
            await adapter.connect()
            self.platforms[name] = adapter
    
    async def handle_message(self, platform: str, user_id: str,
                             text: str) -> str:
        """Route incoming message to appropriate handler."""
        if text.startswith("/"):
            return await self._handle_command(platform, user_id, text)
        return await self._handle_project_spec(platform, user_id, text)
```

### 6.2 Platform Adapters (minimal — Telegram + Webhook only)

```python
# orchestrator/gateway/platforms/base.py
class PlatformAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def send_message(self, user_id: str, text: str) -> None: ...

# orchestrator/gateway/platforms/telegram.py
class TelegramAdapter(PlatformAdapter): ...
# orchestrator/gateway/platforms/webhook.py  
class WebhookAdapter(PlatformAdapter): ...
```

### 6.3 Gateway Session Management (`orchestrator/gateway/session.py`)

Tracks active project sessions per user, handles timeout/cleanup.

### 6.4 Kanban Work Queue (`orchestrator/kanban/`)

SQLite-backed persistent work queue for long-running orchestration projects.

```python
# orchestrator/kanban/board.py
class KanbanBoard:
    """SQLite-backed multi-project work queue."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS kanban_tasks (
        task_id      TEXT PRIMARY KEY,
        project_spec TEXT NOT NULL,   -- JSON: {description, criteria, budget}
        status       TEXT NOT NULL DEFAULT 'todo',
        assignee     TEXT,            -- worker profile name
        created_at   REAL NOT NULL,
        claimed_at   REAL,
        completed_at REAL,
        priority     INTEGER NOT NULL DEFAULT 0
    );
    """
    
    async def enqueue(self, spec: dict) -> str: ...
    async def claim_next(self, assignee: str) -> dict | None: ...
    async def complete(self, task_id: str, result: dict) -> None: ...
    async def fail(self, task_id: str, error: str) -> None: ...

# orchestrator/kanban/dispatcher.py
class KanbanDispatcher:
    """Long-lived loop that monitors the board and spawns workers."""
    async def run(self, board: KanbanBoard, worker_fn: Callable) -> None: ...
```

**Integration:** `cli.py` adds `hermes gateway` and `hermes kanban` subcommands. Gateway runs in background; Kanban dispatcher runs alongside.

**Existing infrastructure used:**
- `state.py` / `StateManager` — existing SQLite pattern for persistence
- `command_registry.py` — from Phase 1, drives gateway slash commands
- `cli.py` — existing argparse subcommand structure

---

## Summary: File Manifest

| Phase | New Files | Modified Files | Lines | Key Deliverable |
|-------|-----------|---------------|-------|-----------------|
| P1: Foundation | 5-7 | 2-4 | 500-700 | Guardrails + Command Registry + Policy Doc |
| P2: Plugin Surfaces | 6-8 | 2 | 500-800 | MemoryProvider ABC + ContextProvider ABC + Discovery |
| P3: Memory & Context | 8-12 | 3-5 | 700-1,200 | MemoryManager + ContextCompressor + ConsolidationLoop |
| P4: Parallelism | 5-7 | 3-4 | 500-800 | SubAgent + BatchRunner + Level parallelism |
| P5: Learning | 10-14 | 4-6 | 800-1,200 | PatternExtractor + PatternStore + Curator + Injector |
| P6: Delivery | 15-20 | 5-8 | 1,200-2,000 | Gateway + Telegram/Webhook adapters + Kanban |
| **Total** | **49-68** | **19-29** | **4,200-6,700** | |

---

## Integration Map: How Each Phase Connects to Existing Code

```
Existing Infrastructure          →  New Component

engine.py._execute_task()        →  tool_guardrails.py (Phase 1)
engine.py.run_project()          →  memory_manager.py (Phase 3)
engine.py._execute_task()        →  pattern_learner/injector.py (Phase 5)
engine.py.run_project()          →  delegation/batch_runner.py (Phase 4)

dependency_resolver.py           →  context_compressor.py (Phase 3)
dependency_resolver.py           →  delegation/integration.py (Phase 4)

plugins/base.py                  →  plugins/memory_provider.py (Phase 2)
plugins/base.py                  →  plugins/context_provider.py (Phase 2)
plugins/base.py                  →  plugins/discovery.py (Phase 2)

telemetry_store.py               →  memory/builtin_provider.py (Phase 3)
telemetry_store.py               →  pattern_learner/pattern_store.py (Phase 5)

state.py (StateManager)          →  kanban/board.py (Phase 6)
state.py (StateManager)          →  gateway/session.py (Phase 6)

cli.py (argparse)                →  command_registry.py (Phase 1)
cli.py (subcommands)             →  gateway/run.py (Phase 6)
cli.py (subcommands)             →  kanban/dispatcher.py (Phase 6)

hooks.py (HookRegistry)          →  tool_guardrails.py hooks (Phase 1)
unified_events/core.py           →  consolidation.py events (Phase 3)

budget.py (reserve/commit)       →  delegation/batch_runner.py (Phase 4)

concurrency_controller.py        →  delegation/batch_runner.py (Phase 4)

UnifiedClient (api_clients.py)   →  context_compressor.py (Phase 3)
UnifiedClient (api_clients.py)   →  pattern_learner/curator.py (Phase 5)

models.py (Task, TaskResult)     →  pattern_learner/extractor.py (Phase 5)
models.py (Model enum)           →  context_compressor.py (Phase 3)
```

---

## Testing Strategy

| Phase | Test Type | Test Files | Key Scenarios |
|-------|-----------|-----------|---------------|
| P1 | Unit + Integration | 3-4 | Guardrail repeat-detection, destructive-pattern matching, command resolution with aliases |
| P2 | Unit | 2-3 | MemoryProvider discovery, BuiltinMemoryProvider prefetch, ContextProvider enrichment |
| P3 | Unit + Integration | 4-5 | Compression preserves signatures, MemoryManager consolidation cycle, empty-corpus edge cases |
| P4 | Integration | 3-4 | BatchRunner parallel execution, budget slicing, depth tracking, failure isolation |
| P5 | Unit + Integration | 5-6 | Pattern extraction thresholds, duplicate detection, curator archive/restore, injector output format |
| P6 | Integration + Smoke | 5-7 | Gateway message routing, Telegram adapter connect/disconnect, Kanban claim/complete cycle |

**All tests follow existing patterns:**
- `pytest-asyncio` for async tests (consistent with codebase)
- `conftest.py` fixtures for temporary DB paths (PatternStore, MemoryManager)
- Mock `UnifiedClient` for LLM-call-heavy tests (curator, compressor)
- `@pytest.mark.slow` for consolidation/curator review tests

---

## Rollout Strategy

1. **Phase 1 ships as v6.1** — Tool guardrails immediately prevent runaway generations; command registry unifies CLI help. Zero breaking changes.

2. **Phase 2 ships as v6.2** — Plugin ABCs are additive; existing plugins continue working. Memory provider registration is opt-in.

3. **Phase 3 ships as v6.3** — Context compression is opt-in (`ORCH_CONTEXT_COMPRESSION=true`). Memory consolidation runs silently in background. Non-breaking.

4. **Phase 4 ships as v6.4** — Subagent parallelism is opt-in (`ORCH_BATCH_PARALLELISM=true`, default off for backward compatibility). Existing serial execution is preserved.

5. **Phase 5 ships as v6.5** — Pattern injection is opt-in (`ORCH_PATTERN_INJECTION=true`). Patterns are stored but not injected by default — users explicitly enable the learning loop.

6. **Phase 6 ships as v7.0** — Gateway and Kanban are major new features. Gateway is opt-in (must explicitly run `orchestrator gateway start`). Kanban is opt-in.
