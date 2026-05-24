# Agentic System Implementation Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0 — Agentic Transformation  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The AI Orchestrator currently operates as a **single-agent pipeline**: decompose spec → route task → generate → critique → revise → evaluate → deliver. To become an **agentic system** that can cooperatively develop any kind of application, it needs a **multi-agent coordination architecture** — specialized agents collaborating through a shared workspace, with recursive goal decomposition, tool access, and a self-improving feedback loop.

This plan defines 10 new capabilities across 46 days of implementation. The **minimal viable agentic system** (MVAS) requires the first 5 capabilities (~27 days). Once those exist, every subsequent enhancement compounds the system's effectiveness.

---

## Current vs Target Architecture

### Today (Single-Agent Pipeline)

```
User spec → Decompose → Task queue → Route → Generate → Critique → Revise → Validate → Output
  │
  └─ Single Orchestrator instance does everything sequentially
```

### Target (Multi-Agent Cooperative System)

```
User goal: "Build a social media app"
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  AgentOrchestrator (coordinator)                     │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐  │
│  │Architect│  │Developer│  │Reviewer │  │  Tester  │  │
│  │(design) │  │(write)  │  │(audit)  │  │(verify)  │  │
│  └────┬────┘  └───┬─────┘  └────┬────┘  └─────┬─────┘  │
│       │           │             │              │        │
│       └───────────┴──────┬──────┴──────────────┘        │
│                          │                               │
│                   ┌──────▼──────┐                        │
│                   │  Workspace   │  (shared blackboard)  │
│                   │  - codebase  │                        │
│                   │  - decisions │                        │
│                   │  - test logs │                        │
│                   │  - knowledge │                        │
│                   └──────┬──────┘                        │
│                          │                               │
│                   ┌──────▼──────┐                        │
│                   │    Tools     │                        │
│                   │  shell, git, │                        │
│                   │  pkg, build, │                        │
│                   │  test, search│                        │
│                   └─────────────┘                        │
└──────────────────────────────────────────────────────┘
  │
  ▼
Deliverable: working, tested, deployed application
```

---

## Capability 1: Multi-Agent Architecture

**Status:** ❌ Does not exist. Currently a single `Orchestrator` monolith.  
**Target:** Specialized agents with roles, tools, and inter-agent communication.  
**New package:** `orchestrator/agents/`  
**Effort:** 8 days | **Priority:** P0 (blocking)

### Architecture

```
orchestrator/agents/
  ├── __init__.py          # Public API
  ├── base.py              # AgentBase abstract class
  ├── coordinator.py       # AgentOrchestrator (plan → dispatch → monitor)
  ├── architect.py         # ArchitectAgent (design decisions, tech stack)
  ├── developer.py         # DeveloperAgent (write/modify code)
  ├── reviewer.py          # ReviewerAgent (code review, security audit)
  ├── tester.py            # TesterAgent (generate + run tests)
  ├── devops.py            # DevOpsAgent (build, deploy, CI)
  ├── researcher.py        # ResearcherAgent (web search, docs lookup)
  └── agent_registry.py    # Agent registry and capability discovery
```

### AgentBase

```python
class AgentBase(ABC):
    """Base class for all specialized agents.

    Each agent has:
    - A role description (system prompt)
    - A set of tools it can use
    - Model preferences for different sub-tasks
    - Access to the shared workspace
    - A message inbox for inter-agent communication
    """

    def __init__(
        self,
        role: AgentRole,
        tools: list[Tool],
        model_preferences: dict[TaskType, Model],
        workspace: ProjectWorkspace,
        client: UnifiedClient,
        message_bus: AgentMessageBus,
    ):
        self.role = role
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.model_preferences = model_preferences
        self.workspace = workspace
        self.client = client
        self.inbox: list[AgentMessage] = []

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's core identity and instructions."""

    @abstractmethod
    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Execute an assigned task.

        The agent:
        1. Reads relevant workspace state
        2. Reasons about the task using its system prompt
        3. Invokes tools as needed
        4. Writes results back to the workspace
        5. Returns a structured result
        """

    async def send_message(self, recipient: str, content: str, msg_type: MessageType):
        """Send a structured message to another agent."""

    async def read_inbox(self) -> list[AgentMessage]:
        """Read pending messages."""
```

### AgentOrchestrator

```python
class AgentOrchestrator:
    """Coordinates multiple specialized agents toward a shared goal.

    Responsibilities:
    1. Receive a high-level goal
    2. Decompose into sub-goals (via HTN planning)
    3. Dispatch sub-goals to appropriate agents
    4. Monitor progress via workspace state
    5. Resolve conflicts between agents
    6. Report final results
    """

    def __init__(self, agents: dict[AgentRole, AgentBase], workspace: ProjectWorkspace):
        self.agents = agents
        self.workspace = workspace
        self.goal_decomposer = GoalDecomposer()

    async def execute_goal(self, goal: str) -> ExecutionResult:
        # 1. Plan: decompose goal into agent tasks
        plan = await self.goal_decomposer.decompose(goal, self.workspace)

        # 2. Dispatch: route tasks to agents
        for agent_task in plan.tasks:
            agent = self.agents[agent_task.target_role]
            result = await agent.handle_task(agent_task)
            self.workspace.record_task_result(result)

            # 3. Adapt: if any task fails, replan
            if result.status == TaskStatus.FAILED:
                plan = await self.goal_decomposer.replan(goal, self.workspace)
```

---

## Capability 2: Shared Workspace (Blackboard Architecture)

**Status:** ❌ Does not exist. Tasks execute in isolation.  
**Target:** Shared state where agents post findings, read each other's work, and build on previous results.  
**New package:** `orchestrator/workspace/`  
**Effort:** 5 days | **Priority:** P0

### Architecture

```
orchestrator/workspace/
  ├── __init__.py
  ├── workspace.py        # ProjectWorkspace (central blackboard)
  ├── file_version.py     # FileVersion tracking (snapshots)
  ├── decision_log.py     # ArchitectureDecision log
  ├── test_log.py         # TestRun history
  ├── message_bus.py      # AgentMessageBus (publish/subscribe)
  └── knowledge_graph.py  # KnowledgeGraph (learned patterns)
```

### Data Model

```python
@dataclass
class ProjectWorkspace:
    """Central blackboard shared by all agents."""

    # Codebase state
    codebase: dict[str, FileVersion]  # path → versioned file history
    git_state: GitState               # current branch, HEAD, uncommitted changes

    # Decision history
    architecture_decisions: list[ArchitectureDecision]
    technology_choices: dict[str, str]  # "database" → "PostgreSQL"
    trade_off_log: list[TradeOffRecord]

    # Execution state
    task_queue: list[AgentTask]
    completed_tasks: dict[str, AgentTaskResult]
    active_agents: set[AgentRole]

    # Quality state
    test_results: dict[str, TestRun]
    lint_results: dict[str, LintOutput]
    build_artifacts: list[BuildArtifact]
    security_audits: list[SecurityAudit]

    # Learning state
    knowledge_graph: KnowledgeGraph
    pattern_memory: ExperienceBuffer
    strategy_effectiveness: dict[str, float]  # "CoVE on CODE_GEN" → 0.15 score improvement
```

### Key Behaviors

- **Conflict detection:** If Agent A modifies `auth.py` while Agent B also modifies it, the workspace detects the conflict and alerts the coordinator.
- **Precondition checking:** Agents can register interest in workspace state changes. When `test_results[module]` updates, the TesterAgent can trigger.
- **Audit trail:** Every workspace mutation is timestamped and attributed to the agent that made it. Replayable.

---

## Capability 3: Recursive Goal Decomposition (HTN Planning)

**Status:** Partial. `Decomposer.decompose()` does single-pass JSON-to-task-list.  
**Target:** Hierarchical Task Network (HTN) planning — break goals into sub-goals recursively.  
**New package:** `orchestrator/planning/`  
**Effort:** 5 days | **Priority:** P0

### Architecture

```
orchestrator/planning/
  ├── __init__.py
  ├── goal.py             # Goal, SubGoal, Plan dataclasses
  ├── decomposer.py       # GoalDecomposer (HTN recursive)
  ├── dependency_graph.py # DependencyResolver (DAG construction)
  ├── cost_estimator.py   # CostEstimator (budget per sub-goal)
  └── strategy.py         # PlanningStrategy (which ARA method per level)
```

### Recursive Decomposition

```python
class GoalDecomposer:
    """Recursive HTN planner that breaks goals into atomic actions.

    Uses different ARA methods at each decomposition level:
    - Top-level: Multi-Perspective (architecture)
    - Mid-level: SoT (skeleton-of-thought for sub-problems)
    - Leaf-level: PersuasionDefense (verification before execution)
    """

    async def decompose(
        self, goal: str, workspace: ProjectWorkspace, depth: int = 0
    ) -> Plan:
        if depth > MAX_DEPTH:  # Safety limit
            return self._atomic_task(goal)

        # Use ARA method appropriate for this depth
        method = self._select_method(depth, goal)
        result = await self._ara.execute(goal, workspace, method)

        if result.is_atomic:
            return Plan(tasks=[self._to_agent_task(result)])

        # Recursive: decompose each sub-goal
        sub_plans = []
        for sub_goal in result.sub_goals:
            sub_plan = await self.decompose(sub_goal, workspace, depth + 1)
            sub_plans.append(sub_plan)

        return Plan.merge(sub_plans, dependencies=self._resolve_deps(sub_plans))
```

### Planning Levels

| Depth | Name | ARA Method | Example |
|-------|------|-----------|---------|
| 0 | **Architecture** | Multi-Perspective | "Which framework for this app?" |
| 1 | **Sub-system** | SoT | "Authentication system → login, registration, JWT" |
| 2 | **Component** | Deliberative (default) | "Login endpoint → route, handler, validator" |
| 3 | **Atomic action** | PersuasionDefense (verify) | "Write login.py" |

---

## Capability 4: Tool Integration Layer

**Status:** Partial. CodebaseReader/Writer exist. No shell, git, package manager, or test runner tools.  
**Target:** Standardized tool interface that every agent can use.  
**New package:** `orchestrator/tools/`  
**Effort:** 6 days | **Priority:** P0

### Architecture

```
orchestrator/tools/
  ├── __init__.py
  ├── base.py             # Tool, ToolResult, ToolPermission
  ├── file_tool.py        # FileReadTool, FileWriteTool, FileDeleteTool
  ├── shell_tool.py       # ShellTool (subprocess in venv)
  ├── git_tool.py         # GitTool (branch, commit, diff, log, PR)
  ├── package_tool.py     # PackageTool (pip, npm, cargo, go get)
  ├── test_tool.py        # TestTool (run tests, parse output)
  ├── build_tool.py       # BuildTool (compile, bundle)
  ├── search_tool.py      # WebSearchTool (Nexus Search wrapper)
  ├── db_tool.py          # DatabaseTool (schema design, migrations)
  └── tool_registry.py    # ToolRegistry (discovery + permissions)
```

### Tool Interface

```python
@dataclass
class ToolResult:
    success: bool
    output: str           # stdout or structured result
    artifacts: list[Path] # created files
    metrics: dict         # timing, memory usage

class Tool(ABC):
    """Standardized tool interface for all agents."""

    name: str
    description: str
    required_permissions: list[ToolPermission]

    @abstractmethod
    async def execute(self, params: dict) -> ToolResult: ...

    @abstractmethod
    def validate_params(self, params: dict) -> bool: ...

class ShellTool(Tool):
    """Execute shell commands in a project-specific virtual environment."""
    name = "shell"
    description = "Execute shell commands (install deps, run scripts, etc.)"
    required_permissions = [ToolPermission.SHELL_EXECUTE]

    async def execute(self, params: dict) -> ToolResult:
        cmd = params["command"]
        cwd = Path(params.get("cwd", "."))
        # Run in isolated venv, with timeout, capturing stdout/stderr
        process = await asyncio.create_subprocess_shell(
            cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=params.get("timeout", 120)
        )
        return ToolResult(
            success=process.returncode == 0,
            output=stdout.decode() or stderr.decode(),
            metrics={"exit_code": process.returncode},
        )

class GitTool(Tool):
    """Git operations: branch, commit, diff, status, log, PR."""
    name = "git"
    ...
```

### Tool Permissions

```python
class ToolPermission(str, Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL_EXECUTE = "shell_execute"
    NETWORK_REQUEST = "network_request"
    PACKAGE_INSTALL = "package_install"
    GIT_COMMIT = "git_commit"
    GIT_PUSH = "git_push"
```

Agents request permissions. The `ToolRegistry` enforces them. Critical operations (FILE_DELETE, GIT_PUSH) require human approval via `HumanInTheLoop`.

---

## Capability 5: Agent Communication Protocol

**Status:** Partial. `unified_events/` event bus exists but only for project progress streaming.  
**Target:** Structured inter-agent messaging — task delegation, status updates, conflict resolution.  
**Extend:** `orchestrator/unified_events/`  
**Effort:** 3 days | **Priority:** P0

### Data Model

```python
@dataclass
class AgentMessage:
    """A structured message between agents."""
    id: str
    sender: AgentRole
    recipient: AgentRole | None  # None = broadcast
    type: MessageType
    content: str                 # LLM-generated message body
    task_reference: str | None   # Link to the agent task this relates to
    artifacts: list[Path]        # Attached files or workspace entries
    priority: Priority
    timestamp: datetime
    reply_to: str | None         # Thread reference

class MessageType(str, Enum):
    TASK_REQUEST = "task_request"       # "Please build the auth module"
    TASK_RESPONSE = "task_response"     # "Done, see workspace/auth/"
    QUERY = "query"                     # "What database did we choose?"
    ALERT = "alert"                     # "auth.py conflicts with main.py"
    PROGRESS = "progress"               # "15% complete, working on login"
    CONFLICT = "conflict"               # "I also modified auth.py"
    APPROVAL_REQUEST = "approval"       # "OK to install package X?"
```

### Message Bus Extension

```python
class AgentMessageBus:
    """Publish/subscribe message bus for agent communication.

    Extends the existing unified_events/ event bus with agent-specific
    routing, message history, and delivery guarantees.
    """

    def __init__(self):
        self.subscriptions: dict[AgentRole, list[MessageType]] = {}
        self.message_history: list[AgentMessage] = []
        self._pending_approvals: list[AgentMessage] = []

    async def publish(self, message: AgentMessage) -> None:
        """Deliver a message to the recipient agent's inbox."""
        self.message_history.append(message)
        if message.recipient:
            # Direct delivery
            await self._deliver(message.recipient, message)
        else:
            # Broadcast to all agents
            for role in self.subscriptions:
                await self._deliver(role, message)

    async def subscribe(self, role: AgentRole, message_types: list[MessageType]) -> None:
        """Register an agent's interest in specific message types."""

    async def deliver_to(self, role: AgentRole) -> list[AgentMessage]:
        """Retrieve pending messages for an agent."""
```

---

## Capability 6: Self-Reflection and Cross-Task Learning

**Status:** Partial. `ObservabilityService` tracks per-model metrics but doesn't feed back into decisions.  
**Target:** Learning loop that spans tasks — remembers what worked, adapts strategy per pattern.  
**Extend:** `orchestrator/services/observability.py` + new `orchestrator/learning/`  
**Effort:** 4 days | **Priority:** P1

### Data Model

```python
class ExperienceBuffer:
    """Remembers what worked and what didn't across tasks."""

    success_patterns: dict[str, SuccessPattern]  # pattern_hash → results
    failure_patterns: dict[str, FailurePattern]
    model_performance: dict[Model, PerformanceWindow]  # rolling 100-call window
    method_effectiveness: dict[str, float]  # "CoVE on CODE_GEN" → +0.15 avg
    best_practices: dict[str, list[str]]  # "FastAPI routing" → ["Use async", "Pydantic v2"]

class StrategyAdapter:
    """Adapts execution strategy based on experience."""
    
    def select_model(self, task: Task, buffer: ExperienceBuffer) -> Model:
        """Check history for similar tasks, prefer models with high success rate."""
        pattern_hash = self._hash_pattern(task)
        if pattern_hash in buffer.success_patterns:
            return buffer.success_patterns[pattern_hash].best_model
        return self._default_model(task.type)

    def select_ara_method(self, task: Task, buffer: ExperienceBuffer) -> ReasoningMethod:
        """Which ARA method produced the best results for this task type?"""
        key = f"{task.type.value}"
        return buffer.method_effectiveness.get(key, self._default_method(task))
```

### Feedback Flow

```
Task executed → result recorded → pattern hashed → lookup buffer
  │
  ├─ Pattern seen before → adapt strategy (model, method, retries)
  └─ Pattern new → use defaults, record outcome for future
```

---

## Capability 7: Runtime Execution Environment

**Status:** ❌ Does not exist. Code is validated syntactically but never executed.  
**Target:** Isolated sandbox for executing generated code, running tests, and verifying correctness.  
**New package:** `orchestrator/runtime/`  
**Effort:** 5 days | **Priority:** P1

### Architecture

```
orchestrator/runtime/
  ├── __init__.py
  ├── sandbox.py          # SandboxExecutor (isolated venv/docker)
  ├── test_runner.py      # TestRunner (pytest, jest, go test, cargo test)
  ├── build_runner.py     # BuildRunner (compile, bundle, package)
  ├── lint_runner.py      # LintRunner (ruff, eslint, go vet)
  └── runtime_security.py # Security isolation, timeout enforcement
```

### Self-Correcting Loop

```python
class SelfCorrectingExecutor:
    """Generate → execute → test → revise loop."""

    async def execute_with_correction(
        self, task: AgentTask, max_iterations: int = 3
    ) -> AgentTaskResult:
        for attempt in range(max_iterations):
            # 1. Generate code
            result = await self.developer.handle_task(task)

            # 2. Lint
            lint = await self.lint_runner.run(result.artifacts)
            if lint.has_errors:
                task.critique = lint.errors
                continue  # Revise

            # 3. Test
            tests = await self.test_runner.run(result.artifacts)
            if not tests.all_passed:
                task.critique = tests.failures
                continue  # Revise

            # 4. Build
            build = await self.build_runner.run()
            if build.failed:
                task.critique = build.errors
                continue  # Revise

            return result  # Success!

        return AgentTaskResult(status=TaskStatus.FAILED)
```

---

## Capability 8: Dynamic Scaffold Generator

**Status:** Partial. Fixed scaffold templates (FastAPI, React, Next.js, CLI).  
**Target:** Generate a project scaffold for ANY technology combination, even when no template exists.  
**New file:** `orchestrator/scaffold/dynamic.py`  
**Effort:** 4 days | **Priority:** P2

### Logic

```python
class DynamicScaffoldGenerator:
    """Generate project structure for any tech stack.

    If a template exists → use it.
    If no template → use ARA Multi-Perspective to design from first principles.
    """

    async def generate(self, goal: str, workspace: ProjectWorkspace) -> ScaffoldResult:
        # 1. Detect tech stack from goal + workspace
        stack = self._detect_stack(goal, workspace)

        # 2. Check for existing template
        template = self._find_template(stack)
        if template:
            return self._apply_template(template, workspace)

        # 3. Design new scaffold via ARA
        ara_result = await self._ara.execute(
            goal=f"Design a project scaffold for {stack}",
            workspace=workspace,
            method=ReasoningMethod.MULTI_PERSPECTIVE,
        )

        # 4. Generate files
        return self._generate_files(ara_result.structure, workspace)

    def _detect_stack(self, goal: str, workspace: ProjectWorkspace) -> TechStack:
        # Parse goal text: "build a Go app with HTMX and SQLite"
        # Return: TechStack(language="go", framework=None, frontend="htmx", database="sqlite")
```

---

## Capability 9: Continuous Integration Pipeline

**Status:** ❌ Does not exist. One-shot code generation, no CI.  
**Target:** Run lint → type-check → test → build after every code change, report results.  
**New package:** `orchestrator/ci/`  
**Effort:** 3 days | **Priority:** P1

### Architecture

```python
class CIPipeline:
    """Chain of quality checks that runs after code generation."""

    steps: list[CIStep] = [
        LintStep(),       # ruff, eslint, go vet
        TypeCheckStep(),  # mypy, tsc
        TestStep(),       # pytest, jest, go test
        BuildStep(),      # compile, bundle
        SecurityStep(),   # bandit, npm audit, gosec
    ]

    async def run(self, workspace: ProjectWorkspace) -> CIReport:
        report = CIReport()
        for step in self.steps:
            result = await step.execute(workspace)
            report.add_result(step.name, result)
            if result.failed and step.critical:
                break  # Stop pipeline, feed failures back to agent
        return report
```

### Integration with Agent Loop

```
DeveloperAgent completes code → CIPipeline runs → CIReport generated
  │
  ├─ All pass → TesterAgent verifies → Deploy
  └─ Any fail → CIReport → DeveloperAgent (revise) → CIPipeline (retry)
```

---

## Capability 10: Human-in-the-Loop Coordinator

**Status:** Partial. PreflightStage does PASS/WARN/BLOCK but no human approval.  
**Target:** Pause for human approval on critical decisions.  
**Extend:** `engine_core/stages/preflight.py` → new `orchestrator/hitl/`  
**Effort:** 3 days | **Priority:** P2

### Architecture

```python
class DecisionGate:
    """Criteria that trigger human approval."""

    ALWAYS_APPROVE = [
        "architecture_choice",        # Which framework to use
        "security_sensitive_code",    # Auth, crypto, payment code
        "breaking_change",            # API contract changes
        "data_migration",             # Database schema changes
        "production_deploy",          # Before pushing to production
    ]

class HumanInTheLoop:
    """Pause execution and request human approval."""

    async def request_decision(
        self, decision: Decision, context: str, timeout: int = 300
    ) -> DecisionResult:
        # Present to user via CLI, IDE backend WebSocket, or web dashboard
        # Wait for response with configurable timeout
        # Return: APPROVED, REJECTED, TIMEOUT_DEFAULT
```

### Integration Points

| Decision Gate | Trigger | Default if Timeout |
|--------------|---------|-------------------|
| Architecture choice | First decomposition level | PROCEED (non-blocking) |
| Security-sensitive code | PersuasionDefense FAIL | BLOCK (reject code) |
| Breaking change | Detected by diff analysis | PROCEED_WITH_WARNING |
| Data migration | DatabaseTool detects schema change | BLOCK |
| Production deploy | DevOpsAgent pre-deploy gate | BLOCK |

---

## Implementation Order and Dependencies

```
        Capability 1 (Multi-Agent Architecture)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                  ▼
Capability 2         Capability 3       Capability 5
(Workspace)          (Goal Decomp)      (Agent Comms)
        │                 │                  │
        └─────────────────┼──────────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
Capability 4         Capability 6       Capability 8
(Tools)              (Self-Learning)    (Dynamic Scaffold)
        │                 │                  │
        └─────────────────┼──────────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
Capability 7         Capability 9       Capability 10
(Runtime)            (CI Pipeline)      (HITL)
```

**Parallel opportunities:**
- Capabilities 2, 3, and 5 can be built in parallel (different packages, no shared state)
- Capabilities 4, 6, and 8 can be built in parallel after workspace exists
- Capabilities 7, 9, 10 are sequential (each depends on previous execution results)

---

## Effort Summary

| # | Capability | New Module | Days | Priority | Blocks |
|---|-----------|------------|------|----------|--------|
| 1 | Multi-agent architecture | `orchestrator/agents/` | 8 | **P0** | — |
| 2 | Shared workspace | `orchestrator/workspace/` | 5 | **P0** | Cap 1 |
| 3 | Recursive goal decomposition | `orchestrator/planning/` | 5 | **P0** | Cap 1 |
| 4 | Tool integration layer | `orchestrator/tools/` | 6 | **P0** | Cap 2 |
| 5 | Agent communication protocol | Extend `unified_events/` | 3 | **P0** | Cap 1 |
| 6 | Self-reflection and learning | `orchestrator/learning/` | 4 | P1 | Cap 2 |
| 7 | Runtime execution environment | `orchestrator/runtime/` | 5 | P1 | Cap 4 |
| 8 | Dynamic scaffold generator | `orchestrator/scaffold/dynamic.py` | 4 | P2 | Cap 2 |
| 9 | CI pipeline | `orchestrator/ci/` | 3 | P1 | Cap 7 |
| 10 | Human-in-the-loop | `orchestrator/hitl/` | 3 | P2 | Cap 9 |
| **Total** | | | **46 days** | | |

**Minimum Viable Agentic System (MVAS): P0 only = 27 days**

---

## Files to Create / Modify

| File | Action | Capability |
|------|--------|-----------|
| `orchestrator/agents/__init__.py` | Create | 1 |
| `orchestrator/agents/base.py` | Create | 1 |
| `orchestrator/agents/coordinator.py` | Create | 1 |
| `orchestrator/agents/architect.py` | Create | 1 |
| `orchestrator/agents/developer.py` | Create | 1 |
| `orchestrator/agents/reviewer.py` | Create | 1 |
| `orchestrator/agents/tester.py` | Create | 1 |
| `orchestrator/agents/devops.py` | Create | 1 |
| `orchestrator/agents/researcher.py` | Create | 1 |
| `orchestrator/agents/agent_registry.py` | Create | 1 |
| `orchestrator/workspace/workspace.py` | Create | 2 |
| `orchestrator/workspace/file_version.py` | Create | 2 |
| `orchestrator/workspace/decision_log.py` | Create | 2 |
| `orchestrator/workspace/test_log.py` | Create | 2 |
| `orchestrator/workspace/message_bus.py` | Create | 2, 5 |
| `orchestrator/workspace/knowledge_graph.py` | Create | 2 |
| `orchestrator/planning/goal.py` | Create | 3 |
| `orchestrator/planning/decomposer.py` | Create | 3 |
| `orchestrator/planning/dependency_graph.py` | Create | 3 |
| `orchestrator/planning/cost_estimator.py` | Create | 3 |
| `orchestrator/planning/strategy.py` | Create | 3 |
| `orchestrator/tools/base.py` | Create | 4 |
| `orchestrator/tools/file_tool.py` | Create | 4 |
| `orchestrator/tools/shell_tool.py` | Create | 4 |
| `orchestrator/tools/git_tool.py` | Create | 4 |
| `orchestrator/tools/package_tool.py` | Create | 4 |
| `orchestrator/tools/test_tool.py` | Create | 4 |
| `orchestrator/tools/build_tool.py` | Create | 4 |
| `orchestrator/tools/tool_registry.py` | Create | 4 |
| `orchestrator/unified_events/core.py` | Extend | 5 |
| `orchestrator/learning/experience_buffer.py` | Create | 6 |
| `orchestrator/learning/strategy_adapter.py` | Create | 6 |
| `orchestrator/runtime/sandbox.py` | Create | 7 |
| `orchestrator/runtime/test_runner.py` | Create | 7 |
| `orchestrator/runtime/build_runner.py` | Create | 7 |
| `orchestrator/runtime/lint_runner.py` | Create | 7 |
| `orchestrator/scaffold/dynamic.py` | Create | 8 |
| `orchestrator/ci/pipeline.py` | Create | 9 |
| `orchestrator/hitl/gate.py` | Create | 10 |
| `orchestrator/engine.py` | Modify (use AgentOrchestrator) | 1 |
| `orchestrator/cli.py` | Add `agentic` subcommand | 1 |

**Total: 41 files (30 new, 2 modified, 0 deleted)**

---

## Verification Gates

- [ ] `python -m orchestrator agentic --goal "Build a CLI calculator" --dry-run` produces a valid plan
- [ ] Two agents can exchange messages through the message bus
- [ ] Workspace records mutations with attribution and timestamps
- [ ] GoalDecomposer recursively decomposes "Build a social media app" into 20+ atomic tasks
- [ ] ShellTool executes `python --version` in isolated environment
- [ ] AgentOrchestrator dispatches tasks to correct agents based on role
- [ ] Self-learning loop: after 10 tasks, strategy adapter selects better model than default
- [ ] Self-correcting loop: generated code → lint fails → revise → lint passes → test passes
- [ ] Dynamic scaffold: "Build a Rust CLI with clap" generates working project structure
- [ ] Human-in-the-loop: approving an architecture decision unblocks the ArchitectAgent
- [ ] All existing tests pass (backward compatibility)
- [ ] All new agent tests pass

## Appendix: Target Metrics

| Metric | Before | After (MVAS) | After (Full) |
|---------|--------|-------------|-------------|
| Agent roles | 1 (Orchestrator) | 7 specialized agents | 7 agents |
| Inter-agent communication | None | Publish/subscribe messages | Threaded conversations |
| Shared state | None (per-task isolation) | Workspace with audit trail | Knowledge graph + experience buffer |
| Goal decomposition depth | 1 level | 3 levels (recursive) | 4+ levels |
| Tools available | File I/O only | 8 tools | 8 tools |
| Self-correction | Critique → Revise only | Lint → Test → Build → Revise | Full CI pipeline |
| Human interaction | None | Approval gates for critical decisions | Full HITL protocol |
| Languages supported | Python (primary) | Any (via tool dispatch) | Any |
| Test coverage | 12% | 35% | 50% |

---

**Last updated:** 2026-05-23
