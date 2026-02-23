# Design: Complete App Generation + Constraint Control Plane + Orchestration Agent

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Two coupled improvements to the multi-llm-orchestrator

---

## Problem Statement

### Problem 1 — Build produces Python boilerplate, not web apps

When `python -m orchestrator build --app-type nextjs ...` is run:
- `ScaffoldEngine` has no `nextjs`/`react` templates → falls back to Python `generic`
- `_decompose` never asks the LLM for `target_path` → `AppAssembler` skips every task
- `DependencyResolver` + `AppVerifier` only understand pip/pytest
- Result: `main.py` with `print("Hello World")` instead of a Next.js app

### Problem 2 — Constraint enforcement is prompt-based, not structural

- Hard constraints (jurisdiction, PII, no-training) are expressed as hints in prompts
- No formal separation between the agent (proposes specs) and the control plane (enforces)
- `JobSpec` lacks `slas`, `inputs`, `constraints`; `PolicySpec` lacks routing hints and escalation rules
- No `ReferenceMonitor` that enforces hard rules before task execution
- No `OrchestrationAgent` that converts NL intent → typed specs with human approval

---

## Approved Design

### Topic 1 — Always Complete Apps (Approach A)

#### Root causes and fixes

| Root Cause | Fix |
|-----------|-----|
| No JS/TS scaffold templates | Add `nextjs.py`, `react_vite.py`, `html.py` templates |
| `target_path` always `""` | Extend `_decompose` prompt to request `target_path` per task |
| LLM unaware of `app_type` | Inject `app_type`, `tech_stack`, scaffold file list into decompose prompt |
| Only pip/pytest supported | `DependencyResolver` detects `package.json` → `npm install`; `AppVerifier` runs `npm test` / `npm run build` |
| Default `run` produces flat task files | CLI default routes through `AppBuilder` unless `--raw-tasks` flag given |

#### A1 — New scaffold templates

```
orchestrator/scaffold/templates/
├── nextjs.py       # next.config.js, tailwind.config.ts, tsconfig.json,
│                   # package.json (next, react, tailwindcss, framer-motion),
│                   # app/layout.tsx, app/page.tsx, app/globals.css
├── react_vite.py   # vite.config.ts, index.html, src/main.tsx, src/App.tsx,
│                   # package.json (react, vite, typescript, tailwindcss)
└── html.py         # index.html, styles/main.css, scripts/main.js
```

`_TEMPLATE_MAP` updated:
```python
"nextjs":       nextjs.FILES,
"react-fastapi": react_vite.FILES,   # frontend; backend uses fastapi.FILES
"html":         html.FILES,
```

#### A2 — App-type-aware decompose prompt

The `_decompose` prompt receives additional context when `app_profile` is present:

```
APP_TYPE: nextjs
TECH_STACK: typescript, next.js 14, tailwind css, framer-motion
SCAFFOLD_FILES (already exist — fill or extend these):
  - app/layout.tsx
  - app/page.tsx
  - tailwind.config.ts
  ...

Each task JSON element MUST include:
  - "target_path": the relative file path this task writes (e.g. "app/page.tsx")
  - "tech_context": brief note on tech stack relevant to this file
Tasks producing non-file outputs (code_review, evaluation) use target_path: "".
```

`_parse_decomposition` maps `target_path` and `tech_context` onto `Task`.

#### A3 — AppBuilder passes profile to Orchestrator

```python
# app_builder.py
state = await orchestrator.run_project(
    project_description=description,
    success_criteria=criteria,
    app_profile=profile,          # NEW
)
```

`Orchestrator.run_project` and `_decompose` accept optional `app_profile: Optional[AppProfile]`.

#### A4 — DependencyResolver: npm support

Detection order:
1. `package.json` exists → `npm install --legacy-peer-deps`
2. `requirements.txt` / `pyproject.toml` exists → pip install
3. Both → run both

#### A5 — AppVerifier: npm test + build

```python
# app_verifier.py
if profile.app_type in ("nextjs", "react-fastapi", "html"):
    verify_cmd = "npm run build"
    test_cmd   = "npm test -- --passWithNoTests"
else:
    verify_cmd = profile.run_command
    test_cmd   = profile.test_command
```

#### A6 — CLI: default run → AppBuilder

```python
# cli.py  _async_new_project()
if not args.raw_tasks:
    # Route through AppBuilder (detects app_type automatically)
    result = await AppBuilder().build(
        description=args.project,
        criteria=args.criteria,
        output_dir=output_dir,
    )
else:
    # Legacy flat-file path (opt-in)
    state = await orch.run_project(...)
    write_output_dir(state, output_dir)
```

New CLI flag: `--raw-tasks` (preserves old behaviour for power users).

---

### Topic 2 — Constraint Control Plane + Orchestration Agent (Approach A)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION AGENT                       │
│  • Accepts NL intent from human or system                    │
│  • Produces draft JobSpecV2 + PolicySpecV2                   │
│  • Explains rationale for each constraint chosen             │
│  • Supports refine() loop based on human feedback            │
│  • NEVER calls LLM providers directly                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ human approve / edit
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONTROL PLANE SERVICE                      │
│  Step 1: validate(job, policy)  — schema + static analysis  │
│  Step 2: monitor.check_global() — hard constraints pre-run  │
│  Step 3: solve_constraints()    — routing plan, SLA fit      │
│  Step 4: run_workflow()         — delegates to Orchestrator  │
│  Step 5: audit_log.write()      — immutable structured log   │
└─────────────────────────────────────────────────────────────┘
                      │ per-task, synchronous
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   REFERENCE MONITOR                          │
│  Called before EVERY task execution                          │
│  Returns ALLOW | DENY(reason) | ESCALATE(rule)               │
│  Cannot be bypassed by prompt content                        │
└─────────────────────────────────────────────────────────────┘
```

#### B1 — `specs.py`: JobSpecV2 + PolicySpecV2

```python
@dataclass
class SLAs:
    max_latency_ms: Optional[int] = None
    max_cost_usd: Optional[float] = None
    min_quality_tier: float = 0.85
    reliability_target: float = 0.95

@dataclass
class InputSpec:
    schema: dict = field(default_factory=dict)
    data_locality: str = "any"   # "eu" | "us" | "any"
    contains_pii: bool = False

@dataclass
class Constraints:
    hard: list[str] = field(default_factory=list)
    # e.g. ["no_training", "eu_only", "no_pii_logging"]
    soft: dict[str, float] = field(default_factory=dict)
    # e.g. {"prefer_low_cost": 0.8, "prefer_low_latency": 0.6}

@dataclass
class JobSpecV2:
    goal: str
    inputs: InputSpec = field(default_factory=InputSpec)
    slas: SLAs = field(default_factory=SLAs)
    constraints: Constraints = field(default_factory=Constraints)
    metrics: list[str] = field(default_factory=list)
    task_tree: list[dict] = field(default_factory=list)
    # backward-compat
    budget: Budget = field(default_factory=lambda: Budget(max_usd=8.0))
    policy_set: PolicySet = field(default_factory=PolicySet)

@dataclass
class RoutingHint:
    condition: str    # "eu_only AND contains_pii"
    target: str       # "self_hosted_only" | "eu_models_only"

@dataclass
class ValidationRule:
    node_pattern: str                   # TaskType value or "*"
    mandatory_validators: list[str]

@dataclass
class EscalationRule:
    trigger: str    # "validator_failed AND iterations >= 3"
    action: str     # "human_review" | "abort" | "fallback_model"

@dataclass
class PolicySpecV2:
    allow_deny_rules: list[dict] = field(default_factory=list)
    # [{"effect": "deny", "when": "risk_level == high AND jurisdiction != eu"}]
    routing_hints: list[RoutingHint] = field(default_factory=list)
    validation_rules: list[ValidationRule] = field(default_factory=list)
    escalation_rules: list[EscalationRule] = field(default_factory=list)
```

#### B2 — `reference_monitor.py`

```python
class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"

@dataclass
class MonitorResult:
    decision: Decision
    reason: str = ""
    rule: Optional[EscalationRule] = None

class ReferenceMonitor:
    """
    Synchronous, bypass-proof hard constraint checker.
    Called before every task execution by ControlPlane.
    """
    HARD_RULES = {
        "no_training":       _check_no_training,
        "eu_only":           _check_eu_only,
        "no_pii_logging":    _check_no_pii_logging,
    }

    def check(self, task: Task, job: JobSpecV2, policy: PolicySpecV2) -> MonitorResult:
        for constraint in job.constraints.hard:
            checker = self.HARD_RULES.get(constraint)
            if checker:
                result = checker(task, job, policy)
                if result.decision != Decision.ALLOW:
                    return result
        for rule in policy.allow_deny_rules:
            result = self._eval_rule(rule, task, job)
            if result.decision != Decision.ALLOW:
                return result
        return MonitorResult(Decision.ALLOW)
```

#### B3 — `control_plane.py`

```python
class ControlPlane:
    def __init__(self):
        self._monitor = ReferenceMonitor()
        self._audit = AuditLog()

    async def submit(
        self,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> ProjectState:
        # 1. Validate
        errors = self._validate(job, policy)
        if errors:
            raise SpecValidationError(errors)

        # 2. Pre-run hard constraint check
        global_check = self._monitor.check_global(job, policy)
        if global_check.decision == Decision.DENY:
            raise PolicyViolation(global_check.reason)

        # 3. Solve constraints → routing plan
        routing = self._solve_constraints(job, policy)

        # 4. Run workflow (delegates to Orchestrator engine)
        state = await self._run_workflow(job, routing)

        # 5. Audit
        self._audit.write(job, policy, routing, state)
        return state

    def _solve_constraints(self, job, policy) -> RoutingPlan:
        """Select models per task based on SLAs, routing hints, hard constraints."""
        ...
```

`Orchestrator._execute_task` calls `monitor.check(task, job, policy)` before each execution.

#### B4 — `orchestration_agent.py`

```python
class AgentDraft:
    job: JobSpecV2
    policy: PolicySpecV2
    rationale: str      # "Chose eu_only because data_locality=eu was specified"

class OrchestrationAgent:
    """
    NL intent → draft specs → human approval.
    Uses the same LLM infrastructure as the Orchestrator but
    outputs ONLY specs, never calls task executors.
    """
    async def draft(self, nl_intent: str) -> AgentDraft:
        ...

    async def refine(self, draft: AgentDraft, feedback: str) -> AgentDraft:
        ...
```

**System prompt highlights:**
- "You produce JobSpecV2 and PolicySpecV2 JSON only"
- "You never invoke model APIs directly"
- "For every constraint you choose, explain why in rationale"
- Uses `capability_library` (list of available validators, models, regions) as context

**CLI:**
```bash
python -m orchestrator agent \
  --intent "θέλω pipeline για code review, budget $2, EU data only" \
  --interactive
```
Prints draft specs → user edits → submits to `ControlPlane.submit()`.

#### B5 — Telemetry / Continuous Improvement loop

After each run the agent can be called with audit log context:
```python
agent.analyze_run(state, job, policy)
# → "Suggestion: tighten max_cost_usd from $0.05 to $0.03 (avg was $0.012)"
# → "Suggestion: add ruff to mandatory_validators for code_generation nodes"
```

---

## New Files

| File | Role |
|------|------|
| `orchestrator/specs.py` | `JobSpecV2`, `PolicySpecV2`, `SLAs`, `InputSpec`, `Constraints`, `RoutingHint`, `ValidationRule`, `EscalationRule` |
| `orchestrator/reference_monitor.py` | `ReferenceMonitor`, `Decision`, `MonitorResult` — hard constraint enforcement |
| `orchestrator/control_plane.py` | `ControlPlane` service — validate → solve → run → log |
| `orchestrator/orchestration_agent.py` | `OrchestrationAgent` — NL → draft specs + refine loop |
| `orchestrator/scaffold/templates/nextjs.py` | Next.js 14 + Tailwind + Framer Motion scaffold |
| `orchestrator/scaffold/templates/react_vite.py` | React + Vite + TypeScript scaffold |
| `orchestrator/scaffold/templates/html.py` | Vanilla HTML/CSS/JS scaffold |

## Modified Files

| File | Change |
|------|--------|
| `orchestrator/engine.py` | `_decompose` accepts `app_profile`, injects `target_path` in prompt and parses it |
| `orchestrator/app_builder.py` | Passes `profile` to `_run_orchestrator` |
| `orchestrator/scaffold/__init__.py` | Adds new templates to `_TEMPLATE_MAP` |
| `orchestrator/dep_resolver.py` | npm install support |
| `orchestrator/app_verifier.py` | npm test/build support |
| `orchestrator/cli.py` | Default `run` → AppBuilder; `--raw-tasks` flag; `agent` subcommand |
| `orchestrator/policy_engine.py` | Integrates `ReferenceMonitor` per-task check |
| `orchestrator/__init__.py` | Exports new public classes |

## Testing Strategy

- **A-series**: Unit tests for each template (valid file structure), `_decompose` integration test asserting `target_path` present, npm install/test mock tests
- **B-series**: `ReferenceMonitor` unit tests per hard rule, `ControlPlane` integration tests (submit → state), `OrchestrationAgent` tests with mocked LLM
- **E2E**: `build --app-type nextjs` → assert `app/page.tsx` exists + `npm run build` exits 0

## Implementation Order

1. Scaffold templates (A1) — unblocks A2-A6
2. Engine `_decompose` app-profile injection (A2, A3)
3. DependencyResolver + AppVerifier npm support (A4, A5)
4. CLI routing (A6)
5. `specs.py` (B1)
6. `reference_monitor.py` (B2)
7. `control_plane.py` (B3)
8. `orchestration_agent.py` (B4)
9. Telemetry loop (B5)
10. Tests + exports
