# CodebaseInvestigatorAgent — Implementation Plan

## Context

The project already has a production-grade 4-phase codebase analysis pipeline
(`codebase/reader.py`, `codebase/context.py`, `codebase/decomposer.py`,
`orchestrator/analyzer.py`). The gap is that this capability exists only as a
standalone CLI tool — it is not accessible to other agents during a workflow run.

No agent can currently ask "I need to understand how module X works before I
start coding." The coordinator has no `INVESTIGATOR` role, and `ResearcherAgent`
only searches the web (Perplexity Sonar), with zero overlap on internal codebase
investigation.

This plan wraps the existing `CodebaseAnalyzer` behind the standard `AgentBase`
interface with minimal new code (~70 lines). Once wired, any agent can dispatch
an investigation request mid-workflow via the message bus.

---

## What already exists (do not rebuild)

| Module | Capability |
|--------|-----------|
| `orchestrator/codebase/reader.py` | FileSystemWalker, ASTIndexer, DependencyGraph |
| `orchestrator/codebase/context.py` | RelevanceRanker — ranks files by relevance within token budget |
| `orchestrator/codebase/decomposer.py` | LLM-driven task planning for codebase modifications |
| `orchestrator/analyzer.py` | `CodebaseAnalyzer.analyze()` — full multi-LLM analysis across 5 focus areas |

---

## Files to create

### `orchestrator/agents/investigator.py` (~70 lines)

```python
"""
CodebaseInvestigatorAgent — wraps CodebaseAnalyzer behind the AgentBase interface.

Receives investigation tasks from the coordinator or other agents, runs the
existing codebase analysis pipeline, and returns structured findings as an
AgentTaskResult.
"""
from __future__ import annotations

from orchestrator.agents.base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from orchestrator.analyzer import CodebaseAnalyzer


class CodebaseInvestigatorAgent(AgentBase):
    """Investigates an existing codebase on demand.

    Wraps the production-grade CodebaseAnalyzer CLI tool behind the agent
    interface so that other agents can request codebase understanding
    dynamically during a workflow run.

    Typical dispatch triggers (handled by coordinator._decompose_goal):
      "understand X", "trace X", "explore X", "how does X work",
      "investigate X", "map dependencies of X"
    """

    role = AgentRole.INVESTIGATOR
    system_prompt = (
        "You are a code archaeologist. Given a codebase path and an objective, "
        "produce clear, structured findings: execution paths, module responsibilities, "
        "dependency relationships, and any relevant patterns. "
        "Be precise and cite file paths and function names."
    )

    # Model preference: capable reasoning model, not the most expensive
    model_preferences = {
        "default": "DEEPSEEK_V4_FLASH",
        "complex": "CLAUDE_SONNET_4_6",
    }

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Run CodebaseAnalyzer on the target path with the given objective."""
        codebase_path = task.context.get("codebase_path", ".")
        objective = task.goal

        analyzer = CodebaseAnalyzer(
            codebase_path=codebase_path,
            objective=objective,
            model=self._pick_model(task),
        )
        findings = await analyzer.analyze()

        return AgentTaskResult(
            agent_role=self.role,
            task_id=task.task_id,
            success=bool(findings),
            output=findings,
            summary=f"Investigation complete for: {objective}",
        )

    def _pick_model(self, task: AgentTask) -> str:
        complexity = task.context.get("complexity", "default")
        return self.model_preferences.get(complexity, self.model_preferences["default"])
```

---

## Files to modify

### 1. `orchestrator/agents/base.py` — Add `INVESTIGATOR` to `AgentRole`

Locate the `AgentRole` enum (around line 42) and add:

```python
class AgentRole(str, Enum):
    ARCHITECT       = "architect"
    DEVELOPER       = "developer"
    REVIEWER        = "reviewer"
    TESTER          = "tester"
    DEVOPS          = "devops"
    RESEARCHER      = "researcher"
    USER            = "user"
    PRODUCT_MANAGER = "product_manager"
    QA              = "qa"
    INVESTIGATOR    = "investigator"   # ← add this line
```

### 2. `orchestrator/agents/coordinator.py` — Route investigation tasks

In `_decompose_goal()`, add a detection branch for investigation intent before
the existing keyword routing. Pattern: any goal containing "understand", "trace",
"explore", "investigate", "how does", or "map dependencies" should produce an
INVESTIGATOR task.

```python
INVESTIGATION_TRIGGERS = {
    "understand", "trace", "explore", "investigate",
    "how does", "map dependencies", "dependency map",
}

# Inside _decompose_goal(), before existing routing:
goal_lower = goal.lower()
if any(trigger in goal_lower for trigger in INVESTIGATION_TRIGGERS):
    tasks.append(AgentTask(
        task_id=f"investigate_{uuid4().hex[:8]}",
        goal=goal,
        target_role=AgentRole.INVESTIGATOR,
        context={"codebase_path": workspace.root_path},
        dependencies=[],
    ))
    return tasks
```

### 3. `orchestrator/agents/coordinator.py` — Register agent in `_build_agents()`

Alongside the other agent instantiations:

```python
from .investigator import CodebaseInvestigatorAgent

# Inside _build_agents():
self._agents[AgentRole.INVESTIGATOR] = CodebaseInvestigatorAgent(
    workspace=self._workspace,
    message_bus=self._message_bus,
)
```

---

## Hexagonal architecture fit

```
Domain (models.py, domain/)          ← unchanged
    ↑
Application layer:
    orchestrator/agents/investigator.py   ← NEW (inherits AgentBase)
    orchestrator/agents/base.py           ← MODIFIED (AgentRole enum)
    orchestrator/agents/coordinator.py    ← MODIFIED (routing + registration)
    orchestrator/analyzer.py             ← REUSED (no changes needed)
    orchestrator/codebase/              ← REUSED (no changes needed)
    ↑
Infrastructure (api_clients, cache, state)  ← unchanged
```

The agent lives entirely in the application layer and imports only from
application-layer modules (`agents/`, `analyzer.py`, `codebase/`). No new
infrastructure dependencies are introduced.

---

## Overlap analysis

| Agent | Scope | Overlap with Investigator |
|-------|-------|--------------------------|
| `ResearcherAgent` | External web research (Perplexity Sonar) | None — different domain |
| `ReviewerAgent` | Code review of *generated* code | None — reviews, doesn't explore |
| `QCAgent` | Quality metrics on *delivered* code | None — metrics, not understanding |
| `ArchitectAgent` | System design for *new* features | Partial — both need codebase understanding, but Architect designs new things; Investigator understands existing ones |
| `DeveloperAgent` | Code generation with self-correction | None — generates, doesn't explore first |
| `CodebaseAnalyzer` (CLI) | LLM multi-focus analysis (5 areas) | Full — same job, different interface. The Investigator is the agent wrapper of this tool. |

---

## Build order

1. `orchestrator/agents/base.py` — add `INVESTIGATOR` to `AgentRole` enum
2. `orchestrator/agents/investigator.py` — create `CodebaseInvestigatorAgent`
3. Write unit tests (RED):
   - `tests/test_investigator_agent.py`
   - `test_handle_task_calls_analyzer` — mock `CodebaseAnalyzer.analyze()`; verify result wrapping
   - `test_picks_complex_model_when_complexity_set` — context `{"complexity": "complex"}` → Sonnet
   - `test_default_model_used_when_no_complexity` — no complexity key → DeepSeek
4. Implement until tests pass (GREEN)
5. `orchestrator/agents/coordinator.py` — add routing + registration
6. Write integration test:
   - `test_coordinator_routes_investigate_goal_to_investigator` — goal "understand auth flow" → dispatches `INVESTIGATOR` task
7. Verify no regressions: `pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --tb=short -q`

---

## Verification

```bash
# 1. Unit tests
pytest tests/test_investigator_agent.py -v

# 2. Coordinator routing test
pytest tests/ -k "investigator" -v

# 3. Regression check
pytest tests/ -m "not slow and not requires_api and not stress and not e2e" \
  --tb=short -q --no-cov

# 4. Manual smoke test (no API key needed — dry run)
python -c "
from orchestrator.agents.base import AgentRole
print('INVESTIGATOR' in [r.value for r in AgentRole])
"
# Expected: True
```
