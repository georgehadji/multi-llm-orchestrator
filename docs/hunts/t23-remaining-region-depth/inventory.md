# T23 — Remaining region depth (closed, explicitly PARTIAL)

**Base:** `master` @ `f390414`. **Plan:** `docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` §4.

**Scope as specified:** `meta/`, `nash/`, `context_mgmt/`, `agents/`, `scaffold/`,
`pattern_learner/`, `dashboard_core/`, `nexus_search/agents/`, `testing/first_generator.py`,
and the `operations/` files T16 left shallow.

**Method as specified:** coverage-ordered — reachable-from-a-live-entry-point first; dead code
gets a **recorded disposition, not a deep read**. The dead-module census built for P3
(`p3-deep-region/inventory.md`, P3-ORPHAN0) is what makes that ordering executable rather
than guessed, so it was built first and consumed here.

## Reachability of the T23 regions (VERIFIED, measured)

| region | unreferenced / total files |
|---|---|
| `meta/` | 0 / 6 |
| `nash/` | 0 / 5 |
| `context_mgmt/` | 0 / 6 |
| `pattern_learner/` | 0 / 4 |
| `dashboard_core/` | 0 / 5 |
| `testing/` | 0 / 3 |
| `scaffold/` | 1 / 8 |
| `nexus_search/` | 5 / 17 |
| `agents/` | **9 / 14** |
| `operations/` | **13 / 32** |

Six of the ten regions are fully referenced — which is itself worth recording, because it
contradicts the prior assumption (carried since T16) that these were the shallow, likely-dead
corners of the repo. The dead weight is concentrated in `agents/` and `operations/`.

**Coverage claim.** Per the plan's §5 wording for depth waves: *the regions above were swept
for the seven detector classes listed in `p4-p11-region/inventory.md` §Method and for
import-integrity, across 100% of their files; no VERIFIED finding remains unlisted for those
classes. Their remaining taxonomy classes were not exhaustively read* — this is a partial,
ranked claim, not a clean one.

---

## T23-AGENT1 — 9 of 14 agent implementations are unreachable, one of them a role the coordinator actually targets — VERIFIED, HIGH (wiring)

**Files:** `orchestrator/agents/__init__.py`, `agents/coordinator.py:31,36,54`,
`agents/base.py:37-46`.

`AgentOrchestrator.__init__(self, agents: dict[AgentRole, AgentBase], ...)` takes its agents
by **injection** — it never constructs them. Its `_run_one` does
`agent = self.agents.get(task.target_role)` and, on a miss, returns a failed `AgentTaskResult`
rather than raising, so a missing role degrades silently.

Three facts, each verified, that together make the package non-functional as shipped:

1. **Nothing in the product constructs `AgentOrchestrator` at all.** Every construction
   repo-wide is in `tests/` (`test_e2e_full_suite.py:160`, `test_investigator_agent.py` ×7,
   `test_agentic_system.py` ×2), and all but two pass `agents={}`.
2. **`agents/__init__.py` exports implementations for only 3 of the 10 declared roles** —
   `DeveloperAgent`, `ArchitectAgent`, `TesterAgent` (all from `developer.py`), plus `base`
   and `coordinator`. Nine agent modules are exported by nothing and imported by nothing:
   `devops`, `investigator`, `metrics`, `product_manager`, `qc`, `rate_limiter`,
   `researcher`, `reviewer`, `user`.
3. **`_decompose_goal` targets `AgentRole.INVESTIGATOR`**, whose implementation
   (`agents/investigator.py`) is in that unexported nine. So even a caller who wired the
   orchestrator correctly through the package's public API would get a silent
   "no agent for role" failure on every investigator task, because the only way to obtain an
   `InvestigatorAgent` is to reach past `__init__.py` into an unexported module.

`AgentRole` declares ten roles; `_decompose_goal` targets four (ARCHITECT, DEVELOPER,
INVESTIGATOR, TESTER); the remaining six (REVIEWER, DEVOPS, RESEARCHER, USER,
PRODUCT_MANAGER, QA) have implementations that are both unexported and never targeted.

**Compounding factor.** `orchestrator/agents.py` (277 LOC, containing a separate `AgentPool`
with its own `run_job` fan-out) is shadowed by this package and can never be imported at all
— see P3-SHADOW1. There are two agent subsystems here, and neither is reachable.

## T23-DISPOSITION — the dead regions, recorded rather than read

Per the method, these were **not** deep-read. Their disposition is recorded so the ledger is
complete and a future wave need not rediscover their status:

- **`agents/` (9 files)** — unreachable; see T23-AGENT1. Disposition: **wire or delete**, a
  product decision about whether the multi-agent capability is intended to ship.
- **`operations/` (13 files)** — includes `gradual_rollout.py` (713 LOC) and
  `feedback_loop.py` (678 LOC), both of which additionally have twins at the repo root
  (`orchestrator/gradual_rollout.py`, `orchestrator/feedback_loop.py`) that are equally
  unreferenced — dead duplicate pairs, T17's shape but with *both* halves dead.
  Disposition: **delete the twin, then decide on the survivor**.
- **`nexus_search/` (5 files)**, **`scaffold/` (1 file)** — unreachable leaf modules, low LOC.
  Disposition: **delete**, absent a stated intent to wire them.

## Findings carried in from the cross-wave sweep that land in T23 regions

- `dashboard_core/core.py:171` — `get_event_bus()` bound un-awaited, then used as
  `async for event in self.event_bus.subscribe()` at `:230`. Filed as PX-BUS1.
- `operations/diagnostics.py:412` — `state_mgr.load_state(...)`, a method `StateManager` does
  not define. Filed as PX-DIAG1.

## Cleared (innocent)

- `nash/monitor.py:345` `handler(new_level.value, str(violations))` — `self._alert_handlers`
  holds user-registered callbacks, sync by contract, each already wrapped in its own
  `try/except` with a log. The detector matched the unrelated `async def handler` in
  `unified_events/core.py`. No defect.
- `nexus_search/nexus_client.py:207` `suggestions.insert(0, ...)` — `list.insert`, matched
  against `pattern_learner/pattern_store.py`'s `async def insert`. No defect.
