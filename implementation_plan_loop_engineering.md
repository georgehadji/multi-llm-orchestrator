# Implementation Plan — Loop Engineering Hardening

**Project:** Multi-LLM Orchestrator
**Source framework:** *Loop Engineering: The Anthropic Playbook for Designing Systems That Prompt Your Agents* (HuaShu Orange Books, v260615, June 2026)
**Author of plan:** Orchestrator engineering
**Date:** 2026-06-25
**Branch target:** `feat/response-healing` → cut feature branches per work item
**Status:** DRAFT — pending review

---

## 1. Executive Summary

The orchestrator is an autonomous *loop* in the precise sense defined by the Loop Engineering paper: it discovers work (decompose), hands it off (delegation), verifies (critique/evaluate), persists (`state.py`), and can run on a schedule (`operations/automations.py`). Mapping the paper's five moves and four silent costs onto the codebase surfaces **two latent defects and five enhancements**, all clustered around the paper's hardest problem: the **"say no" layer** — independent verification, an enforced human checkpoint, and instrumentation of the costs that accrue silently while a loop runs unattended.

The single highest-leverage finding: the evaluator (`orchestrator/services/evaluator.py`, `orchestrator/application/evaluator.py`) **reads and scores prose** with the prompt *"Score this output on a scale of 0.0 to 1.0"* and system role *"You are a precise evaluator."* It does not run tests, does not default to disbelief, and on the same-provider path risks grading homework it effectively wrote. The paper names this the **nodding loop** — the most common failure mode. Engine already enforces a different *provider* for cross-review (`engine.py:407`), which is good, but the evaluator must be upgraded from **read** to **act** (execute tests/lint as a boolean floor) and from **trust** to **assume-broken**.

Two latent defects compound the risk when the loop runs autonomously:

- **FIX-1 — Silent auto-approval.** `hitl/gate.py:64` auto-approves *every* decision ("no UI channel configured"). The paper's **cognitive surrender** cost is baked into the code: the one door that should pause for a human is welded open.
- **FIX-2 — Parallel without isolation.** `delegation/batch_runner.py` runs sub-agents concurrently in a **shared working directory** — no git worktree per task. The paper's **tangled loop** (handoff skipped): parallel edits collide and corrupt.

Already-resolved (verified during analysis): the previously catalogued cron bugs (weekday off-by-one, `*/0` division) are **fixed** in `operations/automations.py` — `(t.tm_wday + 1) % 7` and a `step == 0` guard. No action required.

This plan delivers, in four phases: the two fixes, an acting/adversarial evaluator with a deterministic test-floor, a fresh-model maker-checker stop condition, four-cost telemetry, unattended-run guardrails (caps + enforced checkpoint), and discovery-as-skill automations.

---

## 2. Current Architecture Assessment

### 2.1 Architecture style

Hexagonal (Ports & Adapters), documented in `CLAUDE.md` and `docs/CODEBASE_MINDMAP.md`:

- **Driving adapters:** `cli.py`, `api_server.py`, webhooks, tests.
- **Application core:** `engine.py` (Mediator — wires services only), domain services under `application/`, `engine_core/`.
- **Driven adapters:** LLM providers (`api_clients.py::UnifiedClient`), persistence (`state.py`, `checkpoints.py`), telemetry, cache.
- **Domain:** `models.py` — pure dataclasses + enums, no I/O.

Four unbreakable rules in force: engine = Mediator (logic → new service module), `models.py` pure, TDD (RED→GREEN), no new root-level modules. Import boundaries enforced in CI via `import-linter` (Phase A remediation complete).

### 2.2 Loop Engineering mapping (paper → code)

| Paper organ (move/part) | Current implementation | Maturity | Gap |
|---|---|---|---|
| **Discovery** (skills) | `skills/skills.py`; decomposition in engine | Good | Automations pass inline prompts, not named skills (ENH-5) |
| **Handoff** (worktrees) | `delegation/batch_runner.py`, `subagent.py` | **Defect** | No worktree isolation (FIX-2) |
| **Verification** (generator/evaluator split) | `services/evaluator.py`, `application/evaluator.py`; `engine.py:407` forces different provider | **Weak** | Reads not acts; trusts not doubts (ENH-1) |
| **Stop condition** (`/goal` maker-checker) | loop to `max_iterations` + score threshold; `_should_exit_early` | **Weak** | Generator self-score gates completion, not a fresh judge (ENH-2) |
| **Persistence** (memory) | `state.py` (aiosqlite, WAL), `checkpoints.py` (Memento), `memory/` | Good | — |
| **Scheduling** (automations) | `operations/automations.py` (cron/interval/event/webhook) | Good | Discovery-as-skill (ENH-5) |
| **Connectors** | `connectors/connectors.py` (MCP) | Good | — |
| **Human checkpoint** ("one door open") | `hitl/gate.py` | **Defect** | Auto-approves everything (FIX-1) |
| **Budget caps** (cap before ship) | `budget.py` (per-run `max_usd`), `cost.py` (`BudgetHierarchy` org/team/job) | Partial | No "all-caps-set" precondition for unattended runs; no idle max-retry cap (ENH-4) |
| **Four silent costs** | `telemetry/`, `capability_logger.py` | Missing | Not measured (ENH-3) |

### 2.3 Data flow (control loop)

`engine.py` primary loop: `decompose → [per task: generate → critique → revise → evaluate] (iterate ≤ max_iterations)`. Persistence to `~/.orchestrator_cache/state.db` after each task. `_evaluate` (`engine.py:1157`) delegates to `EvaluatorService.evaluate` → returns a `float`/`CritiqueReport`. Completion governed by score threshold + `BudgetEnforcer.should_exit_early`.

### 2.4 Technical debt relevant to this plan

- Evaluator self-consistency is multi-run but single-modality (LLM text scoring only).
- HITL has no transport (CLI/WebSocket) — stubbed to auto-approve.
- Parallel execution opt-in (`ORCH_BATCH_PARALLELISM=true`) but unguarded against filesystem races.
- No cost-of-autonomy observability surface despite existing dashboard (`dashboard_core/`).

### 2.5 Non-functional posture

- **Security:** secrets env-only; bash guardrails; generated-output scanner. HITL stub weakens approval gates for security-sensitive changes — **in scope**.
- **Scalability:** async throughout; circuit breaker + rate limiter present.
- **Maintainability:** strong module boundaries; import-linter enforced.

---

## 3. Detailed Implementation Plan

Four phases, ordered **safety → core value → observability → polish**. Each work item carries its own WBS in §4.

### Phase 0 — Stop the bleeding (safety defects + guardrails)
Low risk, high protection. Ships the two defect fixes and the unattended-run guardrails so that everything built later runs behind a real checkpoint and real caps.

- **FIX-1** Real HITL gate (replace silent auto-approve).
- **FIX-2** Worktree isolation for parallel sub-agents.
- **ENH-4** Unattended-run guardrails (caps-all-set precondition + enforced checkpoint).

**Milestone M0:** No loop can run unattended without (a) all budget caps set and (b) at least one human checkpoint reachable; parallel tasks cannot corrupt a shared tree.

### Phase 1 — The "say no" layer (core value)
The paper's central thesis. Depends on Phase 0 (FIX-1 supplies the checkpoint these reference).

- **ENH-1** Acting + adversarial evaluator (deterministic test/lint floor, "assume broken" stance).
- **ENH-2** Fresh-model maker-checker stop condition.

**Milestone M1:** Completion is decided by a separate, cheaper model on a deterministic pass (tests + lint), not by the generator's self-score; evaluator vetoes code that does not run.

### Phase 2 — Observability of autonomy cost
Depends on Phase 1 (verification debt needs the independent-check signal to count).

- **ENH-3** Four-cost telemetry + dashboard panels.

**Milestone M2:** Verification debt, comprehension rot, cognitive surrender, and token blowout are each a named gauge with thresholds.

### Phase 3 — Scheduling polish
Independent; lowest urgency.

- **ENH-5** Discovery-as-skill automations.

**Milestone M3:** Scheduled triggers invoke a named, maintained skill rather than an inline prompt wall.

### Dependency graph

```
FIX-1 ──┐
FIX-2   ├─ Phase 0 (M0) ──► ENH-1 ──► ENH-2  (Phase 1, M1) ──► ENH-3 (Phase 2, M2)
ENH-4 ──┘
ENH-5 ── independent (Phase 3, M3)
```

---

## 4. Task Breakdown Structure (WBS)

> Each item: Objective · Affected components · Design changes · Implementation tasks · Refactoring · Testing · Acceptance · Rollback.

### FIX-1 — Replace silent HITL auto-approval

- **Objective:** The human-checkpoint gate must actually pause (or fail closed) for critical decisions instead of auto-approving. Removes baked-in cognitive surrender.
- **Affected components:** `orchestrator/hitl/gate.py`; consumers calling `HumanInTheLoop.request_decision`; `api_server.py` / `cli.py` (decision transport); config.
- **Design changes:**
  - Introduce a `DecisionChannel` Protocol (port) with `ask(decision, timeout) -> DecisionResult`. Adapters: `CLIDecisionChannel`, `WebSocketDecisionChannel`, `AutoApproveChannel` (explicit, dev-only).
  - `HumanInTheLoop` receives a channel via DI (constructor). **Default policy fails closed**: if no channel configured and decision `requires_approval`, return `REJECTED` (not `APPROVED`) unless `ORCH_HITL_AUTOAPPROVE=true` is explicitly set.
  - Add `policy` enum: `BLOCK` (wait), `FAIL_CLOSED` (reject on no-channel), `AUTO` (explicit opt-out).
- **Implementation tasks:**
  1. Define `DecisionChannel` Protocol in `hitl/`.
  2. Implement CLI channel (prompt) and WebSocket channel (reuse `api_server` ws).
  3. Rewire `request_decision` to delegate to channel; remove `WARNING auto-approved` default.
  4. Wire channel through `engine_core/container.py`.
  5. Config flag + `.env.example` entry, documented as dev-only.
- **Refactoring:** Move auto-approve into an explicit named adapter; no behavioral default change without config.
- **Testing:** RED — assert `request_decision` on a `requires_approval` decision with no channel returns `REJECTED`. GREEN — implement fail-closed. Integration — CLI channel approve/reject/timeout paths. Security — confirm security-category decisions cannot pass without explicit approval.
- **Acceptance:** No code path silently auto-approves; unattended runs without a channel fail closed; dev opt-out requires explicit env flag.
- **Rollback:** Single feature flag `ORCH_HITL_AUTOAPPROVE=true` restores legacy behavior; revert is one module.

### FIX-2 — Worktree isolation for parallel sub-agents

- **Objective:** Each concurrently-executing sub-agent operates in its own git worktree so parallel edits cannot collide (tangled-loop prevention).
- **Affected components:** `orchestrator/delegation/batch_runner.py`, `orchestrator/delegation/subagent.py`; `workspace/`/`vcs/` helpers; cleanup hooks.
- **Design changes:**
  - Add `WorktreeManager` (in `vcs/` or `workspace/`): `create(task_id) -> path`, `remove(path)`, context-manager form.
  - `BatchRunner._run_one` acquires a worktree per task, injects its path as the sub-agent CWD, and removes it on completion (auto-clean if unchanged; preserve on failure for inspection).
  - Guard: only engage when repo is a git work tree; otherwise fall back to a per-task temp dir with a logged warning.
- **Implementation tasks:**
  1. `WorktreeManager` with create/remove + `git worktree` invocation behind `vcs/`.
  2. Thread worktree path into `SubAgent.execute` as working directory.
  3. Cleanup on success; quarantine on failure; cap stale worktrees.
  4. Honor existing `.claude/worktrees/` gitignored convention.
- **Refactoring:** Extract any hardcoded CWD assumptions in `subagent.py` to injected path.
- **Testing:** RED — two tasks writing the same relative file in parallel must not interfere (assert each worktree's file is independent). Integration — N concurrent tasks each get distinct paths; cleanup verified. Failure path — worktree preserved on sub-agent error.
- **Acceptance:** Under `ORCH_BATCH_PARALLELISM=true`, no two sub-agents share a working directory; merge/collection step is deterministic.
- **Rollback:** Feature flag `ORCH_WORKTREE_ISOLATION` (default on after bake); disabling reverts to shared-CWD legacy path.

### ENH-1 — Acting + adversarial evaluator (test/lint floor)

- **Objective:** Upgrade the evaluator from reading prose to executing the artifact, and from trusting to assuming-broken. Establishes a deterministic boolean veto beneath the LLM quality score.
- **Affected components:** `orchestrator/services/evaluator.py`, `orchestrator/application/evaluator.py`, `engine.py::_evaluate` (`engine.py:1157`); reuse `app_verifier.py`, `browser_testing.py`, `validators.py`.
- **Design changes:**
  - Introduce a `VerificationGate` (Chain of Responsibility, consistent with existing `validators.py`/`preflight.py`): deterministic checks (run tests, run lint, type-check) → returns `pass/fail + reasons`. **Fail = hard veto** (score capped at a configurable floor, e.g. ≤ 0.2) regardless of LLM opinion.
  - Adversarial prompt: change system role to *"Adversarial reviewer. ASSUME this output is BROKEN until proven otherwise. Do not praise. Find what fails."* Default stance = doubt.
  - Compose: `final = deterministic_gate ? llm_quality_score : capped_floor`.
  - Keep self-consistency multi-run for the LLM layer.
- **Implementation tasks:**
  1. `VerificationGate` service with pluggable deterministic checks (tests/lint/type), each sandboxed via existing bash guardrails.
  2. Swap evaluator system/user prompts to adversarial "assume broken."
  3. Wire gate result into `EvaluatorService.evaluate` aggregation; expose `CritiqueReport.deterministic_pass: bool`.
  4. Container wiring; opt-in `ORCH_VERIFY_ACTS=true` then default-on after bake.
- **Refactoring:** Centralize scoring composition in one place; remove implicit trust in single LLM float.
- **Testing:** RED — feed loop a known-broken artifact (failing test); assert evaluator returns `deterministic_pass=False` and score ≤ floor. RED — feed lint-violating but functional code; assert lint veto. GREEN — implement gate. Regression — passing artifact still scores via LLM. Property — adversarial prompt does not inflate scores vs baseline on a fixture set.
- **Acceptance:** No artifact that fails tests/lint can score above the floor; evaluator stance is doubt-by-default; different-provider rule (`engine.py:407`) preserved.
- **Rollback:** `ORCH_VERIFY_ACTS=false` reverts to LLM-only scoring; gate is additive.

### ENH-2 — Fresh-model maker-checker stop condition

- **Objective:** Completion ("done") decided by a separate, cheaper model judging a deterministic pass — not the generator's self-score. Maker ≠ checker.
- **Affected components:** `engine.py` loop / `_should_exit_early` (`engine.py:1187`), `application/budget_enforcer.py`, `ModelCascadeService`, `engine_core/`.
- **Design changes:**
  - Add `CompletionJudge` service: given task + artifact + `VerificationGate` result, a model from the cheapest cascade tier (FREE/BUDGET) returns `done: bool + reasons`. Generator stays on premium tier.
  - Stop when: `deterministic_gate.pass AND completion_judge.done`. Score threshold becomes advisory, not sole gate.
  - Maker-checker invariant: judge model ≠ generator model (assert at wiring).
- **Implementation tasks:**
  1. `CompletionJudge` reading the gate result, prompted on "is the stop condition met" (analogous to `/goal`).
  2. Integrate into loop exit decision alongside `should_exit_early`.
  3. Enforce judge-tier = cheapest available; assert distinct from generator.
  4. Telemetry: record judge verdict + cost per turn.
- **Refactoring:** Decouple "looks reasonable" (LLM score) from "is right" (deterministic + judge) in the exit logic.
- **Testing:** RED — generator self-scores high but tests fail → loop must NOT exit complete. GREEN — judge + gate block completion. Integration — judge runs on cheap tier; cost recorded. Invariant — judge model id ≠ generator model id.
- **Acceptance:** Loop completion requires deterministic pass + independent fresh-model approval; generator cannot self-certify done.
- **Rollback:** `ORCH_FRESH_STOP=false` reverts to threshold-only exit.

### ENH-3 — Four-cost telemetry

- **Objective:** Make the paper's four silent costs first-class, measured gauges with thresholds and dashboard panels.
- **Affected components:** `telemetry/`, `capability_logger.py`, `dashboard_core/`, `events/`.
- **Design changes:** Four gauges via the existing Observer/EventBus:
  - `verification_debt` = count of merges/completions without `deterministic_pass=True` (from ENH-1).
  - `comprehension_rot` = ratio of generated output bytes to human-reviewed bytes (HITL-reviewed from FIX-1).
  - `cognitive_surrender` = auto-approve / total-decision rate (HITL channel data).
  - `token_blowout` = idle/retry spend ÷ estimated spend (budget + retry counters).
  - Each emits an event; dashboard subscribes; thresholds configurable.
- **Implementation tasks:**
  1. Define four metric events in `events/`.
  2. Emit from evaluator (verification debt), HITL (surrender), reviewer flow (rot), budget/retry (blowout).
  3. Aggregator service computing rolling windows.
  4. Four dashboard panels + threshold alerts.
- **Refactoring:** Route all four through one `AutonomyCostCollector` to keep emission DRY.
- **Testing:** Unit — each gauge computes correctly from synthetic events. Integration — a simulated unattended run produces non-zero gauges; thresholds trip. Dashboard — panels render from event stream.
- **Acceptance:** Four gauges visible, thresholded, and populated by real loop events; no gauge hardcoded.
- **Rollback:** Telemetry is additive/observational; disable collector via flag with zero functional impact.

### ENH-4 — Unattended-run guardrails (caps + door)

- **Objective:** Enforce the paper's "cap before you ship" and "keep one door open" disciplines as preconditions for any unattended run.
- **Affected components:** `budget.py`, `cost.py` (`BudgetHierarchy`), `engine.py` run entry, `operations/automations.py`, `hitl/gate.py` (from FIX-1).
- **Design changes:**
  - `UnattendedGuard` precondition checked at run start when no interactive session: require per-run cap (`budget.max_usd`), daily/cross-run cap (`BudgetHierarchy`), and `max_retries` all set to finite values; require ≥1 reachable human checkpoint (HITL channel configured) OR explicit `ORCH_NO_CHECKPOINT_ACK=true`.
  - Add idle/retry cap: a per-run `max_retries` ceiling that converts open-ended retry spend into a bounded one (circuit-breaker semantics).
  - Fail closed with an actionable error listing which cap/door is missing.
- **Implementation tasks:**
  1. `UnattendedGuard.validate(run_context)` returning structured missing-requirements.
  2. Invoke at `run_project` entry and at automation trigger.
  3. Add `max_retries` ceiling to budget enforcement.
  4. Document required env/config for autonomous operation.
- **Refactoring:** Consolidate scattered cap checks behind the guard (DRY).
- **Testing:** RED — start unattended run with daily cap unset → guard rejects with named missing cap. RED — no checkpoint + no ack → rejected. GREEN — all set → proceeds. Integration — automation trigger honors guard.
- **Acceptance:** An unattended run cannot start without all caps set and a checkpoint reachable (or explicit ack); idle bug cannot burn unbounded spend.
- **Rollback:** Guard behind `ORCH_UNATTENDED_GUARD` (default on); disabling restores legacy unguarded start.

### ENH-5 — Discovery-as-skill automations

- **Objective:** Scheduled triggers invoke a named, version-controlled skill (`SKILL.md`-style) rather than an inline prompt wall, so discovery logic is maintained, not rotting in a cron entry.
- **Affected components:** `operations/automations.py`, `skills/skills.py`, `connectors/`.
- **Design changes:** `ScheduledTask` references a `skill_name` + inputs instead of a raw prompt; scheduler resolves and invokes the registered skill; discovery findings written to state file/board (persistence).
- **Implementation tasks:**
  1. Add `skill_name` field to `ScheduledTask`; deprecate inline-prompt path with a warning.
  2. Scheduler resolves skill from `skills/` registry at fire time.
  3. Findings persisted via existing state writer.
- **Refactoring:** Replace any inline-prompt automations with named skills.
- **Testing:** Unit — scheduled fire invokes resolved skill, not raw prompt. Integration — missing skill name fails fast with clear error. Regression — cron/interval matching unchanged.
- **Acceptance:** Automations trigger named skills; inline-prompt path deprecated; findings persisted.
- **Rollback:** Inline-prompt path retained behind deprecation flag for one release.

---

## 5. Risk & Mitigation Matrix

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | FIX-1 fail-closed default blocks existing automated flows that relied on silent approval | High | High | Ship behind `ORCH_HITL_AUTOAPPROVE` opt-out; communicate breaking change; default-on after one bake cycle; document in release notes |
| R2 | ENH-1 deterministic gate slows the loop (running tests every turn costs time) | Medium | Medium | Cache test results by artifact hash; run lint/type before full test; gate opt-in then default-on; budget the gate's own cost |
| R3 | FIX-2 worktree creation overhead / disk pressure under high parallelism | Medium | Medium | Cap concurrent worktrees; auto-clean unchanged; reuse `max_concurrent`; temp-dir fallback off git |
| R4 | ENH-2 fresh judge on cheap tier under-judges (false "done") | Medium | High | Judge only *after* deterministic gate passes; gate is the hard floor; self-consistency on judge; alert on judge/gate disagreement |
| R5 | Backward compatibility: callers expecting `_evaluate -> float` break on `CritiqueReport` shape | Medium | Medium | Keep float-returning facade; add structured fields additively; type-checked at boundary |
| R6 | Token cost increases (extra judge call + multi-run evaluator) | High | Medium | Cheapest cascade tier for judge; cap eval runs; track via `token_blowout` gauge (ENH-3); per-run budget enforced |
| R7 | Telemetry gauges miscount and mislead operators | Low | Medium | Unit-test each computation against synthetic events; mark advisory until validated |
| R8 | Scope creep across seven items destabilizes `feat/response-healing` | Medium | High | One feature branch per work item; phase gating; CI green before merge; import-linter contracts must stay KEPT |
| R9 | Architectural drift — logic added to `engine.py` instead of services | Low | High | Enforce Rule 1 (engine wires only); new logic → new service module; code review checklist |

**Architectural constraints honored:** no new root-level modules (all under existing subpackages); `models.py` stays pure; engine wires services only; import-linter contracts must remain KEPT in CI.

---

## 6. Testing & Quality Assurance Strategy

### 6.1 TDD per the four unbreakable rules

Every item starts RED. Representative failing tests are specified per WBS item in §4. No implementation merges without its originating failing test now passing plus regression suite green.

### 6.2 Test layers

- **Unit** (`-m unit`): VerificationGate checks, CompletionJudge invariant (judge ≠ generator), gauge computations, UnattendedGuard validation, Cron/skill resolution.
- **Integration** (`-m integration`): full loop with acting evaluator on a known-broken fixture; parallel worktree isolation; automation → skill invocation; HITL channel approve/reject/timeout.
- **Security:** security-category HITL decisions cannot pass without explicit approval; bash guardrails wrap deterministic test execution; no secrets in telemetry.
- **Property / adversarial:** adversarial evaluator prompt does not inflate scores vs baseline fixtures; idle-retry cap bounds spend.

### 6.3 Coverage & gates

- Target ≥ 80% on new/changed modules (per project testing rules).
- Run `pytest tests/ -m unit` and `-m integration` (avoid full dir — 200+ mixed files per `CLAUDE.md`).
- Quality gates per change: `ruff check`, `black --check`, `mypy`, `bandit -r orchestrator/`, import-linter contracts KEPT.

### 6.4 Key acceptance fixtures

1. **Nodding-loop fixture:** code with a failing unit test → evaluator must veto (score ≤ floor, `deterministic_pass=False`) and loop must not complete.
2. **Tangled-loop fixture:** two tasks editing the same relative path in parallel → independent worktrees, no cross-contamination.
3. **Surrender fixture:** unattended run with no HITL channel → fail closed unless explicit ack.
4. **Blowout fixture:** induced retry storm → `max_retries` ceiling bounds spend; `token_blowout` gauge trips.

---

## 7. Deployment & Rollback Plan

### 7.1 Branching & CI

- One feature branch per work item off `feat/response-healing` (or `master` per workflow). Worktree isolation for parallel dev (`.claude/worktrees/`).
- CI must be green on all gates (Lint, Type Check, Security Scan, Architecture Boundaries, Test) before merge — matching the existing PR-gate pipeline.

### 7.2 Phased rollout with feature flags

Every behavioral change is flag-guarded and ships **opt-in → bake → default-on**:

| Item | Flag | Initial | After bake |
|---|---|---|---|
| FIX-1 | `ORCH_HITL_AUTOAPPROVE` | legacy via flag | fail-closed default |
| FIX-2 | `ORCH_WORKTREE_ISOLATION` | on (temp-dir fallback) | on |
| ENH-1 | `ORCH_VERIFY_ACTS` | off | on |
| ENH-2 | `ORCH_FRESH_STOP` | off | on |
| ENH-3 | (observational) | on | on |
| ENH-4 | `ORCH_UNATTENDED_GUARD` | on | on |
| ENH-5 | deprecation flag | inline allowed | inline deprecated |

### 7.3 Rollback

- Each item reverts via its single flag (no schema migration in any item — telemetry uses event bus, not new tables).
- `state.db` untouched structurally; checkpoints (Memento) allow run-state recovery.
- Revert order = reverse dependency order: ENH-2 → ENH-1 → ENH-4/FIX-2/FIX-1.

### 7.4 Observability during rollout

ENH-3 gauges double as rollout canaries: a spike in `verification_debt` or `cognitive_surrender` after enabling a flag signals regression — halt and revert.

---

## 8. Post-Implementation Validation Checklist

- [ ] No code path silently auto-approves a `requires_approval` decision (FIX-1).
- [ ] Security-category decisions require explicit human approval; verified by test.
- [ ] Two parallel sub-agents never share a working directory; collisions impossible (FIX-2).
- [ ] Failed sub-agent worktrees quarantined, not auto-deleted; stale count capped.
- [ ] Evaluator vetoes any artifact failing tests or lint (score ≤ floor) (ENH-1).
- [ ] Evaluator system prompt is adversarial ("assume broken"); baseline score inflation eliminated.
- [ ] Different-provider cross-review rule (`engine.py:407`) still holds.
- [ ] Loop completion requires deterministic pass + fresh-model approval; generator cannot self-certify (ENH-2).
- [ ] Judge model id asserted ≠ generator model id; judge on cheapest tier.
- [ ] Four cost gauges populated by real events, thresholded, visible on dashboard (ENH-3).
- [ ] Unattended run blocked unless all caps set + checkpoint reachable (or explicit ack) (ENH-4).
- [ ] `max_retries` ceiling bounds idle/retry spend; verified by blowout fixture.
- [ ] Scheduled automations invoke named skills; inline-prompt path deprecated (ENH-5).
- [ ] All new/changed modules ≥ 80% coverage; `-m unit` and `-m integration` green.
- [ ] `ruff`, `black --check`, `mypy`, `bandit` clean; import-linter contracts KEPT.
- [ ] No new root-level modules; `models.py` pure; `engine.py` wires services only.
- [ ] Every behavioral change flag-guarded with documented rollback.
- [ ] Release notes document FIX-1 fail-closed breaking change.

---

## Appendix A — Code anchors

| Concern | File:line |
|---|---|
| Engine evaluate delegation | `orchestrator/engine.py:1157` |
| Different-provider cross-review rule | `orchestrator/engine.py:407` |
| Early-exit logic | `orchestrator/engine.py:1187` |
| LLM-only evaluator (services) | `orchestrator/services/evaluator.py:97` |
| LLM-only evaluator (application) | `orchestrator/application/evaluator.py:98` |
| HITL silent auto-approve | `orchestrator/hitl/gate.py:64` |
| Parallel exec, no isolation | `orchestrator/delegation/batch_runner.py:142` |
| Per-run budget cap | `orchestrator/budget.py:44` |
| Cron parser (bugs already fixed) | `orchestrator/operations/automations.py:54` |
| Scheduled task model | `orchestrator/operations/automations.py:29` |

## Appendix B — Source mapping (paper → plan)

- Five moves (Discovery/Handoff/Verification/Persistence/Scheduling) → §2.2 table.
- Generator/evaluator separation (§V) → ENH-1, ENH-2.
- Five anti-patterns (§VI): nodding → ENH-1; tangled → FIX-2; amnesiac → persistence (already covered); manual/blind → ENH-5.
- Four silent costs (§VIII) → ENH-3.
- Three disciplines (§XI): read a sample → comprehension_rot gauge; cap before ship → ENH-4; keep one door open → FIX-1 + ENH-4.
- First-Loop Checklist (Table VI) → §8 validation checklist.
