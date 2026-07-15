# Implementation Plan — AI Orchestrator × Spec-Kit Integration & Governance Hardening

**Version:** 1.0
**Date:** 2026-07-11
**Owner:** Orchestrator core
**Source analysis:** [`docs/spec-kit-vs-ai-orchestrator-report.md`](docs/spec-kit-vs-ai-orchestrator-report.md)
**Status:** Proposed — awaiting go/no-go per track

---

## 1. Executive Summary

The comparative analysis (`spec-kit-vs-ai-orchestrator-report.md`) established that AI Orchestrator has best-in-class **execution** (multi-model routing, cross-model critique, budget hierarchy, crash recovery) but three critical front-end gaps: **no structured specification phase (GAP-1)**, **no constitutional enforcement (GAP-2)**, and **no human-in-the-loop gates (GAP-3)**. Spec-Kit is the inverse — strong specification discipline, weak execution.

This plan does **not** rebuild a specification methodology from scratch. It adopts the report's Phase 11 recommendation: **ingest Spec-Kit's output as orchestrator input** through a thin adapter, and **activate the orchestrator's already-present-but-passive governance primitives** (`ProjectConstitution`) as real pipeline gates.

The work is decomposed into **6 independently shippable tracks**, ordered by value/effort. The two highest-ROI items — Spec-Kit Mode A ingestion and Constitutional enforcement — together discharge GAP-1 through GAP-5 and GAP-13 in ~5 weeks, without violating any of the Four Unbreakable Rules or the 5 import-linter contracts.

| Track | Discharges | Effort | Priority | Ship gate |
|-------|-----------|--------|----------|-----------|
| **0 — Fact/metric corrections** | doc drift | done | — | ✅ complete |
| **A — Spec-Kit Mode A ingestion** | GAP-1, 4, 5 | ~2 wk | Critical | independent |
| **B — Constitutional enforcement gate** | GAP-2 | ~2 wk | Critical | after A (shares config loader) |
| **C — Human gate stage** | GAP-3 | ~3 wk | High | after B |
| **D — Clarification workflow** | GAP-5 (deep) | ~2 wk | High | after A |
| **E — Shell-command safety guard** | GAP-13 | ~1 wk | Med (pre-emptive) | independent |
| **F — engine.py strangler cont.** | tech debt | ongoing | Med | independent |

**Recommended first cut:** Track A + Track B + Track E (~5 wk) — the governance-and-ingestion core. Defer C/D/F to a second milestone.

---

## 2. Current Architecture Assessment

### 2.1 Structure (verified against source, 2026-07-11)

Hexagonal / Ports & Adapters, enforced mechanically:

```
Interfaces (cli.py, api_server.py)
   → Infrastructure (llm_client, state/SQLite, cache, telemetry)
      → Application (engine.py Mediator, engine_core/ pipeline+stages, application/ services)
         → Domain (models.py, domain/ports.py Protocols, domain/constitution.py)
```

**Enforcement already in place** (constraints the plan must respect, not build):
- **5 import-linter contracts** (`.importlinter`): `domain-purity`, `application-no-concrete-infra`, `application-services-no-engine`, `engine-core-no-loose-infra`, `root-modules-no-infra`.
- **Coverage ratchet:** `fail_under = 7` (`pyproject.toml`) — every track must raise, never lower.
- **Strict mypy** on `domain/`, `application/`, `engine_core/`.
- **Bandit HIGH gate**, root-file freeze (`check_new_root_files.py`), config-drift check.

### 2.2 Integration seams the plan exploits

| Seam | Location | Verified shape | Used by |
|------|----------|----------------|---------|
| Task decomposition | `application/decomposer_service.py::decompose_project()` | returns `dict[str, Task]` (`{task.id: Task}`) | **Track A** drop-in |
| Governance value object | `domain/constitution.py::ProjectConstitution` | `protect_paths`, `require_review_above`, `require_tests`, `required_validators`, `forbidden_imports`, `max_file_size_bytes`; methods `is_path_protected()`, `is_import_forbidden()`, `validate()` | **Track B** enforcement source |
| Constitution loader | `infrastructure/constitution_loader.py::ConstitutionLoader` | loads `.orchestrator/constitution.json` | **Track A/B** |
| Pipeline stage protocol | `engine_core/pipeline.py::PipelineStage` (Protocol, `async def run(ctx)->ctx`) + `PipelineContext` (`task`, `output`, `score`, `model`, `should_abort`, `abort_reason`) | mutable context, abort channel already present | **Track B, C, E** |
| Service wiring | `engine_core/container.py::ServiceContainer.build()` | single factory, 30+ fields | all stage insertions |

### 2.3 Technical debt (from report, confirmed)

- `engine.py` ≈ 50 KB Mediator; ~325 lines of feature-flag conditional imports (report Appendix D).
- 200+ root-level modules (god-file pattern, acknowledged, Strangler Fig in progress).
- `ProjectConstitution` exists but is **passive** — loaded, never enforced at generation time.

---

## 3. Detailed Implementation Plan (per track)

### Track A — Spec-Kit Mode A Ingestion (Critical)

**Objective.** Let a user run `orchestrator run --from-speckit ./specs/001-feature/` so orchestrator consumes Spec-Kit artifacts (`spec.md`, `plan.md`, `tasks.md`, `.specify/memory/constitution.md`) instead of decomposing a raw prompt. Discharges GAP-1 (spec phase), GAP-4 (template constraint), GAP-5 (partial).

**Affected components.**
- NEW `orchestrator/ingest/speckit_adapter.py` (application layer — pure parsing, file read via a port).
- NEW `orchestrator/ingest/__init__.py`.
- `cli.py` — add `--from-speckit PATH` flag (interfaces layer).
- `engine.py` / `Orchestrator.run_project` — branch: when spec-kit tasks supplied, **skip** `decompose_project`.
- `infrastructure/constitution_loader.py` — extend to also parse `constitution.md` (markdown → `ProjectConstitution`) OR map to existing JSON loader.

**Design changes.**
- Adapter emits **exactly** `dict[str, Task]` (same contract as `decompose_project`) → zero pipeline changes downstream.
- `tasks.md` grammar (verified from `spec-kit-main/templates/tasks-template.md`): lines `[ID] [P?] [Story] Description`, file paths embedded. Parser extracts: `id`, `parallel: bool` (`[P]`), `story` group, `description`, `file_paths`.
- `plan.md` tech-context → optional routing bias (keyword → `ROUTING_TABLE` hint); **read-only, best-effort** (never hard-fail on parse miss).
- `spec.md` success criteria → appended to `Task` acceptance criteria consumed by EvaluateStage.

**Implementation tasks.**
1. Define `SpecArtifacts` dataclass (paths + parsed content).
2. `parse_tasks_md(text) -> list[Task]` — regex line parser, dependency edges from `[Story]` + explicit "depends on".
3. `parse_constitution_md(text) -> ProjectConstitution` — map documented articles/principles to the JSON fields (protect_paths from "Governance", forbidden_imports from "Anti-Abstraction", etc.); unmatched → warnings, never crash.
4. `SpecKitAdapter.load(dir) -> (dict[str,Task], ProjectConstitution, routing_hints)`.
5. CLI flag + wiring; `Orchestrator` branch to bypass decomposition.
6. File read via a `FileReaderPort` (keeps application layer pure — no direct `open()` in app code; contract-compliant).

**Refactoring requirements.** None to existing pipeline. Only additive. New code lives under `orchestrator/ingest/` (application subpackage) — compliant with Rule 4 "no new depth-1 root modules".

**Testing strategy.**
- Unit: `tests/test_speckit_adapter.py` — golden `tasks.md`/`constitution.md` fixtures → assert exact `dict[str,Task]` and `ProjectConstitution` fields. RED first.
- Contract: adapter output satisfies the same shape assertions as `decompose_project` (shared parametrized test).
- Integration: `tests/integration/test_from_speckit_flow.py` — end-to-end with mocked LLM, assert pipeline runs the ingested tasks (no decompose call — assert via spy).
- Fixtures sourced from real `spec-kit-main/templates/` to prevent format drift.

**Acceptance criteria.**
- [ ] `orchestrator run --from-speckit ./specs/NNN/` executes ingested tasks, `decompose_project` **not** called (asserted).
- [ ] Malformed `tasks.md` → clear error naming file+line, non-zero exit, no partial run.
- [ ] `constitution.md` fields land in a `ProjectConstitution` passed into the run.
- [ ] Coverage on new package ≥ 80% (raise ratchet).
- [ ] All 5 import-linter contracts still KEEP.

**Rollback.** Feature-flagged behind `--from-speckit` (opt-in). No flag → identical legacy behavior. Revert = delete `orchestrator/ingest/` + CLI flag; zero migration.

---

### Track B — Constitutional Enforcement Gate (Critical)

**Objective.** Make `ProjectConstitution` active: a `ConstitutionGate` pipeline stage that runs **before** `GenerateStage` and aborts tasks that would violate declared constraints. Discharges GAP-2.

**Affected components.**
- NEW `orchestrator/engine_core/stages/constitution_gate.py`.
- `engine_core/container.py::ServiceContainer.build()` — inject constitution + gate.
- `engine_core/pipeline.py` — register gate as first stage (before generate).
- `domain/constitution.py` — add convenience `check_task(task) -> list[str]` aggregator (pure; no new deps).

**Design changes.** Stage uses the **real, verified** API (not a fabricated `check()`):

```python
# orchestrator/engine_core/stages/constitution_gate.py
from ..pipeline import PipelineContext
from ...domain.constitution import ProjectConstitution

class ConstitutionGate:
    """Phase -1 gate: reject a task BEFORE spending generation tokens
    if it would violate declared project constraints."""

    def __init__(self, constitution: ProjectConstitution) -> None:
        self._c = constitution

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        violations: list[str] = []
        for path in getattr(ctx.task, "target_paths", []) or []:
            if self._c.is_path_protected(path):
                violations.append(f"writes protected path: {path}")
        for imp in getattr(ctx.task, "declared_imports", []) or []:
            if self._c.is_import_forbidden(imp):
                violations.append(f"forbidden import: {imp}")
        # require_tests / required_validators are appended to task hard-gates,
        # enforced downstream by ValidateStage (not aborted here).
        if violations:
            ctx.should_abort = True
            ctx.abort_reason = "constitution: " + "; ".join(violations)
        return ctx
```

- Uses existing `ctx.should_abort` / `ctx.abort_reason` — **no new pipeline plumbing**.
- `required_validators` / `require_tests` are **appended** to `task.hard_validators` at wiring time so the existing `ValidateStage` enforces them (DRY — reuse validation path, don't duplicate).

**Implementation tasks.**
1. `ProjectConstitution.check_task()` pure aggregator + unit tests (RED).
2. `ConstitutionGate` stage.
3. Wire in `ServiceContainer.build()`; append `required_validators` to each task's hard validators.
4. Register as first pipeline stage; verify abort short-circuits generate.
5. Emit a compliance event via existing `EventPort` (observability) on abort.

**Refactoring requirements.** None. Purely additive stage. `domain/constitution.py` stays I/O-free (Rule 2 — domain purity contract).

**Testing strategy.**
- Unit: protected-path abort, forbidden-import abort, clean-pass no-abort.
- Integration: task targeting a `protect_paths` glob → run aborts before generate (spy on GenerateStage).
- Contract: gate satisfies `PipelineStage` protocol (runtime_checkable assert).

**Acceptance criteria.**
- [ ] Task writing a protected path aborts with `abort_reason` naming the path; **zero** generation tokens spent (assert client not called).
- [ ] Forbidden import → abort.
- [ ] `required_validators` run on every task via ValidateStage.
- [ ] No-constitution / empty-constitution run behaves exactly as today.
- [ ] Coverage ≥ 80% new code; contracts KEEP; mypy strict passes on new stage.

**Rollback.** Gate is a no-op when constitution is empty (default). Wiring guarded by `constitution_enforcement_enabled` flag (default on for `--from-speckit`, off otherwise for first release). Revert = unregister stage.

---

### Track C — Human Gate Stage (High)

**Objective.** Formal approve/reject/retry pause points with resumable state (GAP-3).

**Affected components.** NEW `engine_core/stages/gate.py`; `state.py` (persist gate state); `cli.py` (`orchestrator resume <id> --approve|--reject`).

**Design.** `GateStage` sets `ctx.should_abort` with `abort_reason="awaiting_human:<gate_id>"`, persists `PipelineContext` snapshot via existing `StatePort.save_checkpoint()`, exits cleanly. Resume rehydrates and continues. Reuses Memento/checkpoint machinery already in `state.py` (`save_checkpoint`/`load_checkpoint`) — no new persistence layer.

**Testing.** Pause→persist→resume→complete round-trip; reject path aborts run; state survives process restart (SQLite).

**Acceptance.** Gate pauses, state persists, resume continues from exact task; reject terminates with status DEGRADED/ABORTED.

**Rollback.** Flag `human_gates_enabled` (default off). No gates configured → no pauses.

---

### Track D — Clarification Workflow (High)

**Objective.** Detect `[NEEDS CLARIFICATION]` markers (from Spec-Kit specs or orchestrator's own spec pass) and run structured Q&A before planning (GAP-5 deep).

**Design.** New `application/clarifier_service.py`: scans spec text for markers; for each, routes a targeted question to a cheap model (cost-aware, per phase policy); in autonomous mode auto-answers with best-guess + logs assumption; in interactive mode surfaces to user. Feeds resolved answers back into task acceptance criteria.

**Testing.** Marker extraction unit tests; auto-answer path deterministic with mocked LLM; assumptions logged to event bus.

**Acceptance.** Specs with markers block progression (interactive) or annotate assumptions (autonomous); zero unresolved markers reach GenerateStage.

**Rollback.** Flag `clarification_enabled`. Off → markers ignored (today's behavior).

---

### Track E — Shell-Command Safety Guard (Medium, pre-emptive)

**Objective.** Add a 4-tier risk classifier (`SAFE / SUSPICIOUS / DANGEROUS / BLOCKED`) gating any shell dispatch, before any feature executes generated commands (GAP-13).

**Affected components.** NEW `orchestrator/safety/command_guard.py` (pairs with existing `safety/generated_output_scanner.py`); any future shell executor imports it.

**Design.** Pure classifier: regex/allowlist rules → `RiskLevel` enum + rationale. `BLOCKED` raises; `DANGEROUS` requires explicit `allow_dangerous=True`. Secure-by-default (deny on unknown). No execution logic in this track — guard only.

**Testing.** Table-driven: known-safe (`ls`, `pytest`), suspicious (`curl | sh`), dangerous (`rm -rf`), blocked (fork bombs, `:(){ :|:& };:`). Bandit clean.

**Acceptance.** Classifier covers OWASP command-injection patterns; default-deny on unrecognized; 100% branch coverage (small surface).

**Rollback.** Additive module, imported by nothing yet — zero risk. Delete file to revert.

---

### Track F — engine.py Strangler Continuation (Medium, ongoing)

**Objective.** Continue extracting control flow from the 50 KB `engine.py` Mediator into `engine_core/` modules (report IMP-1). Not a gate for A–E.

**Design.** Per-extraction: identify a cohesive control-flow block → move to a named `engine_core/` module → `engine.py` delegates. One extraction per PR, each behind full regression suite. Respect `application-services-no-engine` contract.

**Acceptance per extraction.** Behavior-identical (golden run diff); `engine.py` shrinks; contracts KEEP; coverage non-decreasing.

**Rollback.** Each extraction is an isolated PR — revert individually.

---

## 4. Task Breakdown Structure (WBS)

```
1.0 Milestone 1 — Governance & Ingestion Core (~5 wk)
├── 1.1 Track A — Spec-Kit Mode A ingestion
│   ├── 1.1.1 SpecArtifacts + FileReaderPort            (0.5 d)
│   ├── 1.1.2 parse_tasks_md + tests (RED→GREEN)        (2 d)
│   ├── 1.1.3 parse_constitution_md + tests             (2 d)
│   ├── 1.1.4 SpecKitAdapter.load + tests               (1 d)
│   ├── 1.1.5 CLI --from-speckit + Orchestrator branch  (1.5 d)
│   ├── 1.1.6 integration test (decompose-skipped spy)  (1 d)
│   └── 1.1.7 docs + USAGE_GUIDE section                (0.5 d)
├── 1.2 Track B — Constitutional enforcement
│   ├── 1.2.1 ProjectConstitution.check_task + tests    (1 d)
│   ├── 1.2.2 ConstitutionGate stage + tests            (1.5 d)
│   ├── 1.2.3 ServiceContainer wiring + validator merge (1.5 d)
│   ├── 1.2.4 pipeline registration + abort integration (1 d)
│   └── 1.2.5 compliance event + observability          (0.5 d)
└── 1.3 Track E — Shell safety guard
    ├── 1.3.1 RiskLevel + rule table + tests            (2 d)
    └── 1.3.2 command_guard classifier + bandit         (1 d)

2.0 Milestone 2 — Human Collaboration (~5 wk)
├── 2.1 Track C — Human gate stage                      (3 wk)
└── 2.2 Track D — Clarification workflow                (2 wk)

3.0 Continuous — Track F strangler extractions          (per-PR, ongoing)
```

**Critical path:** 1.1 → 1.2 (B reuses A's constitution loader) → 2.1 → 2.2. Track E and F are parallelizable at any time.

---

## 5. Risk & Mitigation Matrix

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Spec-Kit `tasks.md`/`constitution.md` format drifts across versions | Med | Med | Golden fixtures pulled from pinned `spec-kit-main`; parser tolerant (warn, never crash); version-check the input dir's `.specify` |
| R2 | `ConstitutionGate` too aggressive → blocks legitimate tasks | Med | Med | Default-off except `--from-speckit`; empty constitution = no-op; abort_reason explicit + actionable |
| R3 | Adapter parsing produces malformed `Task` → downstream crash | Med | High | Strict validation at adapter boundary (fail-fast, named file+line); contract test shares `decompose_project` shape assertions |
| R4 | New stage violates an import-linter contract | Low | High | Stages live in `engine_core/stages/` (already contract-clean); CI lint-imports gate blocks merge |
| R5 | Coverage ratchet regression | Low | Med | Each track raises `fail_under`; new code ≥80% enforced in PR |
| R6 | Human-gate resume corrupts pipeline state | Low | High | Reuse proven `save_checkpoint`/`load_checkpoint`; round-trip tests incl. process restart |
| R7 | Scope creep: rebuilding Spec-Kit instead of ingesting | Med | High | Hard rule — adapter only; report IMP-2 (build-own) is explicitly the *fallback*, not the plan |
| R8 | License entanglement vendoring Spec-Kit templates | Low | Med | Confirmed MIT (`spec-kit-main/LICENSE`); Mode A reads user's own files (no vendoring); attribute in NOTICE if Mode B |

---

## 6. Testing & Quality Assurance Strategy

**Methodology:** TDD without exception (Rule 3) — RED (failing test named + committed) → GREEN → refactor → commit.

| Layer | Tool | Gate |
|-------|------|------|
| Unit | pytest `-m unit` | new code ≥80%, ratchet raised |
| Integration | pytest `-m integration`, mocked LLM | flows assert no real API calls |
| Contract | port/protocol conformance + shared-shape tests | adapter ≡ `decompose_project` shape |
| Static | mypy strict (domain/application/engine_core), ruff, black --check | zero errors |
| Security | bandit HIGH gate | zero HIGH on new code |
| Architecture | `lint-imports` (5 contracts), `check_new_root_files.py` | all KEEP, no depth-1 root modules |
| Config | `check_config_drift.py` | no drift introduced |

**Golden-fixture discipline:** Spec-Kit parser fixtures are copied from pinned `spec-kit-main/templates/` and checked in, so format drift surfaces as a test failure, not a production bug.

**Coverage ratchet plan:** A raises `fail_under` 7→8, B →9, E →10 (adjust to actuals; never lower). Rule: a PR that adds a module must add its tests in the same PR.

---

## 7. Deployment & Rollback Plan

**Delivery model:** every track is **feature-flagged and additive** — no migration, no breaking change to the default (`orchestrator run "<prompt>"`) path.

| Track | Flag | Default | Rollback |
|-------|------|---------|----------|
| A | `--from-speckit` (CLI, opt-in) | absent | delete `orchestrator/ingest/`, remove flag |
| B | `constitution_enforcement_enabled` | on only w/ `--from-speckit` | unregister stage / flag off |
| C | `human_gates_enabled` | off | flag off (no gates configured) |
| D | `clarification_enabled` | off | flag off (markers ignored) |
| E | (module, imported by nothing) | n/a | delete file |

**Deployment sequence per track:** merge behind flag → smoke run with mocked LLM → enable flag in a canary run against a sample Spec-Kit project → monitor telemetry (latency/cost/success EMA) → default-on decision documented in an ADR.

**Rollback trigger:** any contract failure, coverage regression, or >X% success-rate drop on canary → flag off (instant, no redeploy — flags are runtime).

**Observability on rollout:** each gate emits an `EventPort` event (constitution abort, human-gate pause, clarification assumption) → visible in existing telemetry, so enforcement actions are auditable.

---

## 8. Post-Implementation Validation Checklist

**Per-track gate (all must pass before default-on):**
- [ ] All new code has RED-first tests committed in history.
- [ ] Coverage ratchet raised; new modules ≥80%.
- [ ] `lint-imports` — all 5 contracts KEEP.
- [ ] `check_new_root_files.py` — no new depth-1 root modules.
- [ ] mypy strict clean on domain/application/engine_core additions.
- [ ] bandit — zero HIGH.
- [ ] `check_config_drift.py` — clean.
- [ ] Default path (`orchestrator run "<prompt>"`) behavior byte-identical with all flags off (golden run diff).

**Track A specific:**
- [ ] `--from-speckit` runs a real `spec-kit-main`-generated `specs/NNN/` dir end-to-end (mocked LLM).
- [ ] `decompose_project` provably not called on that path (spy asserted).
- [ ] Malformed input fails fast with file+line diagnostic.

**Track B specific:**
- [ ] Protected-path task aborts with **zero** generation tokens (client-call spy asserted).
- [ ] Forbidden-import task aborts; `required_validators` enforced via ValidateStage.

**Track E specific:**
- [ ] Classifier default-denies unknown commands; covers rm-rf / fork-bomb / pipe-to-shell.

**Documentation:**
- [ ] `USAGE_GUIDE.md` — `--from-speckit` section.
- [ ] `docs/CODEBASE_MINDMAP.md` — new `ingest/` package + `ConstitutionGate` stage added.
- [ ] ADR per default-on decision (A, B).
- [ ] `spec-kit-vs-ai-orchestrator-report.md` gap statuses updated (GAP-1/2/4/5/13 → addressed).

---

## Appendix — Traceability (report gap → track → acceptance)

| Report gap | Track | Closes when |
|-----------|-------|-------------|
| GAP-1 No spec phase | A | `--from-speckit` ingests spec/plan/tasks |
| GAP-2 No constitution enforcement | B | ConstitutionGate aborts violations pre-generation |
| GAP-3 No human gates | C | GateStage pause/resume round-trips |
| GAP-4 No template constraint | A | Spec-Kit templates ingested as task constraints |
| GAP-5 No clarification | A + D | markers resolved before generate |
| GAP-13 No shell guard | E | 4-tier classifier default-denies |
| IMP-1 engine.py debt | F | ongoing extractions |
