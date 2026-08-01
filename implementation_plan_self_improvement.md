# AI Orchestrator — Evidence-Driven Self-Improvement Implementation Plan

**Date:** 2026-07-22  
**Status:** Proposed  
**Target:** Multi-LLM Orchestrator v6.x → v7.x  
**Scope:** Paper-derived reliability, harness evolution, evaluator co-evolution, and autonomous-loop hardening

## 1. Executive Summary

The orchestrator has a strong foundation: hexagonal boundaries, protocol-based dependency injection, an asynchronous pipeline, SQLite-backed checkpoints, model routing and fallback, circuit breakers, telemetry, a generate–critique–revise cycle, deterministic-verification abstractions, worktree support, human-in-the-loop gates, and trajectory-based skill optimization.

The main limitation is not the absence of components; it is that several components do not yet form a closed, evidence-based control loop. The evaluator can still be configured without deterministic checks, skill improvement uses a validation proxy instead of executing candidate harnesses, completion is not consistently gated by a fresh independent judge, and the system does not yet version evaluator criteria or preserve evaluator-independent evidence.

The papers point to one governing principle:

> An autonomous improvement must be linked to observable failure evidence, tested against held-out behavior, and promoted only when an independent verification layer confirms that it did not regress.

The recommended target architecture is:

```text
Discovery
  → isolated handoff
  → generate / act
  → deterministic verification
  → independent evaluation
  → durable evidence and lineage
  → controlled harness/evaluator improvement
  → held-out regression gate
  → promotion or rollback
  → scheduled next epoch
```

Implementation should proceed in four releases:

| Release | Outcome | Priority |
|---|---|---:|
| R1 | Safe completion: mandatory verification, fresh judge, artifact receipts, isolation defaults | P0/P1 |
| R2 | Evidence loop: structured traces, failure taxonomy, real held-out regression, candidate lineage | P1 |
| R3 | Evolvable harness: typed processors, variant routing, model-specific harness profiles | P1/P2 |
| R4 | Controlled co-evolution: evaluator epochs, anchor suites, challenger evaluators, scheduled discovery | P2 |

No release should weaken the five import-boundary contracts, add business logic to `engine.py`, or make `models.py` behavioral. Every implementation item begins with a failing test and ends with the required CI sequence.

## 2. Inputs and Evidence Base

This plan synthesizes the repository architecture and the following local papers:

- [The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](<E:/Downloads/Papers%20AI/The%20Red%20Queen%20Gödel%20Machine%20Co-Evolving%20Agents%20and%20Their%20Evaluators.pdf>)
- [Self-Harness: Harnesses That Improve Themselves](<E:/Downloads/Papers%20AI/Self-Harness%20Harnesses%20That%20Improve%20Themselves.pdf>)
- [HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry](<E:/Downloads/Papers%20AI/HarnessX.pdf>)
- [How Coding Agents Fail Their Users](<E:/Downloads/Papers%20AI/How%20Coding%20Agents%20Fail%20Their%20Users%20-%20A%20Large-Scale%20Analysis%20of.pdf>)
- [Loop Engineering](<E:/Downloads/Papers%20AI/Loop-Engineering-IEEE.pdf>)

The papers are design evidence, not production specifications. Their benchmark gains must not be assumed to transfer directly to this repository. The implementation therefore adopts their mechanisms and requires repository-specific fixtures, measurements, and rollback controls.

## 3. Current Architecture Assessment

### 3.1 Architecture and boundaries

The repository is an asynchronous Python 3.10+ orchestrator organized around:

```text
Driving adapters (CLI/API/dashboard)
        ↓
Orchestrator mediator / project runner
        ↓
Engine-core pipeline and stages
        ↓
Application services
        ↓
Domain models, ports, policies
        ↓
Infrastructure adapters (LLM, SQLite, cache, execution, telemetry)
```

The architecture has five enforced import contracts. They are valuable constraints for this work:

1. Domain and models remain infrastructure-free.
2. Application services depend on ports, not concrete infrastructure.
3. Engine-core orchestration does not import loose infrastructure.
4. `container.py` remains the composition root.
5. Root-level compatibility shims must not become new business-logic locations.

New behavior belongs in application services, domain value objects, ports, or engine-core stages. `engine.py` should only coordinate those services.

### 3.2 Current data flow

The principal task flow is:

```text
project request
  → decomposition / task DAG
  → routing and budget reservation
  → context enrichment
  → generation
  → critique / revision cycle
  → evaluator
  → task result and trajectory recording
  → checkpoint persistence
  → project completion or resumption
```

Existing integration points include:

- `orchestrator/application/evaluator.py` — LLM scoring and critique reports.
- `orchestrator/application/verification_gate.py` — deterministic check composition.
- `orchestrator/application/critique_cycle.py` — iterative generation and revision.
- `orchestrator/application/skill_optimizer.py` and `skill_manager.py` — trajectory-driven skill updates.
- `orchestrator/engine_core/pipeline_executor.py` and stages — pipeline execution and trajectory collection.
- `orchestrator/delegation/batch_runner.py` — parallel task execution.
- `orchestrator/vcs/worktree_manager.py` — task worktree isolation capability.
- `orchestrator/infrastructure/state.py` — SQLite persistence and checkpoints.
- `orchestrator/telemetry.py`, tracing, and event modules — metrics and operational signals.
- `orchestrator/hitl/gate.py` — approval and fail-closed decision handling.

### 3.3 Strengths

| Area | Current strength |
|---|---|
| Dependency management | Ports and adapters provide test seams and future backend substitution. |
| Resilience | Circuit breakers, fallback chains, retries, and budget controls exist. |
| Recovery | Checkpoints and resumable project state exist. |
| Verification seam | `VerificationGate` can enforce deterministic checks without coupling the evaluator to infrastructure. |
| Parallel execution | `BatchRunner` and `WorktreeManager` provide an isolation seam. |
| Human safety | HITL gate and unattended-run guard enforce caps and checkpoints. |
| Learning substrate | Trajectories, skill epochs, negative feedback, and skill persistence exist. |
| Testing | Unit, integration, contract, regression, and architecture tests are already separated. |

### 3.4 Gaps and technical debt

1. **Verification is not yet an invariant.** `VerificationGate.default()` may contain no checks, so an LLM score can remain the effective completion signal.
2. **Candidate skill validation is not behavioral.** `_estimate_val_score()` derives a validation score from existing trajectory scores and an edit-size bonus rather than running the proposed candidate on a fixed validation suite.
3. **Completion independence is incomplete.** A fresh-model maker–checker decision is not consistently required after deterministic verification.
4. **Harness state is fragmented.** Prompts, skills, tools, policies, memory, and recovery logic are not represented as one typed, versioned, substitutable `HarnessSpec`.
5. **Evaluator criteria lack epoch lineage.** Scores do not universally record the evaluator version and the evidence on which the score depends.
6. **Trace evidence is incomplete for alignment.** The system records output and score, but needs explicit planned actions, actual actions, constraints, scope, receipts, user corrections, and failure causes.
7. **Parallel isolation is available but must be proven and made policy-driven.** `BatchRunner` can receive a worktree manager; configuration must ensure isolation is enabled whenever concurrent tasks can mutate a repository.
8. **SQLite is a deliberate scale boundary.** It is suitable for single-instance operation but needs clear write serialization, project locking, and a documented PostgreSQL/event-log migration path.
9. **Large compatibility surface.** The repository contains a very large mediator, legacy shims, duplicated modules, and multiple planning documents. Changes must be additive and localized.

## 4. Target Architecture

### 4.1 New domain concepts

Add pure data types under `orchestrator/domain/` or an existing domain data module:

- `VerificationResult`: check results, reasons, duration, artifact identity.
- `EvaluationRecord`: score, critique items, evaluator version, model, confidence, evidence references.
- `TraceRecord`: normalized task/run/turn/tool/action/result events.
- `FailureSignature`: symptom, cause, verifier outcome, causal mechanism.
- `HarnessVersion`: immutable version, parent, change manifest, target failure signatures.
- `EvaluatorEpoch`: frozen evaluator criteria, anchor suite, start/end, promotion decision.
- `PromotionDecision`: candidate, baseline, held-in result, held-out result, regression status, approver.

These types must contain no file I/O, network calls, asyncio, or business behavior.

### 4.2 Application services

Create or extend services behind ports:

```text
TraceRecorder              normalize and persist execution evidence
FailureMiningService        cluster verifier-grounded recurring failures
HarnessCandidateService     propose bounded candidate changes
HarnessValidationService    execute candidates on held-in/held-out suites
PromotionService            apply conservative acceptance rules
IndependentJudgeService     fresh-model completion decision
EvaluatorEpochService       freeze, calibrate, and promote evaluators
AlignmentEvidenceService    compare requirements, actions, receipts, and claims
```

The services should be independently testable and orchestrated by existing pipeline services. Avoid creating a second top-level engine.

### 4.3 Infrastructure adapters

Implement adapters only where needed:

- `TraceStorePort` backed initially by SQLite.
- `EvaluationSuitePort` for deterministic fixtures and repository test commands.
- `WorktreePort` backed by the existing VCS manager.
- `HarnessStorePort` for immutable versions, manifests, and rollback.
- `EvaluatorAnchorStorePort` for evaluator-independent calibration data.
- Optional PostgreSQL adapters after the SQLite contract is stable.

### 4.4 Promotion flow

```text
baseline harness/evaluator
  → execute fixed tasks
  → collect full traces and verifier outcomes
  → mine failure signatures
  → propose K minimal candidates
  → validate every candidate in isolated workspace
  → held-in must improve or remain stable
  → held-out must not regress
  → independent judge and deterministic gate must pass
  → persist immutable lineage
  → promote, or retain baseline
```

## 5. Prioritized Roadmap and Dependencies

### Milestone M0 — Safety invariants

**Dependencies:** none.  
**Deliverables:** mandatory deterministic verification, independent completion judge, artifact receipts, isolation policy, fail-closed autonomous operation.

### Milestone M1 — Evidence substrate

**Dependencies:** M0.  
**Deliverables:** normalized traces, failure signatures, requirement ledger, evaluator/version metadata, persisted evidence references.

### Milestone M2 — Real harness evolution

**Dependencies:** M1.  
**Deliverables:** candidate harness versions, held-in/held-out execution, conservative promotion, rollback and negative-feedback storage.

### Milestone M3 — Composable variants

**Dependencies:** M2.  
**Deliverables:** typed processors, variant isolation/routing, model-specific harness profiles, per-task compatibility tests.

### Milestone M4 — Controlled evaluator co-evolution

**Dependencies:** M1 and M2.  
**Deliverables:** frozen evaluator epochs, anchor suites, challenger promotion, selective invalidation of evaluator-dependent scores, adversarial sample pools.

### Milestone M5 — Operational scale

**Dependencies:** M0–M4 and measured workload data.  
**Deliverables:** dashboards, alerting, scheduled discovery skills, SQLite contention controls, PostgreSQL adapter decision, event-log migration plan.

## 6. Detailed Work Breakdown

### WBS-1 — Mandatory acting verification

**Priority:** P0  
**Objective:** Prevent an LLM from approving an artifact that fails deterministic checks.

**Affected components:**

- `application/verification_gate.py`
- `application/evaluator.py`
- `application/validators.py`
- `app_verifier.py`, `browser_testing.py`, execution/sandbox adapters
- `engine_core` completion decision
- configuration and dependency-injection container
- evaluator and gate tests

**Design changes:**

- Make verification checks explicit in the composition root.
- Keep deterministic checks below the LLM score as a hard veto/floor.
- Return structured failures instead of only a scalar score.
- Distinguish `not_run`, `passed`, `failed`, and `blocked`.
- Fail closed when a required check cannot execute, unless a task policy explicitly permits a skip.

**Implementation tasks:**

1. Define `VerificationPolicy` by task type and artifact type.
2. Add test, lint, type, build, artifact, and security check adapters.
3. Wire the default policy through `container.py`.
4. Add artifact hashes and command receipts to `GateResult`.
5. Make evaluator reports include deterministic results and reasons.
6. Add a compatibility facade for callers expecting a float.

**Refactoring:** Move command-specific behavior into validators/adapters; keep the gate as orchestration only.

**Testing:** Broken fixture, lint-only failure, missing artifact, validator timeout, validator exception, passing fixture, and policy-specific skipped check.

**Acceptance criteria:**

- A failed required check prevents completion regardless of LLM score.
- Validator errors are visible and persisted.
- Passing artifacts still receive quality scoring.
- No application-layer import boundary is violated.

**Rollback:** Feature flag `ORCH_VERIFY_ACTS=false` may disable optional execution during staged rollout, but production unattended mode must refuse to run without an explicit bypass acknowledgment.

### WBS-2 — Independent fresh-model completion judgment

**Priority:** P0  
**Objective:** Ensure the generator cannot self-certify completion.

**Affected components:** evaluator, model cascade, budget enforcer, project runner, pipeline completion policy, telemetry.

**Design changes:**

- Add `IndependentJudgeService` using a model/provider distinct from the generator when available.
- Run it only after deterministic checks pass.
- Give the judge the task contract, artifact, receipts, failures, and acceptance criteria—not the generator’s self-score as authority.
- Record judge model, evaluator epoch, cost, latency, and verdict.

**Implementation tasks:**

1. Add a `CompletionDecision` domain type.
2. Enforce generator-model ≠ judge-model where the provider pool allows it.
3. Add disagreement telemetry between gate, evaluator, and judge.
4. Integrate the decision into `_should_exit_early` and project completion.
5. Use the cheapest suitable independent tier, bounded by budget.

**Acceptance criteria:** Generator score alone cannot end a task; deterministic failure always vetoes; judge disagreement causes revision or escalation.

**Rollback:** `ORCH_FRESH_STOP=false` restores threshold-only completion for interactive legacy operation; unattended mode should not permit this bypass by default.

### WBS-3 — Parallel worktree isolation

**Priority:** P0  
**Objective:** Prevent concurrent agents from corrupting one another’s edits.

**Affected components:** `delegation/batch_runner.py`, `vcs/worktree_manager.py`, `delegation/subagent.py`, task executor, cleanup and state persistence.

**Design changes:**

- Require one isolated worktree per concurrently mutating task.
- Preserve failed worktrees for inspection; clean unchanged successful worktrees.
- Bound worktree count and disk usage.
- Make non-Git/temp fallback explicit and disabled for mutation tasks by default.

**Implementation tasks:**

1. Make the worktree manager a required policy dependency for parallel mutation.
2. Pass the effective working directory into sub-agent execution.
3. Add lifecycle states: created, active, merged, cleaned, quarantined.
4. Store worktree path and branch/commit metadata in the task trace.
5. Add stale-worktree cleanup with safe path validation.

**Acceptance criteria:** Two tasks writing the same path cannot cross-contaminate; failed tasks remain inspectable; successful cleanup never removes user files outside the managed directory.

**Rollback:** Disable parallel mutation and run tasks sequentially. Do not silently fall back to a shared working directory.

### WBS-4 — Structured trace and alignment evidence

**Priority:** P1  
**Objective:** Detect failures involving intent, constraints, scope, action boundaries, and inaccurate progress reports—not only code correctness.

**Affected components:** pipeline context, task models, trajectory recording, state store, telemetry, HITL, prompt/context enrichment, dashboard/API.

**Design changes:**

Add a requirement ledger and action receipt to each task:

```text
requirements: explicit constraints, requested scope, acceptance tests
planned_actions: files, commands, tools, expected effects
actual_actions: executed commands, changed files, tool results
receipts: test/build/upload evidence
claims: agent-reported status
corrections: user pushback and subsequent resolution
failure_signature: symptom/cause/mechanism/outcome
```

Use the paper-derived taxonomy: wrong project diagnosis, misread intent, constraint violation, scope overreach, premature action, context loss, implementation/execution error, and false completion reporting.

**Implementation tasks:**

1. Extend trajectory capture with structured event records.
2. Add post-run consistency checks between claims and receipts.
3. Persist user corrections as negative evidence.
4. Add failure clustering by verifier-grounded signatures.
5. Expose trace IDs and evidence links in API/dashboard responses.

**Acceptance criteria:** Every completion claim has supporting receipts or is marked unverified; user corrections can be associated with the original trace; repeated failure signatures are queryable.

**Rollback:** Keep the new fields additive and ignore them in legacy readers; disable only the analyzer, never the raw trace capture.

### WBS-5 — Real Self-Harness validation

**Priority:** P1  
**Objective:** Replace heuristic skill improvement with behaviorally validated candidate harness changes.

**Affected components:** `skill_optimizer.py`, `skill_manager.py`, `skill_store.py`, task runner, evaluation fixtures, state schema.

**Design changes:**

- Treat each skill/harness edit as an immutable candidate version.
- Mine recurring failure clusters from traces.
- Generate diverse but minimal proposals tied to one failure mechanism.
- Evaluate baseline and candidate on the same held-in and held-out tasks.
- Accept only if held-in improves or remains stable and held-out does not regress.
- Store rejected candidates and reasons.

**Implementation tasks:**

1. Replace `_estimate_val_score()` with `HarnessValidationService`.
2. Introduce deterministic fixture selection and seed control.
3. Add candidate manifests: parent, changed surfaces, target signature, expected effect, risk.
4. Execute candidate variants in isolated worktrees/sandboxes.
5. Add minimum effect size and confidence thresholds for promotion.
6. Retain baseline until candidate validation and persistence succeed.

**Acceptance criteria:** An edit cannot be accepted solely because it is large or because its source trajectories scored highly; held-out regressions reject promotion; every accepted patch is reproducible from its manifest.

**Rollback:** Restore the last accepted `HarnessVersion`; retain rejected candidate artifacts for diagnosis.

### WBS-6 — First-class composable harness

**Priority:** P1  
**Objective:** Make prompts, tools, memory, policies, validators, and recovery mechanisms typed, composable, and replaceable.

**Affected components:** application services, prompt/template carriers, tool registry/guardrails, context management, validators, skill store, container.

**Design changes:**

Define a `HarnessSpec` composed of typed processors with lifecycle hooks:

```text
pre_context → prompt/context processors → tool policy
→ execution controller → verification processors
→ recovery policy → post_run evidence processor
```

Each processor must declare inputs, outputs, version, cost budget, and failure behavior. Harness versions are immutable values assembled by a registry.

**Implementation tasks:**

1. Inventory existing prompt, context, tool, memory, verification, and recovery modules.
2. Define minimal processor protocols in the domain/application boundary.
3. Create adapters around existing components rather than rewriting them.
4. Add composition validation and cycle detection.
5. Store serialized manifests, not executable arbitrary code, in persistent state.

**Acceptance criteria:** A task can select a harness variant by immutable ID; processors are independently testable; incompatible processor combinations fail before execution.

**Rollback:** Continue using the legacy pipeline adapter as a default `HarnessSpec` implementation.

### WBS-7 — Variant isolation and model-specific harness profiles

**Priority:** P2  
**Objective:** Prevent improvements for one task/model cluster from degrading another.

**Affected components:** model registry/router, harness registry, task classifier, skill store, evaluation suite, telemetry.

**Design changes:**

- Maintain variants by task family, model family, or failure cluster.
- Route only compatible tasks to a candidate variant.
- Track aggregate and per-variant non-regression.
- Prefer a stable baseline when classification confidence is low.

**Implementation tasks:**

1. Add `VariantKey` and routing metadata.
2. Add per-variant score/cost/latency metrics.
3. Implement conservative fallback to the default harness.
4. Run cross-variant compatibility and contamination tests.

**Acceptance criteria:** A variant’s improvement is measured on its own population and cannot silently lower unrelated populations; routing decisions are observable and reversible.

**Rollback:** Route all tasks to the last known-good default harness.

### WBS-8 — Controlled evaluator co-evolution

**Priority:** P2  
**Objective:** Let evaluators improve without making historical scores incomparable or creating reward-hacking loops.

**Affected components:** evaluator service, skill optimizer, trace/evaluation stores, anchor fixtures, model registry, telemetry, promotion policy.

**Design changes:**

- Freeze evaluator criteria within epochs.
- Maintain evaluator-independent anchor tasks with deterministic or human-grounded outcomes.
- Promote a challenger only after it statistically outperforms the incumbent on anchors.
- Mark evaluator-dependent records with dependency sets.
- Selectively invalidate or re-score dependent records after promotion.
- Add adversarial pools targeting known evaluator blind spots.

**Implementation tasks:**

1. Define evaluator version and epoch schemas.
2. Build anchor-suite execution and calibration reports.
3. Add challenger evaluation and promotion thresholds.
4. Add dependent-record invalidation/replay.
5. Freeze and unfreeze evaluator slots only at explicit boundaries.

**Acceptance criteria:** Scores within an epoch are comparable; evaluator promotion is anchor-based; historical evidence is not silently mixed across criteria; adversarial examples can be replayed.

**Rollback:** Revert to the last promoted evaluator epoch and preserve all newer records as quarantined evidence.

### WBS-9 — Discovery, scheduling, and human escalation

**Priority:** P2  
**Objective:** Turn autonomous execution into a bounded loop that discovers useful work and pauses at meaningful decision points.

**Affected components:** automations/scheduler, skill registry, project runner, HITL channels, state store, API/CLI, dashboard.

**Design changes:**

- Schedule named skills with versioned inputs rather than raw prompts.
- Discover work from CI failures, open tasks, recent commits, and incomplete checkpoints.
- Persist findings outside the context window.
- Require approval for destructive, external, irreversible, or high-scope actions.
- Fail closed when no approval channel exists for a required decision.

**Acceptance criteria:** A scheduled run is resumable, budget-bounded, traceable to a named skill, and routes unsupported findings to a human inbox.

**Rollback:** Disable scheduled triggers while preserving manually invoked runs.

### WBS-10 — Persistence and scale hardening

**Priority:** P1/P2  
**Objective:** Preserve evidence integrity under concurrent runs and create a measured path beyond SQLite.

**Affected components:** `StatePort`, `infrastructure/state.py`, skill store, telemetry store, checkpoints, ADR-010, deployment configuration.

**Design changes:**

- Serialize or transactionally protect project and evidence writes.
- Add project-level lock/idempotency keys where required.
- Use monotonic internal ordering and UTC display timestamps.
- Keep trace/evidence schemas forward-compatible.
- Add contract tests before introducing PostgreSQL.

**Implementation tasks:**

1. Measure SQLite write contention and checkpoint latency.
2. Add migrations for trace, evaluation, harness, and epoch records.
3. Add state-store contract tests.
4. Document thresholds for moving to PostgreSQL.
5. Keep event-sourcing as a later migration, not a prerequisite for R1–R3.

**Acceptance criteria:** Concurrent writes do not corrupt state; resume returns the latest valid checkpoint; schema migrations are transactional; the adapter remains replaceable through `StatePort`.

**Rollback:** Disable new evidence consumers while preserving additive records; restore the previous reader against the same database.

## 7. Engineering Practices and Design Rules

### SOLID and Clean Architecture

- Single responsibility: gates verify, judges decide, optimizers propose, validators execute, stores persist.
- Open/closed: add validators and processors through protocols/registries.
- Liskov substitution: all adapters pass port contract tests.
- Interface segregation: use focused ports rather than expanding `StatePort` with unrelated behavior.
- Dependency inversion: composition root wires infrastructure; application services consume protocols.

### DRY, KISS, and YAGNI

- Reuse `VerificationGate`, `WorktreeManager`, `SkillStore`, telemetry, and HITL instead of creating parallel frameworks.
- Implement the smallest harness processor set needed for current task types.
- Do not add model training or distributed event sourcing until measurements justify them.
- Consolidate duplicate evaluator and validation paths only after contract tests exist.

### Secure-by-design

- Execute generated code only through the existing sandbox/guardrail boundary.
- Fail closed when isolation is unavailable for untrusted execution.
- Treat harness manifests and evaluator prompts as data, not executable code.
- Validate all paths used by worktree cleanup.
- Redact secrets from traces and never persist raw API credentials.
- Apply least privilege to tools and require HITL for destructive operations.

### Defensive programming

- Use explicit state machines for candidate and promotion lifecycles.
- Make retries bounded and idempotent.
- Preserve failed artifacts and evidence rather than deleting them automatically.
- Treat validator timeout/error as a separate outcome from a clean pass/fail.
- Use schema validation when loading persisted state.

### Observability

Record structured events for:

- verification debt and failed checks;
- judge/evaluator disagreement;
- evaluator epoch changes;
- candidate promotion/rejection;
- held-out regressions;
- worktree creation, quarantine, cleanup, and leakage;
- token/cost blowout;
- repeated retries and idle loops;
- user corrections and escalations.

Every event should include `project_id`, `task_id`, `trace_id`, `harness_version`, `evaluator_version`, model, duration, cost, and outcome where applicable.

### Performance and scalability

- Cache deterministic results by artifact hash and command/environment fingerprint.
- Run cheap syntax/lint checks before expensive tests.
- Batch held-out evaluation where safe, but preserve per-task attribution.
- Bound candidate width, epoch size, trace retention, worktree count, and judge retries.
- Measure p50/p95 latency and cost per successful task, not only raw pass rate.
- Do not increase SQLite pooling without benchmark evidence; follow ADR-010.

## 8. Risk and Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Mandatory verification increases latency/cost | High | Medium | Cache by artifact hash; order cheap checks first; budget validator work; measure p95. |
| Fresh judge falsely rejects good work | Medium | High | Deterministic gate is primary floor; use judge calibration fixtures and disagreement alerts. |
| Candidate harness overfits held-in tasks | High | High | Required held-out suite, minimum effect size, immutable baseline, rollback. |
| Evaluator co-evolution creates reward hacking | High | Critical | Frozen epochs, anchor-only promotion, adversarial pool, selective erasure. |
| Trace volume increases SQLite contention | Medium | High | Compact event schema, batching, retention policy, write serialization, PostgreSQL threshold. |
| Parallel worktree cleanup deletes user data | Low | Critical | Resolve and validate absolute paths; restrict cleanup to managed root; quarantine failures. |
| New processor combinations create hidden incompatibilities | Medium | Medium | Typed hooks, composition validation, compatibility matrix, contract tests. |
| Legacy callers expect scalar evaluator results | Medium | Medium | Add structured report fields while preserving a float facade at boundaries. |
| Model/provider unavailable for independent judge | Medium | High | Explicit fallback policy: another provider, human gate, or fail closed. |
| Existing dirty worktree conflicts with refactoring | Medium | High | Keep changes localized, avoid unrelated files, use additive migrations, review diff before commit. |
| Skill optimizer modifies production guidance unexpectedly | Medium | High | Immutable versions, approval/promotion gate, staged rollout, last-known-good pointer. |
| Large `engine.py` becomes the integration bottleneck | High | Medium | Add services behind ports; keep engine wiring-only; enforce architecture tests. |

## 9. Testing and Quality Assurance Strategy

### Test pyramid

**Unit tests:** domain types, promotion rules, failure-signature clustering, gate aggregation, requirement/receipt consistency, evaluator epoch transitions, path validation, cost limits.

**Integration tests:** pipeline with deterministic gate, fresh judge, real temporary worktrees, SQLite migrations, trajectory persistence, candidate validation, HITL approval/rejection/timeout.

**Contract tests:** `StatePort`, `TraceStorePort`, `HarnessStorePort`, `EvaluationSuitePort`, `LLMClient`, `TelemetryPort`, and worktree adapters.

**Adversarial tests:** broken code with a high LLM score, false completion claims, scope expansion, corrupted tool output, endless exploration, missing artifacts, evaluator bias, candidate regression, stale evaluator records, and concurrent workspace writes.

**Property tests:** promotion never accepts a held-out regression; failed required checks never produce completion; cleanup stays within the managed root; evaluator-dependent evidence is invalidated on evaluator replacement.

### Required CI order

Preserve the repository-mandated order:

1. `black --check orchestrator/ tests/`
2. `ruff check orchestrator/ tests/`
3. `lint-imports`
4. strict mypy command for domain/application/container
5. non-slow, non-API, non-stress, non-e2e pytest with coverage
6. contract tests with empty API keys
7. Bandit high-severity scan

New tests must have appropriate markers, author headers, async I/O, and no new ignored modules. The coverage floor must not be reduced.

### Quality gates per milestone

| Milestone | Required gate |
|---|---|
| M0 | Broken-artifact fixture cannot complete; parallel edits are isolated. |
| M1 | Every task has a trace ID, requirement ledger, outcome, and evidence references. |
| M2 | Candidate promotion requires real held-in/held-out execution and no regression. |
| M3 | Variant routing is reproducible and fallback-safe. |
| M4 | Evaluator changes are epoch-scoped and anchor-calibrated. |
| M5 | Cost, latency, contention, and failure dashboards show baseline and post-change comparisons. |

## 10. Deployment and Rollback Plan

### Feature flags

Introduce flags with safe defaults and documented bypasses:

```text
ORCH_VERIFY_ACTS
ORCH_FRESH_STOP
ORCH_WORKTREE_ISOLATION
ORCH_TRACE_EVIDENCE
ORCH_HARNESS_EVOLUTION
ORCH_EVALUATOR_EPOCHS
ORCH_UNATTENDED_GUARD
```

Flags must be evaluated at the composition root, not scattered through domain code. Unattended mode should default to the safest behavior.

### Rollout stages

1. Shadow mode: collect verification, judge, and trace results without changing completion.
2. Canary projects: enable mandatory verification and isolation for selected task types.
3. Staged promotion: enable real harness evolution only for one task type and a fixed fixture suite.
4. Expand by model/task variant after observing non-regression metrics.
5. Enable evaluator epochs only after anchor calibration and replay tests pass.

### Rollback levels

- **Level 1:** disable candidate promotion; keep evidence collection.
- **Level 2:** route all tasks to last-known-good harness/evaluator.
- **Level 3:** disable autonomous scheduling and require HITL.
- **Level 4:** revert code release and retain additive database records for diagnosis.

Rollback must never delete traces, failed worktrees, evaluator records, or candidate manifests needed to explain the incident.

## 11. Documentation and Code Review Requirements

For each WBS item, require:

- an ADR when persistence, boundary, or promotion semantics change;
- updated port and adapter documentation;
- configuration and feature-flag documentation in `.env.example`;
- a failure-mode and rollback section;
- tests linked from the implementation PR;
- migration notes for state schema changes;
- dashboard/API documentation for new evidence fields.

Code review must verify:

1. no business logic was added to `engine.py`;
2. no behavior or I/O was added to `models.py`;
3. no import boundary was weakened;
4. no validator failure is converted into a success or silent skip;
5. every autonomous mutation has an isolation and rollback path;
6. every self-improvement path has held-out regression evidence;
7. every new cost/latency path has telemetry;
8. tests follow RED → GREEN → refactor.

## 12. Post-Implementation Validation Checklist

### Architecture

- [ ] New domain types are pure data.
- [ ] Application services depend on ports, not infrastructure.
- [ ] `engine.py` remains a mediator.
- [ ] All five import contracts pass.
- [ ] No duplicate evaluator, trace, or promotion framework was introduced.

### Safety and correctness

- [ ] Required deterministic checks run by default.
- [ ] Validator errors and timeouts fail closed for unattended runs.
- [ ] Completion requires independent judgment after deterministic success.
- [ ] Generated artifacts have hashes and execution receipts.
- [ ] Parallel mutation uses isolated worktrees.
- [ ] Failed worktrees are quarantined and inspectable.
- [ ] Destructive actions require the configured approval policy.

### Self-improvement

- [ ] Failure signatures are grounded in verifier evidence.
- [ ] Candidate edits are minimal, bounded, and versioned.
- [ ] Candidates run on held-in and held-out suites.
- [ ] Held-out regressions block promotion.
- [ ] Last-known-good harness/evaluator versions are restorable.
- [ ] Evaluator epochs freeze criteria and preserve dependency lineage.
- [ ] Anchor-suite calibration precedes evaluator promotion.

### Operations

- [ ] Cost, latency, retries, verification debt, regressions, and escalations are visible.
- [ ] Trace retention and SQLite contention are measured.
- [ ] Migrations are transactional and backward-compatible.
- [ ] Feature flags and rollback procedures are documented.
- [ ] Scheduled runs are bounded by per-run, daily, and retry caps.

### Verification commands

Run the repository’s required CI pipeline in order, then run targeted suites:

```text
pytest tests/unit/test_verification_gate.py tests/unit/test_evaluator_adversarial.py -q
pytest tests/unit/test_worktree_isolation.py -q
pytest tests/integration/test_skill_manager.py -q
pytest tests/contracts/ -v --no-cov
```

## 13. Definition of Done

The initiative is complete when:

1. An artifact that fails tests, lint, required checks, or receipt validation cannot be marked complete.
2. The generator cannot self-certify completion.
3. Concurrent agents cannot corrupt a shared working tree.
4. Self-improvement candidates are accepted only through real held-out regression testing.
5. Harness and evaluator versions are immutable, observable, and rollbackable.
6. Evaluator changes are controlled by epochs and anchor evidence.
7. User-intent, constraint, scope, and false-reporting failures are represented in traces.
8. All changes pass architecture, type, test, contract, and security gates.

