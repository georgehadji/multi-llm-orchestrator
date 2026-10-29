# Adaptive Capability Router (ACR) — Implementation Plan

**Project:** Multi-LLM Orchestrator
**Component:** Adaptive Capability Router (model selection)
**Author:** Engineering
**Date:** 2026-07-09
**Status:** Proposed
**Related:** `docs/CODEBASE_MINDMAP.md`, `orchestrator-architecture-contract` skill, memory `vfm-routing-2026`, `phase-policy-reasoning-temperature`

---

## 1. Executive Summary

### 1.1 Problem
Model selection today is **static tier mapping**: `PHASE_TO_TIER` → `TIER_ROUTING` lookup tables ([`model_routing.py`](orchestrator/model_routing.py)) plus a scalar greedy score `quality_score * trust_factor / (cost + ε)` in [`GreedyBackend`](orchestrator/operations/optimization.py). It cannot answer the real question: *"for **this** task, with **these** constraints, **right now**, which model has the highest expected utility?"* It does not learn, does not adapt to server-side model drift, and hand-encodes quality assumptions that go stale on every model release.

### 1.2 Key finding — do not build greenfield
~70% of the ACR already exists in the repo and is **wired to nothing or scalarised away**:

| ACR concept | Existing asset | State |
|---|---|---|
| L1 registry (static + dynamic stats) | [`ModelProfile`](orchestrator/policy.py:76) — cost/capability static + `quality_score`/`trust_factor`/`success_rate`/`avg_latency_ms`/`avg_cost_usd` mutable, updated by `TelemetryCollector` | **live** |
| L4 utility + selection seam | [`OptimizationBackend`](orchestrator/operations/optimization.py) ABC + `planner.set_backend()` Strategy | **live, pluggable** |
| L5 telemetry sink | `TelemetryCollector` (updates `ModelProfile`) + [`capability_logger.py`](orchestrator/state_mgmt/capability_logger.py) `ROUTING_DECISION` | **live** |
| L6 online learning | [`LearningAggregator`](orchestrator/learning/learning_aggregator.py) — per-`(task_type, model)` score/cost/confidence, `get_routing_recommendations()` | **built, dead code** (grep: referenced only inside its own module) |
| Availability gate | `resilience.py` circuit breaker | **live, not read by router** |

### 1.3 Strategy
Implement the ACR as a **new `OptimizationBackend` subclass** (`AdaptiveCapabilityBackend`), swapped in via the existing `set_backend()` seam behind a feature flag. This honours the Four Unbreakable Rules (no new logic in `engine.py`, no new root modules, TDD, pure `models.py`). Then **activate `LearningAggregator`** as the backend's per-task-type prior, corrected for the design defects below.

### 1.4 Scope of corrections (fixes) folded in
1. Replace numerically unstable multiplicative utility (`× quality ÷ cost ÷ latency`) with **gate × log-linear** form.
2. Replace cosine/scalar capability match with **weighted shortfall** over a capability vector.
3. Hard constraints (context, modality, JSON, availability) become a **binary gate**, not smooth terms.
4. **Recency-weighted / discounted posterior** to track non-stationary model drift.
5. External reward signal only (critic/eval/benchmark) — never close the loop on self-reported utility.
6. Bandit context = **request features + requirement vector**, not the coarse `TaskType` enum.
7. **Per-agent `explore_allowed` flag** (ε=0 for determinism-critical agents) + explicit exploration budget cap.
8. **Per-phase reward attribution** in the decompose→generate→critique→revise→evaluate pipeline.
9. Read `resilience.py` circuit-breaker state as availability gate — do not rebuild.
10. Benchmarks (L7) = **decaying Bayesian priors**, lazily computed — not a periodic cron over 300+ models.

### 1.5 Expected outcome
A closed-loop router that measures → compares → selects → evaluates → updates, degrading gracefully to the current static path on any failure. Delivered incrementally; each phase is independently shippable and reversible via one flag.

---

## 2. Current Architecture Assessment

### 2.1 Relevant layers (hexagonal)
- **Application core:** `engine.py` (Mediator, wiring only), `planner.py::ConstraintPlanner` (selection orchestration).
- **Domain:** `models.py` (`Model`, `TaskType`, `ROUTING_TABLE`, `FALLBACK_CHAIN` — pure), `policy.py` (`ModelProfile`, `Policy`, `RunContext`).
- **Strategy adapters:** `operations/optimization.py` (`OptimizationBackend`, `GreedyBackend`, `ParetoBackend`).
- **Driven adapters:** `api_clients.py::UnifiedClient`, `infrastructure/telemetry.py::TelemetryCollector`, `infrastructure/model_capabilities.py` (OpenRouter `/models` fetch + 24h cache), `operations/resilience.py` (circuit breaker), `learning/learning_aggregator.py`.

### 2.2 Data flow (selection today)
```
task_type + policies + budget
   → ConstraintPlanner.select_model()
       → _apply_filters(task_type, policies, budget)     # candidate set
       → self._backend.select(candidates, self._profiles, task_type, cost_fn)
           → GreedyBackend: argmax( quality_score * trust_factor / (cost+ε) )
   → Model | None
```
`ModelProfile` dynamic fields are mutated post-call by `TelemetryCollector`; `capability_logger` records the decision. **No feedback reaches the next selection** — the greedy score reads a single scalar `quality_score` (global EMA), not per-task-type history.

### 2.3 Module boundaries / integration points
- Single selection entrypoint: `planner.select_model` / `select_reviewer`. **All routing flows through the backend seam.** This is the only injection surface required.
- `ModelProfile` is the canonical registry record. Extending it (new immutable capability vector field) is additive and backward compatible.
- `LearningAggregator` has an async, file-backed store (`./learning_data/*.jsonl`) — independent of the SQLite state store; no schema migration needed.

### 2.4 Scalability / maintainability
- Selection is pure/synchronous computation over an in-memory `dict[Model, ModelProfile]` — O(candidates). Adding vector math and a prior lookup keeps it O(candidates·dims); negligible for <500 models.
- `LearningAggregator` loads all history into memory on init. Acceptable now; flagged as a scaling risk (R-7) if history grows unbounded — mitigated by existing `retention_days` cleanup.

### 2.5 Security posture
- No new external inputs at the selection layer (model ids are internal enums). Reward signals originate from internal critic/evaluator — **must be validated/clamped** to `[0,1]` before updating posteriors (defensive: a poisoned score must not skew routing). No secrets involved.

### 2.6 Technical debt intersecting this work
- `GreedyBackend` multiplicative score is the debt this plan retires.
- `LearningAggregator` equal-weight averaging (`sum/len`) — no recency weighting → cannot track drift. Must fix before wiring.
- `quality_score` on `ModelProfile` is a single global scalar, not task-type-segmented — the reason the greedy path "forgets" domain performance.
- Telemetry honesty: confirm `TelemetryCollector` actually populates `quality_score` from evaluator output on the live path (per architecture-contract skill, some telemetry is a documented TODO stub). **Verification gate in Phase 0.**

---

## 3. Detailed Implementation Plan (Phased Roadmap)

Guiding sequence: **deterministic core first, learning second, exploration third.** Never ship exploration before the utility metric is proven and the reward signal is verified real.

| Phase | Goal | Work Items | Ships behind flag | Exit criterion |
|---|---|---|---|---|
| **0** | Foundations & guardrails | Feature flag, telemetry verification, capability-vector schema, test scaffold | `ORCH_ACR_BACKEND=off` | Reward signal proven real; flag toggles backend with zero behavioural change when off |
| **1** | Deterministic utility core | WI-1, WI-2, WI-3 | `off` (opt-in) | `AdaptiveCapabilityBackend` reproduces or beats `GreedyBackend` on replay; gate correctness proven |
| **2** | Activate learning loop | WI-4, WI-8 | `off` | `LearningAggregator` wired as prior; recency weighting live; per-phase reward attributed |
| **3** | Adaptive exploration | WI-5 | `off` | Bandit selects under explore budget; `explore_allowed=false` agents provably pinned |
| **4** | Benchmark priors / cold-start | WI-6 | `off` | New model gets benchmark prior; prior weight decays as telemetry accrues |
| **5** | Observability & rollout | WI-7, flag promotion | `on` (staged) | Metrics/traces on every decision; default-on after canary |

**Dependencies:** 1→0 (needs flag + schema). 2→1 (needs utility core to blend the prior into). 3→2 (bandit needs a working posterior). 4→2 (priors feed the same posterior). 5→3 (observe the full loop). WI-8 (reward attribution) is a prerequisite for trustworthy learning and lands in Phase 2 alongside WI-4.

**Milestones:**
- **M1** (end P1): deterministic ACR selectable, provably safe, off by default.
- **M2** (end P2): learning loop closed, drift-aware, still off by default.
- **M3** (end P3): exploration governed by budget + determinism flags.
- **M4** (end P5): ACR default backend after canary; static path retained as fallback.

---

## 4. Task Breakdown Structure (WBS)

Each work item lists: Objective · Affected components · Design changes · Implementation tasks · Refactoring · Testing · Acceptance · Rollback.

### WI-0 — Feature flag, telemetry verification, schema (Phase 0)
- **Objective:** Establish a zero-risk toggle and prove the reward signal is real before building on it.
- **Affected:** `config.py` FeatureFlags, `engine_core/container.py` (backend wiring), `infrastructure/telemetry.py`.
- **Design changes:** Add `ORCH_ACR_BACKEND` env flag (`off|shadow|on`) to pydantic `FeatureFlags`. `off` = `GreedyBackend` (unchanged). `shadow` = run ACR selection, **log** its choice, but return greedy choice (no behaviour change, collects comparison data). `on` = ACR is authoritative.
- **Implementation tasks:**
  1. Add flag to `FeatureFlags` with wired-status doc (per `orchestrator-config-and-flags` convention).
  2. Container reads flag, calls `planner.set_backend(...)` accordingly.
  3. Instrument `TelemetryCollector` update path; assert `quality_score` is populated from evaluator output on a live (mocked-LLM) run. If it is a stub → fix as a blocking sub-task.
  4. Define `CapabilityVector` immutable dataclass (frozen) in a new module `orchestrator/domain/capability_vector.py`.
- **Refactoring:** None to `engine.py` (wiring only, container-owned).
- **Testing:** Unit — flag resolves to correct backend. Integration — `shadow` mode returns identical selections to `off`. Verification test — evaluator score reaches `ModelProfile.quality_score`.
- **Acceptance:** With flag `off`, byte-identical routing to today. Telemetry verification test green.
- **Rollback:** Delete flag / default `off`. No persisted state touched.

### WI-1 — Capability vector registry (L1/L2, objective-only)
- **Objective:** Give each model a multi-dimensional capability vector sourced from **verifiable facts**, not hand-typed quality guesses (fix #1).
- **Affected:** `domain/capability_vector.py` (new), `policy.py::ModelProfile`, `infrastructure/model_capabilities.py`, `config/*.json`.
- **Design changes:**
  - `CapabilityVector` dims split by provenance: **static/objective** (max_context, vision, tool_use, json_mode, price) hydrated from OpenRouter `/models` payload (already fetched); **derived** (reasoning, legal, coding, citation…) left `None` at registry time and filled by L5/L7, never by hand.
  - Add optional `capability: CapabilityVector | None` field to `ModelProfile` (additive, backward compatible).
- **Implementation tasks:** Extend `build_capability_map()` to also extract `context_length`, `architecture.modality`, pricing, `supported_parameters`; map into `CapabilityVector`. Hydrate profiles at container build.
- **Refactoring:** Consolidate the scattered per-model facts (`costs.json`, `routing.json`) reads through the registry hydration path (DRY) — no key drift (memory `config-enum-id-drift`).
- **Testing:** Unit — vector built from a fixture `/models` payload; unknown model → permissive defaults. Property — objective dims never `None` for catalogued models.
- **Acceptance:** Every routable model resolves a `CapabilityVector` with objective dims populated; zero hand-typed quality constants remain in source.
- **Rollback:** `capability=None` path falls back to scalar `quality_score`; field is optional.

### WI-2 — Gate × log-linear utility (fix #2, #3)
- **Objective:** Replace the unstable multiplicative greedy score with a normalized, bounded, additive utility.
- **Affected:** `operations/optimization.py` (new `AdaptiveCapabilityBackend(OptimizationBackend)`).
- **Design changes:**
  ```
  U(m,t) = gate(m,t)                                  # {0,1}
         × ( α·S_cap + β·S_quality + γ·S_reliability
             − δ·S_cost − ε·S_latency + ζ·S_context )
  ```
  All `S∈[0,1]`, min-max normalized across the candidate set. Coefficients per-agent/task via `phase_policy`-style config (reuse `orchestrator/domain/phase_policy.py` pattern). No division by cost/latency.
- **Implementation tasks:** Implement backend `select()` conforming to the ABC signature `(candidates, profiles, task_type, cost_fn)`. Pure function; no I/O. Coefficient table in config.
- **Refactoring:** Extract normalization helpers as pure functions (testable). Leave `GreedyBackend` intact as fallback.
- **Testing:** Unit — monotonicity (↑quality ⇒ ↑U, ↑cost ⇒ ↓U); zero-cost model doesn't produce inf; single candidate returns it; empty → `None`. Golden test vs `GreedyBackend` on a fixture set.
- **Acceptance:** No `inf`/`nan` reachable; utility bounded; deterministic given fixed profiles.
- **Rollback:** Flag `off` → `GreedyBackend`.

### WI-3 — Shortfall match + hard-constraint/availability gate (fix #2, #7, #9)
- **Objective:** Compute `S_cap` as weighted requirement shortfall, and make constraints (context, modality, JSON, circuit-breaker availability) a binary gate.
- **Affected:** `AdaptiveCapabilityBackend`, `operations/resilience.py` (read-only), agent requirement definitions.
- **Design changes:**
  - `S_cap = 1 − Σ wᵢ·max(0, reqᵢ − capᵢ) / Σ wᵢ·reqᵢ` — only deficits penalize; exceeding a requirement is free.
  - `gate = 0` if any hard constraint fails **or** circuit breaker for the model's provider is OPEN (read `resilience.py` state; do not rebuild an `availability` field).
  - Per-agent **requirement vector** defined declaratively (e.g. Constitutional: Legal≥10, Citation≥10, Determinism≥10).
- **Implementation tasks:** Requirement vectors in config keyed by agent/task_type. Availability adapter method on the resilience component (`is_available(provider) -> bool`) if not present.
- **Refactoring:** Fold existing `_apply_filters` hard filters and the new gate into one place to avoid double-filtering (SoC).
- **Testing:** Unit — model below a floor is gated out; model exceeding floor not penalized; OPEN breaker → gated. Integration — a task whose only capable model is circuit-broken returns `None` (fail-closed, surfaced to fallback handler).
- **Acceptance:** No model violating a hard requirement is ever selected; circuit-broken providers excluded.
- **Rollback:** Flag `off`.

### WI-4 — Wire LearningAggregator as recency-weighted per-task prior (fix #4; activates L6 dead code)
- **Objective:** Blend historical per-`(task_type, model)` performance into `S_quality`, weighted toward recent observations to track drift.
- **Affected:** `learning/learning_aggregator.py`, `AdaptiveCapabilityBackend`, container.
- **Design changes:**
  - Replace equal-weight `sum(scores)/len` with **exponential time decay** (half-life configurable, e.g. 14 days) in `_update_aggregated_stats`.
  - Backend queries `get_routing_recommendations(task_type)` / per-model stats; blends `expected_score` (with `confidence`) into `S_quality`: `S_quality = confidence·hist + (1−confidence)·profile.quality_score`. Low sample count ⇒ leans on global profile (graceful cold-start).
  - Record loop: after evaluation, call `record_task_result(...)` on the live path.
- **Implementation tasks:** Add decay to aggregator; add async-safe read from a sync backend (snapshot cache refreshed out-of-band, since `select()` is sync/pure). Wire `record_task_result` at pipeline completion. Validate/clamp score to `[0,1]` on ingest (defensive, fix #5 hygiene).
- **Refactoring:** Introduce a thin read-model snapshot (`dict[(task_type,model)→stats]`) so the pure sync backend never awaits.
- **Testing:** Unit — decay: a recent low score outweighs old high scores. Unit — confidence blend: 0 samples ⇒ pure profile; many ⇒ pure history. Integration — record→aggregate→influence-next-selection round trip. Regression — clamp rejects out-of-range reward.
- **Acceptance:** Two consecutive runs where model A degrades on `legal` cause the router to shift away from A within N runs; dead-code paths now covered by tests.
- **Rollback:** Backend ignores the snapshot (blend weight 0) via sub-flag; aggregator writes are idempotent/append-only and harmless if unread.

### WI-5 — Contextual bandit, explore budget, per-agent explore flag (fix #4, #6, #7, #10)
- **Objective:** Balance exploit/explore with a contextual bandit whose context is request features + requirement vector, bounded by an exploration budget and disabled where determinism is required.
- **Affected:** New `orchestrator/learning/bandit.py`, `AdaptiveCapabilityBackend`, agent config.
- **Design changes:**
  - Thompson sampling (or LinUCB) over candidate models; **context** = requirement vector + request features (est. tokens, language, needs_citation, domain). Not the coarse `TaskType` enum alone.
  - `explore_allowed: bool` per agent — ε/exploration forced 0 for Determinism-critical agents (Constitutional pinned).
  - **Exploration budget cap** (USD or % of run budget); explore preferentially on low-cost/low-stakes tasks. Integrates with the per-run `Budget` dataclass.
  - Reuse `nash/auto_tuning.py::select_arm` if its bandit is suitable (grep hit — evaluate before writing new).
- **Implementation tasks:** Bandit posterior seeded from WI-4 stats (and WI-6 priors). Gate exploration on budget + flag. Persist posteriors alongside learning data.
- **Refactoring:** If `nash/auto_tuning` bandit is reusable, adapt rather than duplicate (DRY); else new module with shared interface.
- **Testing:** Unit — `explore_allowed=false` ⇒ argmax always (no sampling variance); exploration halts when budget exhausted. Statistical — regret decreases over simulated rounds. Determinism — fixed seed reproducible.
- **Acceptance:** Determinism agents never explore; total exploration spend ≤ configured cap; measured regret trends down on a synthetic benchmark.
- **Rollback:** Sub-flag disables bandit → pure exploit (WI-2/4 utility argmax).

### WI-6 — Benchmark priors, lazy + decaying (fix #5)
- **Objective:** Solve cold-start with benchmark-derived priors that fade as production telemetry accrues; avoid periodic re-benchmarking of unused models.
- **Affected:** New `orchestrator/learning/benchmark_priors.py`, benchmark suite runner.
- **Design changes:** Benchmark score = Bayesian **prior** for `(task_type, model)`. Prior weight `∝ 1/(1+n_observations)` so it decays as real data arrives. Benchmarks computed **lazily** only for models entering a candidate set and never re-run on a schedule.
- **Implementation tasks:** On first encounter of an un-benchmarked model in a candidate set, enqueue a bounded benchmark job; store prior. Feed prior into WI-4 blend / WI-5 posterior seed.
- **Refactoring:** Unify prior + posterior into one `expected_quality(task_type, model)` read model (SoC — backend consumes one interface).
- **Testing:** Unit — prior dominates at n=0, negligible at large n. Integration — new model routed sanely on first sight (no divide-by-nothing, no pure-random pick for determinism agents).
- **Acceptance:** A newly added model is routable immediately with benchmark-justified priors; no scheduled benchmark cron exists.
- **Rollback:** Absent prior ⇒ fall back to WI-4 confidence blend (already cold-start-safe).

### WI-7 — Observability & decision audit (cross-cutting)
- **Objective:** Every routing decision emits structured logs, metrics, and a trace span with the full utility breakdown.
- **Affected:** `capability_logger.py` (extend `ROUTING_DECISION` details), metrics exporter, tracing.
- **Design changes:** Log per-decision: candidate set, per-term `S_*`, gate reasons, chosen model, explore-vs-exploit, prior/posterior contribution. Metrics: selection latency, explore rate, per-model win share, drift alerts.
- **Implementation tasks:** Structured `details` payload; counter/histogram metrics; trace span around `select()`.
- **Testing:** Unit — decision record schema complete. Integration — a decision produces a parseable audit line + metric increment.
- **Acceptance:** Any routing choice is fully reconstructable from logs; drift dashboards populate.
- **Rollback:** Logging is additive; disable via log level.

### WI-8 — Per-phase reward attribution (fix #8)
- **Objective:** Attribute pipeline outcome to the model that performed each phase, not one blanket score to the generator.
- **Affected:** Engine pipeline stages (`engine_core/stages/*`), `LearningAggregator` ingest.
- **Design changes:** Each of decompose/generate/critique/revise/evaluate records its own `(phase, model, reward)`. Evaluate stage's score credits the generator+reviser; critique quality credited to the critic.
- **Implementation tasks:** Thread phase identity + model into the `record_task_result` call per stage.
- **Testing:** Unit — a run attributes distinct rewards to distinct phase models. Integration — poor critic doesn't penalize a good generator.
- **Acceptance:** Learning store shows per-phase credit; no cross-phase contamination.
- **Rollback:** Fall back to single-score attribution (flagged).

---

## 5. Risk & Mitigation Matrix

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | Reward signal is a stub / not populated → learning trains on noise | Med | Critical | **Phase 0 verification gate**; block Phase 2 until proven. Clamp+validate all rewards. |
| R-2 | ACR regresses routing quality vs static path | Med | High | `shadow` mode collects comparison data before `on`; golden replay tests; staged canary; instant flag rollback. |
| R-3 | Numerical instability (inf/nan) in utility | Low | High | Gate × normalized log-linear (no division); property tests forbid inf/nan. |
| R-4 | Non-stationary drift makes stale posterior overconfident | Med | High | Recency decay (WI-4); drift alerts (WI-7). |
| R-5 | Exploration burns budget on worse models | Med | Med | Explore budget cap + low-stakes bias + `explore_allowed=false` for critical agents (WI-5). |
| R-6 | Determinism-critical agent gets explored/varied model | Low | Critical | Hard `explore_allowed=false` ⇒ argmax only; unit test asserts zero sampling variance. |
| R-7 | `LearningAggregator` in-memory history unbounded | Low | Med | Existing `retention_days` cleanup + decay makes old data weightless; snapshot read-model. |
| R-8 | Circular import / new-root-module contract violation | Med | Med | New code under `domain/`, `learning/`, `operations/` — no `orchestrator/*.py` at depth 1; run `lint-imports` in CI (architecture-contract skill). |
| R-9 | Async aggregator vs sync pure `select()` deadlock/await-in-sync | Med | High | Out-of-band snapshot refresh; `select()` reads a plain dict, never awaits. |
| R-10 | Config key drift (model id ≠ enum value) silently drops entries | Med | Med | Registry hydration validates ids against `Model` enum; `check_config_drift.py` in CI (memory `config-enum-id-drift`). |
| R-11 | Reward poisoning skews routing | Low | High | Clamp `[0,1]`, source only from internal critic/evaluator, per-phase attribution limits blast radius. |

---

## 6. Testing & Quality Assurance Strategy

### 6.1 Approach
TDD throughout (RED→GREEN→commit) per project rule 3. Every WI lands with failing tests first. Pure functions (utility, shortfall, decay, gate) are the bulk of coverage; the live wiring is thin.

### 6.2 Test pyramid
- **Unit (majority):** utility monotonicity/bounds, shortfall math, gate logic, decay weighting, confidence blend, bandit determinism under `explore_allowed=false`, reward clamp. Markers: `@pytest.mark.unit`.
- **Integration:** record→aggregate→influence round trip; shadow-mode parity with greedy; circuit-broken exclusion; per-phase attribution. Markers: `@pytest.mark.integration`.
- **Property-based:** no inf/nan reachable; utility ∈ bounds for random profiles; objective capability dims never `None` for catalogued models.
- **Regression/golden:** replay a fixed task set through `GreedyBackend` vs `AdaptiveCapabilityBackend`; assert ACR ≥ greedy on a scored rubric.
- **Statistical (simulation):** synthetic non-stationary environment — verify regret ↓ and drift tracking (marker `slow`).

### 6.3 Coverage & gates
- Meet the repo coverage ratchet (`--cov-fail-under`); target ≥80% on new modules per `common/testing.md`.
- CI gates: `ruff`, `black --check`, `mypy` (new modules must not extend the mypy-ignore baseline), `bandit`, `lint-imports` (5 contracts), `check_new_root_files.py`, `check_config_drift.py`.
- No new module added to the mypy ignore list (architecture-contract).

### 6.4 Verification discipline
Never mark a WI complete without a passing test proving it (project rule + `orchestrator-validation-and-qa`). Reward-signal reality (R-1) is proven by test before any learning code merges.

---

## 7. Deployment & Rollback Plan

### 7.1 Rollout (progressive, flag-gated)
1. **Merge behind `ORCH_ACR_BACKEND=off`** — no behaviour change. Default remains `GreedyBackend`.
2. **Shadow (`=shadow`)** in staging/canary — ACR computes and logs its choice; greedy remains authoritative. Collect divergence + would-be-quality data.
3. **Canary (`=on`)** for a subset (single team/job via `BudgetHierarchy` scoping) — monitor R-2/R-4/R-5 dashboards.
4. **Promote default `=on`** after canary meets acceptance; keep static path as permanent fallback.

### 7.2 CI/CD
- Feature branch → all gates green → PR with full commit-history summary and test plan (`git-workflow` rules).
- No hook/signing bypass. Pre-commit + CI parity.
- Migrations: none (learning store is append-only jsonl; `ModelProfile` field additive).

### 7.3 Rollback
- **Instant:** set `ORCH_ACR_BACKEND=off` — reverts to `GreedyBackend` with zero data migration. No restart-state coupling.
- **Sub-feature rollback:** independent sub-flags disable learning blend (WI-4), exploration (WI-5), or priors (WI-6) without disabling the deterministic utility core.
- **Data:** learning-store writes are append-only and ignored when unread — safe to leave in place or delete `./learning_data/`.
- **Blast radius:** confined to model *selection*; execution/validation/budget layers untouched.

### 7.4 Observability in production
- Metrics: selection latency, explore rate, per-model win share, gate-rejection counts, drift alerts (WI-7).
- Structured decision audit reconstructs any routing choice.
- Alert on: exploration spend approaching cap, sudden win-share shift (possible drift/poisoning), gate rejecting all candidates (fail-closed event).

---

## 8. Post-Implementation Validation Checklist

**Correctness**
- [ ] Flag `off` produces byte-identical routing to pre-change baseline (parity test green).
- [ ] Utility never yields inf/nan (property test).
- [ ] No model violating a hard requirement or with an OPEN circuit breaker is ever selected.
- [ ] `explore_allowed=false` agents show zero selection variance across seeds.

**Learning loop**
- [ ] Reward signal proven sourced from real evaluator output (not stub) — R-1 gate.
- [ ] Recent degradation of a model on a task shifts routing within N runs (drift test).
- [ ] Cold-start: newly added model routable immediately via benchmark prior; prior decays with data.
- [ ] Per-phase reward attribution verified — critic failure does not penalize generator.

**Governance / cost**
- [ ] Total exploration spend ≤ configured cap in a load test.
- [ ] Determinism-critical agents pinned in an end-to-end run.

**Engineering gates**
- [ ] `lint-imports` (5 contracts) + `check_new_root_files.py` pass; no new `orchestrator/*.py` root module.
- [ ] `ruff`, `black`, `mypy` (no new ignores), `bandit` clean.
- [ ] Coverage ≥ ratchet; new modules ≥80%.
- [ ] `check_config_drift.py` clean (registry ids ↔ `Model` enum).

**Observability**
- [ ] Every routing decision emits a complete audit record + metrics.
- [ ] Drift/exploration/gate dashboards populate in canary.

**Docs**
- [ ] `docs/CODEBASE_MINDMAP.md` routing section updated to describe ACR backend.
- [ ] `CLAUDE.md` / `orchestrator-config-and-flags` note `ORCH_ACR_BACKEND` with wired status.
- [ ] ADR recorded: "Adopt Adaptive Capability Router as pluggable OptimizationBackend."

**Rollback proof**
- [ ] Verified `=off` fully reverts with no residual state dependency.

---

## Appendix A — Engineering Practices Applied

- **SOLID:** ACR is a new `OptimizationBackend` (Open/Closed — extend, don't modify greedy); single responsibility per WI module; depends on the `OptimizationBackend` abstraction (DIP), not concretions.
- **Clean/Hexagonal Architecture:** selection logic stays in the Strategy adapter; `engine.py` wires only; `models.py` remains pure data; new domain types under `domain/`.
- **Separation of Concerns:** registry (facts) vs telemetry (observation) vs learning (estimation) vs utility (decision) vs exploration (policy) — each a distinct module with one read interface.
- **DRY/KISS/YAGNI:** reuse `ModelProfile`, `LearningAggregator`, `capability_logger`, `resilience`, and possibly `nash/auto_tuning` bandit rather than rebuild; deterministic core shipped before speculative RL; no periodic benchmark cron until proven needed.
- **Secure-by-Design / Defensive:** reward clamp+validation, fail-closed gate (no capable model ⇒ `None` → fallback handler), circuit-breaker-aware availability, per-phase attribution to bound poisoning.
- **Observability:** structured decision audit, metrics, tracing (WI-7) from Phase 5, with shadow-mode telemetry from Phase 0.
- **CI/CD & Code Review:** flag-gated progressive rollout, all existing gates enforced, PR review mandatory, ADR + mindmap docs.
- **Performance/Scalability:** O(candidates·dims) pure selection; async learning kept off the sync selection path via snapshot; decay + retention bound memory.
