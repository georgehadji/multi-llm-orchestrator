# Outstanding Defects & Architecture Decisions — Remediation Plan

**Date:** 2026-09-06 · **Author:** Claude (session `01NRAPzp9JjkkegwCagDCzA1`) · **Base:** `master` @ `9cb580d` (post-merge of PR #28: T0–T22 + V4 waves P1–P2, 102 verified defects)

## 0. How to read this document

This is a **WORM** planning document (write-once, read-many) per this repo's own docs
convention — don't edit it in place as items get resolved; track disposition in commits/PRs
and, if this plan is superseded, write a new one and mark this one superseded in a one-line
header note. Every factual claim below is tagged:

- **VERIFIED** — I read the cited file/line myself, today, against `9cb580d`. A quote or exact
  line number is given so it can be re-checked in one command.
- **INFERENCE** — follows necessarily from VERIFIED facts but wasn't itself directly observed.
- **HYPOTHESIS** — plausible, not confirmed; treat as a lead, not a fact.
- **UNKNOWN** — insufficient evidence; stated as a gap, not guessed at.

Every item is also classified per `orchestrator-change-control` §1: **Trivial / Standard /
Architectural / Security** — this determines what review and gates apply, not my judgment of
how hard the code is. Architectural and Security classes require **mandatory human review**
before merge; several items below also trip a **hard trigger** (CI workflow edit, new
dependency, anything under `orchestrator/hitl/`) that forces human review regardless of class.

**Do not start any item's code before its decision (§5) is resolved where one is marked
needed.** Writing code against an unresolved product/security decision is exactly the kind
of unilateral call this hunt has consistently declined to make on its own — see the
"Cleared/Escalated" sections of `docs/hunts/INVENTORY.md` for precedent.

---

## 1. Problem / Motivation

T0–T22 (the V7 defect-hunt protocol) and V4 waves P1–P2 fixed 102 verified defects, merged to
`master` in PR #28 (`9cb580d`, 2026-09-06). Of the items that surfaced during that work, **16
were deliberately not fixed**: 15 were formally escalated as `[REQUIRES HUMAN REVIEW]` in
`docs/hunts/INVENTORY.md` and the PR body, because each needs a product or architecture
decision rather than a mechanical patch; a 16th (`operations/diagnostics.py`) was identified
during the P2 wave but fell out of the consolidation and was never fixed, escalated, or
documented — a process gap, not a considered deferral.

This plan re-verifies every one of those 16 items against the current source (not against the
PR's prose, which in a few cases turns out to have been imprecise — see §4 and §5 for the
corrections), classifies each, and either clears it, gives it a mechanical fix, or lays out the
real decision with options and a recommendation.

## 2. Scope

**In scope:** the 16 items enumerated in §4–§5.

**Out of scope:**
- V4 wave plan waves **P3–P11** (~120k more LOC, priority ≥4) and V7 protocol tiers **T23–T24**.
  These are open-ended *discovery* waves, not remediation of a known list, and are explicitly
  gated on the user asking for them by name — referenced in §7 only, not planned here.
- Any **new** discovery. This document fixes the list that already exists; it does not go
  looking for more.

## 3. Architecture ground rules this plan must respect

Condensed from `orchestrator-architecture-contract` and `orchestrator-change-control`
(loaded in full before drafting this plan, per `CLAUDE.md`'s "read before any architectural
decision" instruction). Cite these skills directly rather than re-deriving; only the
project-specific consequences for this plan's items are restated here.

**The Four Unbreakable Rules** (`CLAUDE.md`): `engine.py` is a Mediator only (no new logic
methods); `models.py` is pure data; TDD without exceptions (RED before GREEN); no new
root-level `orchestrator/*.py` modules. Every fix below is placed against these rules
explicitly in its own sub-section.

**Change classification** (`orchestrator-change-control` §1):

| Class | Gate | Human review |
|---|---|---|
| Trivial | pre-commit + CI | No |
| Standard | + TDD + full suite green | Recommended; mandatory if it touches evaluator scoring, budget math, or routing |
| Architectural | + `docs/CODEBASE_MINDMAP.md` read first, plan before code, contracts stay KEPT | **Mandatory** |
| Security | + bandit HIGH clean, no new `#nosec` | **Mandatory + security-focused review** |

**Hard triggers for mandatory human review regardless of class:** any CI workflow edit, any
new third-party dependency, any new `ignore_imports`/allowlist/`xfail`, anything under
`orchestrator/hitl/`. Two items below (CI Python matrix, the `dashboard` extra's dependency
pins) trip these triggers even though the code change itself is small.

**Never weaken a gate to pass it** (`orchestrator-change-control` §7, Unwritten Rule 1). This
matters concretely for two items here: the `dashboard` extra's pin conflict must be solved by
correcting the pins, not by silencing the resolver or dropping the extra from CI; the CI
Python-version gap must be solved by adding a matrix, not by lowering `requires-python`.

**Layer placement rule** (`orchestrator-architecture-contract` §2, applied per item in §5):
data → `models.py` only if pure; new interface → `domain/ports.py`; use-case logic →
`application/`; concrete adapter → `infrastructure/`; wiring → `engine_core/container.py`;
CLI → `commands/` + `entrypoints/cli_dispatch.py`. Never a new root module, never a new method
on `Orchestrator`, never behavior in `models.py`.

---

## 4. Cleared during this plan's own verification — no action needed

One escalated item turned out, on re-reading the current source, to already be a
non-issue — recorded here so nobody re-escalates it a second time.

| ID | Original framing (PR #28) | What re-verification found |
|---|---|---|
| **RunContext-dup** | "A second, independent `RunContext` class exists at `application/unattended_guard.py:36`, distinct from the one `engine.py` actually uses — surfaced while fixing the P1 budget-race defect, not investigated further." | **VERIFIED cleared.** `application/run_context.py:27`'s `RunContext` (the one `engine.py` uses) carries this docstring, verbatim: *"Note: This is distinct from `unattended_guard.RunContext`, which holds pre-flight check metadata (daily cap, max-retries, etc.) and is used by `UnattendedGuard.validate()`. The two serve different phases of the execution lifecycle and are intentionally separate."* This is a documented, intentional design choice (two small, purpose-built dataclasses that happen to share a common name), not an accidental duplication. No decision, no fix. |

---

## 5. Item-by-item disposition

Ordered by the project's own established threat-model priority (`docs/DEFECT_HUNT_PLAN.md`'s
T0–T8 ordering: money → credentials/trust-boundary → concurrency → resilience → execution),
then by whether a decision blocks the fix.

### Group A — Security / trust-boundary (Security class, mandatory + security review)

#### A1. Slack signature verification fails open when unconfigured

- **State (VERIFIED):** `integrations/slack_integration.py::SlashCommandHandler.verify_signature()`
  returns `True` when `signing_secret` is unset, logging only a `WARNING`. The HMAC check
  itself is correct when a secret *is* present (`hmac.compare_digest`, correct base string,
  300s replay window — fixed and tested in P2). Module is currently unwired into any live
  server (dead code today), but its own trailing docstring hands a deployer a verbatim FastAPI
  integration example.
- **Classification:** Security. Mandatory + security-focused review (this touches an auth
  gate, the same standing this repo gives `orchestrator/hitl/`).
- **Decision needed:** yes — what should the unconfigured state do?

  | Option | Behavior | Trade-off |
  |---|---|---|
  | 1. Fail closed always | Reject every request when `signing_secret` is unset | Safest; breaks any existing deployment that (incorrectly) relies on the open default, with no escape hatch |
  | 2. Keep fail-open, louder warning | Current behavior, escalate log to `ERROR` and reject in a `strict` mode flag | Minimal diff; still ships an insecure default, just noisier |
  | **3. Fail closed by default, explicit opt-out (recommended)** | Reject unless `ORCHESTRATOR_SLACK_ALLOW_UNSIGNED=true` is set; log a `CRITICAL` line naming the exact env var whenever that escape hatch is used | Matches this repo's own fail-closed doctrine (HITL gate, `orchestrator-change-control` §7) — "no explicit approval means no" — while leaving a documented, auditable path for local dev/testing |
- **Fix shape (once decided):** `infrastructure/` adapter change only (the class already lives
  in `integrations/`, itself effectively an adapter over the Slack API) — no new port needed,
  no container wiring change since the module is unwired. Standard TDD: RED test asserting
  rejection with no secret configured and no opt-out env var.

#### A2. Insecure default binds — dashboard, and the same pattern repeated in 5 more servers

- **State (VERIFIED):**
  - `crosscutting/config.py:152`: `OrchestratorSettings.dashboard_host: str = "127.0.0.1"` —
    the declared, documented intent.
  - `dashboard_core/core.py:347,372`: `DashboardApp.run()` and `run_dashboard()` both default
    their own `host` **parameter** to `"0.0.0.0"`, independently of the setting above.
  - **Confirmed disconnect:** `grep -rn "dashboard_host" orchestrator/` finds exactly one
    declaration site (`config.py:152`) and zero consumption sites — `OrchestratorSettings
    .dashboard_host` is never read anywhere. `operations/issue_tracking.py`'s
    `dashboard_host` parameter is an unrelated local name on a different class.
  - **New finding, not in the original escalation:** the identical shape — a server
    hardcoding `host="0.0.0.0"` as its own default rather than reading any setting — recurs in
    at least 4 more files, all defaulting to **port 8765** as well: `ide_backend/server.py:124-125`,
    `ide_backend/ide_orchestrator_server.py:3046`, `command_center_server.py:166`,
    `ide_backend/launch.py:24-25`, `ide_backend/standalone_server.py:324`. Only
    `commands/server.py:170`'s `WebSocketServer.start()` defaults to the safe `127.0.0.1`.
    `commands/server.py` and `command_center_server.py` define near-identically-shaped
    `WebSocketServer` classes (same `self._lock = asyncio.Lock()` line immediately before
    `start()`) with **opposite** default binds — this is the same "duplicate root/subpackage
    pair, divergent" shape T17 found 65 instances of; it was not re-run against these two
    files specifically. **INFERENCE**, not verified: these two classes are a genuine
    divergence-candidate pair by T17's own methodology and should be diffed the same way.
- **Classification:** Security (default network exposure).
- **Decision needed:** narrow — the *setting itself* already states the intended default
  (loopback); this isn't a 3-way product fork so much as "wire the existing declared intent
  through," but changing a shipped default bind is a behavior change for anyone currently
  relying on `0.0.0.0` (e.g., running the dashboard inside a container and reaching it from
  outside). Two real options:

  | Option | Behavior | Trade-off |
  |---|---|---|
  | **1. Wire settings through, default loopback (recommended)** | `run_dashboard()`/`DashboardApp.run()` read `settings.dashboard_host` instead of hardcoding `"0.0.0.0"`; same pattern for the 5 IDE/command-center servers, each gaining a settings-or-env-driven host with a loopback default | Matches the already-documented intent; anyone who needs external access sets the env var explicitly, which is the secure-by-default posture this repo already applies elsewhere (WS bind fix, `611c1403`) |
  | 2. Leave runtime default as `0.0.0.0`, only fix the doc | Change `OrchestratorSettings.dashboard_host`'s default to `"0.0.0.0"` to match reality | Zero behavior change, but codifies an insecure default as "intended" — inconsistent with this repo's own security incident history |
- **Fix shape:** `infrastructure`/entrypoint-level parameter change (`dashboard_core/core.py`,
  the 5 server files) reading from `crosscutting.config.settings` — no port/ports architecture
  change, no container wiring. Standard class per file; classify Security given the bind
  default. Do the `commands/server.py` vs `command_center_server.py` diff (per T17 methodology)
  as a first step — it may turn out to be a genuine divergent duplicate warranting its own
  disposition, not just a bind-default fix.

### Group B — Decisions genuinely needed (Architectural or product judgment calls)

#### B1. Policy enforcement is not just "uncalled" — the plumbing is broken in two places

- **State (VERIFIED, corrects the original escalation):** the original framing
  ("`PolicyEngine.enforce()` has zero callers; `spec.policy_set` written to `RunContext` and
  never read") undersold it. Re-verified:
  - `policy_engine.py:285`'s `PolicyEngine.enforce()` — confirmed **zero callers** anywhere in
    `orchestrator/` or `tests/` (only self-reference is its own exception docstring).
  - `engine.py:951`: `self._run_ctx.active_policies = spec.policy_set` — set on every
    `run_job()` call. **Confirmed never read**: no other file reads
    `_run_ctx.active_policies`.
  - **New finding:** `engine.py:1163-1165`'s `_get_active_policies(self, task_id)` reads
    `self._active_policies.policies_for(task_id)` — a *different* attribute
    (`self._active_policies`, no `_run_ctx.` prefix). `self._active_policies` is **assigned
    nowhere** — not in `engine.py`, not in `container.py`'s `ServiceContainer.build()`
    (`grep -n active_policies engine_core/container.py` → no matches). And
    `_get_active_policies()` itself has **zero callers**. This is a second, independent piece
    of dead machinery, and its shape (a reader pointing at an attribute name that no longer
    exists) is the signature of a stranded pre-`RunContext`-refactor leftover, not a
    freshly-introduced bug. **INFERENCE**, not directly confirmed by git blame in this pass.
- **Classification:** Architectural (touches `engine.py`, and real enforcement needs a new
  call site in either the pipeline or the LLM-call path).
- **Decision needed:** yes — CLAUDE.md's own "Known Limitations" already flags this
  ("Enforcement mode selection (HARD/SOFT/MONITOR) not fully integrated"), so this is a known
  open architecture question, not a fresh one.

  | Option | Where enforcement hooks in | Trade-off |
  |---|---|---|
  | 1. Enforce in `UnifiedClient.call_model()` | Every provider call checks policy before dispatch | Closest to "enforced on every API call" (the docstring's actual claim); touches the adapter layer used by every task type, so a bug here is maximally blast-radius |
  | 2. Enforce once per task in the pipeline | A single check in `engine_core/pipeline.py`/`stages.py` before a task's first call | Matches `PolicyEnginePort`'s existing shape better; smaller blast radius; misses mid-task fallback calls |
  | **3. Enforce via a dedicated pipeline stage, `HARD`/`SOFT`/`MONITOR` modes wired to `PolicyEngine.enforce()`'s existing return contract (recommended)** | New stage registered in `engine_core/container.py`, reading the *already-flowing* `spec.policy_set` off `RunContext` | Uses machinery that already exists end-to-end except the last connection; `SOFT`/`MONITOR` modes give a safe rollout path (log-only before hard-blocking) instead of an all-or-nothing flip — directly closes the CLAUDE.md-documented gap |
- **Fix shape:** delete or repoint the orphaned `_get_active_policies`/`self._active_policies`
  pair first (mechanical, Standard, no decision needed — it's provably dead and pointing at
  nothing); the real enforcement wiring is Architectural and follows the decision above,
  landing as a new `application/` service + `container.py` wiring, never a new `engine.py`
  method (Rule 1).

#### B2. `--agent-profile`: not three implementations, two — one throwaway, one already well-built

- **State (VERIFIED, corrects the original escalation):** the original text claimed "a third,
  differently-named profile vocabulary exists in `operations/autonomy.py`." Re-verified: that
  file doesn't exist under that name, and `agent_safety.py`'s `get_agent_profile()` /
  `AgentSafetyProfile` — the only other "profile" hit in the repo — is an **unrelated**
  concept (per-agent trust/safety profiles in the multi-agent safety monitor), a name
  coincidence, not a third competing implementation. There are genuinely two:
  1. `entrypoints/cli_dispatch.py` — an inline `profile_map` dict
     (`{"standard": {...}, "max": {...}, "creative": {...}, "conservative": {...}, "research":
     {...}}`), duplicated verbatim at **4 call sites** (lines ~332, ~402, ~631, ~694), used for
     nothing — each site already logs `"--agent-profile %r is not currently wired to any
     effect on this command"` (this warning was added in P1; the underlying dead code was not
     otherwise touched).
  2. `operations/autonomy_config.py:119`'s `AutonomyConfig.from_agent_profile(profile_name)` —
     maps the **same 5 profile names** to a full `AutonomyConfig` (max_iterations,
     repair_attempts, max_runtime_minutes, verification_mode, critique_passes, model tiers) and
     already has `.apply_to_task(task)`, which sets `max_iterations`/`acceptance_threshold` on
     a task object if those attributes exist. **Zero callers anywhere** — a complete,
     ready-to-wire implementation that nothing invokes.
- **Classification:** Standard (no new layer, no new dependency) — but touches a
  publicly-documented CLI flag's actual behavior, so treat as needing at least code review
  per the "touches routing" bar in the Standard-class table.
- **Decision needed:** narrow, mostly a confirmation rather than a fork — is
  `AutonomyConfig.from_agent_profile(...).apply_to_task(...)` an acceptable replacement for
  the dead inline dicts? **Recommendation: yes** — it is strictly more complete than the
  4 duplicated dicts and was clearly built for exactly this purpose. The one open question is
  whether a `Task`/`TaskSpec`-shaped object is actually in scope at each of the 4
  `cli_dispatch.py` call sites to call `.apply_to_task()` on — **UNKNOWN**, not traced in this
  pass; confirm this first (cheap, 4 read-only lookups) before writing the fix.
- **Fix shape:** delete the 4 duplicated `profile_map` dicts + warnings; replace each with
  `AutonomyConfig.from_agent_profile(args.agent_profile).apply_to_task(<task-or-spec-in-scope>)`.
  No new port/adapter — `AutonomyConfig` is already a plain `application`-layer-appropriate
  dataclass. RED test: assert a project run with `--agent-profile max` actually produces a
  task with `max_iterations` matching `AutonomyConfig.for_level(AutonomyLevel.MAX)`.

#### B3. `operations/autonomy_config.py`'s Multi-Mode Selector, and `integrations/mcp_server.py`'s dead fork

- **State (VERIFIED presence, UNKNOWN full scope):** both were flagged in T9–T16's escalation
  register (`docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` §2, R3: "`integrations/mcp_server.py`'s
  feature-convergence direction is undecided"). Not re-traced line-by-line in this pass —
  this entry inherits T16's own disposition rather than adding new evidence.
- **Classification:** Architectural (feature consolidation, cross-file).
- **Decision needed:** yes.

  | Option | Direction | Trade-off |
  |---|---|---|
  | 1. Delete the unwired fork | Remove `integrations/mcp_server.py`, keep only the live root MCP server | Simplest; loses whatever features the fork has that the root server lacks (unenumerated — **UNKNOWN** what's lost) |
  | 2. Merge forward | Port the fork's extra features into the live root server, then delete the fork | Keeps the good parts; requires actually diffing the two first (unscoped effort — **UNKNOWN** size) |
  | **3. Diff-and-decide (recommended)** | First produce a concrete feature diff between the two (same T17 divergence-pair methodology as A2), *then* pick 1 or 2 per file with real information | Doesn't guess at scope; this is the same "measure before deciding" discipline this hunt has used throughout — a diff is cheap, a wrong consolidation direction is not |
- **Fix shape:** the diff step is Standard/mechanical and can run without a decision; the
  consolidation itself is Architectural and follows from the diff's findings.

#### B4. `TieredModelRouter.next_tier()` / `.escalate_tier()` — dead on an otherwise-live class

- **State (VERIFIED in P2, re-confirmed no change since):** `_MODEL_TIERS` assigns tier 0 to
  every real text model and tier 1 only to an image model (`NANO_BANANA_2`), so escalation
  would be a no-op for virtually all real task types even if wired. `M2-1`'s duplicate-key
  no-op (`Model.ZHIPU_GLM_5_2` mapped twice to the same value) was already cleared in P2 as
  harmless.
- **Classification:** Standard once tier data exists; the data itself is a product judgment
  call, not a code question.
- **Decision needed:** yes — this needs someone to actually rank models by relative
  capability/cost, which this hunt has consistently declined to invent. No sub-options to
  offer here; the only path forward is a maintainer (or a dedicated, separate research task
  against real benchmark data — see `orchestrator-research-frontier`'s Pareto-frontier
  program) supplying the tier assignments.
- **Fix shape:** once tier data exists, wiring `next_tier()`/`escalate_tier()` into the
  fallback-escalation path is a small `application`/`engine_core` change — not itself the
  hard part.

#### B5. `TransferLearningEngine.find_transferable_patterns()` — half-fixed, needs an upstream guarantee

- **State (VERIFIED in P2):** the silent bypass is now visible (P2 added a warning + corrected
  comment). Completing it needs `PatternMiner` to actually populate
  `TransferPattern.source_projects` (currently always `[]`). `meta/orchestrator.py
  ::ExecutionRecord` does carry a `project_id` field, so the data exists somewhere in the
  pipeline — whether it's reliably populated end-to-end was explicitly out of P2's 16-file
  scope and remains **UNKNOWN**.
- **Classification:** Architectural (cross-module data-flow guarantee, not a local fix).
- **Decision needed:** yes, but it's a scoping decision, not a design fork — someone needs to
  trace `ExecutionRecord.project_id`'s reliability across the meta-optimization pipeline
  (outside this plan's scope) before a fix can even be written. Recommend this be scoped as
  its own small investigation task, not bundled with this plan's other items.

#### B6. `dashboard` extra — `pip install -e ".[dashboard]"` is `ResolutionImpossible`

- **State (VERIFIED, root cause narrowed from the original framing):** `pyproject.toml:74-79`
  pins `dashboard = ["fastapi>=0.100.0,<1.0", "uvicorn[standard]>=0.23.0,<1.0",
  "websockets>=11.0,<13.0", "httpx>=0.24.0,<1.0"]`. `pyproject.toml:31` (core, unconditional
  dependency) pins `"google-genai>=1.0,<2.0"`. The original text said "httpx collides too
  (core `<0.28.0` vs google-genai `>=0.28.1`)" — the exact httpx range on the core side was
  **not re-verified** in this pass; treat that half of the claim as **INFERENCE** carried
  forward from the earlier hunt, not freshly confirmed. The `websockets<13.0` vs
  `google-genai`-needs-`>=13.0` conflict is corroborated by the pin values actually present.
- **Classification:** Standard fix (loosen two version ranges), but **hard-triggers mandatory
  human review** regardless — any dependency-version change is change-control's Unwritten
  Rule 3 territory, and CI's current workaround (installing bare `fastapi` instead of the
  `dashboard` extra) is itself evidence the extra has been silently broken for a while.
- **Decision needed:** minimal — confirm the new ranges (`websockets>=13.0`, and re-verify the
  httpx range) don't break anything else that pins the old range. Recommend: bump
  `websockets` and `httpx` floors in the `dashboard` extra to whatever `google-genai>=1.0,<2.0`
  actually requires (get this from `pip install google-genai==1.0` in a scratch venv, not by
  guessing), then flip CI's Test job back to installing the real `dashboard` extra instead of
  bare `fastapi`, so this doesn't silently rot again.
- **Fix shape:** `pyproject.toml` edit + CI workflow edit. No architecture change.

#### B7. `ara_pipelines.py` — a paid LLM call whose result is computed and then never weighted

- **State (VERIFIED, corrects the original framing):** `_phase_jury_verify_and_meta_eval`
  (line 1355, called from line 1257) makes a real `client.call(model=verifier,
  max_tokens=1500, ...)` and writes the result to `state.metadata["meta_evaluation"]`
  (line 1397) — this call is **reachable** whenever this phase runs. The PR's "the dead
  statement is fixed" refers to `_phase_jury_weighted_ranking` (line 1399 on): the pointless
  fetch-and-discard line was already removed and replaced with an explanatory comment
  (lines 1423-1428) — so today the code no longer *pretends* to use `meta_evaluation`, it
  correctly does nothing with it. What remains unfixed is the LLM call itself still executing
  and costing money for output nothing consumes.
- **Classification:** Standard (no architecture change either way) once the decision below is
  made.
- **Decision needed:** yes.

  | Option | Behavior | Trade-off |
  |---|---|---|
  | 1. Stop making the call | Skip `_phase_jury_verify_and_meta_eval`'s LLM call entirely | Saves the spend immediately; discards whatever signal `meta_evaluation` was meant to add — unmeasured, so the loss is unknown |
  | 2. Wire a weighting formula | Use `meta_evaluation` to adjust `candidate_scores` in `_phase_jury_weighted_ranking` | Delivers the feature the comment originally implied; "inventing a formula nothing specifies" (the original objection) still applies — this needs a real design, not a guess |
  | **3. Make the call conditional on a feature flag, default off (recommended)** | Gate the call behind a new `FeatureFlags` entry (e.g. `jury_meta_eval_enabled: bool = False`), default `False` so spend stops now; a maintainer can flip it on once a weighting design exists | Stops the bleeding immediately without deciding the weighting question under time pressure; consistent with this repo's existing pattern of flag-gating experimental phases (`vs_*` flags in `crosscutting/config.py`) |
- **Fix shape:** `crosscutting/config.py` gets one new `FeatureFlags` field (pure data, no
  behavior — consistent with Rule 2); `ara_pipelines.py` gains one `if flags.x:` guard around
  the call. No new dependency, no layer violation.

#### B8. `OrchestratorSettings` — dead-field count needs a proper recount, not either existing number

- **State (VERIFIED field count, UNVERIFIED dead-count):** `crosscutting/config.py:137-174`'s
  `OrchestratorSettings` has exactly **21** fields (counted directly:
  `max_concurrency, max_parallel_tasks, default_budget_usd, default_timeout_seconds,
  context_truncation_limit, rate_limit_per_minute, cache_ttl_hours,
  semantic_cache_threshold, dashboard_port, dashboard_host, mcp_port, mcp_host,
  mcp_http_mode, log_level, log_format, audit_log_path, design_variance,
  motion_intensity, visual_density, cache_home, compression_model`) — confirming the "21"
  half of the original "12 of 21" claim. A narrow re-check (`grep -rn
  "settings\.(field)\b"` for all 21 names) found only **2** real consumption sites
  (`log_level`, `log_format`, both in `project_mgmt/assembler.py:514,516`) — the apparent
  third hit, `crosscutting/config.py:12`, is inside that file's own module **docstring**
  (example code, not live). This regex only catches the `settings.<field>` access pattern
  through the module-level singleton; it cannot see a locally-constructed
  `OrchestratorSettings()` instance under a different variable name, or an `ORCH_<FIELD>`
  environment variable read directly via `os.environ`. **The true dead-field count is
  UNKNOWN** — it could be as low as the previously-recorded 12 or as high as 19; neither
  number should be trusted until every one of the 21 fields is checked individually across
  all three access patterns.
- **Classification:** Standard investigation, then a product decision.
- **Decision needed:** after the recount, for each confirmed-dead field: wire it to a real
  consumer, or delete it. Not resolvable until the recount exists — do not guess at which 12
  (or 19) in the meantime.
- **Fix shape:** recount first (mechanical, cheap); disposition follows per-field.

#### B9. IDE dev-server ports are hardcoded, not session-scoped

- **State (VERIFIED):** `ide_backend/ide_orchestrator_server.py:484-495`'s
  `SessionManager.start_server(self, session_id, output_dir, port: int = 8000, server_type)`
  takes `port` as a parameter (not literally hardcoded at the signature level), but checks
  `self.is_port_available(port)` and returns `False` (a silent-looking failure, not a raised
  error) if the port is taken — meaning two concurrent sessions requesting the same
  `server_type` (and therefore the same default port) will have the second one simply fail to
  start, with the caller responsible for noticing. The false-success half (a dead process
  still reporting as running) was already fixed in T-series/P1; real per-session port
  allocation was not.
- **Classification:** Standard (adding an allocation scheme), no new layer.
- **Decision needed:** yes — allocation scheme.

  | Option | Scheme | Trade-off |
  |---|---|---|
  | 1. Sequential scan | Try `base_port`, `base_port+1`, … until `is_port_available` succeeds | Simple, deterministic, but ports drift session-to-session — harder to document/debug |
  | 2. Random from a configurable range | Pick randomly within e.g. `[9000, 9999]` | No collision-prone sequential pattern, but two sessions could theoretically collide (low probability, non-zero) |
  | **3. Deterministic hash of `session_id` into a fixed range, with sequential-scan fallback on collision (recommended)** | `port = base + (hash(session_id) % range_size)`, then linear-probe if taken | Same session reliably gets the same port across restarts (useful for debugging/bookmarking), while still handling the rare collision case safely |
- **Fix shape:** `application`-layer change to `SessionManager` (or wherever it canonically
  lives per the layer map) plus a new `OrchestratorSettings` field for the port range —
  small, no port/adapter/container change needed.

### Group C — Mechanical fixes, no decision needed (Standard class, can start immediately)

#### C1. `commands/codebase.py`'s `modify_codebase()` — confirmed live crash disguised as a normal failure

- **State (VERIFIED, upgraded from the original "referenced but never implemented, left
  unbuilt" framing):** `Orchestrator` (`engine.py`) has **no `modify_codebase` method at
  all** (`grep -n "def modify_codebase" orchestrator/engine.py` → no matches).
  `commands/codebase.py:31`'s `_run_modify()` calls `await orch.modify_codebase(...)` inside a
  broad `except Exception as exc:` (line 37) that catches the resulting `AttributeError` and
  returns `f"Modification failed: {exc}\n{traceback.format_exc()}"` — indistinguishable, to a
  user, from a legitimate runtime failure. This command (`register()` wires an argparse
  subparser for it, `execute()` handles a real `args.repo`/`args.objective`/`args.dry_run`)
  **always fails this way on every invocation** — this is exactly the T18 "silent failure"
  shape (a broad except disguising a structural absence as a normal error), just not caught by
  T18's own sweep because that sweep targeted `except: pass`/`except: continue` sites, not
  ones that return a formatted message.
- **Classification:** Standard — no decision needed to make the failure honest; a decision
  only exists if someone wants to *implement* `modify_codebase` for real (a genuine feature,
  out of scope here — `implementation_plan_codebase_optimizations.md:98` suggests this was
  once planned).
- **Fix:** narrow the `except` to let an `AttributeError` on a genuinely-missing method surface
  as a clear "codebase modification is not implemented yet" message (matching the existing
  `nash_backup` pattern in `commands/nash.py`), rather than a generic traceback dump. RED test:
  assert the CLI's output names "not implemented," not a raw traceback.

#### C2. `operations/diagnostics.py` — still assumes a direct-provider architecture

- **State (VERIFIED just now, this is the item I flagged as a process gap in my prior
  response):** `_check_api_keys()` (line 185) constructs `UnifiedClient` directly against
  `Model.GPT_4O_MINI`/`DEEPSEEK_V4_FLASH`/`GEMINI_FLASH` and calls `.generate()` on each;
  `_check_network()` (line 232) opens raw sockets to `api.openai.com`, `api.deepseek.com`,
  `generativelanguage.googleapis.com`. This is the same root cause already fixed elsewhere in
  T8 (C6) — a diagnostic that assumes direct per-provider calls in a codebase that has since
  moved toward OpenRouter-mediated routing for these paths (see `llm-orchestration-reference`
  for the routing doctrine) — but this specific pair of methods was never itself touched.
- **Classification:** Standard.
- **Fix:** update both checks to reflect the current routing architecture (OpenRouter
  connectivity/key check instead of, or in addition to, direct provider probes) — same shape
  as whatever T8/C6's fix did elsewhere; consult that commit before writing this one so the
  two diagnostics stay consistent.

#### C3. `nash_backup` — confirm all guarded call sites degrade the same way

- **State (VERIFIED module absence, PARTIALLY VERIFIED guard consistency):**
  `orchestrator.nash_backup` does not exist. `commands/nash.py:46-49` catches its absence and
  prints "Nash backup/restore is not implemented yet." `cli_nash.py` has **4** more import
  sites (`_create_backup`, `_list_backups`, `_restore_backup`, `_show_backup_value`, lines
  172/198/226/256), each inside a `try:` block — but this pass did not read far enough into
  each `except` clause to confirm all 4 degrade as cleanly as `commands/nash.py`'s version.
  **UNKNOWN**, not assumed.
- **Classification:** Trivial/Standard verification task.
- **Fix:** read all 4 `except` blocks; if any lets an unguarded `ImportError` propagate,
  align it with `commands/nash.py`'s pattern. No design work either way.

#### C4. CI never exercises the declared Python 3.10/3.11 floor

- **State (VERIFIED):** `pyproject.toml:10`: `requires-python = ">=3.10"`. `.github/workflows
  /ci.yml`: every one of 7 `python-version` references is pinned to `"3.12"` — no matrix, no
  3.10/3.11 job anywhere.
- **Classification:** Standard change, but **hard-triggers mandatory human review** (CI
  workflow edit, per change-control §1).
- **Fix:** add a `matrix: python-version: ["3.10", "3.11", "3.12"]` to at least the Test job
  (not necessarily every job — Lint/Security Scan/Architecture Boundaries don't need to run
  3x). Decide with the maintainer whether this runs on every push (cost: 3x CI minutes) or
  only on a schedule/pre-release — that's a cost/coverage trade-off worth a one-line ask
  rather than a unilateral pick.

---

## 6. Recommended execution sequencing

| Phase | Contents | Gate to start | Gate to ship |
|---|---|---|---|
| **R0** | Read this document; resolve the decisions in §5 Group B (at minimum A1/A2's security defaults before anything else, since those are the highest-severity class) | This document exists | Repo owner sign-off recorded (a comment on the tracking PR/issue is enough — doesn't need to be formal) |
| **R1** | Group C mechanical fixes (C1–C4) + B1's dead `_get_active_policies` deletion + B8's recount | None — no decision blocks these | Standard gate suite (TDD, black/ruff/bandit/lint-imports, mypy isolated-diff, full pytest) per item, one commit per item, same discipline as the T/P-tier work |
| **R2** | Group A + remaining Group B items, each as its decision resolves | That item's §5 decision resolved | Same gate suite; Architectural/Security items additionally need the mandatory human review pass before merge |
| **R3** *(not opened by this plan)* | V4 P3–P11, V7 T23–T24 | Explicit user request naming the wave/tier | — |

R1 and R2 are not strictly sequential — an R2 item whose decision resolves early can ship
before R1 finishes. The only hard ordering is: no code for a Group A/B item before its
decision is recorded.

## 7. Gated future discovery — reference only

`docs/hunts/PRECISION_AUDIT_V4_WAVE_PLAN.md` (waves P3–P11) and the V7 protocol's own T23–T24
remain queued and unstarted. This plan does not scope their content — by design, a discovery
wave's findings aren't knowable before it runs. They are listed here only so this document is
a complete picture of "what remains," per the standing instruction not to start them without
an explicit, separate ask.

## 8. Test & gate plan (applies to every item above)

- TDD without exception (Rule 3): RED test written first, observed failing for the *predicted*
  reason, then GREEN.
- Full gate suite per commit: `black --check`, `ruff check`, `bandit -r orchestrator
  --severity-level high`, `lint-imports` (5/5 contracts KEPT), `check_new_root_files.py
  --baseline origin/master`, mypy isolated-diff (clean `.mypy_cache`, `--no-incremental` both
  sides — the methodology this session's P1/P2 work settled on after finding incremental-mode
  noise), full `pytest` run with zero new regressions.
- One item per commit, matching this repo's own T/P-tier convention — makes each change
  independently revertable and reviewable.
- Any item touching `models.py` or `orchestrator/config/*.json`: run the drift tests
  (`pytest tests/ -k "drift or vfm_routing or config"`) per Unwritten Rule 2. None of the 16
  items here touch those files directly, so this is a checklist item, not an expected hit.

## 9. Risks

- **Scope creep.** 16 items is a lot of surface; the one-item-per-commit discipline in §6/§8
  is the mitigation, same as every prior tier in this hunt.
- **Gate-weakening pressure.** B6 (dependency pins) and C4 (CI matrix) are exactly the kind of
  item where "just skip the check" is tempting under time pressure. Explicitly banned per
  Unwritten Rule 1 — fix the pins/add the matrix, never loosen a floor or add an
  `ignore_imports` to route around a real conflict.
- **B8's number is genuinely unknown.** Don't let "12 of 21" (or my own rough "as many as 19")
  anchor the eventual fix — recount first.
- **A2's `WebSocketServer` pair is an inferred, not confirmed, duplicate.** Diff before
  assuming both files should converge to one behavior.

## 10. Uncertainty acknowledgment

**Most likely to change on closer inspection:** B8 (dead-field count) and B6's httpx-range
half — both explicitly flagged UNKNOWN/INFERENCE above rather than asserted.

**Most likely to be straightforward once decided:** C1–C4 and B1's dead-code deletion half —
all VERIFIED, all mechanical, none blocked on a design question.

**Highest-value next step if only one item can be picked:** A1/A2 — both are Security-class,
both are cheap to fix once decided, and both are the same "insecure-by-default" shape this
repo has already paid down once (the `611c1403` WS-bind fix cited in
`orchestrator-change-control`) — leaving it unresolved in five more files is the most
repo-consistent argument for prioritizing it.

**Cannot be determined from static reading alone:** whether `AutonomyConfig.apply_to_task()`
(B2) has a `Task`/`TaskSpec` object actually in scope at all 4 `cli_dispatch.py` call sites;
the real httpx conflict range (B6); the true dead-field count (B8); whether
`ExecutionRecord.project_id` (B5) is reliably populated end-to-end.
