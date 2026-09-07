# Escalation Register — T24

**Base:** `master` @ `f390414`. **Plan:** `docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` §4, T24.

Per T24's specification this is **not a bug hunt**. For each open item it restates the
finding, states *the decision actually required*, gives a recommendation with its trade-offs,
and marks it **decided** / **deferred-with-reason** / **withdrawn**.

**Authority note.** Nothing here is marked "decided" on my own authority. Items are marked
**RECOMMENDED** where the evidence supports one option clearly enough that a maintainer can
ratify it in one line, and **OPEN** where the answer depends on product intent that no
artefact in this repo states. Two items are marked **WITHDRAWN** because this session's
evidence dissolved the question.

Status counts: **18 open** (10 carried from the remediation plan, 8 new from P3–P11/T23),
**2 withdrawn**, **1 closed by R1**.

---

## Part 1 — Carried from `docs/plans/2026-09-06-outstanding-remediation-plan.md`

### A1. Slack signature verification fails open when unconfigured — OPEN (Security)

**Finding.** `SlashCommandHandler.verify_signature()` returns success when no signing secret
is configured, so an unconfigured deployment accepts unsigned requests.

**Decision required.** Should a verification gate fail open or closed when its credential is
absent?

**Recommendation: fail closed, with an explicit opt-out.** Refuse to start (or refuse every
request) when the secret is unset, and require an explicit
`SLACK_ALLOW_UNSIGNED=1`-style flag for local development. Trade-off: breaks any existing
deployment that silently relied on the open default — which is exactly the population most at
risk, so breaking it loudly is the point. **Blast radius is currently nil** — the whole
`slack_integration` module is unwired (P2-SLACK1), so this can be fixed before it has users.

### A2. Insecure default binds (`0.0.0.0`) in 6 servers — OPEN (Security)

**Decision required.** Is binding all interfaces the intended default for a developer tool?

**Recommendation: default to `127.0.0.1`, make `0.0.0.0` opt-in** via an explicit `--host`
flag or env var, and log loudly when the wide bind is chosen. Trade-off: container
deployments need the flag set, which is a one-line change in whatever launches them, versus
the current state where every `pip install` exposes a dev server to the local network.

### B1. Policy enforcement plumbing broken in two places — OPEN (Architectural)

R1 removed the dead `_get_active_policies`. What remains is the real decision: whether
`PolicySet` enforcement (HARD/SOFT/MONITOR) is a shipping capability or an abandoned design.
`CLAUDE.md` advertises "Policy-driven enforcement"; the code does not deliver it.

**Decision required.** Ship it, or stop advertising it?

**Recommendation: decide by the cost of the smallest honest option.** Wiring `ConstraintPlanner`
to actually reject on HARD violations is bounded work; the alternative is deleting the
capability claim from `CLAUDE.md` and the `PolicySet` surface. Either is defensible; the
current middle state — a documented guarantee with no enforcement — is not.

### B2. `--agent-profile` accepted and discarded — OPEN (Architectural)

R1 made the flag log that it has no effect. The decision — wire `AutonomyConfig.from_agent_profile`
into `run_project_streaming`, or remove the flag — is unchanged.

**Recommendation: wire it.** The classmethod already exists and is already tested; the missing
piece is a parameter on the call path. Removing a documented CLI flag is a user-visible
regression, and the flag's three competing vocabularies (P1-2) get resolved by picking the
built one.

### B3. `autonomy_config.py` Multi-Mode Selector vs `mcp_server.py` dead fork — OPEN

**Decision required.** Which of the two forks is canonical?

**Blocked on evidence, not judgment:** this needs a diff of the two, which the plan noted was
never taken. **New evidence from this session:** `integrations/mcp_server.py` **cannot be
imported at all** (PX-IMPORT1 — `No module named 'orchestrator.integrations.log_config'`), and
neither can its root twin's dependencies resolve cleanly. So the "dead fork" is dead in a
stronger sense than the plan recorded. **Recommendation: fix the import first
(mechanical, see the fix plan's Phase 1), then diff, then decide** — deciding on a module
nobody can import risks preserving the wrong half.

### B4. `TieredModelRouter.next_tier()` / `.escalate_tier()` — OPEN

Needs per-model tier rankings that no artefact in this repo states, and that I decline to
invent. **Deferred-with-reason: requires a maintainer's product knowledge.** The narrower
question a maintainer can answer cheaply: *is tiered escalation a capability you want?* If no,
delete both methods and the question dissolves.

### B5. `TransferLearningEngine` similarity filtering — OPEN

Needs the upstream fact of whether `ExecutionRecord.project_id` is reliably populated across
the meta-optimization pipeline. **Deferred-with-reason: an unanswered data-availability
question, not a design choice.** The investigation is bounded (trace `project_id` writes in
`meta/orchestrator.py`) and should precede any code.

### B6. `dashboard` extra is `ResolutionImpossible` — OPEN (mechanical, but needs a version call)

**Recommendation: raise the httpx/websockets floors** to the lowest pair that resolves, and
add the install to CI so it cannot regress silently. The only judgment needed is which floor;
that is discoverable by running the resolver, not by deliberation.

### B7. `ara_pipelines.py` — a paid LLM call whose result is never weighted — OPEN

**Recommendation: feature-flag it, default off.** The call costs money and its output is
discarded; gating it off is strictly better than the status quo under every hypothesis about
intent, and preserves the code for whoever meant to weight it.

### B9. IDE dev-server ports hardcoded, not session-scoped — OPEN

**Decision required.** What port-allocation scheme? (fixed base + session offset, OS-assigned
ephemeral, or a configured range.)

**Recommendation: OS-assigned ephemeral (bind port 0) with the chosen port reported back to
the UI.** It eliminates the collision and the TOCTOU race P1-3 half-fixed, at the cost of
losing predictable URLs — which the session-scoped alternative also loses.

### B8. `OrchestratorSettings` dead-field recount — **CLOSED by R1.**

---

## Part 2 — New escalations from P3–P11 and T23

### N1. Disposition of 179 unreferenced product files (44,449 LOC) — OPEN (Architectural, largest)

**Finding (P3-ORPHAN0).** 102 ORPHAN + 64 TEST-ONLY + 13 MAIN-ONLY files, 18% of the backend,
are unreachable from any live entry point.

**Decision required.** Per subsystem: **wire, or delete?** This cannot be answered file-by-file
by an auditor — it is a question about which advertised capabilities are real.

**Recommendation: delete by default, wire by exception**, and make the exceptions explicit.
The argument: dead code here is not inert. This session found that the unreachable regions
carry crash-level defects that have never executed (PX-IMPORT1's 12 unloadable modules,
P3-TE3's missing `await`, PX-ROUTER1's unreachable fallback). Every file kept "in case we wire
it later" is a file that will fail the moment it is wired, and meanwhile inflates the audit
surface every wave has to pay for. Trade-off: deletion is irreversible in practice even
though `git` remembers, and some of these represent real unfinished work someone intends to
return to.

### N2. `safety/guardrails.py` — wire or delete — OPEN (Security-adjacent)

**Finding (P3-GUARD0).** 590 LOC of "CRITICAL: Production safety mechanisms" with five stated
GUARANTEES, imported by nothing. If wired as-is it fails open in three ways
(P3-GUARD1/2/4).

**Recommendation: decide the capability first, then fix, then wire — in that order.** Wiring
it unchanged would be worse than leaving it dead, because it would convert "no guardrails" into
"guardrails that report passing without checking".

### N3. Kill-switch files default to world-writable `/tmp` paths — OPEN (Security)

**Finding (P3-GUARD3).** `/tmp/orchestrator_kill` and `/tmp/orchestrator_force_kill` are
predictable and world-writable; `check_and_exit()` answers the force file with `os._exit(1)`.
Any local user can hard-kill the orchestrator.

**Recommendation: move the default under `~/.orchestrator_cache/`** (already the state
directory) and check ownership before honouring the file. Same decision class as A2.
Currently dormant — the module is dead — so it is cheap to fix now.

### N4. Batch-API cost-optimization subsystem — wire or delete — OPEN

**Finding (P3-BATCH0/1/2).** `BatchClient` is reachable only through
`cost_optimization_integration.py`, which nothing imports. As written it hangs 300s unless a
10th request arrives, and its real-provider polling can never succeed (it retrieves a
locally-generated id the provider has never seen).

**Recommendation: delete, or rewrite — do not wire.** The advertised "50% cost reduction"
cannot be delivered by this implementation; wiring it would introduce a 300-second stall on
the evaluation/critique path.

### N5. Multi-agent subsystem — wire or delete — OPEN

**Finding (T23-AGENT1).** `AgentOrchestrator` is constructed only in tests; 9 of 14 agent
implementations are unexported and unreferenced; the coordinator targets `AgentRole.INVESTIGATOR`
whose implementation is among the unexported. A second, entirely separate agent subsystem
(`orchestrator/agents.py`, 277 LOC) is unimportable by construction (P3-SHADOW1).

**Recommendation: pick one subsystem, delete the other.** Which one is a product call. If
`agents/` is chosen, wiring is bounded: export the nine implementations and provide a default
role→agent factory.

### N6. Seven module/package shadowing collisions — RECOMMENDED (mechanical, needs one call)

**Finding (P3-SHADOW1).** `orchestrator/{skills,verification,gateway,agents,workspace,connectors,plugins}.py`
are each shadowed by a same-named package and can never be imported.

**Recommendation: delete all seven.** Five are 7–11-line deprecation shims whose warnings can
never fire — they are pure misinformation. The two substantial ones (`gateway.py` 477 LOC,
`agents.py` 277 LOC) need a one-line confirmation that their package counterparts are the
intended survivors; the evidence says yes for `gateway` (`commands/gateway.py` imports
`..gateway.run`, the package). **This is the cheapest high-value item in the register.**

### N7. Where do `job_id` / `team` come from? — RECOMMENDED (blocks a HIGH money-path fix)

**Finding (P3-COST3).** `policy.py::JobSpec` has no `job_id`/`team` fields, so
`run_job()`'s `getattr(spec, "job_id", "")` is always `""`, and every job leaks its entire
budget reservation permanently.

**Decision required.** Add the two fields to `JobSpec`, or derive them (e.g. `project_id` +
a configured team), or make `BudgetHierarchy` handle the anonymous case correctly?

**Recommendation: do both halves.** (a) Add `job_id: str = ""` and `team: str = ""` to
`JobSpec` so callers *can* attribute spend, and (b) fix `BudgetHierarchy` to release
reservations for anonymous jobs regardless — the reservation/settlement asymmetry at
`cost.py:248` vs `:363` is a bug on its own terms, not merely a consequence of the empty id.
(b) alone stops the leak; (a) is what makes the hierarchy useful. Low trade-off: both are
additive and backward-compatible.

### N8. No CI gate imports every module — RECOMMENDED (highest leverage in the register)

**Finding (PX-IMPORT1).** 12 modules cannot be imported; all eleven CI gates pass anyway,
because nothing ever tries. `grep -rln "walk_packages|iter_modules" tests/` returns nothing.

**Recommendation: add the gate, unconditionally.** ~10 lines walking
`pkgutil.walk_packages(orchestrator.__path__)` and asserting each imports. It closes the entire
class, prevents recurrence, and needs no product decision. The only judgment is whether to
land it **red** (documenting the 12 as `xfail` and burning them down) or fix the 12 first and
land it green — **recommend fixing first, landing green**, since all 12 fixes are mechanical.

---

## Withdrawn

### W1. "`sagas.py`'s saga events are silently queued forever" — WITHDRAWN

P2-UEB1 recorded this as a live consequence of the event bus never being `start()`ed. This
session established that `engine_core/sagas.py` **cannot be imported at all** (PX-IMPORT1) and
that its `self.event_bus` is a bare coroutine when no bus is injected (PX-BUS1), so
`await self.event_bus.publish(event)` raises `AttributeError` before reaching any queue. The
mechanism P2 described is real for the bus in general; the claim that this path was live is
withdrawn. The underlying `UnifiedEventBus.start()` fix P2 shipped remains correct.

### W2. "`quality/preflight.py` is the Chain-of-Responsibility module `CLAUDE.md` names" — WITHDRAWN

Raised as a candidate architecture-doc inconsistency during P3 and disproved by the innocence
check: `orchestrator/preflight.py` (root) *is* referenced and live, and is what the table
means. Only the `quality/preflight.py` twin is orphaned. `CLAUDE.md`'s Validation row is
correct as written.
