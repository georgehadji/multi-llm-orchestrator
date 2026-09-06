# P2 (DEEP tier) — coverage & residual risk

## Surface audited

All 16 files in the P2 manifest row, full read top-to-bottom (not sampled):
`integrations/slack_integration.py` (1530), `models.py` (1389),
`nash/infrastructure_v2.py` (1239), `architecture_rules.py` (1260),
`domain/ports.py` (853), `infrastructure/caching.py` (789),
`unified_events/core.py` (1122), `events/ab_testing.py` (1043),
`engine_core/container.py` (1029), `transfer_learning.py` (955),
`codebase/context.py` (943), `state_mgmt/telemetry_store.py` (942),
`streaming.py` (718), `operations/diagnostics.py` (496),
`model_selector.py` (245), `cli.py` (246) — 14,799 LOC total, matches the
wave manifest exactly.

Two adjacent files outside the 16 were read as direct dependencies during
Innocence Checks, not as new P2 scope: `orchestrator/streaming.py`'s twin
`infrastructure/streaming.py` (already a P1 file; a residual bug from
P1's own incomplete coverage was found there and fixed under this wave,
see inventory.md P2-S2-2b), and `meta/orchestrator.py`/`meta_orchestrator.py`
(a dependency of `transfer_learning.py` and `events/ab_testing.py`, read to
verify canonical/shim status and check `ExecutionRecord`'s field shape).

## Defect classes covered

All six V4 taxonomy classes were actively hunted per file: Memory/Resource
(found: P2-SLACK1), Concurrency (found: P2-ENGINE1, P2-UEB1, P2-S2-2,
P2-S2-2b, NASH-2 residual, NASH-3 residual), Injection/Taint (checked —
`codebase/context.py::QualityAnalyzer._run_tool` builds subprocess argv
lists with no `shell=True`, not injectable; `slack_integration.py`'s HMAC
verification correctly uses `hmac.compare_digest`; no findings), Edge cases
(found: P2-SLACK3, P2-ARCH2), Dependencies/Config (found: P2-TRANSFER2;
scipy was confirmed an existing declared dependency before using it in
P2-AB1's fix), Logic (found: P2-NASH1, P2-AB1, P2-TRANSFER1, P2-MODELS1,
P2-M2-3, P2-ARCH1, P2-PORTS1, P2-TELEMETRY1 — the majority of this wave's
yield, consistent with full-file reads surfacing "declared machinery, never
wired" more than any other single shape, now confirmed 8 times in this wave
alone across 6 different files).

## Clean-claim scope

Files `operations/diagnostics.py` and `cli.py` were read in full and yielded
no new findings beyond what T2/T8/T15 had already fixed in them (verified
those fixes are still intact in current source). `codebase/context.py` was
read in full with no VERIFIED finding (one weak, non-crashing edge case
noted, not fixed — see below). Claim: **these 3 files, all 6 taxonomy
classes, no VERIFIED defect found beyond prior tiers' fixes.**

For the other 13 files, the claim is narrower: **audited for all 6 taxonomy
classes, with the findings in inventory.md as the complete VERIFIED output**
— not "bug-free". `engine_core/container.py`'s ~500-line `build()` factory
method was read in full but is mostly wiring (try/except ImportError
fallback chains); no new finding beyond confirming P2-UEB1's and P2-M2-2's
mechanisms. `unified_events/core.py`'s ~1100 lines were read in full;
its own logic (EventStore, Projections, HookRegistry) is internally
consistent — the only finding chargeable to this file is P2-UEB1, which is
a wiring gap in a *different* file (`container.py`) that constructs it.

## Runtime-dependent set

- **P2-TRANSFER1 (full fix)**: resolves once `ExecutionRecord.project_id`'s
  reliability across the meta-optimization pipeline is confirmed (or fixed)
  outside this wave's scope.
- **P2-ARCH1's escalated half** (real health-aware model fallback): resolves
  once `ArchitectureRulesEngine` is given an `api_health`/health-tracker
  dependency to check against — an architectural decision, not a data gap.
- **P2-M2-2**: resolves only if someone decides `TieredModelRouter`'s tier
  escalation should be real, and assigns relative "power" tiers to every
  model — an explicit product input this hunt cannot supply.
- **P2-SLACK2**: resolves once a human decides whether missing-config should
  fail open or closed for Slack request verification.

## Highest-value next step

`engine_core/sagas.py` (not a P2 file; discovered as P2-UEB1's live
caller) is worth a dedicated read: it is the *only* confirmed real
`.publish()` caller onto the container's `UnifiedEventBus`, which means it
is also the file most likely to reveal what real behavior was silently
broken by P2-UEB1 before this fix (dead projections, e.g. `ProjectStateProjection`
never updating for saga-originated events). Second: a dedicated shape-sweep
for "`get_event_bus()` called without `await`" specifically — this exact
bug shape recurred **7 times** across this session's P1+P2 work
(`streaming.py` x2 classes, `infrastructure/streaming.py` x1,
`dashboard_core/core.py`, `cli_nash.py` x2, `projections.py` x1 module-level
+ x2, `analysis/projections.py` x1 module-level + x2, `engine_core/sagas.py`
x1) — a mechanical AST check (flag `X = obj_or(…, ASYNC_CALL())` /
bare `ASYNC_FUNC()` assigned without `await`) would catch all of these in
one gate, the same T17-T22 pattern already used for 6 other recurring
shapes in this codebase.

## Uncertainty acknowledgment

**Most likely false positive:** P2-PORTS1 — no current caller passes
`quality_score=`, so if the Protocol's own declaration of that parameter is
itself vestigial (never intended to be used), the "fix" is cheap and
harmless either way, but the finding's practical urgency could be zero
rather than latent.

**Most likely missed real defect:** `slack_integration.py`'s
`SlackNotifier`/`RunSummaryFormatter`/`SequentialABTest`/`MultiArmedBandit`/
`CUPEDAdjustment` (the "Advanced A/B Testing Features" section of
`events/ab_testing.py`) were read but not adversarially stress-tested the
way `RateLimiter`/`parse_overrides`/`StatisticalAnalyzer` were — their
math (Thompson Sampling epsilon-greedy allocation, CUPED covariance
estimator) checked out symbolically on inspection but was not executed
against reference values the way P2-AB1's t-distribution CDF was.

**Tail coverage:** Toggle C (HEAD+TAIL elicitation) was applied per-file as
configured; no atypical-class finding surfaced beyond what's in the table —
this itself is data (either the tail is genuinely quieter than the head at
this priority band, or K=8 with Toggle C still isn't enough to reliably
surface the rarest classes; cannot distinguish between these from one wave).

**Cannot be determined statically:** whether P2-TRANSFER1's `find_transferable_patterns`
is *called* anywhere beyond its constructing site (`meta_integration.py`) —
confirmed the class is live, not confirmed this specific method executes on
any real path. Whether P2-SLACK2's fail-open branch has ever actually fired
in any real deployment of this code (module is unwired in this repo, but the
file is explicitly designed for external consumers to wire in).

**Input that would most raise confidence:** a real `openrouter.ai`-reachable
environment to re-run `test_openrouter_model_audit.py`'s two failures as an
independent cross-check that this wave's changes to `models.py` (docstring
only, no data changes) didn't disturb anything catalogue-related; a grep
across the *whole* repo (not just P2's 16 files) for the "async factory
called without await" shape, since 7 confirmed instances in a 16-file
sample strongly suggests more exist in the other 874 files.
