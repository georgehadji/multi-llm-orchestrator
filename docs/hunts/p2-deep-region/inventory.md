# V4 precision audit — wave P2 (DEEP tier, priority 8–7)

16 files, 14,799 LOC: `integrations/slack_integration.py`, `models.py`,
`nash/infrastructure_v2.py`, `domain/ports.py`, `infrastructure/caching.py`,
`streaming.py`, `operations/diagnostics.py`, `cli.py`, `model_selector.py`,
`architecture_rules.py`, `unified_events/core.py`, `events/ab_testing.py`,
`engine_core/container.py`, `transfer_learning.py`, `codebase/context.py`,
`state_mgmt/telemetry_store.py`. Config: `APPLY_FIXES=ON`,
`TOGGLE_B_INNOCENCE=ON`, `TOGGLE_C_TAIL_SWEEP=ON`, `K=8`, same as P1.

Ledger ingested first per V4 rule 1C: `docs/hunts/INVENTORY.md` (918 lines at
start of this wave) — informed several dispositions below (models.py's
already-fixed enum-aliasing bug not re-raised; `unified_events/core.py`'s
`.start()` gap re-examined fresh rather than carried forward as still
"a bigger wiring decision" — see UEB-1; `architecture_rules.py`/
`transfer_learning.py` root-vs-subpackage import confirmed canonical/shim
correctly beforehand).

| ID | Severity | Evidence | Reach | Location | Category | Violated Property |
|----|----------|----------|-------|----------|----------|--------------------|
| P2-NASH1 | **CRITICAL** | VERIFIED-EXEC | REACHABLE from `nash/__init__.py` (wildcard) + `nash/monitor.py` | `nash/infrastructure_v2.py::WriteAheadLog.append()` | Logic/async-misuse | A two-phase-commit write must not guarantee a crash |
| P2-AB1 | **HIGH** | VERIFIED-EXEC | REACHABLE, `TransferLearningEngine`-adjacent A/B engine, min_samples caller-configurable | `events/ab_testing.py::StatisticalAnalyzer._t_distribution_cdf/_incomplete_beta` | Logic/numerical | A p-value must be a valid probability in [0,1] |
| P2-ENGINE1 | MEDIUM | VERIFIED-STATIC | REACHABLE — `run_project_streaming()` called from `supervisor/service.py` (persistent REPL) and `cli_dispatch.py` | `engine.py::Orchestrator.run_project_streaming` | Concurrency/shared-state | `self._event_bus` must stay the container-wired bus for the instance's lifetime |
| P2-UEB1 | MEDIUM | VERIFIED-STATIC, well-mechanized | REACHABLE — `engine_core/sagas.py:624` publishes real events to this exact bus | `engine_core/container.py` (constructs) / `unified_events/core.py::UnifiedEventBus` | Concurrency/wiring-gap | A queued event must eventually be processed |
| P2-S2-1 | MEDIUM | VERIFIED-EXEC | DEAD (no live constructor of `StreamingPipeline` found) | `streaming.py::StreamingPipeline._run_pipeline` | Logic/scope | A referenced name must be in scope |
| P2-S2-2 | MEDIUM | VERIFIED-STATIC→EXEC | **REACHABLE** — `engine.py::run_project_streaming` (core) + `api_server.py::_stream_via_event_bus` (SSE) both construct this | `streaming.py::ProjectEventBus.__init__` | Concurrency/async-misuse | An async factory must be awaited before use |
| P2-S2-2b | MEDIUM | VERIFIED-STATIC→EXEC | Same class shape, P1-follow-up | `infrastructure/streaming.py::ProjectEventBus.__init__` | Concurrency/async-misuse | Same as P2-S2-2 |
| P2-TRANSFER1 | MEDIUM | VERIFIED-STATIC | REACHABLE — `meta_integration.py` constructs the engine live; specific method's own caller not traced (outside P2 scope) | `transfer_learning.py::TransferLearningEngine.find_transferable_patterns` | Logic/silent-bypass | Results must be filtered by the similarity computation the method performs |
| P2-MODELS1 | MEDIUM | VERIFIED-STATIC | REACHABLE — `engine_core/stages/generate.py:97` | `models.py::vs_variant_for` | Logic/dead-branch | Docstring-promised branches must exist in code |
| P2-M2-3 | MEDIUM | VERIFIED-STATIC | REACHABLE — `engine.py:1180`, `application/decomposer.py:706` pass real descriptions | `model_selector.py::ModelSelector.decomposition_model` | Logic/wiring-gap | An accepted parameter must influence its documented decision |
| P2-ARCH1 | LOW-MEDIUM | VERIFIED-STATIC | REACHABLE when `client` is passed to `ArchitectureRulesEngine` | `architecture_rules.py::ArchitectureRulesEngine._generate_rules_with_llm` | Logic/dead-branch | A "check availability" comment must correspond to a check |
| P2-ARCH2 | LOW | VERIFIED-EXEC | REACHABLE, same call path as ARCH1's sibling method | `architecture_rules.py::ArchitectureRulesEngine._optimize_rules_with_llm` | Edge case | An embedded JSON example must itself be valid JSON |
| P2-PORTS1 | LOW-MEDIUM | VERIFIED-EXEC | DORMANT — no current caller passes `quality_score=`, but the Protocol declares it | `domain/ports.py::NullTelemetry.record_call` | Logic/contract-drift | A Null adapter's call signature must match its Protocol |
| P2-TELEMETRY1 | LOW-MEDIUM | VERIFIED-EXEC (mechanism) | Recovery path had zero callers anywhere | `state_mgmt/telemetry_store.py::TelemetryStore.drain_queue` | Resource/wiring-gap | A documented warm-start recovery step must run at warm-start |
| P2-SLACK1 | LOW | VERIFIED-STATIC | DEAD (whole module unwired into any live server) | `integrations/slack_integration.py::RateLimiter.is_allowed` | Memory/resource | A sliding-window limiter's stored state must be bounded by the window |
| P2-SLACK3 | LOW | VERIFIED-EXEC | DEAD (same module) | `integrations/slack_integration.py::TemplateRegistry.parse_overrides` | Edge case | A malformed user-supplied value must not crash the handler |
| P2-TRANSFER2 | LOW | VERIFIED-STATIC | REACHABLE — fires a real `DeprecationWarning` on every import | `transfer_learning.py` (import statement) | Dependencies/hygiene | A live module should not import through a deprecated shim |

**Fix:** all 17 rows above → `PACKAGE`, applied (`APPLY_FIXES=ON`), except
P2-TRANSFER1 which is a **partial** fix (visibility only; full similarity
filtering is escalated — see below).

## Escalated — `[REQUIRES HUMAN REVIEW]`, not fixed

- **P2-SLACK2**: `SlashCommandHandler.verify_signature()` returns `True`
  (verified) when `signing_secret` is unset, logging only a WARNING — fails
  open, not closed, on missing config. The HMAC check itself is correct
  (`hmac.compare_digest`, correct base string, 300s replay window) when a
  secret *is* present. Currently DEAD (module unwired), but this file's own
  trailing docstring hands a deployer a verbatim FastAPI integration example
  — anyone who follows it and wires in a real `TemplateRunner` while
  forgetting the secret would silently accept unauthenticated slash commands
  that trigger real orchestrator runs. Whether a verification gate should
  fail open at all when unconfigured is a product decision this hunt has
  consistently left to the repo owner, not flipped unilaterally.
- **P2-TRANSFER1 (full fix)**: completing real similarity-based filtering
  requires `PatternMiner` to actually populate `TransferPattern.source_projects`
  (currently hardcoded `[]` — `meta/orchestrator.py::ExecutionRecord` does
  carry a `project_id` field, so the data exists, but whether it's reliably
  populated across the whole meta-optimization pipeline was not traced —
  that pipeline is outside this wave's 16-file scope). Inventing a fallback
  behavior for unattributed patterns would itself be a product decision.
- **P2-M2-2**: `TieredModelRouter.next_tier()`/`.escalate_tier()` have zero
  callers anywhere (confirmed by repo-wide grep), even though the class
  itself is live (`engine.py` uses `.available_models()`). `_MODEL_TIERS`
  assigns tier 0 to every real text model and tier 1 only to an image model
  (`NANO_BANANA_2`), so even if wired, escalation would be a no-op for
  virtually all real task types. A genuine fix requires inventing relative
  "power" rankings for dozens of models — a product judgment call, not a
  mechanical fix. Per V4's own innocence-check option ("unreachable from
  every anchor"), this is close to clearable; recorded rather than silently
  dropped since the underlying data structure is genuinely mischaracterized
  by its own name.

## Cleared (innocent), with evidence

- **M2-1**: `model_selector.py::_MODEL_TIERS` declares `Model.ZHIPU_GLM_5_2`
  twice (lines 42 & 44) mapping to the identical value (`0`). Checked the
  full `Model` enum for a plausible missing third GLM variant — none exists.
  A dict literal reassigning the same key to the same value is a provable
  no-op; VERIFIED-STATIC innocence.
- **CACHE-1** (reconsidered mid-fix): `infrastructure/caching.py::DiskCache.get_stats()`
  does blocking sqlite I/O while its sibling methods (`get`/`set`/`delete`/
  `clear`/`keys`) all correctly use `asyncio.to_thread`. Initially planned as
  a fix, but `get_stats()` is declared **sync** on the `CacheBackend` ABC
  itself — by design, not oversight, and shared by `InMemoryCache`/
  `RedisCache` too. Making it non-blocking would mean async-ifying the whole
  ABC plus `MultiLayerCache`'s aggregator — an architecture change, not a
  minimal fix, for a module confirmed to have zero external callers.
  Downgraded to a documented residual (see coverage.md).
- **CACHE-2**: `infrastructure/caching.py::DiskCache` shares a class name
  with the unrelated, incompatible `infrastructure/cache.py::DiskCache`
  (the one that actually satisfies `CachePort`). Real naming collision, but
  `infrastructure/caching.py` is confirmed fully dead (zero external
  importers of the module or its `orchestrator/caching.py` shim) — no live
  path can actually import the wrong one today.
- **NASH-2**: `nash/infrastructure_v2.py`'s `UnifiedEventBus`/
  `TransactionalStorage` both construct `AsyncIOManager()` directly — the
  constructor its own docstring calls "DEPRECATED: Use
  AsyncIOManager.get_instance() instead" — spinning up redundant
  `ThreadPoolExecutor`s instead of sharing the TD-003 singleton fix. Real,
  but `__init__` cannot be `async def`, so properly sharing the singleton
  needs an async factory method refactor, not a one-line fix.
- **NASH-3**: two unrelated classes are both named `UnifiedEventBus`
  (`unified_events/core.py` and `nash/infrastructure_v2.py`) with
  overlapping method names (`publish`/`subscribe`) but incompatible
  signatures. `nash/infrastructure_v2.py`'s own module docstring literally
  lists "UnifiedEventBus (unified_events/)" as a different system it routes
  between — aware of, not resolving, the collision. No live crash found;
  pure naming-collision hygiene risk in a codebase that already uses this
  name meaningfully elsewhere.

**Scope limit:** two shapes (Concurrency/async-misuse and Logic/wiring-gap)
account for 11 of 17 fixes — the same recurring shapes T0-T22 and P1 already
established, now confirmed to recur at this depth too. Full detail on what
was and wasn't read: `coverage.md`.
