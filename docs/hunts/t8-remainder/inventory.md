# T8 — Remainder, Coverage-Ordered — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4/§7.5, §8 ("T8 is explicitly PARTIAL and
never 'completes'"). Budget: 8 candidates surveyed (self-chosen cap 12), 5
fixed.

## Phase 0 — candidate source

Rather than a fresh full-codebase survey, T8's Phase 0 delta drew its entire
candidate pool from the residual backlog T0–T7 had already flagged in
`docs/hunts/INVENTORY.md` — real gaps each prior tier named but declined to
fix due to scope/budget. This is exactly what the plan's own running-inventory
mechanism (§5) exists for: "a later tier does not need to re-survey these." A
background agent ran Phase 1 (reachability) + Phase 3 (trigger + innocence)
on all 8; Phase 4 below records every disposition.

## Candidates fixed

### C1 — VERIFIED DEFECT — `generators/secrets_manager.py`'s `SecretsFilter` built but never attached to a real logger

- **Property violated:** a security mitigation that exists in code but is
  never wired in provides zero actual protection — indistinguishable from
  not having built it at all.
- **Location:** `orchestrator/log_config.py::configure_logging()` (the one
  real, exported logging-setup function) attached `CorrelationIdFilter` to
  every handler but never `SecretsFilter`.
- **Finding:** confirmed via independent re-trace, not just the T2 summary:
  the only two `.addFilter(SecretsFilter(...))` call sites in the whole repo
  were inside `generators/secrets_manager.py` itself, one in a helper with no
  callers, one inside `setup_secure_logging()`, which itself had zero
  callers anywhere.
- **Reachability:** the *absence* of the mitigation is live — every real
  `logging.getLogger(__name__)` call in the app, once `configure_logging()`
  is invoked, would have gone out unmasked.
- **Innocence attempt:** "maybe some other module wires it via
  `logging.config.dictConfig`." Checked — no `dictConfig`/`fileConfig` call
  anywhere in `orchestrator/`. Innocence fails.
- **Fix:** `configure_logging()` now attaches `SecretsFilter()` to both the
  console and file handlers, alongside the existing `CorrelationIdFilter()`.
- **Not fixed here (separate, larger question):** `configure_logging()`
  itself has zero live callers today (no `cli.py`/`api_server.py` entry
  point calls it) — wiring it into a real startup path is an architecture
  decision (what default level/format/log_file applies process-wide), out
  of scope for a security-filter-installation fix. Flagged
  `[REQUIRES HUMAN REVIEW]`.
- **Test:** `test_c1_configure_logging_installs_secrets_filter` — calls
  `configure_logging()`, logs a real OpenAI-style key pattern through a
  child logger, asserts via `capsys` the raw secret does not appear in
  captured output and a `[REDACTED_...]` marker does (pre-fix: raw secret
  leaked verbatim).

### C4 — VERIFIED DEFECT — `plugin/plugin_isolation_secure.py`'s seccomp network-syscall loop silently swallowed rule-install failures

- **Property violated:** a sandbox boundary (network egress blocking) that
  can silently fail to install, with the process still logging "seccomp
  policy loaded" as if nothing went wrong — the plan's "silent fail-open of
  a security boundary" threat class.
- **Location:** `orchestrator/plugin/plugin_isolation_secure.py`, the
  network-syscalls loop inside `_apply_seccomp()`.
- **Finding:** `except Exception: pass` with no log at all, unlike the
  sibling "dangerous syscalls" loop a few lines above, which at least
  documents its own silence as an accepted arch-dependent tradeoff
  (`# Syscall may not exist on this arch`). A per-syscall failure here (e.g.
  a specific network syscall rule failing to install) was indistinguishable
  from full success.
- **Reachability:** `SecureIsolatedRuntime`/the whole `orchestrator/plugin/`
  (singular) package has zero live callers anywhere in the repo — confirmed
  independently by T9's later, broader survey of the same package. Currently
  dead code.
- **Innocence attempt:** "it's dead, so the silent failure doesn't matter
  today." Holds for current reachability. Fixed anyway because the shape is
  cheap to fix and security-relevant the moment this subsystem is ever wired
  in — matching this hunt's established practice of fixing defects inside
  dormant-but-real subsystems (e.g. T1-C4) without wiring the subsystem
  itself live.
- **Incidental finding while testing this candidate:** the whole
  `orchestrator/plugin/` package could not even be imported —
  `plugin_isolation_secure.py:40-44` imported `Plugin` from
  `.plugin_isolation`, which has no such class. Traced the real `Plugin`
  class to the sibling module `.plugins` (`orchestrator/plugin/plugins.py`,
  a `PluginManifest`-based base class matching this file's usage, not the
  unrelated `orchestrator/plugins/` (plural) package's own differently-shaped
  `Plugin(ABC)`). Fixed the import to reference the correct sibling module.
  Without this fix the module — and therefore this candidate — could not be
  tested at all.
- **Fix:** logs a warning naming the failed syscall and exception; fixed the
  blocking `Plugin` import.
- **Test:** `test_c4_seccomp_network_rule_failure_is_logged` — injects a fake
  `seccomp` module (real `seccomp` isn't installed in this environment) whose
  `add_rule` raises for the `"socket"` syscall, asserts a warning naming it
  is logged (pre-fix: nothing logged).

### C5 — VERIFIED DEFECT — `generators/website_validator.py`'s secret scanner silently skipped unreadable files and reported a clean scan

- **Property violated:** "silent wrong result" — a scan that couldn't read
  every file must not look identical to one that scanned everything and
  found nothing.
- **Location:** `_check_secret_exposure()`'s per-file loop.
- **Finding:** `except Exception: continue` with zero logging; if a file
  failed to read (permission error, TOCTOU delete, symlink loop), it was
  silently excluded and, if no other leaks were found, the check reported
  `passed=True, score=1.0, "No secrets found in frontend code"`.
- **Reachability:** LIVE and the highest-severity fix in this tier. Traced
  the full call chain: `orchestrator website --min-quality`/
  `--require-all-checks` (real, documented CLI flags) →
  `WebsiteGenerator.generate()` → `WebsiteQualityValidator.validate()` →
  `_check_secret_exposure()` → feeds `_apply_quality_gate()`'s
  ship/no-ship decision. By default the gate is a no-op
  (`min_quality=0.0`), but the flags exist specifically for the CI/CD
  gating use case where a false-clean result would matter.
- **Innocence attempt:** "the default invocation doesn't gate on this, so
  it's harmless." Only partially holds — fails for the documented,
  discoverable, intended gating usage. Innocence fails.
- **Fix:** logs a warning naming the unreadable file; an unreadable file now
  makes the check `passed=False` (never rolls into a clean score), and the
  `details`/`recommendations` fields say the scan is incomplete and name
  which file(s) to re-check. Same fix pattern applies to the two dead
  `quality_control.py` duplicate copies T6 flagged (`orchestrator/
  quality_control.py`, `orchestrator/quality/quality_control.py`) but those
  were not touched this tier — zero live callers (`analyze_on_complete`
  defaults `False` everywhere), lower priority, `[REQUIRES HUMAN REVIEW]`
  if that flag is ever flipped live.
- **Test:** `test_c5_website_validator_flags_unreadable_frontend_file` — a
  real trigger: a directory named `bad.js` (reading a directory as text
  raises `IsADirectoryError`, deterministic and root-safe, no reliance on OS
  permission bits), asserts `passed is False` and a warning names the file
  (pre-fix: `passed is True`, nothing logged).

### C6 — VERIFIED DEFECT — `operations/diagnostics.py`'s environment check required keys the live client never reads

- **Property violated:** a health check whose required-variable list doesn't
  match what the code it's diagnosing actually reads is worse than no check
  — it produces both false CRITICALs and false HEALTHYs.
- **Location:** `SystemDiagnostic._check_environment()`.
- **Finding:** required `OPENAI_API_KEY`/`GOOGLE_API_KEY`/
  `ANTHROPIC_API_KEY`/`MINIMAX_API_KEY`. Independently verified (not just
  trusting the survey) via direct read of
  `infrastructure/llm_client.py::UnifiedClient.__init__` and
  `_get_client_for_model()`: the live client reads `OPENROUTER_API_KEY`
  (primary — routes every OpenAI/Google/Anthropic-branded model),
  `DEEPSEEK_API_KEY` (direct fallback), and `XAI_API_KEY` (direct Grok,
  optional). A repo-wide grep confirms zero code anywhere reads
  `OPENAI_API_KEY`/`GOOGLE_API_KEY`/`ANTHROPIC_API_KEY` via
  `os.environ.get`/`os.getenv`.
- **Reachability:** `SystemDiagnostic` has zero live callers (no CLI
  subcommand, no health endpoint) — currently dead, but the drift is real
  and would mislead the first person who wires it up or runs it manually.
- **Innocence attempt:** none needed beyond reachability — the check's own
  `suggested_fix` message actively told an operator to set the wrong
  variable.
- **Fix:** `required_vars` now lists `OPENROUTER_API_KEY`/
  `DEEPSEEK_API_KEY`/`XAI_API_KEY`; `suggested_fix` message updated to match.
- **Explicitly NOT fixed here — flagged `[REQUIRES HUMAN REVIEW]`:**
  `CLAUDE.md`'s own "Required env vars" section and `.env.example` both
  document `OPENAI_API_KEY`/`DEEPSEEK_API_KEY`/`GOOGLE_API_KEY`/
  `ANTHROPIC_API_KEY` as the required keys — the same stale claim, in the
  project's user-facing setup docs, not just this one health check. This
  wasn't rewritten because it's ambiguous whether the project intentionally
  migrated to OpenRouter-as-gateway (in which case the docs are simply
  outdated) or whether direct-provider support was meant to still work and
  is itself a gap — that's a product decision, not a one-line doc fix I can
  make confidently. Recorded here so a future tier (or a human) can decide
  with full context rather than re-discovering it.
- **Test:** `test_c6_diagnostics_accepts_openrouter_key_alone` — clears every
  provider key, sets only `OPENROUTER_API_KEY`, asserts no `ENV001`
  CRITICAL issue is raised (pre-fix: CRITICAL raised for a correctly
  configured setup).

### C7 — VERIFIED DEFECT — `adaptive_router.py::AdaptiveRouter.is_available()` ignored committed DISABLED/DEGRADED state under lock contention

- **Property violated:** a "fast path" that returns a wrong answer under a
  real, non-rare condition (any concurrent writer holding the lock) is not
  an optimization, it's a correctness bug.
- **Location:** `is_available()`.
- **Finding:** `if self._lock.locked(): return True` ran *before* checking
  `self._disabled`/`self._degraded_since` — whenever any coroutine held the
  lock for any reason (recording a timeout/success/latency on *any* model),
  every call to `is_available()` for *every* model returned `True`
  unconditionally, including for a model already permanently `DISABLED` by
  a prior `record_auth_failure()`.
- **Reachability:** `container.py` hardcodes `adaptive_router=None` at
  every site (confirmed: the live `Orchestrator` never constructs a real
  one); the one real `AdaptiveRouter()` construction
  (`engine_core/outcome_router.py`) is itself inside a class with zero live
  callers, one of which (`router_integration.py`) has its own unrelated
  broken import (`get_adaptive_router` doesn't exist). Currently fully dead.
- **Innocence attempt:** "dead today, so it never fires." Holds for
  reachability. Fixed anyway — one `container.py` edit away from being live,
  and the fix is a deletion, not an addition (see below).
- **Fix:** removed the lock-check fast path entirely rather than reordering
  it — the method's own docstring already establishes that reading
  `self._disabled`/`self._degraded_since` needs no lock in CPython (`dict.get()`
  is a single atomic C call with no `await` points), so the "optimistic"
  branch bought no real safety and only introduced the bug.
- **Not fixed here:** whether `AdaptiveRouter`/`OutcomeWeightedRouter`
  should be wired into `container.py` at all, and
  `router_integration.py`'s unrelated broken import (dead, zero importers,
  currently harmless) — both `[REQUIRES HUMAN REVIEW]`/`[UNK]`, architecture
  decisions out of scope for a state-machine correctness fix.
- **Test:** `test_c7_is_available_reflects_disabled_state_during_concurrent_write`
  — disables a model, holds `router._lock` (simulating a concurrent writer),
  asserts `is_available()` still returns `False` (pre-fix: returned `True`
  unconditionally while the lock was held).

## Candidates cleared (innocent) or explicitly not fixed

- **C2 — `gateway.py`/`multi_tenant_gateway.py` key comparisons —
  CLEARED, wrong threat model.** Neither does a literal `key == stored_key`:
  `gateway.py::APIGateway.verify_api_key()` hashes with SHA256 then does a
  dict lookup (no exploitable timing side-channel from a hash digest
  comparison); `integrations/multi_tenant_gateway.py` looks up the raw key
  in a dict via CPython's randomized SipHash, not a linear scan. Both are
  also dead code (zero constructors found; `gateway.py` is additionally
  permanently shadowed by the sibling `gateway/` package and cannot be
  imported by its own dotted path — a pure hygiene landmine, not part of
  this candidate's threat class, recorded here so it isn't re-litigated).
  The live HTTP surface (`api_server.py`) already uses
  `hmac.compare_digest` throughout. Not fixed — nothing to fix.
- **C3 — `safety/sandbox.py`/`safety/secure_execution.py` — SPLIT.**
  `secure_execution.py` — CLEARED, read in full, no defect found; correctly
  resolves symlinks before containment-checking. `sandbox.py` — dead
  (zero constructors anywhere), and its `validate_code()` denylist is
  trivially bypassable (`"from os import system"` contains no `"import os"`
  substring) with resource limits declared but never enforced against the
  subprocess. `[REQUIRES HUMAN REVIEW]` — not fixed: building real
  `resource.setrlimit`/cgroup enforcement into dead code nobody asked for is
  scope creep, not a bug fix; the live sandboxing path is the
  container-based `DockerSandbox` via `code_executor.py`, unaffected.
- **C8 — `services/scorers.py`, `rate_limiter.py::fetch_current_spend` —
  CLEARED as live risks.** Both confirmed fully dead by exhaustive grep
  (zero constructors/callers anywhere). `EvaluatorScorer`'s `except
  Exception: return 0.0` is a defensible fail-safe-low design if ever wired
  in (unlike T6's higher-severity `evaluator.py` 0.5-fallback, a genuinely
  neutral fake score). Not fixed — dead, and the one design note (sentinel
  vs. 0.0) is a judgment call for whoever wires it in, not a bug today.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1/3 survey delegated to one background
agent run). `fix_revisions`: 1/1 per fixed candidate, except C4, which
required one incidental additional fix (the blocking `Plugin` import) before
its own test could even run — recorded above as part of C4's own writeup,
not a revision of the syscall-logging fix itself.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
