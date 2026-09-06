# T9 — Safety/Execution/Plugin Surface — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4/§7.5 and `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
wave definition. Budget: 48 files surveyed (46 discovered + 2 T7 carry-overs), 4 fixed
(spanning 6 source files — 3 of the 4 fixes are duplicate-file shim conversions).

## Phase 0/1/2/3 — survey method

A background agent read all 46 files in the wave's declared scope (`safety/`,
`security/`, `plugin/`+`plugins/`, `gateway/` subpackage, and root-level
security-adjacent modules), plus 2 files T7 explicitly named but never
opened (`safety/sandbox.py`, `safety/secure_execution.py`), tracing live
reachability for each via repo-wide grep before ranking any candidate. Full
per-file disposition table and ranked top-10 preserved verbatim in this
session's transcript; the 4 pursued to a fix this tier are below.

## Headline finding (not itself a single fix, context for the ranking below)

The dominant defect shape in this wave was not duplicate-file drift (the
pattern every prior tier found repeatedly) — it was **security controls
that are fully built, wired into a `ServiceCollection`/dataclass field, and
then never actually read by anything that executes generated code**:
`safety/tool_guardrails.py::ToolCallGuardrailController`,
`safety/command_guard.py::classify_command()`, and the entire
`orchestrator/plugin/` (singular) isolation subsystem all show this shape.
Only one of these three was fixed this tier (tool_guardrails.py's false
documentation claim, below) — wiring any of them into a live execution
path is an architecture decision, consistent with how every prior tier
(T1 C3/C6, T5 adaptive_router, T8 C1) has handled "should this dormant
subsystem go live" questions.

## Candidates fixed

### C1 — VERIFIED DEFECT — three dead safety/ duplicates carried broken imports (same shape T1 already fixed once)

- **Property violated:** a root-vs-subpackage duplicate pair silently
  diverges and one copy breaks — the dominant defect shape T1/T2/T3/T5/T7
  already found and fixed repeatedly in this hunt.
- **Locations:**
  - `safety/architecture_rules.py:830,982` — `from .models import Model as M`
    should have been `from ..models import Model` (one dot short in one
    place after a prior manual "FIXED:" attempt, one dot too many in
    another — confirmed via diff, not just grep, against the canonical
    `orchestrator/architecture_rules.py`).
  - `safety/architecture_advisor.py:516` — `from ...api_clients import
    UnifiedClient`, one dot too many for this file's depth.
  - `safety/reference_monitor.py` — `from ...specs import ...`, one dot too
    many, inside a `TYPE_CHECKING`-only block (verified directly — no
    runtime impact from this specific broken import, unlike the other two).
- **Finding:** confirmed via direct diff against each canonical root module
  (`architecture_rules.py`, `architecture_advisor.py`, `reference_monitor.py`
  — all three genuinely live, wired into `engine_core/architect.py`,
  `appbuilder/detector.py`/`app_detector.py`, and `control_plane.py`
  respectively) that only the relative-import depth differs — no other
  content divergence in `architecture_rules.py`/`reference_monitor.py`.
  `architecture_advisor.py`'s `safety/` copy had additionally fallen behind
  the canonical module: it was missing the `"static"` project type's
  tech-stack/topology dict entries entirely (a real data-staleness gap, not
  just a broken import).
- **Reachability:** all three `safety/` copies confirmed DEAD (zero live
  callers anywhere, exhaustive repo-wide grep) — the canonical root modules
  are what's actually imported and used.
- **Innocence attempt:** "dead code, so the broken import never fires."
  Holds for current reachability. Fixed anyway, matching this hunt's
  established, consistent practice for exactly this situation (T1's
  `costing/core.py`, T2's `codebase_writer.py`, T5's `circuit_breaker.py`,
  T7's `appbuilder/verifier.py`): convert the dead duplicate into a
  re-export shim of its live canonical, closing the landmine and the
  staleness gap in one move, at zero behavioral risk since nothing calls
  the shimmed module today.
- **Fix:** all three converted to `from ..<canonical> import *` shims.
- **Test:** `test_c1_safety_architecture_rules_is_canonical`,
  `test_c1_safety_architecture_advisor_is_canonical`,
  `test_c1_safety_reference_monitor_is_canonical` — each asserts the
  `safety/`-imported class **is** (identity, not just equal to) the
  canonical root class (pre-fix: two distinct class objects).
- **Incidental, not fixed:** converting `reference_monitor.py`'s dead
  `safety/` shim made mypy reach the canonical `orchestrator/
  reference_monitor.py` through this invocation for the first time,
  surfacing 8 pre-existing type errors in that live file (untyped `dict`
  literals, a `dict[Any, Any]` vs `EscalationRule` mismatch) plus 3 in
  `specs.py`. These are real, pre-existing debt in a live file — not
  introduced by this fix, and out of scope for a duplicate-shim conversion.
  `[REQUIRES HUMAN REVIEW]` / a candidate for whichever future tier covers
  `reference_monitor.py`/`specs.py` directly.

### C2 — VERIFIED DEFECT — `plugins/discovery.py::load_plugin()` built the wrong import path for bundled plugins

- **Property violated:** a path-construction bug that silently swallows the
  entire feature it's part of — "return value indistinguishable from
  success" via a caught, logged, and discarded `ImportError`.
- **Location:** `plugins/discovery.py:228` (pre-fix).
- **Finding:** `_bundled_plugin_path()` (line 62) correctly looks under
  `<repo>/orchestrator/plugins/<kind>/` (plural package), but `load_plugin()`
  built the module name to import as
  `f"orchestrator.plugin.plugins.{kind}.{name}"` — singular `plugin.plugins`,
  which is not a package at all (it's the plain module
  `orchestrator/plugin/plugins.py`), so it can never have a `<kind>.<name>`
  submodule. Any bundled plugin that existed would be *discovered*
  (its manifest read successfully) but then fail
  `importlib.import_module()`, silently swallowed by `load_plugin()`'s
  surrounding `except Exception: logger.error(...); return None`.
- **Reachability:** dormant today — no `orchestrator/plugins/<kind>/`
  directories exist yet (confirmed via `find`), so no bundled plugin has
  ever hit this path. The bug is real and would silently break the first
  bundled plugin anyone adds.
- **Innocence attempt:** none needed beyond reachability — this is a plain
  string-construction mistake with no ambiguity about intended behavior
  (the sibling `_bundled_plugin_path()` in the same file already states the
  correct path one function away).
- **Fix:** module name now built as `f"orchestrator.plugins.{kind}.{name}"`.
- **Test:** `test_c2_bundled_plugin_import_path_matches_plugins_package` —
  patches `importlib.import_module` to capture the exact module name
  `load_plugin()` requests, asserts it matches the plural `orchestrator.plugins.*`
  path (pre-fix: captured the singular, unimportable path).

### C3 — VERIFIED DEFECT — `safety/generated_output_scanner.py::scan_output_dir()` silently skipped unreadable files with no count or log

- **Property violated:** the same "silent wrong result" shape as T8's C5,
  now confirmed in a third, independently-maintained secret/insecure-pattern
  scanner (T6 flagged `website_validator.py`, T8 fixed it; `quality_control.py`
  x2 remain dead/unfixed; this is the fourth known instance of the pattern
  and the second live one).
- **Location:** `scan_output_dir()`'s per-file loop.
- **Finding:** `except OSError: continue` with no logging and no counter —
  a file that failed to read was indistinguishable from a file that was
  never scanned because it didn't match the scan's file-type filter.
- **Reachability:** LIVE — wired via `output_organizer.py::_security_scan()`,
  itself called from `OutputOrganizer.organize_project()`'s standard
  pipeline (`security_scan=True` by default).
- **Innocence attempt:** "the docstring already documents this as
  intentional — 'the gate must not crash delivery'." This is true and
  correct for the *pass/fail* behavior (unlike T8's C5, this tier does
  **not** flip the check to fail on a skip — that would contradict a
  deliberate, already-documented design choice). But "must not crash" and
  "must not be silently invisible" are different properties; the innocence
  attempt only covers the former.
- **Fix:** added a `files_skipped` counter to `ScanReport` (surfaced in
  `to_dict()`), a warning log naming the file and exception on each skip,
  and a caller-side warning in `output_organizer.py::_security_scan()` when
  `files_skipped > 0` — all additive, the existing "never block delivery"
  contract is unchanged.
- **Test:** `test_c3_scan_output_dir_counts_and_logs_unreadable_files` — a
  real trigger: monkeypatches `Path.read_text` to raise for one specific
  file only (a directory-named-as-file trick doesn't work here, since
  `_iter_scannable_files()` filters to `path.is_file()` before the read —
  confirmed by first writing the more naive version of this test and
  watching it fail to even reach the code path), asserts `files_skipped == 1`
  and a warning names the file (pre-fix: `AttributeError`, the field didn't
  exist yet).

### C4 — VERIFIED DEFECT (documentation) — `safety/tool_guardrails.py` falsely claimed to be wired into `engine.py`

- **Property violated:** a docstring asserting integration that doesn't
  exist is worse than no docstring — it tells the next reader a safety
  control is active when it isn't.
- **Location:** module docstring, "Integration: Called from
  engine.py._execute_task() BEFORE deterministic validators."
- **Finding:** verified false by repo-wide grep — `.check(`,
  `ToolCallGuardrailController`, `GuardrailDecision` appear nowhere in
  `engine.py` or any executor. `engine_core/service_collection.py`
  constructs a `ToolCallGuardrailController()` instance into a dataclass
  field that is never subsequently read anywhere.
  `engine_core/orchestration_facade.py`'s `getattr(orch, "_tool_guardrails",
  None)` always resolves to `None` since `_tool_guardrails` is never set on
  `Orchestrator`.
- **Reachability:** the *documentation claim* is what's live (read by any
  developer who opens this file) even though the *guardrail* is dead — the
  same "false CLAUDE.md invariant" shape T0 already fixed once for this
  hunt's own planning doc.
- **Fix:** corrected the docstring to state plainly that this is not
  currently wired in, and how to wire it if that's ever decided
  (no code/behavior change — this file's runtime logic is untouched).
- **No test:** a docstring correction has no executable behavior to assert
  against; consistent with how T0's CLAUDE.md/`stress_test.py` citation fix
  needed no pytest test.
- **Not fixed here — `[REQUIRES HUMAN REVIEW]`:** actually wiring
  `ToolCallGuardrailController` into the execution path (this is exactly
  the kind of guard meant to block `rm -rf`/`DROP TABLE`/`os.system` in
  LLM-generated code before it runs — architecturally significant, and
  wiring it is a decision for a human, not a doc-fix-scope task).

## Candidates surveyed, not fixed this tier — `[REQUIRES HUMAN REVIEW]` / `[UNK]`

Recorded so a later tier does not need to re-survey these:

- **`safety/command_guard.py::classify_command()`** — a 4-tier shell-command
  risk classifier ("GAP-13") with zero production callers; the one live
  path that runs a caller-supplied shell command
  (`assembler.py::_run_verify()` → `secure_execution.SecureSubprocess`)
  already forces `shell=False` independently, so this isn't a raw bypass,
  just an unplugged second layer.
- **`orchestrator/plugin/` (singular) subpackage** — the entire isolation/
  sandboxing subsystem (`plugin_isolation.py::IsolatedPluginRuntime`/
  `SecurePluginRegistry`, `plugin/plugins.py::PluginManager`) is dead (zero
  live callers, confirmed exhaustively). If ever wired in:
  `IsolatedPluginRuntime._requires_isolation()` trusts a plugin's own
  self-reported `metadata.author == "orchestrator"` or a `"verified:"`
  prefix with no cryptographic binding — any plugin could self-report its
  way out of sandboxing. Not fixed — building trust verification into dead
  code nobody asked for is scope creep, not a bug fix.
- **`safety/guardrails.py::ProductionGuardrails`/`KillSwitch`** — a full
  589-line budget/kill-switch/error-rate/drift safety system with zero
  callers and zero tests anywhere. More orphaned than T1's already-flagged
  `cost_optimization/` package. Architecturally interesting, not a live
  defect.
- **`gateway/run.py::OrchestratorGateway`** — a real, CLI-reachable feature
  (`orchestrator gateway start`) whose `handle_message()` spends a real
  `Budget` with no auth/rate-limiting on `user_id`. Not currently
  exploitable: its only two platform adapters (`echo`, `webhook`) never
  actually bind a network listener, so `handle_message()` has zero live
  callers today. A landmine for whoever finishes the webhook listener, not
  a live bug.
- **`AgentSafetyMonitor.can_interact()`/`.report_event()`** — same
  dead-service-collection-field shape as C4's `ToolCallGuardrailController`,
  lower severity (a monitoring/quarantine layer, not a hard block).
- **`safety/sandbox.py`, `safety/secure_execution.py`** (T7's originally-named
  carry-over) — confirmed **not** independent implementations: both are
  clean re-export shims to `orchestrator/sandbox.py`/`orchestrator/
  secure_execution.py` respectively (11 lines each, `import *` only, no
  divergence possible). `secure_execution.py`'s `SecureSubprocess` is
  genuinely live (via `assembler.py`, `commands/analyze.py`,
  `commands/agent.py`) and genuinely safe (`shell=False` hardcoded
  regardless of validation outcome — defense-in-depth, not the only
  barrier). This resolves T7's own open question: nothing about these two
  names needed fixing; the naming collision the plan flagged as
  "most promising" turned out to be innocent shims, not divergent copies.
- **`hierarchy.py`** — the node-ID-collision bug a failure-archaeology pass
  would flag is already fixed (a monotonic counter, with a dated comment);
  zero live production callers today regardless.
- **Confirmed clean, no further action:** `safety/red_team.py` vs. live
  root `red_team.py` (diff shows only the correct-for-each-location import
  line differs, no behavioral divergence); `safety/accountability.py`
  (canonical/live) vs. root `accountability.py` (shim, correctly wired,
  reverse of the usual direction); `safety/secure_execution.py`,
  `safety/input_validation.py`, `safety/security_templates.py`,
  `safety/dependency_scanner.py`, `security/enhancer.py`,
  `security/indesign_plugin_rules.py`, `security/ios_hig_prompts.py`,
  `security/wordpress_plugin_rules.py`, `plugins/context_provider.py`,
  `plugins/cost_optimization.py`, `plugins/hallmark_design.py`,
  `plugins/nash_stability.py`, `gateway/session.py` — all dead code with no
  divergence or security-relevant defect found. No hardcoded credentials,
  no non-constant-time secret comparisons, and no live `eval`/`exec`/
  `os.system` reachable from untrusted input were found anywhere in this
  48-file scope.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
