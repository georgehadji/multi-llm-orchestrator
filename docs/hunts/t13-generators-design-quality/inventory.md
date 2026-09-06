# T13 — generators/, appbuilder/, design/, scaffold/, output/, quality/ — Inventory

Fifth of waves T9-T16 per `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`, continuing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7 across backend files with no individual disposition
recorded in earlier tiers.

## Phase 0 — Scope

122 files by direct `find` count (the plan estimated ~145; trusted the direct count):
`generators/` (34), `appbuilder/` (5), `codebase/` (8) plus root-level `code_executor.py`/
`code_validator.py`/`code_post_processor.py`, `design/` (33), `scaffold/` (10), `output/` (3),
`quality/` (19). Framed by the plan itself as **lower architectural blast radius** than
T9-T12 — "bugs here affect generated-output quality, not orchestrator integrity/security."

## Phase 1-3 — Survey, candidates, trigger/innocence

A background agent surveyed all 122 files, prioritizing (a) `generators/website_validator.py`
(a T6-flagged, possibly-still-open false-clean secret scan) and (b) a systematic sweep of an
unusually large number of root-vs-subpackage duplicate-name pairs in this wave's file set.
Every finding below was independently re-verified from source before being applied or
reclassified — most significantly, Candidate 2 below (`design/component_registry.py`) was
**downgraded from the agent's "VERIFIED DEFECT — highest impact this wave" to
`[REQUIRES HUMAN REVIEW]`** after independent verification surfaced context the agent's
report did not carry: this is not a silently-broken subsystem nobody noticed, but a
**previously-known, explicitly documented, deliberate workaround** — see Candidate 2 for
detail. This is this hunt's fourth instance of a background agent's severity claim being
materially corrected during independent verification (after T11's C1 and, in spirit, T12's
Lead D correction of T10's own prior claim).

## Phase 0 lead — `generators/website_validator.py` verdict: already fixed

**(a) — confirmed the same file T8's C5 already fixed.** `generators/website_validator.py`'s
`_check_secret_exposure` (lines 845-861) already tracks an `unscanned` list, logs each
unreadable file, and folds the count into `passed`. Root `orchestrator/website_validator.py`
is a clean 20-line shim to this file (converted by a separate, non-hunt commit `c2de25d`
before T8 ran). No action needed; re-confirmed rather than re-fixed.

## Phase 4 — Fixes (VERIFIED DEFECT)

### C1 — `quality_control.py`'s security scan silently reported false-clean on unreadable files

**File:** `orchestrator/quality_control.py::TestRunner._run_security_checks` (lines 509-546).

The exact false-clean-scan pattern this hunt has now fixed three times (T6 flagged it, T8
fixed `generators/website_validator.py`, T9 fixed `safety/generated_output_scanner.py`) was
present here too, never touched: `except Exception: pass` silently skipped any unreadable
file with zero counting or logging, and `passed=len(issues) == 0` had no way to reflect a
scan that silently examined nothing.

**Reachability, verified independently (not just trusting the survey):** `TestLevel.SECURITY`
is never passed as an argument by either real caller of `run_quality_gate`
(`project_analyzer.py:35`, `project_mgmt/analyzer.py:35` — both hardcode
`levels=[TestLevel.UNIT]`), and grepping the whole repository for `TestLevel.SECURITY` finds
it only inside `quality_control.py` itself (the enum definition and internal dispatch) —
confirmed dead today, in both copies (see C2).

**Fix:** narrowed `except Exception: pass` to `except OSError:`, added an `unreadable` counter
with a `logger.warning(...)` per skipped file, and folded a summary line into `issues` (so
`passed` becomes `False` whenever anything was skipped) — the identical fix shape used in T8
and T9, applied here for the third time.

### C2 — `orchestrator/quality/quality_control.py` was an unshimmed duplicate carrying the same bug

**File:** `orchestrator/quality/quality_control.py` (converted to a shim).

Diffed in full against the canonical, live `orchestrator/quality_control.py`: the only
differences were import-depth-adjustment comments (`# FIXED: from ..log_config import ...`)
— the logic, including C1's bug, was byte-for-byte identical. Confirmed zero live importers
of `orchestrator.quality.quality_control` anywhere (the one grep hit for `from .quality_control
import` resolves to the *root* file, since `project_analyzer.py` lives at package root, not
inside `quality/`). `quality/__init__.py` does not import this module either (plain docstring,
no re-exports), ruling out any indirect live path.

**Fix:** converted into a `from ..quality_control import *` shim, matching this hunt's
established convention (plain relative import, no `DeprecationWarning`, a docstring naming
the canonical file and the specific bug the sync closes). Verified no circular import both by
tracing (root `quality_control.py`'s own imports — `log_config`, `monitoring`, `performance`
— never reach back into `quality/`) and empirically (`python3 -c "import
orchestrator.quality.quality_control"` succeeds, `QualityController` identity-matches the
canonical class).

## Phase 4 — Residual, surveyed but not fixed

### `design/component_registry.py`'s broken import — `[REQUIRES HUMAN REVIEW]`, reclassified from the survey's "VERIFIED DEFECT — highest impact"

`design/component_registry.py:15-19` imports `ComponentSource`/`ComponentSpec` from
`.design_system` — **these two classes have never existed anywhere in this repository's
history** (confirmed via `git log --all -S"class ComponentSpec"` / `-S"class
ComponentSource"` across the whole repo, zero hits, and via direct inspection of every commit
that ever touched `design/design_system.py` — exactly one, and it never added them). The
survey correctly identified this as broken and traced its consequence:
`generators/website_generator.py`'s `_get_registry()` falls back to a `_FakeRegistry`
returning 4 generic placeholder components (`hero`, `features`, `pricing`, `contact`) on
every generation run, meaning the curated `COMPONENT_LIBRARY` (dozens of
Aceternity/shadcn/21st.dev/Magic-UI-sourced specs) has never been used in any generated
website.

**What changes the classification:** reading `website_generator.py` directly (lines 18-53)
shows this is not an accidental, silently-discovered breakage — it is a **previously known,
explicitly documented, deliberate workaround**:
```python
# FIXED: from .component_registry import get_registry
# Lazy import — component_registry has broken dependencies
get_registry = None
```
A prior developer already identified this exact defect, already chose graceful degradation
over completing the dependency, and left a comment saying so. There is no live crash, no
silent-wrong-result in the sense this hunt weights highest (the fallback is honest — generic
components, not corrupted ones) — the cost is entirely in generated-output *richness*, not
correctness or safety.

**Why this is not fixed here:** `ComponentSpec`/`ComponentSource` were never completed, not
merely misplaced. `component_registry.py` uses `ComponentSpec` with a consistent 12-field
shape and calls `candidate.compatibility_score(design_system)` — a method that does not exist
anywhere either. Writing these would mean **designing a compatibility-scoring algorithm from
scratch** (what makes a component "compatible" with a design system — by what criteria, what
weights) — a genuine product/design decision, not a mechanical causal fix. This is squarely
the class of finding this hunt has consistently deferred (T9's plugin isolation, T11's
`UnifiedEventBus.start()`, T12's HITL wiring): a real gap, cheap to observe, expensive and
judgment-laden to close. Flagged for a human to decide whether completing the curated
component library is worth building, or whether the existing fallback is intentionally
sufficient (the inline comment suggests the latter was already the working assumption).

### `orchestrator/output/organizer.py` vs live `orchestrator/output_organizer.py` — diverged, dead

The `output/` subpackage copy lacks two entire pipeline steps present in the live root file:
`_format_code()` and `_security_scan()` (the latter calling T9's C3-hardened
`safety/generated_output_scanner.py`). Confirmed zero live importers of
`orchestrator.output.organizer` — dead today, but a landmine (`output/__init__.py` exists
specifically to house this and reads as the intended eventual import path). Not converted to
a shim: unlike C2, the two files are not near-identical with one small drift — the subpackage
copy is missing entire features, and this hunt's shim-conversion pattern is for confirmed
divergence-of-a-fix, not for filling in a partially-built parallel implementation. Left as a
documented residual rather than force-fit into a shim that would silently change dead code's
apparent feature set.

### `docker_generator.py` divergence — security-adjacent, both copies dead

`generators/docker_generator.py` has non-root-user container hardening and required
(not hardcoded) DB credentials; root `docker_generator.py` hardcodes default database
passwords (`POSTGRES_PASSWORD=postgres`, `MYSQL_ROOT_PASSWORD=root`) directly into generated
`docker-compose.yml` and lacks the hardening. Confirmed `DockerfileBuilder` has zero live
importers in either copy — dead today, but worth recording precisely because it is exactly
the shape of regression T10's live `docker_sandbox.py` path-traversal fix exists to prevent:
if this generator is ever wired up via the root path instead of the hardened
`generators/` path, real projects would generate with weak default credentials. Not fixed
(dead code, no live path), but flagged more prominently than a typical dead-code note given
the security content.

### `design/frontend_security.py` — complete, unused CSP/CSRF generation library

1058 lines of CSP-meta-tag and CSRF-component generation exist, byte-identical in both the
`design/` and root copies, with zero callers of `generate_csp_meta_tag`/`create_csrf_component`
anywhere (grepped). Generated websites currently receive no CSP headers or CSRF protection
from this code path, though it is fully written and ready. Matches this hunt's Pattern 3
(fully-built, never wired) exactly — recorded, not fixed (wiring it in is a product decision:
should every generated site get CSP/CSRF by default, and if so, wired from where).

### Dead legacy import chain — `website_generator.py` (root) → nonexistent `component_registry.py` (root) → `cli_website.py`

Distinct from the `design/component_registry.py` finding above: root
`orchestrator/website_generator.py:17` does `from .component_registry import get_registry` —
`orchestrator/component_registry.py` has never existed at root either. This cascades:
`orchestrator/cli_website.py` imports from the broken `website_generator.py` and is therefore
also fully unimportable (confirmed empirically). Confirmed nothing in the repository imports
`orchestrator.cli_website` except a stale docstring reference — the real, live `website` CLI
subcommand is `orchestrator/commands/website.py`, which correctly uses
`generators/website_generator.py` (a different file) and never touches either broken one.
`[REQUIRES HUMAN REVIEW]`: delete-vs-repair of confirmed-dead legacy code is a scope decision,
not investigated further.

### `orchestrator.CodebaseAnalyzer` vs `orchestrator.analyzer.CodebaseAnalyzer` — naming footgun, `[UNK]`

Two independent classes share an identical public name at two different import paths
(`codebase/analyzer.py`, used by `codebase/understanding.py`; and `analysis/analyzer.py` via
a root shim, used by the real `--analyze-codebase` CLI command). Docstrings suggest genuinely
different scopes (structure-scan-only vs. full LLM analysis pipeline) — not confirmed as an
active mix-up anywhere, flagged as a readability/IDE-autocomplete risk rather than an
asserted defect.

## Phase 4 — Cleared (innocent)

Confirmed clean shims (correct, zero divergence): `database_generator.py`,
`fullstack_generator.py`, `secrets_manager.py`, `secrets_generator.py` (T2's fix, reconfirmed
intact), `frontend_rules.py`, `responsive_layouts.py`, `validators.py`, `verification.py`,
`cicd_generator.py`, `copy_generator.py`, `image_optimizer.py`, `logging_generator.py`,
`multi_platform_generator.py`, `opengraph_generator.py`, `testing_templates.py`,
`adaptive_templates.py`, `benchmark_suite.py`, `browser_testing.py`, `code_post_processor.py`,
`code_validator.py`, `appbuilder/verifier.py` (T7's fix, reconfirmed intact), and
`safety/code_executor.py` (T1's fix, reconfirmed intact).

Confirmed clean depth-adjusted duplicates (two independent files, differ only in import
dot-count, logic byte-identical — no divergence risk since neither has yet been fixed
independently of the other): `diff_generator.py`, `advanced_query_processing.py`,
`pre_submission_testing.py`, `tdd_config.py`, `appbuilder/detector.py`↔`app_detector.py`,
`appbuilder/store_assets.py`↔`app_store_assets.py`,
`appbuilder/store_validator.py`↔`app_store_validator.py`.

Confirmed genuinely different purposes despite name collision (not duplicates):
`image_generator.py` (local SVG placeholder vs. AI image generation), `website_template.py`
(the task's assumed root pairing doesn't exist — the real pair is a CLI-wrapper relationship
with `commands/website_template.py`), `prompt_builder.py` (design-generation constraints vs.
decompose/critique/revise templates), `quality/validators.py` vs `application/validators.py`,
`quality/verification.py` vs `domain/verification.py`, `quality/preflight.py`/root vs
`engine_core/stages/preflight.py`, the 3-way `assembler.py` collision (three genuinely
different tools), `codebase/decomposer.py` vs the three other same-named decomposers
(four genuinely different decomposition concepts), `quality/preflight.py` vs root (root has
an extra feature and is live; no bug, just an enrichment not yet synced backward — low
priority since the live side is the richer one), `design/design_to_code.py` vs root (one
stale model-name string, both fully dead — cosmetic).

`[UNK]`, not fully chased given this wave's own lower-priority framing: `design/atelier/themes.py`
vs `design/catalogs/themes.py`; `design/catalogs/routing.py` vs `routing/routing.py`;
`codebase/context.py` vs `project_mgmt/context.py`.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1-3 survey delegated to one background agent run).
`fix_revisions`: both fixes (C1, C2) correct on first pass, RED→GREEN verified. One
correction was made to the *survey's own severity classification* (component_registry.py,
downgraded from VERIFIED DEFECT to REQUIRES HUMAN REVIEW) during independent verification,
not a fix revision.
