# T17 — Duplicate-pair convergence sweep — Inventory

First wave of `docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` (T17–T24), continuing the
AUTONOMOUS DEFECT-HUNT PROTOCOL V7. First wave of the programme whose Phase-1
surface is an **exhaustively enumerated pattern instance set** rather than a
sampled file list.

## Phase 0 — Scope

**In scope:** every subpackage module under `orchestrator/` sharing a filename
with a root-level module where *both* sides carry their own class/function
definitions. Enumerated by AST (a module body with zero `ClassDef`/
`FunctionDef`/`AsyncFunctionDef` is a re-export shim and therefore cannot
diverge).

**Measured at `56276c7`:** 188 same-name pairs → 26 already shimmed on the
subpackage side, 97 already shimmed on the root side (inverted but resolved),
**65 with definitions on both sides** — the candidate set. Re-verified at
execution time per the plan's Phase-0 requirement; count unchanged.

**Threat model:** a fix or hardening lands on one copy and never reaches the
other, and whichever copy a caller happens to import decides whether the bug is
live. This shape produced confirmed defects in T5 (circuit-breaker HALF_OPEN),
T13 (duplicated false-clean security scan), T14 (un-backported SQL-injection
hardening) and T16 (four `operations/` pairs).

**Budget:** all 65 triaged; convergence limited to the subset where it is
provably behaviour-preserving.

## Phase 1 — Surface map (exact, not sampled)

Four mechanical signals were computed for all 65 pairs before any diff was read
for suspicion:

| Signal | Method | Result |
|---|---|---|
| Divergence size | `diff` line count | 15 pairs share **no** definitions (name collision only); 31 near-identical (diff ≤ 20); 19 diverged (diff > 20) |
| Live side | import-statement scan across the repo | most pairs have importers on the root side only |
| **Package exposure** | does the package `__init__.py` re-export the submodule | **27 of 65** are reachable as `from orchestrator.<pkg> import <name>` |
| Importability | `python -c "import <module>"` in a subprocess, both sides | 62/65 subpackage copies import; **3 fail** |

The package-exposure signal is what converts a "dead duplicate" into a live
one — the same mechanism T16 recorded for `services/`. It is the reason this
wave's severity assessment is not simply "importer count > 0".

## Phase 2–4 — Candidates and verdicts

### C1 — `integrations/swiftstack_integration.py` — **VERIFIED DEFECT, FIXED**

A **byte-identical** (diff = 0) copy of `orchestrator/swiftstack_integration.py`
sitting one package deeper. Because it is byte-identical it carries root's
root-relative imports, which resolve against `orchestrator.integrations.*` from
its location:

```
ModuleNotFoundError: No module named 'orchestrator.integrations.api_builder'
    at orchestrator/integrations/swiftstack_integration.py:40
```

**Trigger:** `import orchestrator.integrations.swiftstack_integration` → raises.
FIRED. **Innocence:** no guard, no alternate resolution — the module genuinely
cannot load; `integrations/__init__.py` does not import it, so nothing masks it.
NO-DEFENSE-FOUND. Severity **LOW** (nothing imports it), but it is a real,
executable defect: a duplicate that could not even load. T15 recorded this file
as an "unshimmed dead duplicate"; this wave establishes it was additionally
**broken**. Fixed by converging to a shim, which also makes it importable.

### C2 — root `website_generator.py` unimportable — **VERIFIED DEFECT, NOT FIXED**

`orchestrator/website_generator.py:17` does `from .component_registry import
get_registry`, but `component_registry.py` lives at `orchestrator/design/`, not
root. The module therefore fails to import, and so does its only importer,
`orchestrator/cli_website.py`.

**Trigger:** `import orchestrator.website_generator` → `ModuleNotFoundError:
No module named 'orchestrator.component_registry'`. FIRED.
**Innocence:** none — the path is simply wrong.

**Not fixed, deliberately.** Correcting the path to `.design.component_registry`
would move the failure rather than remove it: that module raises `ImportError`
on `from .design_system import ComponentSource`, and **`ComponentSource` is
referenced 17 times across the design subsystem but defined nowhere in the
repository**. A path-only change would produce a module that imports and then
fails at construction — masking the symptom rather than breaking the mechanism,
which V7 Phase 5 forbids. The live `generators/website_generator.py` already
handles this deliberately with a guarded lazy import and a `_FakeComponent`
fallback (the gap T13 recorded). Root cause stays `[REQUIRES HUMAN REVIEW]`,
consistent with T13's disposition of the same subsystem. Severity **LOW** —
both the module and its only importer are unreachable.

### C3 — four byte-identical duplicate pairs — **hygiene, FIXED**

`design/component_library.py` (879 lines), `design/frontend_security.py` (1058),
`security/indesign_plugin_rules.py` (1131), `security/ios_hig_prompts.py` (474)
were each byte-identical to their root counterpart, and the two `design/` ones
are wildcard-exposed through `design/__init__.py`. No behavioural divergence
exists **today** — by construction, since the bytes match — so these are not
defects; they are the precondition for one. Converged to shims: provably
behaviour-preserving (verified name-superset before the change, object identity
after).

### C4 — `design/design_system.py` stale fork — **VERIFIED DEFECT (latent), FIXED**

A 155-line fork of the 310-line canonical `orchestrator/design_system.py`,
**wildcard-exposed** through `design/__init__.py`. The fork lacks the
`tone`/`font_heading`/`font_body`/`accessibility` fields and the `__post_init__`
that materialises `spacing`/`shadow`/`animation`/`border_radius` — the root copy
annotates these as *"Additional fields used by WebsiteGenerator"*, and
`website_generator.py` formats exactly those into its output
(`design_system.accessibility.min_contrast_ratio`, `.font_heading`,
`.border_radius.sm/md/lg/full`).

**Innocence attempt — partially successful, and it lowers the severity:** every
real consumer (`generators/website_generator.py`, `website_validator.py`,
`website_factory.py`, `cli_website.py`, `commands/website.py`, and all tests)
imports the **root** module. No live path reaches the fork today, so there is no
live trigger. It is a latent `AttributeError` landmine armed for the first
caller to write `from orchestrator.design import DesignSystem`. Severity
**LOW-MEDIUM**, fixed by convergence.

**Independent corroboration:** mypy's pre-fix run reported, unprompted, on
`design/component_registry.py`:

```
error: "DesignSystem" has no attribute "tone"  [attr-defined]
error: "DesignSystem" has no attribute "to_prompt_context"  [attr-defined]
```

— the same defect, found by a different tool. Both errors are gone post-fix.

### C5 — `design/design_to_code.py` stale model id — **FALSE (innocent)**

The fork's `VISION_MODELS` dict names `claude-sonnet-4.6`; root names
`claude-sonnet-5` (the id that exists in the `Model` enum;
`claude-sonnet-4.6` appears nowhere else in the repository). **Innocence:
`VISION_MODELS` is read by nothing** — grep finds only the two definitions.
An inert constant in a dead fork. Recorded so it is not re-raised.

### C6 — `ide_backend/log_config.py` — **UNKNOWN, routed to T21**

A 35-line module defining its own `configure_logging()` and a `get_logger()`
returning `logging.getLogger(f"ide_backend.{name}")` — a **separate logger
hierarchy** from `orchestrator.*`, with no `SecretsFilter`. Five `ide_backend/`
modules import it. This is the **fifth** independent `configure_logging()` in
the repository and, like the other four, has zero callers.

**No executable trigger can be written in this environment**: importing it pulls
`ide_backend/__init__.py`, which requires `fastapi` (not installed). Per V7
Phase 3, a candidate with no executable trigger stays `[UNK]` and is **not**
promoted on reasoning alone. Routed to T21, whose stated prerequisite is
installing the `dashboard` extra. This strengthens the case for that
prerequisite.

### `integrations/gateway.py` — **excluded from convergence, deliberately**

Byte-identical to root `gateway.py` and name-safe, but `orchestrator/gateway/`
exists as a **package**, so a `from ..gateway import *` shim would resolve to the
package, not to root's `gateway.py` — a semantic change, not a convergence. T2
already recorded root `gateway.py` as permanently shadowed by that package.
Left as-is and kept in the gate baseline.

### Remaining 57 candidates — **triaged, not converged**

Overwhelmingly **import-depth-only** differences: the subpackage copy uses `..X`
where the root copy uses `.X`, both correct for their own location, frequently
with a leftover `# FIXED: <old line>` breadcrumb comment above the corrected
import. Cosmetically divergent, semantically identical. Converging them is
possible but each requires its own name-superset proof and caller check, and a
57-file shim conversion in one commit is precisely the over-broad change V7
Phase 6 vector 6 warns against. They are frozen in the gate baseline so they
cannot grow, and remain available to later waves.

## Phase 5–6 — Fix design and self-review

Six one-line-body shims, each `from ..<module> import *` with a docstring
recording what the fork was and why it was converged. Each was verified by
object identity against the root module after the change, and the C4 fix was
additionally verified by constructing `DesignSystem(tone="luxury")` through the
**package** path and reading the four attributes the fork lacked.

Self-review vector 6 (**new-defect introduction**) is the load-bearing one for a
convergence wave, and it drove two decisions: excluding `integrations/gateway.py`
(package shadowing would change semantics) and declining the other 57 (name-
superset unproven per-file). Vector 4 (**regression**) is covered by the full
suite plus the name-superset precondition, which was checked mechanically before
any file was rewritten.

## Deliverable — `scripts/check_duplicate_pairs.py`

Freezes the remaining **59** pairs. Fails on any new both-sides-define pair,
accepts a pair once either side becomes a shim, and supports `--list` and
`--update`. Self-tested three ways (detects an injected pair; accepts it once
shimmed; `--update` is idempotent and does not corrupt its own source).

Two defects were found in the gate script while building it, both by its own
tests: `re.sub` without `count=1` rewrote the marker literals inside the update
function and corrupted the file; and the failure listing printed to stdout while
its header printed to stderr, so CI could show the error without the offending
filenames. Both fixed.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3. `fix_revisions`: 0 on the six shims (all correct first
pass, RED→GREEN on first attempt); 2 on the gate script, both self-caught before
commit and both in tooling rather than in shipped orchestrator code.

One hypothesis was **falsified during Phase 3 and discarded before any fix was
written**: that converging `design/design_system.py` would repair
`design/component_registry.py`'s `ImportError`. It does not — `ComponentSource`
is absent from *both* copies and from the entire repository, so the registry
stays broken and stays `[REQUIRES HUMAN REVIEW]`.
