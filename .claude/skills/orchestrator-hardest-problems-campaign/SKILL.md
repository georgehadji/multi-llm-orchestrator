---
name: orchestrator-hardest-problems-campaign
description: Execute one of the four hardest live problems in the Multi-LLM Orchestrator as a decision-gated campaign with exact commands, expected numbers, and promotion gates. Load this when the user says "work the hardest-problems campaign", "attack Track A/B/C/D", "fix the engine/container circular imports", "harden response-healing", "fix config drift permanently", or "raise the generated-output quality ceiling". Each track is independently executable — do not run all four in one sitting. This skill is EXECUTABLE (has numbered phases with copy-pasteable commands and pass/fail gates), not reference material — for background theory use llm-orchestration-reference, for the flag catalog use orchestrator-config-and-flags, for gate mechanics use orchestrator-change-control.
---

# Orchestrator Hardest-Problems Campaign

Four independent tracks, each gated by measured evidence, not vibes. Every claim below was
re-verified against the repo on **2026-07-08** on branch `feat/response-healing` — re-run the
verification commands yourself before trusting a stale number.

**Do not start a track without reading its Phase 0 baseline in full.** Two of the four premises
in the original problem brief turned out to be **wrong** when checked against the live repo
(Track A's "circular import" and Track B's "flag doesn't reach the payload" are both stale/false
as of this date) — this is exactly why Phase 0 exists. Trust the command output, not the last
session's memory note.

## How to use this skill

1. Pick ONE track (A/B/C/D). Read its Phase 0 baseline and run every command in it yourself.
2. Compare your output to the "EXPECTED" block. If it matches → proceed down the ranked menu.
   If it diverges → follow the "if you see X instead" branch.
3. Every fix lands through TDD (RED→GREEN) and the full gate suite — see
   [Cross-track promotion protocol](#cross-track-promotion-protocol) at the end. Do not skip this
   even for a "one-line" fix.
4. Record the outcome (even "investigated, no action needed") as a chronicle entry per
   `orchestrator-failure-archaeology` conventions — future sessions must not re-litigate this.

## When NOT to use this skill

- You want the *theory* behind why a design choice was made → `llm-orchestration-reference`.
- You need the flag catalog/env-var reference, not a fix campaign → `orchestrator-config-and-flags`.
- You hit a gate and don't understand why it exists or how to escalate → `orchestrator-change-control`.
- You're debugging a fresh, unrelated live incident, not one of these four tracks →
  `orchestrator-debugging-playbook`.
- You want the measurement tools themselves (drift checker, BOM checker, importtime) →
  `orchestrator-diagnostics-and-tooling`.
- You're about to write an external claim about the results ("we fixed X, now Y is Z% better") →
  `orchestrator-external-positioning` — it has the claim-discipline checklist.

---

## TRACK A — Engine/container circular imports + engine.py demolition

### Phase 0: Baseline (run this first, in full)

```bash
# 1. Current engine.py size
wc -l orchestrator/engine.py
# EXPECTED (2026-07-08): 1250 lines. (Session memory claims a prior 1867→post-remediation
# figure and a "<300 line" target — that specific numeric target could NOT be found anywhere
# in the repo docs (ARCHITECTURE_REMEDIATION_PLAN.md, ARCH-AUDIT-V2.md, CLAUDE.md). Treat
# "<300" as UNVERIFIED folklore. What IS verifiable and binding: CLAUDE.md Unbreakable Rule #1
# — "engine.py = Mediator. New logic → new service module, not engine.py." Measure success by
# the trend line (git log -p --follow -- orchestrator/engine.py commit-by-commit LOC) and by
# "did this PR add new logic to engine.py", not by chasing an unsourced number.

# 2. Count skip markers claiming circular-import blockage
grep -n "circular" tests/test_phase6_10_comprehensive.py
```

EXPECTED grep output — exactly **10** skip decorators, in two groups:
- 4× `@pytest.mark.skip(reason="Relies on container.py imports...")` — lines 498, 513, 526, 536
  (class `TestServiceContainer`)
- 6× `@pytest.mark.skip(reason="Importing Orchestrator from engine.py has circular imports")` —
  lines 819, 828, 837, 856, 866, 876

If your count differs from 10, the file has changed since this was written — re-derive the rest
of this track's numbers from your own grep, don't trust this document's specifics further.

### Phase 0.5: THE CRITICAL CHECK — is the cycle even real anymore?

Do not skip this. Run it exactly as written:

```bash
# Attempt WITHOUT an API key (reproduces what CI/a bare dev shell sees)
python -c "
from orchestrator.engine_core.container import ServiceContainer
from orchestrator.budget import Budget
c = ServiceContainer.build(budget=Budget(max_usd=10.0))
print('OK', c.client is not None)
"
```

EXPECTED (verified 2026-07-08): this raises
`orchestrator.infrastructure.llm_client.AuthenticationError: OpenRouter API key not found.`
— **not** an `ImportError` or `ImportError: cannot import name ... (most likely due to a
circular import)`. Same for `from orchestrator.engine import Orchestrator; Orchestrator()`.

Now retry with a dummy key:

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-test-dummy"
python -c "
from orchestrator.engine_core.container import ServiceContainer
from orchestrator.budget import Budget
c = ServiceContainer.build(budget=Budget(max_usd=10.0))
print('OK', c.client is not None, c.selector is not None, c.decomposer is not None, c.pipeline is not None, c.validator is not None, c.architect is not None, c.telemetry is not None)
from orchestrator.engine import Orchestrator
o = Orchestrator()
print('OK Orchestrator instantiated', type(o))
"
```

EXPECTED: both prints succeed — `OK True True True True True True True` then
`OK Orchestrator instantiated <class 'orchestrator.engine.Orchestrator'>`. **No ImportError at
any point.**

**Conclusion (verified, not inherited from the brief): there is no live circular import today.**
`engine.py` imports `ServiceContainer` lazily *inside* `Orchestrator.__init__` (line 432 —
`from .engine_core.container import ServiceContainer`), so a top-level `import orchestrator.engine`
never triggers the container's imports at all, and `container.py` has no top-level import of
`engine.py` or anything that transitively reaches it (`grep -rn "from ..engine import\|import engine$" orchestrator/engine_core/` → no matches). The 10 skip reasons describe a **stale
premise**. The real reason those specific tests fail in a bare environment is the eager
`AuthenticationError` in `UnifiedClient.__init__` (`orchestrator/infrastructure/llm_client.py:182`)
when `OPENROUTER_API_KEY` isn't set — an environment-setup problem, not an architecture problem.

If you re-run this on a later date and DO get an `ImportError` mentioning "circular import",
the premise has become true again (someone added a top-level cross-import) — in that case skip
to "If Phase 0.5 finds a real cycle" below instead of the near-free fix path.

### Phase 1 (the actual, near-free fix): un-skip with a mocked/dummy client

Ranked menu:

1. **(Do this first — cheapest, ~1 hour, zero architecture risk)** Fix the stale premise.
   For each of the 10 skipped tests: remove the `@pytest.mark.skip`, and either (a) set
   `monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")` in the test / a fixture, or
   (b) monkeypatch `orchestrator.infrastructure.llm_client.UnifiedClient` to a stub for tests
   that only assert wiring shape (`container.client is not None`) and don't need a real client.
   Un-skip **one test at a time**, run it, confirm GREEN, then move to the next — do not bulk
   un-skip and hope.
   ```bash
   pytest tests/test_phase6_10_comprehensive.py::TestServiceContainer::test_build_creates_all_services -v
   ```
   Gate: passes with no `ImportError` anywhere in the traceback of a *failure* (an
   `AuthenticationError` you forgot to mock is a fixture bug, not a circular-import finding).

2. **(Only if Phase 0.5 finds a real cycle)** Port extraction — precedent already exists:
   `VSSamplerPort` at `orchestrator/domain/ports.py:658` is a `Protocol` defined specifically to
   let `engine_core/stages/{generate,critique}.py` depend on an abstraction instead of importing
   `application.verbalized_sampling` directly (cited as `[VERIFIED]` in
   `ARCHITECTURE_REMEDIATION_PLAN.md:135,138`). Follow the same shape: define a narrow `Protocol`
   in `domain/ports.py`, have the concrete implementation registered into `ServiceContainer`,
   have the cycle's other end depend on the port, not the concrete class.

3. Late/lazy imports inside the composition root. Already the codebase's actual pattern —
   `engine.py:432` imports `ServiceContainer` lazily inside `__init__` for exactly this reason.
   If a genuine new cycle appears, moving the offending top-level import to function-local scope
   at the single point of use is a legitimate, already-precedented fix. Do not do this
   silently — leave a comment citing the cycle it breaks (see `engine.py:432`'s pattern of a
   `# noqa: F821` + docstring note as the house style for this).

4. Factory indirection (`Callable[[], T]` passed into the container instead of a concrete
   import) — reach for this only if 2 and 3 are structurally impossible, e.g. the two modules
   must both exist at import time before either can be fully constructed.

**Fenced off — do NOT do this:** add a new `ignore_imports` entry to `.importlinter` to "make
the contract pass." Read `.importlinter` first — Contract 3 (`application-services-no-engine`)
already has exactly 2 sanctioned `ignore_imports` entries (lines 60-66), each with a code comment
explaining *why* it's safe (TYPE_CHECKING-only coupling). A new exemption needs the same bar:
proven zero runtime coupling, documented inline, and — per `orchestrator-change-control` — this
class of change is architectural, route it through review, don't self-approve.

### Gate / promotion

- [ ] Skip count in `tests/test_phase6_10_comprehensive.py` is strictly lower than 10 after your
      change (`grep -c "circular" tests/test_phase6_10_comprehensive.py`).
- [ ] Every un-skipped test passes in isolation (`pytest tests/test_phase6_10_comprehensive.py -v`).
- [ ] `lint-imports` still passes with **zero new** `.importlinter` contracts or `ignore_imports`
      entries (or, if Phase 0.5 found a real cycle requiring one, it went through the
      architectural-change review path in `orchestrator-change-control`, not a solo edit).
      ```bash
      lint-imports
      ```
- [ ] `wc -l orchestrator/engine.py` did not grow. If your fix is a genuine extraction (menu
      item 2), it should shrink.
- [ ] Full suite still green: `pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --cov=orchestrator --cov-fail-under=7`

### Success metric

Skip-count for this file: **10 → 0**, each removal individually proven (not bulk-deleted), zero
new lint-imports exemptions, `engine.py` LOC flat-or-shrinking, full CI gate green.

---

## TRACK B — Response-healing hardening

### Phase 0: Baseline — trace the flag, verify current wiring state

**Re-verification note:** the task brief flags that `llm_client.py` was "substantially edited
this session for an unrelated instructor-import-laziness fix" — re-checked against the file as
it stands today (2026-07-08); the response-healing code described below is what's actually in
the file now.

```bash
grep -n "USE_RESPONSE_HEALING" orchestrator/config.py orchestrator/infrastructure/llm_client.py
```

EXPECTED:
- `orchestrator/config.py:179` — `USE_RESPONSE_HEALING: bool = False` (declared on
  `OpenRouterOptimizations`, default off like its 5 siblings).
- `orchestrator/config.py:191` — `USE_RESPONSE_HEALING=os.getenv("USE_RESPONSE_HEALING", "false").lower() == "true"` inside `.from_env()`.
- `orchestrator/infrastructure/llm_client.py:493` — `def _maybe_add_response_healing(request_params: dict, opts) -> None:` — a **module-level, directly testable function**, not buried in a method.
- Logic (read lines 493-523 yourself): only adds the plugin when
  `getattr(opts, "USE_RESPONSE_HEALING", False)` is true, the request has a
  `response_format` (i.e. structured output), and it's non-streaming. Appends
  `{"id": "response-healing"}` to `extra_body["plugins"]` — routed through `extra_body` because
  the OpenAI SDK used for OpenRouter calls does not accept a top-level `plugins` kwarg.

### Phase 1 status: ALREADY DONE — verify, don't redo

```bash
pytest tests/unit/test_response_healing.py -v
```

EXPECTED: **5 passing unit tests** — `test_added_via_extra_body_for_structured_nonstreaming_request`,
`test_not_added_when_flag_disabled`, `test_not_added_without_response_format`,
`test_not_added_for_streaming`, `test_safe_when_opts_none`. These directly assert
`params["extra_body"]["plugins"] == [{"id": "response-healing"}]` when enabled and
`"extra_body" not in params` in every disabled/ineligible case.

**This means "prove the flag reaches the request payload" (the brief's Phase 1) is already
proven, with tests, today.** Don't re-derive it — cite these 5 tests and move straight to Phase 2.
If you re-run this and any of the 5 fail, something regressed; that's a bug fix via TDD, not this
track's real work.

### Phase 2: Measure healing hit-rate — the actual gap

```bash
grep -n "TODO" orchestrator/infrastructure/telemetry.py
```

EXPECTED: `orchestrator/infrastructure/telemetry.py:167` — `# TODO: Implement proper tracking`.
Read the surrounding function (lines ~150-180) yourself to see what's stubbed before you build on
top of it — do not assume the stub's shape matches what you need.

**Gate before claiming any improvement: add counting first.** Concretely:
1. Instrument `_maybe_add_response_healing` call sites (or the response path that receives the
   healed JSON) to emit a telemetry event: `{healing_enabled, healing_applied, parse_succeeded}`
   per structured-output call. Land this as its own small PR with its own unit test asserting the
   counter increments — this is infrastructure, not evaluation.
2. Only once that counter exists and is unit-tested does "hit rate" mean anything. A hit rate
   computed by eyeballing logs is not a number — it does not satisfy this track's success
   criterion.

**Fenced off:** flipping `USE_RESPONSE_HEALING` to `True` globally (in `config.py`'s default, or
in a `.env` shipped to users) before you have a measured number from step 2/3. That is exactly
the "enabling the flag globally without a measured number" the brief calls out — and it also
silently changes cost/latency (an extra OpenRouter-side repair pass) for every structured-output
call, which is a cost-model change requiring the same rigor as anything in
`llm-orchestration-reference`'s dual-budget section.

### Phase 3: Adversarial corpus, N≥50

Build (or find, if one already exists — search first: `grep -rln "malformed" tests/ | head`)
a corpus of ≥50 deliberately malformed JSON structured-output responses (truncated, trailing
commas, unescaped quotes, wrong types, extra prose wrapping the JSON, etc. — cover the failure
modes OpenRouter's response-healing plugin is documented to repair). For each:
1. Feed it through the **existing** local repair layers first (json5 relaxed-parse,
   partial-repair helpers — locate them: `grep -rln "json5\|partial.*repair\|repair.*json" orchestrator/ | grep -v test`).
2. Separately, with `USE_RESPONSE_HEALING=true` against a live/sandboxed OpenRouter call, measure
   how many of the 50 the server-side plugin repairs that the local layers didn't.
3. Report: `local_repair_rate = X/50`, `healing_additional_repair_rate = Y/50`,
   `still_failing_rate = Z/50`. This triage answers "is response-healing pulling its weight over
   what we already do for free locally" — the actual question this track exists to answer.

### Gate / promotion

- [ ] Telemetry counter for healing applied/succeeded exists, is unit-tested, lands as its own
      TDD'd change.
- [ ] N≥50 adversarial corpus exists under `tests/` (fixtures) with a documented measured
      local-vs-healing repair-rate split.
- [ ] Any change to `USE_RESPONSE_HEALING`'s default is accompanied by the measured numbers in
      the PR description, not asserted from vibes.
- [ ] Full gate suite green (see [promotion protocol](#cross-track-promotion-protocol)).

### Success metric

A written, reproducible rate (`local_repair_rate`, `healing_additional_repair_rate`), not prose.
Default flag state changes only if the additional-repair rate justifies the added latency/cost —
state that tradeoff explicitly in the change.

---

## TRACK C — Config/routing drift permanent fix

### Phase 0: Baseline — run the checker yourself, today

```bash
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py
```

EXPECTED (verified 2026-07-08 — numbers WILL drift, re-run before trusting this):
```
Checked 133 cost keys, 9 routing keys, 51 fallback pairs against 130 Model values and 9 TaskType values.

HARD DRIFT — 5 config entr(ies) silently dropped by models.py:
  costs.json key not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4.1'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-sonnet-4'
  fallbacks.json['qwen/qwen3-coder'] value not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'

INFO: 1 Model value(s) have no costs.json entry:
  internal/nano-banana-2
```

This confirms the brief's claim of "5 hard-drift entries including qwen/qwen3.6-flash and bare
anthropic ids" — the 3 `anthropic/claude-{opus-4,opus-4.1,sonnet-4}` entries are exactly the
"bare ids" (unversioned/pre-normalization forms) referenced. If your run shows a different count,
that's the current truth — use it, don't reconcile against this document.

Do **not** hand-fix these 5 entries as your task-complete criterion — that treats the symptom.
This track's success criterion is a mechanism that makes this class of drift structurally
impossible to reintroduce, not a one-time cleanup.

### Solution menu, ranked

1. **CI drift gate** — add a step to `.github/workflows/ci.yml` that runs
   `check_config_drift.py` and fails the build on any `HARD DRIFT` line.
   **This is a workflow edit, which is an architectural-class change per
   `orchestrator-change-control`** — route it through that skill's review process, don't
   self-merge. Design it to fail closed: a script exit code, not a grep on stdout text
   (`check_config_drift.py` should `sys.exit(1)` on hard drift — verify this yourself; if it
   doesn't currently exit nonzero, that's a prerequisite bug to fix first, since a CI step that
   always exits 0 is worse than no gate — it's a false sense of safety).
2. **Loader that hard-fails on orphan keys** — make the config loader (wherever `costs.json` /
   `fallbacks.json` / `routing.json` are parsed into runtime structures) raise at import/startup
   time if a key doesn't resolve to a `Model` or `TaskType` enum value, instead of silently
   dropping it. This catches drift the moment it's introduced, not just in CI — but it's a
   behavior change (a previously-silent drop becomes a hard crash), so it needs its own test
   coverage proving legitimate configs still load, and should ship together with item 1's
   corpus of currently-known-bad entries either fixed or explicitly allowlisted first (you cannot
   ship a hard-fail loader while 5 known-bad entries still exist — fix those 5 in the same change
   or the loader will crash the app on startup).
3. **Generate JSON from the Model enum directly** — instead of hand-maintained parallel JSON and
   enum, generate (or partially generate) `costs.json`'s key set from `orchestrator/models.py`'s
   `Model` enum at build/lint time, so drift is structurally impossible rather than caught after
   the fact. Highest effort, strongest guarantee — consider this only after 1 and 2 are in place
   and you've observed drift recurring despite them.

### Fenced off — do NOT do this

- Hand-editing the JSON files without rerunning `check_config_drift.py` immediately after. This
  is literally how the current 5 entries got introduced.
- Building "alias-resolution magic" in the config loader that silently maps
  `anthropic/claude-opus-4` → some canonical enum value to make the drift checker quiet. The
  **one legitimate precedent** for this class of thing is OpenRouter's own server-side id
  normalization (hyphenated anthropic ids like `claude-opus-4-6` resolve to `4.6` on OpenRouter's
  side transparently — documented in this repo's memory as "not dead despite absent from
  /models") — that is normalization happening at a boundary you don't control and must tolerate.
  It is not license to build your own alias-resolution layer that masks a config/enum mismatch
  you do control. If a JSON entry doesn't match the enum, the fix is to fix the JSON or the enum
  — not to teach the loader to paper over the mismatch.

### Gate / promotion

- [ ] `check_config_drift.py` run today shows 0 `HARD DRIFT` lines.
- [ ] A CI step now runs this checker and fails the build on future drift (verify by
      intentionally introducing a bad key locally, confirming the CI step would catch it, then
      reverting — do not commit the intentional break).
- [ ] Zero orphan keys in both directions (JSON→enum and, per the INFO line, enum→JSON — decide
      and document whether `internal/nano-banana-2` needing no cost entry is intentional or a
      gap; don't leave it unaddressed silently).
- [ ] Workflow-file change went through the architectural review path.

### Success metric

CI gate red on an intentionally-reintroduced drift entry (proven, then reverted); 0 orphan keys
both directions on `master` after merge.

---

## TRACK D — Generated-output quality ceiling

### Phase 0: Baseline — locate the actual gates

```bash
grep -n "has_blocking_findings" orchestrator/safety/generated_output_scanner.py
```
EXPECTED: `has_blocking_findings` is a computed property (line 96) on the scan report. But check
how it's **consumed**:
```bash
grep -n "blocking\|self.report.errors.append" orchestrator/output_organizer.py
```
Read `orchestrator/output_organizer.py` lines ~236-259 yourself. Confirmed today: the security
scanner's CRITICAL/HIGH findings are logged as warnings and appended to
`self.report.errors`, but the method is wrapped so a scan failure or finding **never raises or
halts delivery** — the comment at line 232 (`# never block delivery on formatting`) and the
structurally identical pattern for the security scan confirm this is **advisory, not blocking**,
exactly as the brief states. If you need this to actually gate delivery, that's a deliberate
behavior change — go through `orchestrator-change-control`, and decide explicitly what
"blocking" should mean (reject the whole delivery? downgrade quality score? require HITL
approval?) before implementing.

```bash
pytest tests/unit/quality/test_design_quality_validator.py -v
```
EXPECTED: the whole module is skipped —
`pytestmark = pytest.mark.skip(reason="validate_design_quality removed during refactoring — re-implement when needed")`
(file header, lines 1-13). Confirmed: `grep -rln "validate_design_quality" orchestrator/` returns
**zero files** — the function genuinely no longer exists anywhere in `orchestrator/`, only in
this orphaned test file. This is a near-free RED→GREEN opportunity: the tests already encode the
expected contract (`result.passed`, `result.validator_name == "design_quality"`,
`"critical" in result.details.lower()` for banned fonts, a stamp-presence check) — re-implement
`validate_design_quality` in `orchestrator/quality/validators.py` (or wherever the sibling
validators live — check `grep -rln "class.*Validator\|def validate_" orchestrator/quality/`
first to match the existing validator shape) to satisfy the existing tests, then remove the
`pytestmark` skip. **Read the whole test file before implementing** — it encodes specific rules
(banned font list, required CSS "Hallmark" comment stamp format, focus-visible/active state
requirements) that came from somewhere; grep the design-quality rules docs/skills
(`web/design-quality.md` rule content, if installed) to make sure your reimplementation matches
intent, not just makes the assertions pass.

```bash
python -c "import packages.orchestrator_persona" 2>&1 | head -3
grep -n "orchestrator-persona\|orchestrator_persona" "E:/Documents/Vibe-Coding/Ai Orchestrator/pyproject.toml"
```
EXPECTED: `packages/orchestrator-persona/` exists as a **standalone, separately-packaged**
directory (own `pyproject.toml`, `src/orchestrator_persona/{__init__,persona,persona_modes}.py`)
with **zero references** anywhere inside `orchestrator/` and **zero references** in the root
`pyproject.toml` (not a workspace member, not a dependency). Confirmed: it is prototype/blueprint
code, entirely unwired into the running app, exactly as the brief states.
`PONYTAIL_INTEGRATION_PLAN.md` (repo root) is a design doc describing the target integration
(§3.1 persona/persona_modes updates, §3.2 slash commands, §3.3 skills) — it is a plan, not a
status report of what's shipped.

### Phase 1: Build a quality scorecard on real generated outputs

Before touching any gate, measure the current state:
1. Locate any real generated outputs in the repo (check `output/`, sample runs, or generate a
   small fresh one via `python -m orchestrator --project "..." --budget <small>` if none exist).
2. For each: lint-clean % (`ruff check` / `black --check` pass rate, since
   `output/formatter.py` already auto-fixes most of this — measure what's left after
   auto-fix, not before), scanner finding counts by severity (from
   `generated_output_scanner.py`'s report), evaluator score distribution (pull from
   `EvaluatorService` results if logged/persisted).
3. Write this down as a table — this is your "before" scorecard. Do not skip this even if it
   feels like busywork; Phase 3's success metric is a delta against this exact baseline.

### Phase 2: Ponytail integration — verify status, don't assume

Cross-check `PONYTAIL_INTEGRATION_PLAN.md` milestone 1 (§3.1-3.3) against what's actually wired
(commands above already confirm: **blueprint-only**, prototype in `packages/orchestrator-persona`
unwired). If you choose to implement milestone 1:
- It touches `orchestrator/persona.py` / `orchestrator/persona_modes.py` per the plan's §3.1 —
  verify these files exist at those exact paths first (`ls orchestrator/persona*.py`) since the
  plan may describe target paths that don't match current layout.
- New slash commands (§3.2) live under `.claude/skills/`, not `orchestrator/` — that's fine, it's
  outside the Python package boundary rules.
- Follow TDD: this is new behavior (a persona mode that changes generation style/verbosity), it
  needs tests proving output actually shrinks/changes under Ponytail mode, not just that the flag
  is threaded through.

### Phase 3: Re-measure

Rerun Phase 1's scorecard against outputs generated *after* your Track D changes (design-quality
validator reinstated, ponytail wired if you did it, any scanner/gate changes). Report the delta
table: before vs. after, same metrics, same methodology.

### Fenced off — do NOT do this

- Claiming "quality improved" without the before/after scorecard from Phase 1/3. A scanner
  finding count you didn't measure before your change is not evidence of anything.
- README or docs claims ahead of implementation. **Precedent**: the ponytail README overclaim,
  documented as an oversell incident in `orchestrator-external-positioning`'s oversell ledger —
  cross-reference that skill before writing any external-facing claim about this track's results,
  and follow its four-part claim-discipline checklist (repro command + number + baseline + date/hash)
  for anything you write publicly.

### Gate / promotion

- [ ] `test_design_quality_validator.py` module skip removed; all its tests pass against a real
      reinstated `validate_design_quality`.
- [ ] Before/after quality scorecard exists with matching methodology on both sides.
- [ ] If ponytail milestone 1 was implemented: TDD'd, tests prove behavioral change (not just
      wiring), and any doc/README claim about it matches what's actually shipped.
- [ ] Full gate suite green.

### Success metric

A concrete before/after scorecard delta (lint-clean %, scanner findings by severity, evaluator
score distribution) on real generated outputs, plus the design-quality validator test module
converted from skip to green.

---

## Cross-track promotion protocol

Every fix in every track, no exceptions:

1. **TDD**: write the failing test first (RED) — for un-skips this means "remove skip, watch it
   fail for the *right* reason" before you fix anything; for new counters/validators, write the
   assertion before the implementation.
2. **Full gate suite**, matching CI exactly:
   ```bash
   ruff check orchestrator/
   black orchestrator/ --check
   lint-imports
   python scripts/check_new_root_files.py --baseline origin/master
   mypy orchestrator/domain orchestrator/application orchestrator/engine_core/container.py
   pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --cov=orchestrator --cov-fail-under=7
   bandit -r orchestrator/ -ll
   ```
   All of these are enforced mechanically — see `orchestrator-validation-and-qa` for what each
   one actually checks and why, and `orchestrator-change-control` for the Four Unbreakable Rules
   and the "never weaken a gate to pass" doctrine (no coverage-floor lowering, no contract
   exemptions, no xfail-to-green, no allowlist expansion without approval).
3. **Chronicle entry**: once a track (or a sub-step) is genuinely resolved, add an entry to
   `orchestrator-failure-archaeology`'s incident chronicle — symptom, root cause, evidence (commit
   hash), status — using its existing format. This is what prevents the next session from
   re-discovering "the circular import isn't real" from scratch.
4. **No track is "done" without its own Success metric section above being satisfied by a number
   you can point to, not a description.**

---

## Provenance and maintenance

Everything in this document was verified against the repo at commit `067ca737` (branch
`feat/response-healing`) on 2026-07-08. Re-verify before trusting any number here:

| Claim | Re-verify with |
|---|---|
| Track A: engine.py LOC | `wc -l orchestrator/engine.py` |
| Track A: skip count = 10 | `grep -c "circular" tests/test_phase6_10_comprehensive.py` |
| Track A: no live circular import | Re-run the Phase 0.5 two-block script in this doc |
| Track A: lint-imports contracts | `cat .importlinter` |
| Track B: flag wiring / 5 passing tests | `pytest tests/unit/test_response_healing.py -v` |
| Track B: telemetry TODO stub | `grep -n TODO orchestrator/infrastructure/telemetry.py` |
| Track C: drift count | `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py` |
| Track D: scanner advisory-only | Read `orchestrator/output_organizer.py` lines 236-259 |
| Track D: validator removed | `grep -rln validate_design_quality orchestrator/` (expect empty) |
| Track D: persona package unwired | `grep -n orchestrator_persona pyproject.toml` (expect empty) |
