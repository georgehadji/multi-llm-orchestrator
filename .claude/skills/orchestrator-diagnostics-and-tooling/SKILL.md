---
name: orchestrator-diagnostics-and-tooling
description: Catalog of every MEASURE-don't-eyeball tool for the Multi-LLM Orchestrator — the three shipped drift/hygiene scripts (check_config_drift.py, check_bom.py, flag_inventory.py), the two repo audit scripts (audit_openrouter_models.py, check_new_root_files.py), lint-imports, pytest/coverage as diagnostics, `python -X importtime` for startup regressions, git-based provenance/bisect techniques, and telemetry honesty (what is/isn't actually wired). Load this when you need to PROVE a claim about the repo instead of asserting it — symptoms like "is this flag dead", "did this file always have a BOM", "which config entries silently drop", "why did CLI startup get slow", "is this root file new or legacy", "are these model ids still live on OpenRouter", "did lint-imports actually pass". For WHY these gates exist use orchestrator-change-control; for the full grimp/Windows story use orchestrator-build-and-env; for evidence doctrine (which marker, where tests go) use orchestrator-validation-and-qa; for the flag catalog itself use orchestrator-config-and-flags.
---

# Orchestrator Diagnostics and Tooling

Everything here answers "how do I MEASURE this" rather than "how do I read the
code and guess." All commands verified against branch `feat/response-healing`
on 2026-07-08 (Windows 11, Python 3.12, PowerShell primary / Git Bash tool
available). Outputs below are real, pasted output from that run — expect
model counts, file counts, and violation counts to drift; re-run before
trusting a stale number.

---

## 1. The three shipped scripts (`.claude/skills/orchestrator-diagnostics-and-tooling/scripts/`)

These are stdlib-only (no `orchestrator` import required — they parse source
via `ast`/regex or read raw bytes), so they still work when the package is too
broken to import. That is deliberate: they exist for exactly the moments when
the normal tooling is unavailable.

### 1.1 `check_config_drift.py` — config/enum drift detector

**WHAT it measures**: whether every key/value in
`orchestrator/config/{costs,routing,fallbacks}.json` exactly matches a
`Model` or `TaskType` enum value in `orchestrator/models.py`. Any mismatch is
**silently dropped** by the builder guards in `models.py`
(`{Model(k): v for k, v in data.items() if k in Model._value2member_map_}`) —
no exception, no log line, just a model that never gets routed to or a
fallback that never fires. This is the single most expensive recurring bug
class in this repo (fixed repeatedly: 2026-06-23, `ab17b5f4`).

**WHEN to run it**: after editing any file under `orchestrator/config/`,
after adding/renaming a `Model` or `TaskType` enum member, before any commit
that touches routing/costs/fallbacks, and whenever a model "mysteriously"
never gets used.

**HOW to interpret output**: "HARD DRIFT" entries are real bugs — fix the
JSON key/value to equal the enum `.value` exactly. The "INFO" section
(Direction B: enum values with no JSON coverage) is not fatal by default —
some models are deliberately uncosted or unrouted — but a growing INFO list
is worth a second look. Pass `--strict` to also fail CI-style on Direction B.

**Exact command**:
```bash
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py --strict
```

**Exit codes**: `0` = no drift, `1` = drift found, `2` = could not run
(missing `models.py` or a config JSON file).

**Real output (2026-07-08, HARD DRIFT present — do not treat as clean)**:
```
Checked 133 cost keys, 9 routing keys, 51 fallback pairs against 130 Model values and 9 TaskType values.

HARD DRIFT — 5 config entr(ies) silently dropped by models.py:
  costs.json key not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4.1'
  costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-sonnet-4'
  fallbacks.json['qwen/qwen3-coder'] value not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'

Fix: make the JSON key/value EXACTLY equal the Model/TaskType enum .value.

INFO: 1 Model value(s) have no costs.json entry:
  internal/nano-banana-2
```
Exit code was `1`. This is a **live, unfixed drift** on this branch as of the
date above — treat it as a known open item, not a tooling bug. Before fixing,
check whether `qwen/qwen3.6-flash` and the three bare `anthropic/claude-*`
ids (missing patch-version suffixes) are typos of already-declared enum
values or genuinely new models that need a `Model` member added.

### 1.2 `check_bom.py` — UTF-8 BOM scanner

**WHAT it measures**: whether any `.py` file under `orchestrator/` or
`tests/` starts with a UTF-8 byte-order-mark (`EF BB BF`). A BOM is invisible
in most Windows editors and in PowerShell's default `Out-File` encoding, but
CPython on Linux raises `SyntaxError: invalid non-printable character
U+FEFF` when it hits one at the top of a file. This exact failure class cost
a full incident (commit `d913d136`, 32 files) and Windows tooling keeps
reintroducing it, so the check must be cheap enough to run after every bulk
file operation.

**WHEN to run it**: after any PowerShell-based bulk edit/rewrite of `.py`
files, after any tool that might resave files with a BOM (some editors, some
`Out-File` invocations), and as a pre-commit sanity check before pushing from
Windows.

**HOW to interpret output**: any file listed is a real Linux-CI landmine even
though it runs fine locally on Windows. Fix with the PowerShell one-liner the
script prints, or `sed -i '1s/^\xef\xbb\xbf//' <file>` in Git Bash.

**Exact command**:
```bash
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py --paths orchestrator scripts
```

**Exit codes**: `0` = clean, `1` = BOM(s) found, `2` = a scan directory does
not exist.

**Real output (2026-07-08 — 33 files currently carry a BOM, a regression
worth knowing about before you assume the repo is clean)**:
```
BOM FOUND in 33 file(s) under orchestrator, tests:
  orchestrator\async_event_store.py
  orchestrator\canary_deployment.py
  orchestrator\config_sync.py
  ... (33 total; includes orchestrator\state_mgmt\*, orchestrator\operations\*,
       tests\integration\test_execute_task_golden_path.py,
       tests\integration\test_resume_golden_path.py)
```
Exit code `1`. This is a bigger count than the historical incident (32 files
in `d913d136`) — either that fix regressed or new files were added with a
BOM since. Treat as an open finding, not evidence the script is broken.

### 1.3 `flag_inventory.py` — env-flag read-site inventory

**WHAT it measures**: every `os.environ[...]` / `os.environ.get(...)` /
`os.getenv(...)` read of a name matching `USE_*`, `ENABLE_*`, `ORCH_*`, or
`*_API_KEY` under a given directory tree, with file:line for each read site.
It is regex-based on literal string names — it will miss dynamic/indirect
lookups, and **a flag being read does not prove it is wired into the call
path**. `USE_PROVIDER_SORTING` is the canonical counterexample: it is read in
`orchestrator/config.py` but confirmed dead in the call path (2026-06-25).
Use `flag_inventory.py` to find candidates, then trace consumers by hand
before trusting a flag.

**WHEN to run it**: before adding a new `USE_*`/`ENABLE_*`/`ORCH_*` flag
(check it doesn't already exist under a slightly different name), when
auditing whether a documented flag is actually consumed, or when a flag
"does nothing" and you need every read site at once.

**HOW to interpret output**: flags with `(1 read site(s))` are marked
`[single read site — verify it is consumed downstream]` — a single read is
often just the config dataclass parsing the env var, with no guarantee
anything downstream branches on the resulting field. Flags with multiple
read sites across different files are lower risk but still worth a
consumer trace for anything load-bearing.

**Exact commands**:
```bash
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py --paths orchestrator tests
python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py --flag USE_RESPONSE_HEALING
```

**Exit codes**: `0` = report printed (or `--flag` found), `1` = `--flag`
given and never read anywhere (dead-flag suspect), `2` = a scan directory
does not exist.

**Real output (2026-07-08, default scan of `orchestrator/`, 19 distinct
flags found)**:
```
ENABLE_AUTO_COMMIT  (2 read site(s))
    orchestrator\git_service.py:102
    orchestrator\vcs\service.py:102
ENABLE_PR_COMMENTS  (2 read site(s))
    orchestrator\git_service.py:101
    orchestrator\vcs\service.py:101
...
ORCH_HITL_AUTOAPPROVE  (1 read site(s))  [single read site — verify it is consumed downstream]
    orchestrator\hitl\gate.py:67
...
USE_JSON_SCHEMA_RESPONSES  (1 read site(s))  [single read site — verify it is consumed downstream]
    orchestrator\config.py:184
USE_PROVIDER_SORTING  (1 read site(s))  [single read site — verify it is consumed downstream]
    orchestrator\config.py:188
USE_RESPONSE_HEALING  (1 read site(s))  [single read site — verify it is consumed downstream]
    orchestrator\config.py:191

Total distinct flags: 19
```
Full read-site catalog (the wired-or-dead judgment call for each flag) is
owned by `orchestrator-config-and-flags` — do not re-derive it here, just
regenerate the raw read-site list with this script and cross-reference.
Note `ENABLE_AUTO_COMMIT`/`ENABLE_PR_COMMENTS`/`GROK_API_KEY`/`XAI_API_KEY`
each show up under **two** files (`orchestrator/git_service.py` /
`orchestrator/vcs/service.py`, `orchestrator/xai_search.py` /
`orchestrator/knowledge/xai_search.py`, `orchestrator/provisioned_throughput.py`
/ `orchestrator/operations/provisioned_throughput.py`) — these are root-shim
vs. subpackage duplicate pairs (see `check_new_root_files.py` in §2.2); a
flag inventory diff between the two paths is a fast way to notice a shim has
drifted from its target.

---

## 2. Repo tools (`scripts/`)

### 2.1 `scripts/audit_openrouter_models.py` — live model-registry audit

**WHAT it measures**: cross-checks every `"provider/model[:variant]"` string
literal in `orchestrator/models.py`, `orchestrator/domain/model_registry.py`,
and `orchestrator/phase_aware_models.py` against the live OpenRouter
catalogue (`GET https://openrouter.ai/api/v1/models`, keyless). Reports any
referenced id absent from the live catalogue as `dead` (or `stale` if a
version-bumped replacement is detectable). It encodes two allowlists so it
does not false-positive: `_ANTHROPIC_HYPHEN` (OpenRouter server-side
normalizes `anthropic/claude-opus-4-6` → `anthropic/claude-opus-4.6`, so
hyphenated Anthropic ids absent from `/models` are not necessarily dead) and
`RUNTIME_ONLY_IDS` (video-gen models like `openai/sora-2-pro`,
`google/veo-3.1*` are billed per-second via the generation endpoint and never
appear in the chat-models snapshot, but do resolve at
`/api/v1/models/<id>/endpoints`).

**WHEN to run it**: periodically (it is wired into CI as a non-blocking
job), and any time a model "mysteriously" 404s/400s at runtime or you see an
"Unknown model X — allowing through" warning in logs.

**HOW to interpret output**: `dead` = referenced id genuinely gone from the
live catalogue and not covered by an allowlist rule — this is a real bug,
fix the id in `models.py`. `stale_replacements` = a newer version of the id
exists (e.g. a patch bump) — consider migrating. Non-zero exit is meant for
CI gating; locally it just means "go look."

**Exact commands**:
```bash
python scripts/audit_openrouter_models.py                       # fetch live + audit, human output
python scripts/audit_openrouter_models.py --json                # machine-readable
python scripts/audit_openrouter_models.py --snapshot snap.json  # offline against a saved snapshot
python scripts/audit_openrouter_models.py --save-snapshot snap.json
```

**Exit codes**: non-zero when any referenced id is dead (suitable for CI);
`0` when everything referenced resolves live or via an allowlist.

**Real output (2026-07-08, live fetch, `--json`)**:
```json
{
  "source": "live OpenRouter catalogue",
  "live_model_count": 509,
  "referenced_id_count": 150,
  "known_deprecated_count": 37,
  "dead": {},
  "stale_replacements": {}
}
```
Clean today — 0 dead ids among 150 referenced against 509 live models. This
is a network-dependent check; if you have no internet access, use
`--snapshot`/`--save-snapshot` to work offline against a previously captured
catalogue (stale snapshots will under-report drift).

### 2.2 `scripts/check_new_root_files.py` — root-kernel freeze gate

**WHAT it measures**: whether the change under review adds a **new**
root-level `orchestrator/*.py` file that is not on the hardcoded
`KERNEL_ALLOWLIST` (40 names, e.g. `models.py`, `engine.py`, `cli.py`,
`config.py`). This enforces CLAUDE.md's Unbreakable Rule #4 ("No new
root-level orchestrator/*.py modules"). It has two modes:

- **`--baseline <ref>`** (CI mode, authoritative): uses
  `git diff --name-only --diff-filter=A <ref> -- 'orchestrator/*.py'`, i.e.
  only files **added** since the baseline, restricted to direct children of
  `orchestrator/` (nested subpackage files are explicitly allowed and
  excluded from the diff by a `p.count("/") == 1` filter). This is the mode
  CI actually runs: `python scripts/check_new_root_files.py --baseline
  origin/master`.
- **No `--baseline`** (full-audit mode): lists **every** current root-level
  file not on the allowlist, regardless of when it was added. This will
  flood with pre-existing legacy violations — it is not a gate, it is a
  standing-debt inventory. Do not treat a non-zero exit from full-audit mode
  as "CI would fail"; only `--baseline` mode matches what CI enforces.

**WHEN to run it**: `--baseline origin/master` before every commit/PR that
touches `orchestrator/` at the root; no-baseline mode occasionally, to see
how large the legacy root pile currently is (tracked in
`orchestrator-architecture-contract` as the "~256-file root kernel" known
weak point).

**HOW to interpret output**: `--allowlist` prints the current 40-file
allowlist and the current total root-file count — use it to check whether a
name you're about to add is already covered. Any violation under
`--baseline` mode means: move the new file into a subpackage
(`orchestrator/domain/`, `orchestrator/application/`,
`orchestrator/infrastructure/`, etc.) or, if it is a deliberate kernel
addition, add it to `KERNEL_ALLOWLIST` in the script itself (that edit is
itself a change-controlled decision — see `orchestrator-change-control`).

**Windows gotcha (verified 2026-07-08)**: the script prints `✅`/`❌` emoji.
On a Windows console using a non-UTF-8 code page (e.g. `cp1253`), this
raises `UnicodeEncodeError: 'charmap' codec can't encode character
'❌'` and the process exits with a traceback instead of the intended
report — the exit code you observe (`1`) is from the crash, not necessarily
from real violations. Fix: run with `PYTHONIOENCODING=utf-8` prefixed, or
`chcp 65001` first in `cmd.exe`, or run inside Git Bash / WSL where UTF-8 is
already default. This is a general trap for any script in this repo that
prints emoji on Windows — not unique to this file.

**Exact commands**:
```bash
# CI-authoritative mode (matches .github/workflows/ci.yml)
PYTHONIOENCODING=utf-8 python scripts/check_new_root_files.py --baseline origin/master

# Full-audit mode (standing-debt inventory, NOT a CI-equivalent gate)
PYTHONIOENCODING=utf-8 python scripts/check_new_root_files.py

# Print the current allowlist
PYTHONIOENCODING=utf-8 python scripts/check_new_root_files.py --allowlist
```

**Exit codes**: `0` = no violations (or `--allowlist` printed), `1` =
violation(s) found, `2` = the underlying `git diff` failed.

**Real output (2026-07-08)**:

`--baseline origin/master` (what CI actually gates on):
```
✅ No root-level violations (vs. origin/master).
```
Exit `0` — no *new* root files have been added on this branch relative to
`origin/master`.

No-baseline full audit (standing debt, not a CI signal):
```
❌ Root-level file violations (current state):
   orchestrator\ab_testing.py
   orchestrator\accountability.py
   orchestrator\adaptive_router.py
   ... (219 total)

219 violation(s) found.
```
Exit `1`. These 219 files are pre-existing legacy root modules, not
something this branch introduced — confirmed by the `--baseline` run above
returning clean. Use this number as a baseline for tracking root-kernel
demolition progress over time, not as a per-PR blocker.

`--allowlist` (first 4 of 40 lines shown):
```
Kernel allowlist (40 files):
  __init__.py
  __main__.py
  api_clients.py
  app_builder.py
  ...
```

---

## 3. `lint-imports` (import-linter) — architecture boundary contracts

**WHAT it measures**: the 5 `.importlinter` contracts (domain purity,
application-no-concrete-infra, application-services-no-engine,
engine-core-no-loose-infra, root-modules-no-infra) that encode the
hexagonal dependency rule as executable checks. CI runs this as a **blocking**
job.

**WHEN to run it**: before any commit touching import statements across
layer boundaries; it also runs as a pre-commit hook (`language: system`).

**Exact command**:
```bash
lint-imports
```

**Real output (2026-07-08, local Windows run — succeeded this time)**:
```
Analyzed 782 files, 1218 dependencies.
--------------------------------------

Domain layer must not import application or infrastructure KEPT
Application layer must not import concrete infrastructure adapters KEPT
Application services must not import from engine.py directly KEPT
engine_core pipeline modules must not import infrastructure directly KEPT
Root modules must not import infrastructure directly (shims excepted) KEPT

Contracts: 5 kept, 0 broken.
```
Exit `0`, all 5 contracts KEPT on this run. **This is not a guarantee it will
always run cleanly on Windows** — `lint-imports`'s dependency `grimp` has a
documented Rust-extension panic on some Windows setups, pinned/worked around
via `grimp==3.3`. **CI on ubuntu-latest is the authoritative signal**, not a
local Windows pass or fail. Do not trust a local grimp panic as "the contract
is broken" (it may be an environment issue) and do not trust a local pass as
"CI will pass" without also checking CI. Full grimp incident story, the pin
rationale, and Windows-specific workarounds live in
`orchestrator-build-and-env` — do not re-derive them here.

---

## 4. pytest / coverage as diagnostics

**WHAT it measures**: correctness (test pass/fail) and the coverage ratchet
(`fail_under = 7` in `pyproject.toml`, raised from a Phase-A baseline of 6 —
re-verify the current value before quoting it, it is designed to keep
climbing). CI's exact marker expression (from `.github/workflows/ci.yml`,
re-verify before quoting):
```
pytest -m "not slow and not requires_api and not stress and not e2e" --cov=orchestrator --cov-fail-under=6
```
Note the CI invocation currently passes `--cov-fail-under=6` on the command
line while `pyproject.toml` declares `fail_under = 7` — the command-line flag
wins when both are present; treat this as a config-drift smell worth
re-checking, not a settled fact (dated 2026-07-08).

**WHEN to run it**: standard TDD loop (RED → GREEN → REFACTOR), before every
commit, and whenever `tests/test_preexisting_problems.py` needs updating
(it's an `xfail(strict)` ledger of catalogued bugs — an unexpected `XPASS`
there is itself a diagnostic signal that something got fixed and the ledger
entry needs removing).

**HOW to interpret output**: this skill only owns the *mechanics* of running
these tools as diagnostics. The full evidence doctrine — which marker to use,
where a new test file goes, what a "contract test" protects, how to react to
an unexpected `XPASS` — is owned by `orchestrator-validation-and-qa`; go
there for interpretation depth. Here, treat pytest/coverage the same as any
other measurement tool: run it, read the number, don't guess it.

**Exact commands** (see also `CLAUDE.md` "Common Commands"):
```bash
pytest tests/ -v --cov=orchestrator --cov-report=term-missing
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest -n auto tests/
```

---

## 5. `python -X importtime` — startup-regression diagnostic

**WHAT it measures**: per-module wall-clock import cost, printed as a
cumulative-time tree to stderr. This is the technique that catches "why did
CLI startup / import get slow" regressions — it is a profiler for import
side effects, not a guess based on reading `import` statements.

**Worked example (real, dated 2026-07-07)**: `orchestrator/infrastructure/
llm_client.py` had `import instructor` at **module scope** (top of file).
On Windows, cold-importing `instructor` cost roughly 30 seconds — enough to
silently blow CLI smoke-test timeouts and make `import orchestrator` (and
anything that imports it, including test collection) pay a 30s tax on every
process start, even for code paths that never construct an LLM client. This
was found by running `python -X importtime -c "import orchestrator"` and
reading off which leaf module dominated the cumulative time column. The fix
(currently an **uncommitted working-tree change** on this branch — see
`git status`, `M orchestrator/infrastructure/llm_client.py`) made the
import lazy: `instructor` is now imported inside a `@staticmethod
_instructor_mode()` helper and inside the two call sites that actually
construct a client (`orchestrator/infrastructure/llm_client.py`, search for
`import instructor` — 3 occurrences, all function-local as of this date).
The docstring on `_instructor_mode()` states the rationale verbatim:
> "``instructor`` costs ~30s to import cold on Windows; deferring keeps
> `import orchestrator` (and CLI startup) fast for paths that never create a
> client."

**Re-measured 2026-07-08 (this session, warm cache, after the lazy-import
fix)**: `import orchestrator` alone took ~5.5s wall time
(`time python -c "import orchestrator"` → `real 0m5.622s`), and `instructor`
does **not** appear anywhere in the `importtime` tree for a bare
`import orchestrator` — confirming the lazy-import fix is effective; the
30s tax is now paid only by code paths that actually build an LLM client.
(A prior note put post-fix warm import at "~12-13s" — this session measured
lower; treat both as points on a noisy distribution across machine/cache
states, not a precise SLA. What matters is the qualitative result: instructor
no longer loads at `import orchestrator` time.)

**WHEN to run it**: whenever CLI/test-collection startup feels slow,
whenever you add a new module-scope `import` of a heavy third-party package
(anything doing network client setup, ML/tokenizer loading, or large regex
compilation at import time is a suspect), and after any refactor that
touches what gets imported transitively by `orchestrator/__init__.py` or
`orchestrator/engine.py` (currently the single largest transitive import,
~569ms cumulative in this session's run — separate from the instructor
issue, just the natural cost of wiring the Mediator).

**HOW to interpret output**: each line is
`import time: <self us> | <cumulative us> | <module>`, indented by import
depth. Sort mentally by the **cumulative** column (2nd number) at the
shallowest depth you care about — a huge self-time on a single leaf module
(as `instructor` was) is the classic smoking gun. A broad, evenly-distributed
cost across hundreds of small modules is a different problem (too many
eager imports, not one culprit) and calls for lazy-loading strategy instead
of a single fix.

**Exact command**:
```bash
python -X importtime -c "import orchestrator" 2> importtime_out.txt
# then inspect importtime_out.txt, or pipe through a search for the module you suspect:
python -X importtime -c "import orchestrator" 2>&1 | grep -i <suspect_module>
```

---

## 6. Git-based diagnostics

### 6.1 `git bisect` — regression localization

Use when something worked at some past commit and is broken now, and you
don't know which commit introduced the break. Standard workflow:
```bash
git bisect start
git bisect bad                      # current HEAD is broken
git bisect good <known-good-ref>    # e.g. a tag or old commit
# git will check out a midpoint; run your reproduction, then:
git bisect good   # or: git bisect bad
# repeat until git reports the first-bad commit
git bisect reset
```
Useful reproduction commands to run at each bisected commit in this repo:
`pytest tests/<specific_test>.py -x`, `python -c "import orchestrator"`, or
`python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py`.

### 6.2 `git log --diff-filter=A` — file provenance

Use to answer "was this file always here, or did it get added recently /
who added it." This is exactly the primitive `check_new_root_files.py`
builds on (§2.2) — `--diff-filter=A` restricts to **added** paths, so
renames/edits don't pollute the answer.
```bash
git log --diff-filter=A --oneline -- <path>          # commits that ADDED this path
git log --follow --oneline -- <path>                 # full history including renames
git show <hash> --stat                                # what else changed in that commit
```
Real example from this session — confirming `llm_client.py` predates the
instructor fix (i.e. the fix is a modification, not a new file):
```
$ git log --diff-filter=A --oneline -- orchestrator/infrastructure/llm_client.py | tail -5
0c699d3a fix: add missing untracked modules + apply black formatting
```

### 6.3 `git diff <ref>...HEAD --stat` — scope-of-change sanity check

Use before any PR to confirm the diff matches your mental model of what
changed (see also CLAUDE.md's PR-creation workflow, which mandates this
before drafting a PR body).

---

## 7. Telemetry honesty — what is/isn't wired

Do not assume every counter or tracking mechanism referenced in the
codebase is actually implemented. Verified stub, quoted exactly, from
`orchestrator/infrastructure/telemetry.py:165-172`:
```python
    def record_validator_failure(self, model: Model) -> None:
        """Record a validator failure for the given model."""
        # TODO: Implement proper tracking
        # For now, just degrade trust factor slightly
        profile = self._profiles.get(model)
        if profile is None:
            return
        profile.trust_factor *= _TRUST_DEGRADE
```
`record_validator_failure` degrades a model's trust factor but does **not**
record the failure anywhere queryable — no counter increment, no
persisted event, just a multiplicative decay applied in place. If you are
debugging "why doesn't the dashboard show validator failure counts" or
"why can't I query how many times model X failed validation," the answer is:
that tracking does not exist yet, this is the TODO. Healing counters and
cost counters elsewhere in the telemetry surface have similarly been found
not fully wired end-to-end (per `docs/CODEBASE_MINDMAP.md` and prior
sessions) — do not trust a telemetry field's *name* as proof it is
populated; grep for where it is actually written before building anything
on top of it, the same discipline as §1.3's flag-inventory approach.

---

## When NOT to use this skill

- You want to know **why** a flag/gate/pattern exists, not how to measure
  it → `orchestrator-config-and-flags` (flag catalog + wired/dead verdicts),
  `orchestrator-change-control` (why gates exist, the Four Unbreakable
  Rules), `llm-orchestration-reference` (domain theory).
- You want the **grimp/Windows Rust-panic story** in full, or general
  environment/dependency setup → `orchestrator-build-and-env`.
- You want to know **which pytest marker to use, where a test file goes, or
  what counts as sufficient evidence** → `orchestrator-validation-and-qa`.
- You're triaging a **live failure right now** and need a symptom→cause
  lookup ranked by cost → `orchestrator-debugging-playbook`.
- You want the **incident chronicle** (root cause + commit hash + status
  for settled bugs) → `orchestrator-failure-archaeology`.
- You want to know **how to run** the orchestrator itself (CLI flags,
  dashboard, MCP server) → `orchestrator-run-and-operate`.

---

## Provenance and maintenance

Re-verify anything below before relying on it — all facts here are dated
2026-07-08 on branch `feat/response-healing`.

| Fact | Re-verification command |
|---|---|
| Config drift count/entries | `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py` |
| BOM file count/list | `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py` |
| Env-flag inventory | `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/flag_inventory.py` |
| Live OpenRouter model audit | `python scripts/audit_openrouter_models.py --json` |
| Root-kernel violations (CI-equivalent) | `PYTHONIOENCODING=utf-8 python scripts/check_new_root_files.py --baseline origin/master` |
| Root-kernel violations (full legacy audit) | `PYTHONIOENCODING=utf-8 python scripts/check_new_root_files.py` |
| lint-imports contracts | `lint-imports` |
| Coverage ratchet floor | `grep -n "fail_under" pyproject.toml` |
| CI marker expression / cov flag | `grep -n "cov-fail-under\|not slow and not requires_api" .github/workflows/ci.yml` |
| `instructor` still lazy (not at module scope) | `grep -n "^import instructor\|import instructor" orchestrator/infrastructure/llm_client.py` (all hits should be indented/function-local, none at column 0 module scope) |
| Import-time regression check | `python -X importtime -c "import orchestrator" 2> out.txt` then inspect `out.txt` |
| Telemetry TODO stub still present | `grep -n "TODO: Implement proper tracking" orchestrator/infrastructure/telemetry.py` |
