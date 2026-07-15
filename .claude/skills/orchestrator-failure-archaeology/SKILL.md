---
name: orchestrator-failure-archaeology
description: The incident chronicle for the Multi-LLM Orchestrator. Load this BEFORE re-investigating any bug that smells familiar, before proposing to "clean up" or re-enable something that looks dead, or before relitigating a settled design decision. Symptom keywords that should trigger this skill — silent auto-approval / tasks approved without a human; cryptic SyntaxError or import errors on Linux/CI only (BOM); 300-second hangs in batch processing; evaluation scores that ignore extra self-consistency runs; hierarchy node-ID collisions after deletes; CLI crashing entirely on startup after a refactor; config entries silently dropping (model-id / enum drift); "why is engine.py so big / can I add a method to engine.py"; "should we use Verbalized Sampling"; "is use_provider_sorting doing anything"; grimp Rust panic on Windows; xfail tests turning green unexpectedly. Every entry: symptom → root cause → evidence (commit hash) → status, so nobody re-fights a settled battle.
---

# Orchestrator Failure Archaeology

Chronicle of every major investigation, dead end, rejected fix, and revert in this repo,
with commit-level evidence. Purpose: **stop repeated work**. If a bug or debate below is
marked SETTLED, do not reopen it without new evidence — escalate via
`orchestrator-change-control` instead.

All commit hashes below were verified against `git log` on **2026-07-07** on branch
`feat/response-healing`. Re-verify any hash with:

```powershell
git log -1 --format='%h %ad %s' --date=short <hash>
git show <hash> --stat
```

## When NOT to use this skill

| You need... | Use instead |
|---|---|
| Rules for making a change / which gates block you | `orchestrator-change-control` |
| Live debugging of a NEW failure (triage steps) | `orchestrator-debugging-playbook` |
| Layer rules, import-linter contracts, Four Unbreakable Rules | `orchestrator-architecture-contract` |
| Flag/env-var catalog (what each flag does today) | `orchestrator-config-and-flags` |
| The four active hard-problem tracks and current plans | `orchestrator-hardest-problems-campaign` |
| Research directions (VS/MAP-Elites future work) | `orchestrator-research-frontier` |

This skill is the **history**. It tells you what already happened and why; it does not
tell you what to do next.

---

## The Chronicle (newest first)

Status legend: **SETTLED** = fixed + locked by tests/gates, do not relitigate.
**FIXED** = fixed, no dedicated lock. **OPEN** = known, catalogued, not fixed.
**RETIRED** = deliberately abandoned with written rationale.

### 2026-07-01 — Test remediation + config resync on feat/response-healing

- **Symptom:** integration and regression suites regressed after the response-healing
  branch merged master (`2fc508ae` merge).
- **Root cause:** accumulated drift between branch and master fixtures/config.
- **Evidence:** `ab17b5f4` "test: remediate integration and regression test regressions
  on feat/response-healing" (2026-07-01).
- **Status:** FIXED. Response-healing hardening itself is an **active campaign track** —
  see `orchestrator-hardest-problems-campaign`. Feature flag `USE_RESPONSE_HEALING`
  (default false, verified in `orchestrator/config.py` 2026-07-07); plugin introduced in
  `80c1fa51` (2026-06-20).

### 2026-06-26 — BOM characters: Linux-only cryptic import errors

- **Symptom:** modules that worked fine on Windows dev machines produced cryptic
  `SyntaxError`/import errors on Linux CI only.
- **Root cause:** UTF-8 BOM (byte-order-mark, the invisible `﻿` prefix some Windows
  editors add) in 32 `.py` files. Windows Python tolerates it; some Linux paths did not.
  Same pass also fixed 4 bare `except:` clauses.
- **Evidence:** `d913d136` "fix: remove BOM characters from 32 Python files" (2026-06-26)
  — 35 files, 36 insertions / 36 deletions (one-char diffs).
- **Status:** fix landed (`d913d136`) but new BOM files have since appeared — **re-run the
  checker before assuming this is closed.** `python
  .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py` found **2 files**
  with BOM as of 2026-07-11 (`tests/integration/test_execute_task_golden_path.py`,
  `tests/integration/test_resume_golden_path.py`) — down from a previously-reported 33, but
  nonzero, confirming this class of regression keeps recurring in new files rather than being
  a one-time, permanently-closed fix. This is the canonical example of the
  **environment/encoding trap** class (Windows dev vs ubuntu-latest CI). If you see a
  Linux-only import error on a file that "looks fine", check for BOM first:
  ```powershell
  Get-Content -Encoding Byte -TotalCount 3 path\to\file.py   # EF BB BF = BOM
  ```
  or run the checker script above rather than eyeballing individual files.

### 2026-06-25 — HITL silent auto-approval → fail-closed gate (FIX-1)

- **Symptom:** unattended runs proceeded through human-in-the-loop (HITL) checkpoints as
  if a human had approved. Nobody had. Zero errors, zero logs — the worst failure class
  in this repo (silent auto-approve).
- **Root cause:** the decision channel defaulted to auto-approval when no real channel
  was configured.
- **Fix:** fail-closed `DecisionChannel` gate in `orchestrator/hitl/` — with no channel
  configured, the run **raises** instead of approving. Legacy escape hatch:
  `ORCH_HITL_AUTOAPPROVE=true` env var (explicit opt-in, verified in
  `orchestrator/hitl/channel.py`).
- **Evidence:** `11deb573` "fix(hitl): replace silent auto-approval with fail-closed gate
  (FIX-1)" (2026-06-25). Locked by `tests/unit/test_hitl_gate.py` (15 test functions as
  of 2026-07-07; FIX-1 landed with 11).
- **Status:** **SETTLED — do not relitigate.** Fail-closed is the default forever.
  Any proposal to auto-approve by default is a change-control escalation, not a PR.

### 2026-06-25 — P4 hierarchy ID collision + P5 batch truthiness hang

- **P4 symptom:** deleting a hierarchy node then adding a new one could silently reuse
  the deleted node's ID (IDs were length-based: `len(nodes)`), corrupting parent/child
  links. **Fix:** monotonic counter for node IDs.
- **P5 symptom:** `BatchClient` waited on `if request.result:` — a **valid falsy** result
  (empty string, 0, empty list) never satisfied the check → 300-second hang on successful
  requests. **Fix:** gate on `status == COMPLETED`, not result truthiness.
- **Evidence:** `416b9e18` "fix(p4,p5): eliminate both pre-existing catalogued problems"
  (2026-06-25) — touched `orchestrator/hierarchy.py` and the xfail ledger
  `tests/unit/test_preexisting_problems.py`.
- **Status:** SETTLED. Both were entries in the xfail(strict) ledger (see Settled
  Decisions below) — the fix flipped their ledger tests from xfail to pass, which
  xfail(strict) forces you to acknowledge explicitly.
- **Lesson (recurring class):** never use truthiness as a completion/validity check on
  LLM or API results. Empty-but-successful is a normal outcome.

### 2026-06-25 — EvaluatorService._aggregate discarded consistency runs

- **Symptom:** evaluation self-consistency was configured for N runs, but scores looked
  suspiciously like single-run scores. Extra runs cost money and changed nothing.
- **Root cause:** `_aggregate` returned `scores[0]` for 3+ runs — runs 3..N were
  silently discarded.
- **Fix:** explicit 0/1/2/3+ handling; **median** for 3+ runs; 2-run delta rule
  (delta ≤ threshold → mean, else conservative) retained.
- **Evidence:** `e863f0c8` "test(bug-scan): proactive invariant suite + fix _aggregate
  dropping runs" (2026-06-25) — `orchestrator/services/evaluator.py` +
  `tests/unit/test_bug_scan.py` (321-line invariant suite).
- **Status: OPEN — do NOT mark this SETTLED.** An earlier revision of this entry said
  SETTLED; that was wrong. There are **two `EvaluatorService` classes with two separate
  `_aggregate` implementations**: `orchestrator/services/evaluator.py` got the median-for-3+
  fix from `e863f0c8`; `orchestrator/application/evaluator.py` still has the unfixed
  `_aggregate` that returns `scores[0]` for 3+ runs, silently discarding runs 2..N. Verified
  live 2026-07-11: `orchestrator/engine_core/container.py:344` —
  `from ..application.evaluator import EvaluatorService` — **production DI wiring
  constructs the unfixed class.** If self-consistency actually runs 3+ evaluation passes in
  production, the original bug may still be live behind the "fixed" commit. This is a human
  decision (consolidate the two classes, or repoint `container.py` at
  `services/evaluator.py`) via `orchestrator-change-control` — do not silently patch it as a
  drive-by. Full reproduction, the exact diff between the two `_aggregate` bodies, and the
  file:line evidence: `orchestrator-proof-and-analysis-toolkit` Recipe 4.
- **Related open caveat:** the response DiskCache can **defeat self-consistency** (N
  "independent" runs hitting the same cached response are 1 run). Catalogued 2026-06-25,
  locked as documented behavior by `tests/unit/test_cost_reduction.py`. OPEN as a design
  tension, not a bug ticket.

### 2026-06-25 — Cost-reduction audit: dead flags and unwired caches

- **Findings (audit, evidence = `tests/unit/test_cost_reduction.py`):**
  - `USE_PROVIDER_SORTING` is a **dead flag**: declared and env-read in
    `orchestrator/config.py` (line ~188) and mirrored in
    `orchestrator/crosscutting/config.py`, but **nothing in the call path consumes it**
    (verified by grep 2026-07-07 — only those two files reference it).
  - The semantic cache is **not wired into the call path** — it exists but does nothing
    for live calls.
  - The **real** cost win is the response DiskCache ($0 on hits, 48h TTL) plus the
    cost/cache/token-optimization defaults.
- **Status:** SETTLED as documented facts. Do not "quickly wire up" the dead flag or the
  semantic cache as a drive-by — either is an architectural change (change-control).
  Do not delete the dead flag without checking `orchestrator-config-and-flags` first.

### 2026-06-25 — VFM routing + phase policy overhaul

- **What:** 19 high-VFM (value-for-money) 2026 OpenRouter models declared and priced;
  routing made free-tier-first per text task; dead `grok-4-mini` removed; per-phase
  temperature/reasoning policy centralized in `orchestrator/domain/phase_policy.py`.
- **Evidence:** `15026700` (models), `23757976` (phase policy), `4c7b2ccc` (policy doc),
  all 2026-06-25. Locked by `tests/unit/test_vfm_routing.py`.
- **Status:** SETTLED. Free-tier-first routing and the phase-policy single source of
  truth are decided. Details live in `llm-orchestration-reference`.

### 2026-06-24 — ARCH-AUDIT-V2 (score 5/10) and the remediation campaign

- **Trigger:** architecture audit scored the codebase 5/10 with 2 CRITICAL and 3 HIGH
  risks. Evidence: `431fc89c` "docs(arch): ARCH-AUDIT-V2 — score 5/10, 2 CRITICAL, 3
  HIGH risks" (2026-06-24), adding `ARCH-AUDIT-V2.md` (repo root, 213 lines).
- **Remediation phases (A1–E2), all verified commits:**

| Phase | What | Evidence (date) |
|---|---|---|
| A1–A3, D1, D4 | Week-1 architecture fixes; explicit `__all__` on root shims | `4fc1310a` (06-23), `c1758019` (06-24) |
| A4 | Reconcile divergent root-shim/subpackage file pairs (structural drift) | `e16e1c0f` (06-23), `5eea5aa0`, `8e91dba3` (06-24) |
| A5 | Move `codebase_*` / `context_*` root files into subpackages | `9a0c084b`, `86664db7` (06-24) |
| B | Safety net: tests, null adapters, port contracts | `c68d8460` / `cfa95d87` (06-24) |
| C1 | `VSSamplerPort` — break engine_core→reasoning concrete dep | `2d468c07` (06-23) |
| C2 | `SkillStore` port, extract aiosqlite from core | `85efa0ec` (06-24) |
| C3 | Reclassify `cli_dispatch` + `chat_cli` | `7957e1d6` (06-24) |
| D2 | Delete 3 zero-importer shims | `485417ef` (06-24) |
| D3 | Unified stage retry constants | `fd387c44` (06-24) |
| E2 | 4 architecture contract tests | `38211763` (06-24) |
| Final audit | 13/15 tasks done, 0 contract errors | `383090ee` (06-24) |

- **engine.py demolition** (part of the same campaign; engine.py was 1867 lines):
  - `4ac9b4f1` / `de7c9c9a` "Phase C.1 — Remove 5 dead methods + fix 2 critical bugs"
    — the bugs: a leftover `self._container` reference where the attribute is `self._c`
    (cleanup bug), and a stale planner reference in `set_optimization_backend`.
  - `9cb92059` "Phase C.1 — Remove 7 more dead methods".
  - `b7263052` / `740a0680` "Phase C.2 — Remove 9 more dead/shadow methods".
  - `9fb2b826` "Phase C.5 — Fix audit_log/cost_predictor + remove orphan imports".
  - (Duplicate hashes for the same message are cherry-picks across branches — both exist.)
- **Current state (verified 2026-07-07):** `orchestrator/engine.py` = **1250 lines**.
  Target <300. Demolition is **incomplete and active** — track (1) of the hardest-problems
  campaign. Engine↔container circular imports still cause 6 skips in
  `tests/test_phase6_10_comprehensive.py`.
- **Status:** Campaign SETTLED as direction (import-linter contracts now BLOCKING in CI);
  demolition itself OPEN/active. **Never add logic to engine.py** — that battle is over.

### 2026-06-23 — Config/enum model-id drift (recurring episode class)

- **Symptom:** entries in `orchestrator/config/{costs,routing,fallbacks}.json` silently
  vanish from effect — a model gets no cost, no route, no fallback — with **no error**.
- **Root cause:** config builders key JSON entries by exact string match against
  `Model` enum values in `orchestrator/models.py` and **do not resolve aliases**.
  A JSON key written as an alias id (e.g. hyphenation/version variants of the same
  model) simply drops.
- **Episodes (all verified):**
  - `912333df` (2026-06-20) resync model registry with live OpenRouter catalog.
  - `4e31dbc6` (2026-06-23) 2026 model roster + single-source-of-truth pass; 9 drifted
    entries fixed this day across `ec5f3d57`, `82868815`, `b4867f29`.
  - `15026700` (2026-06-25) VFM roster, full 147-id live audit.
  - `ab17b5f4` (2026-07-01) another resync during test remediation.
- **Status:** OPEN as a class — a permanent structural fix is campaign track (3). The
  **rule** is settled: JSON key must equal `Model.X.value` exactly, and you run the drift
  tests (`tests/unit/test_openrouter_model_audit.py`, `tests/unit/test_vfm_routing.py`,
  `tests/test_models.py`) after touching either side. Rule text lives in
  `orchestrator-change-control`; flag/config details in `orchestrator-config-and-flags`.

### 2026-06-23 — Security-first pass: 9 vulnerabilities

- **What was fixed** (all in one pass): admin key comparison → `hmac.compare_digest`
  (timing-safe); dev_server port injection; SSRF (server-side request forgery) guards;
  WebSocket bind moved to `127.0.0.1`; SQL identifier allowlists; CSP `unsafe-eval`
  removal.
- **Evidence:** `611c1403` "security: fix 9 CRITICAL+HIGH+MEDIUM vulnerabilities
  (security-first pass)" (2026-06-23). Companion: `bfe31b41` (same day) hardened
  **generated** apps/websites secure-by-default and added
  `orchestrator/safety/generated_output_scanner.py` as a delivery gate.
- **Status:** SETTLED. bandit HIGH is a BLOCKING CI gate. Do not loosen any of these
  (e.g. rebinding WS to 0.0.0.0 "for convenience") without change-control.

### 2026-06-21/22 — CLI command-extraction breakage

- **Symptom:** after extracting handlers from `cli.py` into `orchestrator/commands/*.py`,
  various subcommands failed — and at one point **the entire CLI crashed on startup**.
- **Root causes (multiple, from one refactor):**
  - handler functions registered with wrong `func=` names (extracted as `cmd_*` but
    registered under old names) — fixed by `a4946977` "rename cmd_* -> execute"
    (2026-06-21);
  - star imports the old code depended on were dropped — restored by `d9099e3a`
    (2026-06-21);
  - missing/wrong relative imports in extracted modules; a relative-import bug in
    `orchestrator/commands/integration.py` crashed the whole CLI (every command dies at
    import time because command discovery imports all modules — `d8278b5b` dynamic
    `COMMAND_MODULES` discovery, 2026-06-21). *Exact fixing commit for the
    integration.py import: inference — folded into the 2026-06-22/23 cleanup commits
    (`56d3015b`, `4e31dbc6`); the module's git history shows only aggregate commits.*
- **Extraction milestone commits:** `15cdd393`, `1d9c42a7`, `01b24d0c`, `31daed80`
  (2026-06-21), `1847e025` (2026-06-22, final nash+nexusscope handlers).
- **Status:** FIXED. **Lesson (refactor-fallout class):** when extracting from a
  God-module, one broken import in any extracted module can kill the whole entry point.
  Smoke-test `python -m orchestrator --help` after every extraction batch.

### 2026-06-15 → 2026-06-25 — Verbalized Sampling: built, judged, retired

- **Timeline:** VS (Verbalized Sampling — a prompting technique from arXiv:2510.01171
  for recovering LLM output diversity) was implemented enthusiastically over 2026-06-15/16:
  `a6295cb5`, `33284542`, `a7beeae3` (VS-first GenerateStage), `22386a76`, `d00983b6`,
  `a6415a0f` (architecture selection), `d3b887c1` (code review + decomposition + bug
  hunting).
- **Verdict (2026-06-15 analysis, `docs/VERBALIZED_SAMPLING_ANALYSIS.md`):** the
  pre-existing `VerbalizedSamplingPipeline` (`orchestrator/reasoning/ara_pipelines.py`)
  was (a) **never invoked** — no TaskType maps to it, ARA layer not wired into engine
  mainline — and (b) **not faithful to the paper**: it used a list-level prompt and an
  inverted probability definition, i.e. exactly the variant the paper proves recovers
  only a uniform distribution.
- **Retirement outcome:** dead pipeline documented as dead; concrete engine_core
  dependency broken via `VSSamplerPort` (`2d468c07`, remediation C1). The **salvaged
  value** — MAP-Elites seeding, synthetic/test-data diversity, VS-tail on retry — is
  future work owned by `orchestrator-research-frontier`.
- **Status:** RETIRED. **Do not resurrect the old pipeline.** Any new VS work must be
  paper-faithful (distribution-level prompt) and go through change-control.

### Pre-2026-06 background incidents (context, verified)

| Date | What | Evidence | Status |
|---|---|---|---|
| 2026-06-04 | Phase-1 fixes: event history, orphan tasks, CachePathProvider | `7edcd61f` | FIXED |
| 2026-05-30 | CRITICAL/HIGH findings from 2026-05-29 architectural audit applied; broken imports fixed | `8296af32`, `e07467cd` | FIXED |
| 2026-03-26 | Rate limiter async-lock / state blocking-IO investigation | branch `origin/fix/rate-limiter-async-lock-state-blocking-io`, tip `321547ac` | see branch table |
| 2026-02-14 | Stress suite S1–S12 added; S2, S6, S7 fail pre-existing | `3143b49b`; CLAUDE.md "Known Limitations" | OPEN, documented, non-blocking |

---

## Settled decisions — do NOT relitigate

New evidence → escalate via `orchestrator-change-control`. Otherwise these are closed.

| Decision | Locked by | Origin |
|---|---|---|
| HITL is **fail-closed**; auto-approve only via explicit `ORCH_HITL_AUTOAPPROVE=true` | `tests/unit/test_hitl_gate.py` | `11deb573` |
| Known-bug ledger uses **xfail(strict)**: catalogued bugs get an xfail test that FAILS the suite the moment the bug is accidentally (or deliberately) fixed — no silent drift either direction | `tests/unit/test_preexisting_problems.py` (note: under `tests/unit/`, not `tests/` root) | `416b9e18` era |
| Import-linter **5 contracts are BLOCKING** in CI and pre-commit; `grimp` pinned `>=3.3,<3.4` because 3.4+ Rust-panics on Windows with this codebase | `.importlinter`, `pyproject.toml` lines 49–50, CI `lint-imports` step | remediation campaign |
| Config JSON keys ≡ `Model` enum values exactly; builders never resolve aliases | drift tests (see 2026-06-23 entry) | repeated drift episodes |
| ~~`_aggregate`: median for 3+ self-consistency runs~~ — **REOPENED, not settled** (`services/evaluator.py` fixed, but `container.py` wires the unfixed `application/evaluator.py`; see the 2026-06-25 entry above) | `tests/unit/test_bug_scan.py` (covers only the fixed class) | `e863f0c8` |
| Completion checks gate on **status**, never result truthiness | ledger + `416b9e18` | P5 |
| engine.py = Mediator only; no new logic there; no new root-level `orchestrator/*.py` | CI `check_new_root_files.py --baseline origin/master`, contracts | ARCH-AUDIT-V2 |
| Old VerbalizedSamplingPipeline stays retired | `docs/VERBALIZED_SAMPLING_ANALYSIS.md` | VS retirement |
| Never weaken a gate to pass (no coverage-floor lowering, no contract exemptions, no xfail-to-green, no allowlist expansion) | policy — full text in `orchestrator-change-control` | campaign doctrine |

---

## Stalled / dead branches (as of 2026-07-07)

From `git branch -a`. "Why stalled" is **inference from commit messages** unless noted.
Local branches marked ✓merged are fully contained in `master` (via `git branch --merged master`).

| Branch | Tip (date) | What it attempted | State |
|---|---|---|---|
| `arch/phase-abc` | `36b67d2c` (06-24) | Remediation phases A/B/C staging + CI greening | NOT merged as branch; work landed on master via cherry-picks (inference: duplicate C.1/C.2 hashes) |
| `chore/ci-health` | `66d87241` (06-20) | Test-suite health: stale mocks, skip-vs-xfail hygiene | ✓ merged |
| `chore/lint-format` | `17b45d11` (06-18) | Website sanitizer fixes + lint | ✓ merged |
| `feat/supervisor-spine` | `f6bdaa72` (06-18) | Supervisor implementation **plan** (docs) | ✓ merged; feature itself not built |
| `feat/supervisor-impl` | `66d87241` (06-20) | Supervisor implementation | ✓ merged tip, but tip is a test-health commit — implementation stalled (inference) |
| `fix/website-image-and-assembly` | `8c90d1ea` (06-18) | Website image-response parsing + HTML assembly | ✓ merged |
| `origin/docs/architecture-audit-and-refactoring-plan` | `519ac4ca` (06-04) | Early audit/refactor plan | superseded by ARCH-AUDIT-V2 campaign (inference) |
| `origin/fix/rate-limiter-async-lock-state-blocking-io` | `321547ac` (03-26) | Prevent concurrent `_get_conn()` bypass before state migration completes | stalled 3+ months; verify whether fix landed elsewhere before reviving |
| `origin/release/v5.1` | `4ce7c25f` (03-26) | v5.1 release line + paradigm-shift security fixes | dead — repo is v6.0.0 |
| `origin/claude/*` (5 branches) | 02-20 → 04-02 | Agent worktree sessions (T1-B DependencyResolver wiring, conftest fixes, hook stubs, test assertions) | abandoned worktree branches (inference); mine `c388a531` before re-attempting DependencyResolver-in-engine work |

Before reviving any stalled branch: `git log master..origin/<branch> --oneline` to see
what is genuinely unmerged, and check whether the chronicle above already covers why it
stopped.

---

## How to add a new entry to this chronicle

Add at the **top** of The Chronicle (newest first). Template:

```markdown
### YYYY-MM-DD — <one-line title: what broke or what was decided>

- **Symptom:** what an operator/developer actually observed (error text, hang, silence).
- **Root cause:** the mechanism, one level deeper than the symptom.
- **Fix / decision:** what changed, or what was decided and why alternatives lost.
- **Evidence:** `<commit-hash>` "<subject>" (date) — verify with
  `git log -1 --format='%h %ad %s' --date=short <hash>` BEFORE writing it here.
  Add file paths and the locking test if one exists.
- **Status:** SETTLED | FIXED | OPEN | RETIRED.
- **Lesson (optional):** only if it generalizes to a failure class.
```

Rules for entries:
1. No entry without a verified commit hash or a checked-in document path.
2. If the incident closes a debate, also add a row to "Settled decisions".
3. If it retires code, state explicitly what must NOT be resurrected.
4. Dead ends and reverts are first-class entries — the whole point is that the next
   person learns the path was already walked.
5. Label inference as inference.

---

## Provenance and maintenance

Verified 2026-07-07 on branch `feat/response-healing`. Re-verification one-liners:

| Claim | Re-verify with |
|---|---|
| Any commit hash cited above | `git log -1 --format='%h %ad %s' --date=short <hash>` |
| engine.py line count (1250 as of 2026-07-07) | `(Get-Content orchestrator\engine.py | Measure-Object -Line).Lines` |
| `USE_PROVIDER_SORTING` still dead | `git grep -n "use_provider_sorting" -- "*.py"` (should hit only config declarations) |
| grimp pin still present | `Select-String -Path pyproject.toml -Pattern grimp` |
| xfail ledger location | `Test-Path tests\unit\test_preexisting_problems.py` |
| HITL gate test count | `Select-String -Path tests\unit\test_hitl_gate.py -Pattern "def test" | Measure-Object` |
| Branch table freshness | `git branch -a` + `git branch --merged master` |
| VS retirement doc | `Test-Path docs\VERBALIZED_SAMPLING_ANALYSIS.md` |
| Stress test S2/S6/S7 still failing/documented | CLAUDE.md "Known Limitations" section |

Volatile facts most likely to drift: engine.py line count (demolition active), the
stalled-branch table, and anything on `feat/response-healing` before it merges.
