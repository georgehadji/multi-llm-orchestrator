---
name: orchestrator-external-positioning
description: Load this BEFORE writing anything that leaves the repo as a claim about the project's capabilities — a README edit, a paper/preprint, a release note, a blog post, a "beyond SOTA" pitch, a comparison table, or an answer to "is this novel / is this proven / can we claim X publicly". Also load when auditing an existing external claim for overclaim risk. Covers: the honest capability inventory (what's implemented+tested vs aspirational, verified 2026-07-08), the known-art vs candidate-novel table, the oversell ledger (documented incidents including the ponytail README overclaim and the Verbalized Sampling non-paper-faithful incident), the four-part claim-discipline checklist (repro command + number + baseline + date/hash), the reproducibility standard, and open release-hygiene items (version-string drift, .env-with-live-keys-in-repo-dir). Symptom keywords: "can we say this is novel", "write the README claim", "draft a release note", "is this SOTA", "external positioning", "paper claim", "reproducibility", "overclaim", "what can we prove".
---

# Orchestrator External Positioning

Governs anything this project says about itself to an audience outside the repo:
READMEs, papers, release notes, comparison tables, "beyond SOTA" pitches. The
rule is simple and non-negotiable: **every external claim is either backed by
a verifiable artifact in this repo, or it is labeled unproven.** There is no
third option. This skill exists because the project has already shipped one
verified oversell (ponytail, §3) and one claim that quietly stopped being true
without the docs catching up (Verbalized Sampling, §3) — both are the
canonical cautionary tales cited below.

Date-stamp: all "implemented/tested" verdicts below verified 2026-07-08 against
commit `72e13285d68da75b600c3aa723300e7ae9dad54a` (branch `feat/response-healing`).
Re-verify before reusing — see §7 Provenance.

## When NOT to use this skill

- Deciding whether a change is *allowed* into the repo (gates, coverage, contracts) → `orchestrator-change-control`.
- Understanding *why* a mechanism is designed a certain way (routing theory, dual budget, response-healing internals) → `llm-orchestration-reference`.
- Writing internal docs/codemaps (not external-facing) → `orchestrator-docs-and-writing`.
- Deciding what to work on next among the hard, unproven problems → `orchestrator-research-frontier` and `orchestrator-hardest-problems-campaign`.
- Designing an experiment/benchmark methodology to actually generate the missing evidence → `orchestrator-research-methodology` and `orchestrator-proof-and-analysis-toolkit`.
- Chronicle of past incidents (for narrative/rationale, not for claim-vetting) → `orchestrator-failure-archaeology`.
- Verifying whether a given test suite actually proves what it claims to prove → `orchestrator-validation-and-qa`.

---

## 1. Honest capability inventory

For each headline claim in `README.md`, verified by grepping for the
implementation module and a corresponding test file. "Tested" means a test
file exists that exercises the mechanism, not that it's covered by CI's
blocking gate — check `orchestrator-validation-and-qa` for which markers
actually run in CI.

| Claim (README location) | Implemented? | Test evidence | Verdict |
|---|---|---|---|
| Multi-provider routing (`ROUTING_TABLE`/`FALLBACK_CHAIN`, `models.py`) | Yes — `orchestrator/model_routing.py`, `orchestrator/planner.py` | `tests/unit/test_vfm_routing.py` | **Implemented + tested** |
| VFM / free-tier-first routing (README "Dual Budget" area + routing config) | Yes — routing config prefers `:free` OpenRouter variants per task type | `tests/unit/test_vfm_routing.py` (locks the routing table) | **Implemented + tested** |
| Dual budget (`Budget` per-run + `BudgetHierarchy` cross-run, README line 63) | Yes — `models.py`/`budget.py` (per-run), `cost.py` (`BudgetHierarchy`) | `tests/test_budget_hierarchy_persistence.py` | **Implemented + tested** |
| Resume capability (`--resume <project_id>`, README line 31) | Yes — `resumption_service.py` | `tests/integration/test_resume_golden_path.py`, `tests/integration/test_resumption_service.py` | **Implemented + tested**. Resume-detection heuristic (file mtime) is a documented known limitation — see CLAUDE.md "Known Limitations" — don't claim it's robust without qualifying that. |
| Response-healing (`USE_RESPONSE_HEALING`, server-side JSON repair) | Yes — `orchestrator/config.py`, `orchestrator/infrastructure/llm_client.py` | `tests/unit/test_response_healing.py` | **Implemented + tested**, but flag defaults `false` (per `orchestrator-config-and-flags`) — don't claim it's the default behavior in any external artifact. Say "opt-in" explicitly. |
| Design Quality Gates / "taste-skill" (README §"Design Quality Gates", `ORCH_TASTE_SKILL_ENABLED`) | Yes — `orchestrator/design/taste_skill_injector.py`, `taste_skill_loader.py`, `taste_skill_service.py`, wired through `engine.py` → `container.py` → `pipeline_executor.py` → `context_enricher.py` | `tests/integration/test_taste_skill_prompt_injection.py`, `tests/unit/design/test_taste_skill_*.py`, `tests/unit/design/test_redesign_rubric.py`, `tests/unit/quality/test_design_validators.py`, `tests/unit/test_config_design_dials.py` | **Implemented + tested**. This is the best-evidenced headline claim in the README — dial knobs (`ORCH_DESIGN_VARIANCE`, `ORCH_MOTION_INTENSITY`, `ORCH_VISUAL_DENSITY`) each have a dedicated test. |
| Ponytail anti-over-engineering (README §"Anti-Over-Engineering (ponytail)", lines 133-159) | **No.** `ponytail`/`PONYTAIL`/`Ponytail` has **zero** matches anywhere under `orchestrator/*.py` (verified `grep -rli ponytail orchestrator/ --include="*.py"` → no results; only a stale `__pycache__/persona_modes.cpython-312.pyc` filename false-positive from an unrelated module). The README's `## Anti-Over-Engineering (ponytail)` section describes it as something "the orchestrator integrates" | None in `orchestrator/` | **Pure aspirational / documented overclaim.** See §3 Oversell Ledger item (a). |

### What "persona" actually is (don't conflate with ponytail)

`orchestrator/persona.py`, `orchestrator/persona_modes.py`,
`orchestrator/agents/persona.py`, `orchestrator/agents/persona_modes.py` exist
and are wired into `engine.py`, `engine_core/engine_deps.py`,
`engine_core/orchestration_facade.py`, `engine_core/service_collection.py`,
and the MCP server. This is the **"Mnemo Cortex" persona system**
(STRICT / CREATIVE / BALANCED / CUSTOM behavior modes) — a real, tested,
unrelated mechanism. It is not ponytail and does not implement YAGNI/laziness
enforcement. If someone points at `persona.py` as evidence ponytail is wired
in, that is a category error — correct it.

Where ponytail *does* exist in this repo: as Claude-Code-harness tooling under
`.claude/skills/ponytail*` (coding-assistant skills invoked by a human/agent
session, not by the orchestrator's own runtime), and as an unwired prototype
package at `packages/orchestrator-persona/src/orchestrator_persona/persona.py`
+ `persona_modes.py` (note: this is a *separate* prototype package, distinct
from the wired `orchestrator/persona.py` module above — confirm which one any
future ponytail-integration work would extend before assuming they're the
same code).

---

## 2. Known-art vs candidate-novel table

**Default posture: everything below is "candidate" until benchmarked against
a real, named baseline with a published number.** Nothing in this project has
external validation (no published benchmark run, no third-party replication,
no comparison table with citations) as of 2026-07-08. Don't let a mechanism's
internal sophistication be mistaken for external proof of novelty.

| Mechanism | What it does | Known-art or candidate? | Evidence it's real (not just claimed) | External validation? |
|---|---|---|---|---|
| Adversarial evaluator / `CompletionJudge` maker-checker | Separate judge-role LLM scores completions; maker/checker split to reduce self-grading bias | Candidate — the maker-checker pattern itself is known-art (LLM-as-judge literature); this implementation's specific scoring/aggregation is unbenchmarked | `orchestrator/services/completion_judge.py`, referenced in `orchestrator/services/autonomy_costs.py` | **None.** No published comparison against single-pass self-grading or a human-labeled eval set. |
| 2-pass self-consistency + conservative aggregation | Runs evaluation twice, uses median for 3+ runs (fixed from an earlier bug that discarded runs 2..N — see `orchestrator-failure-archaeology` commit e863f0c8) | Known-art technique (self-consistency is a published pattern), candidate as applied here | `EvaluatorService._aggregate` | **None.** No ablation showing it actually improves score reliability on this project's tasks; the fix corrected a bug, it didn't prove the technique's value. |
| Verbalized Sampling (VS) pipeline | Meant to sample diverse candidate outputs per the VS paper technique | **Retired / not a claim anymore** — see §3(b) | `docs/VERBALIZED_SAMPLING_ANALYSIS.md` | N/A — do not cite this as a capability at all going forward. |
| MAP-Elites seeding | Quality-diversity search over solution variants | Candidate — MAP-Elites itself is decades-old known-art from evolutionary computation; using it to seed orchestrator outputs is the (unbenchmarked) candidate-novel part | `orchestrator/engine_core/stages/map_elites.py`, `orchestrator/crosscutting/config.py` | **None.** No comparison against random/greedy seeding. |
| VFM free-tier-first routing | Prefers `:free` OpenRouter model variants per task type before paid tiers | Candidate-novel as a *cost* strategy (routing-by-quality is known-art; routing-by-free-tier-availability-first is less commonly documented) | `tests/unit/test_vfm_routing.py` locks the table; see `llm-orchestration-reference` for the doctrine | **None.** No published cost/quality frontier chart against a baseline (e.g., always-premium or always-cheapest-paid). |
| Response-healing (server-side JSON repair via OpenRouter plugin) | Repairs malformed structured-output JSON before it reaches the app | Known-art (JSON repair / retry-on-parse-failure is standard practice) | `tests/unit/test_response_healing.py` | N/A — not being positioned as novel, just useful. Fine to describe factually; don't inflate to "novel resilience technique." |

**Rule for any of the above appearing in a paper/README/pitch:** state the
mechanism, state that it is unvalidated externally, and do not use words like
"beyond SOTA," "state-of-the-art," or "proven" without satisfying the
Claim-Discipline Checklist in §4 for that specific claim.

---

## 3. Oversell ledger — concrete documented incidents

This is the living list of times this project's *external-facing* claims
outran its *internal* reality. Each entry is closed only when the doc/claim
is fixed to match the code, or the code is fixed to match the claim — not
when someone stops noticing.

### (a) Ponytail README overclaim — OPEN as of 2026-07-08

`README.md` lines 133-159 (`## Anti-Over-Engineering (ponytail)`) states:
"The orchestrator integrates the **ponytail** extension, a specialized agent
mode that forces the laziest, simplest, and most minimal solution that
actually works." This is written as a runtime capability of the orchestrator
itself.

**Verified reality:** `grep -rli "ponytail" orchestrator/ --include="*.py"`
returns **zero** matches (only an unrelated `persona_modes.cpython-312.pyc`
filename false-positives on the substring, not ponytail itself). Ponytail
exists only as:
1. Claude Code harness skills at `.claude/skills/ponytail`, `ponytail-audit`,
   `ponytail-help`, `ponytail-review` — these operate on the *developer's
   coding session*, not on orchestrator-generated output.
2. An unwired prototype at `packages/orchestrator-persona/src/orchestrator_persona/`
   (`persona.py`, `persona_modes.py`) that is not imported by any code under
   `orchestrator/`.

**Impact if uncorrected:** any external reader (paper reviewer, prospective
user reading the README, a benchmark comparison) would reasonably conclude the
orchestrator enforces anti-over-engineering on its own generated code at
runtime. It does not. The Design Quality Gates / taste-skill system (§1) is
real and tested and *could* have been the actual load-bearing claim — this
looks like a case of a real internal-tooling feature (Claude-Code ponytail)
getting described as if it were a product feature (orchestrator runtime
behavior).

**Required fix (not performed by this skill — flag to whoever owns
`orchestrator-docs-and-writing` / the README):** either (1) rewrite the
section to accurately describe ponytail as developer-tooling used *while
building* the orchestrator, not a runtime capability of it, or (2) actually
wire a ponytail-equivalent check into the generated-output pipeline (e.g., as
part of `taste_skill_service.py` or a new validator in `validators.py`) and
then the claim becomes true. Until one of those happens, do not cite "ponytail
integration" as an orchestrator capability in any external artifact.

### (b) Verbalized Sampling — "paper-faithful" claim, found false — CLOSED (retired)

Earlier docs/positioning described the orchestrator's Verbalized Sampling (VS)
pipeline as implementing the VS paper technique. `docs/VERBALIZED_SAMPLING_ANALYSIS.md`
found two concrete faithfulness breaks:
- **G1 — never invoked (dead code).** `ara_execution_strategy.py:41-59`'s
  `default_methods`/`retry_methods` maps task types only to
  PERSUASION_DEFENSE / SOT / JURY / MULTI_PERSPECTIVE / DEBATE / COVE — nothing
  maps to `VERBALIZED_SAMPLING` or `BRAINSTORMING`. Per
  `ARA_IMPLEMENTATION_PLAN.md:12`, ARA methods are "registered in
  `PipelineFactory`, but none are wired into the core execution path."
- List-level prompt construction and inverted probability handling that
  diverge from the paper's method (see the analysis doc for the specifics).

**Resolution:** the analysis doc documents retirement of the VS-paper-fidelity
claim. The real residual value identified was reuse of VS-adjacent code for
MAP-Elites seeding, synthetic/test data generation, and retry tail-escape —
*not* "we implement the VS paper." **This is the canonical cautionary tale for
this skill: a specific, checkable claim ("paper-faithful") was made, someone
actually checked it against the paper, and it was false.** Every claim in §4's
checklist exists to prevent a repeat of this exact failure mode.

### Ledger discipline

New entries append here. Format: `(letter) <claim> — OPEN/CLOSED as of <date>`,
with verified reality, impact, and required fix. Do not delete closed entries
— they're the evidence the discipline works.

---

## 4. Claim-discipline checklist — required before ANY external claim ships

No claim (README bullet, paper sentence, release note line, tweet, pitch deck
slide) about this project's capabilities, performance, or novelty ships
without **all four** of the following. If you can't produce all four, the
claim is not ready — soften it to "candidate"/"unvalidated" or cut it.

1. **Reproduction command.** An exact, copy-pasteable command that reproduces
   the claim's evidence from a clean checkout (or points to a script under
   `orchestrator-proof-and-analysis-toolkit` that does). "Trust me, I ran it"
   is not a reproduction command.
2. **A number.** Not "fast," "cheap," "high-quality" — an actual measured
   value (latency, $ cost, score, pass rate, token count).
3. **A named baseline it's compared against.** "Better than X" requires
   stating what X is and how it was run under the same conditions. A number
   with no comparison point is not evidence of an advantage.
4. **Date + commit hash.** Every measured claim decays — model behavior,
   OpenRouter pricing/availability, and this codebase all change. Stamp the
   commit hash (`git rev-parse HEAD`) and date the measurement was taken so a
   reader can judge staleness.

**Template for a compliant claim:**

> "On task type X, response-healing (`USE_RESPONSE_HEALING=true`) reduced
> JSON-parse failures from N% to M% across K runs, vs. the same K runs with
> the flag off, measured on commit `<hash>` (2026-0X-XX). Reproduce:
> `pytest tests/unit/test_response_healing.py -v` plus `<benchmark script
> path>`."

Compare against the ponytail incident (§3a): the claim had none of the four —
no repro command, no number, no baseline, no date/hash tied to when ponytail
was actually verified present. That absence is exactly why it went
unnoticed.

---

## 5. Reproducibility standard

Any external artifact citing a run, benchmark, or comparison must disclose:

| Disclosure | What to state | Where the source of truth lives |
|---|---|---|
| **Pinned environment** | Python version (3.10 min, 3.12 primary per CLAUDE.md), install command (`pip install -e ".[dev]"` + which extras), OS (Windows dev vs ubuntu-latest CI — behavior can differ, e.g. BOM/encoding issues per `orchestrator-build-and-env`) | `orchestrator-build-and-env` |
| **Seed / temperature / reasoning disclosure** | Which phase-policy settings were active (Decompose/Critique/Evaluate use reasoning+HIGH+temp 0.1-0.2; Creative uses 0.8; Extract uses 0.0) — a claim run under Creative-phase temperature is not reproducible if rerun under Decompose defaults | `llm-orchestration-reference` (owns phase_policy doctrine); source: `orchestrator/domain/phase_policy.py` |
| **Cost disclosure** | Actual $ spent for the run, which budget mode was active (per-run `Budget` vs cross-run `BudgetHierarchy` vs both), and which models were actually routed to (not just the routing table's intent — VFM free-tier routing can silently change which model serves a request run-to-run based on availability) | `llm-orchestration-reference` (dual-budget model), `orchestrator-config-and-flags` (routing table) |
| **Cache-state disclosure** | Whether the response DiskCache (48h TTL, `$0` on hit) was warm or cold. A warm cache produces a $0/instant-response number that is not representative of a cold-cache run, and a warm cache **defeats eval self-consistency** (repeated "runs" against a cache hit return the identical cached response, not independent samples) — state explicitly whether self-consistency numbers were measured cache-cold | `orchestrator-validation-and-qa` (owns cache-state verification practice) |

**Do not publish a benchmark number without stating all four.** A number
with unstated cache-state is close to meaningless for self-consistency claims
specifically — see the cross-reference above.

---

## 6. Release hygiene

Verified 2026-07-08 against commit `72e13285d68da75b600c3aa723300e7ae9dad54a`:

| Item | Status | Detail |
|---|---|---|
| Version in `orchestrator/__init__.py` | `__version__ = "6.0.0"` (line 22) — this is the **authoritative** version; `pyproject.toml`'s `[tool.hatch.version] path = "orchestrator/__init__.py"` reads from here at build time | Confirm via `grep "__version__" orchestrator/__init__.py` |
| Version in `pyproject.toml` | `dynamic = ["version"]` — no static version string to drift, correctly delegates to hatch | Confirm via `grep "^version\|dynamic" pyproject.toml` |
| `[tool.bumpversion] current_version` | **`"1.2.0"`** — stale, inconsistent with the 6.0.0 actually shipped. This is a real drift item: if `bump2version`/`bumpversion` is ever run, it will compute the next version from `1.2.0`, not `6.0.0`, silently corrupting the next release's version number | **Open hygiene item.** Someone with write access to `pyproject.toml` should sync `[tool.bumpversion] current_version` to `"6.0.0"` before any bumpversion invocation, or remove the stale `[tool.bumpversion]` block if it's unused tooling. This skill does not fix it (not owned by external-positioning). |
| `.env` with live-looking API keys in repo working directory | **Present.** `.env` at repo root contains `OPENROUTER_API_KEY=` and `XAI_API_KEY=` with populated (not placeholder) values | `.env` **is** listed in `.gitignore` (lines 29-30: `.env`, `.env.*`) and `git ls-files .env` returns nothing — **not committed to git history**, so it is not a git-leak risk as verified. It is still a local-machine hygiene risk (any process/tool with filesystem read access on this machine can read live keys) and a copy-paste risk when zipping/sharing the repo directory outside git (e.g., emailing a folder, uploading to a non-git file share). **Do not attempt to fix or delete `.env` from this skill** — out of scope per this skill's mandate; flag to the repo owner. |
| `README.md` capability claims vs code | Mixed — see §1. Most are accurate and evidenced; ponytail section is not (§3a) | Re-run §1's grep commands before any external release that includes README content verbatim |

**Never** commit a real `.env`. If `.env` handling ever needs to change
(rotation, migration to a secrets manager, git-history scrub), that is a
security/ops task — route it through `security.md`'s Secret Management
protocol and `orchestrator-change-control`, not through documentation work.

---

## 7. Provenance and maintenance

Re-verify before reusing any claim from this skill — facts here decay as the
codebase changes:

```bash
# Re-check ponytail wiring status (should stay 0 until fixed)
grep -rli "ponytail" orchestrator/ --include="*.py"

# Re-check version consistency
grep -n "__version__" orchestrator/__init__.py
grep -n "current_version" pyproject.toml

# Re-check .env git-tracking status (should stay empty)
git ls-files .env

# Re-check test evidence for each README headline claim (adjust globs as code moves)
grep -rl "USE_RESPONSE_HEALING\|response_healing" tests/ --include="*.py"
find tests -iname "*vfm*"
grep -rl "BudgetHierarchy" tests/ --include="*.py"
grep -rl "resume_project\|resumption_service" tests/ --include="*.py"
grep -rl "taste_skill" tests/ --include="*.py"

# Re-check VS pipeline stays retired (should show no new wiring into ara_execution_strategy.py)
grep -n "VERBALIZED_SAMPLING\|BRAINSTORMING" orchestrator/reasoning/ara_execution_strategy.py

# Confirm current commit this skill's verdicts are pinned to
git log -1 --format="%H %ad" --date=short
```

If any of the above disagrees with §1-§6, this skill is stale — update the
relevant table row and bump the date-stamp at the top of this file before
trusting it for a new external claim.
