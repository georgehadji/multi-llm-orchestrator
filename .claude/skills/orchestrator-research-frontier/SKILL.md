---
name: orchestrator-research-frontier
description: Load this skill when scoping "beyond SOTA" / research-grade work on the Multi-LLM Orchestrator — i.e. when the question is not "is it broken" but "could this be a publishable or genuinely novel advance." Covers four research programs (cost-quality Pareto frontier, zero-silent-failure autonomy, generated-output quality ceiling, orchestration science) each with a verified current-state asset check, first concrete repo steps, and a falsifiable numeric milestone. Symptom keywords: "beyond state of the art", "is this novel", "research program", "Pareto frontier", "ablation study", "publishable result", "what's the open problem here", "VFM routing benchmark", "zero silent failure", "design quality gate", "blind eval scorecard", "self-consistency ablation". Everything here is CANDIDATE/OPEN, not a claim — pair with orchestrator-research-methodology for the evidence bar before asserting any result, and orchestrator-external-positioning before writing any of this up externally.
---

# Orchestrator Research Frontier

Date-stamped 2026-07-08. This skill is a map of four open research programs where this
project's actual assets (verified against the repo below, not asserted) could push past
the SOTA baseline of "single premium model, hope it works, no evaluation loop." None of
the milestones below have been run yet as of this writing — that is the point of a
frontier skill. If you run one and get a number, that result belongs in
`orchestrator-proof-and-analysis-toolkit` output or a dated doc, not silently folded
back into this file as if it were always true.

**Golden rule of this skill:** every "Asset" line was checked with Read/Grep against
the repo on 2026-07-08. If you're reading this later, re-run the one-line verification
commands in "Provenance and maintenance" before trusting a line here — several prior
sessions in this repo shipped stale claims (see `orchestrator-failure-archaeology`).

---

## When NOT to use this skill

- You need to fix something that is currently broken → `orchestrator-debugging-playbook`.
- You need the incident history / "has this been tried before" → `orchestrator-failure-archaeology`.
- You need to know WHY a mechanism (routing, budgets, evaluation) is designed the way
  it is, not whether it's novel → `llm-orchestration-reference`.
- You need the flag catalog to turn an experiment on/off → `orchestrator-config-and-flags`.
- You're about to run one of these experiments and need the rigor bar (sample size,
  what counts as evidence, how to report negative results) → `orchestrator-research-methodology`.
- You're about to write these results up for anyone outside this repo →
  `orchestrator-external-positioning` (claim-discipline checklist gates all external claims).
- You want the standing list of hardest live engineering problems (not research
  programs) → `orchestrator-hardest-problems-campaign`.
- You need scripts/harnesses to actually run an experiment →
  `orchestrator-proof-and-analysis-toolkit`.

## Ground rules that apply to all four programs

1. **Experiment flags default OFF.** Any new benchmarking/ablation flag you add must
   follow the existing `USE_*` / `ORCH_*` off-by-default convention in
   `orchestrator/config.py` (see `orchestrator-config-and-flags`). Never flip a
   production default to make a research result look better.
2. **No external claims until the checklist passes.** A milestone hit in a scratch
   script is not a claim. Before it appears in a README, blog post, or PR description,
   run it through `orchestrator-external-positioning`'s claim-discipline checklist.
3. **Evidence bar.** "Candidate" until `orchestrator-research-methodology`'s bar is met
   (repeat runs, stated sample size, negative results reported alongside positive ones,
   no cherry-picking the one green run out of five).
4. **Never weaken a gate to produce a nicer number.** Don't lower `--cov-fail-under`,
   don't relax an import-linter contract, don't quietly xfail a test that's
   inconvenient for the story. See `orchestrator-change-control`'s unwritten rules.

---

## Program 1 — Cost-quality Pareto frontier

### Why current SOTA fails
The default posture in most agent frameworks is "call the single most capable model for
everything." That's a single point, not a frontier — it can't answer "how much quality
do I actually buy per marginal dollar," and it silently overpays on tasks a cheap model
would nail.

### This project's asset (verified)
- `orchestrator/models.py::ROUTING_TABLE` + `orchestrator/config/routing.json` +
  `orchestrator/config/costs.json` implement **VFM (value-for-money) free-tier-first
  routing**: `tests/unit/test_vfm_routing.py` exists and is a live contract test. It
  asserts, per text `TaskType` (CODE_GEN, CODE_REVIEW, REASONING, WRITING,
  DATA_EXTRACT, SUMMARIZE, EVALUATE), that 2026 VFM/free-tier models are declared with
  a real `Model` enum member and a priced `costs.json` entry, and that routing tries a
  free/ultra-cheap model before escalating to premium. This is a **routing-order
  guarantee**, not a measured quality outcome — the test proves the frontier is wired,
  not that it's good.
- `orchestrator/cost.py::BudgetHierarchy` — cross-run Org→Team→Job budget caps,
  independent of the per-run `Budget` in `models.py`/`budget.py` (dual-budget system,
  see `llm-orchestration-reference`).
- `orchestrator/infrastructure/cache.py` (and `caching.py`) — response `DiskCache`
  with 48h TTL is the actual measured cost win per `cost-reduction-audit` memory
  (2026-06-25); confirmed live in the call path (unlike `use_provider_sorting`, which
  is a dead flag — verify with `orchestrator-config-and-flags`).
- `projects/*.yaml` — 28 real project specs already in the repo (e.g.
  `backend_rest_api.yaml`, `frontend_react_dashboard.yaml`,
  `scientific_nbody_symplectic.yaml`, `polytonic_ocr.yaml`) usable as a benchmark
  corpus without inventing synthetic tasks.

### First 3 concrete steps in this repo
1. Pick N≥10 of the existing `projects/*.yaml` specs spanning at least 3 categories
   (backend, frontend, scientific/analysis) and run each once through
   `python -m orchestrator --project <spec>` (or the equivalent programmatic
   `run_project()` call) with VFM routing on (current default), recording per-task
   `cost_usd` (from `APIResponse`) and the evaluator's quality score.
2. Re-run the same N tasks with routing forced to a single premium model baseline
   (temporarily override `ROUTING_TABLE` resolution in a throwaway script — do not
   edit `orchestrator/models.py` in place; this is a benchmark harness, not a product
   change) and record the same two numbers.
3. Plot cost (x) vs quality score (y) for both conditions — this is the Pareto curve.
   Store the harness script and raw output under
   `orchestrator-proof-and-analysis-toolkit`'s conventions, not ad hoc in repo root
   (root-file freeze — see `orchestrator-change-control`).

### Milestone (falsifiable)
A measured Pareto frontier — cost vs quality-score pairs for **N≥10 real tasks**,
VFM-routed vs single-premium-model baseline, both conditions run and plotted — not an
asserted "VFM is cheaper" claim. If the VFM curve does not dominate (lower cost at
equal-or-better quality) for a meaningful fraction of tasks, that is itself the result;
report it, don't discard it.

---

## Program 2 — Autonomy reliability (zero-silent-failure unattended runs)

### Why current SOTA fails
Long unattended agent runs commonly fail silently: a step errors, gets swallowed, the
loop continues on stale/wrong state, and the operator finds out hours later from bad
output rather than from an alert. "It didn't crash" gets mistaken for "it worked."

### This project's asset (verified)
- `orchestrator/hitl/` — `channel.py`, `gate.py`, `__init__.py` implement the
  **fail-closed `DecisionChannel`** (commit `11deb573`, verified via `git show --stat`:
  "replace silent auto-approval with fail-closed gate (FIX-1)"). Default channel is
  `FailClosedChannel` unless `ORCH_HITL_AUTOAPPROVE=true` is explicitly set — the
  legacy silent-approve escape hatch is now opt-in, not default. 11 unit tests cover
  channel paths per the commit message (re-verify count before citing).
- `orchestrator/application/unattended_guard.py` — `UnattendedGuard` class exists
  (confirmed via grep for `class UnattendedGuard`).
- `orchestrator/services/completion_judge.py` — `CompletionJudge` class exists
  (confirmed via grep for `class CompletionJudge`); this is the maker-checker pattern
  referenced in `llm-orchestration-reference` — read that skill for how it scores
  completion, not this one.
- `tests/test_preexisting_problems.py` — the `xfail(strict)` regression ledger
  discipline: known bugs are captured as strict-xfail tests so a silent "fix" (test
  starts passing without an explicit review) is caught by CI, not shipped quietly.

### First 3 concrete steps in this repo
1. Build a fault-injection harness (new script under a research/benchmark location,
   not root — see root-file freeze) that deliberately triggers N distinct failure
   modes against a running orchestrator session: network timeout mid-call, malformed
   JSON from a model, budget exhaustion mid-task, a `requires_approval=True` decision
   with no channel configured. For each, assert the failure surfaces as a detected/
   logged/raised event — not a silently-continued loop.
2. Measure detection rate = (failures surfaced) / (failures injected). This is the
   headline instrument for the whole program — build it before running long sessions.
3. Run M unattended hours (start small: M=2, then scale) with
   `ORCH_HITL_AUTOAPPROVE` unset (fail-closed default) and count silent failures —
   defined precisely as: a failure that occurred but produced no log entry, no raised
   exception, no HITL gate trigger, and no entry in whatever telemetry surface is
   wired (telemetry is a known partial-TODO per `orchestrator-architecture-contract` —
   confirm what's actually collecting before trusting "0 detected" as "0 occurred").

### Milestone (falsifiable)
X consecutive unattended hours with **zero undetected silent failures** — detected-and-
handled failures (a raised exception, a HITL block, a logged circuit-breaker trip) are
expected and fine; the failure mode being measured is specifically "something broke and
nothing recorded it." Start with a small X (e.g. 2 hours) and report the fault-injection
detection rate alongside it — a clean unattended run with an untested fault-injection
harness is not evidence of anything.

---

## Program 3 — Generated-output quality ceiling

### Why current SOTA fails
Generated code/websites/apps from most agent pipelines are recognizably AI-generated:
generic Tailwind-template look, boilerplate nobody asked for, no design point of view.
"Compiles and passes tests" is a much lower bar than "a senior engineer would ship this."

### This project's asset (verified — status corrected from prior assumptions)
- `orchestrator/safety/generated_output_scanner.py` exists and **is wired** into
  `orchestrator/output_organizer.py` (grep-confirmed) as a delivery gate — this part is
  real and live, not aspirational.
- **Design-quality validator is NOT live.** `tests/unit/quality/test_design_quality_validator.py`
  carries `pytestmark = pytest.mark.skip(reason="validate_design_quality removed
  during refactoring — re-implement when needed")` at the module level (verified by
  reading the file 2026-07-08). The function `validate_design_quality` was removed
  from `orchestrator.quality.validators` during a refactor and never reinstated. Every
  test in that file — banned-font detection, macrostructure stamp requirement,
  `VALIDATORS` dict registration, genre override — is currently a no-op. Do not claim
  this gate exists in production until it is re-implemented and this skip is removed.
- **Ponytail anti-over-engineering doctrine is blueprint-only.**
  `PONYTAIL_INTEGRATION_PLAN.md` (repo root) is a design document — grep for
  `ponytail`/`Ponytail`/`PONYTAIL` across `orchestrator/` returns **zero matches**.
  The persona/steering/quality-gate integration it describes has not been implemented
  in application code; it exists as `.claude/skills/ponytail*` tooling for the coding
  *assistant* (Claude Code), not as a runtime feature of the orchestrator itself. Don't
  conflate the two.

### First 3 concrete steps in this repo
1. Build a blind-eval scorecard: an LLM-judge (or human) rubric that scores a code/
   website artifact on the axes `PONYTAIL_INTEGRATION_PLAN.md` and the web
   design-quality rules already describe (hierarchy, intentional rhythm, no default-
   template look, etc.) without telling the judge which artifacts came from this
   orchestrator vs a baseline generator. Keep the rubric and scoring script in
   `orchestrator-proof-and-analysis-toolkit`'s territory.
2. Run the scorecard against whatever real generated artifacts already exist under
   `outputs/` (check what's actually there before assuming a corpus — a prior session's
   memory notes reference a generated dental-clinic website and other outputs; verify
   current contents with `Glob outputs/**` rather than trusting this line by date).
3. Ship the design-quality gate for real: re-implement `validate_design_quality` in
   `orchestrator/quality/validators.py` (or wherever it now belongs post-refactor),
   remove the `pytest.mark.skip`, and get all 5 tests in
   `tests/unit/quality/test_design_quality_validator.py` green under TDD (RED already
   exists — the tests are written, just skipped; this is close to free RED→GREEN work,
   not net-new design).

### Milestone (falsifiable)
Blind evaluators (human or LLM-judge, stated which) cannot distinguish orchestrator
output from senior-human work at rate **≥X%** on a held-out task set (state X and the
task-set size before running, not after — post-hoc threshold selection is not evidence
per `orchestrator-research-methodology`). Step 3 (un-skipping the validator) is a
prerequisite gate, not the milestone itself — a re-implemented validator with all tests
green is necessary infrastructure, not a quality-ceiling result.

---

## Program 4 — Orchestration science (novel decomposition/evaluation methods)

### Why current SOTA fails
Naive single-pass generation (one prompt, one output, ship it) has no self-checking
loop and no way to know if a decomposition strategy or an evaluation method is actually
buying quality versus just adding latency and cost.

### This project's asset (verified)
- `orchestrator/services/completion_judge.py::CompletionJudge` — maker-checker pattern
  (grep-confirmed to exist; read the file directly before citing its internals, this
  skill doesn't restate them — see `llm-orchestration-reference`).
- 2-pass self-consistency evaluation is real and live in
  `orchestrator/application/evaluator.py`: `consistency_runs: int = 2` default,
  `_consistency_runs` loop at line ~118, and the aggregation comment at line ~205
  ("Apply self-consistency: if delta > threshold, take the lower score") — cross-check
  against the `cross-project-learning`/`e863f0c8` fix noted in
  `orchestrator-failure-archaeology` (the aggregator used to discard runs 3+ and return
  `scores[0]`; now uses median for 3+ runs). Confirm current behavior by reading
  `evaluator.py` directly, since this is exactly the kind of drift-prone claim this
  skill warns about.
- HTN-style task decomposition — `orchestrator/application/decomposer.py` (present in
  the working tree per `git status`; read it directly for current shape before citing).
- `docs/VERBALIZED_SAMPLING_ANALYSIS.md` exists — a real, already-written **negative
  result**: the VS pipeline was judged dead code and not paper-faithful (list-level
  prompt, inverted probability vs the VS paper's design). This is citable as-is; the
  documented salvage value is MAP-Elites-style seeding, synthetic/test-data generation,
  and retry tail-escape — not the VS pipeline as originally built.
- `orchestrator/domain/phase_policy.py` — per-phase temperature/reasoning policy
  (Decompose/Critique/Evaluate = reasoning + HIGH effort + temp 0.1–0.2; Creative =
  0.8; Extract = 0.0), file confirmed present.

### First 3 concrete steps in this repo
1. Ablation harness: run the same N task set from Program 1 four ways — (a) full
   pipeline as-is, (b) self-consistency forced to `consistency_runs=1` (single pass,
   no aggregation), (c) decomposition disabled/flattened to one big task, (d) phase
   policy temperature/reasoning pinned to one flat setting instead of per-phase. Record
   quality score for each.
2. Compare effect sizes against literature baselines where findable (self-consistency
   sampling, MAP-Elites-style diversity seeding) — cite real papers if you find them
   during this step; do not invent citations.
3. Write up whichever single ablation shows the largest quality delta, and — per the
   evidence bar — report the smallest/negative deltas too, explicitly, in the same
   writeup. The VS negative result in `docs/VERBALIZED_SAMPLING_ANALYSIS.md` is the
   house style to match: "we tried X, it didn't work, here's why, here's the salvage
   value" is a valid and expected outcome, not a failure to hide.

### Milestone (falsifiable)
A mechanism (self-consistency, decomposition granularity, or phase-policy tuning) shown
to move final-task quality by **≥X points** in a controlled ablation (same task set,
only that mechanism toggled), with negative/null results for the other mechanisms
reported alongside in the same writeup — not cherry-picked.

---

## Provenance and maintenance

Re-verify before trusting any line in this file — it is a frontier map, not a fact
sheet, and every asset here has drifted before in this repo:

| Claim | Re-verification command |
|---|---|
| VFM routing test exists/still passes | `pytest tests/unit/test_vfm_routing.py -v` |
| Design-quality validator still skipped | open `tests/unit/quality/test_design_quality_validator.py`, check for `pytest.mark.skip` at module level |
| Ponytail still blueprint-only in orchestrator code | `grep -ril ponytail orchestrator/` (expect no matches) |
| Generated-output scanner still wired | `grep -n "generated_output_scanner" orchestrator/output_organizer.py` |
| HITL fail-closed default unchanged | `grep -n "ORCH_HITL_AUTOAPPROVE\|FailClosedChannel" orchestrator/hitl/*.py` |
| Self-consistency aggregation behavior | read `orchestrator/application/evaluator.py`, search `_consistency_runs` and the aggregation method |
| CompletionJudge / UnattendedGuard still present | `grep -rl "class CompletionJudge\|class UnattendedGuard" orchestrator/` |
| Project benchmark corpus size | `ls projects/*.yaml \| wc -l` (28 as of 2026-07-08) |
| use_provider_sorting still dead / DiskCache still the real win | see `orchestrator-config-and-flags` and `orchestrator-diagnostics-and-tooling` |

Last verified: 2026-07-08 against branch `feat/response-healing`. This skill makes no
claim about `master` or any other branch.
