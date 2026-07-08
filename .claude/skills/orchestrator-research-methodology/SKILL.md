---
name: orchestrator-research-methodology
description: The discipline that turns a hunch about this codebase into an accepted, defensible result — write a mechanism statement, predict numbers before running anything, get an adversarial pass, and either promote through change-control or retire with a document. Load this BEFORE writing a root-cause claim, an "improvement" PR description, or a research/analysis doc; BEFORE running an experiment or A/B; and whenever you catch yourself asserting "X caused Y" or "this makes it better" without a command whose output you predicted in advance. Symptom keywords — "I think the bug is", "this should improve", "let's try", "I ran it and it looks better", "root cause is probably", "the fix works" (with no falsifiable prediction attached), "should we add a flag for", "is this worth keeping". Not a testing/gates skill (see orchestrator-change-control, orchestrator-validation-and-qa) and not the incident log itself (see orchestrator-failure-archaeology) — this is the epistemic process that produces entries for those.
---

# Research Methodology

How a hunch about this repo becomes an accepted result — or a documented retirement. This is
process, not a specific bug or flag. If you're debugging a live incident, use
`orchestrator-debugging-playbook` instead; come back here when you're ready to write down *why*
you believe your fix is correct.

Date-stamp: verified against repo state 2026-07-08 on branch `feat/response-healing`.

## When NOT to use this skill

- Debugging a live, currently-broken thing → `orchestrator-debugging-playbook` (symptom-ranked
  triage, fastest discriminating experiment).
- Looking up whether a flag/config entry is wired → `orchestrator-config-and-flags` or
  `orchestrator-diagnostics-and-tooling` (measurement scripts).
- You already have a result and need to know which gate blocks merging it →
  `orchestrator-change-control`.
- You want the list of past incidents/retirements themselves, not the process that produced
  them → `orchestrator-failure-archaeology`.
- You need domain theory (why routing/evaluation/caching work the way they do) to form your
  hypothesis in the first place → `llm-orchestration-reference`.
- You're deciding whether a big speculative idea is even worth an experiment →
  `orchestrator-research-frontier` (candidate backlog) or `orchestrator-hardest-problems-campaign`.

## 1. The evidence bar

**One mechanism must explain ALL observations — including the ones that don't fit — and it must
survive an adversarial pass assigned to break it.** This project has been burned by claims that
explained the happy path but not the edge cases (see §4 for real examples). A root-cause or
improvement claim is not accepted because it sounds plausible or because a metric moved after
you changed something; it is accepted because nobody who tried to disprove it could.

### Required artifact before you claim anything

Before you write "the bug is X" or "this change improves Y" in a commit message, PR, or doc,
produce this in your working notes (doesn't need to be a separate file for small claims — inline
in the PR description is fine, but it must exist somewhere reviewable):

```markdown
## Mechanism statement
<One or two sentences: the causal chain, precise enough to predict a number (see §2).>

## Observations
- [x] Observation A — explained: <how>
- [x] Observation B — explained: <how>
- [ ] Observation C — does NOT fit: <why not, and what that implies>

## Adversarial pass
Assigned to: <a second agent/model instance, or a human reviewer who did not write the fix>
Result: <survived / broke the claim — if broke, the mechanism statement is wrong, not the evidence>
```

The "observations that don't fit" row is not optional decoration — it's the part that actually
filters false claims. A mechanism that quietly ignores an inconvenient log line or test result is
not yet a mechanism, it's a story.

### Who plays adversary

This project already has a maker/checker pattern for exactly this purpose — reuse the concept
even when the "claim" is a debugging conclusion, not generated code:

- **`CompletionJudge`** (`orchestrator/services/completion_judge.py`) — enforces
  `judge_model != generator_model` (`SameModelError` if violated) so the model that produced a
  claim never grades its own claim. Verdict is `PASS`/`FAIL`, fail-closed on any parse error or
  exception (`JudgeVerdict.FAIL`, docstring: "prevents the generator from grading its own work
  (cognitive surrender)"). See `llm-orchestration-reference` §"maker-checker (`CompletionJudge`)"
  for the full mechanics.
- For a debugging claim you're making by hand (not through the pipeline), the equivalent is:
  hand your mechanism statement + the observation list to a **different** Claude Code session, a
  different agent type, or a human colleague, and explicitly ask them to find the observation it
  doesn't explain. Do not ask "does this look right?" — ask "what breaks this?"
- A claim that only you have reasoned about, with no adversarial pass logged, is a hunch, not a
  result. It can still go behind an experiment flag (§3) — it just can't be promoted yet.

## 2. Hypothesis predicts numbers before running anything

Template — fill this in **before** you touch a keyboard to run the command:

```
If H is true, then running `<exact command>` will output `<X>` (± `<δ>` if numeric, or an exact
string/exit-code if discrete).
If instead `<Y>` comes out, H is dead — not "weakened," dead.
```

A prediction that can't be falsified by a specific command's output is not a hypothesis test,
it's vibes with extra steps. "Should improve latency" is not a prediction. "P50 latency for
`decompose()` on the 12-task fixture drops from 340ms±20 to under 250ms" is.

### Worked example A — real bug, real prediction (retroactive reconstruction)

This is how the `EvaluatorService._aggregate` bug (commit `e863f0c8`,
`orchestrator/services/evaluator.py`) should have been framed, and is the shape to imitate:

- **Hypothesis (H):** `_aggregate` silently discards consistency runs beyond the first two —
  for `consistency_runs >= 3` it falls through to returning `scores[0]` instead of aggregating
  all N runs.
- **Prediction:** If H is true, `EvaluatorService(None, None, lambda t: [])._aggregate([0.2, 0.9,
  0.9], "t")` returns `0.2` (the first element), not a value reflecting the other two runs.
- **Falsifier:** If instead it returns something in `[0.8, 0.9]` (a mean or median of all three),
  H is dead.
- **Verified now (post-fix):** confirmed by reading `tests/unit/test_bug_scan.py:100-104`
  (`TestAggregateNeverDiscardsRuns.test_three_runs_not_reduced_to_first`) — the fixed
  implementation returns `0.9` (median of `[0.2, 0.9, 0.9]`), with the test's own comment noting
  "The fixed bug: `[0.2, 0.9, 0.9]` previously returned `0.2` (`scores[0]`)." The commit message
  for `e863f0c8` independently confirms the pre-fix behavior. This is the pattern: a proactive
  invariant-scan test suite (`tests/unit/test_bug_scan.py`, added in that same commit — 77
  cross-cutting invariant tests) is what turned a vague worry ("does aggregation handle N runs
  right?") into a numeric, falsifiable prediction — and caught a real bug other correctness tests
  had missed because they never exercised `consistency_runs >= 3`.

### Worked example B — config drift, prediction verified live today

- **Hypothesis (H):** Some `orchestrator/config/{costs,fallbacks,routing}.json` keys/values do
  not exactly equal a `Model` or `TaskType` enum `.value`, so those entries are silently dropped
  by the config builders (see `orchestrator-config-and-flags` for the mechanism).
- **Prediction:** If H is true, running
  `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py`
  reports a non-empty "HARD DRIFT" list. If H is false, it reports zero drifted entries.
- **Ran it 2026-07-08** (this session, on `feat/response-healing`, cold — no prior run this
  session): the script reported **5 hard-drift entries**, all confirmed real:
  ```
  HARD DRIFT — 5 config entr(ies) silently dropped by models.py:
    costs.json key not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'
    costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4'
    costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-opus-4.1'
    costs.json key not a Model value (SILENTLY DROPPED): 'anthropic/claude-sonnet-4'
    fallbacks.json['qwen/qwen3-coder'] value not a Model value (SILENTLY DROPPED): 'qwen/qwen3.6-flash'
  ```
  plus one informational note (`internal/nano-banana-2` has no `costs.json` entry). **H
  confirmed as of 2026-07-08** — this is a currently-open drift, not a historical one; if you're
  reading this later, re-run the script (it's cheap, seconds, no API calls) before trusting this
  number, since config files churn.

Both examples share the shape that makes them worked examples and not just anecdotes: a command
you can literally paste, a specific expected output, and an explicit "if X instead, the
hypothesis is dead" clause.

## 3. Idea lifecycle

```
hunch
  ↓  (optional but recommended: check it's not a re-fight of a settled question)
chronicle check — orchestrator-failure-archaeology "Settled decisions" section
  ↓
experiment behind a flag — default OFF, USE_* naming (orchestrator-config-and-flags)
  ↓
measured result — numbers pre-registered per §2, not chosen after seeing the data
  ↓
        ┌───────────────────────┴───────────────────────┐
        ▼                                                ▼
  PROMOTION                                        RETIREMENT
  through orchestrator-change-control              documented, not deleted silently
  (TDD RED→GREEN, gates, docs)                     (see §3.1 below)
```

### 3.1 Retirement is a successful outcome — the Verbalized Sampling precedent

`docs/VERBALIZED_SAMPLING_ANALYSIS.md` (verified present, 18.1KB, dated 2026-06-15) is the
canonical example of what a *good* retirement looks like. Its actual finding, verified by reading
the doc:

- The repo already had a `VerbalizedSamplingPipeline`
  (`orchestrator/reasoning/ara_pipelines.py:3094`) — but it was **dead code**: no `TaskType`
  mapped to it in `default_methods`/`retry_methods`
  (`orchestrator/reasoning/ara_execution_strategy.py:41-59`), so it was never invoked from the
  main pipeline.
- Worse, the implementation was **not faithful to the source paper** (arXiv:2510.01171v3): it
  used a *list-level* prompt ("write 5 things") instead of the paper's *distribution-level*
  prompt ("write 5 things, each tagged with its probability"). The paper's own Claim 2 says
  list-level prompts recover only a uniform distribution at best — the exact failure mode the
  existing code had, by construction.
- The doc does **not** recommend reviving the pipeline as-is. It identifies three narrower,
  paper-grounded wins that are cheap and isolated (fix MAP-Elites seeding, use VS for synthetic
  test-data generation, VS-tail on the self-consistency retry path) — each explicitly scoped as
  its own future experiment behind its own flag, not a blanket "turn VS back on."

The lesson to copy: when a full investigation shows a feature was speculative, unwired, or wrong
relative to its own justification, **write the finding down with the mechanism and the
evidence**, even though nothing ships. The failure mode this prevents is someone re-discovering
"hey there's a `VerbalizedSamplingPipeline` sitting right there, why don't we wire it up" every
few months with no memory of why it was left alone. A retirement with a doc closes the loop; a
retirement by silent abandonment (comment out the code, never explain why) does not — the next
person burns another investigation cycle re-deriving what you already knew.

**Where retirements/dead-ends get indexed:** `orchestrator-failure-archaeology` — do not
duplicate its "Settled decisions — do NOT relitigate" and "Stalled / dead branches" sections
here; check them before starting new work in case your hunch is already a settled question.

## 4. Where good ideas came from historically (verify, don't invent)

Four repeatable sourcing patterns, each grounded in a real commit/artifact in this repo — use
these as templates for where to look for your *next* idea, not just history trivia:

1. **Proactive invariant bug-scan suites** found real bugs nobody was looking for. `e863f0c8`
   (`test(bug-scan): proactive invariant suite + fix _aggregate dropping runs`,
   `tests/unit/test_bug_scan.py`) added 77 tests asserting cross-cutting invariants over pure
   logic — `parse_score` output always in `[0,1]`, `_aggregate` never discards runs, `CronParser`
   never raises on garbage input, `BudgetHierarchy.remaining` never negative, etc. — and *in the
   same commit* found and fixed the `_aggregate` bug from Worked Example A. The pattern: write
   tests for properties that should always hold, not just for the happy path you're currently
   implementing; they catch bugs orthogonal to whatever you were originally working on.
2. **Post-incident generalization** — fixing one instance of an anti-pattern, then auditing for
   siblings. `11deb573` (`fix(hitl): replace silent auto-approval with fail-closed gate (FIX-1)`)
   fixed `HumanInTheLoop.request_decision` auto-approving every critical decision with only a
   warning log. The commit message explicitly frames this as one instance of "cognitive
   surrender baked into the default path" — the generalization step (documented in
   `orchestrator-failure-archaeology`) was auditing other decision/approval flows in the codebase
   for the same silent-fail-open shape, not just patching the one call site.
3. **External-tool-inspired docs** — a deliberate research sweep of competitor/adjacent tools,
   written up before any code changed. Verified present in `docs/`: `BASE44_INSPIRED_ENHANCEMENTS.md`,
   `BLACKBOX_INSPIRED_ENHANCEMENTS.md`, `BOLT_INSPIRED_ENHANCEMENTS.md`,
   `CREATE_XYZ_INSPIRED_ENHANCEMENTS.md`, `DYAD_INSPIRED_ENHANCEMENTS.md`,
   `LOVABLE_INSPIRED_ENHANCEMENTS.md`, `NEWLY_INSPIRED_ENHANCEMENTS.md`,
   `REPLIT_INSPIRED_ENHANCEMENTS.md`, `RETOOL_INSPIRED_ENHANCEMENTS.md`, `V0_INSPIRED_ENHANCEMENTS.md`
   (10 docs, 18KB–57KB each). Each is analysis-first: what does tool X do differently, does it
   map to something this project's architecture actually needs, before any implementation. Read
   one before writing your own before/after competitive-analysis doc to match the format.
4. **Cost-audit sweeps** — a focused pass whose only question is "where is money/tokens actually
   going, and does the mechanism we assume is saving cost actually run." This is how
   `use_provider_sorting` was found to be a **dead flag** (declared in config, never consulted on
   the call path) and the response `DiskCache` (48h TTL) was confirmed as the *actual* cost win —
   see `llm-orchestration-reference` and `orchestrator-config-and-flags` for the mechanism detail;
   the point here is the sourcing pattern: don't trust that a cost-saving feature is doing
   anything until you've traced the call path, the same way `check_config_drift.py` in Worked
   Example B doesn't trust that a config key survives to `models.py` until it's actually checked.

## 5. Experiment hygiene

Every experiment run whose result you intend to cite (in a PR, a doc, a claim to the user) must
disclose all of these, or the number is not trustworthy enough to act on:

| Disclosure | Why it matters here | Where to check |
|---|---|---|
| **Cache state** | Response `DiskCache` (48h TTL) means a "fresh" fast/cheap result may just be a cache hit, not the mechanism you think you're measuring. A warm cache also **defeats self-consistency** — repeated calls return the identical cached completion instead of N independent samples, silently degrading evaluation quality. | `llm-orchestration-reference` (cache theory), `orchestrator-config-and-flags` (`ORCH_CACHE_HOME`) |
| **Seed / temperature** | Per-phase temperature is policy-driven, not a fixed global (Decompose/Critique/Evaluate use low temp + reasoning; Creative phases use 0.8). Citing "the model said X" without stating which phase/temperature ran is not reproducible. | `llm-orchestration-reference` §phase policy, `docs/REASONING_AND_TEMPERATURE.md` |
| **Cost budget** | State the `$` spent and against which `Budget`/`BudgetHierarchy` scope (per-run vs cross-run — see CLAUDE.md "Dual-Budget System"). An experiment with no stated budget ceiling is not a controlled experiment. | `Budget` (`models.py`), `BudgetHierarchy` (`cost.py`) |
| **Model IDs used, verbatim** | OpenRouter `:free` variants and canonical vs. alias ids are not interchangeable for cost accounting — see the config-drift trap in §2 Worked Example B. State exact ids, not "the free tier model." | `orchestrator-config-and-flags`, `llm-orchestration-reference` |

### The hard rule: experiment code must never weaken a gate

This is `orchestrator-change-control`'s "Unwritten Rule 1: Never weaken a gate to pass it"
(verified present at that skill's SKILL.md, that exact heading) applied to research specifically:
an experiment flag may add a new, additive code path behind `USE_*` (default off), but it must
**never**:

- lower the coverage floor (`--cov-fail-under`) to make an experimental branch's tests pass,
- add an import-linter contract exemption to let an experiment cross a hexagonal boundary,
- flip an `xfail(strict)` to green without actually fixing the underlying bug it documents,
- expand an allowlist (bandit, mypy ignore list, etc.) to silence a finding your experiment
  introduced.

If your experiment can only "succeed" by weakening a gate, the experiment has failed — the
correct write-up is a retirement note (§3.1), not a gate change. Route any genuine gate change
through `orchestrator-change-control`'s classification/approval path, entirely separately from
the experiment's own result.

## 6. Negative-result ledger

Retirements, dead branches, and settled "don't relitigate" questions are indexed in
`orchestrator-failure-archaeology` (its "Settled decisions" and "Stalled / dead branches"
sections) — that skill owns the ledger; this skill owns the process that feeds it. Check there
before starting a new investigation, and write your own retirement note there (or reference it
from a `docs/*_ANALYSIS.md` doc per the VS precedent in §3.1) when your experiment concludes
negative.

## Provenance and maintenance

Re-verify before trusting anything above if it's been a while:

- `git show e863f0c8 --stat` and `Read tests/unit/test_bug_scan.py` (around
  `TestAggregateNeverDiscardsRuns`, currently lines 84–104) — confirms Worked Example A's numbers
  still match the code.
- `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_config_drift.py` —
  re-run to get today's real drift count; the 5-entry count in Worked Example B is a snapshot from
  2026-07-08, not a permanent fact. Config files churn — expect this number to change.
- `git show 11deb573 --stat` — confirms the HITL fail-closed commit and its message wording.
- `ls docs/*_INSPIRED_ENHANCEMENTS.md` — confirms which competitor-analysis docs currently exist
  (10 as of this writing); new ones may be added, old ones renamed.
- `Read docs/VERBALIZED_SAMPLING_ANALYSIS.md` (§0 TL;DR, §2) — confirms the retirement finding
  summary above still matches the doc if it's been revised.
- `grep -n "Never weaken a gate" .claude/skills/orchestrator-change-control/SKILL.md` — confirms
  the unwritten-rule heading this skill cross-references still exists under that name.
- `grep -n "CompletionJudge" .claude/skills/llm-orchestration-reference/SKILL.md` — confirms the
  maker-checker section this skill points to is still there.
