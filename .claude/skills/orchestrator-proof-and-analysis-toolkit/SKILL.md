---
name: orchestrator-proof-and-analysis-toolkit
description: Worked-example recipes for "prove it, don't just install it" — proving a flag is wired vs dead, a model id is actually live, a cache saves real money without silently breaking self-consistency evaluation, an aggregation formula is mathematically correct, a bug fix hasn't regressed, a paper's claimed technique matches the code that cites it, a cost number adds up by hand, and a new dependency is genuinely needed. Load this when you are about to assert a fact about this repo ("this flag does X", "this is wired", "this is dead code", "the cache saves $Y", "the fix landed") and need the verification method, not just the conclusion. Symptom keywords — "is this flag actually used", "prove this is dead code", "is the response-healing plugin wired", "does use_provider_sorting do anything", "is this model id still valid on OpenRouter", "does the cache defeat self-consistency", "is aggregate() correct for N runs", "how do I retire an xfail", "does our VS implementation match the paper", "recompute this cost by hand", "do we actually need this new dependency". For the flag catalog itself use orchestrator-config-and-flags; for the incident chronicle use orchestrator-failure-archaeology; for the measurement scripts (check_config_drift.py etc.) use orchestrator-diagnostics-and-tooling; for gate/process rules use orchestrator-change-control.
---

# Orchestrator Proof and Analysis Toolkit

Eight recipes. Each one is a **method**, demonstrated on a **real, currently
verified example from this repo** (branch `feat/response-healing`, verified
2026-07-08 — re-verify anything you rely on; the codebase moves). The point
of this skill is not the specific conclusions below — those will drift — it
is the *discipline*: grep every call site, read the actual branch, run the
actual command, before you write the word "wired" or "dead" or "fixed" in a
report. Two of the worked examples below **overturn claims from this
project's own memory notes**, discovered by simply tracing the code instead
of trusting the prior write-up. That is the whole lesson of this skill.

---

## Recipe 1 — Prove a flag is wired vs dead

**Method**: `Grep` the flag's exact string across `orchestrator/`. For every
hit, classify it as (a) **declaration** — a dataclass field or
`os.getenv(...)` read that produces a Python value, or (b) **consumer** — code
that branches on that value (`if opts.FLAG:`, `getattr(opts, "FLAG", ...)`
inside a function that changes behavior). A flag with only (a) hits is dead.
A flag with a (b) hit is wired **only if you can trace that consumer function
is itself actually called** from the live request path — a well-tested
helper function that nothing calls is still dead. This last trap is the one
most people miss (see Case B below).

### Case A — `USE_PROVIDER_SORTING`: dead, single read site, no consumer

```
orchestrator/config.py:173   USE_PROVIDER_SORTING: bool = False              (declaration)
orchestrator/config.py:188   USE_PROVIDER_SORTING=os.getenv(...)             (declaration)
orchestrator/crosscutting/config.py:53  use_provider_sorting: bool = False   (a second, unrelated declaration)
```
`Grep -i provider_sorting` across the entire `orchestrator/` package turns up
**no third file** — no `if opts.USE_PROVIDER_SORTING`, no
`getattr(..., "USE_PROVIDER_SORTING", ...)` anywhere outside the two
declarations above. There is no code path this flag could possibly affect.
**Verdict: dead.** (This matches `orchestrator-diagnostics-and-tooling`'s
`flag_inventory.py` finding — cross-reference, don't re-derive.)

### Case B — `USE_RESPONSE_HEALING`: the subtler failure — tested, but never called

This one is more interesting because a naive check says "wired": there IS a
real consumer function.

```python
# orchestrator/infrastructure/llm_client.py:493-514
def _maybe_add_response_healing(request_params: dict, opts) -> None:
    """Add the OpenRouter ``response-healing`` plugin to a request when enabled. ..."""
    if opts is None or not getattr(opts, "USE_RESPONSE_HEALING", False):
        return
    if "response_format" not in request_params or request_params.get("stream"):
        return
    extra_body = request_params.setdefault("extra_body", {})
    plugins = extra_body.setdefault("plugins", [])
    if not any(isinstance(p, dict) and p.get("id") == "response-healing" for p in plugins):
        plugins.append({"id": "response-healing"})
```
This function is real, well-written, and has 6 passing unit tests in
`tests/unit/test_response_healing.py` that exercise every branch (flag off,
no `response_format`, streaming, idempotency, `opts=None`).

**The trap**: a unit test importing and calling a function directly proves
the function works in isolation. It proves **nothing** about whether the
function is reachable from production code. `Grep "_maybe_add_response_healing"`
across the whole repo returns exactly **three** hits: the `def` line, the
`__all__` export two lines below it, and the test file. Read
`UnifiedClient.call()` → `_dispatch()` → `_create_openrouter_client()` in the
same file (`llm_client.py:246-452`, the entire live request path) end to end:
none of them call `_maybe_add_response_healing`, and no `opts`/
`OpenRouterOptimizations` object is even threaded into `call()`'s parameters
in the first place. **Verdict: dead in the call path**, despite being the
namesake feature of the very branch (`feat/response-healing`) this was
verified on. The same is true of the sibling helper
`_resolve_provider_variant` (`:nitro`/`:floor`/`:exacto` suffix handling,
`llm_client.py:475-490`, unit-tested in `tests/unit/test_provider_variants.py`,
also never called from `_dispatch`).

### The recipe, generalized
1. `Grep` the flag name (case-sensitive and once case-insensitive) across
   `orchestrator/` — list every hit with file:line.
2. Classify each hit: declaration or consumer.
3. If zero consumers → dead, stop.
4. If a consumer exists, find its caller. Repeat "who calls this" until you
   either reach a method on a class that is instantiated and invoked on the
   real request path (wired), or you run out of callers and land only in a
   test file (dead-in-practice — tested but unreachable).
5. Do not stop at step 2. A single unit test file is not evidence of wiring.

**You have proven it when**: you can name the exact call chain from an entry
point (`UnifiedClient.call()`, `engine.py`, a CLI command) down to the
consumer line, with no test file in that chain — or you have exhausted every
grep hit and confirmed none of them form such a chain.

---

## Recipe 2 — Prove a model id is live

**Method**: follow the four-link chain and don't stop at any one link.

1. **Enum**: is the id a `Model` member in `orchestrator/models.py`
   (`Model.X.value == "provider/model[:variant]"`)?
2. **`costs.json`**: does `orchestrator/config/costs.json` have a key that is
   the **exact string** of `Model.X.value`? (`COST_TABLE` is built by
   `_build_cost_table`, `orchestrator/models.py:816` — it maps `costs.json`
   keys through `Model(k)`; a key that isn't a valid enum value is **silently
   dropped**, no exception. Same drop pattern applies to `routing.json`/
   `fallbacks.json`.)
3. **`routing.json`** (and `fallbacks.json`): same exact-string-match
   requirement for any task-type routing/fallback entry referencing the id.
4. **`scripts/audit_openrouter_models.py`**: cross-checks every
   `"provider/model[:variant]"` literal referenced in `models.py`,
   `domain/model_registry.py`, and `phase_aware_models.py` against the live
   OpenRouter catalogue (`GET /api/v1/models`, keyless). Run:
   ```bash
   python scripts/audit_openrouter_models.py --json
   ```
   Verified 2026-07-08: `live_model_count: 509`, `referenced_id_count: 150`,
   `dead: {}`, `stale_replacements: {}` — clean at that date. Re-run before
   trusting this number; it is network-dependent and models get deprecated.

**The id-normalization trap** (do not skip this step): OpenRouter
**server-side normalizes hyphenated Anthropic ids**. The audit script encodes
this explicitly:
```python
# scripts/audit_openrouter_models.py:69
_ANTHROPIC_HYPHEN = re.compile(r"^anthropic/claude-(opus|sonnet|haiku)-(\d+)-(\d+)$")
# :103-106 — anthropic/claude-opus-4-6  normalizes to  anthropic/claude-opus-4.6
```
An id like `anthropic/claude-opus-4-6` can be **absent from the live
`/api/v1/models` catalogue and still resolve correctly at call time** —
OpenRouter accepts the hyphenated form and serves the dotted model. **Do not
conclude "not in catalog" ⇒ "dead"** for any `anthropic/claude-*-N-M` shaped
id; check whether it matches this pattern before flagging it. The audit
script also encodes a second allowlist, `RUNTIME_ONLY_IDS`
(`:76-81`) — video-generation models (`openai/sora-2-pro`,
`google/veo-3.1*`) are billed per-second via the generation endpoint and
never appear in the chat-models snapshot at all, but resolve at
`/api/v1/models/<id>/endpoints`; `verify_runtime_only_ids()` probes that
endpoint directly rather than trusting the allowlist blindly.

**You have proven it when**: the id is a `Model.value`, appears with an
exact-string match in every config file that should reference it (or you've
confirmed its absence is intentional — e.g. deliberately uncosted), and
`audit_openrouter_models.py --json` shows it under neither `dead` nor
(unexplained) `stale_replacements` — or, if it's an Anthropic hyphenated id
or a `RUNTIME_ONLY_IDS` member, you've confirmed the specific allowlist rule
that covers it instead of trusting a live-catalogue miss at face value.

---

## Recipe 3 — Prove a cache saves money (and doesn't quietly break eval)

**First, verify the TTL claim itself** — do not repeat a number from a prior
session's notes without checking. A previous internal note claimed "response
DiskCache, 48h TTL." Tracing it:

```python
# orchestrator/cache.py — a backward-compat shim
from .infrastructure.cache import DiskCache  # the canonical implementation

# orchestrator/infrastructure/cache.py:118-157 — get()/put()
async def get(self, model, prompt, max_tokens, system="", temperature=0.3):
    h = prompt_hash(model, prompt, max_tokens, system, temperature)
    ...
    "SELECT response, tokens_input, tokens_output FROM cache WHERE hash = ?"
    # no created_at / age filter anywhere in the WHERE clause
async def put(self, ...):
    ...
    "INSERT OR REPLACE INTO cache (hash, model, response, ..., created_at) VALUES (...)"
    # created_at IS stored (time.time()) but never read back to expire anything
```
**Verdict: the response cache used by `UnifiedClient.call()` has no expiry
logic at all** — an entry persists until `DiskCache.clear()` is called
explicitly. There is no 48h TTL anywhere in this class. (The `48` you'll find
elsewhere in the repo — `meta_config.py`/`meta/config.py`
`timeout_hours=48` — is an unrelated rollout-stage timeout, not a cache TTL;
don't let a stray grep hit confirm a wrong belief.) The number this skill's
authoring brief handed down was **wrong** — this is exactly the kind of
claim you must re-derive, not copy forward.

**Proving the $0-on-hit claim** (this part IS true): `UnifiedClient.call()`
(`llm_client.py:282-296`) checks the cache before dispatching, and on a hit
returns an `APIResponse` built entirely from the cached row with
`cost_usd=cached.get("cost", 0.0)` — no network call, so genuinely $0. Verify
by instrumenting or reading the code path directly; don't just trust the
docstring.

**The self-consistency trap**: `EvaluatorService` (whichever module is
actually wired — see Recipe 4) runs N scoring calls and aggregates them for
"self-consistency." If those N calls hit the **same** cache key (same
model/prompt/max_tokens/system/temperature — see `prompt_hash(...)` inputs
above), call #2..N return the **identical cached text**, not independent
samples. "2-pass self-consistency" silently degenerates into "1 real pass,
1 free replay of the same answer" — the aggregation math in Recipe 4 becomes
meaningless because `scores` isn't actually N independent samples.

**Detection recipe**:
1. Run the same evaluation call twice with `bypass_cache=False` (default) —
   confirm the second call's `latency_ms`/`cached` field shows a hit
   (`APIResponse.cached=True`, `llm_client.py:287-295`).
2. Re-run with `bypass_cache=True` explicitly passed in `kwargs` — confirm
   `call()` pops it (`llm_client.py:282`) and skips the cache lookup/write.
3. Compare: if a self-consistency loop does not pass `bypass_cache=True` for
   its repeat calls, its "N runs" are at risk of being fewer than N distinct
   samples whenever the prompt is byte-identical across runs.

**You have proven it when**: you have read `get()`/`put()` yourself and can
state the actual expiry behavior (currently: none), and you have confirmed
whether the self-consistency call site you're auditing passes
`bypass_cache=True` — with a file:line citation either way, not a guess.

---

## Recipe 4 — Prove aggregation math is right

**The invariant**: `aggregate([0.2, 0.9, 0.9])` must **not** equal `0.2`. A
correct N-run aggregator must not silently discard runs 2..N.

**Write the smallest failing input first** (do this before reading any
implementation, so you aren't anchored by what the code already does):
```python
def test_three_run_aggregate_does_not_discard():
    result = evaluator._aggregate([0.2, 0.9, 0.9], "task-x")
    assert result != 0.2, "3-run aggregate must not silently return only scores[0]"
```
Then check every run-count boundary: `[]` (0 runs), `[0.7]` (1 run),
`[0.8, 0.82]` (2 runs, low spread → mean), `[0.9, 0.5]` (2 runs, high spread →
lower score per the `_consistency_delta` gate), `[0.2, 0.9, 0.9]` (3 runs),
`[0.2, 0.4, 0.6, 0.8]` (4 runs).

**A verified, currently-live surprise**: this repo has **two separate
`EvaluatorService` classes with two separate `_aggregate` implementations**,
and they disagree:

| | `orchestrator/services/evaluator.py:249` | `orchestrator/application/evaluator.py:204` |
|---|---|---|
| 0/1 runs | `0.5` / `scores[0]` | not handled (falls to the `else` branch) |
| 2 runs | mean, or `min()` if spread > `_consistency_delta` | same logic |
| **3+ runs** | **median**, outlier-robust, logs high-spread disagreement | **`scores[0]`** — silently discards runs 2..N |
| Fixed in commit | `e863f0c8` ("fix _aggregate dropping runs") | **never fixed** |
| Covered by | `tests/unit/test_bug_scan.py` (`self._ev()` imports
`orchestrator.services.evaluator.EvaluatorService`) | `tests/unit/test_evaluator.py` (imports
`orchestrator.application.evaluator.EvaluatorService`, and — check yourself
— has **no 3+-run test case** at all, only 0/1/2-run assertions) |

The commit `e863f0c8` message describes the fix as landing in
`orchestrator/services/evaluator.py`. But trace what actually gets
constructed in production:
```python
# orchestrator/engine_core/container.py:325
from ..application.evaluator import EvaluatorService
```
**The container wires the unfixed class.** This is not a hypothetical —
confirm it yourself with:
```bash
grep -n "application.evaluator import EvaluatorService\|services.evaluator import EvaluatorService" orchestrator/engine_core/container.py orchestrator/services/__init__.py orchestrator/services/scorers.py
```
As of 2026-07-08, `container.py` (the production DI wiring, per
`orchestrator-architecture-contract`) and `services/scorers.py` both import
from `application.evaluator`; only `services/__init__.py` re-exports from
`services.evaluator`. If self-consistency in production actually runs 3+
evaluation passes (check `_consistency_delta`/`consistency_runs`
configuration for the currently-active `EvaluatorService`), it may still be
silently discarding runs 2..N in the live path — **the e863f0c8 fix may not
be reaching production**. This needs a human decision (which class should
`container.py` wire, or should the two be merged) — do not silently "fix" it
without going through `orchestrator-change-control`; flag it instead.

**You have proven it when**: you have run the `[0.2, 0.9, 0.9]` case against
the **specific class instance that `container.py` actually constructs** (not
whichever `EvaluatorService` import came up first in an editor's autocomplete)
and can state its behavior with a file:line citation.

---

## Recipe 5 — Prove no silent regression (the xfail(strict=True) ledger)

**Pattern**: `tests/unit/test_preexisting_problems.py` catalogs known,
unfixed bugs. Each test asserts the **correct** behavior and is marked
`@pytest.mark.xfail(strict=True)` while the bug exists — because `strict`
makes an unexpected `XPASS` a hard failure, the moment someone fixes the
underlying bug without touching this file, CI breaks and forces them to
remove the marker. This makes "silently fixed and nobody documented it" and
"quietly regressed a fix" both structurally impossible to miss.

**Adding an entry** (when you discover a new pre-existing bug you're not
fixing right now):
1. Write a test asserting the **correct** behavior (not the buggy one).
2. Add a comment block above it: file/line of the bug, root cause, blast
   radius.
3. Mark `@pytest.mark.xfail(strict=True, reason="...")`.
4. Confirm it actually fails as xfail (`pytest tests/unit/test_preexisting_problems.py -v`
   should show `xfail`, not `xpass` or `error`).

**Retiring an entry — the worked example, verified**: commit `416b9e18`
("fix(p4,p5): eliminate both pre-existing catalogued problems") is the
correct shape for a retirement:
- **P4** (`orchestrator/hierarchy.py`): node IDs were `f"{type}_{len(nodes)}"`
  — reused after a delete, causing silent overwrite. Fixed with a monotonic
  `self._id_counter` via `_next_id()`.
- **P5** (`orchestrator/cost_optimization/batch_client.py`): poll loop did
  `if request.result:` — a valid falsy result (`""`, `0`, `{}`) was treated
  as "not ready," blocking to the 300s timeout. Fixed by gating on
  `request.status == BatchStatus.COMPLETED` instead of truthiness.
- Both tests in `test_preexisting_problems.py` had their `xfail` marker
  **removed** in the same commit — they are visible today
  (`test_hierarchy_ids_survive_removal`,
  `test_batch_result_falsy_is_recognized_as_complete`) as plain regression
  guards with `[FIXED]` in their docstring header, asserting the fixed
  behavior with no xfail wrapper. Commit message states the before/after
  test count explicitly: "603 passed, 0 failed, 0 xfailed (was 601 passed +
  2 xfail)" — a concrete, checkable number.
- P1/P2/P3 (automations sync-handler counting, cron weekday convention,
  cron `*/0` step) remain `xfail(strict=True)` in the same file as of
  2026-07-08 — confirm with `pytest tests/unit/test_preexisting_problems.py -v`
  and count 3 `xfail` outcomes.

**You have proven it when**: `pytest tests/unit/test_preexisting_problems.py -v`
shows exactly the xfail/pass split you expect for the current state of the
codebase, and any entry you just fixed has its marker removed **in the same
commit** as the fix (never leave a fixed bug's test still marked xfail — the
next unrelated `XPASS` from someone else's unrelated fix will get confused
with yours during triage).

---

## Recipe 6 — Prove a paper/technique claim before implementing

**Method**: before writing code that claims to implement Paper X's Technique
Y, (1) read the paper's precise mechanism, not just its abstract/name, (2)
find every place the codebase already claims to implement it, (3) diff the
actual prompt/algorithm against the paper's specification line by line, (4)
write down every mismatch as a named gap.

**Worked example**: `docs/VERBALIZED_SAMPLING_ANALYSIS.md` (dated
2026-06-15) is the full worked protocol-diff for the "Verbalized Sampling"
paper (Zhang et al., arXiv:2510.01171v3) against
`orchestrator/reasoning/ara_pipelines.py::VerbalizedSamplingPipeline`. The
paper's central claim: asking for *k* items (**list-level** prompt) recovers
at best a **uniform** distribution (Claim 2); only asking for *k* items
**each tagged with a probability** (**distribution-level** prompt) recovers
the model's actual pre-training diversity (Claim 3). The gap analysis found:

- **G1 — dead code**: nothing in `ara_execution_strategy.py`'s
  `default_methods`/`retry_methods` maps any `TaskType` to
  `VERBALIZED_SAMPLING` — it is implemented and registered in
  `PipelineFactory` but never invoked on the real execution path.
- **G2 — wrong prompt shape**: the generation prompt asks for "k distinct
  plausible candidate answers" with **no per-item probability** requested —
  probability defaults to a uniform `1.0/k` after the fact. This is exactly
  the paper's Sequence/list-level **baseline**, not VS.
- **G3 — inverted probability semantics**: a separate call asks for "the
  probability this is the correct/optimal solution" and keeps the
  highest-scoring candidate — the paper's probability means *typicality
  under the base distribution*, used to **enable exploring the low-typicality
  tail**, not to pick the most-typical answer. Selecting the highest-probability
  candidate re-introduces the exact typicality bias VS exists to remove.
- **G4 — no real tail-sampling knob**: the paper's diversity control is a
  generation-time instruction ("probability of each response must be below
  {threshold}"); the code only has a post-hoc **selection** cutoff.
- **G5 — synthesis collapses diversity back to one answer**, defeating the
  purpose when combined with G3.

Conclusion in the doc: the real, paper-grounded value for an orchestrator is
NOT "more creative prose" — it's better MAP-Elites seed diversity, more
diverse synthetic/test data, and a principled escape hatch on the
self-consistency retry path (ranked options B/C/E in the doc), each scoped as
a small, isolated, flagged change — not a wholesale rewrite of the existing
(wrong) pipeline.

**You have proven it when**: you can point to the specific paper section
(figure/claim/equation) that each implementation choice either satisfies or
violates, with a file:line for the code side of each comparison — "we
implement VS" is not a claim you can make from the class name
`VerbalizedSamplingPipeline` existing; it has to survive this diff.

---

## Recipe 7 — Prove cost math by hand

**Method**: pick a real `(model, input_tokens, output_tokens)` triple,
look up the rate in `orchestrator/config/costs.json`, and recompute
independently of any code, then compare against what the code returns.

**The formula** (`orchestrator/models.py:1176-1180`):
```python
def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})  # fallback if unmapped
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
```
Rates in `costs.json` are **USD per 1,000,000 tokens** (confirm by checking a
known-price model, e.g. `"openai/gpt-4o": {"input": 2.50, "output": 10.00}` —
OpenAI's published GPT-4o pricing is $2.50/$10.00 per 1M tokens as of the
model's release, matching).

**Worked example**: 12,000 input tokens + 3,500 output tokens against
`openai/gpt-4o-mini` (`costs.json`: `{"input": 0.15, "output": 0.60}`):
```
input_cost  = 12000 * 0.15 / 1_000_000 = 0.0018
output_cost = 3500  * 0.60 / 1_000_000 = 0.0021
total       = 0.0039  →  $0.0039
```
The same computation appears independently in `UnifiedClient.call()`'s
non-cached branch (`llm_client.py:347-351`) using `self._cost_service.get_cost(model_enum)`
— trace that `CostService` resolves the *same* `costs.json` row (not a
different table) before trusting both paths agree; two independently
maintained cost formulas is exactly the kind of thing that drifts.

**A relevant gotcha while doing this**: any cache hit returns
`cost_usd=cached.get("cost", 0.0)` (see Recipe 3) — if you're reconciling a
dashboard total against hand-computed costs, remember cache hits contribute
literal `$0`, which is correct behavior, not a missing-data bug, but it will
make your by-hand total for "N calls" diverge from "N × per-call cost" unless
you also know the cache hit rate for that run.

**You have proven it when**: your hand computation (rate from `costs.json` ×
tokens ÷ 1,000,000) matches the value the code actually returns/records for
the same model+token triple, and you've confirmed which cost table
(`COST_TABLE` vs whatever `CostService` reads) both paths are really using —
don't assume they're the same table without checking, given this repo's
demonstrated config/enum drift history (`orchestrator-config-and-flags`).

---

## Recipe 8 — Prove a dependency is actually needed before adding it

**Method** (stdlib-first, per `orchestrator-change-control`'s "no new
dependencies without explicit approval" rule): before adding any package,
(1) check the standard library, (2) check whether an already-declared
dependency already solves it, (3) if truly novel, name the exact capability
gap it fills, then (4) **declare it in `pyproject.toml` in the same commit
you start using it** — this last step is the part people skip.

**Counter-example, live, from this session**: `instructor` (structured LLM
output parsing/validation on top of the OpenAI SDK — no stdlib or
already-declared equivalent provides typed-response coercion against a
Pydantic schema for chat completions). This genuinely passed the "is it
needed" test — `orchestrator/infrastructure/llm_client.py`'s entire
`UnifiedClient` is built around `instructor.from_openai(...)`.

**But it landed without being declared.** Commit `067ca737`
("fix(ci): resolve mypy errors, undeclared instructor dep, and test
regressions on feat/response-healing") describes the failure mode exactly:
> "`infrastructure/llm_client.py` imports instructor at module scope but it
> was never declared in `pyproject.toml`/`requirements.txt`; a clean CI
> install hit `ModuleNotFoundError` during conftest collection."

This is the cautionary half of the recipe: **"needed" is necessary but not
sufficient.** A dependency that's genuinely required but undeclared is still
a proof failure — it just fails later and more expensively (a clean-checkout
CI run, not a code review) than an unneeded dependency that at least shows up
as a diff in `pyproject.toml` for someone to question. The same commit also
fixed a second, related problem the missing declaration was masking: the
import was at **module scope**, so every `import orchestrator` paid
`instructor`'s ~30s cold-import cost on Windows (see
`orchestrator-diagnostics-and-tooling` §5 for the `importtime` measurement)
— the fix both declared the dependency (`pyproject.toml`:
`"instructor>=1.0,<2.0"`) and made the import lazy (moved inside
`_instructor_mode()` and the two client-construction call sites).

**The recipe, checkable**:
1. `grep -rn "^import <pkg>\|^from <pkg>" orchestrator/` — find every module-scope
   import of the candidate package.
2. `grep -n "<pkg>" pyproject.toml` — confirm it's declared in `[project]`
   `dependencies` (or the correct extra).
3. If (1) has hits and (2) doesn't: this is a live, undeclared-dependency
   bug identical in shape to `067ca737` — a clean `pip install -e .` +
   `pytest` collection will `ModuleNotFoundError`. Fix by declaring it, and
   check whether the import should also be lazy (any package with a
   noticeable cold-import cost, or one only needed by a subset of code
   paths, belongs inside a function, not at module scope — see
   `orchestrator-diagnostics-and-tooling` §5's `importtime` technique to
   measure "noticeable").
4. Before adding anything new, first check `pip show <candidate>` /
   `pyproject.toml`'s existing dependency list for something that already
   covers the need — the ponytail doctrine (stdlib/existing-dep first)
   applies before you get to step 1.

**You have proven it when**: `pip install -e ".[dev]"` on a genuinely clean
checkout (`git clone` to a fresh directory, or at minimum a clean virtualenv)
followed by `pytest --collect-only` succeeds with zero `ModuleNotFoundError`
— that is the actual failure mode `067ca737` fixed, and it is the only
proof that a currently-imported package is both needed *and* correctly
declared.

---

## When NOT to use this skill

- You want the **flag catalog with wired/dead verdicts already compiled** →
  `orchestrator-config-and-flags` (this skill teaches the *method*; that one
  owns the *current answers* — cross-check them against each other and
  prefer whichever was verified more recently).
- You want the **measurement scripts themselves** (`check_config_drift.py`,
  `flag_inventory.py`, `audit_openrouter_models.py`, `check_bom.py`) →
  `orchestrator-diagnostics-and-tooling`.
- You want the **incident chronicle** (root cause + commit hash + status for
  settled bugs) rather than a proof method → `orchestrator-failure-archaeology`.
- You're triaging a **live failure right now** and need a symptom→cause
  lookup ranked by cost → `orchestrator-debugging-playbook`.
- You want **why** a gate/rule exists or the process for changing one →
  `orchestrator-change-control`.
- You want **evidence doctrine** — which pytest marker, where a new test
  file goes, what counts as sufficient coverage → `orchestrator-validation-and-qa`.
- You want the **domain theory** behind routing/budgets/resilience, not a
  proof technique → `llm-orchestration-reference`.

---

## Provenance and maintenance

All findings above are dated 2026-07-08 on branch `feat/response-healing`;
several **correct prior claims recorded elsewhere in this project's memory**
(the 48h cache TTL, and an assumption that `e863f0c8`'s aggregation fix is
live in production). Both were found wrong or incomplete by direct tracing
during authoring of this skill — treat every fact below as due for the same
treatment on your next pass, not as settled.

| Fact | Re-verification command |
|---|---|
| `USE_PROVIDER_SORTING` has zero consumers | `grep -rn "USE_PROVIDER_SORTING" orchestrator/` — expect exactly 3 hits, all declarations |
| `USE_RESPONSE_HEALING` plugin helper is uncalled | `grep -rn "_maybe_add_response_healing" orchestrator/ tests/` — expect def + `__all__` + test file only, no call site in `llm_client.py`'s `call()`/`_dispatch()` |
| `_resolve_provider_variant` is uncalled | `grep -rn "_resolve_provider_variant" orchestrator/ tests/` — same pattern |
| Response cache has no TTL/expiry | Read `orchestrator/infrastructure/cache.py`'s `get()` — confirm the `SELECT` has no age/`created_at` filter |
| `bypass_cache` kwarg exists and is honored | `grep -n "bypass_cache" orchestrator/infrastructure/llm_client.py` |
| Two `EvaluatorService._aggregate` implementations disagree | `diff <(sed -n '200,290p' orchestrator/application/evaluator.py) <(sed -n '245,300p' orchestrator/services/evaluator.py)` (line numbers drift — locate `_aggregate` in each file first) |
| Which `EvaluatorService` `container.py` wires | `grep -n "evaluator import EvaluatorService" orchestrator/engine_core/container.py` |
| xfail ledger current state | `pytest tests/unit/test_preexisting_problems.py -v` |
| Verbalized Sampling gaps (G1-G5) still open | Re-read `docs/VERBALIZED_SAMPLING_ANALYSIS.md` §2.2 against current `orchestrator/reasoning/ara_pipelines.py` and `ara_execution_strategy.py` |
| `costs.json` rate for any model | `grep -n "\"<model-id>\"" orchestrator/config/costs.json` |
| `instructor` still declared + lazy-imported | `grep -n "instructor" pyproject.toml` and `grep -n "import instructor" orchestrator/infrastructure/llm_client.py` (all hits indented/function-local) |
| Live model-id catalogue check | `python scripts/audit_openrouter_models.py --json` |
