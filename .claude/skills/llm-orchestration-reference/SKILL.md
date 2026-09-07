---
name: llm-orchestration-reference
description: Domain-theory knowledge pack for the Multi-LLM Orchestrator — load this when you need to understand WHY the system is designed the way it is, not how to run it. Covers model routing theory (ROUTING_TABLE, fallback chains, VFM free-tier-first doctrine, OpenRouter model-id semantics like ":free" variants and alias ids), the dual-budget cost model and token pricing math, resilience theory (circuit breaker states, retry vs fallback, backoff/jitter), evaluation science (LLM-as-judge, 2-pass self-consistency, conservative aggregation, maker-checker CompletionJudge, score-parsing hazards), HTN-style decomposition and topological execution, structured-output healing layers (json5, partial repair, OpenRouter response-healing plugin), LLM caching theory (key design, why caching defeats self-consistency), per-phase temperature/reasoning policy, and HITL fail-closed doctrine. Symptom keywords: "why is routing empty", "score is always 0.5", "eval runs identical", "what does :free mean", "why two budgets", "circuit breaker open", "JSON parse failed from model", "silently dropped model", "why fail-closed".
---

# LLM Orchestration Reference — Theory As It Applies HERE

Audience: a mid-level engineer or Sonnet-class model with zero context on multi-LLM
orchestration. This skill explains the *concepts* and *why the code implements them
this way*, with every claim grounded in a file:line. It owns THEORY only.

**When NOT to use this skill**
| You want... | Go to |
|---|---|
| Commands to run/resume/operate the orchestrator | `orchestrator-run-and-operate` |
| The catalog of env flags and config knobs | `orchestrator-config-and-flags` |
| Incident history (what broke, commit hashes) | `orchestrator-failure-archaeology` |
| Where new code may live, layering rules | `orchestrator-architecture-contract` |
| Gates, TDD rules, what you may not weaken | `orchestrator-change-control` |
| Step-by-step failure triage | `orchestrator-debugging-playbook` |

---

## 1. Model routing theory

### 1.1 Task-type → model routing

Routing answers: *given a task of type X, which models do we try, in what order?*

- `TaskType` enum — 9 task types: `code_generation`, `code_review`, `complex_reasoning`,
  `creative_writing`, `data_extraction`, `summarization`, `evaluation`, `image_generation`,
  `video_generation` (`orchestrator/models.py:65-83`).
- `ROUTING_TABLE: dict[TaskType, list[Model]]` is built lazily from
  `orchestrator/config/routing.json` by `_build_routing_table()`
  (`orchestrator/models.py:778-784`). The list order IS the try order.
- `FALLBACK_CHAIN: dict[Model, Model]` maps each model to its *next hop* on failure,
  built from `orchestrator/config/fallbacks.json` (`orchestrator/models.py:787-793`).
- All five config tables (`COST_TABLE`, `ROUTING_TABLE`, `FALLBACK_CHAIN`,
  `DEFAULT_THRESHOLDS`, `MAX_OUTPUT_TOKENS`) load on first attribute access via module
  `__getattr__` (`orchestrator/models.py:815-831`) — no disk I/O at import time
  (Unbreakable Rule #2: models.py is pure data).

### 1.2 THE silent-drop drift trap (memorize this)

The builders filter with `if m in Model._value2member_map_`
(`orchestrator/models.py:781,792`). A JSON key/value that is not an *exact* Model enum
value is **silently dropped** — no error, no log. Downstream consequence: with an empty
eval model list, `EvaluatorService` returns the 0.5 default score
(`orchestrator/application/evaluator.py:91-93`), so the whole quality loop degrades
without any visible failure.

> Live demonstration (date-stamped 2026-07-07): the uncommitted working tree at
> authoring time had `routing.json`/`fallbacks.json` rewritten to bare ids like
> `"gpt-4o-mini"` (the enum value is `"openai/gpt-4o-mini"`). Verified by executing
> `_build_routing_table()`: **every task type resolved to an empty list**. The
> committed HEAD version is free-tier-first and fully qualified. If you see this,
> restore the config from HEAD and run the drift tests below.

Contract tests that lock this: `tests/unit/test_vfm_routing.py`
(`test_no_silent_drops`, `test_every_routed_model_is_priced`).

### 1.3 Quality/cost tiers and the VFM doctrine

**VFM (value-for-money) free-tier-first doctrine**: every *text* task's routing list
must lead with a `$0` free-tier model and escalate to paid models only on failure,
while the *last* entry must be a reliable paid model (never `:free`). This is not
folklore — it is an executable contract in `tests/unit/test_vfm_routing.py`:

| Invariant | Test |
|---|---|
| Lead candidate ends with `:free` | `test_lead_candidate_is_free_tier` (line 78) |
| No routing entry silently dropped | `test_no_silent_drops` (line 86) |
| Every routed model has a costs.json entry | `test_every_routed_model_is_priced` (line 94) |
| ≥2 models; last resort is paid | `test_has_premium_fallback` (line 101) |

Cost tiers for cascade decisions: `CostTier` = FREE < BUDGET < PREMIUM with
thresholds $0 / ≤$0.50 / >$0.50 per 1M input tokens
(`orchestrator/operations/resilience.py:357-370`). `classify_model_tier()` defaults to
PREMIUM when a model has no cost entry — *fail-safe: assume expensive, never assume
free* (`orchestrator/operations/resilience.py:373-385`).

### 1.4 OpenRouter model-id semantics

All models route through OpenRouter; ids are `provider/slug` with optional suffixes.

| Suffix / form | Meaning | Handling |
|---|---|---|
| `:free` | *Endpoint variant* — a genuinely distinct $0 endpoint of the same model | Passed to the API **intact** (endpoint variants "returned unchanged", `orchestrator/infrastructure/llm_client.py:469-476`); priced 0.0 in `costs.json` |
| `:nitro` / `:floor` | Sorting aliases (throughput / price) | **Stripped** from the slug and mapped to `provider.sort` (`llm_client.py:458,461-473`) |
| `:exacto` | OpenRouter-native variant | Kept on the slug; flagged so task-strategy sort does not override it (`llm_client.py:474-475`) |

- **Alias enum members**: `CLAUDE_OPUS = CLAUDE_OPUS_4_8`, `CLAUDE_SONNET = CLAUDE_SONNET_5`,
  `CLAUDE_HAIKU = CLAUDE_HAIKU_4_5` (`orchestrator/models.py:256-259`). These are Python
  enum *aliases* — same member object, same `.value`. Config JSON must always use the
  canonical `.value` string, never the Python attribute name.
- **Server-side id normalization**: OpenRouter resolves Anthropic hyphenated ids
  (`anthropic/claude-opus-4-6` style) server-side, which is why they are deliberately
  NOT in the dead-id map (comment at `orchestrator/domain/model_registry.py:134-135`).
- **Dead-id redirect map**: `ModelRegistry.UNAVAILABLE_MODELS` maps verified-dead
  OpenRouter ids to live replacements matched by provider + capability + price tier
  (`orchestrator/domain/model_registry.py:136-188`, verified via live URL checks,
  header comment dated 2026-04-01). `validate_model_available()` redirects so persisted
  state referencing an old id keeps working (`model_registry.py:501,516`).

### 1.5 Fallback chain ≠ routing list

Two different mechanisms, both real:
1. **Routing list** (`ROUTING_TABLE[task_type]`) — the *ordered candidate set* chosen
   before execution.
2. **Fallback chain** (`FALLBACK_CHAIN[model]` / `ResiliencePolicy.fallback_chain`) —
   the *next-hop on failure* once a chosen model has exhausted its retries
   (`orchestrator/operations/resilience.py:52-53,63`).

---

## 2. Cost model

### 2.1 Token pricing math

Prices are USD **per 1 million tokens**, split input/output
(`orchestrator/config/costs.json`, e.g. `"qwen/qwen3-coder:free": {"input": 0.00, "output": 0.00}`):

```
cost_usd = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
```
(`orchestrator/models.py:1176-1180`). An unknown model defaults to `{"input": 5.0,
"output": 20.0}` — again the pessimistic fail-safe.

### 2.2 Per-run `Budget` (orchestrator/budget.py)

Tracks spend/time inside a single `run_project()` call. Defaults: `max_usd=8.0`,
`max_time_seconds=5400` (`budget.py:44-45`). Key ideas:

- **Atomic reserve/commit/release** under an `asyncio.Lock` prevents the TOCTOU race
  where N concurrent tasks each "check budget, then spend" (`budget.py:121-145`).
  `remaining_usd` subtracts both spent AND reserved (`budget.py:70-72`).
- **Phase partitioning (soft caps)**: decomposition 5%, generation 45%, cross_review 25%,
  evaluation 15%, reserve 10% (`BUDGET_PARTITIONS`, `budget.py:21-27`) — stops one phase
  from eating the whole budget.
- Lives in `budget.py`, not `models.py`, precisely because it has async behavior
  (module docstring, `budget.py:1-9`).

### 2.3 Cross-run `BudgetHierarchy` (orchestrator/cost.py) — the Composite

Org → Team → Job caps that persist *across* runs; "all three are checked
independently. The most restrictive constraint wins" (`cost.py:136-155`). It does NOT
replace the per-run Budget — both can be active simultaneously (`cost.py:146-149`).
`can_afford_job()` also *reserves* the estimate on success so concurrent jobs see
committed-not-yet-settled spend (`cost.py:190-253`). Optional SQLite persistence via
`db_path` so caps survive restarts (`cost.py:151-155`).

### 2.4 Cost prediction and pre-flight

- `CostPredictor` — EWMA (exponentially weighted moving average, `alpha=0.1`) of
  actual per-(model, task_type) costs, falling back to a static estimate; also offers
  `cheapest_model()` (`cost.py:444-524`).
- `CostForecaster.forecast()` → frozen `ForecastReport` with `estimated_total_usd`,
  per-phase breakdown, and `RiskLevel` (LOW <50% of budget, MEDIUM 50–80%, HIGH ≥80%)
  plus `will_exceed_budget()` for pre-flight refusal (`cost.py:95-128,525+`).

---

## 3. Resilience theory

### 3.1 Why retry ≠ fallback ≠ circuit breaker

Three distinct answers to "the call failed":

| Mechanism | Question it answers | Where |
|---|---|---|
| **Retry** | "Was this a transient blip on the SAME model?" — exponential backoff, bounded attempts | `ResiliencePolicy.retries` = attempts *per model* (`operations/resilience.py:47,58`) |
| **Fallback** | "Is this MODEL the problem?" — switch to next model after retries exhaust | `fallback_chain` (`operations/resilience.py:52-53`) |
| **Circuit breaker** | "Is this PROVIDER dying? Stop hammering it" — fail fast without calling | `orchestrator/circuit_breaker.py` |

Retrying a dead provider wastes budget and time; falling back without retrying wastes
a healthy model on a network blip; neither prevents a thundering herd — that is the
breaker's job (`circuit_breaker.py:4`).

### 3.2 Circuit breaker state machine

States (`circuit_breaker.py:37-40`, docstring 6-9):
`CLOSED` (normal) → after `failure_threshold=5` consecutive failures → `OPEN`
(calls raise `CircuitBreakerOpen` immediately, marked `retriable = False`,
`circuit_breaker.py:43-47`) → after `reset_timeout=60s` → `HALF_OPEN` (ONE probe call
allowed) → `success_threshold=2` consecutive successes → `CLOSED`; probe failure or
`half_open_timeout=30s` re-trips to `OPEN` (`circuit_breaker.py:79-91`).

A second, simpler per-model health breaker exists in `FallbackHandler`: unhealthy
after 3 consecutive failures, 60s cooldown, success resets the counter
(`orchestrator/application/fallback_handler.py:36-46`).

### 3.3 Retry policy shape

`ResiliencePolicy` (frozen dataclass, `operations/resilience.py:41-70`): default
`retries=2`, `timeout=60s`, exponential backoff base 2.0 capped at 30s, **±20% jitter**
(desynchronizes concurrent retriers so they don't stampede in lockstep), and an
explicit `retryable_exceptions` allowlist (TimeoutError/ConnectionError/OSError) —
non-listed exceptions (e.g. auth errors) fail immediately instead of burning retries.
Execution uses `tenacity` with `stop_after_attempt | stop_after_delay`
(`operations/resilience.py:262-297`). `RetryTemplate` gives per-task presets — e.g.
CODE_GEN retries=3/timeout=120s; REASONING timeout=300s with backoff cap 60s
(`operations/resilience.py:85-158`). Note `orchestrator/resilience.py` is only a
backward-compat shim re-exporting from `operations/resilience.py`.

### 3.4 Rate limiting

`orchestrator/rate_limiter.py`: `RateLimiter.acquire(tokens=...)` gates calls by token
volume (line 56-68); `GrokRateLimiter` adds per-tier limits (`TierLimits`, lines
97-132) with an acquire timeout. Rate limiting is *pre-emptive* (don't send what the
provider will 429) whereas the breaker is *reactive* (stop sending after it did).

---

## 4. Evaluation science

### 4.1 LLM-as-judge with 2-pass self-consistency

An LLM scores each task output in [0.0, 1.0]. One judgment is noisy, so
`EvaluatorService` runs `consistency_runs=2` independent scoring passes and compares
them (`orchestrator/application/evaluator.py:43-62`). Aggregation (`_aggregate`,
`application/evaluator.py:203-219`):

- 2 runs, |Δ| ≤ `consistency_delta=0.05` → **mean** (runs agree; average out noise).
- 2 runs, |Δ| > 0.05 → **min** (runs disagree; the score is unreliable, so take the
  *conservative* lower bound — a falsely low score costs one extra iteration, a
  falsely high score ships bad output).

**Known discrepancy (verified 2026-07-07):** commit `e863f0c8` fixed `_aggregate`
discarding runs 3..N by adding median aggregation — but that fix landed in
`orchestrator/services/evaluator.py:249-297` while the container actually wires
`orchestrator/application/evaluator.py` (`orchestrator/engine_core/container.py:325,454`),
whose `_aggregate` still returns `scores[0]` for 3+ runs
(`application/evaluator.py:219`). Latent because the default is 2 runs, but do not
raise `consistency_runs` above 2 on the wired copy, and do not "clean up" either file
without reading `orchestrator-failure-archaeology` first.

### 4.2 Why evaluation runs cold (temperature/reasoning)

EVALUATE phase policy: temperature 0.1, thinking ON at HIGH effort,
`exclude_reasoning=True` — the chain-of-thought is generated (and paid for) but
stripped from the response so it cannot pollute the structured score
(`orchestrator/domain/phase_policy.py:91`, attribute docs at 66-74).

### 4.3 Score parsing hazards

`parse_score()` (`application/evaluator.py:221-304`) exists because judges return
scores in wildly inconsistent formats. Layered defense, in order:

1. Strip `<think>...</think>` blocks (closed AND truncated-tail) **first** so numbers
   inside a reasoning trace are never mistaken for the score (lines 234-237).
2. Strip markdown fences; parse with `json5` (lenient) falling back to `json`
   (lines 242-253).
3. Regex for human formats: `score: 0.8`, Chinese 评分/得分, `8/10`, `85/100`, `85%`,
   `out of 10` — with scale renormalization to [0,1] (lines 270-294).
4. Any bare float in [0,1] (lines 296-301).
5. **0.5 safe default** — unparseable output means "unknown quality", not zero
   (line 304). Every path clamps to [0.0, 1.0].

Invariants are locked by `tests/unit/test_bug_scan.py` (commit `e863f0c8`).

### 4.4 Adversarial evaluation: maker-checker (`CompletionJudge`)

Theory: a generator grading its own work converges on self-approval ("cognitive
surrender" — see §9). `CompletionJudge` (`orchestrator/services/completion_judge.py`)
is the stop-condition antidote:

- **Independence invariant**: `judge_model != generator_model`, enforced at
  construction with `SameModelError` (lines 63-77).
- **Fail-closed**: unparseable verdict, unknown verdict string, or any exception →
  `FAIL`, forcing another iteration (lines 101-111). PASS must be earned.
- **Cheap by design**: `from_models()` picks the cheapest candidate that differs from
  the generator; returns `None` (caller skips judge) when none differs (lines 113-131).
- Judge runs at `temperature=0.0`, `max_tokens=150`, with a system prompt asserting
  "You did NOT write this output" (lines 44-50, 93-100).

The core loop this all serves: decompose → per task: generate → critique → revise →
evaluate, iterating to `max_iterations` (CLAUDE.md "Core Execution Pipeline").

---

## 5. Decomposition: HTN-style task graphs

"HTN-style" here means: an LLM decomposes the project into a flat set of tasks with
explicit dependency edges — a DAG (directed acyclic graph), not a nested plan tree.

- The decomposition prompt demands "Dependencies MUST form a DAG — no circular
  dependencies" (`orchestrator/application/decomposer.py:365`).
- Each parsed task carries a `dependencies` list; a bare string is normalized to a
  one-element list (`decomposer.py:667-677`).
- `DecomposerService` wraps the engine's `_decompose` via injected callback and
  returns a `DecomposerResult` with success/metrics (`decomposer.py:40-114`).

**Topological execution**: `DependencyResolver` implements Kahn's algorithm
(in-degree counting + zero-degree queue) for a valid execution order
(`orchestrator/application/dependency_resolver.py:77-113`) and `get_ready_tasks()`
for "all dependencies satisfied" scheduling (line 125). The engine exposes
`_topological_sort` (`orchestrator/engine.py:1159`) and *topological levels* —
tasks at the same level share no path between them, so a whole level can run in
parallel (`orchestrator/application/project_runner.py:199,396`;
`orchestrator/delegation/batch_runner.py:18`).

---

## 6. Structured output & healing

### 6.1 Why LLM JSON breaks

Failure modes seen in this codebase (each with a matching repair layer): markdown
code fences around the JSON; trailing commas / single quotes / unquoted keys;
`max_tokens` truncation mid-object; reasoning text before/around the JSON; schema
deviations from non-structured-output models (`orchestrator/task_schemas.py:274`;
`orchestrator/infrastructure/model_capabilities.py:9`).

### 6.2 The repair ladder (cheapest first)

1. **Lenient local parse**: try `json5` (tolerates trailing commas, comments,
   single quotes), fall back to stdlib `json`
   (`decomposer.py:573-616`; `application/evaluator.py:248-253`).
2. **Partial extraction**: `_repair_partial_tasks()` recovers whatever complete task
   objects exist in a truncated/garbled decomposition payload rather than discarding
   everything (`decomposer.py:437,532-560`).
3. **Server-side healing** (branch `feat/response-healing`, flag-gated):
   `_maybe_add_response_healing()` appends the OpenRouter `response-healing` plugin
   via `extra_body.plugins` — only when `USE_RESPONSE_HEALING` is true AND the request
   is non-streaming AND carries a `response_format`
   (`orchestrator/infrastructure/llm_client.py:479-500`). It repairs brackets,
   trailing commas, fences server-side; it **cannot fix max_tokens truncation**
   (`orchestrator/config.py:176-179`). It must ride `extra_body` because `plugins`
   is not an OpenAI-SDK kwarg (docstring, `llm_client.py:487-490`).
4. **Regex last resort** for scores only (§4.3).

All `USE_*` OpenRouter flags default false and come from env
(`orchestrator/config.py:168-195`); the full flag catalog belongs to
`orchestrator-config-and-flags`.

---

## 7. Caching for LLMs

### 7.1 The response DiskCache — key design

The cache that is actually in the call path: `DiskCache`
(`orchestrator/infrastructure/cache.py:29`; `orchestrator/cache.py` is a shim).
SQLite + WAL via aiosqlite. Key:

```
sha256(f"{model}||{system}||{prompt}||{max_tokens}||{temperature}")
```
(`prompt_hash`, `orchestrator/models.py:1167-1173`). Every request-shaping input is
in the key — change any one and you miss. The client checks it before dispatch and
returns the stored response with `cached=True` (near-$0 hit), writing through on
success (`orchestrator/infrastructure/llm_client.py:272-284,353-354`). Empty
responses are never cached — "a bad model reply should not poison the cache"
(`infrastructure/cache.py:147-148`).

### 7.2 TTL trade-offs — read carefully (2026-07-07)

- `DiskCache._CONN_TTL = 3600s` is the **SQLite connection** TTL, not entry expiry
  (`infrastructure/cache.py:37-38`).
- Entry rows store `created_at`, but `get()` applies **no expiry filter** — entries
  live until `clear()` (`infrastructure/cache.py:118-134,159-162`).
- The `cache_ttl_hours: int = 48` setting exists (`orchestrator/crosscutting/config.py:124`)
  but `DiskCache.get()` does not consult it. Treat "48h TTL" as intent, not enforced
  behavior, until you verify otherwise.

Trade-off theory: long/no TTL maximizes $0 hits but serves stale answers after model
or prompt-template upgrades; the key includes the prompt text, so template changes
naturally miss — staleness only bites for *identical* re-runs, which is usually what
you want for cost.

### 7.3 Why caching can defeat self-consistency evaluation

The 2-pass evaluation (§4.1) sends the *same* model, prompt, system, max_tokens, and
temperature twice → identical `prompt_hash` → run 2 is served run 1's cached text →
Δ=0 always → the disagreement check is vacuously satisfied. The escape hatch is the
`bypass_cache` kwarg on the client call (`llm_client.py:272-274`). If you see two
eval runs returning byte-identical text, suspect the cache before praising the
judge's consistency.

### 7.4 Related but NOT the same thing

- `orchestrator/infrastructure/caching.py` — **deleted 2026-09-07 (SEC-002)**. It
  was a generic multi-layer framework (`InMemoryCache` / `RedisCache` /
  `MultiLayerCache`) that `pickle.loads`'d cache blobs. It had zero importers and
  raised `AttributeError` on every write (`datetime.now(datetime.timezone.utc)()`),
  so nothing was cached and there was nothing to migrate. Do not confuse it with
  `orchestrator/infrastructure/cache.py`, the live response cache above.
  Guard: `tests/unit/security/test_no_pickle_deserialization.py`.
- Semantic cache (`semantic_cache_threshold=0.85`, `crosscutting/config.py:125`) —
  **not wired into the call path** as of the 2026-06-25 cost audit (locked by
  `tests/unit/test_cost_reduction.py`). Same audit: `use_provider_sorting` is a
  declared-but-dead flag. Details: `orchestrator-failure-archaeology`.

---

## 8. Prompt-side: personas and phase policy

### 8.1 Phase policy — single source of truth

`orchestrator/domain/phase_policy.py` centralizes per-phase temperature/reasoning
because literals had drifted across 50+ call sites (module docstring, lines 7-12).
The table (`phase_policy.py:87-101`):

| Phase | Temp | Thinking | Effort | Prefer reasoning model |
|---|---|---|---|---|
| DECOMPOSE | 0.2 | yes | HIGH | yes |
| CRITIQUE | 0.1 | yes | HIGH | yes |
| EVALUATE | 0.1 | yes | HIGH (CoT excluded) | yes |
| GENERATE | 0.2 | no | — | no |
| REVISE | 0.2 | no | — | no |
| EXTRACT | 0.0 | no | — | no |
| SUMMARIZE | 0.3 | no | — | no |
| CREATIVE | 0.8 | no | — | no |
| SAMPLING | 0.9 | no | — | no |

Theory: verification phases (decompose/critique/evaluate) get reasoning + near-zero
temperature because you want *deterministic, careful* judgment; creative/sampling
phases get high temperature because you want *spread*. Task-type overrides win over
phase defaults when GENERATE covers a creative or extraction task
(`temperature_for()`, `phase_policy.py:104-145`). Reasoning models ignore/forbid
temperature; the client omits it for them (`phase_policy.py:66-68`).

### 8.2 Personas

Persona support is flag-gated: `persona_enabled: bool = True`
(`orchestrator/crosscutting/config.py:77`), conditionally imported by the engine
(`orchestrator/engine.py:245-253`). `PersonaMode` = `STRICT` ("business/production,
strict validation") vs `CREATIVE` ("brainstorming, flexible output")
(`orchestrator/persona.py:37-41`); `PersonaManager` manages per-project persona
configuration (`persona.py:184+`). Tests: `tests/unit/test_persona.py`. Theory: a
persona is a *system-prompt-level* behavior contract, orthogonal to the numeric
phase policy — mode sets expectations, temperature sets variance.

---

## 9. HITL theory: fail-closed vs fail-open

**Fail-open**: when the approval machinery is missing, approve and continue.
**Fail-closed**: when it is missing, REJECT and stop. This repo learned the
difference the hard way (silent auto-approval incident, FIX-1, commit `11deb573` —
chronicle in `orchestrator-failure-archaeology`).

The gate (`orchestrator/hitl/gate.py`, policy in docstring lines 6-13):

| Situation | Result |
|---|---|
| `requires_approval=False` | APPROVED without channel (informational, not a gate) |
| Channel wired (CLI / WebSocket) | Delegated to the human |
| No channel + `ORCH_HITL_AUTOAPPROVE=true` | `AutoApproveChannel` — dev/test escape hatch, "never in production" (line 13) |
| No channel, no env flag | **`FailClosedChannel` → REJECTED** (`gate.py:59-70`) |

`has_real_channel()` deliberately counts neither FailClosed nor AutoApprove as a
human (`gate.py:72-84`).

**Cognitive surrender anti-pattern**: deferring judgment to the machine because
objecting takes effort. It appears twice in this codebase's defenses: (1) the HITL
gate refuses to let *absence of a human* count as approval; (2) `CompletionJudge`
refuses to let the *generator* count as its own reviewer (docstring names it
explicitly, `services/completion_judge.py:4-6`). The shared principle: any path
where "nobody actually checked" must resolve to NO.

---

## 10. Glossary

| Term | Meaning here |
|---|---|
| ROUTING_TABLE | TaskType → ordered model candidate list, built from routing.json (`models.py:778`) |
| FALLBACK_CHAIN | Model → next-hop model on failure, from fallbacks.json (`models.py:787`) |
| VFM | Value-for-money: quality per dollar; doctrine = free-tier model first, paid last resort |
| Free-tier variant (`:free`) | Distinct $0 OpenRouter endpoint of a model; id sent to API unchanged |
| Sorting alias (`:nitro`/`:floor`) | Suffix stripped client-side into `provider.sort` throughput/price |
| Canonical id vs alias | Enum `.value` string vs Python enum alias member (same object, `models.py:256-259`) |
| UNAVAILABLE_MODELS | Dead OpenRouter id → live replacement redirect map (`model_registry.py:136`) |
| Silent drop | Config key/value not matching an enum value being filtered out without error |
| Budget (per-run) | Async spend/time tracker with reserve/commit/release (`budget.py`) |
| BudgetHierarchy | Cross-run Org/Team/Job caps, most restrictive wins (`cost.py:136`) |
| TOCTOU | Time-of-check-to-time-of-use race; fixed by atomic reservation in both budgets |
| EWMA | Exponentially weighted moving average; `CostPredictor` cost smoothing (`cost.py:458`) |
| Pre-flight | Cost forecast before execution; refuse if `will_exceed_budget` (`cost.py:126`) |
| Circuit breaker | CLOSED/OPEN/HALF_OPEN fail-fast state machine per provider (`circuit_breaker.py`) |
| Retry vs fallback | Same-model bounded re-attempt vs next-model switch after retries exhaust |
| Jitter | ±20% randomization of backoff waits to avoid synchronized retry stampedes |
| LLM-as-judge | Using an LLM to score another LLM's output on [0,1] |
| Self-consistency (2-pass) | Score twice; agree → mean, disagree (Δ>0.05) → conservative min |
| Conservative aggregation | On disagreement take the lower bound; false-low is cheaper than false-high |
| Maker-checker | Independent judge model ≠ generator model gives PASS/FAIL stop verdict |
| CompletionJudge | The maker-checker implementation; fail-closed FAIL (`services/completion_judge.py`) |
| Cognitive surrender | Letting the machine (or the maker itself) approve because no one objects |
| Fail-closed / fail-open | Missing approval machinery → REJECT / → approve; this repo is fail-closed |
| HTN-style decomposition | LLM produces a task DAG with dependency edges, executed topologically |
| Kahn's algorithm | In-degree-based topological sort (`dependency_resolver.py:77`) |
| Topological level | Set of mutually independent tasks runnable in parallel |
| json5 | Lenient JSON parser (trailing commas, comments) used before strict json |
| Partial extraction | Recovering complete objects from truncated JSON (`_repair_partial_tasks`) |
| Response-healing plugin | OpenRouter server-side JSON repair; non-streaming + response_format only |
| prompt_hash | sha256 over model‖system‖prompt‖max_tokens‖temperature — the cache key (`models.py:1167`) |
| Write-through | Cache populated on successful call, checked before dispatch |
| Cache-defeats-self-consistency | Identical eval passes share a key, so pass 2 replays pass 1 |
| Phase policy | Per-phase temperature/thinking/effort table (`domain/phase_policy.py`) |
| exclude_reasoning | Generate CoT but strip it from the response (you still pay for it) |
| Persona | STRICT/CREATIVE system-prompt behavior contract (`persona.py:37`) |
| HITL | Human-in-the-loop approval gate (`hitl/gate.py`) |

---

## Provenance and maintenance

All file:line citations verified against the working tree on 2026-07-07, branch
`feat/response-healing`. Volatile facts are date-stamped inline. Re-verify with:

| Claim | One-line re-verification (repo root, PowerShell-safe) |
|---|---|
| Routing/fallback builders + silent drop | `python -c "import orchestrator.models as m; print({k.value: len(v) for k,v in m.ROUTING_TABLE.items()})"` (empty lists = drift) |
| VFM doctrine still locked | `pytest tests/unit/test_vfm_routing.py -q` |
| Config keys ≡ enum values | `pytest tests/unit/test_cost_reduction.py tests/unit/test_vfm_routing.py -q` |
| Wired evaluator is the application copy | `Select-String -Path orchestrator/engine_core/container.py -Pattern "application.evaluator"` |
| _aggregate 3+-run behavior (both copies) | `Select-String -Path orchestrator/application/evaluator.py,orchestrator/services/evaluator.py -Pattern "median|scores\[0\]"` |
| Circuit breaker defaults | `Select-String -Path orchestrator/circuit_breaker.py -Pattern "failure_threshold|reset_timeout"` |
| Response-healing gating | `Select-String -Path orchestrator/infrastructure/llm_client.py -Pattern "response-healing" -Context 2` |
| Cache key composition | `Select-String -Path orchestrator/models.py -Pattern "def prompt_hash" -Context 6` |
| DiskCache entry expiry (none) | `Select-String -Path orchestrator/infrastructure/cache.py -Pattern "created_at|SELECT response"` |
| Phase policy table | `Select-String -Path orchestrator/domain/phase_policy.py -Pattern "_PHASE_POLICY" -Context 15` |
| HITL fail-closed default | `Select-String -Path orchestrator/hitl/gate.py -Pattern "FailClosedChannel"` |
| Dead-id redirect map | `Select-String -Path orchestrator/domain/model_registry.py -Pattern "UNAVAILABLE_MODELS"` |
| Cost tier thresholds | `Select-String -Path orchestrator/operations/resilience.py -Pattern "_TIER_THRESHOLDS" -Context 5` |

Drift-prone items to re-check before relying on them: (1) the working-tree
routing.json/fallbacks.json bare-id corruption (may have been fixed or committed
since 2026-07-07); (2) the unwired median `_aggregate` in
`orchestrator/services/evaluator.py`; (3) DiskCache entry-TTL enforcement;
(4) `UNAVAILABLE_MODELS` contents (OpenRouter catalog churns).
