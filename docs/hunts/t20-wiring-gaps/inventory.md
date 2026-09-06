# T20 — wiring gaps: registered but never read

Wave 20 of the depth pass. Shape hunted: something **declared** as a control
surface — a CLI flag, a feature flag, a settings field — that no code ever
consults, so setting it appears to work and does nothing.

## Phase 0–1 — census

Four sub-shapes were enumerated by AST:

| Sub-shape | Declared | Never read |
|---|---|---|
| CLI flags (`add_argument`) | 109 | **0** |
| `FeatureFlags` fields | 53 | 8 |
| `OrchestratorSettings` fields | 21 | 12 |
| DI container services | — | n/a (see below) |

### Detector calibration — two wrong answers before the right one

The first CLI census reported 18 never-read flags. It subtracted declaration
hits from a count that never contained them, so `args.deps` (read at
`commands/website.py:98`) was scored unread. The second swung the other way —
collecting every `.attr` load and every keyword name repo-wide — and reported
0 by matching almost anything.

The third scopes reads to the argparse namespace itself (`args.X`,
`getattr(args, "X")`, `vars(args)`) and was **calibrated against seven flags
confirmed read by hand** before its answer was believed: 0 false positives.
Only then was "0 never-read CLI flags" recorded as a result.

That inverse check — names read off a namespace that nothing declares —
produced 15 candidates, **all false**: `parsed` is a dict in
`conversation_agent.py` and a `urlparse` result in `deployment_feedback.py`,
`options` is a config dataclass in `fullstack_generator.py`, `args` is a list
in `validator.py` and a string in `slack_integration.py`; and `func`,
`meta_cmd`, `subcommand`, `template_action` are declared via
`set_defaults(func=)` (34 sites) and `add_subparsers(dest=)`, which the
detector did not model.

**The CLI surface is clean.** That is a negative result, and it is reported as
one rather than padded.

### DI container

`engine_core/container.py` is hand-wired: `build()` constructs each service
explicitly into dataclass fields. There is no name-keyed register/resolve
registry, so "registered but never resolved" has no separate meaning here —
it collapses into "a dataclass field nothing reads", which is the settings
sub-shape. Not hunted separately; recorded so the gap is visible.

## Phase 3–4 — triage

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `knowledge_rerank_enabled` — see below |
| C2 | **VERIFIED — ESCALATED** | `bilevel_tabu_enabled` is inert, and `engine_core/tabu_search.py` is imported by nothing. Every other `tabu_search` occurrence is a string literal inside a docstring example (`mechanism_registry.py:12-16`) or a comment (`mechanism_researcher.py:25`). Flag *and* subsystem are dead. |
| C3 | **VERIFIED — ESCALATED** | `bilevel_level15_enabled` is inert, but unlike C2 its feature is live: `SearchStrategyTuner` is injected into `BilevelAutoresearch` through a constructor parameter. The flag is redundant, not the feature. |
| C4 | **VERIFIED — ESCALATED** | 12 of 21 `OrchestratorSettings` fields are read nowhere, including `dashboard_port`, `dashboard_host`, `mcp_port`, `audit_log_path`, `default_budget_usd`. With `env_prefix="ORCH_"` and `extra="ignore"`, `ORCH_DASHBOARD_PORT=9000` is accepted in silence and discarded. |
| C5 | **VERIFIED — ESCALATED (worth a second look)** | `dashboard_host: str = "127.0.0.1"` in config.py, while `dashboard_core/core.py:347,372` hardcode `host="0.0.0.0"`. The settings file states loopback-only; the dashboard binds every interface. An operator reading the config would conclude the opposite of what runs. Also `dashboard_port: int = 8000` vs. the real `8888`. |
| C6 | FALSE (innocent) | Six `use_*` flags (`use_json_schema_responses`, `use_model_variants`, `use_native_fallbacks`, `use_provider_sorting`, `use_streaming`, `use_embedding_cache`) look dead but their env vars are read directly via `os.getenv`. CLAUDE.md documents them as working, and they do. The **field** is redundant, not the feature. |
| C7 | FALSE (innocent) | `cache_home` — same shape: `infrastructure/path_provider.py:37` reads `ORCH_CACHE_HOME` from the environment directly. |

### C1 — the defect

`knowledge_rerank_enabled` was declared in `crosscutting/config.py:121` and
read by **no module in the repository**. Behind it:

* `KnowledgeBase.find_similar(rerank=..., fetch_k=...)` — the two-stage
  cosine → LLM rerank, fully implemented;
* `infrastructure/reranker.py` — `LLMReranker`, live and injected into
  `HybridSearchPipeline` by `service_collection.py:123`;
* `tests/unit/test_knowledge_rerank.py` — five cases, all passing.

`implementation_plan_reranking.md` lists the implementation steps, the config
flag, and the five tests — every one of which was carried out — and then the
final step, which was not:

> Caller (`find_similar` consumers, e.g. `find_similar(top_k=3)`) passes
> `rerank=flags.knowledge_rerank_enabled`.

That consumer is `KnowledgeBase.get_recommendations`. It has no callers inside
`orchestrator/`, which nearly got it dismissed as dead code — but
`CAPABILITIES.md:247` and `USAGE_GUIDE.md:1141` both document it as *the*
public way to query the knowledge base:

```python
recs = await kb.get_recommendations("Build payment service")
```

So a user who followed the guide and set `ORCH_KNOWLEDGE_RERANK_ENABLED=true`
got no reranking and no warning. Fixed by taking the plan's last step: 7
lines, one function, one added import.

## Phase 5 — fix

```python
similar = await self.find_similar(
    current_task, top_k=3, rerank=flags.knowledge_rerank_enabled
)
```

`from .crosscutting.config import flags` at module top level, matching
`engine.py` and `api_server.py`; a lazy in-function import was written first
and replaced, because "lazy imports hiding breakage from top-level import
sweeps" is itself one of the seven shapes this depth pass exists to remove.
No cycle: verified by importing the module directly.

The flag defaults to `False`, so default behaviour is unchanged, and
`find_similar` already falls back to cosine when no reranker is configured
(covered by an existing test).

## Phase 7 — proof

`tests/unit/test_hunt_t20_wiring.py`, 3 tests, RED verified — all three fail
with the fix stashed. The clearest is the third, which states the defect in
its own failure message:

```
AssertionError: knowledge_rerank_enabled is declared but read by no module
assert []
```

## Phase 8 — gate

`scripts/check_config_wiring.py` freezes the 20 currently-unread fields and
fails on any **new** one, so the shape can only shrink. Verified by adding a
probe field: reported as `FeatureFlags.t20_gate_probe_flag`, exit 1.

The gate is deliberately generous about what counts as a read (any `.name` or
quoted occurrence outside `config.py`) — it exists to stop new dead settings,
not to adjudicate the existing twenty.
