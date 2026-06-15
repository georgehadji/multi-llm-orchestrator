# Verbalized Sampling — Implementation Plan

> **Companion to:** [`VERBALIZED_SAMPLING_ANALYSIS.md`](VERBALIZED_SAMPLING_ANALYSIS.md) (the *why*). This document is the *how*.
> **Paper:** *Verbalized Sampling* (arXiv:2510.01171v3).
> **Scope:** Make VS real, faithful, and wired into the stages where it pays — under feature flags, TDD, respecting hexagonal boundaries.
> **Status:** Plan only. No code written yet. Awaiting go-ahead.

---

## 0. Guiding constraints (from this repo's architecture)

These are non-negotiable and shape every phase below:

1. **`engine.py` is a Mediator.** New VS logic goes into **new modules**, never into `engine.py`. The engine only wires.
2. **`models.py` is pure data.** VS *config* (a frozen dataclass + enum) may live there; VS *behavior* (prompt building, parsing, calls) must not.
3. **TDD, no exceptions.** Every phase starts with a failing test (RED), then implementation (GREEN), then refactor.
4. **Dependency rule + import-linter (4 contracts).** The reusable VS primitive depends only on the **`LLMClient` port** (`domain/ports.py:108`), never on a concrete adapter. Verified by `lint-imports` in CI.
5. **Feature-flagged.** Every behavior change is gated by an `ORCH_*` flag in `crosscutting/config.py::FeatureFlags`, default **off**, so mainline behavior is unchanged until opted in.

### 0.1 Two client contracts — pick the right one
The repo has **two** LLM call signatures:

| Contract | Signature | Returns | Used by |
|---|---|---|---|
| **`LLMClient` port** (canonical) | `call(model, prompt, system="", max_tokens, temperature, timeout, retries, task_type, response_schema, policy)` | `APIResponse` (`.text`, `.cost_usd`, `.input_tokens`, `.output_tokens`) | `GenerateStage`, `EvaluatorService`, `CritiqueStage` |
| **ARA tuple contract** (legacy/ported) | `call(model=, system_prompt=, user_prompt=, max_tokens=, temperature=)` | `tuple(response, _)` | every `BasePipeline` in `reasoning/ara_pipelines.py` |

> **Finding:** the ARA contract does **not** match the real `UnifiedClient`. The existing `VerbalizedSamplingPipeline` would raise `TypeError` (unexpected `system_prompt`) and fail tuple-unpacking against the live client — confirming it is dead, never-run code. **All new VS code standardizes on the `LLMClient` port.** Migrating the ARA pipeline onto the port is part of Phase 1.

---

## 1. Architecture: one primitive, many call sites

The mistake to avoid is re-implementing VS in each stage. Instead, build **one** reusable, port-only primitive and let every opportunity (B/C/D/E) call it.

```
                       ┌──────────────────────────────────────────┐
                       │  application/verbalized_sampling.py        │
                       │  VerbalizedSampler  (depends on LLMClient) │
                       │   .sample(prompt, k, threshold, fmt) ->    │
                       │        list[VSCandidate(text, probability)]│
                       └───────────────┬──────────────────────────┘
                                       │ (port only — no infra import)
        ┌──────────────────┬───────────┼────────────────┬───────────────────┐
        ▼                  ▼           ▼                ▼                   ▼
  MAP-Elites seed    Test/synthetic   Self-consistency   VS-first         Fixed ARA
  (map_elites.py)    data generator   tail-escape        GenerateStage    VerbalizedSampling
   [Phase 2/B]        [Phase 4/C]      [Phase 3/E]        [Phase 6/D]      Pipeline [Phase 1]
```

### 1.1 New modules

| Module | Layer | Responsibility | May import |
|---|---|---|---|
| `models.py` (additions) | Domain (pure) | `ProbabilityFormat` enum, `VSConfig` frozen dataclass | stdlib only |
| `application/verbalized_sampling.py` | Application | `VerbalizedSampler` (prompt build → call → parse → `list[VSCandidate]`); `VSCandidate` dataclass | `domain.ports.LLMClient`, `models`, `budget` |
| `tests/test_verbalized_sampling.py` | Tests | unit + integration coverage | — |

**Why `application/`:** it's reusable business logic (Rule #1), it depends only on the `LLMClient` port (passes contract `application-no-concrete-infra`), and both `engine_core/stages/*` and `reasoning/*` are allowed to import `application/*` (no contract forbids it).

### 1.2 The primitive (design sketch — real signatures)

```python
# models.py  (PURE — no I/O, no behavior beyond data)
from dataclasses import dataclass
from enum import Enum

class ProbabilityFormat(str, Enum):
    EXPLICIT = "explicit"      # best for VS-Standard (paper H.3)
    CONFIDENCE = "confidence"  # best for VS-Multi  (paper H.3)

@dataclass(frozen=True)
class VSConfig:
    k: int = 5                       # paper default; H.1 shows diminishing returns above
    probability_threshold: float | None = None   # None = no tail tuning; e.g. 0.10 = tail
    fmt: ProbabilityFormat = ProbabilityFormat.EXPLICIT
    temperature: float = 0.9         # VS is orthogonal to temp (paper §5.3); keep high for gen
    top_p: float = 0.95              # paper H.2 optimum
```

```python
# application/verbalized_sampling.py
from dataclasses import dataclass
from ..domain.ports import LLMClient
from ..models import Model, TaskType, VSConfig, ProbabilityFormat
from ..budget import Budget

@dataclass(frozen=True)
class VSCandidate:
    text: str
    probability: float   # verbalized typicality in [0,1]; ordering hint, NOT calibrated truth

_FORMAT_DEFS = {
    ProbabilityFormat.EXPLICIT:
        "the estimated probability from 0.0 to 1.0 of this response given the "
        "input prompt (relative to the full distribution)",
    ProbabilityFormat.CONFIDENCE:
        "the normalized likelihood score between 0.0 and 1.0 that indicates how "
        "representative or typical this response is compared to the full distribution",
}

class VerbalizedSampler:
    """One distribution-level VS call -> diverse candidates. Port-only; no selection."""

    def __init__(self, client: LLMClient, budget: Budget | None = None) -> None:
        self._client = client
        self._budget = budget

    async def sample(
        self, *, prompt: str, model: Model, cfg: VSConfig = VSConfig(),
        system_extra: str = "", task_type: TaskType | None = None,
        max_tokens: int = 4096, timeout: int = 160,
    ) -> list[VSCandidate]:
        system = self._build_system(cfg, system_extra)
        resp = await self._client.call(
            model=model, prompt=prompt, system=system,
            max_tokens=max_tokens, temperature=cfg.temperature, timeout=timeout,
            task_type=task_type, response_schema=bool(task_type),  # opt-in json_schema
        )
        if self._budget is not None:
            await self._budget.charge(resp.cost_usd, "verbalized_sampling")
        return self._parse(resp.text, cfg.k)

    def _build_system(self, cfg: VSConfig, extra: str) -> str:
        tail = ("" if cfg.probability_threshold is None else
                f" Randomly sample from the distribution such that the probability of "
                f"each response is below {cfg.probability_threshold}.")
        return (
            f"{extra}\nGenerate {cfg.k} possible responses to the user prompt. "
            f"Return ONLY JSON: {{\"responses\": [{{\"text\": str, \"probability\": float}}]}}. "
            f"For each, \"probability\" is {_FORMAT_DEFS[cfg.fmt]}.{tail}"
        ).strip()

    def _parse(self, text: str, k: int) -> list[VSCandidate]:
        ...  # robust: strip ``` fences, json/json5, partial-array recovery, clamp prob to [0,1]
```

**Key correctness choices (fix the analysis's G2–G5):**
- **G2:** `text` + `probability` requested **in the same call** → distribution-level prompt (Claim 3), not list-level.
- **G3:** probability is **typicality**, not correctness. The primitive **never selects** — callers do. Quality selection (when needed) uses the existing `EvaluatorService`, kept strictly separate.
- **G4:** `probability_threshold` injects the paper's exact tail instruction.
- **G5:** returns the full candidate list; no synthesis collapse inside the primitive.

### 1.3 Parsing robustness (no new brittle regex)
`BasePipeline._extract_json` only matches the first `{...}` and can't handle a `responses` array reliably; the analysis flagged it. Reuse the hardened recovery already in the codebase (`engine_core/decomposer.py::_try_parse_partial_json_array`, referenced in `KIMI_K2_7_CODE_INTEGRATION_PLAN.md:67`) — extract it to a shared helper if needed, or call json5 then partial-array fallback. JSON-schema (`response_schema=True`) is an *optimization* layered on top, not the primary guarantee (it defaults off and not all providers honor it).

---

## 2. Phased delivery

Each phase is independently shippable, flag-gated, and adds value alone. Order = ascending risk.

### Phase 0 — Primitive + tests (foundation, no behavior change)
- **Build** `models.py` additions + `application/verbalized_sampling.py` + `VerbalizedSampler`.
- **Flag:** none (inert until a caller uses it).
- **Tests (RED→GREEN):**
  - builds a distribution-level prompt containing both `text` and `probability`;
  - `EXPLICIT`/`CONFIDENCE` inject the correct definition string;
  - `probability_threshold=0.1` appends the tail instruction; `None` omits it;
  - parser handles: clean JSON, ```json fences, partial/truncated array, missing probabilities (default uniform `1/k` **only as a parse fallback**, logged), out-of-range prob clamped;
  - charges `budget` when provided (mock); never raises on bad JSON (returns `[]` + warns).
- **Acceptance:** 100% of primitive unit tests pass; `lint-imports` clean; `mypy`/`ruff`/`black` clean.

### Phase 1 — Make the ARA `VerbalizedSamplingPipeline` faithful + runnable
- **Rewrite** [`reasoning/ara_pipelines.py:3094`](../orchestrator/reasoning/ara_pipelines.py) to:
  - delegate generation to `VerbalizedSampler` via the **port** (fixes the `system_prompt`/tuple mismatch that makes it un-runnable today);
  - stop defaulting probability to `1/k` in the generation step;
  - separate **typicality** (from VS) from **quality** (delegate to `EvaluatorService`) — pick best by *quality*, optionally bias toward tail for exploration;
  - implement the advertised **TAIL** mode via `VSConfig.probability_threshold`.
- Apply the same delegation to `BrainstormingPipeline` (`:2910`) so "VS-Multi" is real.
- **Flag:** existing ARA gating (`ARAExecutionStrategy.enabled`).
- **Tests:** pipeline issues exactly one generation call with a distribution-level prompt; TAIL mode forwards threshold; selection uses the evaluator, not verbalized prob; runnable against a fake `LLMClient` (port) end-to-end.
- **Acceptance:** pipeline executes against the port-typed fake without `TypeError`; produces ≥`k` distinct candidates.

### Phase 2 — (B) MAP-Elites seeding ⭐ highest ROI
- **Replace** `MAPElitesPipeline._initialize` ([`engine_core/stages/map_elites.py:80`](../orchestrator/engine_core/stages/map_elites.py)) list-level prompt with `VerbalizedSampler.sample(..., cfg=VSConfig(k=9, probability_threshold=0.10))` — deliberately seed unconventional grid cells from the tail.
- Migrate this file off the **tuple contract** onto the port (the audit flagged `map_elites.py` for the tuple contract; this is a clean moment to fix it).
- *(Optional, noted not required)* the `_heuristic_score` line-count metric is weak; leave behavior as-is this phase, file a follow-up to score via real features. Keep scope to seeding diversity.
- **Flag:** `ORCH_VS_MAP_ELITES_SEEDING` (default off) → falls back to current `_initialize`.
- **Tests:** when flag on, seeds come from `VerbalizedSampler` (assert call shape) and populate ≥N distinct grid cells; when off, legacy path unchanged.
- **Acceptance:** with flag on, grid coverage (distinct occupied cells) ≥ legacy on a fixed fake-client fixture.

### Phase 3 — (E) Self-consistency tail-escape ⭐
- After a model-swap retry has already failed once, route the **second** retry through VS-tail: map `TaskType.* → VERBALIZED_SAMPLING` in `retry_methods` ([`reasoning/ara_execution_strategy.py:53`](../orchestrator/reasoning/ara_execution_strategy.py)), guarded so it only fires as the *escalation after* `retry_for_quality`.
- The retry uses `VSConfig(probability_threshold=0.10)` to sample unconventional approaches, then the existing `EvaluateStage` scores them.
- **Flag:** `ORCH_VS_RETRY_ESCAPE` (default off).
- **Tests:** first sub-threshold retry still does model-swap; second retry (flag on) selects `VERBALIZED_SAMPLING` with tail threshold; disabled flag preserves current behavior exactly ([`engine_core/stages/self_consistency.py`](../orchestrator/engine_core/stages/self_consistency.py)).
- **Acceptance:** on a seeded "stuck" task fixture, tail-escape changes the candidate set (not a re-draw of the same mode).

### Phase 4 — (C) VS for test / synthetic-data generation ⭐ strongest evidence
- Add a VS path for tasks that generate tests/seed data (gate on `tdd_enabled` + new flag). Use `VSConfig(k=5, probability_threshold≈0.05)` so tail samples surface **edge cases**.
- Keep the existing quality gate/validators — diverse candidates still pass through validation.
- **Flag:** `ORCH_VS_TEST_GENERATION` (default off).
- **Tests:** generator requests a distribution-level batch; produced cases are de-duplicated; tail config increases distinct-input coverage on a fixture vs. direct prompting.
- **Acceptance:** measurable ↑ in distinct test inputs vs. baseline on a fixed fixture (mirrors paper §8 coverage intuition; no live-API assertion).

### Phase 5 — (H) Tier-aware VS variant selection (config win)
- Encode the scale trend (paper §B, Fig 4e-f) in routing: choose VS variant by model tier.
  - **STANDARD/PREMIUM** → allow VS-CoT/VS-Multi (quality *and* diversity gain).
  - **FREE/ULTRA-LOW/BUDGET** → VS-Standard only, or skip VS (cognitive burden can *lower* quality).
- Implement as a small pure helper `vs_variant_for(model) -> VSConfig|None` near `model_routing.py`/`ModelSelector`; consumed by callers in Phases 2–4/6.
- **Flag:** none (pure policy; covered by callers' flags).
- **Tests:** tier→variant mapping table; premium enables CoT, budget downgrades/None.

### Phase 6 — (D) VS-first GenerateStage (optional, hot path)
- Optional `VerbalizedSampler` front to `GenerateStage` ([`engine_core/stages/generate.py`](../orchestrator/engine_core/stages/generate.py)): one distribution-level call → k candidates → existing `EvaluateStage` selects best. Converts sequential retries into parallel exploration. Restrict to STANDARD/PREMIUM via Phase 5.
- **Flag:** `ORCH_VS_GENERATE` (default off). Cost guard: respect `BudgetEnforcer`; cap `k`.
- **Tests:** flag off → byte-identical to current single-call path; flag on → k candidates generated then scored by evaluator; budget charged once per candidate batch.
- **Acceptance:** on a fixture, best-of-k quality ≥ single-shot at ≤ k× cost; no change when flag off.

---

## 3. Cross-cutting concerns

| Concern | Decision |
|---|---|
| **Cost/latency** | VS ≈ k× tokens. Default `k=5`. All paths flag-gated + behind `BudgetEnforcer`. Phase 5 restricts expensive variants to capable models. |
| **Determinism** | Keep VS **off** the strict-format codegen path (single right answer → collapse is desirable). |
| **Probability trust** | Verbalized probs are *ordering hints* for tail sampling, never ground truth. Never used as the quality signal. |
| **Telemetry** | Charge `Budget` with reason `"verbalized_sampling"`; record candidate count + chosen index in `TaskResult.metadata` for later analysis. |
| **Config** | New flags in `FeatureFlags` (pydantic, `ORCH_` prefix, default `False`). VS constants in `VSConfig` (pure dataclass), not scattered literals. |
| **Schema** | Prefer robust parser; enable `response_schema`/json_schema as an optimization where `use_json_schema_responses` is on. |
| **Backwards-compat** | Every flag defaults off → zero behavior change until explicitly enabled. |

---

## 4. Test strategy (TDD, 80%+ target)

- **Unit** (`-m unit`): primitive prompt building, format strings, tail injection, parser robustness, tier mapping. Fast, no I/O, fake `LLMClient`.
- **Integration** (`-m integration`): each phase's stage/pipeline against a scripted fake client returning canned `{"responses":[...]}`; assert call shape, candidate handling, flag on/off equivalence.
- **Fixtures:** a `FakeLLMClient` implementing the `LLMClient` port (returns deterministic `APIResponse`) — reusable across phases; lives in `tests/`.
- **No `requires_api` assertions** for diversity numbers — assert *mechanism* (call shape, distinct-candidate counts on canned data), not live model quality.
- **Regression guard:** "flag off ⇒ identical behavior" test for every phase that touches a live path (2/3/4/6).

---

## 5. Rollout & verification

1. Per phase: RED → GREEN → `pytest -m "unit or integration"` green → `lint-imports` → `ruff`/`black`/`mypy` → commit (conventional message).
2. Branch per phase (or git worktree per `CLAUDE.md`), small PRs.
3. Enable flags in a non-prod run; compare candidate-diversity + quality + cost deltas via existing telemetry before defaulting any flag on.
4. Update [`docs/CODEBASE_MINDMAP.md`](CODEBASE_MINDMAP.md) (new module + flags) and [`ARA_INTEGRATION_ANALYSIS.md`](ARA_INTEGRATION_ANALYSIS.md) (VS now real) when Phases 1–2 land.

---

## 6. Effort & sequencing summary

| Phase | Opportunity | Files (primary) | Flag | Effort | Risk |
|---|---|---|---|---|---|
| 0 | Primitive | `models.py`, `application/verbalized_sampling.py`, tests | — | M | very low |
| 1 | Fix ARA VS | `reasoning/ara_pipelines.py` | existing ARA | M | low |
| 2 ⭐ | MAP-Elites seed | `engine_core/stages/map_elites.py` | `ORCH_VS_MAP_ELITES_SEEDING` | S | low |
| 3 ⭐ | Retry tail-escape | `reasoning/ara_execution_strategy.py`, `engine_core/stages/self_consistency.py` | `ORCH_VS_RETRY_ESCAPE` | S–M | low |
| 4 ⭐ | Test/synthetic data | new generator + TDD path | `ORCH_VS_TEST_GENERATION` | M | low–med |
| 5 | Tier routing | `model_routing.py`/`ModelSelector` | — | S | low |
| 6 | VS-first generate | `engine_core/stages/generate.py` | `ORCH_VS_GENERATE` | M | med |

**Recommended first slice:** Phases **0 → 1 → 2**. It delivers a faithful, reusable, tested VS primitive and a measurable win (MAP-Elites diversity) with near-zero blast radius, and unblocks 3/4/6.

---

## 7. Open questions for the user

1. **Scope of first PR** — ship the recommended 0→1→2 slice, or the full 0→6?
2. **Cost ceiling** — acceptable `k` and per-task token multiplier for the gated paths (default `k=5`)?
3. **MAP-Elites scorer** — fix the line-count `_heuristic_score` now (broader change) or defer to a follow-up (kept out of scope here)?
4. **JSON-schema** — turn on `use_json_schema_responses` for VS paths, or rely on the robust parser only?
