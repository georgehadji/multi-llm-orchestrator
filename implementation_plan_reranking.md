# Implementation Plan — Reranking in the Core Loop

**Status:** Draft · **Date:** 2026-06-25 · **Branch target:** `feat/reranking`

Two reranking gaps, both wiring (not building) over existing infra:

- **#1 VS Selector** — Verbalized Sampling produces k diverse candidates but nothing
  selects among them (`verbalized_sampling.py:10` — *"the primitive never selects"*). No
  caller consumes `.sample()`. VS is dead weight without a selector.
- **#2 Two-stage knowledge recall** — `knowledge_base.find_similar()` is single-stage
  bi-encoder cosine. Add a rerank stage to refine recall before it feeds task context.

---

## Architecture constraints (must hold)

From `CLAUDE.md` Four Unbreakable Rules + Hexagonal layering:

1. **`engine.py` = Mediator** — new logic → new service/application module, engine only wires.
2. **`models.py` = pure data** — no behaviour added there.
3. **TDD** — failing test (RED) → impl (GREEN) → commit, per unit.
4. **No new root modules** — code lands in `application/`, `services/`, `domain/`, `infrastructure/`.
5. **Ports & Adapters** — application/domain depend on Protocol ports, never on infra concretes.
6. **Cost-first** ([[cost-reduction-audit]]) — reranking is opt-in, cheap-tier, prefilter before paying.

---

## Existing pieces to reuse (do NOT rebuild)

| Piece | Location | Role in plan |
|-------|----------|--------------|
| `VerbalizedSampler.sample()` → `list[VSCandidate]` | `application/verbalized_sampling.py` | #1 candidate source |
| `VSCandidate(text, probability)` | same | #1 candidate dataclass (frozen) |
| `EvaluatorService.evaluate(task, output, policy)` → `CritiqueReport(score)` | `services/evaluator.py` | #1 quality signal |
| `LLMReranker.rerank(query, results, top_k, min_score)` | `infrastructure/reranker.py` | #2 rerank stage |
| `KnowledgeBase.find_similar(query, top_k, min_similarity)` | `knowledge_base.py` | #2 stage-1 recall |
| `FeatureFlags` (pydantic) | `crosscutting/config.py:~90` | both — opt-in flags |
| `phase_policy` (temp/effort) | `domain/phase_policy.py` | #1 scorer temp = EVALUATE |

> Note: all reranking here is **LLM-based** (no cross-encoder dep installed). Keep the scorer
> behind a port so a real cross-encoder (`bge-reranker`/Cohere) can drop in later without
> touching callers.

---

## Part #1 — VS Candidate Selector

### Goal
Turn `list[VSCandidate]` → the single best candidate using a swappable quality signal,
with a free probability prefilter to bound cost.

### New files
```
orchestrator/domain/ports.py            # ADD QualityScorer Protocol (no new file)
orchestrator/application/vs_selector.py  # NEW — CandidateSelector
orchestrator/services/scorers.py         # NEW — EvaluatorScorer + ProbabilityScorer adapters
tests/unit/test_vs_selector.py           # NEW — RED first
```

### Port (domain) — `QualityScorer`
```python
class QualityScorer(Protocol):
    async def score(self, task: Task, text: str) -> float: ...  # 0.0–1.0
```

### `CandidateSelector` (application/vs_selector.py)
- `__init__(self, scorer: QualityScorer, *, prefilter_keep: int = 4)`
- `async select(task, candidates: list[VSCandidate]) -> VSCandidate | None`
  1. Empty → `None`.
  2. **Free prefilter:** sort by `VSCandidate.probability` desc, keep top `prefilter_keep`
     (cap LLM scoring cost — don't pay to score the long tail).
  3. Score survivors via `scorer.score(task, c.text)` (parallel, `asyncio.gather`).
  4. Return argmax score; tie-break by `probability`.
- Pure orchestration over the port. No infra import. No engine reference.

### Adapters (services/scorers.py)
- `EvaluatorScorer(evaluator: EvaluatorService)` — `score()` calls
  `evaluator.evaluate(task, text)` and returns `report.score`. For reranking, construct the
  evaluator with `consistency_runs=1` (single pass — reranking needs ranking, not the
  2-pass Δ-guard; halves eval cost). Temp already `phase_policy.EVALUATE` (0.1).
- `ProbabilityScorer()` — free fallback: returns `VSCandidate.probability`. Used when
  `vs_reranking_enabled` is off or budget is exhausted (degrade gracefully, never block).

### Config flag (crosscutting/config.py FeatureFlags)
```python
vs_reranking_enabled: bool = False   # ORCH_VS_RERANKING_ENABLED — opt-in
vs_rerank_prefilter_keep: int        # OrchestratorSettings, default 4
```

### Wiring (the Mediator boundary)
- VS currently has **no consumer**. Pick ONE concrete first consumer to avoid scope creep:
  the **CREATIVE / diversity path** (`phase_policy.SAMPLING`, temp 0.9) — VS's natural home.
- In the relevant stage (`engine_core/stages/generate.py` creative branch) OR
  `application/` orchestrator: if `flags.vs_reranking_enabled` →
  `candidates = await sampler.sample(...)` then `best = await selector.select(task, candidates)`.
- Engine only constructs `CandidateSelector(EvaluatorScorer(...))` and passes it in —
  no selection logic in engine.

### Tests (RED → GREEN)
- `select([])` → None.
- prefilter keeps only top-N by probability (assert scorer called N times, not k).
- argmax: scorer stub returns highest for candidate #2 → #2 returned.
- tie on score → higher probability wins.
- scorer raises on one candidate → that candidate scored 0, others still ranked (no crash).
- flag off → ProbabilityScorer path (zero eval calls).

### Cost guard
k≤6, prefilter_keep=4, consistency_runs=1 → ≤4 cheap eval calls per VS use. Flag default OFF.

---

## Part #2 — Two-stage knowledge recall

### Goal
`find_similar` becomes: stage-1 cheap cosine recall (widened) → stage-2 rerank → top_k.
Backward compatible; off by default.

### Touched files
```
orchestrator/knowledge_base.py          # find_similar gains optional rerank
orchestrator/domain/ports.py            # reuse/add Reranker Protocol
tests/unit/test_knowledge_rerank.py     # NEW — RED first
```

### Port — `Reranker`
```python
class Reranker(Protocol):
    async def rerank(self, query: str, results: list[dict], top_k: int,
                     min_score: float = 0.3) -> list[dict]: ...
```
`LLMReranker` (infrastructure/reranker.py) already matches this shape (returns
`RerankResult`; add a thin `.rerank_dicts()` or map in the adapter to keep KB port-pure).

### `find_similar` change (knowledge_base.py)
- Signature add: `rerank: bool = False, fetch_k: int = 20`.
- Inject reranker via constructor (`KnowledgeBase(__init__(..., reranker: Reranker | None = None))`),
  NOT imported inside the method (keeps KB testable + port-pure).
- Flow:
  1. Stage 1: cosine as today but take top `fetch_k` (not `top_k`) above `min_similarity`.
  2. If `rerank and self._reranker and len(stage1) > top_k`:
     map artifacts → `[{"doc_id": a.id, "content": a.content}]`,
     `ranked = await reranker.rerank(query, docs, top_k=top_k)`,
     reorder artifacts by ranked doc_id, set `artifact.similarity_score = relevance_score`.
  3. Else: return `stage1[:top_k]` (unchanged behaviour).
- **Cache key must include `rerank` + `fetch_k`** (line 217) — else reranked and raw share a key.

### Config flag
```python
knowledge_rerank_enabled: bool = False   # ORCH_KNOWLEDGE_RERANK_ENABLED
```
Caller (`find_similar` consumers, e.g. line 362 `find_similar(top_k=3)`) passes
`rerank=flags.knowledge_rerank_enabled`.

### Tests (RED → GREEN)
- rerank=False → identical to current cosine order (regression lock).
- rerank=True, fewer than top_k hits → reranker NOT called (no-op short circuit).
- rerank=True → reranker reorders; output follows reranker order, not cosine order.
- reranker=None but rerank=True → silent fallback to cosine (no crash).
- cache: rerank=True and rerank=False produce distinct cache entries.

### Cost guard
Stage-1 cosine is free (local vectors). Stage-2 LLM rerank = `fetch_k` cheap calls **only
when flag on AND hits > top_k**. Default OFF. Consider capping `fetch_k` to 20.

---

## Build order (TDD units, each its own commit)

1. `domain/ports.py`: add `QualityScorer` + `Reranker` Protocols. (no test — types)
2. **#1** `test_vs_selector.py` RED → `application/vs_selector.py` + `services/scorers.py` GREEN.
3. **#1** config flags + engine wiring into creative/sampling path; integration smoke test.
4. **#2** `test_knowledge_rerank.py` RED → `knowledge_base.find_similar` rerank stage GREEN.
5. **#2** config flag + inject `LLMReranker` adapter at KB construction; consumer passes flag.
6. Docs: `docs/RERANKING.md` (when + why), update `CODEBASE_MINDMAP.md` pattern table.
7. Full `pytest tests/unit/ -q` — no regressions.

## Out of scope (explicit)
- Real cross-encoder dep (`sentence-transformers`/Cohere) — port leaves the door open; not now.
- Reranking model-routing candidate lists (#4 in research) — separate effort.
- MAP-Elites rerank (#5) — already has a score signal.

## Risks
- **Cost creep** if flags default ON → both default **OFF**, prefilter + fetch_k caps.
- **Eval as scorer is slow** → `consistency_runs=1` for rerank path; parallel scoring.
- **VS has no consumer today** → #1 step 3 must add exactly one, behind a flag, or VS selection
  is untested in situ. Keep that consumer minimal.

## Verification
- `pytest tests/unit/test_vs_selector.py tests/unit/test_knowledge_rerank.py -v`
- Flag-off path proves zero added LLM calls (assert on stubbed client call count).
- `import-linter` contracts stay KEPT ([[architecture-remediation-phase-a]]).
