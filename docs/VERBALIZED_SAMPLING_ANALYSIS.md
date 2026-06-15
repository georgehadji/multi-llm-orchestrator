# Verbalized Sampling — Value Analysis for the AI Orchestrator

> **Paper:** *Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity* (Zhang, Yu, Chong, Sicilia, Tomz, Manning, Shi — Northeastern / Stanford / WVU, arXiv:2510.01171v3, Oct 2025)
> **Status:** Research analysis. No code changed.
> **Author:** analysis pass 2026-06-15
> **Related:** [`ARA_INTEGRATION_ANALYSIS.md`](ARA_INTEGRATION_ANALYSIS.md), [`ARA_IMPLEMENTATION_PLAN.md`](ARA_IMPLEMENTATION_PLAN.md)

---

## 0. TL;DR

1. **We already have a `VerbalizedSamplingPipeline`** ([`reasoning/ara_pipelines.py:3094`](../orchestrator/reasoning/ara_pipelines.py)). It is **(a) never invoked** — no `TaskType` maps to it, and the ARA layer isn't wired into `engine.py`'s mainline — and **(b) not faithful to the paper** (it uses a *list-level* prompt and an *inverted* probability definition).
2. The paper's mechanism is precise and easy to get wrong. **Our current implementation gets it wrong in the exact way the paper proves does not work** (Claim 2: list-level prompts recover a *uniform* distribution at best; only distribution-level prompts recover the base model's diversity — Claim 3).
3. The real, paper-grounded value for an *orchestrator* is **not** "more creative prose." It is: **better candidate exploration at fixed compute, more diverse synthetic/test data, and a principled escape hatch from repeated-failure mode collapse.** These map onto pipeline stages we already own.
4. Highest-leverage, lowest-risk wins: **(B) fix MAP-Elites seeding**, **(C) VS for test/synthetic-data generation**, **(E) VS-tail on the self-consistency retry path**. Each is a small, isolated change behind a feature flag.

---

## 1. What the paper actually claims (and the parts that matter to us)

### 1.1 The mechanism
- **Mode collapse** in aligned models is driven by **typicality bias** in human preference data: annotators systematically prefer familiar/fluent text. RLHF/DPO sharpens the base distribution `π_ref` by a power `γ = 1 + α/β > 1`, concentrating mass on the mode. This happens *even with a perfect reward model* — it's a data property, not an algorithm bug (§3).
- **Different prompt types collapse to different modes** (§4.1, §E.4, proven as Claims 1–3):
  | Prompt type | Example | What you get back |
  |---|---|---|
  | **Instance-level** ("write a story") | one answer | the **mode** of `π_ref` (Claim 1) |
  | **List-level** ("write 5 stories") | a list, no probabilities | a **uniform** distribution over related items, at best (Claim 2) |
  | **Distribution-level (VS)** ("write 5 stories *with their probabilities*") | items **+ verbalized probability each** | an **approximation of `π_ref`** itself (Claim 3) |

  > This is the crux. Asking for *k* items is **not** VS. Asking for *k* items **each tagged with a probability, in the same call** is VS. The probability tag is what makes the model recover its pre-training diversity instead of a flat list of near-duplicates.

### 1.2 The variants (verbatim prompt structure, App. I.2)
- **VS-Standard** — one call returns JSON `{"responses":[{"text", "probability"}, ...]}`. Probability = *"the estimated probability from 0.0 to 1.0 of this response given the input prompt (relative to the full distribution)"* (the **Explicit** format, which the H.3 ablation found best for VS-Standard).
- **VS-CoT** — same, but a single `reasoning` field precedes the list. Best diversity-quality trade-off on capable models (pushes the Pareto front, Fig 4d).
- **VS-Multi** — first turn returns a small batch with a `confidence` field; later turns say *"generate N more"*. Best for multi-turn / when you want to keep pulling more samples. Prefers the **Confidence** format.
- **Tail tuning (the diversity knob)** — append: *"Randomly sample the responses from the distribution, with the probability of each response must be below {threshold}."* Lower threshold → more diverse / more unconventional ("tail") outputs. This is **prompt-only**, no decoding changes.

### 1.3 Results that are relevant to an orchestrator (not just prose)
- **Synthetic data generation (§8, Table 4)** — fine-tuning on VS-generated math problems beats direct prompting on downstream MATH500/Olympiad/Minerva: avg **32.8 (direct) → 37.5 (VS-Multi)**. Direct prompting can *hurt* vs. baseline because of mode collapse. **This is the strongest "engineering" result in the paper.**
- **Open-ended QA (§7, Fig 9)** — VS lowers KL-divergence to the true answer distribution and raises coverage, **while keeping precision ≈ 0.96** (diversity does not cost correctness).
- **Orthogonal to decoding (§5.3, H.2)** — VS stacks with temperature, top-p (~0.95 optimal), and min-p. It's a *different axis* from sampling params.
- **Emergent scale trend (§5.1, Fig 4e-f)** — **larger/more-capable models benefit more**, and on capable models the more complex variants (VS-CoT/VS-Multi) improve **both** diversity **and** quality. On weak models the extra "cognitive burden" can *reduce* quality (§B).
- **Factuality & safety preserved** (§G.7–G.8). **Cost:** N candidates ≈ N× tokens/latency vs. a single response (§B) — the main downside.

---

## 2. Current state in the AI Orchestrator

### 2.1 What exists
| Component | File | Reality |
|---|---|---|
| `VerbalizedSamplingPipeline` | [`reasoning/ara_pipelines.py:3094`](../orchestrator/reasoning/ara_pipelines.py) | Implemented, **registered** in `PipelineFactory` (`:4175`) |
| `BrainstormingPipeline` ("VS-Multi-ish") | [`reasoning/ara_pipelines.py:2910`](../orchestrator/reasoning/ara_pipelines.py) | Implemented, registered (`:4174`) |
| MAP-Elites (quality-diversity) | [`engine_core/stages/map_elites.py`](../orchestrator/engine_core/stages/map_elites.py) | Implemented; seeds via a list-level prompt |
| ARA retry dispatch | [`engine_core/stages/self_consistency.py:74`](../orchestrator/engine_core/stages/self_consistency.py) | Calls `ara_strategy.get_retry_method()` |
| Structured JSON output | `USE_JSON_SCHEMA_RESPONSES`, `response_format` (`openrouter_models.json`) | **Supported** — we can *enforce* the `{text, probability}` schema |

### 2.2 The five gaps between our VS and the paper's VS

**G1 — It is never invoked (dead code).** `default_methods` and `retry_methods` map task types only to PERSUASION_DEFENSE / SOT / JURY / MULTI_PERSPECTIVE / DEBATE / COVE ([`ara_execution_strategy.py:41-59`](../orchestrator/reasoning/ara_execution_strategy.py)). Nothing maps to `VERBALIZED_SAMPLING` or `BRAINSTORMING`. And per [`ARA_IMPLEMENTATION_PLAN.md:12`](ARA_IMPLEMENTATION_PLAN.md), the ARA methods are "registered in `PipelineFactory`, but **none are wired into the core execution path**."

**G2 — The generation prompt is *list-level*, not *distribution-level*.** [`ara_pipelines.py:3139-3152`](../orchestrator/reasoning/ara_pipelines.py): *"Generate exactly k distinct, plausible candidate answers… Each must be substantively different."* No per-item probability is requested in the generation call; probability **defaults to `1.0/k`** (uniform) at `:3164-3166`. By Claim 2 this is exactly the construction that yields a *uniform* distribution — the paper's **Sequence baseline**, not VS. The single most important fix.

**G3 — Probability semantics are inverted.** A *separate* call ([`:3171-3193`](../orchestrator/reasoning/ara_pipelines.py)) asks a "probability calibration expert" for *"the probability that it is the correct/optimal solution"* and then **keeps the highest-probability candidate**. The paper's probability is **typicality** (how representative under `π_ref`), used to *enable* diversity — especially to sample the **low**-probability tail. Selecting the *most typical/optimal* candidate **re-introduces the typicality bias the paper is trying to remove.**

**G4 — No tail sampling / diversity knob.** The docstring advertises a "TAIL" mode but only `K_DEFAULT=5` and `QUALITY_THRESHOLD=0.15` exist; the threshold is a *post-hoc selection cutoff* ([`:3216-3231`](../orchestrator/reasoning/ara_pipelines.py)), not the paper's generation-time *"probability < threshold"* instruction.

**G5 — Synthesis collapses diversity back to one answer.** [`:3233-3263`](../orchestrator/reasoning/ara_pipelines.py) merges/picks one output. For a task that needs a single deliverable that's fine — but combined with G3 it means we generate near-duplicates and then pick the most typical one, i.e. ~the direct-prompting mode at k× the cost.

**Separately, the mainline default path is fully mode-collapsed by design:** `GenerateStage` does a single call at **`temperature=0.3`** ([`generate.py:58-66`](../orchestrator/engine_core/stages/generate.py)). Low temperature + instance-level prompt is the textbook mode-collapse setting. The generate→critique→revise→evaluate loop then *re-generates from the same mode* on retry.

---

## 3. Where Verbalized Sampling adds *real* value (ranked)

Ranking = (paper-grounded impact) × (fit to an existing stage) ÷ (effort + risk).

### ⭐ B. Fix MAP-Elites seeding — *highest ROI, smallest change*
- **Problem today:** `MAPElitesPipeline._initialize` ([`map_elites.py:80`](../orchestrator/engine_core/stages/map_elites.py)) seeds the quality-diversity grid with a **list-level** "generate 9 diverse variants" prompt, and scores them with a **trivial heuristic** (`_heuristic_score`: line count + avg line length, `:112`). Garbage-diverse seeds → weak grid coverage.
- **Why VS fits perfectly:** MAP-Elites *is* a diversity-search algorithm; its quality depends entirely on seed diversity. The paper exists to produce exactly this. Replace `_initialize` with a **VS distribution-level** call (`{text, probability}`), and use a **low tail threshold** to deliberately seed unconventional cells of the grid. The paper explicitly names QD/RL exploration and hypothesis generation as target use cases (§C, Inference-time Scaling).
- **Effort:** ~1 prompt + parser change in one file. **Risk:** isolated; MAP-Elites is already opt-in.

### ⭐ C. VS for test / synthetic-data generation — *strongest empirical backing*
- **Where:** any path that asks a model to *generate test cases* (TDD, `tdd_enabled` flag) or eval/seed data.
- **Why:** §8 Table 4 is the headline engineering result — VS synthetic data measurably improves *downstream* performance, and direct prompting can *hurt*. For us: a VS test-generator produces genuinely diverse cases; **tail samples = edge cases**, which is exactly what raises real coverage rather than 5 variations of the happy path.
- **Effort:** medium (new generator helper or a `data_extraction`/test task variant). **Risk:** low — more diverse tests can only help; quality gate still applies.

### ⭐ E. VS-tail on the self-consistency retry path — *the principled escape hatch*
- **Problem today:** `EnhancedSelfConsistencyStage` ([`self_consistency.py`](../orchestrator/engine_core/stages/self_consistency.py)) retries with a **fallback model** when score < 0.7. But an instance-level prompt collapses to the *same mode* regardless of model — retrying the same approach.
- **Why VS:** when a task fails repeatedly it's usually stuck in one (wrong) approach. A **VS-tail** call — *"give me k solutions, each with probability < 0.10"* — is the paper's exact tool for escaping a local optimum by sampling unconventional-but-plausible approaches. This is the §C "richer exploration in RL / action-space" direction applied to our retry loop.
- **Effort:** small — map the existing (registered) `VERBALIZED_SAMPLING` method into `retry_methods` *after* a model-swap retry already failed, **but first fix G2–G4** so it's real VS. **Risk:** low (only fires on already-failing tasks, behind the ARA flag).

### D. Diverse candidate generation on the mainline (best-of-k done right)
- Turn `GenerateStage` into an optional **VS-first** stage: one distribution-level call → k diverse candidates → existing `EvaluateStage` selects the best. We already own the selection machinery (`EvaluatorService`, 2-pass self-consistency). The paper shows VS-Standard gives a **better diversity-quality Pareto than the list/Sequence baseline at the same N-token budget** (H.1). Converts *sequential* retries into *parallel* exploration.
- **Effort:** medium; **Risk:** medium (touches the hot path + cost). Gate behind a flag and restrict to STANDARD/PREMIUM models (see H).

### F. Decomposition diversity
- `decompose_project` produces one task breakdown. VS could propose k *strategies* (test-first, vertical-slice, monolith-first…) with probabilities; planner evaluates and picks. Aligns with the existing Multi-Perspective idea in [`ARA_INTEGRATION_ANALYSIS.md`](ARA_INTEGRATION_ANALYSIS.md). **Higher risk** (decomposition quality gates everything downstream) → behind a flag, off by default.

### G. Diverse research/sub-queries
- Already noted in [`ARA_INTEGRATION_ANALYSIS.md:104-106`](ARA_INTEGRATION_ANALYSIS.md). VS generates search queries from different angles; backed by the open-ended-QA coverage result (§7).

### H. Routing implication (cheap config win, do alongside any of the above)
- The scale trend (§B, Fig 4e-f) is directly actionable for `ModelSelector`/`model_routing`:
  - Apply **VS-CoT / VS-Multi** preferentially on **STANDARD/PREMIUM** models (they turn the "cognitive burden" into a quality *gain*).
  - On **FREE/ULTRA-LOW/BUDGET** models, prefer **VS-Standard** or skip VS — the burden can *lower* quality on weak models.
- Encode as: VS variant is a function of the model tier, not a global constant.

---

## 4. A faithful design (drop-in corrections)

### 4.1 The generation prompt should be distribution-level (fixes G2/G3/G4)
Single call, structured output enforced via `USE_JSON_SCHEMA_RESPONSES`:

```text
System: You are a helpful assistant. For the task below, generate {k} possible
responses, each in its own object. Each object must include:
  - "text": the response only.
  - "probability": the estimated probability from 0.0 to 1.0 of this response
    given the input prompt (relative to the full distribution).
Return ONLY JSON: {"responses": [{"text": "...", "probability": 0.0}, ...]}
[tail mode] Sample from the distribution such that each response's probability
is below {threshold}.

User: {task.prompt}
```
- Use **Explicit** probability for VS-Standard; **Confidence** for VS-Multi (H.3).
- JSON schema: `responses: array<{text: string, probability: number}>` — kills the brittle `_extract_json` regex path ([`map_elites.py:202`](../orchestrator/engine_core/stages/map_elites.py), [`ara_pipelines.py` `_extract_json`](../orchestrator/reasoning/ara_pipelines.py)).

### 4.2 Use probability for *diversity*, not for *picking the typical one*
- **Selection tasks** (need 1 deliverable): generate diverse via VS → score with the **real `EvaluatorService`** (correctness), not the verbalized typicality → pick best. Keep these two scores distinct. (Today they're conflated, which is G3.)
- **Exploration tasks** (MAP-Elites seeds, retry escape, test cases): **prefer low-probability tail** samples; do **not** select the highest-probability candidate.

### 4.3 Combine with, don't replace, decoding params
- VS at the orchestrator's existing temperatures is fine; for max diversity raise top-p toward ~0.95 (H.2) on the VS call. Don't lower temperature to 0.3 for VS generation (that's tuned for deterministic single-shot codegen).

### 4.4 Cost control
- VS is ~k× tokens. Keep `k=5` (paper default; H.1 shows diminishing returns and quality drop for large k). Restrict mainline VS (opportunity D) to tasks/models where the trade-off pays, via the routing rule in H.

---

## 5. Suggested roadmap (TDD, feature-flagged)

| Phase | Change | Files | Flag |
|---|---|---|---|
| 1 | Make VS faithful: distribution-level prompt + Explicit probability + tail threshold + json_schema; stop defaulting probability to `1/k`; separate typicality from quality | [`reasoning/ara_pipelines.py:3094`](../orchestrator/reasoning/ara_pipelines.py) | existing ARA |
| 2 | **B** — reseed MAP-Elites with the Phase-1 VS generator (tail-biased) | [`engine_core/stages/map_elites.py`](../orchestrator/engine_core/stages/map_elites.py) | reuse MAP-Elites gating |
| 3 | **E** — add `VERBALIZED_SAMPLING` to `retry_methods` as the *second-stage* escape after a failed model-swap retry | [`reasoning/ara_execution_strategy.py:53`](../orchestrator/reasoning/ara_execution_strategy.py) | `ORCH_*` (new) |
| 4 | **C** — VS test/synthetic-data generator; measure downstream (mirror §8) | new helper + TDD path | `tdd_enabled` |
| 5 | **H** — tier-aware VS variant selection | `model_routing.py` / `ModelSelector` | — |
| 6 | **D** (optional) — VS-first `GenerateStage` for STANDARD/PREMIUM | [`engine_core/stages/generate.py`](../orchestrator/engine_core/stages/generate.py) | new flag, default off |

Each phase: RED test first (e.g., "generation call requests a `probability` field per item"; "tail mode forwards `threshold` into the prompt"; "MAP-Elites seeds come from VS, not the list prompt"), then GREEN.

---

## 6. Risks, limits, and when *not* to use VS

- **Cost/latency:** k× tokens. Don't put VS on every task — only exploration/diversity-bound stages, and gate by model tier.
- **Weak models:** VS can *reduce* quality on FREE/BUDGET models (§B). Honor the tier rule (H).
- **Single-answer determinism:** for strict-format codegen where there's one right answer, mode "collapse" is *desirable* — VS adds cost for no gain. Keep it off the deterministic codegen path.
- **Probability is self-reported, not calibrated.** Treat verbalized probabilities as *ordering hints* for tail sampling, never as ground-truth likelihoods.
- **Don't double-count diversity:** if a stage already runs Multi-Perspective/Debate, layering VS may be redundant.

---

## 7. Bottom line

The paper's contribution to *this* codebase is less "a new feature" and more **"a correctness fix plus three high-value placements."** We have the scaffolding (`VerbalizedSamplingPipeline`, MAP-Elites, ARA retry, json_schema), but the VS we shipped is the very *list-level* construction the paper proves doesn't work, and it's not wired in. Fixing it to be a true distribution-level prompt and aiming it at **MAP-Elites seeding (B)**, **test/synthetic-data generation (C)**, and **the self-consistency tail-escape (E)** turns a dead, mis-specified pipeline into three measurable wins — at the cost of a few prompts, a schema, and a feature flag.
