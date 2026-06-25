# Reasoning Models, Thinking, and Temperature — Per-Phase Policy

> Single source of truth: [`orchestrator/domain/phase_policy.py`](../orchestrator/domain/phase_policy.py).
> This doc is the *why*; the module is the *what* the code reads at runtime.

## TL;DR decision table

| Pipeline phase | Reasoning model? | Thinking / effort | Temperature\* | Rationale |
|----------------|------------------|-------------------|---------------|-----------|
| **Decompose / plan** | ✅ prefer | ON · high | 0.2 | Multi-step task breakdown; depth pays off, but plan must be near-deterministic |
| **Generate (code)** | ⚪ optional | OFF (ON for hard algos) | 0.2 | Correct code needs low variance; reasoning only for genuinely hard logic |
| **Generate (creative)** | ❌ avoid | OFF | 0.8 | Reasoning models write stiff prose; want voice + diversity |
| **Critique / review** | ✅ prefer | ON · high | 0.1 | Finding subtle flaws is the canonical reasoning win; reproducible |
| **Revise** | ⚪ optional | OFF (light) | 0.2 | Applying known fixes; depth rarely needed |
| **Evaluate / score** | ✅ prefer | ON · high | 0.1 | Adversarial judgment (ASSUME BROKEN); must be consistent run-to-run |
| **Extract (structured)** | ❌ avoid | OFF | 0.0 | Exact, deterministic field extraction; reasoning is pure waste |
| **Summarize** | ❌ avoid | OFF | 0.3 | Faithful but fluent; no multi-step reasoning needed |
| **Diversity / sampling (VS)** | ⚪ optional | OFF | 0.8–0.9 | Need spread across candidates, not a single best answer |

\* **Temperature only applies to non-reasoning models.** o-series (o1/o3/o4), GPT-5 Pro, and
similar **forbid or ignore** `temperature` — the client omits it for reasoning models
(`_call_reasoning_model`). DeepSeek-R1 recommends 0.5–0.7; Grok-4 accepts temperature.

## When to use a reasoning model

Reasoning models trade **latency + token cost** for deeper multi-step inference. Worth it when
the task is **verification-heavy or compositional**:

- **Decomposition / planning** — breaking a project into a correct dependency graph.
- **Critique / code review** — finding subtle bugs, security holes, missing edge cases.
- **Evaluation / scoring** — the adversarial "assume broken" judge (see ENH-1).
- **Hard algorithmic code** — non-trivial logic, math, concurrency, proofs.
- **Root-cause debugging** — tracing failure across layers.

**Not worth it** (latency + token blowout for no quality gain):

- Structured extraction, classification, formatting, simple boilerplate.
- Summarization.
- Creative writing (reasoning models are measurably stiffer).

## The thinking attribute (`reasoning` / `:thinking`)

OpenRouter exposes reasoning control two ways:

1. **Endpoint variant** `model:thinking` — forces the thinking endpoint of a hybrid model.
2. **`reasoning` request field** — `{"effort": "high|medium|low"}` or `{"exclude": true}`.

Policy:

- **Effort `high`** — decompose, critique, evaluate (depth is the whole point).
- **Effort `medium`** — hard code generation.
- **OFF / `exclude`** — extraction, summarization, creative, simple revise.
- **`reasoning.exclude = true`** — when the caller only needs the final answer and the chain
  of thought would pollute structured (JSON) output. The evaluator already strips `<think>`
  blocks defensively; setting `exclude` avoids paying to *return* the CoT (you still pay to
  generate it). Use it on structured-output reasoning calls.

## Temperature — first principles

- **0.0** — must be exact/reproducible: structured extraction, deterministic tools.
- **0.1** — judgment that must be consistent run-to-run: review, evaluation.
- **0.2** — mostly-deterministic with a little flexibility: code gen, planning, revise.
- **0.3** — faithful-but-fluent: summarization.
- **0.7–0.9** — diversity/voice: creative writing, Verbalized-Sampling candidate spread.

Avoid scattering raw temperature literals across services. Read
[`phase_policy.temperature_for(phase, task_type)`](../orchestrator/domain/phase_policy.py)
so the policy lives in exactly one place.

## Reasoning-model roster

The authoritative set is `ModelRegistry.REASONING_MODELS`. It must track the live OpenRouter
catalogue — stale entries (e.g. a removed `grok-4-mini`) cause mis-routing, and missing 2026
models (gpt-5.2, grok-4.3, minimax-m3, the Nemotron-3 reasoning tier, qwen `*-thinking`) lose
the reasoning path. Kept in sync with `scripts/audit_openrouter_models.py`.
