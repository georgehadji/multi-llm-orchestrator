---
type: object
cluster: routing
universe: live
status: verified
verified: 2026-10-22 at 26fbe258 (source unchanged since d74829e6)
entity: orchestrator/domain/model_registry.py
---

# ModelRegistry

A class of model-id string constants, tier sets, timeouts and a cost table, with lookup classmethods.

## Why this shape

Written to stop model id strings being scattered around the code. Its header calls it the "single source of truth for all model configurations, timeouts, and costs" (`model_registry.py:7-9`). In practice `models.Model` and `config/*.json` also define models and prices (see [Model](Model.md)).

## Shape

- Constants such as `QWEN_3_CODER`, `DEEPSEEK_V4_PRO`, `CLAUDE_SONNET_5` (`model_registry.py:65-163`).
- `UNAVAILABLE_MODELS`: dead id → live replacement (`model_registry.py:177-255`).
- `TIMEOUT_CONFIG` by prefix, `MODEL_TIMEOUT_OVERRIDES` per model (`:264-289`); `COST_TABLE` (60 entries, `:296`) and `MODEL_MAX_TOKENS` (`:380`).
- Tier sets: `CODING_SPECIALISTS`, `REASONING_MODELS`, `BUDGET_MODELS`, `PREMIUM_MODELS`, `LONG_CONTEXT_MODELS`, `MULTIMODAL_MODELS` (`:461-623`).
- Classmethods: `get_timeout`, `get_cost`, `is_valid_model`, `get_replacement_model`, `is_reasoning_model`, and others (`:629-770`).
- `ModelConfig` dataclass (`:34`); its users were not traced.

Citations: `orchestrator/domain/model_registry.py:34-770`

## Connected to

- **owns:** the constants and sets above.
- **owned-by:** imported through the root shim `orchestrator/model_registry.py` by `application/critique_cycle.py:22` (`is_reasoning_model` at `:303`, `:373`) and `phase_aware_models.py:30`; also used in tests.
- **looks-like-but-is-not:** `models.Model` and `config/costs.json`.

## If you change this

- **Hits:** `critique_cycle.py` (reasoning-model checks), `phase_aware_models.py`, the OpenRouter audit test `tests/unit/test_openrouter_model_audit.py`.
- **Does not hit:** `ROUTING_TABLE`, `FALLBACK_CHAIN`, `models.COST_TABLE`.
- **`is_valid_model` is narrow:** it is true only for ids in this class's 60-entry `COST_TABLE` and not in `UNAVAILABLE_MODELS` (`model_registry.py:688-690`). `config/costs.json` has 94 ids that are not in it, so those ids return `False` here.
- **Duplicate constant:** `GPT_5_CODEX` is assigned twice with the same value (`:88`, `:92`).

## Surfaces

| Surface | Role |
|---|---|
| `critique_cycle.py`, `phase_aware_models.py` | reads |
| `scripts/audit_openrouter_models.py`, tests | reads (not traced) |

## See

- Source: `orchestrator/domain/model_registry.py:49`; shim `orchestrator/model_registry.py`
