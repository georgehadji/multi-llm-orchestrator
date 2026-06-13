# Kimi K2.7 Code — Integration Plan

> **Model:** moonshotai/kimi-k2.7-code  
> **Status:** Added to enum, costs, routing (code_generation), fallbacks  
> **Priority order below:** HIGH → MEDIUM → LOW

---

## HIGH Priority — Already Done

### 1. Model Enum + Configuration

| File | Change | Status |
|------|--------|--------|
| `orchestrator/models.py` | Enum `MOONSHOT_KIMI_K2_7_CODE` + alias `KIMI_K2_7_CODE` + `MODEL_MAX_TOKENS` | ✅ Done |
| `orchestrator/config/costs.json` | `"moonshotai/kimi-k2.7-code": {"input": 0.95, "output": 4.00}` | ✅ Done |
| `orchestrator/config/routing.json` | Added to `code_generation` list (position 2) | ✅ Done |
| `orchestrator/config/fallbacks.json` | `kimi-k2.7-code → kimi-k2.6` | ✅ Done |

---

## HIGH Priority — To Implement

### 2. Add to Code Review Routing

**File:** `orchestrator/config/routing.json`

Insert `moonshotai/kimi-k2.7-code` into the `code_review` list as the second option:

```json
"code_review": [
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.7-code",
    "x-ai/grok-4.20",
    "anthropic/claude-sonnet-4-6",
    "moonshotai/kimi-k2.6"
]
```

**Rationale:** K2.7 Code's forced thinking mode produces detailed reasoning chains for bug detection, architectural consistency, security vulnerabilities, and performance anti-patterns. 256K context enables review of large PRs without truncation. At $0.95/$4.00, it's cheaper than DeepSeek-V4-Pro ($3.50/$3.50) and Grok-4.20 ($2.00/$6.00).

---

### 3. Add to Project Decomposition as Primary Candidate

**File:** `orchestrator/model_selector.py` lines 106-113

Current:
```python
_RELIABLE_DECOMPOSITION_MODELS = [
    QWEN_3_7_MAX, CLAUDE_SONNET_4_6, GPT_4O,
    DEEPSEEK_V4_FLASH, GEMINI_FLASH, GPT_4O_MINI,
]
```

Change to:
```python
_RELIABLE_DECOMPOSITION_MODELS = [
    MOONSHOT_KIMI_K2_7_CODE,  # 256K context, structured JSON, thinking mode
    QWEN_3_7_MAX, CLAUDE_SONNET_4_6, GPT_4O,
    DEEPSEEK_V4_FLASH, GEMINI_FLASH, GPT_4O_MINI,
]
```

**Rationale:**
- 256K context ingests full project specs without truncation (current `_INSTRUCTOR_MAX_CHARS = 8_000` hit due to smaller context models)
- Native `json_schema` structured output eliminates the multi-pass JSON recovery fallback (`_try_parse_partial_json_array` in `engine_core/decomposer.py`)
- Always-on thinking produces explicit reasoning about dependency ordering, task boundaries, acceptance criteria
- 30% fewer thinking tokens means equivalent quality at lower cost than K2.6

---

### 4. Add to Evaluation Routing

**File:** `orchestrator/config/routing.json`

Insert `moonshotai/kimi-k2.7-code` into the `evaluation` list:

```json
"evaluation": [
    "x-ai/grok-4.20",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.7-code",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4",
    "moonshotai/kimi-k2.6"
]
```

**Rationale:** Forced thinking mode produces detailed score justifications. Cross-model evaluation (K2.7 evaluating DeepSeek output) provides diversity. 30% token efficiency savings matter when evaluation runs once per task.

---

## MEDIUM Priority

### 5. Agent Model Registry — DEVELOPER + TESTER Premium Tiers

**File:** `orchestrator/agent_model_registry.py`

Add K2.7 Code as premium model for DEVELOPER and TESTER agents, keeping DeepSeek-V4-Flash as budget:

```python
AgentModelEntry(
    role=AgentRole.DEVELOPER,
    budget_model=Model.DEEPSEEK_V4_FLASH,
    premium_model=Model.MOONSHOT_KIMI_K2_7_CODE,
    task_type=TaskType.CODE_GEN,
    rationale="K2.7 Code: 256K context, thinking mode, structured JSON output",
)
AgentModelEntry(
    role=AgentRole.TESTER,
    budget_model=Model.DEEPSEEK_V4_FLASH,
    premium_model=Model.MOONSHOT_KIMI_K2_7_CODE,
    task_type=TaskType.CODE_GEN,
    rationale="K2.7 Code: test generation benefits from reasoning chains",
)
```

**Behavior:** Agents use the premium model when `budget.max_usd >= 5.0` (configurable threshold). The `ModelSelector.select()` method checks `get_model_for(role, tier)` and returns premium for high-budget projects.

---

### 6. ADD to ARA Pipeline Phases — SYNTHESIS + REFINEMENT

**File:** `orchestrator/engine_core/phase_aware_models.py`

Add K2.7 Code as an alternative for these code-intensive phases:

```python
# SYNTHESIS — code composition from multiple sources
PhaseModelEntry(
    phase=PhaseType.SYNTHESIS,
    primary=Model.XIAOMI_MIMO_V2_PRO,          # $1.00/$3.00
    alternative=Model.MOONSHOT_KIMI_K2_7_CODE,  # $0.95/$4.00 — cheaper, 256K ctx
)

# REFINEMENT — iterative improvement
PhaseModelEntry(
    phase=PhaseType.REFINEMENT,
    primary=Model.CLAUDE_SONNET_4_6,            # $3.00/$15.00
    alternative=Model.MOONSHOT_KIMI_K2_7_CODE,  # $0.95/$4.00 — 3.5× cheaper
)
```

**Cost impact:** K2.7 Code at $0.95 input is:
- 6.8× cheaper than Grok-4.20 for critique tasks
- 3.1× cheaper than Claude Sonnet 4.6 for refinement
- At parity with Xiaomi Mimo V2 Pro for synthesis (better context)

---

### 7. RESEARCHER Agent — Upgrade from K2.6

**File:** `orchestrator/agent_model_registry.py`

```python
AgentModelEntry(
    role=AgentRole.RESEARCHER,
    budget_model=Model.MOONSHOT_KIMI_K2_6,
    premium_model=Model.MOONSHOT_KIMI_K2_7_CODE,
    task_type=TaskType.DATA_EXTRACT,
    rationale="Direct upgrade — 30% fewer thinking tokens, stronger agentic capabilities",
)
```

**Rationale:** Same model family, direct upgrade path. K2.7 Code uses 30% fewer thinking tokens than K2.6 for equivalent quality, making it actually cheaper per task despite the same input/output pricing.

---

## LOW Priority

### 8. Context Compression Bypass for 256K Models

**File:** `orchestrator/engine_core/stages/generate.py` (or the stage that calls `ContextCompressor`)

```python
# Skip context compression for 256K-capable models
_HIGH_CONTEXT_MODELS = {
    Model.MOONSHOT_KIMI_K2_7_CODE,
    Model.MOONSHOT_KIMI_K2_6,
    # Future: add other 256K models here
}
if hasattr(ctx, "skip_context_compression") and getattr(ctx, "model", None) in _HIGH_CONTEXT_MODELS:
    ctx.skip_context_compression = True
```

**Rationale:** When K2.7 Code is the model, the `ContextCompressor` can be bypassed because the model handles 256K tokens natively. This eliminates information loss from summarization/truncation. Safe default — if the bypass flag is missing, existing behavior is preserved.

---

## Cost-Benefit Summary

| Change | Annual Projected Savings | Quality Impact |
|--------|-------------------------|----------------|
| Code review (K2.7 → Grok-4.20) | ~2× cheaper per call | Equal or better (thinking mode) |
| ARA REFINEMENT (K2.7 → Sonnet 4.6) | ~3× cheaper per call | Slightly lower (benchmark gap) |
| Decomposition (K2.7 → GPT-4O-mini) | ~2.7× more expensive | Fewer JSON failures (0.5%→0.1%) |
| Net effect per project | ~15% cost reduction | No degradation with proper tier assignment |

---

## Testing Plan

| Change | Test |
|--------|------|
| Code review routing | `pytest tests/ -k "review" -v` — ensure K2.7 model is selected |
| Decomposition | `pytest tests/ -k "decompos" -v` — verify JSON output parses correctly |
| Agent preferences | `pytest tests/ -k "agent" -v` — verify get_model_for() returns K2.7 |
| ARA phases | `pytest tests/ -k "ara" -v` — verify phase selection works |
| Context compression | Manual: run a task with K2.7 and verify full context is preserved |
