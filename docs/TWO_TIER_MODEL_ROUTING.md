# AI Orchestrator — Two-Tier Model Routing Configuration
>
> Author: Georgios-Chrysovalantis Chatzivantsidis  
> Date: 2026-05-23  
> Version: 1.0  
>
> Defines budget and premium model tiers for every task type, agent role,
> and pipeline stage in the AI Orchestrator.

---

## Overview

All 88 models are split into two tiers:

| Tier | Budget | Quality | Use Case |
|------|--------|---------|----------|
| **Budget** | < $1.00/M input | High enough for most tasks | Default for standard execution |
| **Premium** | $1.00-$25/M input | Best-in-class | Critical tasks, architecture, security review |

---

## Code Generation (CODE_GEN)

### Budget Tier (input < $1.00/M)

| Model | Input/M | Output/M | Context | Strength |
|-------|---------|----------|---------|----------|
| `XIAOMI_MIMO_V2_FLASH` | $0.10 | $0.30 | 262K | MiMo, very popular on OpenRouter |
| `PHI_4` | $0.07 | $0.14 | 128K | Ultra-cheap, surprisingly good code |
| `QWEN_3_5_FLASH` | $0.065 | $0.26 | 1M | Cheapest in class, large context |
| `GLM_4_5_AIR` | $0.13 | $0.85 | 131K | Cheap, strong for its price |
| `BAIDU_ERNIE_4_5_THINKING` | $0.07 | $0.28 | 131K | Chinese model, thinking variant, ultra-cheap |
| `DEEPSEEK_V4_FLASH` | $0.15 | $0.30 | 1M | Fast, large context, great value |
| `GPT_4O_MINI` | $0.15 | $0.60 | 128K | Solid all-rounder, reliable |
| `QWEN_3_CODER_FLASH` | $0.20 | $0.98 | 1M | Specialized coder, fast |
| `CODESTRAL_2508` | $0.30 | $0.90 | 256K | Mistral's best coder |
| `QWEN_3_CODER_PLUS` | $0.65 | $3.25 | 1M | Best open-source coder |

**Recommended default:** `CODESTRAL_2508` — best quality/$ ratio for code.
**Ultra-budget:** `QWEN_3_5_FLASH` ($0.065/M) — cheapest.
**1M context value:** `XIAOMI_MIMO_V2_5` ($0.40/M) — 1M context, 22% of OpenRouter traffic.
**For large projects:** `DEEPSEEK_V4_FLASH` (1M context, cheap, fast).

### Premium Tier (input $1.00-$5.00/M)

| Model | Input/M | Output/M | Context | Strength |
|-------|---------|----------|---------|----------|
| `GPT_5` | $1.25 | $10.00 | 200K | Top-tier general purpose |
| `GPT_5_3_CODEX` | $1.75 | $14.00 | 400K | Specialized coding model |
| `GPT_5_2_CODEX` | $1.75 | $14.00 | 400K | Coding variant |
| `CLAUDE_SONNET_4_6` | $3.00 | $15.00 | 200K | Excellent code, fast |
| `GPT_4O` | $2.50 | $10.00 | 128K | Well-rounded, widely tested |
| `GPT_5` | $1.25 | $10.00 | 200K | **Best premium coder** |

**Recommended for critical code:** `GPT_5` ($1.25/M) or `CLAUDE_SONNET_4_6` ($3.00/M).

---

## Complex Reasoning (REASONING)

### Budget Tier

| Model | Input/M | Output/M | Context | Strength |
|-------|---------|----------|---------|----------|
| `DEEPSEEK_REASONER` | $0.28 | $0.42 | 128K | **Best budget reasoner** |
| `MINIMAX_M2_5` | $0.30 | $1.20 | 128K | Strong reasoning |
| `QWEN_3_6_PLUS` | $0.33 | $1.95 | 1M | Good all-rounder, large context |
| `MINIMAX_M1` | $0.40 | $2.20 | 1M | 1M context, good for long-form |

**Recommended: `DEEPSEEK_REASONER`** — best reasoning at $0.28/M.
**Budget alternative:** `QWEN_3_5_FLASH` ($0.065/M) — ultra-cheap, fast responses.

### Premium Tier

| Model | Input/M | Output/M | Context | Strength |
|-------|---------|----------|---------|----------|
| `GPT_5` | $1.25 | $10.00 | 200K | Top-tier reasoning |
| `GEMINI_PRO` | $1.25 | $10.00 | 200K | Strong reasoning, multi-modal |
| `O3` | $2.00 | $8.00 | 200K | OpenAI's dedicated reasoning |
| `XAI_GROK_4_20` | $1.25 | $2.50 | **2M** | **Newest Grok, 2M context, multi-agent** |
| `XAI_GROK_4_3` | $1.25 | $2.50 | **1M** | Updated Grok, 60% cheaper than Grok 4 |
| `XAI_GROK_4` | $3.00 | $15.00 | 256K | Older Grok, still capable |
| `GPT_5` | $1.25 | $10.00 | 200K | **Best premium for hard problems** |

**Recommended: `O3`** ($2.00/M) for dedicated reasoning, or `GPT_5` ($1.25/M) for general.

---

## Code Review (CODE_REVIEW)

### Budget Tier

| Model | Input/M | Output/M | Strength |
|-------|---------|----------|----------|
| `LLAMA_4_MAVERICK` | $0.17 | $0.17 | Extremely cheap, decent review |
| `QWEN_3_6_PLUS` | $0.33 | $1.95 | Good review, large context |

### Premium Tier

| Model | Input/M | Output/M | Strength |
|-------|---------|----------|----------|
| `CLAUDE_SONNET_4_6` | $3.00 | $15.00 | Excellent security review |
| `GPT_4O` | $2.50 | $10.00 | Thorough, structured feedback |
| `XAI_GROK_3` | $3.00 | $15.00 | Good review, different perspective |
| `SONAR_PRO` | $3.00 | $15.00 | Web-search enhanced review |
| `CLAUDE_SONNET_4_6` | $3.00 | $15.00 | **Best code review** |

---

## Agents (per role routing)

Each agent role gets its own model routing:

### DeveloperAgent (code generation)

| Phase | Budget | Premium |
|-------|--------|---------|
| Initial generation | `CODESTRAL_2508` | `GPT_5` |
| Self-correction (retry 1) | `GPT_4O_MINI` | `GPT_5` |
| Self-correction (retry 2) | `DEEPSEEK_V4_FLASH` | `CLAUDE_SONNET_4_6` |

### ArchitectAgent (design decisions)

| Phase | Budget | Premium |
|-------|--------|---------|
| Initial architecture | `DEEPSEEK_REASONER` | `GPT_5` |
| Multi-model deliberation | Jury (3 models) | Jury (3 premium models) |

### TesterAgent (test generation)

| Phase | Budget | Premium |
|-------|--------|---------|
| Test writing | `CODESTRAL_2508` | `GPT_5` |
| Test review | `QWEN_3_6_PLUS` | `CLAUDE_SONNET_4_6` |

### ReviewerAgent (code audit)

| Phase | Budget | Premium |
|-------|--------|---------|
| Security audit | `SONAR_PRO` | `SONAR_PRO` |
| Code review | `QWEN_3_6_PLUS` | `CLAUDE_SONNET_4_6` |

---

## Pipeline Stage Routing

| Stage | Budget Model | Premium Model |
|-------|-------------|---------------|
| GenerateStage | `CODESTRAL_2508` | `GPT_5` |
| CritiqueStage | `QWEN_3_6_PLUS` | `CLAUDE_SONNET_4_6` |
| EvaluateStage | `Qwen/Qwen3.6-plus` | Jury (3 premium models) |
| ValidateStage | Deterministic only | Deterministic only |
| PersuasionDefenseStage | `DEEPSEEK_REASONER` | `GPT_5` |
| SelfConsistencyStage | Fallback chain | Premium fallback chain |

---

## ARA Method → Model Mapping
Directly from Reasoner project analysis.

| ARA Method | Budget Models | Premium Models |
|------------|---------------|----------------|
| MultiPerspective | `DEEPSEEK_V4_FLASH` + `QWEN_3_6_PLUS` | `MISTRAL_LARGE_3` + `CLAUDE_SONNET_4_6` |
| Debate | `QWEN_3_6_PLUS` (2 judges) | `CLAUDE_SONNET_4_6` + `GPT_5_2_CODEX` |
| Jury | `GPT_4O` + `DEEPSEEK_V4_FLASH` + `QWEN_3_6_PLUS` | `GPT_4O` + `CLAUDE_SONNET_4_6` + `GEMINI_2_FLASH` |
| CoVE | `DEEPSEEK_REASONER` + `QWEN_3_6_FLASH` | `O3` + `GPT_5_2_CODEX` |
| PersuasionDefense | `DEEPSEEK_REASONER` + `QWEN_3_6_FLASH` | `O3` + `CLAUDE_SONNET_4_6` |
| Iterative | `DEEPSEEK_V3` + `GPT_4O_MINI` | `GPT_4O` + `CLAUDE_SONNET_4_6` |
| Research | `SONAR_PRO` + `GPT_4O_MINI` | `SONAR_DEEP_RESEARCH` + `GPT_5_2_CODEX` |
| Scientific | `DEEPSEEK_REASONER` | `O3` |
| PreMortem | `DEEPSEEK_V3` + `QWEN_3_6_PLUS` | `CLAUDE_SONNET_4_6` + `GPT_5_2_CODEX` |
| Delphi | `GPT_4O` + `DEEPSEEK_V4_FLASH` + `QWEN_3_6_PLUS` | `GPT_4O` + `CLAUDE_SONNET_4_6` + `MISTRAL_LARGE_3` |
| Brainstorming | `GPT_4O_MINI` + `QWEN_3_CODER_FLASH` | `GPT_4O` + `QWEN_3_6_PLUS` |
| Socratic | `GPT_4O_MINI` + `DEEPSEEK_V3` | `GPT_4O` + `CLAUDE_SONNET_4_6` |
| Dialectical | `DEEPSEEK_V4_FLASH` + `QWEN_3_6_PLUS` | `CLAUDE_SONNET_4_6` + `GPT_5_2_CODEX` |
| Analogical | `MISTRAL_LARGE_3` + `QWEN_3_6_PLUS` | `GPT_5_2_CODEX` + `CLAUDE_SONNET_4_6` |
| VerbalizedSampling | `DEEPSEEK_V3` + `GPT_4O_MINI` | `GPT_4O` + `CLAUDE_SONNET_4_6` |
| SoT | `CODESTRAL_2508` (parallel sub-problems) | `GPT_5_2_CODEX` (parallel sub-problems) |
| ToT | `DEEPSEEK_V4_FLASH` | `CLAUDE_SONNET_4_6` |
| Self-Discover | `DEEPSEEK_V4_FLASH` + `QWEN_3_6_PLUS` | `GPT_5_2_CODEX` + `CLAUDE_SONNET_4_6` |


## Fallback Chains

When the primary model fails, fall through to these:

### Budget Fallback Chain

```
CODESTRAL_2508 → DEEPSEEK_V4_FLASH → GPT_4O_MINI → QWEN_3_CODER_FLASH → PHI_4
```

**Cost range:** $0.07 → $0.30/M input  
**Ends at PHI_4** ($0.07/M) — always completes.

### Premium Fallback Chain

```
GPT_5 → CLAUDE_SONNET_4_6 → GEMINI_PRO → O3 → GPT_4O
```

**Cost range:** $1.25 → $3.00/M input  
**Ends at GPT_4O** ($2.50/M) — always affordable.

---

## Implementation

The routing tables in `models.py` should be updated with these priorities:

```python
# models.py — ROUTING_TABLE updates

# CODE_GEN — Budget then Premium
ROUTING_TABLE[TaskType.CODE_GEN] = [
    Model.CODESTRAL_2508,      # $0.30/M (budget primary)
    Model.DEEPSEEK_V4_FLASH,   # $0.15/M (budget fallback)
    Model.GPT_4O_MINI,         # $0.15/M (general fallback)
    Model.QWEN_3_CODER_PLUS,   # $0.65/M (premium coder)
    Model.GPT_5,               # $1.25/M (premium primary)
    Model.GPT_5,               # $1.25/M (premium fallback)
]

# REASONING — Budget then Premium
ROUTING_TABLE[TaskType.REASONING] = [
    Model.DEEPSEEK_REASONER,    # $0.28/M (budget primary)
    Model.QWEN_3_6_PLUS,        # $0.33/M (budget fallback)
    Model.GPT_5,                # $1.25/M (premium fallback)
    Model.O3,                   # $2.00/M (premium primary)
    Model.CLAUDE_SONNET_4_6,    # $3.00/M (premium ultimate)
]

# CODE_REVIEW
ROUTING_TABLE[TaskType.CODE_REVIEW] = [
    Model.LLAMA_4_MAVERICK,     # $0.17/M (budget)
    Model.QWEN_3_6_PLUS,        # $0.33/M (budget)
    Model.CLAUDE_SONNET_4_6,    # $3.00/M (premium)
    Model.CLAUDE_SONNET_4_6,    # $3.00/M (premium ultimate)
]

# EVALUATE
ROUTING_TABLE[TaskType.EVALUATE] = [
    Model.QWEN_3_6_PLUS,        # $0.33/M (budget)
    Model.JURY,                 # Multi-model (premium)
    Model.CLAUDE_SONNET_4_6,    # $3.00/M (premium ultimate)
]
```
