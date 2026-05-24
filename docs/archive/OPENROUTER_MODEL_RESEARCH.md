# OpenRouter Model Research Report
**Date:** 2026-04-01  
**Source:** https://openrouter.ai/models

---

## Executive Summary

### ✅ Confirmed Valid Model IDs

| Model ID | Input ($/1M) | Output ($/1M) | Context | Best For |
|----------|--------------|---------------|---------|----------|
| `qwen/qwen-3-coder-next` | $0.12 | $0.75 | 262K | **Coding agents** ⭐ |
| `qwen/qwen-3.5-397b-a17b` | $0.39 | $2.34 | 262K | **SOTA reasoning/coding** ⭐ |
| `deepseek/deepseek-chat` | $0.32 | $0.89 | 164K | **Best value coding** ⭐ |
| `anthropic/claude-3.5-sonnet` | $6.00 | $30.00 | 200K | Premium coding |
| `openai/gpt-4o-mini` | $0.15 | $0.60 | 128K | Budget general |
| `openai/gpt-5` | $1.25 | $10.00 | 400K | Reasoning |
| `openai/gpt-5-codex` | $1.25 | $10.00 | 400K | **Coding specialist** ⭐ |
| `google/gemini-2.5-flash` | $0.30 | $2.50 | 1M+ | Large context |
| `moonshotai/kimi-k2` | $0.57 | $2.30 | 128K | Agentic coding |

---

## ❌ INVALID Model IDs (From Our Codebase)

The following models in our codebase are **NOT AVAILABLE** on OpenRouter:

| Our Model ID | Status | Replacement |
|--------------|--------|-------------|
| `qwen/qwen-3-coder` | ❌ Not available | → `qwen/qwen-3-coder-next` |
| `qwen/qwen-3.5-235b-a22b-thinking-2507` | ❌ Not available | → `qwen/qwen-3.5-397b-a17b` |
| `x-ai/grok-4.20-beta` | ❌ Not available | → `x-ai/grok-2-vision-1212` |
| `x-ai/grok-4.1-fast` | ❌ Not available | → `x-ai/grok-2-1212` |
| `xiaomi/mimo-v2-flash` | ❌ Not available | → `qwen/qwen-2.5-coder-32b-instruct` |
| `xiaomi/mimo-v2-pro` | ❌ Not available | → `qwen/qwen-3.5-397b-a17b` |
| `stepfun/step-3.5-flash` | ❌ Not available | → `deepseek/deepseek-chat` |
| `z-ai/glm-4.7-flash` | ❌ Not available | → `openai/gpt-4o-mini` |
| `z-ai/glm-4.7` | ❌ Not available | → `openai/gpt-5` |
| `z-ai/glm-5` | ❌ Not available | → `anthropic/claude-3.5-sonnet` |
| `z-ai/glm-5-turbo` | ❌ Not available | → `deepseek/deepseek-chat` |
| `aionlabs/aion-2.0` | ❌ Not available | → `meta-llama/llama-3.1-70b-instruct` |
| `nvidia/nemotron-3-super` | ❌ Not available | → `meta-llama/llama-3.1-70b-instruct` |
| `minimax/minimax-m2.7` | ❌ Not available | → `moonshotai/kimi-k2` |
| `google/gemini-3.1-pro` | ❌ Not available | → `google/gemini-2.5-flash` |
| `google/gemini-3.1-flash` | ❌ Not available | → `google/gemini-2.5-flash` |

---

## 🎯 Recommended Models by Task Type

### **Code Generation** (Best to Budget)
| Priority | Model ID | Cost ($/1M) | Why |
|----------|----------|-------------|-----|
| 🥇 Premium | `openai/gpt-5-codex` | $1.25/$10.00 | Optimized for software engineering |
| 🥈 Balanced | `qwen/qwen-3-coder-next` | $0.12/$0.75 | Coding agents, incredible value |
| 🥉 Budget | `deepseek/deepseek-chat` | $0.32/$0.89 | Best value, 15T token training |
| 💰 Ultra | `qwen/qwen-2.5-coder-32b-instruct` | $0.66/$1.00 | Code-specific, 32B params |

### **Code Review / Critique**
| Priority | Model ID | Cost ($/1M) | Why |
|----------|----------|-------------|-----|
| 🥇 Premium | `anthropic/claude-3.5-sonnet` | $6.00/$30.00 | 49% SWE-Bench Verified |
| 🥈 Balanced | `qwen/qwen-3.5-397b-a17b` | $0.39/$2.34 | SOTA reasoning |
| 🥉 Budget | `deepseek/deepseek-chat` | $0.32/$0.89 | Strong analysis |

### **Reasoning / Analysis**
| Priority | Model ID | Cost ($/1M) | Why |
|----------|----------|-------------|-----|
| 🥇 Premium | `openai/gpt-5` | $1.25/$10.00 | Optimized for complex reasoning |
| 🥈 Balanced | `qwen/qwen-3.5-397b-a17b` | $0.39/$2.34 | 397B MoE SOTA |
| 🥉 Budget | `qwen/qwen-3-coder-next` | $0.12/$0.75 | 80B MoE reasoning |

### **Large Context (100K+ tokens)**
| Priority | Model ID | Context | Cost ($/1M) |
|----------|----------|---------|-------------|
| 🥇 | `google/gemini-2.5-flash` | 1M+ | $0.30/$2.50 |
| 🥈 | `qwen/qwen-3-coder-next` | 262K | $0.12/$0.75 |
| 🥉 | `qwen/qwen-3.5-397b-a17b` | 262K | $0.39/$2.34 |
| 🏅 | `openai/gpt-5-codex` | 400K | $1.25/$10.00 |

### **Agentic Workflows**
| Priority | Model ID | Cost ($/1M) | Why |
|----------|----------|-------------|-----|
| 🥇 | `moonshotai/kimi-k2` | $0.57/$2.30 | 1T params, tool use |
| 🥈 | `qwen/qwen-3-coder-next` | $0.12/$0.75 | Coding agents |
| 🥉 | `qwen/qwen-3.5-397b-a17b` | $0.39/$2.34 | Agent tasks, GUI |

---

## 💰 Cost Comparison (Per 1M Output Tokens)

```
Ultra-Budget Tier (<$1):
  qwen/qwen-3-coder-next        $0.75  ⭐ BEST VALUE
  openai/gpt-4o-mini            $0.60
  qwen/qwen-2.5-coder-32b       $1.00

Budget Tier ($1-2):
  deepseek/deepseek-chat        $0.89  ⭐ BEST VALUE
  openai/gpt-5                  $10.00
  openai/gpt-5-codex            $10.00

Balanced Tier ($2-5):
  google/gemini-2.5-flash       $2.50
  moonshotai/kimi-k2            $2.30
  qwen/qwen-3.5-397b-a17b       $2.34  ⭐ BEST VALUE

Premium Tier (>$5):
  anthropic/claude-3.5-sonnet   $30.00
```

---

## 📋 Updated Model Registry Recommendations

### **Replace Our Current Constants With:**

```python
# CODING SPECIALISTS
QWEN_CODER_NEXT = "qwen/qwen-3-coder-next"      # $0.12/$0.75, 262K - KEEP ✅
QWEN_2_5_CODER_32B = "qwen/qwen-2.5-coder-32b-instruct"  # $0.66/$1.00, 33K - ADD
GPT_5_CODEX = "openai/gpt-5-codex"              # $1.25/$10.00, 400K - ADD

# REASONING / SYNTHESIS
QWEN_3_5_397B_A17B = "qwen/qwen-3.5-397b-a17b"  # $0.39/$2.34, 262K - KEEP ✅
GPT_5 = "openai/gpt-5"                          # $1.25/$10.00, 400K - UPDATE

# BUDGET OPTIONS
DEEPSEEK_CHAT = "deepseek/deepseek-chat"        # $0.32/$0.89, 164K - UPDATE PRICE
GPT_4O_MINI = "openai/gpt-4o-mini"              # $0.15/$0.60, 128K - KEEP ✅

# LARGE CONTEXT
GEMINI_2_5_FLASH = "google/gemini-2.5-flash"    # $0.30/$2.50, 1M+ - UPDATE

# AGENTIC
KIMI_K2 = "moonshotai/kimi-k2"                  # $0.57/$2.30, 128K - ADD

# PREMIUM
CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"  # $6.00/$30.00, 200K - UPDATE PRICE
```

### **Remove (Not Available):**
```python
# These models don't exist on OpenRouter
QWEN_CODER = "qwen/qwen-3-coder"                    # ❌
QWEN_LARGE_REASONING = "qwen/qwen-3.5-235b-a22b-thinking-2507"  # ❌
GROK_4_20 = "x-ai/grok-4.20-beta"                   # ❌
MIMO_V2_FLASH = "xiaomi/mimo-v2-flash"              # ❌
MIMO_V2_PRO = "xiaomi/mimo-v2-pro"                  # ❌
STEPFUN_STEP_3_5_FLASH = "stepfun/step-3.5-flash"   # ❌
GLM_4_7 = "z-ai/glm-4.7"                            # ❌
GLM_5 = "z-ai/glm-5"                                # ❌
AION_2_0 = "aionlabs/aion-2.0"                      # ❌
NEMOTRON_3_SUPER = "nvidia/nemotron-3-super"        # ❌
MINIMAX_M2_7 = "minimax/minimax-m2.7"               # ❌
```

---

## 🎯 Optimal Model Configuration for Our Orchestrator

### **TDD Configuration (Balanced)**
```python
test_generation = "anthropic/claude-3.5-sonnet"  # Best test design
implementation = "qwen/qwen-3-coder-next"        # Best value coding
test_review = "anthropic/claude-3.5-sonnet"      # Best analysis
refactoring = "qwen/qwen-3-coder-next"           # Best value
```

### **TDD Configuration (Budget)**
```python
test_generation = "deepseek/deepseek-chat"       # Strong test writing
implementation = "qwen/qwen-3-coder-next"        # Incredible value
test_review = "deepseek/deepseek-chat"           # Good analysis
refactoring = "qwen/qwen-3-coder-next"           # Incredible value
```

### **Phase-Aware Models**
```python
ANALYSIS: "qwen/qwen-3.5-397b-a17b"    # Strong reasoning
GENERATION: "qwen/qwen-3-coder-next"   # Coding specialist
CRITIQUE: "anthropic/claude-3.5-sonnet" # Low hallucination
SYNTHESIS: "qwen/qwen-3.5-397b-a17b"   # Integration
RESEARCH: "google/gemini-2.5-flash"    # 1M+ context
EVALUATION: "anthropic/claude-3.5-sonnet" # Accurate scoring
```

---

## ✅ Action Items

1. **Revert ModelRegistry changes** - The models we changed TO are invalid!
2. **Original models were CORRECT:**
   - `qwen/qwen-3-coder-next` ✅ (NOT `qwen/qwen-3-coder`)
   - `qwen/qwen-3.5-397b-a17b` ✅ (NOT `qwen/qwen-3.5-235b-a22b-thinking-2507`)

3. **Remove unavailable models** from phase_aware_models.py:
   - xiaomi/mimo-* (not available)
   - stepfun/step-* (not available)
   - z-ai/glm-* (not available)
   - aionlabs/* (not available)
   - nvidia/* (not available)
   - minimax/* (not available)

4. **Update prices** for valid models:
   - `deepseek/deepseek-chat`: $0.32/$0.89 (was $0.27/$1.10)
   - `anthropic/claude-3.5-sonnet`: $6.00/$30.00 (was $3.00/$15.00)

5. **Add new valid models:**
   - `openai/gpt-5-codex` - Coding specialist
   - `moonshotai/kimi-k2` - Agentic workflows
   - `google/gemini-2.5-flash` - Large context
   - `qwen/qwen-2.5-coder-32b-instruct` - Budget coding

---

**Conclusion:** Our original model IDs were mostly CORRECT. The issue may have been a temporary OpenRouter API issue or the models were briefly unavailable. We should revert to the original model IDs and only remove the models that are confirmed unavailable (Xiaomi, StepFun, Z-AI, etc.).
