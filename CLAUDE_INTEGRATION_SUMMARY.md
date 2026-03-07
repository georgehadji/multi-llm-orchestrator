# Claude Integration Summary

## Overview
Removed Kimi K2.5 (Moonshot) LLM and added Anthropic Claude models support.

## Changes Made

### 1. models.py
**Removed:**
- `KIMI_K2_5 = "kimi-k2.5"` enum value

**Added:**
- `CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"` - Best coding performance
- `CLAUDE_3_OPUS = "claude-3-opus-20240229"` - Most capable model
- `CLAUDE_3_HAIKU = "claude-3-haiku-20240307"` - Fastest, cheapest

**Updated:**
- `get_provider()`: Added "anthropic" provider detection
- `COST_TABLE`: Claude pricing ($3/$15 for Sonnet, $15/$75 for Opus, $0.25/$1.25 for Haiku)
- `ROUTING_TABLE`: Added Claude models to all task types
- `FALLBACK_CHAIN`: Claude → OpenAI fallbacks

### 2. api_clients.py
**Removed:**
- Kimi K2.5 client initialization
- `_call_kimi()` method

**Added:**
- Anthropic client initialization (requires `ANTHROPIC_API_KEY` env var)
- `_call_anthropic()` method using native Anthropic API
- Support for Claude's `system` parameter and token usage

### 3. engine.py
**Updated:**
- Comments referencing Kimi → now reference Claude
- `_is_reasoning_model` detection to include Anthropic provider
- `_TIER_BALANCED` to use `CLAUDE_3_HAIKU`
- Model tier mappings

### 4. diagnostics.py
**Updated:**
- Replaced `KIMI_API_KEY` with `ANTHROPIC_API_KEY` in required env vars

### 5. project_file.py
**Updated:**
- Example blocked model from `kimi-k2.5` to `claude-3-opus-20240229`

## Environment Variables

**New required variable:**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Removed:**
```bash
KIMI_API_KEY
MOONSHOT_API_KEY
```

## Installation Requirements

Install Anthropic SDK:
```bash
pip install anthropic
```

## Model Pricing (per 1M tokens)

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| Claude 3.5 Sonnet | $3.00 | $15.00 | Best coding performance |
| Claude 3 Opus | $15.00 | $75.00 | Most capable (complex reasoning) |
| Claude 3 Haiku | $0.25 | $1.25 | Fastest, cheapest |

## Usage Example

```python
from orchestrator import Orchestrator
from orchestrator.models import Model

# Claude 3.5 Sonnet for code generation
orch = Orchestrator()
result = await orch.run_project(
    project_description="Build a REST API",
    # Will use Claude 3.5 Sonnet from routing table
)
```

## Files Modified
- orchestrator/models.py
- orchestrator/api_clients.py
- orchestrator/engine.py
- orchestrator/diagnostics.py
- orchestrator/project_file.py

## Notes
- Claude models are now in the routing tables for all task types
- Claude 3.5 Sonnet is prioritized for CODE_GEN tasks (best coding performance)
- Claude 3 Opus is available for REASONING tasks (most capable)
- Claude 3 Haiku is in the balanced tier (cost-effective)
- Fallback chain: Claude → OpenAI models
