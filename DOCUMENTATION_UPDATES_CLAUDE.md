# Documentation Updates for Claude Integration

## Summary
Updated all documentation to replace Kimi K2.5 (Moonshot) references with Anthropic Claude models.

## Files Updated

### 1. README.md
- **Line 7:** Changed "7 LLM providers" → "6 LLM providers"
- **Lines 36-39:** Replaced Kimi/Moonshot API key setup with Anthropic API key setup

### 2. CLAUDE.md
- **Line 11:** Removed "Kimi" from list of LLM providers (now: Anthropic, OpenAI, Google, DeepSeek)

### 3. CAPABILITIES.md
- **Line 109:** Replaced "Kimi K2.5" → "Claude 3 Haiku" in Standard tier table
- **Line 116:** Updated CODE_GEN routing table: `[DEEPSEEK_CHAT, CLAUDE_3_5_SONNET, GPT_4O, GPT_4O_MINI, GEMINI_FLASH]`

### 4. USAGE_GUIDE.md
- **Lines 1236-1239:** Added Claude 3 Haiku and Claude 3.5 Sonnet to pricing table, removed Kimi K2.5
- **Line 1249:** Replaced `KIMI_API_KEY` with `ANTHROPIC_API_KEY`
- **Lines 1264-1265:** Replaced Kimi K2.5 capabilities row with Claude 3.5 Sonnet and Claude 3 Haiku
- **Lines 1314-1320:** Updated cheapest models ranking and recommendations

### 5. .env.example
- **Lines 33-34:** Replaced Kimi (Moonshot) configuration with Anthropic Claude configuration

## Key Changes Summary

### API Keys
**Removed:**
- `KIMI_API_KEY`
- `MOONSHOT_API_KEY`

**Added:**
- `ANTHROPIC_API_KEY` (format: `sk-ant-...`)

### Models
**Removed:**
- Kimi K2.5 ($0.56/$2.92 per 1M tokens)

**Added:**
- Claude 3.5 Sonnet ($3/$15 per 1M tokens) - Best coding performance
- Claude 3 Opus ($15/$75 per 1M tokens) - Most capable
- Claude 3 Haiku ($0.25/$1.25 per 1M tokens) - Fast & cheap

### Provider Count
Changed from "7 LLM providers" to "6 LLM providers"

## Installation
Users now need to install the Anthropic SDK:
```bash
pip install anthropic
```

And set the API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Verification
All documentation now consistently references Claude models instead of Kimi K2.5.
