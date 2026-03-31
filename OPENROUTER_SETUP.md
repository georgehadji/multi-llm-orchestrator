# OpenRouter-Only Setup Guide

## Overview

The AI Orchestrator now uses **OpenRouter exclusively** for all LLM access. This simplifies configuration and provides access to 200+ models through a single API.

## Benefits

- **Single API Key**: Only need `OPENROUTER_API_KEY`
- **200+ Models**: Access to Meta LLaMA, Microsoft Phi, Google Gemma, Claude, GPT-4, and more
- **Unified Pricing**: All billing through OpenRouter
- **Simplified Code**: No need to maintain multiple provider SDKs
- **Cost Optimization**: Access to cheapest open-source models

## Quick Start

### 1. Get OpenRouter API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Create an account
3. Generate a new API key

### 2. Configure Environment

Update your `.env` file:

```bash
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY_HERE
```

That's it! No other API keys needed.

### 3. Run a Project

```bash
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT authentication" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 2.0
```

## Available Models

All models are accessible via OpenRouter. Recommended models by task:

### Code Generation
1. **phi-4** - $0.07/$0.14 per 1M tokens (Microsoft 14B)
2. **gemma-3-27b-it** - $0.08/$0.20 (Google open-weights)
3. **llama-3.3-70b-instruct** - $0.12/$0.30 (Meta 70B)
4. **llama-4-scout** - $0.11/$0.34 (Meta 109B MoE)
5. **llama-4-maverick** - $0.17/$0.17 (Meta 400B MoE)

### Code Review
1. **phi-4-reasoning-plus** - $0.07/$0.35 (CoT reasoning)
2. **gemma-3-27b-it** - $0.08/$0.20
3. **llama-3.3-70b-instruct** - $0.12/$0.30

### Reasoning
1. **phi-4-reasoning-plus** - $0.07/$0.35
2. **llama-4-maverick** - $0.17/$0.17
3. **llama-3.1-405b-instruct** - $2.00/$2.00

### Writing
1. **llama-4-maverick** - $0.17/$0.17
2. **hermes-3-llama-3.1-70b** - $0.40/$0.40
3. **claude-sonnet-4-6** - $3/$15 (via OpenRouter)

## Cost Comparison

| Model | Input/1M | Output/1M | Provider |
|-------|----------|-----------|----------|
| phi-4 | $0.07 | $0.14 | Microsoft |
| gemma-3-27b-it | $0.08 | $0.20 | Google |
| llama-3.3-70b | $0.12 | $0.30 | Meta |
| llama-4-scout | $0.11 | $0.34 | Meta |
| llama-4-maverick | $0.17 | $0.17 | Meta |
| llama-3.1-405b | $2.00 | $2.00 | Meta |
| gpt-4o | $2.50 | $10.00 | OpenAI |
| claude-sonnet-4-6 | $3.00 | $15.00 | Anthropic |

## Model Selection

The orchestrator automatically selects the best model for each task type:

- **CODE_GEN**: Starts with phi-4, escalates to llama-4-maverick if needed
- **CODE_REVIEW**: Uses phi-4-reasoning for accurate reviews
- **REASONING**: phi-4-reasoning → llama-4-maverick → llama-3.1-405b
- **WRITING**: llama-4-maverick → claude-sonnet-4-6

## Migration from Multi-Provider

If you were using multiple provider keys:

### Before
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...
DEEPSEEK_API_KEY=sk-...
```

### After
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

All models are now accessible through OpenRouter's unified API.

## Technical Changes

### Modified Files

1. **orchestrator/api_clients.py**: Simplified to only use OpenRouter
2. **orchestrator/models.py**: Updated routing table for OpenRouter models
3. **orchestrator/models.py**: `get_provider()` now returns "openrouter" for all models
4. **.env**: Only requires `OPENROUTER_API_KEY`

### API Client

```python
# All providers now use OpenRouter
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
)
```

## Troubleshooting

### "OPENROUTER_API_KEY not set"

Ensure your `.env` file contains:
```bash
OPENROUTER_API_KEY=sk-or-v1-YOUR_KEY
```

### Model not found

Check the [OpenRouter models page](https://openrouter.ai/models) for available models and update `models.py` if needed.

### Rate limits

OpenRouter has built-in rate limiting. The orchestrator handles retries automatically.

## Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Available Models](https://openrouter.ai/models)
- [Pricing](https://openrouter.ai/models/pricing)
- [API Keys](https://openrouter.ai/keys)

---

**Author**: Georgios-Chrysovalantis Chatzivantsidis  
**Updated**: 2026-03-30
