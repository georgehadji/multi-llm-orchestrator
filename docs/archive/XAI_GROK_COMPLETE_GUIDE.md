# xAI Grok Models — Complete Integration Guide

**Version:** 2.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Comprehensive analysis of xAI Grok models** with rate limits, regions, capabilities, and integration recommendations for AI Orchestrator.

---

## Executive Summary

### Current Status ✅

| Aspect | Status | Details |
|--------|--------|---------|
| **Models Supported** | ✅ 4 models | grok-3, grok-3-mini, grok-4, grok-4.1-fast |
| **Latest Model** | ❌ Missing | **grok-4.20** NOT yet integrated |
| **API Integration** | ✅ Complete | OpenAI-compatible via `api.x.ai/v1` |
| **Rate Limits** | ⚠️ Tier-based | Based on cumulative spend |
| **Regions** | ❌ Not configured | Default global endpoint only |
| **X Search** | ❌ Not integrated | Unique xAI feature |
| **Provisioned Throughput** | ❌ Not configured | Enterprise option available |

### Key Findings 🎯

1. **grok-4.20** — Latest model with **industry-leading speed**, NOT yet in orchestrator
2. **2M Context** — grok-4.1-fast and grok-4.20 both support 2M tokens
3. **Rate Limits** — Tier system based on spend ($0 → $5,000+)
4. **Regions** — `us-east-1`, `eu-west-1` available (regional endpoints)
5. **Reasoning Models** — grok-4 requires **encrypted content** for reasoning traces
6. **Provisioned Throughput** — $10/day per unit, 30-day minimum

---

## 1. Rate Limits

### Tier System

xAI uses a **spend-based tier system** for rate limits:

| Tier | Cumulative Spend | RPM | TPM | TPD | How to Unlock |
|------|-----------------|-----|-----|-----|---------------|
| **Tier 1** | $0 (default) | * | * | * | Automatic |
| **Tier 2** | $50+ | * | * | * | Automatic |
| **Tier 3** | $200+ | * | * | * | Automatic |
| **Tier 4** | $500+ | * | * | * | Automatic |
| **Tier 5** | $1,000+ | * | * | * | Automatic |
| **Tier 6** | $5,000+ | * | * | * | Automatic |
| **Enterprise** | Custom | Custom | Custom | Custom | Contact sales |

**Notes:**
- Tiers are **permanent** once qualified
- Based on **cumulative spend since January 1, 2026**
- Specific RPM/TPM values vary by model (view in xAI Console)
- **No burst limits** documented

### How to Increase Limits

1. **Increase Spend** — Automatic tier progression
2. **Cloud Console** — Request limits without spend
3. **Email Sales** — For Voice/Imagine APIs: `sales@x.ai`

### Integration Impact

```python
# orchestrator/rate_limiter.py
class GrokRateLimiter:
    """xAI Grok rate limiter with tier tracking."""
    
    TIERS = {
        0: {"rpm": 10, "tpm": 10_000},      # Tier 1 (estimate)
        50: {"rpm": 60, "tpm": 100_000},    # Tier 2
        200: {"rpm": 120, "tpm": 500_000},  # Tier 3
        500: {"rpm": 300, "tpm": 1_000_000},# Tier 4
        1000: {"rpm": 600, "tpm": 2_000_000},# Tier 5
        5000: {"rpm": 1200, "tpm": 5_000_000},# Tier 6
    }
    
    async def check_limit(self, model: str, tokens: int):
        # Check current tier based on spend
        # Apply appropriate limits
        pass
```

---

## 2. Provisioned Throughput

### Overview

**Provisioned Throughput** is an **enterprise feature** for guaranteed capacity:

| Feature | Details |
|---------|---------|
| **Cost** | $10.00 per day per unit |
| **Commitment** | Minimum 30 days |
| **Capacity per Unit** | ~31,500 Input TPM + 12,500 Output TPM (model-dependent) |
| **Overage** | Falls back to standard pay-as-you-go pricing |
| **SLA** | 99.9% uptime guarantee |

### How It Works

1. **Purchase Units** — Contact `support@x.ai` with expected TPM
2. **Custom Quote** — Receive pricing based on models and capacity
3. **Activation** — Sign order form, capacity added to account
4. **Automatic Application** — All API requests use provisioned capacity first
5. **Fallback** — Excess usage billed at standard rates

### Benefits

- ✅ **Predictable Latency** — Consistent response times during peak
- ✅ **Uncapped Scale** — Add capacity as needed
- ✅ **High Reliability** — 99.9% SLA
- ✅ **Cost Predictability** — Fixed daily cost per unit

### Calculation Example

```
Required: 100,000 Input TPM for grok-4.1-fast
Per Unit: 31,500 Input TPM
Units Needed: 100,000 ÷ 31,500 = 3.17 → 4 units

Cost: 4 units × $10/day × 30 days = $1,200/month
```

### Integration Recommendation

```python
# orchestrator/config.py
GROK_PROVISIONED_THROUGHPUT = {
    "enabled": False,  # Enable for enterprise deployments
    "units": 0,
    "models": ["grok-4.1-fast", "grok-4.20"],
    "max_daily_cost": 100,  # $10 × units
}
```

---

## 3. Regions

### Available Regions

| Region | Code | Endpoint | Location |
|--------|------|----------|----------|
| **US East** | `us-east-1` | `https://us-east-1.api.x.ai/v1` | US East Coast |
| **Europe** | `eu-west-1` | `https://eu-west-1.api.x.ai/v1` | Europe |
| **Global** | (default) | `https://api.x.ai/v1` | Auto-routed |

### Endpoints

```python
# Default (auto-routed to lowest latency)
base_url = "https://api.x.ai/v1"

# Regional (forced routing)
base_url = "https://us-east-1.api.x.ai/v1"
base_url = "https://eu-west-1.api.x.ai/v1"
```

### Model Availability

- **Global Endpoint:** Access to ALL models available to your team
- **Regional Endpoints:** Model availability varies by region
- **Check Console:** Verify model availability per region in xAI Console

### Data Residency

- **Processing:** Regional endpoints ensure in-region processing
- **Data at Rest:** Contact `sales@x.ai` for in-region storage (additional cost)

### Latency & Fallback

| Endpoint | Latency | Fallback Behavior |
|----------|---------|-------------------|
| **Global** | Auto-optimized | Routes to nearest available region |
| **Regional** | Fixed to region | **Fails** if region unavailable |

### Integration

```python
# orchestrator/api_clients.py
class UnifiedClient:
    def __init__(self, region: Optional[str] = None):
        if region:
            self.base_url = f"https://{region}.api.x.ai/v1"
        else:
            self.base_url = "https://api.x.ai/v1"  # Auto-route
```

---

## 4. Model Capabilities

### Text Generation

| Feature | Support | Details |
|---------|---------|---------|
| **API** | ✅ Responses API (preferred) | Stateful, server-side storage |
| **Legacy API** | ✅ Chat Completions | Stateless, deprecated |
| **Streaming** | ✅ Supported | Via SSE |
| **Function Calling** | ✅ Supported | Native tool integration |
| **Structured Outputs** | ✅ Supported | JSON mode, schema validation |
| **Timeout** | ⚠️ 3600s recommended | For reasoning models |

### Parameters

```python
# Supported parameters
{
    "model": "grok-4.20-reasoning",
    "input": [...],  # Messages array
    "store": True,   # Server-side storage (30 days)
    "previous_response_id": "resp_123",  # Continue conversation
    "include": ["reasoning.encrypted_content"],  # For reasoning models
    "temperature": 0.7,  # Sampling
    "max_tokens": 4096,  # Max output
    "top_p": 0.9,  # Nucleus sampling
}
```

### Reasoning Models

| Model | Reasoning Effort | Encrypted Content | Notes |
|-------|-----------------|-------------------|-------|
| **grok-3-mini** | ✅ `low`/`high` | ❌ Plain text | Only model with effort control |
| **grok-3** | ❌ Fixed | ❌ Not returned | Standard reasoning |
| **grok-4** | ❌ Fixed | ✅ Encrypted | Use `include` parameter |
| **grok-4.20** | ❌ Fixed | ✅ Encrypted | Latest reasoning model |

### Reasoning Parameters

```python
# grok-3-mini ONLY
{
    "model": "grok-3-mini",
    "reasoning_effort": "high",  # or "low"
}

# grok-4 / grok-4.20
{
    "model": "grok-4.20-reasoning",
    "include": ["reasoning.encrypted_content"],  # Get reasoning trace
}
```

### Unsupported Parameters (Reasoning Models)

❌ `presence_penalty`
❌ `frequency_penalty`
❌ `stop`

---

## 5. Updated Models Comparison

### All Grok Models

| Model | Context | Input ($/1M) | Output ($/1M) | Reasoning | Speed | Status |
|-------|---------|-------------|---------------|-----------|-------|--------|
| **grok-4.20** ⭐ | 2M | ~$0.25 | ~$0.75 | ✅ Encrypted | **Fastest** | ❌ NOT INTEGRATED |
| grok-4.1-fast | 2M | $0.20 | $0.50 | ✅ Encrypted | Very Fast | ✅ Integrated |
| grok-4 | ~128K | $3.00 | $15.00 | ✅ Encrypted | Fast | ✅ Integrated |
| grok-3 | ~128K | $2.00 | $10.00 | ✅ Standard | Medium | ✅ Integrated |
| grok-3-mini | ~128K | $0.10 | $0.30 | ✅ Effort control | Fast | ✅ Integrated |

### Model Recommendations

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **Long Documents (2M)** | grok-4.20 / grok-4.1-fast | Best value, fastest |
| **Code Generation** | grok-4.20 | Industry-leading speed |
| **Reasoning Tasks** | grok-4.20-reasoning | Latest reasoning capabilities |
| **Budget Tasks** | grok-3-mini | Cheapest option |
| **Real-time Search** | grok-4.20 + X Search | X/Twitter integration |
| **Complex Analysis** | grok-4.20 | Lowest hallucination rate |

---

## 6. Integration Updates Required

### PRIORITY 1: Add grok-4.20 Model

**File: `orchestrator/models.py`**

```python
class Model(str, Enum):
    # XAI GROK MODELS (Updated)
    GROK_3 = "grok-3"
    GROK_3_MINI = "grok-3-mini"
    GROK_4 = "grok-4"
    GROK_4_1_FAST = "grok-4.1-fast"
    GROK_4_20 = "grok-4.20"  # ← NEW
    GROK_4_20_REASONING = "grok-4.20-reasoning"  # ← NEW
    GROK_4_LATEST = "grok-4-latest"  # ← NEW (alias)
```

**File: `orchestrator/models.py` — Pricing**

```python
# XAI GROK (Updated)
Model.GROK_3:             {"input": 2.00,  "output": 10.00},
Model.GROK_3_MINI:        {"input": 0.10,  "output": 0.30},
Model.GROK_4:             {"input": 3.00,  "output": 15.00},
Model.GROK_4_1_FAST:      {"input": 0.20,  "output": 0.50},
Model.GROK_4_20:          {"input": 0.25,  "output": 0.75},  # ESTIMATED
Model.GROK_4_20_REASONING:{"input": 0.25,  "output": 0.75},  # ESTIMATED
```

**File: `orchestrator/models.py` — Routing Table**

```python
# CODE_GEN: Promote grok-4.20 to Tier 1
TaskType.CODE_GEN: [
    Model.MISTRAL_NEMO,
    Model.MISTRAL_SMALL_3_1,
    Model.GEMINI_2_5_FLASH_LITE,
    Model.GROK_4_20,        # ← NEW: Best speed + 2M context
    Model.GROK_4_1_FAST,
    Model.ERNIE_SPEED_PRO,
    Model.GPT_5_NANO,
    # ...
]

# REASONING: Promote grok-4.20
TaskType.REASONING: [
    Model.O3_MINI,
    Model.GROK_4_20_REASONING,  # ← NEW: Latest reasoning
    Model.GROK_4_1_FAST,
    Model.QWEN_3_235B,
    # ...
]

# DATA_EXTRACT: Add grok for 2M context
TaskType.DATA_EXTRACT: [
    Model.MISTRAL_NEMO,
    Model.GEMINI_2_5_FLASH_LITE,
    Model.GROK_4_20,        # ← NEW: 2M context, fast
    Model.GROK_4_1_FAST,
    # ...
]

# SUMMARIZE: Add grok for long documents
TaskType.SUMMARIZE: [
    Model.MISTRAL_NEMO,
    Model.GEMINI_2_5_FLASH_LITE,
    Model.GROK_4_1_FAST,    # 2M context
    Model.GROK_4_20,        # ← NEW: Faster
    # ...
]
```

---

### PRIORITY 2: Region Support

**File: `orchestrator/api_clients.py`**

```python
class UnifiedClient:
    def __init__(
        self,
        region: Optional[str] = None,
        **kwargs
    ):
        # xAI Grok (api.x.ai) — OpenAI-compatible API
        xai_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if xai_key:
            # Support regional endpoints
            if region:
                base_url = f"https://{region}.api.x.ai/v1"
            else:
                base_url = "https://api.x.ai/v1"  # Auto-route
            
            self._clients["xai"] = AsyncOpenAI(
                api_key=xai_key,
                base_url=base_url,
            )
            logger.info(f"xAI client initialized ({region or 'global'})")
```

**File: `orchestrator/config.py`**

```python
# xAI Grok Configuration
XAI_CONFIG = {
    "region": None,  # None = auto-route, or "us-east-1", "eu-west-1"
    "provisioned_throughput": False,
    "rate_limit_tier": "auto",  # Based on spend
}
```

---

### PRIORITY 3: Reasoning Model Support

**File: `orchestrator/api_clients.py`**

```python
async def _call_reasoning_model(
    self,
    model: Model,
    prompt: str,
    **kwargs
) -> APIResponse:
    """Call reasoning model with proper parameters."""
    
    # grok-3-mini supports reasoning_effort
    if model == Model.GROK_3_MINI:
        effort = kwargs.get("reasoning_effort", "high")
        response = await self._client.chat.completions.create(
            model=model.value,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=effort,  # grok-3-mini only
            timeout=3600,  # Extended timeout for reasoning
        )
        return APIResponse(
            content=response.choices[0].message.content,
            reasoning_content=response.choices[0].message.reasoning_content,
            usage=response.usage,
        )
    
    # grok-4 / grok-4.20 require encrypted content
    elif model in [Model.GROK_4, Model.GROK_4_20_REASONING]:
        response = await self._client.responses.create(
            model=model.value,
            input=[{"role": "user", "content": prompt}],
            include=["reasoning.encrypted_content"],  # Get reasoning
            timeout=3600,
        )
        return APIResponse(
            content=response.output_text,
            reasoning_content=response.reasoning.encrypted_content,
            usage=response.usage,
        )
    
    # Standard reasoning models
    else:
        return await self._call_standard_model(model, prompt, **kwargs)
```

---

### PRIORITY 4: Rate Limiter Integration

**File: `orchestrator/rate_limiter.py`**

```python
class GrokRateLimiter:
    """xAI Grok rate limiter with tier-based limits."""
    
    # Estimated limits per tier (verify in xAI Console)
    TIERS = {
        0: {"rpm": 10, "tpm": 10_000},
        50: {"rpm": 60, "tpm": 100_000},
        200: {"rpm": 120, "tpm": 500_000},
        500: {"rpm": 300, "tpm": 1_000_000},
        1000: {"rpm": 600, "tpm": 2_000_000},
        5000: {"rpm": 1200, "tpm": 5_000_000},
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_tier = 0
        self.cumulative_spend = 0.0
    
    async def update_tier(self):
        """Update tier based on cumulative spend."""
        # Fetch spend from xAI API
        # Update tier accordingly
        pass
    
    async def acquire(self, tokens: int):
        """Acquire tokens for request."""
        tier_limits = self.TIERS[self.current_tier]
        
        # Check TPM limit
        if self.current_tpm + tokens > tier_limits["tpm"]:
            await self._wait_for_capacity()
        
        self.current_tpm += tokens
```

---

## 7. Environment Variables

### Required

```bash
# xAI API Key
export XAI_API_KEY="xai-..."
# or
export GROK_API_KEY="xai-..."
```

### Optional

```bash
# Region (optional, default = auto-route)
export XAI_REGION="us-east-1"
# or
export XAI_REGION="eu-west-1"

# Provisioned Throughput
export XAI_PROVISIONED_THROUGHPUT="true"
export XAI_PROVISIONED_UNITS="4"

# Rate Limit Tier Override
export XAI_RATE_LIMIT_TIER="tier_3"
```

---

## 8. Testing Strategy

### Unit Tests

```python
# tests/test_grok_integration.py
class TestGrokModels:
    async def test_grok_4_20_routing(self):
        """Test grok-4.20 is routed correctly."""
        orch = Orchestrator()
        task = Task(
            type=TaskType.CODE_GEN,
            prompt="Build a fast API",
            context_size=500_000,  # Long context
        )
        
        model = orch.select_model(task)
        assert model == Model.GROK_4_20
    
    async def test_grok_reasoning_encrypted(self):
        """Test reasoning model with encrypted content."""
        client = UnifiedClient()
        response = await client.generate(
            model=Model.GROK_4_20_REASONING,
            prompt="Solve this complex math problem...",
            include_reasoning=True,
        )
        
        assert response.reasoning_content is not None
    
    async def test_grok_regional_endpoint(self):
        """Test regional endpoint routing."""
        client = UnifiedClient(region="us-east-1")
        assert "us-east-1.api.x.ai" in client.base_url
```

### Integration Tests

```python
# tests/test_grok_e2e.py
class TestGrokEndToEnd:
    async def test_full_project_grok_4_20(self):
        """Test full project using grok-4.20."""
        orch = Orchestrator(
            preferred_model=Model.GROK_4_20,
            region="us-east-1",
        )
        
        result = await orch.run_project(
            project_description="Build a real-time analytics dashboard",
            success_criteria="All tests pass, <2s response time",
        )
        
        assert result.status == "completed"
        assert result.metadata["primary_model"] == "grok-4.20"
```

---

## 9. Cost Analysis

### Updated Pricing Comparison

**Scenario: Processing 1M tokens (input + output)**

| Model | Input | Output | Total | vs grok-4.20 |
|-------|-------|--------|-------|--------------|
| **grok-4.20** | $0.25 | $0.75 | **$1.00** | **-** |
| grok-4.1-fast | $0.20 | $0.50 | $0.70 | -30% |
| grok-3-mini | $0.10 | $0.30 | $0.40 | -60% |
| gpt-4o-mini | $0.15 | $0.60 | $0.75 | -25% |
| gpt-4o | $2.50 | $10.00 | $12.50 | +1150% |
| claude-sonnet-4-6 | $3.00 | $15.00 | $18.00 | +1700% |

### Provisioned Throughput ROI

**Scenario: 100K TPM sustained usage**

| Option | Monthly Cost | Notes |
|--------|-------------|-------|
| **Pay-as-you-go** | ~$3,000 | Variable, peak pricing |
| **Provisioned (4 units)** | $1,200 | Fixed, predictable |
| **Savings** | **$1,800/month** | **60% savings** |

---

## 10. Migration Guide

### From grok-4.1-fast to grok-4.20

```python
# Before
model = Model.GROK_4_1_FAST

# After
model = Model.GROK_4_20  # Drop-in replacement
```

### Enabling Regional Endpoints

```python
# Before
client = UnifiedClient()  # Global endpoint

# After
client = UnifiedClient(region="us-east-1")  # Regional
```

### Using Reasoning Models

```python
# grok-3-mini
response = await client.generate(
    model=Model.GROK_3_MINI,
    prompt="Solve this puzzle",
    reasoning_effort="high",  # Only for grok-3-mini
)

# grok-4.20
response = await client.generate(
    model=Model.GROK_4_20_REASONING,
    prompt="Solve this puzzle",
    include_reasoning=True,  # Returns encrypted reasoning
)
```

---

## 11. Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `429 Too Many Requests` | Rate limit exceeded | Wait or upgrade tier |
| `400 reasoning_effort not supported` | Using on grok-4 | Remove parameter (grok-4 only) |
| `503 Service Unavailable` | Region unavailable | Switch to global endpoint |
| `402 Payment Required` | Insufficient credits | Add credits to account |

### Debug Mode

```python
# Enable xAI debug logging
export XAI_DEBUG="true"
export ORCHESTRATOR_LOG_LEVEL="DEBUG"
```

---

## 12. Implementation Checklist

### Phase 1: Model Updates (Week 1)

- [ ] Add `GROK_4_20` to `models.py`
- [ ] Add `GROK_4_20_REASONING` to `models.py`
- [ ] Update pricing table
- [ ] Update routing tables (CODE_GEN, REASONING, DATA_EXTRACT, SUMMARIZE)
- [ ] Update fallback chains
- [ ] Run tests

### Phase 2: Region Support (Week 2)

- [ ] Add region parameter to `UnifiedClient`
- [ ] Add `XAI_REGION` environment variable
- [ ] Test regional endpoints
- [ ] Update documentation

### Phase 3: Reasoning Support (Week 3)

- [ ] Implement encrypted reasoning content
- [ ] Add `reasoning_effort` for grok-3-mini
- [ ] Update ARA Pipeline for reasoning models
- [ ] Create tests

### Phase 4: Rate Limiter (Week 4)

- [ ] Implement tier-based rate limiting
- [ ] Add spend tracking
- [ ] Add provisioned throughput support
- [ ] Create monitoring dashboard

---

## Conclusion

### Summary

**Current Integration:** ⭐⭐⭐⭐ (4/5) — Good but incomplete

**Key Gaps:**
1. ❌ grok-4.20 not integrated
2. ❌ Regional endpoints not configured
3. ❌ Reasoning model features incomplete
4. ❌ Rate limiter not tier-aware

**Recommendations:**
1. **Immediate:** Add grok-4.20 to routing tables
2. **Short-term:** Implement region support
3. **Medium-term:** Add reasoning model features
4. **Long-term:** Enterprise features (provisioned throughput)

### Competitive Position

| Feature | xAI Grok | OpenAI | Anthropic | Google |
|---------|----------|--------|-----------|--------|
| **Max Context** | **2M** ⭐ | 128K | 200K | 2M |
| **Speed** | **Fastest** ⭐ | Fast | Medium | Fast |
| **Price (2M)** | **$0.20/$0.50** ⭐ | N/A | N/A | $0.075/$0.30 |
| **Real-time Search** | **X + Web** ⭐ | Web | None | Web |
| **Regions** | 2+ | Global | Global | Global |

**Verdict:** xAI Grok offers **best value for long-context tasks** with unique X Search integration.

---

**References:**
- [xAI Models](https://docs.x.ai/developers/models)
- [xAI Rate Limits](https://docs.x.ai/developers/rate-limits)
- [xAI Provisioned Throughput](https://docs.x.ai/developers/provisioned-throughput)
- [xAI Regions](https://docs.x.ai/developers/regions)
- [xAI Quickstart](https://docs.x.ai/developers/quickstart)
- [xAI Text Generation](https://docs.x.ai/developers/model-capabilities/text/generate-text)
- [xAI Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)
- [xAI Model Comparison](https://docs.x.ai/developers/model-capabilities/text/comparison)

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
