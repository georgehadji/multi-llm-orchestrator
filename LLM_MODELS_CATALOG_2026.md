# LLM Models Catalog 2026
## Complete Reference for AI Orchestrator

**Generated:** March 2026  
**Purpose:** Comprehensive list of all available LLM APIs for integration

---

## Table of Contents
1. [Western Providers](#western-providers)
   - [OpenAI](#1-openai)
   - [Anthropic (Claude)](#2-anthropic-claude)
   - [Google (Gemini)](#3-google-gemini)
   - [DeepSeek](#4-deepseek)
   - [MiniMax](#5-minimax)
   - [Mistral AI](#6-mistral-ai)
   - [xAI (Grok)](#7-xai-grok)
   - [Cohere](#8-cohere)
2. [Chinese Providers](#chinese-providers)
   - [Alibaba (Qwen)](#1-alibaba-qwen)
   - [ByteDance (Seed/Doubao)](#2-bytedance-seeddoubao)
   - [Zhipu AI (GLM)](#3-zhipu-ai-glm)
   - [Baidu (Ernie)](#4-baidu-ernie)
   - [Moonshot AI (Kimi)](#5-moonshot-ai-kimi)
   - [Tencent (Hunyuan)](#6-tencent-hunyuan)
   - [Baichuan Intelligence](#7-baichuan-intelligence)
3. [Aggregators / Proxies](#aggregators--proxies)
4. [Cost Comparison Matrix](#cost-comparison-matrix)
5. [Recommended Routing](#recommended-routing)

---

## Western Providers

### 1. OpenAI
**API:** `https://api.openai.com/v1`  
**Documentation:** https://platform.openai.com/docs

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| GPT-5.2 Pro | $21.00 | $168.00 | 200K | Hardest reasoning tasks |
| GPT-5.2 | $1.75 | $14.00 | 200K | Coding, agents |
| GPT-5 | $1.25 | $10.00 | 128K | General flagship |
| GPT-5 Mini | $0.25 | $2.00 | 200K | Fast, affordable |
| GPT-5 Nano | $0.05 | $0.40 | 128K | High-volume simple tasks |
| o4-mini | $1.10 | $4.40 | 200K | Best value reasoning |
| o3 | $2.00 | $8.00 | 200K | Mid-tier reasoning |
| o3-pro | $20.00 | $80.00 | 200K | Strong reasoning |
| o1 | $15.00 | $60.00 | 200K | Legacy reasoning |
| GPT-4.1 | $2.00 | $8.00 | 1M | Previous gen |
| GPT-4.1 Mini | $0.40 | $1.60 | 1M | Previous gen budget |
| GPT-4.1 Nano | $0.10 | $0.40 | 1M | Previous gen fast |
| GPT-4o | $2.50 | $10.00 | 128K | Current flagship |
| GPT-4o Mini | $0.15 | $0.60 | 128K | Best value current |

**Features:**
- Batch API: 50% off for async workloads
- Cached input: 90% discount
- Widest ecosystem (GPT Store, plugins)

---

### 2. Anthropic (Claude)
**API:** `https://api.anthropic.com/v1`  
**Documentation:** https://docs.anthropic.com

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Claude Opus 4.6 | $5.00 | $25.00 | 200K | Most capable |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 200K | Best coding performance |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | Fast, cheapest Claude |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 200K | Previous gen (excellent) |
| Claude 3 Opus | $15.00 | $75.00 | 200K | Legacy premium |
| Claude 3 Haiku | $0.25 | $1.25 | 200K | Legacy fast |

**Features:**
- Constitutional AI
- 200K context window
- Excellent code understanding
- Cached input: 90% discount
- Computer Use (autonomous tasks)

---

### 3. Google (Gemini)
**API:** `https://generativelanguage.googleapis.com`  
**Documentation:** https://ai.google.dev

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Gemini 2.5 Pro | $1.25 | $10.00 | 2M | Document analysis, reasoning |
| Gemini 3.1 Pro Preview | $2.00 | $12.00 | 2M | Advanced reasoning |
| Gemini 2.5 Flash | $0.30 | $2.50 | 1M | Fast, balanced |
| Gemini 3.1 Flash-Lite | $0.25 | $1.50 | 1M | Best value (86.9% GPQA) |
| Gemini 2.0 Flash | $0.15 | $0.60 | 1M | General tasks |
| Gemini 2.0 Flash-Lite | $0.075 | $0.30 | 1M | Cheapest mainstream |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 1M | New best value |
| Gemini 1.5 Pro | $3.50 | $10.50 | 2M | Previous gen |

**Features:**
- 🎁 Free tier: 1,000 requests/day on most models
- Largest context: 2M tokens
- Multimodal (text, image, video, audio)
- Cached input: 75% discount
- Grounding (Google Search)

---

### 4. DeepSeek
**API:** `https://api.deepseek.com/v1` (OpenAI-compatible)  
**Documentation:** https://platform.deepseek.com

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| DeepSeek-V3.2 | $0.28 | $0.42 | 128K | Best cost/performance |
| DeepSeek-V3 | $0.27 | $1.10 | 128K | General |
| DeepSeek-R1 (Reasoner) | $0.28 | $0.42 | 128K | o1-class reasoning |
| DeepSeek-R1 | $0.55 | $2.19 | 128K | Advanced reasoning |
| DeepSeek-Coder | $0.28 | $0.42 | 128K | Code generation |

**Features:**
- 🇨🇳 Chinese provider
- 90% cache hit discount ($0.028/M input)
- MoE architecture
- Open source culture
- Reasoning model at same price as chat

---

### 5. MiniMax
**API:** `https://api.minimaxi.chat/v1` (OpenAI-compatible)

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| MiniMax-Text-01 | $0.50 | $1.50 | 1M | Alternative reasoning |
| MiniMax M2 | $0.50 | $1.50 | 1M | General |
| MiniMax M2.5 | $0.50 | $1.50 | 1M | "Too cheap to meter" |

**Features:**
- 🇨🇳 Chinese provider
- 1M context window
- Hong Kong IPO filed
- $850M+ raised

---

### 6. Mistral AI
**API:** `https://api.mistral.ai/v1` (OpenAI-compatible)  
**Documentation:** https://docs.mistral.ai

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| **Mistral Nemo** | $0.02 | $0.04 | 128K | 🔥 Cheapest capable |
| **Mistral Small 3.1** | $0.03 | $0.11 | 128K | 🔥 Best value overall |
| Ministral 3B | $0.04 | $0.04 | 128K | Edge/mobile |
| Ministral 8B | $0.10 | $0.10 | 128K | Lightweight |
| Codestral | $0.30 | $0.90 | 128K | Code specialist |
| Mistral Medium 3 | $0.40 | $2.00 | 128K | Balanced |
| Mistral Large 3 | $0.50 | $1.50 | 128K | Complex reasoning |
| Devstral | $0.10 | $0.30 | 128K | Agentic coding |
| Magistral Small | $0.50 | $1.50 | 128K | Reasoning |
| Magistral Medium | $2.00 | $5.00 | 128K | Advanced reasoning |

**Features:**
- 🇪🇺 European provider (France)
- Apache 2.0 open source
- Free tier available
- La Plateforme console

---

### 7. xAI (Grok)
**API:** `https://api.x.ai/v1` (OpenAI-compatible)  
**Console:** https://console.x.ai

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Grok 4 | $3.00 | $15.00 | 256K | Premium reasoning |
| Grok 3 | $2.00 | $10.00 | 128K | General tasks |
| **Grok 4.1 Fast** | $0.20 | $0.50 | 2M | 🔥 2M context, fast |
| **Grok 3 Mini** | $0.10 | $0.30 | 128K | Budget option |

**Features:**
- 🎁 $175 free credits/month ($25 signup + $150 data sharing)
- Largest context: 2M tokens (Grok 4.1 Fast)
- Real-time X (Twitter) data access
- Built-in web search
- Cached input: 50-75% discount

---

### 8. Cohere
**API:** `https://api.cohere.ai/v1`  
**Documentation:** https://docs.cohere.com

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Command A | $2.50 | $10.00 | 256K | Enterprise |
| Command R+ | $2.50 | $10.00 | 128K | RAG, long context |
| Command R | $0.15 | $0.60 | 128K | Balanced |
| Command R7B | $0.15 | $0.0375 | 128K | Ultra-cheap |

**Features:**
- Excellent for RAG pipelines
- Strong embeddings
- Enterprise focus

---

## Chinese Providers

### 1. Alibaba (Qwen)
**API:** `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (OpenAI-compatible)  
**Console:** https://www.alibabacloud.com/model-studio

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| **Qwen3-235B-A22B** | $0.136 | $0.544 | 128K | 🔥 Flagship, best value |
| Qwen3-Coder-30B | $0.15 | $0.60 | 128K | Code generation |
| Qwen3-32B | $0.20 | $0.80 | 128K | Balanced |
| Qwen-Plus | $0.50 | $1.50 | 128K | General |
| Qwen-Turbo | $0.20 | $0.60 | 128K | Fast |
| Qwen-Max | $2.00 | $6.00 | 128K | Premium |
| Qwen-Math | $0.50 | $1.50 | - | Math specialist |
| **Qwen-Long** | $0.10 | $0.40 | 10M | 🔥 Document analysis |
| Qwen-VL | $0.50 | $1.50 | 128K | Vision-language |

**Features:**
- 🇨🇳 Chinese provider
- 🎁 Free tier: 1M tokens/month
- 600M+ downloads worldwide (Hugging Face)
- Apache 2.0 open source
- 180,000+ derivatives
- Multimodal (text, image, code)
- Singapore/US/China regions

---

### 2. ByteDance (Seed/Doubao)
**API:** `https://ark.cn-beijing.volces.com/api/v3` (OpenAI-compatible)  
**Console:** https://www.volcengine.com

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| **Seed 2.0 Pro** | $0.47 | $2.37 | 128K | Frontier reasoning |
| Seed 2.0 Lite | $0.20 | $1.00 | 128K | General production |
| **Seed 2.0 Mini** | $0.05 | $0.25 | 128K | 🔥 High-throughput |
| Seed 2.0 Code | $0.30 | $1.20 | 128K | Code specialist |

**Features:**
- 🇨🇳 Chinese provider
- 155 million weekly active users (Doubao app)
- AIME 2025: 98.3 score
- Codeforces: 3020 rating
- 3.7x cheaper than GPT-5.2
- 10x cheaper than Claude Opus 4.5

---

### 3. Zhipu AI (GLM)
**API:** `https://api.z.ai/api/coding/paas/v4` (OpenAI-compatible)  
**Platform:** https://www.z.ai

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| GLM-5 | ~$7-9 | ~$14-17 | 2M | Flagship (744B params) |
| **GLM-4.7** | $3.00 | $15.00 | 128K | Coding specialist |
| GLM-4.6 | $2.37 | $11.06 | 128K | General |
| GLM-4 | ~$4 | ~$8 | 128K | Previous gen |
| GLM-4-Flash | ~$0.50 | ~$1.00 | 128K | Fast |
| GLM-4-Air | ~$1 | ~$2 | 128K | Balanced |

**Flat Rate Plans:**
- Lite: $3/month - 120 prompts
- Pro: $15/month - 600 prompts

**Features:**
- 🇨🇳 Chinese provider
- 744B parameters (GLM-5)
- Trained on Chinese chips (Huawei Ascend)
- SWE-bench: 73.8%
- HumanEval: 85.2%
- Open source models available

---

### 4. Baidu (Ernie)
**API:** Available through Novita AI, Krater AI, Puter.js  
**Official:** https://cloud.baidu.com (China only)

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| **Ernie 4.5-300B-A47B** | $0.224 | $0.88 | 128K | Flagship |
| Ernie 4.5-VL-424B | $0.336 | $1.00 | 128K | Vision-language |
| **Ernie 4.5-21B-A3B** | $0.056 | $0.224 | 128K | 🔥 Small/fast |
| Ernie 4.5-VL-28B | $0.112 | $0.448 | 128K | Vision small |
| Ernie 4.0-Turbo | $4.20 | $8.40 | 8K | Previous gen |
| Ernie 4.0-8K | $4.20 | $8.40 | 8K | Standard |
| **Ernie Speed** | $0.56 | $0.56 | 128K | Fast |
| **Ernie Speed Pro-128K** | $0.08 | $0.08 | 128K | 🔥 Cheapest |
| Ernie Novel-8K | $5.60 | $5.60 | 8K | Creative writing |

**Features:**
- 🇨🇳 Chinese provider
- 2.4T parameters (Ernie 5.0)
- #1 on LMArena
- MoE architecture
- PaddlePaddle framework
- Free tier available

---

### 5. Moonshot AI (Kimi)
**API:** `https://api.moonshot.cn/v1` (OpenAI-compatible)  
**Platform:** https://www.moonshot.cn

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| **Kimi K2.5** | $0.50 | $1.50 | 128K | Agentic (100 parallel) |
| Kimi K2 | $0.50 | $1.50 | 128K | Coding champion |
| Kimi K1.5 | $0.50 | $1.50 | 128K | Long context |

**Features:**
- 🇨🇳 Chinese provider
- 65.8% SWE-bench
- 100 parallel sub-agents
- $500M Series C at $4.3B valuation
- Long context specialist

---

### 6. Tencent (Hunyuan)
**API:** https://cloud.tencent.com  
**Access:** Tencent Cloud AI

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Hunyuan Pro | ~$2 | ~$6 | 256K | Flagship |
| Hunyuan Standard | ~$1 | ~$3 | 128K | General |
| Hunyuan Lite | ~$0.30 | ~$1 | 128K | Fast |

**Features:**
- 🇨🇳 Chinese provider
- Integrated with WeChat, QQ
- Tencent Games integration
- Multimodal (text, image, video)
- Gaming & entertainment focus

---

### 7. Baichuan Intelligence
**API:** Limited international access

| Model | Input/M | Output/M | Context | Best For |
|-------|---------|----------|---------|----------|
| Baichuan 4 | ~$2 | ~$6 | 128K | General |
| Baichuan 3 | ~$1 | ~$3 | 128K | Previous |
| Baixiaoying | Free | Free | - | Consumer app |

**Features:**
- 🇨🇳 Chinese provider
- Series A (~$693M) with Alibaba/Tencent

---

## Aggregators / Proxies

These services provide unified access to multiple models:

| Provider | Pricing | Models | Features |
|----------|---------|--------|----------|
| **Together AI** | Variable | 100+ open-source | Fast inference |
| **Fireworks AI** | Variable | 50+ models | Serverless |
| **Groq** | $0.05-0.50 | Llama, Mixtral | ⚡ Fastest (LPUs) |
| **Novita AI** | Markup | 350+ models | One API key |
| **Krater AI** | Markup | 350+ models | Credit-based |
| **APIYI** | ~90% official | Chinese models | China proxy |
| **Puter.js** | User-pays | Various | Free for devs |

---

## Cost Comparison Matrix

### Code Generation Cost (2K input + 1.5K output tokens)

| Provider | Model | Cost/Request | Cost/500 req/month |
|----------|-------|-------------|-------------------|
| Alibaba | Qwen3-235B | $0.00109 | $0.55 |
| Baidu | Ernie 4.5-21B | $0.00045 | $0.22 |
| ByteDance | Seed 2.0 Mini | $0.00040 | $0.20 |
| **Mistral** | **Nemo** | **$0.00008** | **$0.04** 🔥 |
| Mistral | Small 3.1 | $0.00023 | $0.11 |
| Google | Gemini 2.0 Flash-Lite | $0.00019 | $0.09 |
| xAI | Grok 4.1 Fast | $0.00065 | $0.33 |
| DeepSeek | V3.2 | $0.00119 | $0.60 |
| OpenAI | GPT-5 Nano | $0.00070 | $0.35 |
| OpenAI | GPT-4o Mini | $0.00120 | $0.60 |

---

## Recommended Routing

### Tier 1: Ultra-Cheap (<$0.30/M blended)
1. **Mistral Nemo** - $0.02/$0.04 - Simple tasks
2. **Gemini 2.0 Flash-Lite** - $0.075/$0.30 - General tasks
3. **Baidu Ernie Speed Pro** - $0.08/$0.08 - China access

### Tier 2: Best Value ($0.30-$1.00/M blended)
1. **Mistral Small 3.1** - $0.03/$0.11 - Best overall value
2. **Alibaba Qwen3-235B** - $0.136/$0.544 - Open source leader
3. **DeepSeek V3.2** - $0.28/$0.42 - Reasoning + chat
4. **Gemini 3.1 Flash-Lite** - $0.25/$1.50 - Fast & capable

### Tier 3: Mid-Range ($1.00-$5.00/M blended)
1. **GPT-4o Mini** - $0.15/$0.60 - Reliable, fast
2. **Gemini 2.5 Pro** - $1.25/$10.00 - 2M context
3. **Claude Haiku 4.5** - $1.00/$5.00 - Fast Claude
4. **ByteDance Seed 2.0 Pro** - $0.47/$2.37 - China best

### Tier 4: Premium (>$5.00/M blended)
1. **GPT-4o** - $2.50/$10.00 - General purpose
2. **Claude Sonnet 4.6** - $3.00/$15.00 - Best coding
3. **GPT-5** - $1.25/$10.00 - New flagship
4. **Claude Opus 4.6** - $5.00/$25.00 - Most capable

---

## Integration Priority for Orchestrator

### High Priority (Add Immediately)
1. ✅ **Alibaba Qwen3-235B** - OpenAI compatible, open source, $0.136/$0.544
2. ✅ **ByteDance Seed 2.0** - OpenAI compatible, $0.47/$2.37, 155M users
3. ✅ **Mistral Small 3.1** - OpenAI compatible, $0.03/$0.11, European
4. ✅ **xAI Grok 4.1 Fast** - OpenAI compatible, $0.20/$0.50, 2M context

### Medium Priority
5. **Zhipu GLM-4.7** - Coding specialist, $3/$15 or flat $15/month
6. **Baidu Ernie 4.5** - Cheapest option, $0.056/$0.224
7. **Moonshot Kimi K2.5** - 100 parallel agents

### Low Priority
8. Tencent Hunyuan - WeChat ecosystem
9. Cohere Command R - RAG specialist
10. Baichuan - Limited access

---

## API Key Environment Variables

```bash
# Western Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...
MISTRAL_API_KEY=
XAI_API_KEY=xai-...
COHERE_API_KEY=

# Chinese Providers
DASHSCOPE_API_KEY=sk-...          # Alibaba Qwen
VOLCENGINE_API_KEY=               # ByteDance
ZHIPU_API_KEY=                    # Zhipu GLM
BAIDU_API_KEY=                    # Baidu Ernie
MOONSHOT_API_KEY=                 # Kimi
TENCENT_API_KEY=                  # Hunyuan
```

---

## Notes

- 🇨🇳 Chinese providers may require phone verification
- All prices in USD per 1M tokens (input/output)
- Context = maximum token window
- Cached input discounts not shown (typically 75-90%)
- Prices as of March 2026 - verify current rates

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-04  
**Author:** AI Research Assistant
