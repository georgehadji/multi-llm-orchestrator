# API Keys Reference for Multi-LLM Orchestrator

This document provides a comprehensive reference for configuring API keys for all supported LLM providers.

**Quick Setup:** Copy `.env.example` to `.env` and uncomment the providers you want to use.

---

## Table of Contents

- [Western Providers](#western-providers)
  - [OpenAI](#openai)
  - [Google Gemini](#google-gemini)
  - [Anthropic Claude](#anthropic-claude)
  - [Mistral AI](#mistral-ai)
  - [xAI (Grok)](#xai-grok)
  - [Cohere](#cohere)
  - [DeepSeek](#deepseek)
- [Chinese Providers](#chinese-providers)
  - [Alibaba / Qwen](#alibaba--qwen)
  - [ByteDance / Seed](#bytedance--seed)
  - [Zhipu / GLM](#zhipu--glm)
  - [Baidu / Ernie](#baidu--ernie)
  - [Tencent / Hunyuan](#tencent--hunyuan)
  - [MiniMax](#minimax)
  - [Baichuan](#baichuan)
  - [Moonshot / Kimi](#moonshot--kimi)

---

## Western Providers

### OpenAI

**Models:** GPT-4o, GPT-4o-mini, GPT-5, GPT-5.2-Pro, o1, o3, o4, o4-mini

**Environment Variable:**
```bash
OPENAI_API_KEY=sk-...
```

**Get Key:** https://platform.openai.com/api-keys

**Base URL:** `https://api.openai.com/v1` (default)

---

### Google Gemini

**Models:** Gemini 2.0 Flash, 2.0 Flash-Lite, 2.5 Flash, 2.5 Pro

**Environment Variable:**
```bash
GOOGLE_API_KEY=...
# Alternative: GEMINI_API_KEY=...
```

**Get Key:** https://makersuite.google.com/app/apikey

---

### Anthropic Claude

**Models:** Claude 4, 4.6, 4.6-Opus, 4.5-Sonnet, 4.1, 3.7-Sonnet, 3.5-Haiku

**Environment Variable:**
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Get Key:** https://console.anthropic.com/

**Note:** Native Anthropic SDK (not OpenAI-compatible)

---

### Mistral AI

**Models:** Mistral Large 2, Medium 3, Small 3.1, Nemo, Codestral, Pixtral

**Environment Variable:**
```bash
MISTRAL_API_KEY=...
```

**Get Key:** https://console.mistral.ai/

**Base URL:** `https://api.mistral.ai/v1`

**Pricing:** Very competitive - Nemo at $0.02/$0.04 per 1M tokens

---

### xAI (Grok)

**Models:** Grok 4, Grok 3, Grok 4 Vision, Grok Code Fast

**Environment Variables:**
```bash
XAI_API_KEY=...
# Alternative: GROK_API_KEY=...
```

**Get Key:** https://console.x.ai/

**Base URL:** `https://api.x.ai/v1`

**Note:** Requires X Premium subscription for API access

---

### Cohere

**Models:** Command A, Command R7B, Command R+, R

**Environment Variable:**
```bash
COHERE_API_KEY=...
```

**Get Key:** https://dashboard.cohere.com/

**Note:** Uses native Cohere SDK (not OpenAI-compatible)

---

### DeepSeek

**Models:** DeepSeek-V3.2, DeepSeek-R1 (Reasoner)

**Environment Variable:**
```bash
DEEPSEEK_API_KEY=...
```

**Get Key:** https://platform.deepseek.com/api_keys

**Base URL:** `https://api.deepseek.com/v1`

**⚠️ Warning:** Can be slow (180s+ timeout). Used as quality fallback only.

**Pricing:** Very cheap - $0.28/$0.42 per 1M tokens

---

## Chinese Providers

### Alibaba / Qwen

**Models:** Qwen-3-235B, Qwen-Plus, Qwen-Max, Qwen-Turbo, Qwen-VL

**Environment Variable:**
```bash
DASHSCOPE_API_KEY=...
```

**Get Key:** https://dashscope.console.aliyun.com/

**Base URLs:**
- International: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- China: `https://dashscope.aliyuncs.com/compatible-mode/v1`

**Optional:** Set `DASHSCOPE_API_BASE` to override default

---

### ByteDance / Seed

**Models:** Seed-2.0-Pro, Seed-1.5-VL, Doubao-1.5-Pro

**Environment Variables:**
```bash
ARK_API_KEY=...
# Alternative: VOLCENGINE_API_KEY=...
```

**Get Key:** https://console.volcengine.com/

**Base URL:** `https://ark.cn-beijing.volces.com/api/v3`

---

### Zhipu / GLM

**Models:** GLM-5, GLM-4.7, GLM-4.6, GLM-4.5, GLM-4.5-Flash (FREE)

**Environment Variables:**
```bash
ZHIPUAI_API_KEY=...
# Alternative: ZHIPU_API_KEY=...
```

**Get Key:** 
- International: https://z.ai/
- China: https://open.bigmodel.cn/

**Base URL:** `https://api.z.ai/api/coding/paas/v4`

---

### Baidu / Ernie

**Models:** Ernie 4.5, 4.0, 3.5, Speed-Pro, Lite-Pro

**Environment Variables (Method 1 - Recommended):**
```bash
QIANFAN_ACCESS_KEY=...
QIANFAN_SECRET_KEY=...
```

**Environment Variables (Method 2 - Legacy):**
```bash
QIANFAN_AK=...
QIANFAN_SK=...
```

**Get Key:** https://console.bce.baidu.com/qianfan/

**Note:** Baidu uses its own API format. Currently requires custom implementation or proxy service.

---

### Tencent / Hunyuan

**Models:** Hunyuan-Pro, Standard, Lite, Role

**Environment Variables (Method 1 - Tencent Cloud):**
```bash
TENCENTCLOUD_SECRET_ID=...
TENCENTCLOUD_SECRET_KEY=...
```

**Environment Variables (Method 2):**
```bash
HUNYUAN_API_KEY=...
# Alternative: TENCENT_API_KEY=...
```

**Get Key:** https://console.cloud.tencent.com/

**Note:** Tencent uses standard Tencent Cloud authentication.

---

### MiniMax

**Models:** MiniMax-Text-01, Speech-01

**Environment Variable:**
```bash
MINIMAX_API_KEY=...
```

**Get Key:** https://www.minimaxi.com/

**Base URL:** `https://api.minimaxi.chat/v1`

---

### Baichuan

**Models:** Baichuan-4-Air, 4-Turbo, 3-Turbo, 2-Turbo

**Environment Variable:**
```bash
BAICHUAN_API_KEY=...
```

**Get Key:** https://platform.baichuan-ai.com/

---

### Moonshot / Kimi

**Models:** Kimi K2.5

**⚠️ Note:** Currently removed from this orchestrator due to compatibility issues.

**Environment Variable:**
```bash
MOONSHOT_API_KEY=...
```

**Get Key:** https://platform.moonshot.cn/

---

## Quick Reference Table

| Provider | Variable(s) | Base URL | SDK |
|----------|-------------|----------|-----|
| OpenAI | `OPENAI_API_KEY` | `api.openai.com` | openai |
| Google | `GOOGLE_API_KEY` | - | google-genai |
| Anthropic | `ANTHROPIC_API_KEY` | - | anthropic |
| Mistral | `MISTRAL_API_KEY` | `api.mistral.ai` | openai |
| xAI | `XAI_API_KEY` or `GROK_API_KEY` | `api.x.ai` | openai |
| Cohere | `COHERE_API_KEY` | - | cohere |
| DeepSeek | `DEEPSEEK_API_KEY` | `api.deepseek.com` | openai |
| Alibaba | `DASHSCOPE_API_KEY` | `dashscope-intl.aliyuncs.com` | openai |
| ByteDance | `ARK_API_KEY` or `VOLCENGINE_API_KEY` | `ark.cn-beijing.volces.com` | openai |
| Zhipu | `ZHIPUAI_API_KEY` or `ZHIPU_API_KEY` | `api.z.ai` | openai |
| Baidu | `QIANFAN_ACCESS_KEY` + `SECRET_KEY` | - | custom |
| Tencent | `TENCENTCLOUD_SECRET_ID` + `SECRET_KEY` | - | custom |
| MiniMax | `MINIMAX_API_KEY` | `api.minimaxi.chat` | openai |
| Baichuan | `BAICHUAN_API_KEY` | - | custom |
| Moonshot | `MOONSHOT_API_KEY` | `api.moonshot.cn` | openai |

---

## Cost Comparison (Per 1M Tokens)

| Provider | Cheapest Model | Input | Output |
|----------|---------------|-------|--------|
| Mistral | Nemo | $0.02 | $0.04 |
| Gemini | 2.0 Flash-Lite | $0.075 | $0.30 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| DeepSeek | Chat | $0.28 | $0.42 |
| Qwen | Turbo | $0.27 | $0.82 |
| Baichuan | 2-Turbo | $0.55 | $1.32 |
| Zhipu | 4.5-Flash | **FREE** | **FREE** |

---

## Troubleshooting

### API Key Not Recognized

1. Ensure the `.env` file is in the project root
2. Restart your terminal/IDE after adding keys
3. Check for typos in variable names

### "Provider not available" Warning

The orchestrator will skip providers without API keys. To use a provider:
1. Get an API key from the provider's console
2. Add it to your `.env` file
3. Restart the application

### Timeout Errors

- **DeepSeek:** Known to be slow. The orchestrator routes to faster models by default.
- **xAI:** Requires X Premium subscription. Free tier may have rate limits.

### Region-Specific Issues

- **Alibaba Qwen:** Use `DASHSCOPE_API_BASE` to switch between international and China endpoints
- **Tencent/Baidu:** May require China-region accounts for some features

---

## Security Best Practices

1. **Never commit `.env` files** - Add `.env` to `.gitignore`
2. **Use separate keys per environment** - Dev, Staging, Production
3. **Rotate keys regularly** - Especially after team changes
4. **Set budget limits** - Use `MAX_DAILY_BUDGET_USD` and `MAX_MONTHLY_BUDGET_USD`
5. **Monitor usage** - Check provider dashboards for unexpected spikes
