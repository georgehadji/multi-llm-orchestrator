# AI Orchestrator — Agent Model Assignments
>
> Author: Georgios-Chrysovalantis Chatzivantsidis  
> Date: 2026-05-23  
>
> Best-in-class models for each agent role, derived from OpenRouter
> live API analysis of 358 models. Each agent gets budget + premium tiers.

---

## ArchitectAgent (design decisions, architecture)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `OWL_ALPHA` | $0.00 | 1M | Free fallback |
| **Budget 1** | `MAESTRO_REASONING` | $0.90 | 131K | Primary budget reasoner |
| **Budget 2** | `DEEPSEEK_REASONER` | $0.70 | 164K | Fallback budget reasoner |
| **Premium 1** | `CLAUDE_SONNET_4_6` | $3.00 | 1M | Primary premium architect |
| **Premium 2** | `GEMINI_2_5_PRO` | $1.25 | 1M | Cheaper premium, huge context |

## DeveloperAgent (code generation)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `DEEPSEEK_V4_FLASH_FREE` | $0.00 | 1M | Free fallback |
| **Budget 1** | `CODESTRAL_2508` | $0.30 | 256K | Primary budget coder |
| **Budget 2** | `QWEN_3_CODER_FLASH` | $0.195 | 1M | Cheaper coder, huge context |
| **Premium 1** | `GPT_5` | $1.25 | 400K | Primary premium coder |
| **Premium 2** | `CLAUDE_SONNET_4_6` | $3.00 | 1M | Best code quality |

## ReviewerAgent (code review, security audit)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `NEMOTRON_NANO_OMNI_FREE` | $0.00 | 256K | Free reasoning reviewer |
| **Budget 1** | `QWEN_3_6_PLUS` | $0.325 | 1M | Primary budget reviewer |
| **Budget 2** | `PERPLEXITY_SONAR_PRO` | $3.00 | 200K | Web-search enhanced review |
| **Premium 1** | `CLAUDE_SONNET_4_6` | $3.00 | 1M | Best security review |
| **Premium 2** | `XAI_GROK_4_3` | $1.25 | 1M | Cheaper premium reviewer |

## TesterAgent (test generation)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `LING_2_6_FLASH` | $0.01 | 262K | Cheapest paid ($0.01/M) |
| **Budget 1** | `QWEN_3_CODER_NEXT` | $0.11 | 262K | Cheapest code model |
| **Budget 2** | `CODESTRAL_2508` | $0.30 | 256K | Best test generation |
| **Premium 1** | `GPT_5` | $1.25 | 400K | Primary premium tester |
| **Premium 2** | `CLAUDE_SONNET_4_6` | $3.00 | 1M | Best test coverage |

## DevOpsAgent (infrastructure, CI/CD)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `DEEPSEEK_V4_FLASH_FREE` | $0.00 | 1M | Free config generation |
| **Budget 1** | `DEEPSEEK_V4_FLASH` | $0.10 | 1M | Primary budget DevOps |
| **Budget 2** | `QWEN_3_CODER_FLASH` | $0.195 | 1M | Cheaper coder for Dockerfiles |
| **Premium 1** | `GPT_5_2_CODEX` | $1.75 | 400K | Best infra-as-code generation |
| **Premium 2** | `CLAUDE_SONNET_4_6` | $3.00 | 1M | Complex infrastructure review |

## ResearcherAgent (web search, research)

| Tier | Model | $/M Input | Ctx | Why |
|------|-------|-----------|-----|-----|
| **Free** | `OWL_ALPHA` | $0.00 | 1M | Free search alternative |
| **Budget 1** | `SONAR_PRO` | $3.00 | 200K | Web-search enhanced |
| **Budget 2** | `QWEN_3_6_PLUS` | $0.325 | 1M | General purpose, large context |
| **Premium 1** | `SONAR_DEEP_RESEARCH` | $2.00 | 128K | Multi-source deep research |
| **Premium 2** | `GPT_5` | $1.25 | 400K | General research synthesis |

## Cross-Cutting: Free Models

For development/testing, these models cost $0:

- `OWL_ALPHA` — 1M context, free (OpenRouter's own model)
- `DEEPSEEK_V4_FLASH_FREE` — 1M context, free
- `QWEN_3_CODER_FREE` — 1M context, free
- `MINIMAX_M2_5` — 204K context, free

## Cost Summary

| Agent | Budget/max | Premium/max | Savings |
|-------|-----------|-------------|---------|
| ArchitectAgent | $0.90/10K | $3.00/10K | 70% |
| DeveloperAgent | $0.30/10K | $1.25/10K | 76% |
| ReviewerAgent | $0.325/10K | $3.00/10K | 89% |
| TesterAgent | $0.11/10K | $1.25/10K | 91% |
| DevOpsAgent | $0.10/10K | $1.75/10K | 94% |
| ResearcherAgent | $3.00/10K | $2.00/10K | -33% |
