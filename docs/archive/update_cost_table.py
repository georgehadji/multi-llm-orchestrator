#!/usr/bin/env python3
"""
Update models.py to have a clean OpenRouter-only COST_TABLE and remove unused model references.
"""
import re

# Read the file
with open('orchestrator/models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# New simplified COST_TABLE (OpenRouter models only)
new_cost_table = '''# ─────────────────────────────────────────────
# Cost table (per 1M tokens, USD)
# ─────────────────────────────────────────────

COST_TABLE: dict[Model, dict[str, float]] = {
    # OpenAI Models (via OpenRouter)
    Model.GPT_4O:             {"input": 2.50,  "output": 10.00},
    Model.GPT_4O_MINI:        {"input": 0.15,  "output": 0.60},
    Model.GPT_5:              {"input": 1.25,  "output": 10.00},
    Model.GPT_5_MINI:         {"input": 0.25,  "output": 2.00},
    Model.GPT_5_NANO:         {"input": 0.05,  "output": 0.40},
    Model.O1:                 {"input": 15.00, "output": 60.00},
    Model.O3_MINI:            {"input": 1.10,  "output": 4.40},
    Model.O4_MINI:            {"input": 1.50,  "output": 6.00},

    # Google Gemini Models (via OpenRouter)
    Model.GEMINI_PRO:             {"input": 1.25,  "output": 10.00},
    Model.GEMINI_FLASH:           {"input": 0.15,  "output": 0.60},
    Model.GEMINI_FLASH_LITE:      {"input": 0.075, "output": 0.30},

    # Anthropic Claude Models (via OpenRouter)
    Model.CLAUDE_3_5_SONNET:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_3_OPUS:      {"input": 15.00, "output": 75.00},
    Model.CLAUDE_3_HAIKU:     {"input": 0.25,  "output": 1.25},
    Model.CLAUDE_SONNET_4_5:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_SONNET_4_6:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_OPUS_4_5:    {"input": 5.00,  "output": 25.00},
    Model.CLAUDE_OPUS_4_6:    {"input": 5.00,  "output": 25.00},
    Model.CLAUDE_HAIKU_4_5:   {"input": 1.00,  "output": 5.00},

    # DeepSeek Models (via OpenRouter)
    Model.DEEPSEEK_CHAT:      {"input": 0.28,  "output": 0.42},
    Model.DEEPSEEK_REASONER:  {"input": 0.28,  "output": 0.42},
    Model.DEEPSEEK_V3:        {"input": 0.27,  "output": 1.10},
    Model.DEEPSEEK_R1:        {"input": 0.55,  "output": 2.19},

    # Meta LLaMA Models (OpenRouter)
    Model.LLAMA_4_MAVERICK:   {"input": 0.17,  "output": 0.17},  # 400B MoE
    Model.LLAMA_4_SCOUT:      {"input": 0.11,  "output": 0.34},  # 109B MoE
    Model.LLAMA_3_3_70B:      {"input": 0.12,  "output": 0.30},  # 70B
    Model.LLAMA_3_1_405B:     {"input": 2.00,  "output": 2.00},  # 405B

    # Microsoft Phi Models (OpenRouter)
    Model.PHI_4:              {"input": 0.07,  "output": 0.14},  # 14B
    Model.PHI_4_REASONING:    {"input": 0.07,  "output": 0.35},  # 14B + CoT

    # Google Gemma Models (OpenRouter)
    Model.GEMMA_3_27B:        {"input": 0.08,  "output": 0.20},  # 27B

    # Nous Research Hermes (OpenRouter)
    Model.HERMES_3_70B:       {"input": 0.40,  "output": 0.40},  # 70B fine-tuned

    # OpenRouter Auto
    Model.OPENROUTER_AUTO:    {"input": 0.00,  "output": 0.00},  # Dynamic
}
'''

# Find and replace the COST_TABLE section
pattern = r'# ─────────────────────────────────────────────\n# Cost table.*?^}(?=\n\n\n# ─────────────────────────────────────────────\n# Routing table)'
replacement = new_cost_table.rstrip()

content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open('orchestrator/models.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated COST_TABLE to OpenRouter-only")
