#!/usr/bin/env python3
"""
Update models.py to use OpenRouter-only routing table.
"""
import re

# Read the file
with open('orchestrator/models.py', 'r', encoding='utf-8') as f:
    content = f.read()

# New routing table (OpenRouter only)
new_routing_table = '''# ─────────────────────────────────────────────
# Routing table (priority-ordered per task type)
# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER ONLY - All models accessible via OpenRouter
# ═══════════════════════════════════════════════════════════════════════════════

ROUTING_TABLE: dict[TaskType, list[Model]] = {
    # CODE_GEN: Cheapest capable first, then quality
    TaskType.CODE_GEN:     [
        Model.PHI_4,                # $0.07/$0.14 — Microsoft 14B
        Model.GEMMA_3_27B,          # $0.08/$0.20 — Google open-weights
        Model.LLAMA_3_3_70B,        # $0.12/$0.30 — Meta 70B
        Model.LLAMA_4_SCOUT,        # $0.11/$0.34 — Meta 109B MoE
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B MoE
        Model.HERMES_3_70B,         # $0.40/$0.40 — Nous fine-tuned
        Model.LLAMA_3_1_405B,       # $2.00/$2.00 — Meta 405B
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — best coding
        Model.GPT_4O,               # $2.50/$10 — premium
    ],

    # CODE_REVIEW: Fast and accurate
    TaskType.CODE_REVIEW:  [
        Model.PHI_4_REASONING,      # $0.07/$0.35 — Microsoft CoT
        Model.GEMMA_3_27B,          # $0.08/$0.20 — Google
        Model.LLAMA_3_3_70B,        # $0.12/$0.30 — Meta 70B
        Model.LLAMA_4_SCOUT,        # $0.11/$0.34 — Meta fast
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B
        Model.HERMES_3_70B,         # $0.40/$0.40 — Nous
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — premium
    ],

    # REASONING: Reasoning models prioritized
    TaskType.REASONING:    [
        Model.PHI_4_REASONING,      # $0.07/$0.35 — Microsoft CoT
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B
        Model.LLAMA_3_1_405B,       # $2.00/$2.00 — Meta frontier
        Model.O3_MINI,              # $1.10/$4.40 — OpenAI
        Model.O4_MINI,              # $1.50/$6.00 — OpenAI
        Model.CLAUDE_OPUS_4_6,      # $5/$25 — most capable
    ],

    # WRITING: Quality and creativity
    TaskType.WRITING:      [
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B
        Model.HERMES_3_70B,         # $0.40/$0.40 — Nous
        Model.LLAMA_3_1_405B,       # $2.00/$2.00 — Meta frontier
        Model.GPT_4O,               # $2.50/$10 — excellent
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — excellent prose
    ],

    # DATA_EXTRACT: Cheapest first
    TaskType.DATA_EXTRACT: [
        Model.PHI_4,                # $0.07/$0.14 — Microsoft
        Model.GEMMA_3_27B,          # $0.08/$0.20 — Google
        Model.LLAMA_3_3_70B,        # $0.12/$0.30 — Meta 70B
        Model.LLAMA_4_SCOUT,        # $0.11/$0.34 — Meta
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
    ],

    # SUMMARIZE: Cheap with good context
    TaskType.SUMMARIZE:    [
        Model.PHI_4,                # $0.07/$0.14 — fast
        Model.GEMMA_3_27B,          # $0.08/$0.20 — concise
        Model.LLAMA_3_3_70B,        # $0.12/$0.30 — accurate
        Model.LLAMA_4_SCOUT,        # $0.11/$0.34 — fast
    ],

    # EVALUATE: Reliable evaluation
    TaskType.EVALUATE:     [
        Model.LLAMA_4_MAVERICK,     # $0.17/$0.17 — Meta 400B
        Model.HERMES_3_70B,         # $0.40/$0.40 — Nous
        Model.LLAMA_3_1_405B,       # $2.00/$2.00 — Meta
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — excellent
        Model.GPT_4O,               # $2.50/$10 — reliable
    ],
}
'''

# Find and replace the ROUTING_TABLE section
pattern = r'# ─────────────────────────────────────────────\n# Routing table.*?^}(?=\n\n\n# ─────────────────────────────────────────────\n# Fallback chains)'
replacement = new_routing_table.rstrip()

content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open('orchestrator/models.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated ROUTING_TABLE to OpenRouter-only")
