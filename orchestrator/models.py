"""
Multi-LLM Orchestrator — Core Models & Types
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis
All data structures, enums, routing tables, cost tables, budget logic.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskType(str, Enum):
    CODE_GEN = "code_generation"
    CODE_REVIEW = "code_review"
    REASONING = "complex_reasoning"
    WRITING = "creative_writing"
    DATA_EXTRACT = "data_extraction"
    SUMMARIZE = "summarization"
    EVALUATE = "evaluation"


class Model(str, Enum):
    # ═══════════════════════════════════════════════════════
    # OPENAI MODELS
    # ═══════════════════════════════════════════════════════
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_PRO = "gpt-5.2-pro"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    O1 = "o1"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"
    O4_MINI = "o4-mini"
    
    # ═══════════════════════════════════════════════════════
    # GOOGLE GEMINI MODELS
    # ═══════════════════════════════════════════════════════
    GEMINI_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    
    # ═══════════════════════════════════════════════════════
    # ANTHROPIC CLAUDE MODELS
    # ═══════════════════════════════════════════════════════
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    
    # ═══════════════════════════════════════════════════════
    # DEEPSEEK MODELS
    # ═══════════════════════════════════════════════════════
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_V3_2 = "deepseek-v3.2"
    DEEPSEEK_R1 = "deepseek-r1"
    
    # ═══════════════════════════════════════════════════════
    # MINIMAX MODELS
    # ═══════════════════════════════════════════════════════
    MINIMAX_TEXT_01 = "MiniMax-Text-01"
    MINIMAX_M2 = "minimax-m2"
    MINIMAX_M2_5 = "minimax-m2.5"
    
    # ═══════════════════════════════════════════════════════
    # MISTRAL AI MODELS
    # ═══════════════════════════════════════════════════════
    MISTRAL_NEMO = "mistral-nemo"
    MISTRAL_SMALL_3_1 = "mistral-small-3.1"
    MISTRAL_MEDIUM_3 = "mistral-medium-3"
    MISTRAL_LARGE_3 = "mistral-large-3"
    CODESTRAL = "codestral"
    DEVSTRAL = "devstral"
    MAGISTRAL_SMALL = "magistral-small"
    MAGISTRAL_MEDIUM = "magistral-medium"
    MINISTRAL_3B = "ministral-3b"
    MINISTRAL_8B = "ministral-8b"
    
    # ═══════════════════════════════════════════════════════
    # XAI GROK MODELS
    # ═══════════════════════════════════════════════════════
    GROK_3 = "grok-3"
    GROK_3_MINI = "grok-3-mini"
    GROK_4 = "grok-4"
    GROK_4_1_FAST = "grok-4.1-fast"
    
    # ═══════════════════════════════════════════════════════
    # COHERE MODELS
    # ═══════════════════════════════════════════════════════
    COMMAND_R = "command-r"
    COMMAND_R_PLUS = "command-r-plus"
    COMMAND_R7B = "command-r7b"
    COMMAND_A = "command-a"
    
    # ═══════════════════════════════════════════════════════
    # ALIBABA QWEN MODELS
    # ═══════════════════════════════════════════════════════
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"
    QWEN_MAX = "qwen-max"
    QWEN_LONG = "qwen-long"
    QWEN_3_235B = "qwen3-235b-a22b"
    QWEN_3_CODER_30B = "qwen3-coder-30b"
    QWEN_3_32B = "qwen3-32b"
    QWEN_VL = "qwen-vl"
    QWEN_MATH = "qwen-math"
    
    # ═══════════════════════════════════════════════════════
    # BYTEDANCE SEED MODELS
    # ═══════════════════════════════════════════════════════
    SEED_2_0_PRO = "seed-2.0-pro"
    SEED_2_0_LITE = "seed-2.0-lite"
    SEED_2_0_MINI = "seed-2.0-mini"
    SEED_2_0_CODE = "seed-2.0-code"
    
    # ═══════════════════════════════════════════════════════
    # ZHIPU GLM MODELS
    # ═══════════════════════════════════════════════════════
    GLM_4 = "glm-4"
    GLM_4_6 = "glm-4.6"
    GLM_4_7 = "glm-4.7"
    GLM_4_FLASH = "glm-4-flash"
    GLM_4_AIR = "glm-4-air"
    GLM_5 = "glm-5"
    
    # ═══════════════════════════════════════════════════════
    # BAIDU ERNIE MODELS
    # ═══════════════════════════════════════════════════════
    ERNIE_4_0_8K = "ernie-4.0-8k"
    ERNIE_4_0_TURBO = "ernie-4.0-turbo"
    ERNIE_4_5_21B = "ernie-4.5-21b"
    ERNIE_4_5_300B = "ernie-4.5-300b"
    ERNIE_SPEED = "ernie-speed"
    ERNIE_SPEED_PRO = "ernie-speed-pro"
    ERNIE_NOVEL = "ernie-novel"
    
    # ═══════════════════════════════════════════════════════
    # MOONSHOT KIMI MODELS
    # ═══════════════════════════════════════════════════════
    KIMI_K1_5 = "kimi-k1.5"
    KIMI_K2 = "kimi-k2"
    KIMI_K2_5 = "kimi-k2.5"
    
    # ═══════════════════════════════════════════════════════
    # TENCENT HUNYUAN MODELS
    # ═══════════════════════════════════════════════════════
    HUNYUAN_LITE = "hunyuan-lite"
    HUNYUAN_STANDARD = "hunyuan-standard"
    HUNYUAN_PRO = "hunyuan-pro"
    
    # ═══════════════════════════════════════════════════════
    # BAICHUAN MODELS
    # ═══════════════════════════════════════════════════════
    BAICHUAN_3 = "baichuan-3"
    BAICHUAN_4 = "baichuan-4"


class ProjectStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    COMPLETED_DEGRADED = "COMPLETED_DEGRADED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"


# ─────────────────────────────────────────────
# Provider detection
# ─────────────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=256)
def get_provider(model: Model) -> str:
    """
    Get provider name for a model.
    
    Uses LRU cache for O(1) repeated lookups.
    Cache size 256 covers all current + future models.
    """
    val = model.value.lower()
    if val.startswith("gpt") or val.startswith("o1") or val.startswith("o3") or val.startswith("o4"):
        return "openai"
    elif val.startswith("gemini"):
        return "google"
    elif val.startswith("claude"):
        return "anthropic"
    elif val.startswith("minimax"):
        return "minimax"
    elif val.startswith("deepseek"):
        return "deepseek"
    elif val.startswith("mistral") or val.startswith("codestral") or val.startswith("devstral") or val.startswith("magistral"):
        return "mistral"
    elif val.startswith("grok"):
        return "xai"
    elif val.startswith("command"):
        return "cohere"
    elif val.startswith("qwen"):
        return "alibaba"
    elif val.startswith("seed"):
        return "bytedance"
    elif val.startswith("glm"):
        return "zhipu"
    elif val.startswith("ernie"):
        return "baidu"
    elif val.startswith("kimi"):
        return "moonshot"
    elif val.startswith("hunyuan"):
        return "tencent"
    elif val.startswith("baichuan"):
        return "baichuan"
    return "unknown"


# ─────────────────────────────────────────────
# Cost table (per 1M tokens, USD)
# ─────────────────────────────────────────────

COST_TABLE: dict[Model, dict[str, float]] = {
    # ═══════════════════════════════════════════════════════
    # OPENAI
    # ═══════════════════════════════════════════════════════
    Model.GPT_4O:             {"input": 2.50,  "output": 10.00},
    Model.GPT_4O_MINI:        {"input": 0.15,  "output": 0.60},
    Model.GPT_5:              {"input": 1.25,  "output": 10.00},
    Model.GPT_5_MINI:         {"input": 0.25,  "output": 2.00},
    Model.GPT_5_NANO:         {"input": 0.05,  "output": 0.40},
    Model.GPT_5_2:            {"input": 1.75,  "output": 14.00},
    Model.GPT_5_2_PRO:        {"input": 21.00, "output": 168.00},
    Model.GPT_4_1:            {"input": 2.00,  "output": 8.00},
    Model.GPT_4_1_MINI:       {"input": 0.40,  "output": 1.60},
    Model.GPT_4_1_NANO:       {"input": 0.10,  "output": 0.40},
    Model.O1:                 {"input": 15.00, "output": 60.00},
    Model.O3:                 {"input": 2.00,  "output": 8.00},
    Model.O3_MINI:            {"input": 1.10,  "output": 4.40},
    Model.O3_PRO:             {"input": 20.00, "output": 80.00},
    Model.O4_MINI:            {"input": 1.50,  "output": 6.00},
    
    # ═══════════════════════════════════════════════════════
    # GOOGLE GEMINI
    # ═══════════════════════════════════════════════════════
    Model.GEMINI_PRO:             {"input": 1.25,  "output": 10.00},
    Model.GEMINI_FLASH:           {"input": 0.15,  "output": 0.60},
    Model.GEMINI_FLASH_LITE:      {"input": 0.075, "output": 0.30},
    Model.GEMINI_3_1_FLASH_LITE:  {"input": 0.25,  "output": 1.50},
    # GEMINI_2_0_FLASH removed - deprecated by Google
    # GEMINI_2_0_FLASH_LITE removed - deprecated by Google
    Model.GEMINI_2_5_FLASH_LITE:  {"input": 0.10,  "output": 0.40},
    Model.GEMINI_1_5_PRO:         {"input": 3.50,  "output": 10.50},
    
    # ═══════════════════════════════════════════════════════
    # ANTHROPIC CLAUDE
    # ═══════════════════════════════════════════════════════
    Model.CLAUDE_3_5_SONNET:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_3_OPUS:      {"input": 15.00, "output": 75.00},
    Model.CLAUDE_3_HAIKU:     {"input": 0.25,  "output": 1.25},
    Model.CLAUDE_SONNET_4_5:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_SONNET_4_6:  {"input": 3.00,  "output": 15.00},
    Model.CLAUDE_OPUS_4_5:    {"input": 5.00,  "output": 25.00},
    Model.CLAUDE_OPUS_4_6:    {"input": 5.00,  "output": 25.00},
    Model.CLAUDE_HAIKU_4_5:   {"input": 1.00,  "output": 5.00},
    
    # ═══════════════════════════════════════════════════════
    # DEEPSEEK
    # ═══════════════════════════════════════════════════════
    Model.DEEPSEEK_CHAT:      {"input": 0.28,  "output": 0.42},
    Model.DEEPSEEK_REASONER:  {"input": 0.28,  "output": 0.42},
    Model.DEEPSEEK_V3:        {"input": 0.27,  "output": 1.10},
    Model.DEEPSEEK_V3_2:      {"input": 0.28,  "output": 0.40},
    Model.DEEPSEEK_R1:        {"input": 0.55,  "output": 2.19},
    
    # ═══════════════════════════════════════════════════════
    # MINIMAX
    # ═══════════════════════════════════════════════════════
    Model.MINIMAX_TEXT_01:    {"input": 0.50,  "output": 1.50},
    Model.MINIMAX_M2:         {"input": 0.50,  "output": 1.50},
    Model.MINIMAX_M2_5:       {"input": 0.50,  "output": 1.50},
    
    # ═══════════════════════════════════════════════════════
    # MISTRAL AI
    # ═══════════════════════════════════════════════════════
    Model.MISTRAL_NEMO:       {"input": 0.02,  "output": 0.04},
    Model.MISTRAL_SMALL_3_1:  {"input": 0.03,  "output": 0.11},
    Model.MISTRAL_MEDIUM_3:   {"input": 0.40,  "output": 2.00},
    Model.MISTRAL_LARGE_3:    {"input": 0.50,  "output": 1.50},
    Model.CODESTRAL:          {"input": 0.30,  "output": 0.90},
    Model.DEVSTRAL:           {"input": 0.10,  "output": 0.30},
    Model.MAGISTRAL_SMALL:    {"input": 0.50,  "output": 1.50},
    Model.MAGISTRAL_MEDIUM:   {"input": 2.00,  "output": 5.00},
    Model.MINISTRAL_3B:       {"input": 0.04,  "output": 0.04},
    Model.MINISTRAL_8B:       {"input": 0.10,  "output": 0.10},
    
    # ═══════════════════════════════════════════════════════
    # XAI GROK
    # ═══════════════════════════════════════════════════════
    Model.GROK_3:             {"input": 2.00,  "output": 10.00},
    Model.GROK_3_MINI:        {"input": 0.10,  "output": 0.30},
    Model.GROK_4:             {"input": 3.00,  "output": 15.00},
    Model.GROK_4_1_FAST:      {"input": 0.20,  "output": 0.50},
    
    # ═══════════════════════════════════════════════════════
    # COHERE
    # ═══════════════════════════════════════════════════════
    Model.COMMAND_R:          {"input": 0.15,  "output": 0.60},
    Model.COMMAND_R_PLUS:     {"input": 2.50,  "output": 10.00},
    Model.COMMAND_R7B:        {"input": 0.15,  "output": 0.0375},
    Model.COMMAND_A:          {"input": 2.50,  "output": 10.00},
    
    # ═══════════════════════════════════════════════════════
    # ALIBABA QWEN
    # ═══════════════════════════════════════════════════════
    Model.QWEN_PLUS:          {"input": 0.50,  "output": 1.50},
    Model.QWEN_TURBO:         {"input": 0.20,  "output": 0.60},
    Model.QWEN_MAX:           {"input": 2.00,  "output": 6.00},
    Model.QWEN_LONG:          {"input": 0.10,  "output": 0.40},
    Model.QWEN_3_235B:        {"input": 0.136, "output": 0.544},
    Model.QWEN_3_CODER_30B:   {"input": 0.15,  "output": 0.60},
    Model.QWEN_3_32B:         {"input": 0.20,  "output": 0.80},
    Model.QWEN_VL:            {"input": 0.50,  "output": 1.50},
    Model.QWEN_MATH:          {"input": 0.50,  "output": 1.50},
    
    # ═══════════════════════════════════════════════════════
    # BYTEDANCE SEED
    # ═══════════════════════════════════════════════════════
    Model.SEED_2_0_PRO:       {"input": 0.47,  "output": 2.37},
    Model.SEED_2_0_LITE:      {"input": 0.20,  "output": 1.00},
    Model.SEED_2_0_MINI:      {"input": 0.05,  "output": 0.25},
    Model.SEED_2_0_CODE:      {"input": 0.30,  "output": 1.20},
    
    # ═══════════════════════════════════════════════════════
    # ZHIPU GLM
    # ═══════════════════════════════════════════════════════
    Model.GLM_4:              {"input": 4.00,  "output": 8.00},
    Model.GLM_4_6:            {"input": 2.37,  "output": 11.06},
    Model.GLM_4_7:            {"input": 3.00,  "output": 15.00},
    Model.GLM_4_FLASH:        {"input": 0.50,  "output": 1.00},
    Model.GLM_4_AIR:          {"input": 1.00,  "output": 2.00},
    Model.GLM_5:              {"input": 7.00,  "output": 17.00},
    
    # ═══════════════════════════════════════════════════════
    # BAIDU ERNIE
    # ═══════════════════════════════════════════════════════
    Model.ERNIE_4_0_8K:       {"input": 4.20,  "output": 8.40},
    Model.ERNIE_4_0_TURBO:    {"input": 4.20,  "output": 8.40},
    Model.ERNIE_4_5_21B:      {"input": 0.056, "output": 0.224},
    Model.ERNIE_4_5_300B:     {"input": 0.224, "output": 0.88},
    Model.ERNIE_SPEED:        {"input": 0.56,  "output": 0.56},
    Model.ERNIE_SPEED_PRO:    {"input": 0.08,  "output": 0.08},
    Model.ERNIE_NOVEL:        {"input": 5.60,  "output": 5.60},
    
    # ═══════════════════════════════════════════════════════
    # MOONSHOT KIMI
    # ═══════════════════════════════════════════════════════
    Model.KIMI_K1_5:          {"input": 0.50,  "output": 1.50},
    Model.KIMI_K2:            {"input": 0.50,  "output": 1.50},
    Model.KIMI_K2_5:          {"input": 0.50,  "output": 1.50},
    
    # ═══════════════════════════════════════════════════════
    # TENCENT HUNYUAN
    # ═══════════════════════════════════════════════════════
    Model.HUNYUAN_LITE:       {"input": 0.30,  "output": 1.00},
    Model.HUNYUAN_STANDARD:   {"input": 1.00,  "output": 3.00},
    Model.HUNYUAN_PRO:        {"input": 2.00,  "output": 6.00},
    
    # ═══════════════════════════════════════════════════════
    # BAICHUAN
    # ═══════════════════════════════════════════════════════
    Model.BAICHUAN_3:         {"input": 1.00,  "output": 3.00},
    Model.BAICHUAN_4:         {"input": 2.00,  "output": 6.00},
}


# ─────────────────────────────────────────────
# Routing table (priority-ordered per task type)
# ─────────────────────────────────────────────

ROUTING_TABLE: dict[TaskType, list[Model]] = {
    # ═══════════════════════════════════════════════════════════════════════════════
    # UPDATED 2026-03-04: Full integration of 70+ models from 15 providers
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # CODE_GEN: Cheapest capable first, then quality
    TaskType.CODE_GEN:     [
        # Tier 1: Ultra-cheap capable models
        Model.MISTRAL_NEMO,         # $0.02/$0.04 — cheapest capable
        Model.MISTRAL_SMALL_3_1,    # $0.03/$0.11 — best value overall
        Model.GEMINI_2_5_FLASH_LITE,# $0.075/$0.30 — cheapest mainstream
        Model.ERNIE_SPEED_PRO,      # $0.08/$0.08 — cheapest from China
        Model.GPT_5_NANO,           # $0.05/$0.40 — OpenAI entry level
        
        # Tier 2: Best value
        Model.GEMINI_3_1_FLASH_LITE,# $0.25/$1.50 — fast, good quality
        Model.QWEN_3_235B,          # $0.136/$0.544 — best Chinese value
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
        Model.GROK_4_1_FAST,        # $0.20/$0.50 — 2M context
        Model.SEED_2_0_MINI,        # $0.05/$0.25 — ByteDance cheap
        
        # Tier 3: Quality
        # DEEPSEEK_CHAT removed - too slow (180s+), use as fallback only
        Model.GEMINI_FLASH,         # $0.15/$0.60 — 1M context
        Model.CODESTRAL,            # $0.30/$0.90 — code specialist
        Model.DEVSTRAL,             # $0.10/$0.30 — agentic coding
        
        # Tier 4: Premium
        Model.GPT_4O,               # $2.50/$10 — premium quality
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — best coding performance
        Model.QWEN_MAX,             # $2/$6 — Chinese premium
        Model.GLM_4_7,              # $3/$15 — Zhipu coding specialist
    ],
    
    # CODE_REVIEW: Fast and accurate
    TaskType.CODE_REVIEW:  [
        Model.MISTRAL_SMALL_3_1,    # $0.03/$0.11 — excellent value
        Model.GEMINI_3_1_FLASH_LITE,# $0.25/$1.50 — fast reviews
        Model.QWEN_3_235B,          # $0.136/$0.544 — accurate
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
        # DEEPSEEK_CHAT removed - too slow, use as fallback only
        Model.GEMINI_FLASH,         # $0.15/$0.60 — 1M context for large reviews
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — premium review
        Model.QWEN_MAX,             # $2/$6 — Chinese premium
    ],
    
    # REASONING: Reasoning models prioritized
    TaskType.REASONING:    [
        # DEEPSEEK models moved to fallback - too slow (180s+)
        Model.O3_MINI,              # $1.10/$4.40 — OpenAI reasoning
        Model.O4_MINI,              # $1.50/$6.00 — OpenAI reasoning
        Model.QWEN_3_235B,          # $0.136/$0.544 — good reasoning
        Model.GROK_4_1_FAST,        # $0.20/$0.50 — 2M context reasoning
        Model.CLAUDE_OPUS_4_6,      # $5/$25 — most capable
        Model.O3,                   # $2/$8 — OpenAI mid-tier
        Model.GEMINI_PRO,           # $1.25/$10 — 2M context
        Model.GLM_5,                # $7/$17 — massive 744B model
    ],
    
    # WRITING: Quality and creativity
    TaskType.WRITING:      [
        Model.GPT_5,                # $1.25/$10 — best writing
        Model.GPT_4O,               # $2.50/$10 — excellent
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — excellent prose
        Model.QWEN_LONG,            # $0.10/$0.40 — 10M context for long docs
        Model.GEMINI_PRO,           # $1.25/$10 — 2M context
        Model.GEMINI_3_1_FLASH_LITE,# $0.25/$1.50 — bulk content
        Model.ERNIE_NOVEL,          # $5.60/$5.60 — creative specialist
        Model.HUNYUAN_PRO,          # $2/$6 — Tencent creative
    ],
    
    # DATA_EXTRACT: Cheapest first
    TaskType.DATA_EXTRACT: [
        Model.MISTRAL_NEMO,         # $0.02/$0.04 — cheapest
        Model.GEMINI_2_5_FLASH_LITE,# $0.075/$0.30 — cheapest mainstream
        Model.ERNIE_SPEED_PRO,      # $0.08/$0.08 — very cheap
        Model.GPT_5_NANO,           # $0.05/$0.40 — OpenAI cheap
        Model.QWEN_LONG,            # $0.10/$0.40 — 10M context for large data
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
        Model.QWEN_3_235B,          # $0.136/$0.544 — accurate
    ],
    
    # SUMMARIZE: Cheap with good context
    TaskType.SUMMARIZE:    [
        Model.MISTRAL_NEMO,         # $0.02/$0.04 — cheapest
        Model.GEMINI_2_5_FLASH_LITE,# $0.075/$0.30 — cheap + good
        Model.QWEN_LONG,            # $0.10/$0.40 — 10M context
        Model.GEMINI_FLASH,         # $0.15/$0.60 — 1M context
        Model.GPT_4O_MINI,          # $0.15/$0.60 — reliable
    ],
    
    # EVALUATE: Reliable evaluation
    TaskType.EVALUATE:     [
        Model.GPT_4O,               # Most reliable evaluator
        Model.QWEN_3_235B,          # $0.136/$0.544 — accurate & cheap
        Model.CLAUDE_HAIKU_4_5,     # $1/$5 — cheap Claude
        Model.CLAUDE_SONNET_4_6,    # $3/$15 — excellent evaluation
        Model.O4_MINI,              # $1.50/$6.00 — reasoning
        # DEEPSEEK removed from primary - too slow (180s+)
    ],
}


# ─────────────────────────────────────────────
# Fallback chains (always cross-provider)
# ─────────────────────────────────────────────

FALLBACK_CHAIN: dict[Model, Model] = {
    # OPTIMIZED 2026-03-04: Gemini 3.1 Flash-Lite added, all fallbacks cross-provider
    # Each fallback goes to a different provider to maximize availability
    
    # OpenAI fallbacks → DeepSeek (cost-effective, same quality tier)
    Model.GPT_4O:              Model.DEEPSEEK_CHAT,      # Premium → DeepSeek
    Model.GPT_4O_MINI:         Model.GEMINI_FLASH,       # Budget → Gemini
    Model.O4_MINI:             Model.DEEPSEEK_REASONER,  # Reasoning → DeepSeek-R1
    
    # Gemini fallbacks → OpenAI or DeepSeek
    Model.GEMINI_PRO:          Model.GPT_4O,             # Pro → GPT-4o
    Model.GEMINI_FLASH:        Model.GPT_4O_MINI,        # Flash → GPT-4o-mini
    Model.GEMINI_FLASH_LITE:   Model.GPT_4O_MINI,        # Lite → GPT-4o-mini
    Model.GEMINI_3_1_FLASH_LITE: Model.DEEPSEEK_CHAT,    # NEW: 3.1 Flash-Lite → DeepSeek
    
    # Claude fallbacks → OpenAI (both high-quality Western providers)
    Model.CLAUDE_3_5_SONNET:   Model.GPT_4O,             # Claude Sonnet → GPT-4o
    Model.CLAUDE_3_OPUS:       Model.GPT_4O,             # Claude Opus → GPT-4o
    Model.CLAUDE_3_HAIKU:      Model.GPT_4O_MINI,        # Claude Haiku → GPT-4o-mini
    
    # MiniMax fallback → OpenAI
    Model.MINIMAX_TEXT_01:     Model.GPT_4O,             # Minimax → GPT-4o
    
    # DeepSeek fallbacks → OpenAI (premium escalation)
    Model.DEEPSEEK_CHAT:       Model.GPT_4O,             # DeepSeek → GPT-4o
    Model.DEEPSEEK_REASONER:   Model.O4_MINI,            # R1 → o4-mini (both reasoning)
}


# ─────────────────────────────────────────────
# Per-task thresholds & limits
# ─────────────────────────────────────────────

DEFAULT_THRESHOLDS: dict[TaskType, float] = {
    TaskType.DATA_EXTRACT: 0.90,
    TaskType.SUMMARIZE:    0.80,
    TaskType.CODE_GEN:     0.85,
    # CODE_REVIEW: lowered from 0.85 — review quality depends on how much
    # source context the LLM received, which is often partial due to truncation.
    # 0.75 is a realistic target; scores above this reflect genuine analysis.
    TaskType.CODE_REVIEW:  0.75,
    TaskType.REASONING:    0.90,
    TaskType.WRITING:      0.80,
    # EVALUATE: lowered from 0.95 — evaluation outputs are open-ended prose;
    # scoring ≥ 0.95 requires near-perfect structured responses which LLMs
    # rarely produce without a domain-specific rubric.
    TaskType.EVALUATE:     0.80,
}

MAX_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     8192,  # raised: avoid unterminated strings mid-class
    TaskType.CODE_REVIEW:  4096,  # raised: full analysis without truncation
    TaskType.REASONING:    4096,
    TaskType.WRITING:      4096,
    TaskType.DATA_EXTRACT: 2048,
    TaskType.SUMMARIZE:    1024,
    TaskType.EVALUATE:     2048,  # raised: evaluation tasks need more room
}

# Model-specific max tokens limits (override MAX_OUTPUT_TOKENS)
MODEL_MAX_TOKENS: dict[Model, int] = {
    # Anthropic Claude models
    Model.CLAUDE_3_HAIKU:     4096,
    Model.CLAUDE_3_5_SONNET:  8192,
    Model.CLAUDE_3_OPUS:      4096,
    Model.CLAUDE_HAIKU_4_5:   4096,
    Model.CLAUDE_SONNET_4_5:  8192,
    Model.CLAUDE_SONNET_4_6:  8192,
    Model.CLAUDE_OPUS_4_5:    4096,
    Model.CLAUDE_OPUS_4_6:    4096,
    
    # Google Gemini models (high limits)
    # GEMINI_2_0_FLASH removed - deprecated by Google
    # GEMINI_2_0_FLASH_LITE removed - deprecated by Google
    Model.GEMINI_2_5_FLASH_LITE:  8192,
    Model.GEMINI_FLASH:           8192,
    Model.GEMINI_FLASH_LITE:      8192,
    Model.GEMINI_3_1_FLASH_LITE:  8192,
    
    # Mistral models
    Model.MISTRAL_NEMO:       8192,
    Model.MISTRAL_SMALL_3_1:  8192,
    Model.MINISTRAL_3B:       4096,
    Model.MINISTRAL_8B:       4096,
    
    # Cohere models
    Model.COMMAND_R7B:        4096,
    
    # Chinese models
    Model.QWEN_3_235B:        8192,
    Model.QWEN_3_CODER_30B:   8192,
    Model.QWEN_3_32B:         8192,
    Model.QWEN_LONG:          8192,  # Actually supports 10M context but output limited
    Model.GLM_4:              4096,
    Model.GLM_4_6:            4096,
    Model.GLM_4_7:            4096,
    Model.GLM_4_FLASH:        8192,
    Model.GLM_4_AIR:          4096,
    Model.GLM_5:              8192,
    Model.KIMI_K1_5:          8192,
    Model.KIMI_K2:            8192,
    Model.KIMI_K2_5:          8192,
}


def get_max_iterations(task_type: TaskType) -> int:
    if task_type == TaskType.CODE_GEN:
        return 3  # heavy — needs generate + critique + revise
    if task_type == TaskType.CODE_REVIEW:
        return 4  # extra iteration: reviews depend on context quality and
                  # often need a full second pass after critique improves framing
    if task_type == TaskType.REASONING:
        return 3
    return 2  # light


# ─────────────────────────────────────────────
# Budget partitioning (soft caps)
# ─────────────────────────────────────────────

BUDGET_PARTITIONS: dict[str, float] = {
    "decomposition": 0.05,
    "generation":    0.45,
    "cross_review":  0.25,
    "evaluation":    0.15,
    "reserve":       0.10,
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Task:
    id: str
    type: TaskType
    prompt: str
    context: str = ""
    dependencies: list[str] = field(default_factory=list)
    acceptance_threshold: float = 0.85
    max_iterations: int = 3
    max_output_tokens: int = 1500
    status: TaskStatus = TaskStatus.PENDING
    hard_validators: list[str] = field(default_factory=list)
    # App Builder fields (Improvement 8)
    target_path: str = ""     # e.g. "src/routes/auth.py"
    module_name: str = ""     # e.g. "src.routes.auth"
    tech_context: str = ""    # brief note on tech stack for this file

    def __post_init__(self):
        self.acceptance_threshold = DEFAULT_THRESHOLDS.get(
            self.type, self.acceptance_threshold
        )
        self.max_iterations = get_max_iterations(self.type)
        self.max_output_tokens = MAX_OUTPUT_TOKENS.get(
            self.type, self.max_output_tokens
        )


@dataclass
class TaskResult:
    task_id: str
    output: str
    score: float
    model_used: Model
    reviewer_model: Optional[Model] = None
    tokens_used: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0}
    )
    iterations: int = 0
    cost_usd: float = 0.0
    status: TaskStatus = TaskStatus.COMPLETED
    critique: str = ""
    deterministic_check_passed: bool = True
    degraded_fallback_count: int = 0
    attempt_history: list["AttemptRecord"] = field(default_factory=list)


@dataclass
class AttemptRecord:
    """Records one failed iteration attempt so the next retry has failure context."""
    attempt_num: int          # 1-based
    model_used: str           # Model.value — str for easy serialization
    output_snippet: str       # first 200 chars of output
    failure_reason: str       # human-readable description
    validators_failed: list[str] = field(default_factory=list)


@dataclass
class Budget:
    max_usd: float = 8.0
    max_time_seconds: float = 5400.0  # 90 min
    spent_usd: float = 0.0
    start_time: float = field(default_factory=time.time)
    phase_spent: dict[str, float] = field(default_factory=lambda: {
        "decomposition": 0.0,
        "generation": 0.0,
        "cross_review": 0.0,
        "evaluation": 0.0,
        "reserve": 0.0,
    })

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_time_seconds - self.elapsed_seconds)

    def can_afford(self, estimated_cost: float) -> bool:
        return self.remaining_usd >= estimated_cost

    def time_remaining(self) -> bool:
        return self.elapsed_seconds < self.max_time_seconds

    def phase_budget(self, phase: str) -> float:
        return self.max_usd * BUDGET_PARTITIONS.get(phase, 0.0)

    def phase_remaining(self, phase: str) -> float:
        return max(0.0, self.phase_budget(phase) - self.phase_spent.get(phase, 0.0))

    def charge(self, amount: float, phase: str = "generation"):
        self.spent_usd += amount
        if phase in self.phase_spent:
            self.phase_spent[phase] += amount

    def to_dict(self) -> dict:
        return {
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "phase_spent": {k: round(v, 4) for k, v in self.phase_spent.items()},
        }


@dataclass
class ProjectState:
    """Full serializable state for resume capability."""
    project_description: str
    success_criteria: str
    budget: Budget
    tasks: dict[str, Task] = field(default_factory=dict)
    results: dict[str, TaskResult] = field(default_factory=dict)
    api_health: dict[str, bool] = field(default_factory=dict)
    status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS
    execution_order: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# JobSpec — App Builder job specification
# ─────────────────────────────────────────────

@dataclass
class JobSpec:
    """
    App Builder job specification.

    A lightweight spec for the App Builder pipeline, separate from the
    policy-oriented JobSpec in policy.py.

    Fields
    ------
    description      : Human-readable description of the app to build.
    success_criteria : Acceptance criteria for the build.
    app_type         : Optional override for the app type (e.g. "fastapi",
                       "cli", "library").  Empty string means auto-detect.
    docker           : Whether to run Docker-based verification.
    output_dir       : Where to write the generated app.  Empty string means
                       auto-generate a temp directory.
    """
    description: str
    success_criteria: str
    app_type: str = ""
    docker: bool = False
    output_dir: str = ""


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def prompt_hash(model: str, prompt: str, max_tokens: int,
                system: str = "", temperature: float = 0.3) -> str:
    payload = f"{model}||{system}||{prompt}||{max_tokens}||{temperature}"
    return hashlib.sha256(payload.encode()).hexdigest()


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


def build_default_profiles() -> "dict[Model, ModelProfile]":
    """
    Build a ModelProfile for every Model enum value using the static
    COST_TABLE and ROUTING_TABLE as the source of truth.

    Called once at Orchestrator construction time. Telemetry fields
    (quality_score, trust_factor, avg_latency_ms, …) start at their
    defaults and are updated at runtime by TelemetryCollector.

    Lazy-imports ModelProfile from policy to avoid a circular import
    (policy.py → models.py, models.py → policy.py would be circular).
    """
    # Lazy import to avoid circular dependency: policy.py imports models.py
    from .policy import ModelProfile  # noqa: PLC0415

    # Build capability map: {TaskType → priority_rank} for each model
    # Priority rank = index in ROUTING_TABLE list (0 = highest priority)
    capability_map: dict[Model, dict[TaskType, int]] = {m: {} for m in Model}
    for task_type, model_list in ROUTING_TABLE.items():
        for rank, model in enumerate(model_list):
            capability_map[model][task_type] = rank

    profiles: dict[Model, ModelProfile] = {}
    for model in Model:
        costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})
        profiles[model] = ModelProfile(
            model=model,
            provider=get_provider(model),
            cost_per_1m_input=costs["input"],
            cost_per_1m_output=costs["output"],
            capable_task_types=capability_map.get(model, {}),
        )
    return profiles
