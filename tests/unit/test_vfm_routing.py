"""
VFM routing contract — proves 2026 value-for-money models are declared, priced,
and lead the routing cascade for each text task.

Goal lock: every text task should try a free / ultra-cheap capable model FIRST
and only escalate to premium on failure. Each routed model must be a real enum
member with a cost entry (no silent drops, no un-priced spend).
"""
from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

from orchestrator.models import Model, ROUTING_TABLE, TaskType


_CONFIG = Path(__file__).resolve().parents[2] / "orchestrator" / "config"
_TEXT_TASKS = [
    TaskType.CODE_GEN,
    TaskType.CODE_REVIEW,
    TaskType.REASONING,
    TaskType.WRITING,
    TaskType.DATA_EXTRACT,
    TaskType.SUMMARIZE,
    TaskType.EVALUATE,
]


def _costs() -> dict:
    return json.load(open(_CONFIG / "costs.json", encoding="utf-8"))


def _routing() -> dict:
    return json.load(open(_CONFIG / "routing.json", encoding="utf-8"))


# ── New 2026 VFM models are declared & usable ────────────────────────────────

_VFM_MODELS = [
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "z-ai/glm-4.7-flash",
    "minimax/minimax-m2.5",
    "openai/gpt-5.2",
]


class TestVfmModelsDeclared:
    @pytest.mark.parametrize("model_id", _VFM_MODELS)
    def test_model_is_enum_member(self, model_id):
        assert model_id in Model._value2member_map_, f"{model_id} not declared in Model enum"

    @pytest.mark.parametrize("model_id", _VFM_MODELS)
    def test_model_has_cost_entry(self, model_id):
        assert model_id in _costs(), f"{model_id} missing from costs.json"

    def test_free_models_cost_zero(self):
        costs = _costs()
        for mid in _VFM_MODELS:
            if mid.endswith(":free"):
                c = costs[mid]
                assert c["input"] == 0.0 and c["output"] == 0.0, f"{mid} free must cost 0"


# ── Routing: VFM-first, no drops, fully priced ───────────────────────────────

class TestRoutingIsVfmFirst:
    @pytest.mark.parametrize("task", _TEXT_TASKS)
    def test_lead_candidate_is_free_tier(self, task):
        """Each text task must try a $0 free-tier model first (max VFM)."""
        lead = ROUTING_TABLE[task][0]
        assert lead.value.endswith(":free"), (
            f"{task.value} should lead with a free model, got {lead.value}"
        )

    @pytest.mark.parametrize("task", _TEXT_TASKS)
    def test_no_silent_drops(self, task):
        """Every routing.json ref must survive into the built ROUTING_TABLE."""
        raw = _routing()[task.value]
        built = [m.value for m in ROUTING_TABLE[task]]
        for ref in raw:
            assert ref in built, f"{task.value}: {ref} silently dropped (not in Model enum)"

    @pytest.mark.parametrize("task", _TEXT_TASKS)
    def test_every_routed_model_is_priced(self, task):
        """No routed model may be un-priced (would mean untracked spend)."""
        costs = _costs()
        for model in ROUTING_TABLE[task]:
            assert model.value in costs, f"{task.value}: {model.value} has no cost entry"

    @pytest.mark.parametrize("task", _TEXT_TASKS)
    def test_has_premium_fallback(self, task):
        """Cascade must keep a paid fallback after the free lead (one door open)."""
        chain = ROUTING_TABLE[task]
        assert len(chain) >= 2, f"{task.value} needs at least one fallback after the free lead"
        assert not chain[-1].value.endswith(":free"), (
            f"{task.value} last-resort model must be a reliable paid model, not free"
        )
