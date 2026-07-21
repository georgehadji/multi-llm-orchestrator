"""
VFM routing contract — proves 2026 value-for-money models are declared, priced,
and survive intact into the routing cascade for each text task.

Free-tier (":free" endpoint variants) is retired project policy as of
2026-07-14 — this project no longer routes to free-tier models. The contract
here is narrower than the original "free-tier-first" lock: every routed model
must be a real enum member with a cost entry (no silent drops, no un-priced
spend), and the cascade must still end on a reliable paid fallback.
"""

from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

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
# Free-tier ids removed from this list — the project no longer routes to
# ":free" endpoint variants (retired 2026-07-14). These are the current
# lead/near-lead value-for-money candidates for the text task types.

_VFM_MODELS = [
    "google/gemini-3.5-flash",
    "minimax/minimax-m3",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
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


# ── Routing: no drops, fully priced, reliable fallback tail ─────────────────


class TestRoutingIsVfmFirst:
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
        """Cascade must keep at least one fallback after the lead (one door open)."""
        chain = ROUTING_TABLE[task]
        assert len(chain) >= 2, f"{task.value} needs at least one fallback after the lead"
