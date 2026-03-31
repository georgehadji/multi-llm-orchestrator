"""
Unit tests for orchestrator/models.py (and orchestrator/budget.py).
Replaces the old standalone verification script with proper pytest tests.
"""
import asyncio
import pytest

from orchestrator.models import (
    Model,
    TaskType,
    COST_TABLE,
    ROUTING_TABLE,
    get_provider,
    estimate_cost,
)
from orchestrator.budget import Budget


# ─────────────────────────────────────────────────────────────────
# TestModelEnum
# ─────────────────────────────────────────────────────────────────

class TestModelEnum:
    def test_no_duplicate_values(self):
        """Each Model entry must have a unique string value (no accidental aliases)."""
        values = [m.value for m in Model]
        # Python Enum silently creates aliases for duplicate values;
        # list(Model) deduplicates them, so check the raw _value2member_map_ size.
        assert len(values) == len(set(values)), (
            "Duplicate Model enum values detected — check for accidental aliases"
        )

    def test_tasktype_no_duplicate_values(self):
        values = [t.value for t in TaskType]
        assert len(values) == len(set(values))

    def test_known_models_present(self):
        assert Model.GPT_4O.value == "gpt-4o"
        assert Model.GPT_4O_MINI.value == "gpt-4o-mini"
        assert Model.O3_MINI.value == "o3-mini"

    def test_known_task_types_present(self):
        assert TaskType.CODE_GEN.value == "code_generation"
        assert TaskType.EVALUATE.value == "evaluation"


# ─────────────────────────────────────────────────────────────────
# TestCostTable
# ─────────────────────────────────────────────────────────────────

class TestCostTable:
    def test_all_models_have_cost_entry(self):
        missing = [m for m in Model if m not in COST_TABLE]
        assert not missing, f"Models missing from COST_TABLE: {missing}"

    def test_no_negative_costs(self):
        for model, costs in COST_TABLE.items():
            assert costs["input"] >= 0, f"{model} has negative input cost"
            assert costs["output"] >= 0, f"{model} has negative output cost"

    def test_cost_entries_have_input_output_keys(self):
        for model, costs in COST_TABLE.items():
            assert "input" in costs, f"{model} missing 'input' key"
            assert "output" in costs, f"{model} missing 'output' key"


# ─────────────────────────────────────────────────────────────────
# TestGetProvider
# ─────────────────────────────────────────────────────────────────

class TestGetProvider:
    def test_returns_string_for_all_models(self):
        for m in Model:
            result = get_provider(m)
            assert isinstance(result, str) and result, f"{m} returned empty provider"

    def test_no_unknown_provider(self):
        for m in Model:
            assert get_provider(m) != "unknown", f"{m} returned 'unknown' provider"


# ─────────────────────────────────────────────────────────────────
# TestEstimateCost
# ─────────────────────────────────────────────────────────────────

class TestEstimateCost:
    def test_zero_tokens_returns_zero_cost(self):
        cost = estimate_cost(Model.GPT_4O_MINI, 0, 0)
        assert cost == 0.0

    def test_cost_scales_linearly(self):
        cost1 = estimate_cost(Model.GPT_4O_MINI, 1000, 0)
        cost2 = estimate_cost(Model.GPT_4O_MINI, 2000, 0)
        assert cost2 == pytest.approx(cost1 * 2, rel=1e-6)

    def test_positive_tokens_returns_positive_cost(self):
        cost = estimate_cost(Model.GPT_4O, 100, 100)
        assert cost > 0.0


# ─────────────────────────────────────────────────────────────────
# TestRoutingTable
# ─────────────────────────────────────────────────────────────────

class TestRoutingTable:
    def test_all_task_types_covered(self):
        for task_type in TaskType:
            assert task_type in ROUTING_TABLE, f"{task_type} missing from ROUTING_TABLE"

    def test_all_referenced_models_exist(self):
        valid_models = set(Model)
        for task_type, models in ROUTING_TABLE.items():
            for m in models:
                assert m in valid_models, (
                    f"ROUTING_TABLE[{task_type}] references unknown model {m!r}"
                )

    def test_each_task_type_has_at_least_one_model(self):
        for task_type, models in ROUTING_TABLE.items():
            assert len(models) >= 1, f"{task_type} has no models in ROUTING_TABLE"


# ─────────────────────────────────────────────────────────────────
# TestBudgetArithmetic
# ─────────────────────────────────────────────────────────────────

class TestBudgetArithmetic:
    def test_initial_remaining_equals_max(self):
        b = Budget(max_usd=10.0)
        assert b.remaining_usd == pytest.approx(10.0)

    def test_charge_reduces_remaining(self):
        b = Budget(max_usd=10.0)
        asyncio.get_event_loop().run_until_complete(b.charge(2.5, "generation"))
        assert b.spent_usd == pytest.approx(2.5)
        assert b.remaining_usd == pytest.approx(7.5)

    def test_reserve_reduces_remaining(self):
        b = Budget(max_usd=10.0)
        result = asyncio.get_event_loop().run_until_complete(b.reserve(3.0))
        assert result is True
        assert b.remaining_usd == pytest.approx(7.0)
        assert b._reserved_usd == pytest.approx(3.0)

    def test_reserve_fails_when_insufficient(self):
        b = Budget(max_usd=5.0)
        result = asyncio.get_event_loop().run_until_complete(b.reserve(6.0))
        assert result is False
        assert b._reserved_usd == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_commit_reservation(self):
        b = Budget(max_usd=10.0)
        await b.reserve(5.0)
        await b.commit_reservation(5.0, 4.0, "generation")
        assert b._reserved_usd == pytest.approx(0.0)
        assert b.spent_usd == pytest.approx(4.0)
        assert b.remaining_usd == pytest.approx(6.0)

    @pytest.mark.asyncio
    async def test_release_reservation(self):
        b = Budget(max_usd=10.0)
        await b.reserve(3.0)
        await b.release_reservation(3.0)
        assert b._reserved_usd == pytest.approx(0.0)
        assert b.remaining_usd == pytest.approx(10.0)

    def test_can_afford(self):
        b = Budget(max_usd=10.0)
        assert b.can_afford(9.99) is True
        assert b.can_afford(10.01) is False

    def test_reserve_negative_raises(self):
        b = Budget(max_usd=10.0)
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(b.reserve(-1.0))
