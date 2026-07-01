"""
Tests for orchestrator/models.py — Budget, Task, Model enums, routing tables.
"""

from __future__ import annotations

import asyncio
import pytest

from orchestrator.models import (
    Budget,
    Model,
    Task,
    TaskType,
    TaskStatus,
    ROUTING_TABLE,
    COST_TABLE,
    estimate_cost,
)
from orchestrator.application.model_profile_builder import build_default_profiles

# ═══════════════════════════════════════════════════════════════════════════
# Budget Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBudget:
    """Unit tests for Budget class."""

    # ── Construction ──
    def test_default_budget(self):
        """Budget with defaults has $10 and 10800s."""
        b = Budget()
        assert b.max_usd >= 5.0
        assert b.max_time_seconds >= 1000
        assert b.spent_usd == 0.0

    @pytest.mark.parametrize(
        "max_usd,timeout",
        [
            (5.0, 300),
            (100.0, 3600),
            (0.01, 60),
        ],
    )
    def test_budget_parametrized(self, max_usd, timeout):
        """Budget accepts various limits."""
        b = Budget(max_usd=max_usd, max_time_seconds=timeout)
        assert b.max_usd == max_usd
        assert b.max_time_seconds == timeout

    # ── Thread Safety ──
    @pytest.mark.asyncio
    async def test_charge_is_thread_safe(self):
        """Concurrent charges must not exceed budget."""
        b = Budget(max_usd=10.0)

        async def charge_five():
            for _ in range(5):
                await b.charge(1.0, "generation")
                await asyncio.sleep(0.001)

        await asyncio.gather(charge_five(), charge_five())
        assert b.spent_usd == 10.0

    @pytest.mark.asyncio
    async def test_reserve_is_atomic(self, small_budget):
        """Reserve must be atomic under concurrent access."""

        async def try_reserve(amount):
            return await small_budget.reserve(amount)

        results = await asyncio.gather(try_reserve(3.0), try_reserve(3.0))
        assert sum(results) == 1
        assert small_budget._reserved_usd == 3.0

    # ── Lock Eager Initialization ──
    def test_lock_is_eagerly_initialized(self):
        """Budget._lock must not be None after construction (BUG-001 fix)."""
        b = Budget()
        assert b._lock is not None
        assert isinstance(b._lock, asyncio.Lock)

    # ── Properties ──
    def test_remaining_usd_excludes_reserved(self, small_budget):
        """Remaining USD must exclude both spent and reserved."""
        assert small_budget.remaining_usd == 5.0
        asyncio.run(small_budget.charge(1.0, "generation"))
        assert small_budget.remaining_usd == 4.0

    def test_can_afford(self):
        """can_afford must return correct boolean."""
        b = Budget(max_usd=10.0)
        assert b.can_afford(5.0)
        assert not b.can_afford(15.0)

    def test_phase_budget(self):
        """Phase budget must be proportional to max_usd."""
        b = Budget(max_usd=100.0)
        gen = b.phase_budget("generation")
        assert gen > 0

    # ── Edge Cases ──
    def test_charge_zero(self, small_budget):
        """Charging zero should not change spent_usd."""
        import asyncio

        asyncio.run(small_budget.charge(0.0, "generation"))
        assert small_budget.spent_usd == 0.0

    @pytest.mark.asyncio
    async def test_reserve_negative_raises(self, small_budget):
        """Reserving negative amount must raise ValueError."""
        with pytest.raises(ValueError):
            await small_budget.reserve(-1.0)

    @pytest.mark.asyncio
    async def test_reserve_when_exhausted(self):
        """Reserve must fail when budget is exhausted."""
        b = Budget(max_usd=1.0)
        assert await b.reserve(0.5)
        assert await b.reserve(0.5)
        assert not await b.reserve(0.5)

    # ── Reservation Lifecycle ──
    @pytest.mark.asyncio
    async def test_reserve_commit_release(self):
        """Reserve -> commit -> release lifecycle must work correctly."""
        b = Budget(max_usd=10.0)
        assert await b.reserve(3.0)
        await b.commit_reservation(3.0, 2.5, "generation")
        assert b.spent_usd == 2.5
        assert b._reserved_usd == 0.0

        assert await b.reserve(2.0)
        await b.release_reservation(2.0)
        assert b._reserved_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Task Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTask:
    """Unit tests for Task dataclass."""

    def test_task_defaults(self):
        """Task must have sensible defaults."""
        t = Task(id="test", type=TaskType.CODE_GEN, prompt="write code")
        assert t.status == TaskStatus.PENDING
        assert t.target_path == ""
        assert t.module_name == ""
        assert t.dependencies == []

    def test_task_with_dependencies(self):
        """Task must accept dependency list."""
        t = Task(id="t2", type=TaskType.CODE_GEN, prompt="p", dependencies=["t1", "t3"])
        assert len(t.dependencies) == 2
        assert "t1" in t.dependencies

    @pytest.mark.parametrize("task_type", list(TaskType))
    def test_all_task_types(self, task_type):
        """All TaskType values must be valid."""
        t = Task(id="test", type=task_type, prompt="test")
        assert t.type == task_type


# ═══════════════════════════════════════════════════════════════════════════
# Model & Routing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelRouting:
    """Tests for model enums and routing tables."""

    def test_routing_table_has_all_task_types(self):
        """ROUTING_TABLE must have entries for all TaskTypes."""
        for task_type in TaskType:
            assert task_type in ROUTING_TABLE, f"Missing: {task_type}"

    def test_cost_table_has_all_models(self):
        """COST_TABLE must have entries for all Models."""
        for model in Model:
            if model is not None:
                assert model in COST_TABLE, f"Missing cost for: {model}"

    def test_build_default_profiles_returns_all_models(self):
        """Default profiles must cover all models."""
        profiles = build_default_profiles()
        for model in Model:
            if model is not None:
                assert model in profiles

    def test_estimate_cost_returns_positive(self):
        """Estimate cost must return positive value."""
        cost = estimate_cost(Model.GPT_4O, 100, 50)
        assert cost > 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetIntegration:
    """Integration tests for Budget with Task workflows."""

    @pytest.mark.asyncio
    async def test_budget_with_task_workflow(self, small_budget, code_task):
        """Budget must track costs across a task workflow."""
        await small_budget.charge(0.01, "generation")
        assert small_budget.spent_usd == 0.01
        assert small_budget.can_afford(1.0)

        await small_budget.charge(0.02, "evaluation")
        assert small_budget.spent_usd == 0.03
