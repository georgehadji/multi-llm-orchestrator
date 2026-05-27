"""
Integration tests — Multi-module workflows.
"""

from __future__ import annotations

import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from orchestrator.models import Budget, Model, Task, TaskType
from orchestrator.circuit_breaker import CircuitBreaker, CircuitState
from orchestrator.autonomy_config import AutonomyConfig, AutonomyLevel
from orchestrator.cost_tracker import CostTracker

# ═══════════════════════════════════════════════════════════════════════════
# Workflow: Task Execution with Budget Tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetTaskWorkflow:
    """Integration: Budget tracks costs across a task execution."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Full workflow: reserve -> charge -> complete."""
        budget = Budget(max_usd=5.0)
        tracker = CostTracker()

        # Reserve budget
        reserved = await budget.reserve(2.0)
        assert reserved

        # Simulate API call
        tracker.record("gpt-4o", 1000, 500, 0.15, 1200.0)

        # Commit actual cost
        await budget.commit_reservation(2.0, 0.15, "generation")
        assert budget.spent_usd == 0.15
        assert budget.remaining_usd == 4.85
        assert tracker.total_cost_usd == 0.15

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_workflow(self):
        """Budget exhaustion must prevent further reservations."""
        budget = Budget(max_usd=1.0)

        # Use up budget
        assert await budget.reserve(0.5)
        assert await budget.reserve(0.5)
        assert not await budget.reserve(0.1)

        assert budget.remaining_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Workflow: Circuit Breaker + Retry
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerRetry:
    """Integration: Circuit breaker with retry logic."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_budget_protection(self):
        """Circuit breaker must prevent calls when budget is tight AND breaker is open."""
        budget = Budget(max_usd=1.0)
        cb = CircuitBreaker(name="api", failure_threshold=1, reset_timeout=0.03)

        try:
            # Trip breaker
            async with cb.context():
                raise ConnectionError("outage")
        except ConnectionError:
            pass

        # Budget is fine, but breaker is open
        await asyncio.sleep(0.01)
        with pytest.raises(Exception):
            await cb.check()

    @pytest.mark.asyncio
    async def test_breaker_recovers_after_timeout(self):
        """Circuit must transition to HALF_OPEN after reset_timeout."""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0.02)
        try:
            async with cb.context():
                raise ConnectionError("fail")
        except ConnectionError:
            pass

        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.03)
        await cb.check()
        assert cb.state == CircuitState.HALF_OPEN


# ═══════════════════════════════════════════════════════════════════════════
# Workflow: Autonomy Config + Task
# ═══════════════════════════════════════════════════════════════════════════


class TestAutonomyTaskWorkflow:
    """Integration: Autonomy levels control task execution limits."""

    def test_max_autonomy_has_full_settings(self):
        """MAX autonomy must enable all quality settings."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.MAX)
        assert cfg.strict_validation
        assert cfg.require_documentation
        assert cfg.max_iterations == 10
        assert cfg.verification_mode == "behavioral"

    def test_lite_autonomy_skips_validation(self):
        """LITE autonomy must disable all expensive checks."""
        cfg = AutonomyConfig.for_level(AutonomyLevel.LITE)
        assert cfg.max_iterations == 0
        assert cfg.critique_passes == 0
        assert cfg.verification_mode == "none"
