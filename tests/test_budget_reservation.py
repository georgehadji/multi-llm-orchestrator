"""
Tests for BudgetHierarchy pessimistic-reservation pattern (Scenario 1 fix).

The TOCTOU gap: can_afford_job() checks a snapshot, but concurrent run_job()
calls all see the same snapshot before any charge lands. Pessimistic reservation
atomically claims the budget before execution, making concurrent over-spend
impossible without coordination.
"""

import asyncio
import pytest
from orchestrator.cost import BudgetHierarchy


# ── Reservation basics ────────────────────────────────────────────────────────

class TestReservationBasics:

    def test_can_afford_job_reserves_estimated_cost(self):
        """After can_afford_job returns True, the org remaining drops by the estimate."""
        h = BudgetHierarchy(org_max_usd=100.0)
        result = h.can_afford_job("job1", "team_a", 40.0)
        assert result is True
        assert h.remaining("org") == 60.0  # 100 - 40 reserved

    def test_second_job_sees_reduced_budget_due_to_reservation(self):
        """The critical fix: a second concurrent can_afford sees the reserved amount."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "team_a", 70.0)   # reserves 70
        result = h.can_afford_job("job2", "team_b", 40.0)  # 70 + 40 > 100
        assert result is False  # must block, not pass

    def test_two_concurrent_jobs_cannot_both_exceed_org_cap(self):
        """Scenario: two jobs, each 60 usd, on a 100 usd org cap. Only one fits."""
        h = BudgetHierarchy(org_max_usd=100.0)
        first  = h.can_afford_job("job1", "t", 60.0)
        second = h.can_afford_job("job2", "t", 60.0)
        assert first is True
        assert second is False   # reservation from job1 blocks job2

    def test_charge_job_settles_reservation(self):
        """After charge_job, remaining reflects actual spend, not estimate."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "team_a", 50.0)   # reserve 50
        h.charge_job("job1", "team_a", 30.0)        # actual was 30
        # remaining should be 100 - 30 = 70, not 100 - 50 = 50
        assert h.remaining("org") == pytest.approx(70.0)

    def test_charge_job_releases_excess_reservation(self):
        """If actual < estimate, the excess reservation is freed."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "t", 50.0)
        h.charge_job("job1", "t", 20.0)
        # The freed 30 should allow a new job
        result = h.can_afford_job("job2", "t", 75.0)  # 20 + 75 = 95 ≤ 100
        assert result is True

    def test_release_reservation_without_charge(self):
        """On abort/error, release_reservation() frees the hold without charging."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "t", 80.0)
        h.release_reservation("job1")
        result = h.can_afford_job("job2", "t", 80.0)  # full budget restored
        assert result is True

    def test_release_reservation_idempotent(self):
        """Double-releasing a reservation does not corrupt the counter."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "t", 50.0)
        h.release_reservation("job1")
        h.release_reservation("job1")  # no-op, should not raise or go negative
        assert h.remaining("org") == pytest.approx(100.0)

    def test_reserved_usd_never_goes_negative(self):
        """Guard: _reserved_usd cannot go below 0 even with mismatched releases."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.release_reservation("nonexistent")  # no prior reservation
        assert h.remaining("org") == pytest.approx(100.0)


# ── Team-level reservation ────────────────────────────────────────────────────

class TestTeamReservation:

    def test_team_reservation_blocks_concurrent_team_job(self):
        """Reservations are checked per team cap as well as org cap."""
        h = BudgetHierarchy(org_max_usd=200.0, team_budgets={"eng": 80.0})
        h.can_afford_job("job1", "eng", 60.0)   # reserves 60 of 80 team budget
        result = h.can_afford_job("job2", "eng", 30.0)  # 60 + 30 > 80
        assert result is False

    def test_charge_job_releases_team_reservation(self):
        """Settling a team job frees the team reservation."""
        h = BudgetHierarchy(org_max_usd=200.0, team_budgets={"eng": 80.0})
        h.can_afford_job("job1", "eng", 50.0)
        h.charge_job("job1", "eng", 40.0)
        result = h.can_afford_job("job2", "eng", 35.0)  # 40 + 35 = 75 ≤ 80
        assert result is True


# ── Concurrent simulation ─────────────────────────────────────────────────────

class TestConcurrentSimulation:

    @pytest.mark.asyncio
    async def test_concurrent_jobs_cannot_collectively_exceed_org_cap(self):
        """
        Simulate N concurrent run_job()-style calls: each checks can_afford,
        then awaits (simulating real work), then charges. Total must stay ≤ cap.
        """
        h = BudgetHierarchy(org_max_usd=100.0)
        approved_jobs = []

        async def fake_run_job(job_id: str, estimate: float, actual: float):
            if h.can_afford_job(job_id, "t", estimate):
                approved_jobs.append(job_id)
                await asyncio.sleep(0)   # yield — simulates real concurrent work
                h.charge_job(job_id, "t", actual)

        # 5 jobs of 30 usd each against a 100 usd cap
        await asyncio.gather(*[
            fake_run_job(f"job{i}", 30.0, 28.0) for i in range(5)
        ])

        total_charged = sum(28.0 for _ in approved_jobs)
        assert total_charged <= 100.0, (
            f"Org cap 100 exceeded: {total_charged} charged "
            f"({len(approved_jobs)} jobs approved)"
        )
        # At most 3 jobs fit: 3 × 30 = 90 ≤ 100; 4 × 30 = 120 > 100
        assert len(approved_jobs) <= 3

    @pytest.mark.asyncio
    async def test_to_dict_reflects_reserved_and_spent_correctly(self):
        """to_dict should report both spent and reserved in a useful way."""
        h = BudgetHierarchy(org_max_usd=100.0)
        h.can_afford_job("job1", "t", 40.0)   # reserve 40
        h.charge_job("job1", "t", 35.0)        # settle at 35
        d = h.to_dict()
        assert d["org"]["spent"] == pytest.approx(35.0)
        assert d["org"]["max"] == pytest.approx(100.0)
