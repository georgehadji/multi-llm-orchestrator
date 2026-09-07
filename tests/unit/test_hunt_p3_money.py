"""P3 money-path regressions — BudgetHierarchy reservation and persistence.

Covers P3-COST3 (reservation leak on anonymous jobs), P3-COST2 (reset_spend not
surviving a restart) and P3-COST1 (remaining("job") ignoring reservations), all
found by the V4 P3 wave. See docs/hunts/p3-deep-region/inventory.md.
"""

from __future__ import annotations

import pytest

from orchestrator.cost import BudgetHierarchy


@pytest.mark.unit
class TestP3Cost3AnonymousReservationLeak:
    """P3-COST3: can_afford_job reserved unconditionally, charge_job released by job_id.

    ``policy.py::JobSpec`` declares no ``job_id``/``team``, so
    ``ProjectRunner.run_job``'s ``getattr(spec, "job_id", "")`` is always ``""``.
    ``can_afford_job`` incremented ``_reserved_usd`` regardless (cost.py:248) but
    ``charge_job`` released via ``_reservations.pop(job_id, 0.0)`` — ``pop("")``
    is a no-op — so every job permanently leaked its estimate from the org cap.
    """

    def test_settled_anonymous_job_releases_its_reservation(self) -> None:
        hier = BudgetHierarchy(org_max_usd=10.0)

        assert hier.can_afford_job("", "", 8.0) is True
        hier.charge_job("", "", 1.0)

        # Only $1 was actually spent, so $8 must fit again. Before the fix the
        # first job's $8 was still reserved and this returned False.
        assert hier.can_afford_job("", "", 8.0) is True

    def test_aborted_anonymous_job_releases_its_reservation(self) -> None:
        hier = BudgetHierarchy(org_max_usd=10.0)

        assert hier.can_afford_job("", "", 8.0) is True
        hier.release_reservation("", "")

        assert hier.can_afford_job("", "", 8.0) is True

    def test_reservation_still_blocks_a_concurrent_job(self) -> None:
        """The leak fix must not cost the TOCTOU protection the reservation exists for."""
        hier = BudgetHierarchy(org_max_usd=10.0)

        assert hier.can_afford_job("", "", 8.0) is True
        # Second job, before the first settles: $8 is genuinely committed.
        assert hier.can_afford_job("", "", 8.0) is False

    def test_repeated_cycles_do_not_accumulate(self) -> None:
        """The failure mode was monotonic: N jobs eventually refused everything."""
        hier = BudgetHierarchy(org_max_usd=10.0)

        for _ in range(20):
            assert hier.can_afford_job("", "", 8.0) is True
            hier.charge_job("", "", 0.1)

        assert hier._org_spent == pytest.approx(2.0)
        assert hier._reserved_usd == pytest.approx(0.0)

    def test_named_jobs_are_unaffected(self) -> None:
        """Regression guard for the path that already worked."""
        hier = BudgetHierarchy(org_max_usd=10.0)

        assert hier.can_afford_job("job-1", "eng", 8.0) is True
        hier.charge_job("job-1", "eng", 1.0)

        assert hier.can_afford_job("job-2", "eng", 8.0) is True
        assert hier._reserved_usd == pytest.approx(8.0)


@pytest.mark.unit
class TestP3Cost2ResetSpendPersistence:
    """P3-COST2: reset_spend() cleared memory but left the DB rows behind."""

    def test_reset_survives_a_reload(self, tmp_path) -> None:
        db = tmp_path / "budget.db"

        hier = BudgetHierarchy(org_max_usd=100.0, db_path=db)
        hier.charge_job("job-1", "eng", 25.0)
        hier.reset_spend("all")

        # A fresh instance is what a process restart looks like.
        reloaded = BudgetHierarchy(org_max_usd=100.0, db_path=db)

        assert reloaded._org_spent == pytest.approx(0.0)
        assert reloaded._team_spent == {}
        assert reloaded._job_spent == {}

    def test_single_key_reset_survives_a_reload(self, tmp_path) -> None:
        db = tmp_path / "budget.db"

        hier = BudgetHierarchy(org_max_usd=100.0, db_path=db)
        hier.charge_job("job-1", "eng", 10.0)
        hier.charge_job("job-2", "eng", 5.0)
        hier.reset_spend("job", "job-1")

        reloaded = BudgetHierarchy(org_max_usd=100.0, db_path=db)

        assert "job-1" not in reloaded._job_spent
        assert reloaded._job_spent["job-2"] == pytest.approx(5.0)


@pytest.mark.unit
class TestP3Cost1JobRemainingIgnoresReservations:
    """P3-COST1: org and team deducted reservations; job level did not."""

    def test_job_remaining_deducts_its_reservation(self) -> None:
        hier = BudgetHierarchy(org_max_usd=100.0, job_budgets={"job-1": 10.0})

        assert hier.can_afford_job("job-1", "", 4.0) is True

        # $4 of the job's $10 is committed, so $6 is genuinely available.
        assert hier.remaining("job", "job-1") == pytest.approx(6.0)
