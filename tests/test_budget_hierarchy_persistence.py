"""
Test BudgetHierarchy SQLite persistence.

Verifies that budget state is persisted across restarts.
"""
import tempfile
from pathlib import Path

import pytest

from orchestrator.cost import BudgetHierarchy


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "budget.db"


def test_budget_hierarchy_creates_db(temp_db_path: Path) -> None:
    """BudgetHierarchy creates SQLite DB on init."""
    bh = BudgetHierarchy(
        org_max_usd=100.0,
        db_path=temp_db_path,
    )
    
    assert temp_db_path.exists()


def test_budget_hierarchy_persists_spending(temp_db_path: Path) -> None:
    """Spending is persisted and loaded on restart."""
    # Create hierarchy and spend some budget
    bh1 = BudgetHierarchy(
        org_max_usd=100.0,
        db_path=temp_db_path,
    )
    
    # Simulate spending
    bh1._org_spent = 50.0
    bh1._team_spent["team_a"] = 30.0
    bh1._job_spent["job_123"] = 20.0
    bh1._save_to_db()
    
    # Create new instance with same DB
    bh2 = BudgetHierarchy(
        org_max_usd=100.0,
        db_path=temp_db_path,
    )
    
    # Verify spending was loaded
    assert bh2._org_spent == 50.0
    assert bh2._team_spent["team_a"] == 30.0
    assert bh2._job_spent["job_123"] == 20.0


def test_budget_hierarchy_in_memory_no_persistence() -> None:
    """Without db_path, BudgetHierarchy is in-memory only."""
    bh = BudgetHierarchy(org_max_usd=100.0)
    
    bh._org_spent = 50.0
    
    # No DB path means no persistence
    assert bh._db_path is None


def test_budget_hierarchy_charge_job_saves_to_db(temp_db_path: Path) -> None:
    """charge_job() method persists to DB."""
    bh = BudgetHierarchy(
        org_max_usd=100.0,
        db_path=temp_db_path,
    )
    
    # Charge a job
    bh.charge_job("job_123", "team_a", 25.0)
    
    # Create new instance
    bh2 = BudgetHierarchy(
        org_max_usd=100.0,
        db_path=temp_db_path,
    )
    
    # Verify charge was persisted
    assert bh2._org_spent == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
