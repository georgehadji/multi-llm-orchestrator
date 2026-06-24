"""
Test that models.py does not perform I/O at import time.

Rule #2: Domain models must not execute I/O at import time.
This test verifies that COST_TABLE, ROUTING_TABLE, FALLBACK_CHAIN, etc.
are loaded lazily via __getattr__, not at module import time.
"""

import sys
import pytest


def test_tables_not_loaded_at_import():
    """Verify config tables are not loaded when models module is imported."""
    # Clear any cached orchestrator modules
    for mod_name in list(sys.modules.keys()):
        if "orchestrator" in mod_name:
            del sys.modules[mod_name]

    # Import models fresh
    from orchestrator import models

    # Tables should NOT be cached after import
    assert "COST_TABLE" not in models.__dict__, "COST_TABLE loaded at import time"
    assert "ROUTING_TABLE" not in models.__dict__, "ROUTING_TABLE loaded at import time"
    assert "FALLBACK_CHAIN" not in models.__dict__, "FALLBACK_CHAIN loaded at import time"
    assert "DEFAULT_THRESHOLDS" not in models.__dict__, "DEFAULT_THRESHOLDS loaded at import time"
    assert "MAX_OUTPUT_TOKENS" not in models.__dict__, "MAX_OUTPUT_TOKENS loaded at import time"


def test_tables_load_on_first_access():
    """Verify tables are loaded on first attribute access."""
    # Clear any cached orchestrator modules
    for mod_name in list(sys.modules.keys()):
        if "orchestrator" in mod_name:
            del sys.modules[mod_name]

    from orchestrator import models

    # Access COST_TABLE - should trigger lazy load
    cost_table = models.COST_TABLE
    assert "COST_TABLE" in models.__dict__
    assert isinstance(cost_table, dict)
    assert len(cost_table) > 0

    # Other tables should still not be loaded
    assert "ROUTING_TABLE" not in models.__dict__

    # Access ROUTING_TABLE
    routing_table = models.ROUTING_TABLE
    assert "ROUTING_TABLE" in models.__dict__
    assert isinstance(routing_table, dict)


def test_orchestrator_package_import_no_io():
    """Verify importing the main orchestrator package doesn't trigger table loading."""
    # Clear any cached orchestrator modules
    for mod_name in list(sys.modules.keys()):
        if "orchestrator" in mod_name:
            del sys.modules[mod_name]

    # Import main package
    from orchestrator import Orchestrator, Budget, Model, TaskType

    # Tables should NOT be loaded
    from orchestrator import models

    assert "COST_TABLE" not in models.__dict__
    assert "ROUTING_TABLE" not in models.__dict__
    assert "FALLBACK_CHAIN" not in models.__dict__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
