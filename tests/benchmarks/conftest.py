"""
Pytest configuration for benchmarks.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")


@pytest.fixture(scope="session")
def benchmark_context():
    """Shared context for benchmarks."""
    return {
        "iterations": 100,
        "warmup_iterations": 10,
    }
