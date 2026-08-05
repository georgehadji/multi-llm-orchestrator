"""
Tests for F-8: testing config single source of truth.
======================================================
The testing limits come from config/limits.json via testing_config.py;
env vars are operational overrides only.
"""

from __future__ import annotations

import pytest

from orchestrator.infrastructure.testing_config import load_testing_config


@pytest.mark.unit
class TestTestingConfig:
    def test_defaults_from_limits_json(self) -> None:
        cfg = load_testing_config()
        assert cfg.max_repair_iterations == 5
        assert cfg.suite_timeout_s == 120.0
        assert cfg.mutation_sample_size == 12
        assert cfg.mutation_timeout_s == 60.0

    def test_env_override_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_SUITE_TIMEOUT_S", "30")
        monkeypatch.setenv("ORCH_MUTATION_SAMPLE_SIZE", "4")
        cfg = load_testing_config()
        assert cfg.suite_timeout_s == 30.0
        assert cfg.mutation_sample_size == 4

    def test_invalid_env_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_SUITE_TIMEOUT_S", "not-a-number")
        cfg = load_testing_config()
        assert cfg.suite_timeout_s == 120.0
