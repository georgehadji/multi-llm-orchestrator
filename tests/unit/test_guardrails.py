"""Guardrails — the fail-open defects found before wiring it (P3-GUARD0..4).

safety/guardrails.py was imported by nothing at all, so none of its five stated
guarantees had any effect on any run. Wiring it unchanged would have been worse
than leaving it dead: three of its checks answered a safety question with the
permissive value when they had not actually run.
"""

from __future__ import annotations

import pytest

from orchestrator.safety.guardrails import (
    GuardrailConfig,
    GuardrailStatus,
    ProductionGuardrails,
    reset_guardrails,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_guardrails()
    yield
    reset_guardrails()


@pytest.fixture
def config(tmp_path) -> GuardrailConfig:
    return GuardrailConfig(
        kill_switch_file=str(tmp_path / "kill"),
        force_kill_file=str(tmp_path / "force_kill"),
    )


@pytest.mark.unit
class TestKillSwitch:
    def test_latches_once_activated(self, config) -> None:
        """P3-GUARD2: the 5s throttle returned False -- "keep running" -- for any
        call inside the window, so a caller re-checking during shutdown saw the
        switch flip back off. The sibling KillSwitch class has always latched."""
        g = ProductionGuardrails(config)
        g.activate_kill_switch()

        assert g.check_kill_switch() is True
        # Immediately again: inside the throttle window, and previously False.
        assert g.check_kill_switch() is True
        assert g.check_kill_switch() is True

    def test_deactivate_clears_the_latch(self, config) -> None:
        g = ProductionGuardrails(config)
        g.activate_kill_switch()
        assert g.check_kill_switch() is True

        g.deactivate_kill_switch()
        g._kill_switch_checked = 0.0  # bypass the poll throttle
        assert g.check_kill_switch() is False

    def test_inactive_by_default(self, config) -> None:
        assert ProductionGuardrails(config).check_kill_switch() is False

    def test_default_paths_are_not_world_writable(self) -> None:
        """P3-GUARD3: defaults were /tmp/orchestrator_kill and its force twin, so
        any local user could halt the orchestrator -- and KillSwitch's force path
        answers with os._exit(1)."""
        cfg = GuardrailConfig()
        assert not cfg.kill_switch_file.startswith("/tmp/")
        assert not cfg.force_kill_file.startswith("/tmp/")


@pytest.mark.unit
class TestUnrunChecksAreNotPasses:
    def test_status_records_whether_it_ran(self) -> None:
        assert GuardrailStatus("x", True, 0, 0, "").checked is True

    def test_throttled_memory_check_is_marked_unchecked(self, config) -> None:
        """P3-GUARD1: a throttled check returned passed=True, and
        all_checks_pass() counted it as a passing check."""
        g = ProductionGuardrails(config)
        g.check_memory()  # first call consumes the 10s window
        second = g.check_memory()

        assert second.checked is False

    def test_unchecked_results_do_not_count_as_failures_either(self, config) -> None:
        g = ProductionGuardrails(config)
        g.check_memory()
        assert g.all_checks_pass(spent=1.0, max_budget=100.0) is True


@pytest.mark.unit
class TestBudgetIsDetectionNotPrevention:
    def test_overrun_is_reported_only_after_it_happens(self, config) -> None:
        """P3-GUARD4: the docstring claimed "budget never exceeded (hard limit)".
        It reports failure only once spent already exceeds the cap."""
        g = ProductionGuardrails(config)

        assert g.check_budget(spent=99.0, max_budget=100.0).passed is True
        assert g.check_budget(spent=101.0, max_budget=100.0).passed is False

    def test_all_checks_fail_when_budget_is_blown(self, config) -> None:
        g = ProductionGuardrails(config)
        assert g.all_checks_pass(spent=101.0, max_budget=100.0) is False

    def test_kill_switch_blocks_all_checks(self, config) -> None:
        g = ProductionGuardrails(config)
        g.activate_kill_switch()
        assert g.all_checks_pass(spent=0.0, max_budget=100.0) is False
