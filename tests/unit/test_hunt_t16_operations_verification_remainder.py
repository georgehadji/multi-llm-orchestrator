"""
T16 (final remainder: operations/, verification/, policy*, telemetry/logging
misc) proof-of-defect and no-regression tests.

C1 — application/cli_helpers.py::setup_logging() is the actual live CLI
     logging entry point (called from entrypoints/cli_dispatch.py on every
     invocation) but never attached generators/secrets_manager.py's
     SecretsFilter — unlike log_config.py::configure_logging() (fixed in
     hunt T8), which has zero live callers. The safety net was built and
     correctly wired to the wrong, unreachable function.
C2 — testing/validator.py::TestValidator._generate_test() called
     self.client.call_model(...), a method UnifiedClient has never had (only
     .call() exists). The AttributeError was silently caught by a broad
     except, falling back to a trivial `assert True` stub that then gets
     written to disk, run, and reported as a passing generated test.
C3 — project_mgmt/analyzer.py was a stale, unshimmed duplicate of the live
     project_analyzer.py, missing the ArchitectureScorer delegation and
     save_report()/print_suggestions() the root gained since.
C4 — four operations/ duplicate pairs (hitl_workflow.py,
     concurrency_controller.py, deployment_feedback.py, memory_tier.py) were
     unshimmed, dead-on-the-operations/-side forks of live root modules;
     two of the four (concurrency_controller, deployment_feedback) diverge
     functionally (root carries an asyncio task-reference fix and an SSRF
     guard the operations/ copies lack).
C5 — services/executor.py, services/generator.py, services/observability.py
     independently redefined the same classes services/__init__.py already
     re-exports from application/ as canonical, so 8 existing test files
     importing the submodule path directly silently exercised the shadowed
     copy instead of the one production code actually runs.
C6 — crosscutting/config.py's re-export of orchestrator/config.py's
     TIMEOUT_DEFAULT_SECONDS/TOKENS_MAX_OUTPUT/BUDGET_DEFAULT_USD always
     ImportError'd (those names never existed; real ones are namespaced
     under Timeout/TokenLimits/BudgetDefaults) and silently fell back to
     defaults including a stale $10 budget that didn't match the real $8.
C7 — operations/quick_self_test.py executed a full ad-hoc integration test
     at module import time: wrote a log file into the source tree
     (orchestrator/self_test_log.txt) and called sys.exit(1) on failure at
     module scope, which would crash any future code that imported or
     iterated the operations/ package's submodules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_setup_logging_installs_secrets_filter(capsys):
    from orchestrator.application.cli_helpers import setup_logging

    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    try:
        setup_logging(verbose=False, suppress_cache=False)
        child = logging.getLogger("orchestrator.test_c1_secrets_filter")
        child.info("using key sk-abcdefghij1234567890klmno for auth")

        captured = capsys.readouterr()
        assert (
            "sk-abcdefghij1234567890klmno" not in captured.err
        ), "raw secret leaked into log output unmasked"
        assert "REDACTED" in captured.err
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
        root_logger.setLevel(original_level)


# --- C2 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_generate_test_calls_real_client_call_method():
    from orchestrator.testing.validator import TestValidator

    mock_response = AsyncMock()
    mock_response.text = "```python\ndef test_add_basic():\n    assert add(1, 2) == 3\n```"
    mock_client = AsyncMock()
    mock_client.call = AsyncMock(return_value=mock_response)

    validator = TestValidator(client=mock_client)
    result = await validator._generate_test(
        source_code="def add(a, b):\n    return a + b",
        import_path="mypkg.mod",
        function_name="add",
        func_info={"args": ["a", "b"], "docstring": "Add two numbers."},
        fixtures="",
    )

    mock_client.call.assert_awaited_once()
    _, kwargs = mock_client.call.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 2000
    assert "def test_add_basic" in result
    assert "test_add_exists" not in result  # not the fallback stub


# --- C3 -----------------------------------------------------------------


def test_c3_project_mgmt_analyzer_is_root_project_analyzer():
    from orchestrator.project_analyzer import ProjectAnalyzer as RootAnalyzer
    from orchestrator.project_mgmt.analyzer import ProjectAnalyzer as ShimAnalyzer

    assert ShimAnalyzer is RootAnalyzer


# --- C4 -----------------------------------------------------------------


def test_c4_operations_duplicates_are_root_shims():
    from orchestrator.concurrency_controller import TaskConcurrencyGuard as RootGuard
    from orchestrator.deployment_feedback import DeploymentFeedbackLoop as RootLoop
    from orchestrator.hitl_workflow import HITLWorkflow as RootWorkflow
    from orchestrator.memory_tier import MemoryTierManager as RootMemory
    from orchestrator.operations.concurrency_controller import (
        TaskConcurrencyGuard as OpsGuard,
    )
    from orchestrator.operations.deployment_feedback import (
        DeploymentFeedbackLoop as OpsLoop,
    )
    from orchestrator.operations.hitl_workflow import HITLWorkflow as OpsWorkflow
    from orchestrator.operations.memory_tier import MemoryTierManager as OpsMemory

    assert OpsWorkflow is RootWorkflow
    assert OpsGuard is RootGuard
    assert OpsLoop is RootLoop
    assert OpsMemory is RootMemory


# --- C5 -----------------------------------------------------------------


def test_c5_services_duplicates_are_application_shims():
    from orchestrator.application.decomposer import DecomposerService
    from orchestrator.application.executor import ExecutorService as AppExecutor
    from orchestrator.application.observability import (
        ObservabilityService as AppObservability,
    )
    from orchestrator.services.executor import ExecutorService as SvcExecutor
    from orchestrator.services.generator import GeneratorService as SvcGenerator
    from orchestrator.services.observability import (
        ObservabilityService as SvcObservability,
    )

    assert SvcExecutor is AppExecutor
    assert SvcObservability is AppObservability
    assert SvcGenerator is DecomposerService


# --- C6 -----------------------------------------------------------------


def test_c6_crosscutting_config_values_match_real_config():
    from orchestrator.config import BudgetDefaults, Timeout, TokenLimits
    from orchestrator.crosscutting.config import (
        DEFAULT_BUDGET_USD,
        MAX_TOKENS_OUTPUT,
        TIMEOUT_SECONDS,
    )

    assert TIMEOUT_SECONDS == Timeout.API_CALL_LONG
    assert MAX_TOKENS_OUTPUT == TokenLimits.CODE_STANDARD
    assert DEFAULT_BUDGET_USD == BudgetDefaults.MAX_USD_DEFAULT == 8.0


# --- C7 -----------------------------------------------------------------


def test_c7_quick_self_test_import_has_no_side_effects():
    import orchestrator.operations.quick_self_test as qst

    source_tree_log = Path(
        os.path.join(os.path.dirname(os.path.dirname(qst.__file__)), "self_test_log.txt")
    )
    assert not source_tree_log.exists(), "importing must not write into the source tree"
    assert callable(qst.main)
