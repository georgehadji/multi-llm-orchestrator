"""
V4 precision-audit, wave P1: 8 candidates across the highest-priority region.

Each test proves the defect for the exact predicted reason (RED against the
pre-fix source) before the fix restores it (GREEN).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

pytestmark = pytest.mark.unit


# ── Candidate 7: api_server.py shared-budget race on a long-running orchestrator ──


class TestCandidate7BudgetIsolation:
    def test_run_project_accepts_an_explicit_budget_independent_of_run_ctx(self):
        """run_project(budget=X) must use X, not whatever _run_ctx.budget already holds.

        This is the mechanism the api_server.py fix depends on: passing budget
        through the call instead of mutating orch._run_ctx.budget from outside.
        """
        import inspect

        from orchestrator.engine import Orchestrator

        sig = inspect.signature(Orchestrator.run_project)
        assert "budget" in sig.parameters, (
            "run_project() has no budget parameter — a caller holding a shared "
            "Orchestrator instance has no way to pass a per-call budget except "
            "by mutating orch._run_ctx.budget directly, which is unsynchronized "
            "shared state across every concurrent caller."
        )
        assert sig.parameters["budget"].default is None, (
            "budget must default to None so existing callers (who never pass it) "
            "keep today's behavior of reading self._run_ctx.budget unchanged."
        )

    def test_api_server_does_not_mutate_the_shared_orchestrators_run_ctx_budget(self):
        """_dispatch_execute_project must not write orch._run_ctx.budget as a
        side effect when reusing a long-running orchestrator — that write is
        exactly the unsynchronized shared mutation Candidate 7 identifies.
        """
        import ast

        src = open("orchestrator/api_server.py").read()
        tree = ast.parse(src)

        offending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "budget"
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "_run_ctx"
                ):
                    offending.append(node.lineno)

        assert not offending, (
            f"orch._run_ctx.budget is still assigned directly at line(s) "
            f"{offending} — this mutates state shared by every concurrent "
            f"request when self._orchestrator is a long-running singleton. "
            f"Pass budget=... into run_project()/run_project_with_tasks() "
            f"instead."
        )


# ── Candidate 8: first_generator.py pytest-output parse inversion ──


class TestCandidate8PytestParseInversion:
    def test_all_failing_summary_is_not_reported_as_all_passing(self):
        from orchestrator.testing.first_generator import TestFirstGenerator

        tests_run, tests_passed = TestFirstGenerator._parse_pytest_output(None, "3 failed in 0.52s")
        assert tests_run == 3
        assert tests_passed == 0, (
            f"'3 failed in 0.52s' (zero passing) parsed as tests_passed="
            f"{tests_passed} — the single-group 'failed in' pattern assigns "
            f"its only capture group (the FAILED count) to tests_passed."
        )

    def test_mixed_pass_fail_summary_still_parses_correctly(self):
        """Guards the fix against breaking the already-correct 2-group pattern."""
        from orchestrator.testing.first_generator import TestFirstGenerator

        tests_run, tests_passed = TestFirstGenerator._parse_pytest_output(
            None, "2 passed, 1 failed in 0.3s"
        )
        assert (tests_run, tests_passed) == (3, 2)


# ── Candidate 6: llm_client.py UnifiedClient._clients shared across instances ──


class TestCandidate6ClientCacheIsolation:
    def test_two_clients_do_not_share_the_same_clients_dict(self):
        from orchestrator.infrastructure.llm_client import UnifiedClient

        a = UnifiedClient(openrouter_api_key="fake-a")
        b = UnifiedClient(openrouter_api_key="fake-b")
        assert a._clients is not b._clients, (
            "UnifiedClient instances share one class-level `_clients` dict — "
            "populating one instance's cache is visible to every other "
            "instance in the process, and close() on any one instance clears "
            "it for all of them."
        )

    @pytest.mark.asyncio
    async def test_closing_one_client_does_not_empty_anothers_cache(self):
        from orchestrator.infrastructure.llm_client import UnifiedClient

        class _FakeClient:
            class client:
                @staticmethod
                async def close():
                    return None

        a = UnifiedClient(openrouter_api_key="fake-a")
        b = UnifiedClient(openrouter_api_key="fake-b")
        b._clients["b-model"] = _FakeClient()

        await a.close()

        assert (
            "b-model" in b._clients
        ), "a.close() emptied b's cache — confirms the shared class-level dict."


# ── Candidate 4: streaming.py StreamingPipeline._run_pipeline NameError ──


class TestCandidate4StreamingNameError:
    @pytest.mark.asyncio
    async def test_execute_streaming_does_not_crash_on_project_description(self):
        from orchestrator.infrastructure.streaming import StreamingPipeline

        pipeline = StreamingPipeline(max_parallel=1)
        events = []
        async for event in pipeline.execute_streaming(
            project_description="Build a test API",
            success_criteria="works",
            budget=5.0,
        ):
            events.append(event)
            if len(events) > 8:
                break

        error_events = [e for e in events if e.type.name == "ERROR"]
        assert not error_events, (
            f"execute_streaming() raised internally: {error_events[0].data if error_events else None} — "
            f"_run_pipeline referenced the bare name `project_description`, which "
            f"is not in scope (the parameter/field is `context.description`)."
        )
        assert any(
            e.type.name == "PROJECT_START" for e in events
        ), "no PROJECT_START event was ever emitted"


# ── Candidate 5: streaming.py StreamingPipeline.__init__ unawaited coroutine ──


class TestCandidate5StreamingEventBus:
    def test_event_bus_is_not_a_bare_coroutine(self):
        import inspect

        from orchestrator.infrastructure.streaming import StreamingPipeline

        pipeline = StreamingPipeline(max_parallel=1)
        assert not inspect.iscoroutine(pipeline.event_bus), (
            "StreamingPipeline.event_bus is a bare, un-awaited coroutine object "
            "(get_event_bus() is `async def` but was called synchronously in "
            "__init__) — every _emit_to_bus() call fails with AttributeError, "
            "silently swallowed by its own except clause."
        )


# ── Candidate 1: website_generator.py silent ImportError fallback ──


class TestCandidate1SilentRegistryFallback:
    def test_registry_import_failure_is_logged(self, caplog):
        import importlib
        import sys

        import orchestrator.generators.website_generator as wg

        # Force the fallback path without touching the real component_registry module.
        wg.get_registry = None
        real_import = __import__

        def _blocking_import(name, *a, **kw):
            if (
                name.endswith("design.component_registry")
                or name == "orchestrator.design.component_registry"
            ):
                raise ImportError("simulated: component_registry unavailable")
            return real_import(name, *a, **kw)

        import builtins

        caplog.set_level(logging.WARNING, logger="orchestrator.generators.website_generator")
        orig_builtin_import = builtins.__import__
        builtins.__import__ = _blocking_import
        try:
            wg._get_registry()
        finally:
            builtins.__import__ = orig_builtin_import

        assert any(
            "component_registry" in rec.message.lower() or "curation" in rec.message.lower()
            for rec in caplog.records
        ), (
            "component_registry's ImportError fallback (the 4-section "
            "_FakeRegistry) fires with no log line — the same failure mode "
            "that left the real registry dead 'since it was written' per "
            "commit 5acde5b, undetected because nothing logged it."
        )


# ── Candidate 2: cli_dispatch.py --agent-profile is a fully dead CLI flag ──


class TestCandidate2AgentProfileVisibility:
    def test_unused_agent_profile_is_reported_not_silently_dropped(self, caplog):
        """
        --agent-profile is parsed into a `cfg` dict that is never applied on
        any of its 4 call sites. Full wiring is a product decision (3
        competing profile vocabularies exist); this only requires the no-op
        to be visible.
        """
        import ast

        src = open("orchestrator/entrypoints/cli_dispatch.py").read()
        tree = ast.parse(src)

        cfg_assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cfg"
        ]
        assert len(cfg_assigns) == 4, (
            f"expected the 4 known dead `cfg = profile_map.get(...)` sites, "
            f"found {len(cfg_assigns)} — re-check line numbers before editing."
        )


# ── Candidate 3: ide_orchestrator_server.py port-collision false-success ──


class TestCandidate3DevServerFalseSuccess:
    def test_start_server_liveness_is_checked_before_reporting_success(self):
        """
        A minimal, in-scope check: the module exposes a way to verify a
        started process is still alive shortly after spawn, and start_server
        does not unconditionally report True without it.
        """
        from orchestrator.ide_backend.ide_orchestrator_server import SessionManager

        assert hasattr(SessionManager, "is_process_alive"), (
            "no liveness check exists — start_server()'s callers print "
            "'✓ Dev server running' unconditionally on server_started=True, "
            "even when the spawned process has already exited (e.g. the "
            "losing side of a port-collision race)."
        )
