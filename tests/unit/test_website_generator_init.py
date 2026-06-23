"""
Unit tests for WebsiteGenerator construction / dependency wiring.

Regression coverage for the ``self._engine`` AttributeError bug: the LLM-powered
generation path reads ``self._engine`` (content brief, concurrency gate, syntax
auto-fix) but ``__init__`` only ever set ``self._executor`` — so every CLI-driven
website build crashed at the top of ``generate()`` and silently fell back to a
near-empty page.
"""

import pytest


class _FakeEngine:
    """Minimal stand-in for Orchestrator with the attrs WebsiteGenerator reads."""

    max_concurrency = 5

    async def _execute_task(self, task):  # pragma: no cover - not invoked here
        return None


@pytest.mark.unit
class TestWebsiteGeneratorInit:
    def test_engine_attribute_exists_when_engine_passed(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        engine = _FakeEngine()
        gen = WebsiteGenerator(orchestrator_engine=engine)

        # Must not raise AttributeError and must bind the engine.
        assert gen._engine is engine

    def test_engine_attribute_defaults_to_none_without_engine(self):
        from orchestrator.domain.ports import TaskExecutorAdapter
        from orchestrator.generators.website_generator import WebsiteGenerator

        adapter = TaskExecutorAdapter(lambda task: None)
        gen = WebsiteGenerator(executor=adapter)

        # Reading the attribute must not raise; absence of engine => None.
        assert gen._engine is None
        assert gen._executor is adapter

    def test_executor_adapted_from_engine_when_only_engine_passed(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        engine = _FakeEngine()
        gen = WebsiteGenerator(orchestrator_engine=engine)

        # _run_section calls self._executor.execute(task); the executor must
        # expose .execute() even when only a raw engine was supplied.
        assert hasattr(gen._executor, "execute")
