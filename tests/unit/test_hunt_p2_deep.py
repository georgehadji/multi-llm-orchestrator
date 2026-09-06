"""
V4 precision-audit, wave P2: DEEP tier, priority 8-7 region (16 files).

Each test proves the defect for the exact predicted reason (RED against the
pre-fix source) before the fix restores it (GREEN).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytestmark = pytest.mark.unit


# ── S2-1/S2-2: orchestrator/streaming.py (root) ──────────────────────────────


class TestS2StreamingRoot:
    def test_run_pipeline_uses_context_description_not_bare_name(self):
        """_run_pipeline referenced a bare `project_description` name that is
        not in its own scope (only `execute_streaming`'s scope has it) —
        NameError on the very first statement of every real invocation.
        """
        from orchestrator.streaming import StreamingContext, StreamingPipeline
        from orchestrator.budget import Budget

        pipeline = StreamingPipeline(max_parallel=1)
        context = StreamingContext(
            project_id="p1",
            description="Build a FastAPI REST API",
            budget=Budget(max_usd=1.0),
        )
        queue: asyncio.Queue = asyncio.Queue()

        asyncio.run(pipeline._run_pipeline(context, queue))

        event = queue.get_nowait()
        assert event.data["description"] == "Build a FastAPI REST API"

    def test_project_event_bus_init_does_not_leak_an_unawaited_coroutine(self):
        """ProjectEventBus.__init__ called the async get_event_bus() without
        awaiting it, binding self._event_bus to a bare coroutine that was
        never read again anywhere in the class. Nothing in ProjectEventBus
        reads self._event_bus, so the attribute should not exist at all.
        """
        from orchestrator.streaming import ProjectEventBus

        bus = ProjectEventBus()

        assert not hasattr(bus, "_event_bus"), (
            "ProjectEventBus still assigns a dead self._event_bus attribute "
            "(get_event_bus() called without await, never read again)."
        )

    def test_infrastructure_twin_project_event_bus_has_the_same_fix(self):
        """P1 follow-up, found incidentally while auditing this file's root
        twin for P2: infrastructure/streaming.py::ProjectEventBus had the
        identical bug (P1 only fixed StreamingPipeline in this file, not this
        second class) — same one-line deletion fix.
        """
        from orchestrator.infrastructure.streaming import ProjectEventBus

        bus = ProjectEventBus()

        assert not hasattr(bus, "_event_bus")


# ── PORTS-1: orchestrator/domain/ports.py ────────────────────────────────────


class TestPorts1NullTelemetrySignature:
    def test_null_telemetry_accepts_quality_score_like_its_protocol(self):
        """TelemetryPort.record_call declares quality_score; NullTelemetry's
        override omitted it entirely. isinstance() (runtime_checkable) cannot
        catch this — only an actual call with the keyword can.
        """
        from orchestrator.domain.ports import NullTelemetry

        null_telemetry = NullTelemetry()
        # Must not raise TypeError: unexpected keyword argument 'quality_score'
        null_telemetry.record_call(
            "model-x", latency_ms=1.0, cost_usd=0.01, success=True, quality_score=0.9
        )


# ── MODELS-1: orchestrator/models.py ─────────────────────────────────────────


class TestModels1VsVariantForDeadPremiumTier:
    def test_no_dead_premium_tuple_remains(self):
        """_premium was computed and never read — the function only ever had
        two effective branches (budget -> None, else -> one fixed VSConfig),
        not the three the old docstring claimed.
        """
        from orchestrator import models

        src = inspect.getsource(models.vs_variant_for)
        assert "_premium" not in src, (
            "vs_variant_for() still declares a dead _premium tuple that the "
            "docstring's 3-tier claim depends on but the code never reads."
        )

    def test_still_skips_vs_for_budget_models_and_returns_config_otherwise(self):
        """Behavior-preserving: budget models -> None, everything else -> VSConfig."""
        from orchestrator.models import Model, vs_variant_for

        assert vs_variant_for(Model.GEMINI_FLASH_LITE) is None
        cfg = vs_variant_for(Model.CLAUDE_OPUS_5)
        assert cfg is not None
        assert cfg.k == 5


# ── M2-3: orchestrator/model_selector.py ─────────────────────────────────────


class TestM23DecompositionModelIgnoresDescription:
    def test_logs_when_a_real_description_is_silently_ignored(self, caplog):
        """decomposition_model(project_description) never reads its argument;
        _COMPLEXITY_KEYWORDS/_TECH_STACK_KEYWORDS are declared and never used
        anywhere in the repo. Real callers (engine.py, application/decomposer.py)
        pass real descriptions expecting some effect; make the gap visible.
        """
        from orchestrator.model_selector import ModelSelector

        selector = ModelSelector(api_health={}, routing_service=None, cost_service=None)
        with caplog.at_level("DEBUG", logger="orchestrator.model_selector"):
            selector.decomposition_model("Build a real-time chat app with websockets")

        assert any("does not use project_description" in r.message for r in caplog.records)

    def test_empty_description_stays_quiet(self, caplog):
        """fast_decomposition_model()'s deliberate `""` call should not warn."""
        from orchestrator.model_selector import ModelSelector

        selector = ModelSelector(api_health={}, routing_service=None, cost_service=None)
        with caplog.at_level("DEBUG", logger="orchestrator.model_selector"):
            selector.decomposition_model("")

        assert not any("does not use project_description" in r.message for r in caplog.records)


# ── SLACK-1/SLACK-3: orchestrator/integrations/slack_integration.py ─────────


class TestSlack1RateLimiterUnboundedGrowth:
    def test_is_allowed_prunes_stored_timestamps_not_just_a_local_copy(self):
        """is_allowed() computed a pruned list but discarded it — only
        record_request() ever mutated self._requests, and it only appended.
        Stored state grew without bound for any repeatedly-used key.
        """
        from orchestrator.integrations.slack_integration import RateLimiter

        limiter = RateLimiter(max_requests=100, window_seconds=0.1)
        key = "team:channel"

        for _ in range(10):
            limiter.is_allowed(key)
            limiter.record_request(key)

        assert len(limiter._requests[key]) == 10

        import time

        time.sleep(0.15)
        limiter.is_allowed(key)  # window has fully elapsed — must prune to 0

        assert limiter._requests[key] == [], (
            "expired timestamps were never pruned from stored state — "
            f"got {limiter._requests[key]!r}"
        )


class TestSlack3ParseOverridesMalformedNumber:
    def test_multi_dot_numeric_value_does_not_crash(self):
        """ "1.2.3".replace(".", "").isdigit() is True, so the code unconditionally
        tried float("1.2.3") next -> uncaught ValueError, outside _handle_run's
        own try/except (which only wraps run_template()).
        """
        from orchestrator.integrations.slack_integration import (
            PresetTemplate,
            TemplateRegistry,
        )

        registry = TemplateRegistry()
        template = PresetTemplate(
            name="t",
            description="d",
            policy_overrides={},
            allowed_overrides=["budget"],
        )

        overrides = registry.parse_overrides("budget=1.2.3", template)

        assert "budget" not in overrides


# ── NASH-1: orchestrator/nash/infrastructure_v2.py — broken two-phase commit ─


class TestNash1WriteAheadLogTwoPhaseCommit:
    def test_append_large_file_does_not_raise_attribute_error(self, tmp_path):
        """`async with self._get_io() as io:` treated an async factory getter
        (returns an AsyncIOManager, which has no __aenter__/__aexit__) as if
        it were an async context manager — guaranteed AttributeError on every
        write >= WALEntry.MAX_STORED_SIZE (100KB), the two-phase-commit path
        this file's own TD-history calls out as hardened multiple times.
        Live via nash/__init__.py's wildcard export + nash/monitor.py's import.
        """
        from orchestrator.nash.infrastructure_v2 import WriteAheadLog

        async def scenario():
            wal = WriteAheadLog(wal_dir=tmp_path / "wal")
            target = tmp_path / "big_file.bin"
            large_data = b"x" * (150 * 1024)  # > MAX_STORED_SIZE (100KB)
            entry = await wal.append("write", target, large_data)
            return entry, target

        entry, target = asyncio.run(scenario())

        assert target.exists()
        assert target.read_bytes() == b"x" * (150 * 1024)
        assert entry.temp_path is not None


# ── ENGINE-1: orchestrator/engine.py — self._event_bus clobbering ───────────


class TestEngine1EventBusAttributeStability:
    def test_run_project_streaming_does_not_touch_self_event_bus(self):
        """run_project_streaming() used to overwrite self._event_bus (the
        container-wired bus __init__ sets at construction and assert_healthy()
        checks) with an unrelated per-call ProjectEventBus, then null it out
        in its finally block — corrupting shared instance state for any later
        call on a reused (long-running) Orchestrator, e.g. the supervisor REPL.
        """
        from orchestrator import engine

        src = inspect.getsource(engine.Orchestrator.run_project_streaming)
        code_lines = [line for line in src.splitlines() if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "self._event_bus" not in code_only, (
            "run_project_streaming() still reads/writes self._event_bus — "
            "the container-wired bus can still be clobbered by a streaming call."
        )
        assert "event_bus = ProjectEventBus()" in src


# ── UEB-1: orchestrator/unified_events/core.py — never started ──────────────


class TestUeb1EventBusNeverStarted:
    def test_publish_before_start_is_never_processed(self):
        """Reproduces the underlying mechanism: publish() queues but nothing
        drains the queue until start() runs a background _process_loop.
        """
        from orchestrator.unified_events.core import UnifiedEventBus

        async def scenario() -> bool:
            bus = UnifiedEventBus()
            bus.subscribe(lambda e: None)
            processed_before = bus._events_routed if hasattr(bus, "_events_routed") else None
            await bus.publish(_DummyEvent())
            await asyncio.sleep(0.05)
            # Without start(), _process_loop never runs, so the event sits in
            # the internal queue forever.
            return bus._event_queue.qsize() == 1

        assert asyncio.run(scenario()) is True

    def test_orchestrator_aenter_starts_the_event_bus(self):
        """The container builds UnifiedEventBus() directly (not via the async
        get_instance() singleton) and build() is itself a sync classmethod, so
        it structurally cannot await start() — __aenter__ is the natural,
        already-existing async lifecycle hook to do it instead.
        """
        from orchestrator import engine

        src = inspect.getsource(engine.Orchestrator.__aenter__)
        assert "_event_bus" in src and ".start()" in src, (
            "__aenter__() does not start the event bus — UnifiedEventBus.publish() "
            "warns forever that events are queued but never processed."
        )


class _DummyEvent:
    pass


# ── TELEMETRY-1: orchestrator/state_mgmt/telemetry_store.py ─────────────────


class TestTelemetry1DrainQueueWiredAtWarmStart:
    def test_drain_queue_promotes_enqueued_snapshot_to_model_snapshots(self, tmp_path):
        """enqueue_snapshot()/drain_queue() are the documented WAL recovery
        path ("Called at warm-start time") but had zero callers anywhere —
        verify the mechanism itself works before checking it's wired in.
        """
        from orchestrator.state_mgmt.telemetry_store import TelemetryStore
        from orchestrator.models import Model, TaskType

        class _FakeProfile:
            quality_score = 0.8
            trust_factor = 0.9
            avg_latency_ms = 100.0
            latency_p95_ms = 150.0
            success_rate = 1.0
            avg_cost_usd = 0.01
            call_count = 5
            failure_count = 0
            validator_fail_count = 0

        async def scenario() -> int:
            store = TelemetryStore(db_path=tmp_path / "telemetry.db")
            await store.enqueue_snapshot(
                "proj-1", Model.GPT_4O_MINI, TaskType.CODE_GEN, _FakeProfile()
            )
            return await store.drain_queue()

        drained = asyncio.run(scenario())
        assert drained == 1

    def test_orchestrator_aenter_drains_the_telemetry_queue(self):
        from orchestrator import engine

        src = inspect.getsource(engine.Orchestrator.__aenter__)
        assert "drain_queue" in src, (
            "__aenter__() never calls telemetry_store.drain_queue() — orphaned "
            "writes from a prior crashed session are never recovered."
        )


# ── ARCH-1/ARCH-2: orchestrator/architecture_rules.py ────────────────────────


class TestArch1FakeModelFallbackLoop:
    def test_no_dead_try_except_model_selection_loop_remains(self):
        """`model = m; break` inside `try: ... except Exception: continue` can
        never raise, so the loop always picked architecture_models[0] and the
        other 4 declared fallbacks plus the final GPT_4O default were dead.
        """
        from orchestrator import architecture_rules

        src = inspect.getsource(architecture_rules.ArchitectureRulesEngine._generate_rules_with_llm)
        assert "for m in architecture_models" not in src


class TestArch2PromptJsonExampleBraceBalance:
    def test_optimize_prompt_json_example_is_brace_balanced(self):
        """The 'respond in this JSON format' example template had 3 literal
        `{` opens vs 4 literal `}` closes once the f-string was rendered — a
        stray trailing brace in instructional text shown to the LLM.
        """
        from orchestrator.architecture_rules import (
            ArchitectureDecision,
            ArchitecturalStyle,
            ArchitectureRulesEngine,
            APIStyle,
            CodingStandard,
            DatabaseType,
            ProgrammingParadigm,
            ProjectRules,
            TechnologyStack,
        )

        captured = {}

        class _FakeResponse:
            text = ""

        class _FakeClient:
            async def call(self, *args, **kwargs):
                captured["prompt"] = args[1] if len(args) > 1 else kwargs.get("prompt")
                return _FakeResponse()

        async def scenario():
            engine = ArchitectureRulesEngine(client=_FakeClient())
            rules = ProjectRules(
                architecture=ArchitectureDecision(
                    style=ArchitecturalStyle.LAYERED,
                    paradigm=ProgrammingParadigm.OBJECT_ORIENTED,
                    api_style=APIStyle.REST,
                    database_type=DatabaseType.RELATIONAL,
                    stack=TechnologyStack(primary_language="python"),
                ),
                coding_standards=CodingStandard(),
            )
            return await engine._optimize_rules_with_llm(rules, "desc", "criteria")

        result = asyncio.run(scenario())

        assert result is None  # empty fake response -> no optimization, as designed
        prompt = captured["prompt"]
        assert prompt.count("{") == prompt.count("}"), (
            f"prompt has unbalanced braces: {prompt.count('{')} opens vs "
            f"{prompt.count('}')} closes"
        )


# ── TRANSFER-1/TRANSFER-2: orchestrator/transfer_learning.py ────────────────


class TestTransfer1SimilarityFilteringBypass:
    def test_warns_that_results_are_unfiltered_by_similarity(self, caplog, tmp_path):
        """similar_projects was computed and used only for the empty-result
        guard — every ACTIVE pattern was returned regardless of similarity.
        """
        from orchestrator.transfer_learning import (
            PatternType,
            ProjectEmbedding,
            ProjectFeatures,
            TransferLearningEngine,
            TransferPattern,
        )

        class _FakeArchive:
            _records: list = []

        engine = TransferLearningEngine(archive=_FakeArchive(), storage_path=tmp_path)
        engine._patterns["p1"] = TransferPattern(
            pattern_id="p1",
            pattern_type=PatternType.MODEL_ROUTING,
            source_projects=[],
            pattern_data={},
        )
        engine._similarity_engine.add_embedding(
            ProjectEmbedding(
                project_id="current",
                embedding=[1.0, 0.0],
                features=ProjectFeatures(project_id="current"),
            )
        )
        engine._similarity_engine.add_embedding(
            ProjectEmbedding(
                project_id="other",
                embedding=[1.0, 0.0],
                features=ProjectFeatures(project_id="other"),
            )
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="orchestrator.transfer"):
            patterns = asyncio.run(engine.find_transferable_patterns("current", min_similarity=0.0))

        assert len(patterns) == 1
        assert any("unfiltered by" in r.message for r in caplog.records)


class TestTransfer2DeprecatedImportPath:
    def test_does_not_import_via_the_deprecated_shim(self):
        from orchestrator import transfer_learning

        src = inspect.getsource(transfer_learning)
        assert "from .meta_orchestrator import" not in src
        assert "from .meta.orchestrator import" in src


# ── AB-1: orchestrator/events/ab_testing.py — broken t-distribution CDF ─────


class TestAb1TDistributionCdfAccuracy:
    def test_cdf_matches_known_reference_values(self):
        """Old approximation gave t_cdf(1.0, df=10) ~= 0.019 against a true
        value of ~0.83 — a two-tailed p-value computation that came out as
        1.96 (impossible for a real probability) and could never detect
        significance for any experiment with Welch-Satterthwaite df <= 30.
        """
        from orchestrator.events.ab_testing import StatisticalAnalyzer

        cdf = StatisticalAnalyzer._t_distribution_cdf(1.0, 10)
        assert 0.0 <= cdf <= 1.0
        assert abs(cdf - 0.8296) < 0.01

    def test_two_sample_t_test_produces_a_valid_probability(self):
        from orchestrator.events.ab_testing import StatisticalAnalyzer

        control = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
        treatment = [0.9, 0.88, 0.92, 0.91, 0.89, 0.90, 0.93, 0.87]

        t_stat, p_value = StatisticalAnalyzer.two_sample_t_test(control, treatment)

        assert 0.0 <= p_value <= 1.0
        assert p_value < 0.05  # a huge, obvious effect must be detected as significant
