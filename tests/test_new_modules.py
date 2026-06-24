"""Comprehensive test suite for newly added orchestrator modules.

Covers: autonomy, brainstorming, auto-fix, quick actions, security,
config-as-code, modules, automations, versioning, checkpoints, context,
documentation, self-review, plan review, cost tracking, git integration,
i18n, type generation, sandbox, dev server, skills, file scope,
release management, slash integrations, RLS, multi-context, design system,
progress, diagnostics, team templates, and cross-project references.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Autonomous Execution
# ═══════════════════════════════════════════════════════════════════════════


class TestAutonomyConfig:
    """Tests for autonomy_config.py — Multi-Mode Selector."""

    def test_for_level_lite(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig, AutonomyLevel

        cfg = AutonomyConfig.for_level(AutonomyLevel.LITE)
        assert cfg.max_iterations == 0
        assert cfg.repair_attempts == 0
        assert cfg.is_lite
        assert not cfg.is_autonomous

    def test_for_level_standard(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig, AutonomyLevel

        cfg = AutonomyConfig.for_level(AutonomyLevel.STANDARD)
        assert cfg.max_iterations == 3
        assert cfg.critique_passes == 1
        assert cfg.is_standard

    def test_for_level_max(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig, AutonomyLevel

        cfg = AutonomyConfig.for_level(AutonomyLevel.MAX)
        assert cfg.max_iterations == 10
        assert cfg.repair_attempts == 5
        assert cfg.verification_mode == "behavioral"
        assert cfg.strict_validation
        assert cfg.require_documentation
        assert cfg.is_max

    def test_from_agent_profile_mapping(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig

        cfg = AutonomyConfig.from_agent_profile("max")
        assert cfg.level.value == "max"
        cfg2 = AutonomyConfig.from_agent_profile("conservative")
        assert cfg2.level.value == "standard"

    def test_apply_to_task_sets_limits(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig, AutonomyLevel

        task = type("Task", (), {"max_iterations": 0, "acceptance_threshold": 0.0})()
        cfg = AutonomyConfig.for_level(AutonomyLevel.AUTO)
        cfg.apply_to_task(task)
        assert task.max_iterations == 5
        assert task.acceptance_threshold == 0.85

    def test_model_tier_for_purposes(self):
        from orchestrator.operations.autonomy_config import AutonomyConfig, AutonomyLevel

        cfg = AutonomyConfig.for_level(AutonomyLevel.MAX)
        assert cfg.model_tier_for("generation") == "reasoning"
        assert cfg.model_tier_for("decomposition") == "premium"


class TestBrainstormingDecomposer:
    """Tests for brainstorming.py."""

    def test_clarifying_questions_complete(self):
        from orchestrator.reasoning.brainstorming import ClarifyingQuestions

        cq = ClarifyingQuestions(questions=["Q1?", "Q2?"])
        assert not cq.is_complete
        cq.answers = ["A1", "A2"]
        assert cq.is_complete

    def test_clarifying_questions_context(self):
        from orchestrator.reasoning.brainstorming import ClarifyingQuestions

        cq = ClarifyingQuestions(questions=["What framework?"], answers=["FastAPI"])
        ctx = cq.build_context()
        assert "What framework?" in ctx
        assert "FastAPI" in ctx

    def test_brainstorming_without_client_returns_empty(self):
        from orchestrator.reasoning.brainstorming import BrainstormingDecomposer

        decomposer = BrainstormingDecomposer(client=None)
        result = asyncio.run(decomposer.ask("Build an API"))
        assert result == []

    def test_inject_into_description(self):
        from orchestrator.reasoning.brainstorming import (
            BrainstormingDecomposer,
            ClarifyingQuestions,
        )

        decomposer = BrainstormingDecomposer(client=None)
        # Manually set pending state
        decomposer._pending = ClarifyingQuestions(questions=["Q1?"], answers=["A1"])
        result = decomposer.inject_into_description("Build API", ["A1"])
        assert "Q1?" in result
        assert "A1" in result


class TestAutoErrorFixer:
    """Tests for auto_error_fix.py."""

    def test_extract_syntax_error(self):
        from orchestrator.auto_error_fix import AutoErrorFixer

        fixer = AutoErrorFixer()
        error = fixer.extract_error_from_output(
            "Traceback...\nSyntaxError: invalid syntax at line 5\n"
        )
        assert "SyntaxError" in error

    def test_extract_type_error(self):
        from orchestrator.auto_error_fix import AutoErrorFixer

        fixer = AutoErrorFixer()
        error = fixer.extract_error_from_output("TypeError: 'NoneType' object is not callable\n")
        assert "TypeError" in error

    def test_validate_fix_valid_syntax(self):
        from orchestrator.auto_error_fix import AutoErrorFixer

        result = asyncio.run(AutoErrorFixer._validate_fix("def foo(): pass"))
        assert result

    def test_validate_fix_bad_syntax(self):
        from orchestrator.auto_error_fix import AutoErrorFixer

        result = asyncio.run(AutoErrorFixer._validate_fix("def foo(: pass"))
        assert not result


class TestQuickActionResolver:
    """Tests for quick_actions.py."""

    def test_suggest_basic(self):
        from orchestrator.operations.quick_actions import QuickActionResolver

        resolver = QuickActionResolver()
        actions = resolver.suggest(tasks_count=4, estimated_cost=1.0)
        labels = [a.label for a in actions]
        assert "Implement Plan" in labels
        assert "Generate Docs" in labels  # tasks > 3

    def test_suggest_complex_plan(self):
        from orchestrator.operations.quick_actions import QuickActionResolver

        resolver = QuickActionResolver()
        actions = resolver.suggest(tasks_count=12, estimated_cost=8.0, has_dependencies=True)
        labels = [a.label for a in actions]
        assert "Refine Plan" in labels  # tasks > 5, cost > 5
        assert "Show Estimate" in labels  # cost > 2
        assert "Review Plan" in labels  # tasks > 10

    def test_resolve_hotkey(self):
        from orchestrator.operations.quick_actions import QuickActionResolver

        resolver = QuickActionResolver()
        action = resolver.resolve("Enter")
        assert action is not None
        assert action.label == "Implement Plan"

    def test_present_formats_actions(self):
        from orchestrator.operations.quick_actions import QuickActionResolver

        resolver = QuickActionResolver()
        actions = resolver.suggest(tasks_count=2, estimated_cost=1.0)
        output = resolver.present(actions[:3])
        assert "Implement Plan" in output


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Output Quality
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigAsCode:
    """Tests for config_as_code.py."""

    def test_entity_schema_pydantic_generation(self):
        from orchestrator.config_as_code import EntitySchema, EntityField, FieldType

        entity = EntitySchema(
            name="User",
            fields=[
                EntityField(name="email", type=FieldType.EMAIL, required=True),
                EntityField(name="name", type=FieldType.STRING, description="Full name"),
            ],
        )
        code = entity.to_pydantic()
        assert "class User(BaseModel):" in code
        assert "email: str" in code
        assert "Optional[str] = None" in code  # name is not required

    def test_entity_schema_typescript_generation(self):
        from orchestrator.config_as_code import EntitySchema, EntityField, FieldType

        entity = EntitySchema(
            name="Product",
            fields=[EntityField(name="price", type=FieldType.FLOAT, required=True)],
        )
        code = entity.to_typescript()
        assert "export interface Product" in code
        assert "price: number" in code
        assert "?" not in code.split("price")[1][:5]  # required = no ?

    def test_app_config_save_load_json(self):
        from orchestrator.config_as_code import AppConfig, EntitySchema, EntityField, FieldType

        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig(app_name="TestApp", api_prefix="/api/v1")
            entity = EntitySchema(
                name="Item", fields=[EntityField(name="id", type=FieldType.UUID, required=True)]
            )
            cfg.entities = [entity]
            path = str(Path(d) / "config.json")
            cfg.save(path)
            loaded = AppConfig.load(path)
            assert loaded.app_name == "TestApp"
            assert len(loaded.entities) == 1
            assert loaded.entities[0].name == "Item"

    def test_generate_all_types(self):
        from orchestrator.config_as_code import AppConfig, EntitySchema, EntityField, FieldType

        cfg = AppConfig(app_name="Test")
        entity = EntitySchema(
            name="Book", fields=[EntityField(name="title", type=FieldType.STRING, required=True)]
        )
        cfg.entities = [entity]
        types = cfg.generate_all_types()
        assert "Book_pydantic.py" in types
        assert "Book.ts" in types


class TestModuleRegistry:
    """Tests for module_system.py."""

    def test_register_and_get(self):
        from orchestrator.operations.module_system import (
            ModuleRegistry,
            ModuleDefinition,
            ModuleKind,
        )

        registry = ModuleRegistry()
        module = ModuleDefinition(
            name="auth", kind=ModuleKind.UTILITY, version="1.0.0", description="Auth module"
        )
        registry.register(module)
        result = registry.get("auth")
        assert result is not None
        assert result.name == "auth"
        assert result.version == "1.0.0"

    def test_resolve_dependencies_linear(self):
        from orchestrator.operations.module_system import ModuleRegistry, ModuleDefinition

        registry = ModuleRegistry()
        registry.register(ModuleDefinition(name="c", dependencies=["b"]))
        registry.register(ModuleDefinition(name="b", dependencies=["a"]))
        registry.register(ModuleDefinition(name="a"))
        resolved = registry.resolve_dependencies("c")
        assert set(resolved) == {"a", "b", "c"}, f"Got: {resolved}"

    def test_circular_dependency_detection(self):
        from orchestrator.operations.module_system import ModuleRegistry, ModuleDefinition

        registry = ModuleRegistry()
        registry.register(ModuleDefinition(name="x", dependencies=["y"]))
        registry.register(ModuleDefinition(name="y", dependencies=["x"]))
        with pytest.raises(ValueError, match=r"Circular"):
            registry.resolve_dependencies("x")

    def test_validate_inputs(self):
        from orchestrator.operations.module_system import ModuleDefinition, ModuleInput

        module = ModuleDefinition(
            name="test",
            inputs=[
                ModuleInput(name="api_key", required=True),
                ModuleInput(name="debug", required=False),
            ],
        )
        missing = module.validate_inputs({"debug": True})
        assert "api_key" in missing
        no_missing = module.validate_inputs({"api_key": "sk-123", "debug": True})
        assert no_missing == []


class TestAutomationScheduler:
    """Tests for automations.py."""

    def test_cron_parser_wildcard(self):
        from orchestrator.operations.automations import CronParser

        # Every minute: * * * * *
        t = time.localtime()
        result = CronParser.matches("* * * * *", time.mktime(t))
        assert result

    def test_cron_parser_specific_minute(self):
        from orchestrator.operations.automations import CronParser

        # At minute 99 (should never match)
        result = CronParser.matches("99 * * * *")
        assert not result

    def test_cron_parser_every_n(self):
        from orchestrator.operations.automations import CronParser

        # */1 * * * * matches every minute
        result = CronParser.matches("*/1 * * * *")
        assert result

    def test_scheduled_task_to_dict(self):
        from orchestrator.operations.automations import ScheduledTask, ScheduleType

        task = ScheduledTask(
            name="cleanup",
            schedule_type=ScheduleType.CRON,
            cron_expr="0 */6 * * *",
            run_count=5,
        )
        d = task.to_dict()
        assert d["name"] == "cleanup"
        assert d["type"] == "cron"
        assert d["cron"] == "0 */6 * * *"
        assert d["run_count"] == 5

    @pytest.mark.asyncio
    async def test_automation_scheduler_register(self):
        from orchestrator.operations.automations import (
            AutomationScheduler,
            ScheduledTask,
            ScheduleType,
        )

        scheduler = AutomationScheduler()
        task = ScheduledTask(
            name="test-task", schedule_type=ScheduleType.INTERVAL, interval_seconds=3600
        )

        async def handler():
            pass

        scheduler.register(task, handler)
        assert "test-task" in scheduler._tasks


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Version Control
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionManager:
    """Tests for version_manager.py."""

    @pytest.mark.asyncio
    async def test_capture_first_version(self):
        from orchestrator.version_manager import VersionManager

        with tempfile.TemporaryDirectory() as d:
            vm = VersionManager(d)
            v = await vm.capture("initial commit")
            assert v.version_id == "v1"
            assert vm.version_count == 1
            assert vm.latest.version_id == "v1"

    @pytest.mark.asyncio
    async def test_capture_multiple_versions(self):
        from orchestrator.version_manager import VersionManager

        with tempfile.TemporaryDirectory() as d:
            vm = VersionManager(d)
            (Path(d) / "test.py").write_text("print('hello')")
            await vm.capture("first")
            (Path(d) / "test.py").write_text("print('world')")
            await vm.capture("second")
            assert vm.version_count == 2
            assert vm.latest.version_id == "v2"

    def test_version_chain(self):
        from orchestrator.version_manager import VersionManager, CodeVersion

        with tempfile.TemporaryDirectory() as d:
            vm = VersionManager(d)
            vm._versions = [
                CodeVersion(version_id="v1", description="first"),
                CodeVersion(version_id="v2", description="second"),
            ]
            chain = vm.chain()
            assert len(chain) == 2
            assert chain[0]["id"] == "v1"
            assert chain[1]["id"] == "v2"


class TestReleaseManager:
    """Tests for release_manager.py."""

    def test_bump_patch(self):
        import tempfile
        from orchestrator.release_manager import ReleaseManager

        with tempfile.TemporaryDirectory() as d:
            rm = ReleaseManager(project_dir=d)
            v = rm.bump("patch")
            assert v == "0.1.1"

    def test_bump_minor(self):
        import tempfile
        from orchestrator.release_manager import ReleaseManager

        with tempfile.TemporaryDirectory() as d:
            rm = ReleaseManager(project_dir=d)
            v = rm.bump("minor")
            assert v == "0.2.0"

    def test_create_and_publish_release(self):
        from orchestrator.release_manager import ReleaseManager

        rm = ReleaseManager()
        rel = rm.create_release(
            version="1.0.0", description="First release", changes=["Added auth"]
        )
        assert rel.status == "draft"
        published = rm.publish("1.0.0")
        assert published
        assert rel.status == "published"

    def test_generate_changelog(self):
        from orchestrator.release_manager import ReleaseManager

        rm = ReleaseManager()
        rm.create_release(version="1.0.0", description="Initial", changes=["Setup"])
        rm.publish("1.0.0")
        rm.create_release(version="1.1.0", description="Feature", changes=["Auth module"])
        changelog = rm.generate_changelog()
        assert "1.1.0" in changelog
        assert "1.0.0" in changelog


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Security
# ═══════════════════════════════════════════════════════════════════════════


class TestSecurityReviewer:
    """Tests for security_review.py."""

    def test_quick_scan_detects_hardcoded_key(self):
        from orchestrator.safety.security_review import SecurityReviewer

        sr = SecurityReviewer()
        report = sr.quick_scan("api_key = 'sk-abc123def456'")
        assert report.total_findings > 0
        assert any(f.rule_id == "SEC-001" for f in report.findings)

    def test_quick_scan_detects_debug_mode(self):
        from orchestrator.safety.security_review import SecurityReviewer

        sr = SecurityReviewer()
        report = sr.quick_scan("DEBUG = True")
        assert any(f.rule_id == "SEC-007" for f in report.findings)

    def test_quick_scan_detects_md5(self):
        from orchestrator.safety.security_review import SecurityReviewer

        sr = SecurityReviewer()
        report = sr.quick_scan("hashlib.md5(b'data')")
        assert any(f.rule_id == "SEC-005" for f in report.findings)

    def test_clean_code_passes(self):
        from orchestrator.safety.security_review import SecurityReviewer

        sr = SecurityReviewer()
        report = sr.quick_scan("def add(a, b): return a + b")
        assert report.total_findings == 0
        assert report.passed

    def test_report_markdown_generation(self):
        from orchestrator.safety.security_review import (
            SecurityReport,
            SecurityFinding,
            Severity,
            Category,
        )

        report = SecurityReport(
            findings=[
                SecurityFinding(
                    "SEC-001",
                    "Hardcoded key",
                    Severity.CRITICAL,
                    Category.SECRETS,
                    "Found API key",
                    location="config.py:5",
                    cwe_id="CWE-798",
                ),
            ],
            total_findings=1,
            critical_count=1,
        )
        md = report.to_markdown()
        assert "CRITICAL" in md
        assert "CWE-798" in md


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Context & Knowledge
# ═══════════════════════════════════════════════════════════════════════════


class TestContextSystem:
    """Tests for context_system.py."""

    def test_knowledge_file_add_and_get(self):
        from orchestrator.context_mgmt.system import WorkspaceKnowledge

        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        kf = wk.add("conventions", "# Coding Style\nUse type hints.", "architecture")
        assert kf.name == "conventions"
        retrieved = wk.get("conventions")
        assert retrieved is not None
        assert "type hints" in retrieved.content

    def test_knowledge_build_context(self):
        from orchestrator.context_mgmt.system import WorkspaceKnowledge

        wk = WorkspaceKnowledge(str(Path(tempfile.mkdtemp())))
        wk.add("security", "# Security\nNo hardcoded keys.")
        ctx = wk.build_context()
        assert "Security" in ctx

    def test_skill_from_file(self):
        content = "# Python Testing - Write unit tests\n## Triggers\n- pytest\n- test\n## Instructions\nUse pytest fixtures."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            fpath = f.name
        try:
            from orchestrator.context_mgmt.system import Skill

            skill = Skill.from_file(Path(fpath))
            assert skill is not None
            assert skill.name == "Python Testing"
            assert "pytest" in skill.triggers
            assert "fixtures" in skill.content
        finally:
            Path(fpath).unlink()


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Category: Cost Tracking, Docs, Self-Review, Plan Review, Git
# ═══════════════════════════════════════════════════════════════════════════


class TestCostTracker:
    """Tests for cost_tracker.py."""

    def test_record_and_total(self):
        import tempfile
        from orchestrator.cost_tracker import CostTracker

        with tempfile.TemporaryDirectory() as d:
            ct = CostTracker(storage_dir=d)
            ct.record("gpt-4o", 100, 50, 0.005, 245.0)
            ct.record("gpt-4o", 200, 100, 0.010, 300.0)
            assert ct.total_cost_usd == 0.015
            assert ct.total_tokens == 450

    def test_last_call(self):
        from orchestrator.cost_tracker import CostTracker

        ct = CostTracker()
        ct.record("gpt-4o", 100, 50, 0.005)
        last = ct.last_call()
        assert last is not None
        assert last.model == "gpt-4o"
        assert last.cost_usd == 0.005

    def test_status_line(self):
        from orchestrator.cost_tracker import CostTracker
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            ct = CostTracker(storage_dir=d)
            ct.record("gpt-4o", 100, 50, 0.005, 245.0)
            status = ct.status_line("gpt-4o")
            assert "gpt-4o" in status
            assert "100+50" in status

    def test_per_model_breakdown(self):
        from orchestrator.cost_tracker import CostTracker

        ct = CostTracker()
        ct.record("gpt-4o", 100, 50, 0.005)
        ct.record("claude", 80, 40, 0.003)
        models = ct.per_model()
        assert len(models) >= 2


class TestSelfReviewer:
    """Tests for self_review.py."""

    def test_should_skip_cross_review_high_score(self):
        from orchestrator.quality.self_review import SelfReviewer, SelfReviewConfig

        reviewer = SelfReviewer()
        skip = reviewer.should_skip_cross_review(0.85, 0, SelfReviewConfig(pass_threshold=0.7))
        assert skip

    def test_should_not_skip_with_deterministic_issues(self):
        from orchestrator.quality.self_review import SelfReviewer, SelfReviewConfig

        reviewer = SelfReviewer()
        skip = reviewer.should_skip_cross_review(0.90, 3, SelfReviewConfig(pass_threshold=0.7))
        assert not skip

    def test_should_not_skip_when_disabled(self):
        from orchestrator.quality.self_review import SelfReviewer, SelfReviewConfig

        reviewer = SelfReviewer()
        skip = reviewer.should_skip_cross_review(0.99, 0, SelfReviewConfig(enabled=False))
        assert not skip


class TestPlanReviewer:
    """Tests for plan_reviewer.py."""

    def test_estimate_resources(self):
        from orchestrator.plan_reviewer import PlanReviewer

        pr = PlanReviewer()
        tasks = [
            {"id": "1", "description": "task1"},
            {"id": "2", "description": "task2", "dependencies": ["1"]},
            {"id": "3", "description": "task3"},
        ]
        est = pr.estimate_resources(tasks)
        assert est["task_count"] == 3
        assert est["estimated_cost_usd"] == pytest.approx(0.45)
        assert est["has_dependencies"]

    def test_validate_plan_catches_missing_dependency(self):
        from orchestrator.plan_reviewer import PlanReviewer

        pr = PlanReviewer()
        tasks = [{"id": "1", "description": "task", "dependencies": ["nonexistent"]}]
        issues = pr.validate_plan(tasks)
        assert len(issues) > 0
        assert any(
            "unknown" in i.lower() or "does not exist" in i.lower() or "depends" in i.lower()
            for i in issues
        )

    def test_validate_plan_catches_missing_description(self):
        from orchestrator.plan_reviewer import PlanReviewer

        pr = PlanReviewer()
        tasks = [{"id": "1"}]
        issues = pr.validate_plan(tasks)
        assert any("description" in i.lower() for i in issues)


class TestDocGenerator:
    """Tests for doc_generator.py."""

    def test_generate_readme_fast(self):
        from orchestrator.generators.doc_generator import DocGenerator

        dg = DocGenerator()
        readme = dg.generate_readme_fast("MyApp", "A great app", ["Add auth", "Add API"])
        assert "# MyApp" in readme
        assert "A great app" in readme
        assert "Add auth" in readme

    def test_generate_changelog(self):
        from orchestrator.generators.doc_generator import DocGenerator

        dg = DocGenerator()
        changelog = dg.generate_changelog([("1.0.0", "Initial"), ("1.1.0", "Auth module")])
        assert "1.0.0" in changelog
        assert "1.1.0" in changelog
        assert "Initial" in changelog


class TestGitIntegrationMessages:
    """Tests for git_integration.py."""

    def test_generate_message_fast(self):
        from orchestrator.git_integration import GitIntegration

        git = GitIntegration()
        msg = git.generate_message_fast("task_001", "Add login endpoint", 0.95)
        assert "task_001" in msg.summary
        assert "Add login endpoint" in msg.summary

    def test_commit_message_full(self):
        from orchestrator.git_integration import CommitMessage

        msg = CommitMessage(
            summary="feat(auth): add JWT login",
            body="Implemented JWT authentication",
            task_id="task_002",
        )
        full = msg.full_message
        assert "feat(auth)" in full
        assert "JWT authentication" in full
        assert "Co-Authored-By" in full


class TestI18nManager:
    """Tests for i18n.py."""

    def test_extract_translatable_strings(self):
        from orchestrator.operations.i18n import I18nManager

        i18n = I18nManager()
        keys = i18n.extract_from_code('_("Hello World")', "test.py")
        assert len(keys) > 0
        assert keys[0].default_text == "Hello World"

    def test_text_to_key_conversion(self):
        from orchestrator.operations.i18n import I18nManager

        key = I18nManager._text_to_key("Hello World!")
        assert key == "hello_world"

    def test_generate_base_locale(self):
        from orchestrator.operations.i18n import I18nManager

        with tempfile.TemporaryDirectory() as d:
            i18n = I18nManager(locales_dir=d)
            i18n.extract_from_code('_("Submit")', "test.py")
            lf = i18n.generate_base("en")
            assert lf.locale == "en"
            assert "submit" in lf.translations
            assert lf.translations["submit"] == "Submit"


class TestTypeGenerator:
    """Tests for type_generator.py."""

    def test_generate_pydantic(self):
        from orchestrator.generators.type_generator import DynamicTypeGenerator, TargetLanguage

        gen = DynamicTypeGenerator()
        entity = type(
            "Entity",
            (),
            {
                "name": "User",
                "fields": [
                    type("F", (), {"name": "email", "type": "email", "required": True}),
                    type("F", (), {"name": "age", "type": "integer", "required": False}),
                ],
            },
        )()
        output = gen.generate(entity, TargetLanguage.PYTHON)
        assert "class User" in output.code
        assert "email: str" in output.code
        assert output.filename.endswith(".py")

    def test_generate_typescript(self):
        from orchestrator.generators.type_generator import DynamicTypeGenerator, TargetLanguage

        gen = DynamicTypeGenerator()
        entity = type(
            "Entity",
            (),
            {
                "name": "Product",
                "fields": [type("F", (), {"name": "price", "type": "float", "required": True})],
            },
        )()
        output = gen.generate(entity, TargetLanguage.TYPESCRIPT)
        assert "export interface Product" in output.code
        assert "number" in output.code

    def test_generate_sql(self):
        from orchestrator.generators.type_generator import DynamicTypeGenerator, TargetLanguage

        gen = DynamicTypeGenerator()
        entity = type(
            "Entity",
            (),
            {
                "name": "Order",
                "fields": [
                    type("F", (), {"name": "amount", "type": "float", "required": True}),
                    type("F", (), {"name": "email", "type": "email", "unique": True}),
                ],
            },
        )()
        output = gen.generate(entity, TargetLanguage.SQL)
        assert "CREATE TABLE order" in output.code


class TestSandboxExecutor:
    """Tests for sandbox_executor.py."""

    @pytest.mark.asyncio
    async def test_run_writes_to_sandbox(self):
        from orchestrator.safety.sandbox_executor import SandboxExecutor

        with tempfile.TemporaryDirectory() as d:
            sandbox = SandboxExecutor(project_dir=d)
            result = await sandbox.run(
                task_id="test-001",
                code="def hello(): pass",
                output_path="src/hello.py",
            )
            assert result.success
            assert "src/hello.py" in result.files_changed
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_reject_discards_changes(self):
        from orchestrator.safety.sandbox_executor import SandboxExecutor

        with tempfile.TemporaryDirectory() as d:
            sandbox = SandboxExecutor(project_dir=d)
            await sandbox.run("t1", "x = 1", "out.py")
            sandbox.reject()
            assert sandbox._pending is None


class TestDevServer:
    """Tests for dev_server.py."""

    def test_detect_fastapi_project(self):
        from orchestrator.dev_server import DevServer

        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").write_text("from fastapi import FastAPI")
            Path(d, "requirements.txt").write_text("fastapi")
            server = DevServer(d)
            pt = server.detect()
            assert pt is not None
            assert pt.name == "FastAPI"

    def test_detect_react_project(self):
        from orchestrator.dev_server import DevServer

        with tempfile.TemporaryDirectory() as d:
            Path(d, "src", "App.jsx").parent.mkdir(parents=True)
            Path(d, "package.json").write_text("{}")
            server = DevServer(d)
            pt = server.detect()
            assert pt is not None
            assert pt.name in ("React", "Next.js")

    def test_detect_on_empty_dir(self):
        from orchestrator.dev_server import DevServer

        with tempfile.TemporaryDirectory() as d:
            server = DevServer(d)
            pt = server.detect()
            assert pt is None


class TestSkillRegistry:
    """Tests for context_system.py SkillRegistry."""

    def test_skill_match_triggers(self):
        from orchestrator.context_mgmt.system import SkillRegistry
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            skills_dir = Path(d) / "skills"
            skills_dir.mkdir()
            (skills_dir / "python.md").write_text(
                "# Python Testing - Write tests\n## Triggers\n- pytest\n- unittest\n## Instructions\nUse pytest."
            )
            registry = SkillRegistry(workspace_dir=d)
            matched = registry.match_triggers("I need pytest for this")
            assert len(matched) > 0
            assert matched[0].name == "Python Testing"


class TestFileScope:
    """Tests for file_scope.py."""

    def test_can_modify_unlocked(self):
        from orchestrator.operations.file_scope import FileScopeManager

        fs = FileScopeManager()
        fs.target(["src/main.py", "src/utils.py"])
        assert fs.can_modify("src/main.py")
        assert not fs.can_modify("src/secret.py")

    def test_locked_files_cannot_modify(self):
        from orchestrator.operations.file_scope import FileScopeManager

        fs = FileScopeManager()
        fs.lock(["config.json", ".env"])
        assert not fs.can_modify("config.json")
        assert not fs.can_modify(".env")
        assert fs.can_modify("src/main.py")

    def test_filter_files(self):
        from orchestrator.operations.file_scope import FileScopeManager

        fs = FileScopeManager()
        fs.target(["src/app.py"])
        fs.lock(["src/secrets.py"])
        result = fs.filter_files(["src/app.py", "src/secrets.py", "src/other.py"])
        assert result == ["src/app.py"]


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: Integrations
# ═══════════════════════════════════════════════════════════════════════════


class TestSlashIntegrations:
    """Tests for slash_integrations.py."""

    def test_get_registered_integration(self):
        from orchestrator.slash_integrations import SlashIntegrationManager

        mgr = SlashIntegrationManager()
        integration = mgr.get("stripe")
        assert integration is not None
        assert integration.name == "stripe"
        assert integration.category == "payments"

    def test_list_by_category(self):
        from orchestrator.slash_integrations import SlashIntegrationManager

        mgr = SlashIntegrationManager()
        by_cat = mgr.list_by_category()
        assert "ai" in by_cat
        assert "chatgpt" in by_cat["ai"]

    def test_resolve_command(self):
        from orchestrator.slash_integrations import SlashIntegrationManager

        mgr = SlashIntegrationManager()
        result = mgr.resolve("chatgpt", "Explain quantum computing")
        assert result is not None
        assert result["command"] == "chatgpt"


class TestEntityRLS:
    """Tests for entity_rls.py."""

    def test_generate_row_level_policy(self):
        from orchestrator.analysis.entity_rls import EntityRLSManager

        mgr = EntityRLSManager()
        policy = mgr.generate_row_level("users", owner_field="user_id")
        assert policy.entity == "users"
        assert "user_id = auth.uid()" in policy.condition

    def test_generate_tenant_isolation(self):
        from orchestrator.analysis.entity_rls import EntityRLSManager

        mgr = EntityRLSManager()
        policy = mgr.generate_tenant_isolation("projects")
        assert "tenant_id = auth.tenant_id()" in policy.condition

    def test_to_sql_generation(self):
        from orchestrator.analysis.entity_rls import EntityRLSManager

        mgr = EntityRLSManager()
        mgr.generate_row_level("users", "id", "auth.uid()")
        sql = mgr.to_sql("users")
        assert "ENABLE ROW LEVEL SECURITY" in sql
        assert "CREATE POLICY" in sql

    def test_validate_missing_policies(self):
        from orchestrator.analysis.entity_rls import EntityRLSManager

        mgr = EntityRLSManager()
        issues = mgr.validate("nonexistent_table")
        assert len(issues) > 0


class TestMultiContext:
    """Tests for multi_context.py."""

    def test_create_and_list_threads(self):
        from orchestrator.operations.multi_context import MultiContextManager

        with tempfile.TemporaryDirectory() as d:
            mgr = MultiContextManager(project_dir=d)
            t1 = mgr.create_thread("Frontend work", "Working on UI")
            t2 = mgr.create_thread("Backend work", "Building API")
            threads = mgr.list_threads()
            assert len(threads) >= 2

    def test_switch_thread(self):
        from orchestrator.operations.multi_context import MultiContextManager

        with tempfile.TemporaryDirectory() as d:
            mgr = MultiContextManager(project_dir=d)
            t = mgr.create_thread("Testing")
            switched = mgr.switch_to(t.thread_id)
            assert switched is not None
            assert switched.name == "Testing"


# ═══════════════════════════════════════════════════════════════════════════
# Category 4 + 7: Design + UI Data Providers
# ═══════════════════════════════════════════════════════════════════════════


class TestDesignRegistry:
    """Tests for design_registry.py."""

    def test_seed_defaults(self):
        from orchestrator.design.design_registry import DesignRegistry

        registry = DesignRegistry()
        count = registry.seed_defaults()
        assert count > 0

    def test_register_and_get(self):
        from orchestrator.design.design_registry import DesignRegistry, RegistryItem

        registry = DesignRegistry()
        item = RegistryItem("hero", "component", "Hero section", files=["hero.tsx"])
        registry.register(item)
        found = registry.get("hero")
        assert found is not None
        assert found.name == "hero"

    def test_resolve_dependencies(self):
        from orchestrator.design.design_registry import DesignRegistry, RegistryItem

        registry = DesignRegistry()
        registry.register(RegistryItem("dialog", "component", dependencies=["button"]))
        registry.register(RegistryItem("button", "component"))
        deps = registry.resolve_dependencies("dialog")
        assert "button" in deps
        assert "dialog" in deps


class TestDesignSystemManager:
    """Tests for design_system.py."""

    def test_generate_css(self):
        from orchestrator.design_system import DesignSystem, ColorTokens, TypographyTokens

        ds = DesignSystem(
            name="Brand",
            version="1.0",
            colors=ColorTokens(primary="#ff0000", background="#ffffff"),
            typography=TypographyTokens(font_sans="Arial", font_mono="Courier"),
        )
        css = ds.to_css()
        assert "--color-primary: #ff0000" in css
        assert "--font-sans: Arial" in css

    def test_generate_tailwind(self):
        from orchestrator.design_system import DesignSystem, ColorTokens

        ds = DesignSystem(name="Brand", colors=ColorTokens(primary="#818cf8"))
        tw = ds.to_tailwind()
        assert "#818cf8" in tw

    def test_prompt_injection(self):
        from orchestrator.design_system import DesignSystem

        ds = DesignSystem(name="Brand")
        prompt = ds.to_prompt_injection()
        assert "Brand" in prompt
        assert "DESIGN SYSTEM" in prompt


class TestDiffViewProvider:
    """Tests for diff_view.py."""

    def test_diff_files(self):
        from orchestrator.generators.diff_view import DiffViewProvider

        provider = DiffViewProvider()
        diff = provider.diff_files("old line", "new line", "test.py")
        assert diff.filename == "test.py"
        assert diff.added_lines >= 0
        assert diff.removed_lines >= 0
        assert diff.unified_diff != ""

    def test_record_checkpoint_and_timeline(self, tmp_path):
        from orchestrator.generators.diff_view import DiffViewProvider

        provider = DiffViewProvider(project_dir=str(tmp_path))
        provider.record_checkpoint("after-auth")
        provider.record_version("v2")
        timeline = provider.get_timeline()
        assert len(timeline) == 2


class TestKnowledgeSidebar:
    """Tests for knowledge_sidebar.py."""

    def test_get_sections(self):
        from orchestrator.knowledge.knowledge_sidebar import KnowledgeSidebarData

        sidebar = KnowledgeSidebarData()
        sections = sidebar.get_sections()
        assert len(sections) == 5
        section_ids = {s["id"] for s in sections}
        assert section_ids == {"knowledge", "skills", "references", "commands", "context"}


class TestPlanReviewData:
    """Tests for plan_review_data.py."""

    def test_create_review_and_approve(self):
        from orchestrator.plan_review_data import PlanReviewData

        data = PlanReviewData()
        tasks = [{"id": "t1", "description": "Setup"}, {"id": "t2", "description": "Auth"}]
        data.create_review("plan-1", tasks)
        data.approve_task("plan-1", "t1")
        data.approve_task("plan-1", "t2")
        review = data.get_review("plan-1")
        assert review["status"] == "approved"

    def test_reject_task(self):
        from orchestrator.plan_review_data import PlanReviewData

        data = PlanReviewData()
        data.create_review("plan-1", [{"id": "t1", "description": "Task"}])
        data.reject_task("plan-1", "t1", "Too risky")
        review = data.get_review("plan-1")
        assert review["status"] == "rejected"

    def test_approve_all(self):
        from orchestrator.plan_review_data import PlanReviewData

        data = PlanReviewData()
        tasks = [{"id": "t1", "description": "A"}, {"id": "t2", "description": "B"}]
        data.create_review("plan-1", tasks)
        data.approve_all("plan-1")
        review = data.get_review("plan-1")
        assert review["status"] == "approved"


# ═══════════════════════════════════════════════════════════════════════════
# Category 9: Team Templates
# ═══════════════════════════════════════════════════════════════════════════


class TestTeamTemplates:
    """Tests for team_templates.py."""

    def test_create_template(self):
        from orchestrator.generators.team_templates import TeamTemplateManager

        mgr = TeamTemplateManager()
        t = mgr.create(
            "fastapi-starter",
            "FastAPI starter",
            entities=[{"name": "User"}],
            auth_config={"method": "jwt"},
        )
        assert t.name == "fastapi-starter"

    def test_template_to_dict_roundtrip(self):
        from orchestrator.generators.team_templates import TeamTemplate

        t = TeamTemplate(
            name="test", version="2.0", entities=[{"name": "Item"}], modules=["auth", "api"]
        )
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0"
        assert d["entities"] == [{"name": "Item"}]


# ═══════════════════════════════════════════════════════════════════════════
# Category 10+11: Config & Cost
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigSync:
    """Tests for config_sync.py."""

    def test_add_target(self):
        from orchestrator.config_sync import ConfigSync, SyncTarget

        sync = ConfigSync()
        sync.add_target(SyncTarget(name="supabase", type="supabase", url="https://xyz.supabase.co"))
        assert len(sync.targets) >= 1

    def test_sync_target_from_env(self):
        from orchestrator.config_sync import SyncTarget
        import os

        os.environ["TEST_SYNC_TYPE"] = "custom_api"
        os.environ["TEST_SYNC_URL"] = "https://api.example.com"
        target = SyncTarget.from_env("test", "TEST_SYNC")
        assert target.name == "test"
        assert target.type == "custom_api"
        assert target.url == "https://api.example.com"


class TestProgressCollector:
    """Tests for progress_collector.py."""

    def test_task_lifecycle(self):
        from orchestrator.analysis.progress_collector import ProgressCollector

        pc = ProgressCollector()
        pc.task_start("t1", "Setup project")
        pc.task_progress("t1", 0.5, "generating", "Code gen in progress")
        pc.task_complete("t1", score=0.95, cost=0.01, model="gpt-4o")
        status = pc.get_status()
        assert status["completed"] == 1
        assert status["total"] == 1
        assert status["cost_usd"] == 0.01

    def test_record_cost(self):
        from orchestrator.analysis.progress_collector import ProgressCollector

        pc = ProgressCollector()
        pc.record_cost("gpt-4o", 0.005, 100, 50, 245.0)
        status = pc.get_status()
        assert status["cost_usd"] == 0.005
        assert status["total_tokens"] == 150


class TestSystemDiagnostics:
    """Tests for system_diagnostics.py."""

    @pytest.mark.asyncio
    async def test_check_python(self):
        from orchestrator.operations.system_diagnostics import SystemDiagnostics

        diag = SystemDiagnostics()
        check = diag._check_python()
        assert check.name == "python"
        assert check.status == "ok"
        assert "Python" in check.message

    @pytest.mark.asyncio
    async def test_run_all(self):
        from unittest.mock import MagicMock, patch

        from orchestrator.operations.system_diagnostics import SystemDiagnostics

        diag = SystemDiagnostics()
        # The "tests" diagnostic shells out to `pytest --co` via subprocess; under
        # the test runner that recursively collects the whole suite and hangs.
        # Stub subprocess.run so the diagnostic stays fast and deterministic.
        fake = MagicMock(returncode=0, stdout="1 selected", stderr="")
        with patch("orchestrator.operations.system_diagnostics.subprocess.run", return_value=fake):
            report = await diag.run_all()
        assert report.overall_status in ("ok", "warning")
        assert len(report.checks) >= 4


class TestCrossProjectReferencer:
    """Tests for cross_project.py."""

    def test_index_project(self):
        from orchestrator.learning.cross_project import CrossProjectReferencer

        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproject"
            proj.mkdir()
            (proj / "app.py").write_text("print('hello')")
            ref = CrossProjectReferencer(projects_root=d)
            result = ref.index("test-proj", str(proj), "Test project")
            assert result is not None
            assert "app.py" in result.files

    def test_resolve_at_reference(self):
        from orchestrator.learning.cross_project import CrossProjectReferencer

        ref = CrossProjectReferencer()
        ref._projects["demo"] = type(
            "P", (), {"path": "", "files": {"api.py": "def run(): pass"}, "summary": "Demo project"}
        )()
        result = ref.resolve("See @demo/api.py for reference")
        assert "def run" in result

    def test_find_file(self):
        from orchestrator.learning.cross_project import CrossProjectReferencer

        with tempfile.TemporaryDirectory() as d:
            proj = Path(d) / "myproject"
            proj.mkdir()
            (proj / "utils.py").write_text("def helper(): pass")
            ref = CrossProjectReferencer(projects_root=d)
            ref.index("myproject", str(proj))
            content = ref.find_file("myproject", "utils.py")
            assert content == "def helper(): pass"


class TestTwoWayGitSync:
    """Tests for git_sync.py build_context."""

    def test_build_context_empty(self):
        from orchestrator.git_sync import TwoWayGitSync

        sync = TwoWayGitSync()
        ctx = sync.build_context([])
        assert "No changes" in ctx


class TestSiteManager:
    """Tests for site_manager.py."""

    def test_unpublished_changes_new_project(self):
        from orchestrator.site_manager import SiteManager

        sm = SiteManager()
        assert sm.unpublished_changes  # new project, never published

    def test_publish_and_version(self):
        from orchestrator.site_manager import SiteManager

        sm = SiteManager()
        ps = sm.publish(version="1.0.0", description="Initial release")
        assert ps.version == "1.0.0"
        assert ps.files_count >= 0


class TestProjectCopier:
    """Tests for project_copier.py."""

    def test_duplicate_project(self):
        from orchestrator.project_copier import ProjectCopier

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            (src / "app.py").write_text("print('hello')")
            copier = ProjectCopier(workspace=d)
            cp = copier.duplicate(str(src), "Experiment")
            assert cp is not None
            assert cp.description == "Experiment"


# ═══════════════════════════════════════════════════════════════════════════
# Existing regression tests (BUG-001..004)
# ═══════════════════════════════════════════════════════════════════════════

# Note: Tests for BUG-001 through BUG-004 are in tests/test_bug_fixes_v2.py
# and remain unchanged. They verify:
#   - Budget lock atomicity
#   - CircuitBreaker HALF_OPEN probe race
#   - Evaluator latency tracking
#   - Telemetry store null guard
