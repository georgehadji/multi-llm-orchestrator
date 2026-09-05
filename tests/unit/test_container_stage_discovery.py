"""
Regression tests for ServiceContainer pipeline-stage discovery.

Four defects, each hidden behind the previous one:

1. `stages = [cls._build_stage(cls, ...) for cls in discovered]` — inside a
   @classmethod, `cls` is ServiceContainer (which owns the _build_stage
   staticmethod), but the comprehension's `for cls in discovered` SHADOWED it,
   so the lookup landed on a stage class:
       AttributeError: type object 'ConstitutionGate' has no attribute '_build_stage'

2. DesignCritiqueStage.build_kwargs returned {} while its __init__ requires
   `client`, so the build then raised TypeError.

3. TaskContextEnricher (a helper) and MAPElitesPipeline (a BasePipeline) were
   listed as stages but define no `process()`, which TaskPipeline awaits — so
   every task failed once discovery worked.

4. Three entries in pyproject.toml's `orchestrator.pipeline.stages` group named
   classes that do not exist (ContextEnricherStage, SelfConsistencyStage,
   MapElitesStage). Where the package is installed with entry points registered
   — i.e. CI — the first ep.load() raised, _discover_stages()'s bare
   `except Exception` swallowed it at debug level, and discovery returned None
   on EVERY run. Entry-point discovery had never once worked in CI.

Defect 4 is why these tests must not depend on `_discover_stages()` returning
stages: whether it uses the entry-point group or the hardcoded fallback is an
environment detail. They validate BOTH declared sources directly instead.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SENTINEL_DEPS = {
    "client": object(),
    "budget": object(),
    "selector": object(),
    "vs_sampler": object(),
    "lsp_validator": object(),
    "evaluator": object(),
    "ara": object(),
    "ara_strategy": object(),
    "validator": object(),
}


def _load(target: str) -> type:
    """Import a ``module.path:ClassName`` target."""
    module_path, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_path), class_name)


def _declared_entry_point_targets() -> dict[str, str]:
    if sys.version_info < (3, 11):  # pragma: no cover
        pytest.skip("tomllib requires Python 3.11+")
    import tomllib

    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return config["project"]["entry-points"]["orchestrator.pipeline.stages"]


def _fallback_targets() -> list[str]:
    from orchestrator.engine_core.container import _FALLBACK_ENTRY_POINTS

    return list(_FALLBACK_ENTRY_POINTS)


def _all_declared_targets() -> list[str]:
    return sorted({*_declared_entry_point_targets().values(), *_fallback_targets()})


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")


@pytest.mark.unit
class TestStageDeclarations:
    def test_every_declared_entry_point_imports(self):
        """Defect 4: three entry points named classes that do not exist."""
        broken = []
        for name, target in _declared_entry_point_targets().items():
            try:
                _load(target)
            except Exception as exc:  # noqa: BLE001 — report all, not the first
                broken.append(f"{name} -> {target}: {type(exc).__name__}: {exc}")
        assert not broken, "unimportable entry point(s):\n  " + "\n  ".join(broken)

    def test_every_fallback_entry_imports(self):
        broken = []
        for target in _fallback_targets():
            try:
                _load(target)
            except Exception as exc:  # noqa: BLE001
                broken.append(f"{target}: {type(exc).__name__}: {exc}")
        assert not broken, "unimportable fallback entr(y/ies):\n  " + "\n  ".join(broken)

    def test_entry_points_and_fallback_list_agree(self):
        """The two sources must not drift; the environment picks between them."""
        declared = set(_declared_entry_point_targets().values())
        fallback = set(_fallback_targets())
        assert declared == fallback, (
            f"only in pyproject entry points: {sorted(declared - fallback)}; "
            f"only in _FALLBACK_ENTRY_POINTS: {sorted(fallback - declared)}"
        )

    def test_every_declared_stage_implements_process(self):
        """Defect 3: TaskPipeline awaits stage.process(ctx)."""
        missing = [t for t in _all_declared_targets() if not hasattr(_load(t), "process")]
        assert not missing, (
            f"declared stage(s) without process(): {missing}. "
            f"TaskPipeline awaits stage.process(ctx), so these break every task."
        )

    def test_every_declared_stage_can_be_constructed_by_build_stage(self):
        """Defect 2: build_kwargs must supply every required __init__ argument."""
        broken = []
        for target in _all_declared_targets():
            stage_cls = _load(target)
            builder = getattr(stage_cls, "build_kwargs", None)
            kwargs = builder(**_SENTINEL_DEPS) if builder is not None else {}
            params = list(inspect.signature(stage_cls.__init__).parameters.values())[1:]
            required = {
                p.name
                for p in params
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            }
            if required - set(kwargs):
                broken.append(
                    f"{stage_cls.__name__}: build_kwargs omits {sorted(required - set(kwargs))}"
                )
        assert not broken, "stage(s) _build_stage cannot construct:\n  " + "\n  ".join(broken)


@pytest.mark.unit
class TestBuildStageWiring:
    def test_build_stage_is_a_staticmethod_of_the_container(self):
        """Defect 1: it belongs to ServiceContainer, not to a stage class."""
        from orchestrator.engine_core.container import ServiceContainer

        assert isinstance(ServiceContainer.__dict__["_build_stage"], staticmethod)

    def test_no_declared_stage_defines_build_stage(self):
        """If one did, the old shadowed `cls._build_stage` lookup could 'work'."""
        offenders = [t for t in _all_declared_targets() if hasattr(_load(t), "_build_stage")]
        assert not offenders, f"stage(s) unexpectedly defining _build_stage: {offenders}"

    def test_discover_stages_returns_usable_stages_or_none(self):
        """None is a legitimate 'discovery unavailable' signal the caller handles."""
        from orchestrator.engine_core.container import _discover_stages

        discovered = _discover_stages()
        if discovered is None:
            pytest.skip("entry-point discovery unavailable in this environment")
        assert discovered, "discovery returned an empty list rather than None"
        assert all(hasattr(c, "process") for c in discovered)


@pytest.mark.unit
class TestContainerBuilds:
    def test_container_builds(self):
        from orchestrator.budget import Budget
        from orchestrator.engine_core.container import ServiceContainer

        assert ServiceContainer.build(budget=Budget(max_usd=1.0)) is not None

    def test_built_pipeline_has_stages(self):
        from orchestrator.budget import Budget
        from orchestrator.engine_core.container import ServiceContainer

        container = ServiceContainer.build(budget=Budget(max_usd=1.0))
        pipeline = getattr(container, "pipeline", None)
        if pipeline is None:
            pytest.skip("container exposes no pipeline attribute in this configuration")
        # TaskPipeline keeps its stage list private.
        stages = getattr(pipeline, "stages", None) or getattr(pipeline, "_stages", None)
        assert stages, "a built pipeline must carry stages"
