"""
Regression tests for ServiceContainer stage discovery.

The bug: inside ServiceContainer.build (a @classmethod, so `cls` is
ServiceContainer), the pipeline was assembled with

    stages = [cls._build_stage(cls, ...) for cls in discovered]

The comprehension's `for cls in discovered` SHADOWS the classmethod's `cls`,
so `cls._build_stage` was looked up on a discovered stage class rather than on
ServiceContainer. `_build_stage` is a @staticmethod of ServiceContainer, so
every build raised:

    AttributeError: type object 'ConstitutionGate' has no attribute '_build_stage'

_discover_stages() returns 11 stages via the hardcoded fallback list, so this
branch is always taken — ServiceContainer.build() could not construct at all.
mypy reported it as `container.py:780: "type" has no attribute "_build_stage"`,
but the Type Check job also reported 1422 errors from unrelated layers, so the
signal was invisible.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_discovered_stages_do_not_expose_build_stage():
    """Pins the precondition: the shadowed lookup could never have worked."""
    from orchestrator.engine_core.container import _discover_stages

    discovered = _discover_stages()
    assert discovered, "expected the fallback entry-point list to yield stages"
    assert not hasattr(discovered[0], "_build_stage"), (
        "a stage class must not define _build_stage; if it does, this test's "
        "premise (and the fix) needs revisiting"
    )


@pytest.mark.unit
def test_build_stage_is_a_staticmethod_of_the_container():
    from orchestrator.engine_core.container import ServiceContainer

    assert hasattr(ServiceContainer, "_build_stage")
    assert isinstance(
        ServiceContainer.__dict__["_build_stage"], staticmethod
    ), "_build_stage is expected to be a @staticmethod taking the stage class"


@pytest.mark.unit
def test_container_builds_with_discovered_stages(monkeypatch):
    """ServiceContainer.build() must actually construct."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")
    from orchestrator.budget import Budget
    from orchestrator.engine_core.container import ServiceContainer

    container = ServiceContainer.build(budget=Budget(max_usd=1.0))
    assert container is not None


@pytest.mark.unit
def test_built_pipeline_has_stages(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")
    from orchestrator.budget import Budget
    from orchestrator.engine_core.container import ServiceContainer

    container = ServiceContainer.build(budget=Budget(max_usd=1.0))
    pipeline = getattr(container, "pipeline", None)
    if pipeline is None:  # pipeline is optional wiring in some configurations
        pytest.skip("container exposes no pipeline attribute in this configuration")
    # TaskPipeline keeps its stage list private.
    stages = getattr(pipeline, "stages", None) or getattr(pipeline, "_stages", None)
    assert stages, "a built pipeline must carry stages"


@pytest.mark.unit
def test_every_discovered_stage_can_actually_be_built():
    """Each stage's build_kwargs must supply every required __init__ argument.

    DesignCritiqueStage returned `{}` while its __init__ required `client`, so
    repairing the `cls` shadowing above simply moved the failure from
    AttributeError to TypeError. This invariant catches that class of mismatch
    directly, without needing a full container build.
    """
    import inspect

    from orchestrator.engine_core.container import _discover_stages

    sentinel_deps = {
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

    broken: list[str] = []
    for stage_cls in _discover_stages():
        builder = getattr(stage_cls, "build_kwargs", None)
        kwargs = builder(**sentinel_deps) if builder is not None else {}
        params = list(inspect.signature(stage_cls.__init__).parameters.values())[1:]
        required = {
            p.name
            for p in params
            if p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        missing = required - set(kwargs)
        if missing:
            broken.append(f"{stage_cls.__name__}: build_kwargs omits {sorted(missing)}")

    assert not broken, "stage(s) cannot be constructed by _build_stage:\n  " + "\n  ".join(broken)
