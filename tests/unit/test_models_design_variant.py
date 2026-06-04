"""Unit tests for DesignVariant enum and Task.design_variant field."""

import pytest


@pytest.mark.unit
def test_design_variant_enum_values():
    from orchestrator.models import DesignVariant

    assert DesignVariant.DEFAULT == "default"
    assert DesignVariant.SOFT == "soft"
    assert DesignVariant.MINIMALIST == "minimalist"
    assert DesignVariant.BRUTALIST == "brutalist"
    assert DesignVariant.REDESIGN == "redesign"


@pytest.mark.unit
def test_design_variant_is_str_enum():
    from orchestrator.models import DesignVariant

    assert isinstance(DesignVariant.DEFAULT, str)


@pytest.mark.unit
def test_task_design_variant_defaults_none():
    from orchestrator.models import Task, TaskType

    task = Task(id="t1", type=TaskType.CODE_GEN, prompt="build a UI")
    assert task.design_variant is None


@pytest.mark.unit
def test_task_design_variant_can_be_set():
    from orchestrator.models import DesignVariant, Task, TaskType

    task = Task(id="t1", type=TaskType.CODE_GEN, prompt="build a UI", design_variant=DesignVariant.BRUTALIST)
    assert task.design_variant == DesignVariant.BRUTALIST


@pytest.mark.unit
def test_task_is_pure_data_no_methods():
    from orchestrator.models import Task, TaskType

    task = Task(id="t1", type=TaskType.CODE_GEN, prompt="x")
    # pure data — no generate() or business-logic method
    assert not hasattr(task, "generate")
    assert not hasattr(task, "execute")
