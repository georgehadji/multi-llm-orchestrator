"""senior-frontend skill is bundled and injected for web-frontend tasks."""

from unittest.mock import MagicMock

import pytest


def _task(prompt, target_path=""):
    from orchestrator.models import Task, TaskType

    return Task(id="t1", type=TaskType.CODE_GEN, prompt=prompt, target_path=target_path)


@pytest.mark.unit
def test_senior_frontend_is_bundled_and_not_truncated():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    assert loader.is_bundled("senior_frontend")
    text = loader.load("senior_frontend")
    # last section of the upstream file: proves the size cap did not cut it
    assert "Red flags: stop and fix" in text
    assert "truncated for token efficiency" not in text


@pytest.mark.unit
def test_prefix_includes_senior_frontend_for_frontend_task():
    from orchestrator.design.taste_skill_service import TasteSkillService

    prefix = TasteSkillService().build_prefix(_task("build a landing page in HTML", "index.html"))

    assert "<senior_frontend>" in prefix
    assert "design every state" in prefix.lower() or "List the states" in prefix
    # bundled references/scripts are not shipped: the model must be told so
    assert "not available" in prefix


@pytest.mark.unit
def test_prefix_omits_senior_frontend_for_backend_task():
    from orchestrator.design.taste_skill_service import TasteSkillService

    prefix = TasteSkillService().build_prefix(_task("write a fastapi backend", "app.py"))
    assert prefix == ""


@pytest.mark.unit
def test_prefix_omits_senior_frontend_when_flag_disabled():
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = False
    prefix = TasteSkillService(flags=flags).build_prefix(
        _task("build a landing page", "index.html")
    )
    assert "<senior_frontend>" not in prefix
