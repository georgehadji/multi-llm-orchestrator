"""
Unit tests for Ponytail Persona behavior customization.
Verifies Milestone 1 implementation: addition of PONYTAIL persona, enum values,
and configurations to the active persona.py system.
"""

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skip(
        reason="PONYTAIL persona mode (PONYTAIL_INTEGRATION_PLAN.md milestone 1) is "
        "prototyped in packages/orchestrator-persona but not installed as a project "
        "dependency, and not merged into the production orchestrator/persona.py module "
        "that this test suite otherwise exercises. Wiring one or the other in is an "
        "architectural-class change (see orchestrator-change-control) — un-skip once "
        "that decision is made, not as a side effect of a CI fix."
    ),
]

from orchestrator.persona import (  # noqa: E402
    PersonaMode,
    get_persona_manager,
    PersonaSettings,
    Persona,
)


def test_persona_mode_enum_contains_ponytail():
    """Verify that PersonaMode Enum includes PONYTAIL."""
    assert hasattr(PersonaMode, "PONYTAIL")
    assert PersonaMode.PONYTAIL.value == "ponytail"


def test_persona_presets_contain_ponytail():
    """Verify that Ponytail preset settings are correct."""
    manager = get_persona_manager()
    settings = manager.get_preset_settings(PersonaMode.PONYTAIL)

    assert isinstance(settings, PersonaSettings)
    assert settings.temperature == 0.1
    assert settings.max_tokens == 2048
    assert settings.require_tests is False
    assert settings.require_documentation is False
    assert "Ponytail" in settings.system_prompt_addition
    assert "YAGNI" in settings.system_prompt_addition


def test_persona_manager_get_persona():
    """Verify that PersonaManager can assign and retrieve Ponytail persona."""
    manager = get_persona_manager()
    project_id = "test_project_ponytail"

    try:
        manager.set_persona(project_id, PersonaMode.PONYTAIL)
        assert manager.get_persona_mode(project_id) == PersonaMode.PONYTAIL

        persona = manager.get_persona(project_id)
        assert isinstance(persona, Persona)
        assert persona.mode == PersonaMode.PONYTAIL
        assert persona.get_temperature() == 0.1

        system_prompt = persona.get_system_prompt("Base Prompt Instructions.")
        assert "Base Prompt Instructions." in system_prompt
        assert "Ponytail" in system_prompt

    finally:
        manager.clear_persona(project_id)


@pytest.mark.xfail(
    strict=True,
    reason="Ponytail prose truncation not implemented yet — "
    "PONYTAIL_INTEGRATION_PLAN.md milestone 3 (post-processing). "
    "De-xfail when CodePostProcessor gains ponytail_prose_truncated.",
)
def test_ponytail_prose_truncation_post_processor():
    """Verify that CodePostProcessor truncates verbose trailing prose in Ponytail mode."""
    from orchestrator.quality.code_post_processor import CodePostProcessor

    manager = get_persona_manager()
    original_default = manager._default_mode

    try:
        # Set default to PONYTAIL to activate post-processor check
        manager._default_mode = PersonaMode.PONYTAIL

        processor = CodePostProcessor()

        mock_output = (
            "Here is the code:\n"
            "```python\n"
            "def hello():\n"
            "    print('hello')\n"
            "```\n"
            "This is line 1 of verbose explanation.\n"
            "This is line 2 of verbose explanation.\n"
            "This is line 3 of verbose explanation.\n"
            "This is line 4 of verbose explanation.\n"
            "This is line 5 of verbose explanation."
        )

        processed_output = processor.process(mock_output, "test_file.py")

        # Verify it has been truncated to max 3 lines after code block
        assert "ponytail_prose_truncated" in processor.fixes_applied
        assert "line 1" in processed_output
        assert "line 3" in processed_output
        assert "line 4" not in processed_output
        assert "line 5" not in processed_output

    finally:
        manager._default_mode = original_default
