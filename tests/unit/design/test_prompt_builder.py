"""Unit tests for prompt_builder."""

import pytest

from orchestrator.design.prompt_builder import HallmarkPromptBuilder
from orchestrator.design.catalogs import MACROSTRUCTURES, THEMES, ARCHETYPES


@pytest.mark.unit
class TestHallmarkPromptBuilder:
    def test_macrostructure_block_contains_heading_rule(self):
        builder = HallmarkPromptBuilder()
        macro = MACROSTRUCTURES["manifesto"]
        block = builder.build_macrostructure_block(macro)
        assert "Selected Macrostructure" in block
        assert macro.heading in block
        assert "Nav archetype" in block

    def test_theme_block_contains_palette(self):
        builder = HallmarkPromptBuilder()
        theme = THEMES["midnight"]
        block = builder.build_theme_block(theme)
        assert "Selected Theme" in block
        assert "Paper" in block
        assert "Ink" in block
        assert "Accent" in block
        assert theme.font_display in block
        assert "Signature moves" in block
        assert "Anti-patterns" in block

    def test_nav_block(self):
        builder = HallmarkPromptBuilder()
        nav = ARCHETYPES["N1"]
        block = builder.build_nav_block(nav)
        assert nav.name in block
        assert nav.code in block
        assert "Variation knobs" in block

    def test_footer_block(self):
        builder = HallmarkPromptBuilder()
        footer = ARCHETYPES["Ft1"]
        block = builder.build_footer_block(footer)
        assert footer.name in block
        assert footer.code in block

    def test_component_scope_block_has_8_states(self):
        builder = HallmarkPromptBuilder()
        block = builder.build_component_block()
        assert "default" in block
        assert "hover" in block
        assert "focus-visible" in block
        assert "active" in block
        assert "disabled" in block
        assert "loading" in block
        assert "error" in block
        assert "success" in block
        assert "8-state demo wrapper" in block

    def test_self_critique_block_has_axes(self):
        builder = HallmarkPromptBuilder()
        block = builder.build_self_critique_block()
        assert "Philosophy" in block
        assert "Hierarchy" in block
        assert "Execution" in block
        assert "Specificity" in block
        assert "Restraint" in block
        assert "Variety" in block
        assert "pre-emit critique" in block

    def test_full_prefix_component_scope(self):
        builder = HallmarkPromptBuilder()
        prefix = builder.build_full_prefix(None, None, None, None, scope="component")
        assert "Component Scope" in prefix
        assert "default" in prefix
        assert "Pre-emit Self-Critique" in prefix

    def test_full_prefix_page_scope(self):
        builder = HallmarkPromptBuilder()
        macro = MACROSTRUCTURES["bento_grid"]
        theme = THEMES["lumen"]
        nav = ARCHETYPES["N1"]
        footer = ARCHETYPES["Ft1"]
        prefix = builder.build_full_prefix(macro, theme, nav, footer, scope="page")
        assert "Selected Macrostructure" in prefix
        assert "Selected Theme" in prefix
        assert "Selected Nav" in prefix
        assert "Selected Footer" in prefix
        assert "Pre-emit Self-Critique" in prefix
