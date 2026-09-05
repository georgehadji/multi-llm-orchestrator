"""
Regression tests for the TSX component templates.

The bug these lock down: CONTACT_FORM_TEMPLATE and AUTH_TEMPLATE are TSX source
full of literal JavaScript braces (`import { useState } from 'react'`), and were
rendered with ``str.format()``. ``str.format`` treats every ``{...}`` as a
replacement field, so `{ useState }` was looked up as the key `' useState '` and
raised KeyError. Any site whose sections included `contact` therefore failed
generation outright and fell back to a template page — silently, because the
build reported success anyway.

Both templates are free of `$`, so ``string.Template`` substitution is safe and
removes the entire bug class rather than escaping braces one at a time.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def generator():
    from orchestrator.generators.website_generator import WebsiteGenerator

    return WebsiteGenerator.__new__(WebsiteGenerator)


@pytest.fixture
def design_system():
    from orchestrator.design_system import DesignSystem

    return DesignSystem(tone="luxury")


@pytest.mark.unit
@pytest.mark.parametrize("builder", ["_build_contact_form", "_build_auth_component"])
class TestComponentTemplatesRender:
    def test_renders_without_raising(self, generator, design_system, builder):
        out = getattr(generator, builder)("section", "Get in touch", design_system)
        assert isinstance(out, str) and out.strip()

    def test_preserves_literal_js_braces(self, generator, design_system, builder):
        out = getattr(generator, builder)("section", "Get in touch", design_system)
        assert "import { useState } from 'react'" in out

    def test_substitutes_the_headline(self, generator, design_system, builder):
        out = getattr(generator, builder)("section", "Book your visit", design_system)
        assert "Book your visit" in out

    def test_leaves_no_unsubstituted_placeholders(self, generator, design_system, builder):
        out = getattr(generator, builder)("section", "Get in touch", design_system)
        for token in ("$headline", "$primary", "$surface_alt", "$border", "$font_body"):
            assert token not in out, f"{token} was not substituted"


@pytest.mark.unit
class TestContactSectionEndToEnd:
    @pytest.mark.asyncio
    async def test_generate_with_contact_section_succeeds(self, tmp_path):
        """A 'contact' section used to fail the whole build with KeyError."""
        from orchestrator.design_system import DesignSystem
        from orchestrator.generators.website_generator import (
            ClientInfo,
            WebsiteConfig,
            WebsiteGenerator,
        )

        result = await WebsiteGenerator().generate(
            design_system=DesignSystem(tone="luxury"),
            client_info=ClientInfo(
                name="Gadini", industry="barbershop", description="A barber shop"
            ),
            config=WebsiteConfig(framework="html", sections=["hero", "contact", "footer"]),
            output_dir=tmp_path,
        )
        assert result.success is True, f"generation failed: {result.errors}"
        assert not any("useState" in e for e in result.errors)
