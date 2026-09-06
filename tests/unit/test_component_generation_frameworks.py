"""
Svelte/React component generation for generated websites.

`design/component_registry.py` is a 524-line curated component registry —
"multi-source component library with quality scoring and automated selection
based on design system compatibility" — that was entirely dead: it imports
`ComponentSource` and `ComponentSpec` from `design_system`, and neither was
defined anywhere in the repository (hunt T17 confirmed `ComponentSource` was
referenced 17 times and defined zero times). `website_generator.py` therefore
always fell through to a `_FakeRegistry` returning four bare section names
with no component guidance and no design tokens at all.

Defining those two types revives the registry. On top of that, component
selection and the generated OUTPUT contract are now framework-aware, so a
Svelte target is steered to Svelte-native libraries and asked for a `.svelte`
single-file component instead of a React `.tsx` one.
"""

from __future__ import annotations

import pytest

from orchestrator.design_system import (
    ComponentSource,
    ComponentSpec,
    DesignSystem,
    normalize_framework,
    sources_for_framework,
)

pytestmark = pytest.mark.unit


# --- the types that were missing --------------------------------------------


def test_registry_imports_at_all():
    """Regression: this import raised ImportError before the types existed."""
    from orchestrator.design.component_registry import COMPONENT_LIBRARY, get_registry

    assert get_registry() is not None
    assert COMPONENT_LIBRARY, "curated component library must not be empty"
    # Every curated entry must be a real ComponentSpec, not a stub.
    for section, specs in COMPONENT_LIBRARY.items():
        for spec in specs:
            assert isinstance(spec, ComponentSpec), f"{section}/{spec} is not a ComponentSpec"


def test_generator_no_longer_falls_back_to_the_fake_registry():
    """The live generator's lazy import must now resolve to the real registry."""
    from orchestrator.design.component_registry import get_registry as real
    from orchestrator.generators.website_generator import _get_registry

    assert _get_registry() is real


# --- framework normalization -------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("react", "react"),
        ("next.js", "react"),
        ("nextjs", "react"),
        ("svelte", "svelte"),
        ("sveltekit", "svelte"),
        ("SvelteKit", "svelte"),
        ("html", "html"),
        ("", "html"),
    ],
)
def test_normalize_framework(value, expected):
    assert normalize_framework(value) == expected


def test_svelte_and_react_sources_are_disjoint():
    react, svelte = sources_for_framework("react"), sources_for_framework("svelte")
    # CUSTOM is framework-agnostic and belongs to both; nothing else may be.
    assert react & svelte == {ComponentSource.CUSTOM}
    assert ComponentSource.BITS_UI in svelte and ComponentSource.BITS_UI not in react
    assert ComponentSource.REACTBITS in react and ComponentSource.REACTBITS not in svelte


# --- selection is framework-aware -------------------------------------------


@pytest.mark.asyncio
async def test_svelte_build_selects_svelte_native_components():
    from orchestrator.design.component_registry import get_registry

    registry, ds = get_registry(), DesignSystem(tone="modern")
    picked = await registry.select_components(
        page_type="landing",
        design_system=ds,
        sections_needed=["hero", "features"],
        framework="svelte",
    )
    assert [c.source for c in picked] == [ComponentSource.BITS_UI, ComponentSource.SVELTEBITS], (
        "a Svelte build must not be steered toward React-only libraries; got "
        f"{[c.source.value for c in picked]}"
    )


@pytest.mark.asyncio
async def test_react_build_still_selects_react_components():
    """No-regression: the pre-existing React path is unchanged."""
    from orchestrator.design.component_registry import get_registry

    picked = await get_registry().select_components(
        page_type="landing",
        design_system=DesignSystem(tone="modern"),
        sections_needed=["hero", "features"],
        framework="react",
    )
    assert all(c.supports_framework("react") for c in picked)
    assert not any(c.source == ComponentSource.BITS_UI for c in picked)


@pytest.mark.asyncio
async def test_section_without_a_framework_native_component_still_returns_one():
    """Falling back beats returning nothing — the page still needs the section."""
    from orchestrator.design.component_registry import get_registry

    picked = await get_registry().select_components(
        page_type="landing",
        design_system=DesignSystem(),
        sections_needed=["pricing"],  # no Svelte-native pricing component curated
        framework="svelte",
    )
    assert len(picked) == 1


# --- the output contract handed to the model --------------------------------


@pytest.mark.parametrize(
    ("framework", "must_contain", "must_not_contain"),
    [
        ("svelte", "Svelte 5", "React"),
        ("sveltekit", "Svelte 5", "React"),
        ("react", "React/Next.js", "Svelte"),
        ("next.js", "React/Next.js", "Svelte"),
        ("html", "HTML section", "React"),
    ],
)
def test_component_prompt_targets_the_right_framework(framework, must_contain, must_not_contain):
    from orchestrator.design.component_registry import COMPONENT_LIBRARY, get_registry

    prompt = get_registry().get_component_prompt(
        COMPONENT_LIBRARY["hero"][0], DesignSystem(), framework=framework
    )
    assert must_contain in prompt
    assert must_not_contain not in prompt


@pytest.mark.parametrize(
    ("framework", "ext"),
    [
        ("html", ".html"),
        ("react", ".tsx"),
        ("next.js", ".tsx"),
        ("svelte", ".svelte"),
        ("sveltekit", ".svelte"),
    ],
)
def test_component_file_extension(framework, ext):
    from orchestrator.generators.website_generator import _component_extension

    assert _component_extension(framework) == ext


def test_svelte_is_selectable_from_the_cli():
    import argparse

    from orchestrator.commands.website import register

    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["website", "-d", "a site", "--framework", "svelte"])
    assert args.framework == "svelte"


# --- scoring -----------------------------------------------------------------


def test_compatibility_score_is_bounded_and_rewards_accessibility():
    ds = DesignSystem(tone="modern")
    accessible = ComponentSpec(
        name="a",
        source=ComponentSource.SHADCN,
        category="hero",
        has_aria_labels=True,
        keyboard_navigable=True,
        responsive=True,
    )
    inaccessible = ComponentSpec(
        name="b",
        source=ComponentSource.SHADCN,
        category="hero",
        has_aria_labels=False,
        keyboard_navigable=False,
        responsive=False,
    )
    assert 0.0 <= inaccessible.compatibility_score(ds) <= accessible.compatibility_score(ds) <= 1.0
    assert accessible.compatibility_score(ds) > inaccessible.compatibility_score(ds)


def test_default_component_handles_a_plain_string_tone():
    """`DesignSystem.tone` is a plain str; the registry used to call `.value`."""
    from orchestrator.design.component_registry import get_registry

    spec = get_registry()._create_default_component("newsletter", DesignSystem(tone="luxury"))
    assert spec.animation_style == "luxury"
    assert spec.source == ComponentSource.CUSTOM
