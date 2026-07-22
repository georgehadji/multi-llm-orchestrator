"""Unit tests for hallmark_selector."""

import pytest

pytestmark = pytest.mark.unit

from orchestrator.design.hallmark_selector import HallmarkSelector
from orchestrator.design.catalogs import MACROSTRUCTURES, THEMES
from orchestrator.design.design_log import DesignLog, DesignLogEntry
from orchestrator.models import Genre


def _entry(
    macro: str = "bento_grid",
    theme: str = "lumen",
    genre: str = "modern-minimal",
    nav: str = "N1",
    footer: str = "Ft1",
) -> DesignLogEntry:
    from datetime import datetime, timezone

    return DesignLogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        macrostructure=macro,
        theme=theme,
        genre=genre,
        nav_archetype=nav,
        footer_archetype=footer,
    )


@pytest.mark.unit
class TestHallmarkSelector:
    def test_select_macrostructure_returns_valid(self, tmp_path):
        selector = HallmarkSelector()
        log = DesignLog(tmp_path)
        macro = selector.select_macrostructure("Build a SaaS landing page", "saas", log)
        assert macro.slug in MACROSTRUCTURES
        assert macro.heading  # has heading rule

    def test_select_macrostructure_diversifies(self, tmp_path):
        selector = HallmarkSelector()
        log = DesignLog(tmp_path)
        # Force recently-used
        log.append(_entry(macro="bento_grid"))
        log.append(_entry(macro="bento_grid"))
        log.append(_entry(macro="bento_grid"))
        macro = selector.select_macrostructure("Build a SaaS landing page", "saas", log)
        assert macro.slug != "bento_grid"

    def test_select_theme_returns_valid(self, tmp_path):
        selector = HallmarkSelector()
        log = DesignLog(tmp_path)
        macro = selector.select_macrostructure("Build a landing page", "saas", log)
        theme = selector.select_theme("Build a landing page", macro, log)
        assert theme.slug in {t.slug for t in THEMES.values()}
        assert theme.genre

    def test_select_nav_returns_valid_for_genre(self, tmp_path):
        selector = HallmarkSelector()
        log = DesignLog(tmp_path)
        nav = selector.select_nav(Genre.EDITORIAL, log)
        assert nav is not None
        assert nav.category == "nav"

    def test_select_footer_returns_valid_for_genre(self, tmp_path):
        selector = HallmarkSelector()
        log = DesignLog(tmp_path)
        footer = selector.select_footer(Genre.EDITORIAL, log)
        assert footer is not None
        assert footer.category == "footer"

    def test_select_section_archetypes_returns_list(self, tmp_path):
        selector = HallmarkSelector()
        macro = selector.select_macrostructure("Build a landing page", "saas", None)
        archetypes = selector.select_section_archetypes(macro, ["hero", "features", "cta"])
        assert len(archetypes) == 3
        codes = [a.code for a, _ in archetypes]
        assert len(codes) == len(set(codes)), "No duplicate archetypes"

    def test_infer_domain_from_brief(self):
        selector = HallmarkSelector()
        # "blog" hits fallback → "editorial"
        assert selector.infer_domain("Build a blog") == "editorial"
        # "shop" hits first-loop domain key
        assert selector.infer_domain("Create a shop") == "shop"
        # "saas" hits first-loop domain key
        assert selector.infer_domain("Build a SaaS landing page") == "saas"
