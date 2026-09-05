"""WF-100 standard: the check catalogue must satisfy its own arithmetic.

These are invariants of the *standard*, not of any one audit. If the catalogue
stops summing to 100, or two checks collide on an id, every score computed
downstream is quietly wrong — so they are asserted here rather than trusted.
"""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.standard import (
    CATEGORY_WEIGHTS,
    STANDARD,
    Category,
    Evidence,
    Level,
    checks_for,
)

pytestmark = pytest.mark.unit


class TestCatalogueArithmetic:
    def test_total_is_one_hundred_points(self):
        assert sum(c.points for c in STANDARD) == 100

    def test_catalogue_holds_one_hundred_checks(self):
        # WF-100: a hundred checks, one point each. Every point traceable to a
        # single verifiable claim rather than a weighted bundle.
        assert len(STANDARD) == 100

    def test_every_check_is_worth_one_point(self):
        assert {c.points for c in STANDARD} == {1}

    @pytest.mark.parametrize("category", list(Category))
    def test_each_category_sums_to_its_declared_weight(self, category):
        earned = sum(c.points for c in checks_for(category))
        assert earned == CATEGORY_WEIGHTS[category]

    def test_declared_weights_sum_to_one_hundred(self):
        assert sum(CATEGORY_WEIGHTS.values()) == 100

    def test_weights_match_the_published_standard(self):
        assert CATEGORY_WEIGHTS == {
            Category.ARCHITECTURE: 10,
            Category.PERFORMANCE: 15,
            Category.ACCESSIBILITY: 15,
            Category.SEO: 15,
            Category.LOCAL_SEO: 10,
            Category.UX_CONVERSION: 15,
            Category.SECURITY_PRIVACY: 10,
            Category.CONTENT: 10,
        }


class TestCatalogueIntegrity:
    def test_check_ids_are_unique(self):
        ids = [c.id for c in STANDARD]
        assert len(ids) == len(set(ids))

    def test_check_ids_carry_their_category_prefix(self):
        for check in STANDARD:
            assert check.id.startswith(check.category.prefix), check.id

    def test_every_check_has_a_title(self):
        assert all(c.title.strip() for c in STANDARD)

    def test_every_check_declares_required_evidence(self):
        # A check that needs no evidence cannot be honestly decided.
        assert all(c.requires for c in STANDARD)

    def test_every_check_declares_a_level(self):
        assert all(isinstance(c.level, Level) for c in STANDARD)

    def test_every_check_carries_a_remedy(self):
        # The report is meant to be actionable: a failing check must say what
        # to do about it, not just that it failed.
        assert all(c.remedy.strip() for c in STANDARD)


class TestHonestyInvariants:
    def test_field_metrics_are_never_level_one(self):
        # Core Web Vitals come from real users. A lab number is not the metric,
        # and the standard must not let one be scored as though it were.
        for check in STANDARD:
            if Evidence.FIELD_DATA in check.requires:
                assert check.level is not Level.AUTOMATED, check.id

    def test_human_judgement_checks_are_level_three(self):
        for check in STANDARD:
            if Evidence.HUMAN in check.requires:
                assert check.level is Level.HUMAN, check.id

    def test_core_web_vitals_require_field_data(self):
        for cid in ("B1", "B2", "B3"):
            check = next(c for c in STANDARD if c.id == cid)
            assert Evidence.FIELD_DATA in check.requires

    def test_fabricated_credentials_is_critical_and_human(self):
        check = next(c for c in STANDARD if c.id == "H4")
        assert check.critical is True
        assert check.level is Level.HUMAN

    def test_fabricated_testimonials_is_critical_and_human(self):
        check = next(c for c in STANDARD if c.id == "H3")
        assert check.critical is True
        assert check.level is Level.HUMAN

    def test_the_critical_set_is_not_empty(self):
        assert [c for c in STANDARD if c.critical]

    def test_every_level_is_represented(self):
        levels = {c.level for c in STANDARD}
        assert levels == set(Level)
