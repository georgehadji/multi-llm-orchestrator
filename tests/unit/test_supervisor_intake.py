"""
Unit tests for orchestrator.supervisor.intake.

Verifies that human and agent directives collapse to identical JobArgs.
"""

from __future__ import annotations

import pytest

from orchestrator.supervisor.intake import normalize
from orchestrator.supervisor.models import Directive

pytestmark = pytest.mark.unit


def test_human_directive_normalized():
    d = Directive(source="human", text="build a website", criteria="responsive", budget=5.0)
    job = normalize(d)
    assert job.project_description == "build a website"
    assert job.success_criteria == "responsive"
    assert job.budget == 5.0


def test_agent_directive_normalized_identically():
    d = Directive(source="agent", text="build a website", criteria="responsive", budget=5.0)
    job = normalize(d)
    assert job.project_description == "build a website"
    assert job.success_criteria == "responsive"
    assert job.budget == 5.0


def test_default_criteria_when_missing():
    d = Directive(source="human", text="build a website")
    job = normalize(d)
    assert "production-ready" in job.success_criteria


def test_budget_extracted_from_text():
    d = Directive(source="human", text="build a website with $10 budget")
    job = normalize(d)
    assert job.budget == 10.0
