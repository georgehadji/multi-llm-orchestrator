"""
Audit Wave A — live-by-default one-line fixes
=============================================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave A.

A1: dashboard_core/core.py put the raw ?view= value into the error page HTML (XSS).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


async def test_a1_dashboard_error_page_escapes_view_name() -> None:
    from orchestrator.dashboard_core.core import DashboardCore

    page = await DashboardCore().render("<img src=x onerror=alert(1)>")

    assert "<img" not in page
    assert "&lt;img" in page
