"""WF-100 — Website Factory Quality Standard v1.0.

A hundred checks, one point each, across eight categories, applied before a
site launches. Two rules govern the outcome:

    score >= 90  AND  critical failures == 0   ->  launch

and, the one this implementation exists to protect:

    a check the tool could not measure is reported OUTSTANDING, never passed.

That second rule is why the auditor reports three numbers instead of one: what
was *earned*, what is still *outstanding*, and the *ceiling* those two imply.
A tool that scores unmeasured checks as passes reports a launch-ready site and
is wrong in the one direction that costs a client money.
"""

from __future__ import annotations

from .standard import (
    CATEGORY_WEIGHTS,
    STANDARD,
    Category,
    Check,
    Evidence,
    Level,
    checks_for,
)

__all__ = [
    "CATEGORY_WEIGHTS",
    "STANDARD",
    "Category",
    "Check",
    "Evidence",
    "Level",
    "checks_for",
]
