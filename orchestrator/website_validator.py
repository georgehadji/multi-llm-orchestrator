"""
Website Quality Validator — backward-compatibility shim.

The implementation lives in :mod:`orchestrator.generators.website_validator`.
This module used to carry a SECOND, independently-maintained copy of the same
validator, and the two drifted: the generators copy grew three security checks
the root copy never had, and a fix to the shared content-quality logic had to be
applied twice, by hand, to both files. Re-exporting removes the drift surface
entirely — there is now one validator, and `cli_website.py` (which reaches the
validator through the root `website_generator`) gains the security checks it was
silently missing.

Import from ``orchestrator.generators.website_validator`` in new code.
"""

from __future__ import annotations

from .generators.website_validator import WebsiteQualityValidator

__all__ = ["WebsiteQualityValidator"]
