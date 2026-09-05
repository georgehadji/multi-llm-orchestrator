"""
COST_TABLE pricing-sync contract — proves the live-path model pricing in
``orchestrator/models.py::COST_TABLE`` matches OpenRouter's actual per-token
pricing, within rounding tolerance.

Why this test exists
---------------------
``COST_TABLE`` is a **hardcoded module-level dict literal** in models.py — it
is NOT loaded from ``orchestrator/config/costs.json`` at runtime. Python's
module ``__getattr__`` (PEP 562) only fires when normal attribute lookup
fails; since ``COST_TABLE`` already exists as a plain module-level name, the
JSON-driven ``_build_cost_table()`` / ``costs.json`` path is dead code for
this table (verified: ``_build_cost_table()`` returns a 144-entry dict built
from costs.json, while ``orchestrator.models.COST_TABLE`` resolves to a
different, 152-entry hardcoded dict — they are not the same object and do
not have the same content). Whatever this file finds wrong with COST_TABLE
is a REAL bug in the numbers ``estimate_cost()`` / ``Budget`` /
``BudgetHierarchy`` actually use, editing costs.json will not fix it.

This test cross-checks COST_TABLE against ``openrouter_models.json`` (repo
root, git-tracked), a cached snapshot of the OpenRouter ``/api/v1/models``
catalogue in the same format consumed by ``scripts/audit_openrouter_models.py``
and ``tests/unit/test_openrouter_model_audit.py``. It is skipped (not failed)
when that file is absent — regenerate with:

    python scripts/audit_openrouter_models.py --save-snapshot \
        openrouter_models.json

Deliberately excluded from comparison:
* Deprecated ``:free`` endpoint-variant aliases (enum member names starting
  with ``_``) — retired 2026-07-14 project policy, no longer routed to, and
  several no longer exist as distinct ``:free`` catalogue entries at all
  (the base paid id is what a naive id lookup would otherwise resolve to,
  which is not a meaningful comparison).
* ``openrouter/auto`` — the dynamic meta-router; OpenRouter reports its
  pricing as a ``-1`` per-token sentinel (varies by the model actually
  selected), not a comparable fixed price.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import COST_TABLE, Model

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "openrouter_models.json"

# Relative tolerance for float/rounding noise between how this repo records
# $/1M-token prices and how OpenRouter reports $/token (multiplied by 1e6).
_REL_TOL = 0.05

# Enum members intentionally excluded from live-price comparison (see module
# docstring). Identified structurally, not by a hand-maintained id list.
_EXCLUDED_VALUES = {"openrouter/auto"}


def _load_snapshot_prices() -> dict[str, tuple[float, float]]:
    """Return {model_id: (input_$/1M, output_$/1M)} from the cached snapshot.

    Only direct ``id`` matches are indexed (not ``canonical_slug``) so a
    zero-priced ``:free`` variant can never shadow its paid base id or vice
    versa.
    """
    data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    prices: dict[str, tuple[float, float]] = {}
    for entry in data.get("data", []):
        pricing = entry.get("pricing") or {}
        try:
            input_price = float(pricing["prompt"]) * 1_000_000
            output_price = float(pricing["completion"]) * 1_000_000
        except (KeyError, TypeError, ValueError):
            continue
        prices[entry["id"]] = (input_price, output_price)
    return prices


def _comparable_models() -> list[tuple[Model, float, float]]:
    """COST_TABLE entries that have a live snapshot price and are in scope."""
    if not SNAPSHOT_PATH.exists():
        return []
    live_prices = _load_snapshot_prices()
    out = []
    for model, cost in COST_TABLE.items():
        if model.name.startswith("_"):
            continue
        if model.value in _EXCLUDED_VALUES:
            continue
        if model.value not in live_prices:
            continue
        snap_input, snap_output = live_prices[model.value]
        # A live sentinel of exactly -1 means "dynamic pricing" server-side.
        if snap_input < 0 or snap_output < 0:
            continue
        out.append((model, snap_input, snap_output))
    return out


def _rel_close(a: float, b: float, tol: float = _REL_TOL) -> bool:
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


@pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(),
    reason="no cached OpenRouter catalogue snapshot; run "
    "`python scripts/audit_openrouter_models.py --save-snapshot "
    "openrouter_models.json`",
)
class TestCostTablePricingSync:
    def test_snapshot_has_comparable_models(self):
        """Sanity check: the fixture setup actually finds overlap to compare."""
        assert len(_comparable_models()) > 50, (
            "expected substantial overlap between COST_TABLE and the "
            "snapshot; got too few comparable models — check id formats "
            "or the snapshot content"
        )

    def test_input_prices_match_live_catalogue(self):
        mismatches = [
            (model.value, cost_in, live_in)
            for model, live_in, _live_out in _comparable_models()
            if not _rel_close((cost_in := COST_TABLE[model]["input"]), live_in)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} model(s) have a stale input price in "
            f"COST_TABLE (model, COST_TABLE input $/1M, live input $/1M): "
            f"{mismatches}"
        )

    def test_output_prices_match_live_catalogue(self):
        mismatches = [
            (model.value, cost_out, live_out)
            for model, _live_in, live_out in _comparable_models()
            if not _rel_close((cost_out := COST_TABLE[model]["output"]), live_out)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} model(s) have a stale output price in "
            f"COST_TABLE (model, COST_TABLE output $/1M, live output $/1M): "
            f"{mismatches}"
        )
