"""Testing subsystem event builders (E-8).

Pure payload construction — no I/O, no bus dependency. Callers publish the
returned :class:`DomainEvent` on the unified bus; sinks observe hashes and
counts only (never artifact content or secrets).

Event inventory (per plan §E-8):

* ``test.suite.generated``     — test_count, vacuity_rate, mode
* ``test.suite.executed``      — passed/failed/skipped, executed, duration_ms,
                                 isolation, flaky_count, mutation_score
* ``test.repair.iteration``    — iteration, signature, tier, plateau
* ``test.mutation.scored``     — total, killed, score
* ``test.flake.quarantined``   — node_id (hash), observation_count
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from ...unified_events.core import DomainEvent, EventType


def _node_hash(node_id: str) -> str:
    """SHA-256 of a test node id — node ids are emitted as hashes only."""
    return hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:16]


def suite_generated(
    aggregate_id: str,
    *,
    test_count: int,
    vacuity_rate: float,
    mode: str,
) -> DomainEvent:
    """Emit after the RED-gate measures a generated suite."""
    return DomainEvent(
        event_type=EventType.TEST_SUITE_GENERATED,
        aggregate_id=aggregate_id,
        metadata={
            "test_count": test_count,
            "vacuity_rate": round(vacuity_rate, 4),
            "mode": mode,
        },
    )


def suite_executed(
    aggregate_id: str,
    *,
    passed: bool,
    executed: int,
    passed_count: int,
    failed_count: int,
    skipped_count: int,
    duration_ms: float,
    isolation: str,
    flaky_count: int = 0,
    mutation_score: float | None = None,
) -> DomainEvent:
    """Emit after a suite execution completes."""
    return DomainEvent(
        event_type=EventType.TEST_SUITE_EXECUTED,
        aggregate_id=aggregate_id,
        metadata={
            "passed": passed,
            "executed": executed,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "duration_ms": round(duration_ms, 1),
            "isolation": isolation,
            "flaky_count": flaky_count,
            "mutation_score": round(mutation_score, 4) if mutation_score is not None else None,
        },
    )


def repair_iteration(
    aggregate_id: str,
    *,
    iteration: int,
    signature: str,
    tier: str,
    plateau: bool,
) -> DomainEvent:
    """Emit per repair iteration (signature is a normalized, non-PII string)."""
    return DomainEvent(
        event_type=EventType.TEST_REPAIR_ITERATION,
        aggregate_id=aggregate_id,
        metadata={
            "iteration": iteration,
            "signature": signature[:200],
            "tier": tier,
            "plateau": plateau,
        },
    )


def mutation_scored(
    aggregate_id: str,
    *,
    total: int,
    killed: int,
    score: float,
) -> DomainEvent:
    """Emit after mutation scoring completes."""
    return DomainEvent(
        event_type=EventType.TEST_MUTATION_SCORED,
        aggregate_id=aggregate_id,
        metadata={
            "total": total,
            "killed": killed,
            "survived": total - killed,
            "score": round(score, 4),
        },
    )


def flake_quarantined(
    aggregate_id: str,
    *,
    node_id: str,
    observation_count: int,
) -> DomainEvent:
    """Emit when a node id crosses the flake quarantine threshold.

    Only a hash of the node id is emitted (no test content).
    """
    return DomainEvent(
        event_type=EventType.TEST_FLAKE_QUARANTINED,
        aggregate_id=aggregate_id,
        metadata={
            "node_id_hash": _node_hash(node_id),
            "observation_count": observation_count,
        },
    )
