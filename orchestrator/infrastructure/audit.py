"""
AuditLog — structured per-check audit records for policy decisions.
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Every call to PolicyEngine.check() emits one AuditRecord capturing:
  - which model was evaluated
  - which policies were applied
  - the raw compliance result (before EnforcementMode override)
  - the effective result (after EnforcementMode: MONITOR may flip to True)
  - all violation strings

AuditLog stores records in memory and can flush them to a JSONL file
for offline analysis, dashboards, or compliance reporting.

Usage
-----
    from orchestrator.audit import AuditLog
    from orchestrator.policy_engine import PolicyEngine

    log = AuditLog()
    engine = PolicyEngine(audit_log=log)
    engine.check(model, profile, policies, task_id="task_001")
    log.flush_jsonl("audit/policy_audit.jsonl")
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# ── AuditRecord ───────────────────────────────────────────────────────────────


@dataclass
class AuditRecord:
    """
    Immutable snapshot of one PolicyEngine.check() evaluation.

    Fields
    ------
    timestamp : float
        Unix epoch seconds at time of check.
    task_id : str
        Identifier of the task that triggered the check (empty string if not
        provided — e.g. when called from ConstraintPlanner._apply_filters()).
    model : str
        model.value of the evaluated Model enum member.
    passed : bool
        Effective result after EnforcementMode is applied.
        MONITOR mode may set this to True even when raw_passed is False.
    raw_passed : bool
        Raw compliance result before EnforcementMode override.
    violations : list[str]
        Human-readable violation strings (one per violated rule).
        Empty when raw_passed is True.
    enforcement_mode : str
        EnforcementMode.value that was applied ("monitor"/"soft"/"hard").
    policies_applied : list[str]
        Names (Policy.name) of all policies that were evaluated.
    """

    timestamp: float
    task_id: str
    model: str
    passed: bool
    raw_passed: bool
    violations: list[str]
    enforcement_mode: str
    policies_applied: list[str]


# ── AuditLog ──────────────────────────────────────────────────────────────────


class AuditLog:
    """
    In-memory append-only audit log for policy check results.

    Thread-safety: NOT thread-safe by design. The orchestrator runs
    in a single asyncio event loop, which serialises all check() calls.
    External concurrent writers would require a lock.

    Methods
    -------
    record()       — append one AuditRecord
    records()      — return a snapshot copy of all records
    flush_jsonl()  — append all records to a JSONL file
    to_list()      — return records as plain dicts (JSON-serialisable)
    __len__()      — number of records in the log
    clear()        — discard all records (useful between test runs)
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def record(
        self,
        task_id: str,
        model: str,
        passed: bool,
        raw_passed: bool,
        violations: list[str],
        enforcement_mode: str,
        policies_applied: list[str],
    ) -> None:
        """Append one audit record. Called internally by PolicyEngine.check()."""
        self._records.append(
            AuditRecord(
                timestamp=time.time(),
                task_id=task_id,
                model=model,
                passed=passed,
                raw_passed=raw_passed,
                violations=violations,
                enforcement_mode=enforcement_mode,
                policies_applied=policies_applied,
            )
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def records(self) -> list[AuditRecord]:
        """Return a snapshot copy of all records (safe to iterate over)."""
        return list(self._records)

    def to_list(self) -> list[dict]:
        """Return all records as plain dicts (JSON-serialisable via json.dumps)."""
        return [asdict(r) for r in self._records]

    # ── Persistence ────────────────────────────────────────────────────────────

    def flush_jsonl(self, path: str | Path) -> None:
        """
        Append all records to a JSONL file (one JSON object per line).

        The file is opened in append mode so successive flush calls
        accumulate records without overwriting earlier entries.
        """
        with open(path, "a", encoding="utf-8") as fh:
            for r in self._records:
                fh.write(json.dumps(asdict(r)) + "\n")

    async def flush(self, path: str | Path | None = None) -> int:
        """Append pending records to *path*, then drop them from memory.

        Defaults to ``settings.audit_log_path``. Returns how many were written.
        Clearing after the write keeps a long-lived process from growing
        forever and stops a second flush from duplicating lines.
        """
        if not self._records:
            return 0
        if path is None:
            from ..crosscutting.config import settings

            path = settings.audit_log_path
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        count = len(self._records)
        await asyncio.to_thread(self.flush_jsonl, target)
        del self._records[:count]
        return count

    # ── Utilities ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Discard all records. Useful between test runs or audit windows."""
        self._records.clear()
