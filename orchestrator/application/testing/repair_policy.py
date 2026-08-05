"""RepairPolicy — bounded, escalating, non-cheating repair (E-3).

Three mechanisms harden the repair loop:

1. **Immutability lock (hash lock).** The test suite content is hashed when
   the policy is armed; any change to it afterwards raises
   :class:`TestHashLockError`. The agent cannot weaken its own oracle — a
   repair response that modifies the test file is rejected.

2. **Plateau detection.** Failure signatures (exception type + test node id
   + first traceback frame) are normalized and recorded per iteration. Two
   *consecutive identical* signatures mean the current model tier is stuck.

3. **Escalation.** On plateau, the model escalates one tier via the existing
   ``FALLBACK_CHAIN`` (config/fallbacks.json). At most one escalation, then
   the loop fails with a diagnostic artifact instead of retrying forever.

Collection errors (import/setup failures, not test failures) are never
repairable by an LLM and are excluded upstream (tests_run == 0 skips
self-heal); this module's signatures still record them for diagnostics.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SIG_NODE_RE = re.compile(r"[\w./\\]+\.py::[\w\[\].\-()]+")
_SIG_EXC_RE = re.compile(r"(\w+(?:Error|Exception|Failure))(?::|\s)")
_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)', re.MULTILINE)


class TestHashLockError(RuntimeError):
    """Raised when a repair iteration modifies the hash-locked test suite."""


@dataclass(frozen=True)
class FailureSignature:
    """Normalized, comparable failure identity."""

    exception_type: str
    node_id: str
    first_frame: str

    @property
    def is_blank(self) -> bool:
        """True when nothing usable could be extracted."""
        return not (self.exception_type or self.node_id or self.first_frame)


def normalize_failure_signature(error: str) -> FailureSignature | None:
    """Extract a comparable signature from one failure message.

    Args:
        error: A failure message (e.g. ``"test_main.py::test_bad: ..."``,
            a traceback string, or a collection error).

    Returns:
        A normalized signature, or None for empty input.
    """
    if not error or not error.strip():
        return None
    node = _SIG_NODE_RE.search(error)
    exc = _SIG_EXC_RE.search(error)
    frame = _FRAME_RE.search(error)
    return FailureSignature(
        exception_type=exc.group(1) if exc else "",
        node_id=node.group(0) if node else "",
        first_frame=f"{frame.group(1)}:{frame.group(2)}" if frame else "",
    )


def _get_fallback_chain() -> dict[object, object]:
    """Lazily load the model fallback chain (avoids import-time I/O)."""
    from ...models import FALLBACK_CHAIN

    chain = FALLBACK_CHAIN
    return dict(chain) if chain is not None else {}


class RepairPolicy:
    """Tracks repair history and enforces the E-3 invariants."""

    def __init__(self, max_iterations: int = 5, max_escalations: int = 1) -> None:
        """Initialize the policy.

        Args:
            max_iterations: Upper bound on repair iterations per tier.
            max_escalations: Upper bound on model-tier escalations (plan: 1).
        """
        self.max_iterations = max_iterations
        self.max_escalations = max_escalations
        self._signature_history: list[FailureSignature | None] = []
        self.escalations = 0
        self._test_hash: str | None = None

    # ── immutability lock ──────────────────────────────────────────────────

    def lock_tests(self, test_code: str) -> str:
        """Hash-lock the test suite content. Returns the hash."""
        self._test_hash = hashlib.sha256(test_code.encode("utf-8")).hexdigest()
        return self._test_hash

    def assert_tests_locked(self, test_code: str) -> None:
        """Raise TestHashLockError if *test_code* differs from the locked hash."""
        if self._test_hash is None:
            return
        current = hashlib.sha256(test_code.encode("utf-8")).hexdigest()
        if current != self._test_hash:
            raise TestHashLockError(
                "Repair iteration attempted to modify the hash-locked test "
                "suite — the oracle is immutable during repair (E-3)."
            )

    # ── plateau detection ─────────────────────────────────────────────────

    def record_failure(self, errors: list[str]) -> FailureSignature | None:
        """Record the failure signatures for an iteration.

        Returns:
            The last signature (for diagnostics).
        """
        sig = normalize_failure_signature("\n".join(errors or []))
        self._signature_history.append(sig)
        return sig

    def is_plateau(self) -> bool:
        """True when the last two recorded signatures are identical.

        An identical failure twice means this model tier is stuck — repair
        must escalate, not retry (E-3 plateau detection).
        """
        if len(self._signature_history) < 2:
            return False
        a, b = self._signature_history[-2], self._signature_history[-1]
        if a is None or b is None or a.is_blank or b.is_blank:
            return False
        return a == b

    # ── escalation ────────────────────────────────────────────────────────

    def escalate(self, model: object) -> object:
        """Escalate one model tier via FALLBACK_CHAIN (at most max_escalations).

        Args:
            model: The current model.

        Returns:
            The escalated model, or *model* unchanged when the chain has no
            next tier or the escalation budget is exhausted.
        """
        if self.escalations >= self.max_escalations:
            return model
        chain = _get_fallback_chain()
        next_model = chain.get(model, model)
        if next_model is not model and next_model != model:
            self.escalations += 1
        return next_model

    # ── diagnostics ───────────────────────────────────────────────────────

    @property
    def signature_history(self) -> list[FailureSignature | None]:
        """Recorded signatures, in iteration order (for the diagnostic artifact)."""
        return list(self._signature_history)

    def diagnostic(self) -> str:
        """Human-readable summary of the repair session for the failure artifact."""
        lines = [
            f"repair iterations: {len(self._signature_history)}",
            f"escalations: {self.escalations}",
        ]
        for i, sig in enumerate(self._signature_history, 1):
            if sig is None:
                lines.append(f"  {i}: (no extractable signature)")
            else:
                lines.append(f"  {i}: {sig.exception_type} @ {sig.node_id} [{sig.first_frame}]")
        return "\n".join(lines)
