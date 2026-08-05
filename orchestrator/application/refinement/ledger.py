"""Refinement ledger — Command + Memento primitives (Phase 6, E-10).

Every refinement candidate is a :class:`RefinementCommand` with ``apply``
and ``revert``. Revert restores a :class:`WorkspaceMemento` snapshot
(never inverse transforms — unreliable for LLM-authored edits). The
ledger keeps an ordered history so any single candidate can be rolled
back without invalidating accepted ones, and state can be replayed from
snapshots after a crash (plan §3.4.2, §7.3).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class WorkspaceMemento:
    """Byte-level snapshot of a workspace directory tree."""

    def __init__(self, root: Path) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs: list[str] = []
        for p in root.rglob("*"):
            if "__pycache__" in str(p) or ".orch-" in p.name:
                continue
            if p.is_file():
                self._files[str(p.relative_to(root))] = p.read_bytes()
            elif p.is_dir():
                self._dirs.append(str(p.relative_to(root)))

    def restore(self, root: Path) -> None:
        """Restore the workspace to exactly this snapshot's state.

        Files present in the snapshot are rewritten; files not in the
        snapshot are removed; directories are recreated as needed.
        """
        for rel, content in self._files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        # Remove files/dirs that did not exist at snapshot time.
        existing = {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
        for rel, path in existing.items():
            if "__pycache__" in rel or ".orch-" in path.name:
                continue
            if rel not in self._files:
                path.unlink(missing_ok=True)
        for rel in self._dirs:
            (root / rel).mkdir(parents=True, exist_ok=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkspaceMemento):
            return False
        return self._files == other._files


@dataclass
class LedgerEntry:
    """One applied command plus pre/post snapshots.

    ``memento`` is the pre-command state (for revert); ``post_memento`` is
    the state after a successful apply (for replay/roll-forward).
    """

    command_name: str
    memento: WorkspaceMemento
    post_memento: WorkspaceMemento | None = None
    changed_files: tuple[str, ...] = ()


class RefinementCommand(Protocol):
    """A reversible workspace transformation (Command pattern)."""

    name: str

    def apply(self, workspace_root: Path) -> list[str]:
        """Apply the transformation; return the list of changed file paths."""
        ...


@dataclass
class RefinementLedger:
    """Ordered command history with per-entry revert."""

    entries: list[LedgerEntry] = field(default_factory=list)

    def snapshot(self, workspace_root: Path) -> WorkspaceMemento:
        """Capture the pre-command state (Memento)."""
        return WorkspaceMemento(workspace_root)

    def apply(self, command: RefinementCommand, workspace_root: Path) -> LedgerEntry:
        """Snapshot, apply, and record. Raises on failure (no partial state)."""
        memento = self.snapshot(workspace_root)
        changed = command.apply(workspace_root)
        entry = LedgerEntry(
            command_name=command.name,
            memento=memento,
            post_memento=self.snapshot(workspace_root),
            changed_files=tuple(changed),
        )
        self.entries.append(entry)
        return entry

    def revert(self, workspace_root: Path, entry: LedgerEntry | None = None) -> None:
        """Revert *entry* (or the last one) to its pre-command snapshot state."""
        target = entry or (self.entries[-1] if self.entries else None)
        if target is None:
            return
        target.memento.restore(workspace_root)
        if entry is None and self.entries:
            self.entries.pop()

    @property
    def accepted_count(self) -> int:
        """Number of applied (not reverted) commands in history."""
        return len(self.entries)

    def replay(self, workspace_root: Path) -> None:
        """Reconstruct the workspace state from the ledger's snapshots.

        Restores the latest post-command snapshot, so accepted candidates
        survive a crash without re-running any command.
        """
        if not self.entries:
            return
        latest = self.entries[-1]
        if latest.post_memento is not None:
            latest.post_memento.restore(workspace_root)
        else:  # pragma: no cover - legacy entries
            latest.memento.restore(workspace_root)
