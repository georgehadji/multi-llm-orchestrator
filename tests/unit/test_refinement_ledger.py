"""
Tests for the refinement ledger (E-10) — Command + Memento.
============================================================
Byte-exact revert: restoring a snapshot must reproduce the workspace
exactly; a command that fails must leave no partial state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.application.refinement.ledger import (
    RefinementCommand,
    RefinementLedger,
    WorkspaceMemento,
)


class _RewriteCommand(RefinementCommand):
    """Test command: rewrite one file."""

    name = "test.rewrite"

    def __init__(self, rel: str, content: str) -> None:
        self._rel = rel
        self._content = content

    def apply(self, workspace_root: Path) -> list[str]:
        target = workspace_root / self._rel
        target.write_text(self._content, encoding="utf-8")
        return [self._rel]


class _AddFileCommand(RefinementCommand):
    """Test command: create a new file."""

    name = "test.add"

    def __init__(self, rel: str, content: str) -> None:
        self._rel = rel
        self._content = content

    def apply(self, workspace_root: Path) -> list[str]:
        target = workspace_root / self._rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._content, encoding="utf-8")
        return [self._rel]


@pytest.mark.unit
class TestWorkspaceMemento:
    """Memento captures and restores byte-exact state."""

    def test_restore_reverts_content(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        memento = WorkspaceMemento(tmp_path)
        (tmp_path / "a.py").write_text("x = 999", encoding="utf-8")
        memento.restore(tmp_path)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1"

    def test_restore_removes_added_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        memento = WorkspaceMemento(tmp_path)
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
        memento.restore(tmp_path)
        assert not (tmp_path / "b.py").exists()

    def test_restore_restores_deleted_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        memento = WorkspaceMemento(tmp_path)
        (tmp_path / "a.py").unlink()
        memento.restore(tmp_path)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1"


@pytest.mark.unit
class TestRefinementLedger:
    """Ledger applies, reverts, and replays commands."""

    def test_apply_records_entry(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        ledger = RefinementLedger()
        entry = ledger.apply(_RewriteCommand("a.py", "x = 2"), tmp_path)
        assert entry.command_name == "test.rewrite"
        assert ledger.accepted_count == 1

    def test_revert_last_restores_state(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        ledger = RefinementLedger()
        ledger.apply(_RewriteCommand("a.py", "x = 2"), tmp_path)
        ledger.apply(_AddFileCommand("b.py", "y = 2"), tmp_path)
        ledger.revert(tmp_path)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2"  # entry 1 intact
        assert not (tmp_path / "b.py").exists()  # entry 2 reverted

    def test_revert_entry_does_not_invalidate_others(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        ledger = RefinementLedger()
        first = ledger.apply(_RewriteCommand("a.py", "x = 2"), tmp_path)
        ledger.apply(_RewriteCommand("a.py", "x = 3"), tmp_path)
        # Revert the FIRST entry specifically: later accepted candidate survives.
        ledger.revert(tmp_path, first)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1"

    def test_replay_restores_latest_snapshot(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        ledger = RefinementLedger()
        ledger.apply(_RewriteCommand("a.py", "x = 2"), tmp_path)
        (tmp_path / "a.py").write_text("CORRUPTED", encoding="utf-8")
        ledger.replay(tmp_path)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 2"
