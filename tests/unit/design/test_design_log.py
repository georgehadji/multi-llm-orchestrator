"""Unit tests for design_log."""

from datetime import datetime, timezone

import pytest

from orchestrator.design.design_log import DesignLog, DesignLogEntry


def _entry(
    macro: str = "bento_grid",
    theme: str = "lumen",
    genre: str = "modern-minimal",
    nav: str = "N1",
    footer: str = "Ft1",
    timestamp: str | None = None,
) -> DesignLogEntry:
    return DesignLogEntry(
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        macrostructure=macro,
        theme=theme,
        genre=genre,
        nav_archetype=nav,
        footer_archetype=footer,
    )


@pytest.mark.unit
class TestDesignLog:
    def test_is_recently_used_empty(self, tmp_path):
        log = DesignLog(tmp_path)
        assert not log.is_recently_used("bento_grid")

    def test_is_recently_used_within_window(self, tmp_path):
        log = DesignLog(tmp_path)
        log.append(_entry(macro="bento_grid"))
        assert log.is_recently_used("bento_grid", within=3)

    def test_is_recently_used_outside_window(self, tmp_path):
        log = DesignLog(tmp_path)
        # Add an old entry at the beginning, then 3 recent entries
        old = datetime.fromtimestamp(0, tz=timezone.utc).isoformat()
        log.entries.append(_entry(macro="bento_grid", timestamp=old))
        log.append(_entry(macro="a"))
        log.append(_entry(macro="b"))
        log.append(_entry(macro="c"))
        assert not log.is_recently_used("bento_grid", within=3)

    def test_append_and_save(self, tmp_path):
        log = DesignLog(tmp_path)
        entry = _entry(
            macro="manifesto", theme="midnight", genre="editorial", nav="N1a", footer="Ft1"
        )
        log.append(entry)

        log2 = DesignLog(tmp_path)
        assert len(log2.entries) == 1
        assert log2.entries[0].macrostructure == "manifesto"
        assert log2.entries[0].theme == "midnight"

    def test_different_macrostructure_not_recent(self, tmp_path):
        log = DesignLog(tmp_path)
        log.append(_entry(macro="bento_grid"))
        assert not log.is_recently_used("manifesto", within=3)

    def test_persistence_roundtrip(self, tmp_path):
        log = DesignLog(tmp_path)
        log.append(_entry(macro="a", theme="t1", genre="g1", nav="n1", footer="f1"))
        log.append(_entry(macro="b", theme="t2", genre="g2", nav="n2", footer="f2"))

        log2 = DesignLog(tmp_path)
        assert len(log2.entries) == 2
        assert log2.entries[0].macrostructure == "a"
        assert log2.entries[1].macrostructure == "b"
