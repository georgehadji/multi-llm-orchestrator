"""
Tests for the metric collection harness (E-9).
===============================================
Collects a MetricSnapshot over a real temp workspace: complexity, nesting,
length, duplication, dead symbols (graceful when vulture is absent).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.metrics.collector import (
    MetricCollector,
    bundle_size_collector,
    collect_snapshot,
)


def _make_workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "svc.py").write_text(
        "def handle(x):\n"
        "    if x:\n"
        "        for i in range(3):\n"
        "            pass\n"
        "    return x\n"
        "def duplicate(a, b):\n"
        "    return a + b\n"
        "def duplicate2(c, d):\n"
        "    return c + d\n",
        encoding="utf-8",
    )
    (tmp_path / "test_svc.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    return Workspace(root=tmp_path, framework="pytest")


@pytest.mark.unit
class TestCollectSnapshot:
    """End-to-end measurement over a real workspace."""

    def test_snapshot_has_all_fields(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        snapshot = asyncio.run(collect_snapshot(ws))
        assert snapshot.cyclomatic_max >= 3  # handle() has if + for + base
        assert snapshot.max_nesting_depth >= 3  # def > if > for
        assert snapshot.longest_function_lines >= 4
        assert snapshot.duplicated_blocks >= 1  # duplicate/duplicate2
        assert snapshot.total_lines >= 9
        assert snapshot.dead_symbols >= 0
        assert snapshot.bundle_bytes is None

    def test_snapshot_is_frozen_and_comparable(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        before = asyncio.run(collect_snapshot(ws))
        after = asyncio.run(collect_snapshot(ws))
        assert before == after  # deterministic

    def test_empty_workspace_safe(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path, framework="pytest")
        snapshot = asyncio.run(collect_snapshot(ws, use_static=False))
        assert snapshot.cyclomatic_max == 1
        assert snapshot.total_lines == 0


@pytest.mark.unit
class TestBundleCollector:
    """Web bundle size collector (E-12 bundle_bytes)."""

    def test_measures_dist(self, tmp_path: Path) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "app.js").write_text("x" * 100)
        ws = Workspace(root=tmp_path, framework="web")
        collector = bundle_size_collector()
        result = collector.collect(ws)
        assert result["bundle_bytes"] == 100

    def test_absent_bundle_is_none(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path, framework="web")
        assert bundle_size_collector().collect(ws)["bundle_bytes"] is None

    def test_extra_collectors_merge_into_snapshot(self, tmp_path: Path) -> None:
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "app.js").write_text("y" * 50)
        ws = Workspace(root=tmp_path, framework="web")
        snapshot = asyncio.run(
            collect_snapshot(ws, use_static=False, extra_collectors=(bundle_size_collector(),))
        )
        assert snapshot.bundle_bytes == 50


@pytest.mark.unit
class TestDeadCodeGraceful:
    """Missing vulture degrades to a count of 0, never a crash."""

    def test_collect_does_not_raise_without_vulture(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "orchestrator.infrastructure.metrics.dead_code.count_dead_symbols",
            _never_raises,
        )
        ws = _make_workspace(tmp_path)
        snapshot = asyncio.run(collect_snapshot(ws, use_static=False))
        assert snapshot.dead_symbols >= 0


async def _never_raises(*args, **kwargs) -> int | None:
    return None
