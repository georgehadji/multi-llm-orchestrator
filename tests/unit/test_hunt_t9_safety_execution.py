"""
T9 (safety/execution/plugin surface) proof-of-defect and no-regression tests.

Three VERIFIED DEFECTs from docs/hunts/t9-safety-execution/inventory.md:

C1 — safety/architecture_rules.py, safety/architecture_advisor.py, and
     safety/reference_monitor.py were dead duplicates of their live root
     canonicals, each carrying a broken relative import (one dot too many
     for that file's own package depth) that would raise ImportError the
     moment anything reached it — the same defect shape T1 already found
     and fixed once in safety/code_executor.py. Converted all three to
     re-export shims of their canonical root modules, the established
     pattern for every other duplicate pair this hunt has found.
C2 — plugins/discovery.py::load_plugin() built the bundled-plugin import
     path as "orchestrator.plugin.plugins.<kind>.<name>" (singular
     "plugin" — not even a package), which can never match
     _bundled_plugin_path()'s own "orchestrator/plugins/<kind>/" (plural).
     Any bundled plugin would be discovered but then fail to import,
     silently swallowed by the surrounding except Exception.
C3 — safety/generated_output_scanner.py::scan_output_dir() silently
     skipped any file it couldn't read with no counter and no log — a scan
     that missed files looked identical to one that scanned everything and
     found nothing clean.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_safety_architecture_rules_is_canonical():
    from orchestrator.architecture_rules import ArchitectureRulesEngine as canonical
    from orchestrator.safety.architecture_rules import ArchitectureRulesEngine as via_safety

    assert via_safety is canonical


def test_c1_safety_architecture_advisor_is_canonical():
    from orchestrator.architecture_advisor import ArchitectureAdvisor as canonical
    from orchestrator.safety.architecture_advisor import ArchitectureAdvisor as via_safety

    assert via_safety is canonical


def test_c1_safety_reference_monitor_is_canonical():
    from orchestrator.reference_monitor import ReferenceMonitor as canonical
    from orchestrator.safety.reference_monitor import ReferenceMonitor as via_safety

    assert via_safety is canonical


# --- C2 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_bundled_plugin_import_path_matches_plugins_package(tmp_path, monkeypatch):
    from orchestrator.plugins import discovery

    captured: dict[str, str] = {}

    def _fake_import_module(name: str):
        captured["name"] = name
        raise ImportError("stop here — this test only checks the module name built")

    monkeypatch.setattr(discovery.importlib, "import_module", _fake_import_module)

    plugin_info = {"name": "demo", "source": "bundled", "path": tmp_path}
    result = await discovery.load_plugin(plugin_info, kind="memory")

    assert captured.get("name") == "orchestrator.plugins.memory.demo", (
        f"expected the bundled-plugin import path to match orchestrator/plugins/ "
        f"(plural), got: {captured.get('name')!r}"
    )
    assert result is None  # the (unrelated) forced ImportError is swallowed by design


# --- C3 -----------------------------------------------------------------


def test_c3_scan_output_dir_counts_and_logs_unreadable_files(tmp_path, caplog, monkeypatch):
    from pathlib import Path

    from orchestrator.safety.generated_output_scanner import scan_output_dir

    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("y = 2\n", encoding="utf-8")

    # _iter_scannable_files() filters to real files (an unreadable directory
    # would just be excluded, not hit the failure path), so simulate the
    # genuine race/permission failure scan_output_dir()'s except OSError
    # branch exists for: read_text() raises for this one file, unaffected
    # for every other real file.
    real_read_text = Path.read_text

    def _flaky_read_text(self, *args, **kwargs):
        if self == bad_file:
            raise OSError("simulated: permission denied / TOCTOU race")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)

    with caplog.at_level(logging.WARNING):
        report = scan_output_dir(tmp_path)

    assert report.files_scanned == 1
    assert report.files_skipped == 1, "an unreadable file must be counted, not silently dropped"
    assert any(
        "bad.py" in rec.message for rec in caplog.records
    ), f"expected a warning naming the unreadable file, got: {[r.message for r in caplog.records]}"
