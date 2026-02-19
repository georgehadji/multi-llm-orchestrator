"""
Tests for AppAssembler and AssemblyReport (Task 4).
No real LLM calls — uses synthetic TaskResult objects.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.app_assembler import AppAssembler, AssemblyReport
from orchestrator.models import Model, Task, TaskResult, TaskType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_task(task_id: str, target_path: str = "") -> Task:
    return Task(
        id=task_id,
        type=TaskType.CODE_GEN,
        prompt="write code",
        target_path=target_path,
    )


def _make_result(task_id: str, output: str = "# code\n") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        output=output,
        score=1.0,
        model_used=Model.KIMI_K2_5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AssemblyReport — dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_assembly_report_fields():
    """AssemblyReport must have files_written, files_skipped, import_issues fields."""
    report = AssemblyReport(
        files_written=["src/main.py"],
        files_skipped=["src/missing.py"],
        import_issues=["could not resolve: foo"],
    )
    assert report.files_written == ["src/main.py"]
    assert report.files_skipped == ["src/missing.py"]
    assert report.import_issues == ["could not resolve: foo"]


# ─────────────────────────────────────────────────────────────────────────────
# AppAssembler.assemble — basic file writing
# ─────────────────────────────────────────────────────────────────────────────

def test_assemble_writes_task_output_to_target_path(tmp_path):
    """Each TaskResult.output must be written to task.target_path."""
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", target_path="src/main.py")}
    results = {"t1": _make_result("t1", output="# generated main\n")}
    scaffold = {}

    report = assembler.assemble(results, tasks, scaffold, tmp_path)

    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "# generated main\n"
    assert "src/main.py" in report.files_written


def test_assemble_multiple_tasks(tmp_path):
    """All tasks with target_path must have their files written."""
    assembler = AppAssembler()
    tasks = {
        "t1": _make_task("t1", "src/main.py"),
        "t2": _make_task("t2", "src/utils.py"),
    }
    results = {
        "t1": _make_result("t1", "# main\n"),
        "t2": _make_result("t2", "# utils\n"),
    }
    report = assembler.assemble(results, tasks, {}, tmp_path)

    assert (tmp_path / "src" / "main.py").exists()
    assert (tmp_path / "src" / "utils.py").exists()
    assert len(report.files_written) == 2


def test_assemble_skips_task_with_empty_output(tmp_path):
    """Tasks with empty output string must be skipped (not written)."""
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/main.py")}
    results = {"t1": _make_result("t1", output="")}

    report = assembler.assemble(results, tasks, {}, tmp_path)

    assert not (tmp_path / "src" / "main.py").exists()
    assert any("t1" in s for s in report.files_skipped)


def test_assemble_skips_task_with_no_target_path(tmp_path):
    """Tasks with empty target_path must be skipped."""
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", target_path="")}
    results = {"t1": _make_result("t1", output="# code\n")}

    report = assembler.assemble(results, tasks, {}, tmp_path)

    assert len(report.files_written) == 0
    assert any("t1" in s for s in report.files_skipped)


def test_assemble_preserves_scaffold_files(tmp_path):
    """Files in scaffold not overwritten by tasks must remain on disk."""
    assembler = AppAssembler()
    scaffold = {"README.md": "# My App\n"}
    (tmp_path / "README.md").write_text("# My App\n", encoding="utf-8")

    tasks = {}
    results = {}

    report = assembler.assemble(results, tasks, scaffold, tmp_path)

    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# My App\n"


# ─────────────────────────────────────────────────────────────────────────────
# ImportFixer — __init__.py generation
# ─────────────────────────────────────────────────────────────────────────────

def test_assemble_creates_init_py_for_package(tmp_path):
    """After writing src/routes/auth.py, src/__init__.py and src/routes/__init__.py must exist."""
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/routes/auth.py")}
    results = {"t1": _make_result("t1", output="# auth\n")}

    assembler.assemble(results, tasks, {}, tmp_path)

    assert (tmp_path / "src" / "__init__.py").exists()
    assert (tmp_path / "src" / "routes" / "__init__.py").exists()


def test_assemble_does_not_overwrite_existing_init_py(tmp_path):
    """Existing __init__.py files must not be overwritten by ImportFixer."""
    assembler = AppAssembler()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("# existing init\n", encoding="utf-8")

    tasks = {"t1": _make_task("t1", "src/main.py")}
    results = {"t1": _make_result("t1", output="# main\n")}

    assembler.assemble(results, tasks, {}, tmp_path)

    assert (tmp_path / "src" / "__init__.py").read_text(encoding="utf-8") == "# existing init\n"


def test_assemble_returns_assembly_report_type(tmp_path):
    """assemble() must return an AssemblyReport instance."""
    assembler = AppAssembler()
    report = assembler.assemble({}, {}, {}, tmp_path)
    assert isinstance(report, AssemblyReport)
