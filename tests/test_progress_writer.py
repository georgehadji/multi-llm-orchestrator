"""
Tests — Improvement 13: ProgressWriter (Streaming/Incremental Output)
======================================================================
Covers:
  - task_completed writes the per-task output file immediately
  - PROGRESS.jsonl is appended with one line per completed task
  - summary.json is rewritten after every task completion
  - concurrent task_completed calls are safe (asyncio.Lock guards shared files)
  - CODE_GEN tasks with named file blocks cause extract_named_files to run
  - tasks with empty output are silently skipped
  - ProgressEntry exported from orchestrator package
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from orchestrator import ProgressWriter, ProgressEntry
from orchestrator.models import (
    Budget, Model, ProjectState, ProjectStatus,
    Task, TaskResult, TaskStatus, TaskType,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_state() -> ProjectState:
    return ProjectState(
        project_description="test project",
        success_criteria="pass",
        status=ProjectStatus.PARTIAL_SUCCESS,
        budget=Budget(max_usd=1.0, max_time_seconds=60),
    )


def _make_task(task_id: str = "t1", task_type: TaskType = TaskType.CODE_GEN) -> Task:
    return Task(
        id=task_id,
        type=task_type,
        prompt="do something",
        acceptance_threshold=0.7,
        max_iterations=3,
    )


def _make_result(
    task_id: str = "t1",
    output: str = "def foo():\n    return 42",
    score: float = 0.9,
    status: TaskStatus = TaskStatus.COMPLETED,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        output=output,
        score=score,
        model_used=Model.GEMINI_FLASH,
        status=status,
    )


def _run(coro):
    return asyncio.run(coro)


# ─── ProgressEntry export ─────────────────────────────────────────────────────

class TestProgressEntryExport:
    def test_progress_entry_importable(self):
        assert ProgressEntry is not None

    def test_progress_writer_importable(self):
        assert ProgressWriter is not None

    def test_progress_entry_fields(self):
        entry = ProgressEntry(
            task_id="t1",
            status="completed",
            score=0.9,
            cost_usd=0.001,
            model_used="gemini-2.5-flash",
            timestamp_iso="2025-01-01T00:00:00",
            output_file="t1_code_generation.py",
        )
        assert entry.task_id == "t1"
        assert entry.score == 0.9


# ─── task_completed writes output file ───────────────────────────────────────

class TestTaskCompletedWritesFile:
    def test_output_file_created(self):
        state = _make_state()
        task = _make_task("t1", TaskType.CODE_GEN)
        result = _make_result("t1", "def foo():\n    return 42")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            files = list(Path(tmpdir).iterdir())
            # Should have written a task output file
            assert any(f.name.startswith("t1_") for f in files)

    def test_output_file_name_includes_task_type(self):
        state = _make_state()
        task = _make_task("t1", TaskType.CODE_GEN)
        result = _make_result("t1")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            files = list(Path(tmpdir).iterdir())
            code_files = [f for f in files if "code_generation" in f.name]
            assert len(code_files) == 1

    def test_output_file_content_preserved(self):
        state = _make_state()
        task = _make_task("t1", TaskType.WRITING)
        result = _make_result("t1", output="Hello world report content")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            md_files = list(Path(tmpdir).glob("*.md"))
            assert len(md_files) >= 1
            content = md_files[0].read_text(encoding="utf-8")
            assert "Hello world report content" in content

    def test_empty_output_skipped(self):
        state = _make_state()
        task = _make_task("t1", TaskType.CODE_GEN)
        result = _make_result("t1", output="")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            # Only the output_dir itself should exist (no task files)
            files = [f for f in Path(tmpdir).iterdir() if f.is_file()]
            assert len(files) == 0


# ─── PROGRESS.jsonl ──────────────────────────────────────────────────────────

class TestProgressJsonl:
    def test_progress_jsonl_created(self):
        state = _make_state()
        task = _make_task("t1")
        result = _make_result("t1")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            progress_path = Path(tmpdir) / "PROGRESS.jsonl"
            assert progress_path.exists()

    def test_progress_jsonl_one_line_per_task(self):
        state = _make_state()
        task1 = _make_task("t1", TaskType.CODE_GEN)
        task2 = _make_task("t2", TaskType.WRITING)
        result1 = _make_result("t1", output="code output here")
        result2 = _make_result("t2", output="prose output here")
        state.results["t1"] = result1
        state.results["t2"] = result2

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result1, task1))
            _run(pw.task_completed("t2", result2, task2))

            progress_path = Path(tmpdir) / "PROGRESS.jsonl"
            lines = [l for l in progress_path.read_text().splitlines() if l.strip()]
            assert len(lines) == 2

    def test_progress_jsonl_fields(self):
        state = _make_state()
        task = _make_task("t1")
        result = _make_result("t1", score=0.88)
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            progress_path = Path(tmpdir) / "PROGRESS.jsonl"
            entry = json.loads(progress_path.read_text().strip())
            assert entry["task_id"] == "t1"
            assert entry["status"] == "completed"
            assert entry["score"] == 0.88
            assert "model_used" in entry
            assert "timestamp_iso" in entry
            assert "output_file" in entry

    def test_progress_jsonl_task_id_in_output_file(self):
        state = _make_state()
        task = _make_task("task_001", TaskType.CODE_GEN)
        result = _make_result("task_001")
        state.results["task_001"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("task_001", result, task))

            entry = json.loads(
                (Path(tmpdir) / "PROGRESS.jsonl").read_text().strip()
            )
            assert "task_001" in entry["output_file"]


# ─── summary.json updated incrementally ─────────────────────────────────────

class TestSummaryJsonIncremental:
    def test_summary_json_created_after_first_task(self):
        state = _make_state()
        task = _make_task("t1")
        result = _make_result("t1")
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            summary_path = Path(tmpdir) / "summary.json"
            assert summary_path.exists()

    def test_summary_json_grows_with_tasks(self):
        state = _make_state()
        task1 = _make_task("t1", TaskType.CODE_GEN)
        task2 = _make_task("t2", TaskType.WRITING)
        result1 = _make_result("t1", output="code")
        result2 = _make_result("t2", output="prose")
        state.results["t1"] = result1

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result1, task1))

            summary1 = json.loads((Path(tmpdir) / "summary.json").read_text())
            count1 = len(summary1["tasks"])

            # Add second task to shared state
            state.results["t2"] = result2
            _run(pw.task_completed("t2", result2, task2))

            summary2 = json.loads((Path(tmpdir) / "summary.json").read_text())
            count2 = len(summary2["tasks"])

            assert count2 > count1


# ─── Concurrent safety ───────────────────────────────────────────────────────

class TestConcurrentSafety:
    def test_concurrent_completions_all_written(self):
        """Multiple concurrent task_completed calls all append to PROGRESS.jsonl."""
        state = _make_state()
        tasks = [_make_task(f"t{i}", TaskType.WRITING) for i in range(5)]
        results = [_make_result(f"t{i}", output=f"output for task {i}") for i in range(5)]
        for i in range(5):
            state.results[f"t{i}"] = results[i]

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)

            async def run_all():
                await asyncio.gather(*[
                    pw.task_completed(f"t{i}", results[i], tasks[i])
                    for i in range(5)
                ])

            asyncio.run(run_all())

            progress_path = Path(tmpdir) / "PROGRESS.jsonl"
            lines = [l for l in progress_path.read_text().splitlines() if l.strip()]
            assert len(lines) == 5


# ─── CODE_GEN named file extraction ──────────────────────────────────────────

class TestCodeGenExtraction:
    def test_named_files_extracted_from_code_gen(self):
        """ProgressWriter extracts **filename** blocks for CODE_GEN tasks."""
        state = _make_state()
        task = _make_task("t1", TaskType.CODE_GEN)
        code_output = (
            "Here is the implementation:\n\n"
            "**src/main.py**\n"
            "```python\n"
            "def main():\n    print('hello')\n"
            "```\n"
        )
        result = _make_result("t1", output=code_output)
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            extracted = Path(tmpdir) / "app" / "src" / "main.py"
            assert extracted.exists()

    def test_no_extraction_for_non_code_gen(self):
        """Writing tasks with code-like content do not trigger extraction."""
        state = _make_state()
        task = _make_task("t1", TaskType.WRITING)
        prose_output = (
            "**src/main.py**\n"
            "```python\n"
            "def main(): pass\n"
            "```\n"
        )
        result = _make_result("t1", output=prose_output)
        state.results["t1"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            pw = ProgressWriter(Path(tmpdir), state)
            _run(pw.task_completed("t1", result, task))

            app_dir = Path(tmpdir) / "app"
            assert not app_dir.exists()
