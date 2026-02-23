# tests/test_assembler.py
"""
Tests for orchestrator.assembler — ProjectAssembler and assemble_project.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.assembler import AssemblyResult, ProjectAssembler, assemble_project
from orchestrator.models import (
    Budget, Model, ProjectState, ProjectStatus,
    Task, TaskResult, TaskStatus, TaskType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_state(tasks_and_results: list[tuple[str, TaskType, str, str]]) -> ProjectState:
    """
    Build a minimal ProjectState.

    tasks_and_results: list of (task_id, task_type, prompt, output)
    """
    state = ProjectState(
        project_description="Test project",
        success_criteria="Tests pass",
        budget=Budget(max_usd=1.0),
        status=ProjectStatus.SUCCESS,
    )
    for task_id, task_type, prompt, output in tasks_and_results:
        state.tasks[task_id] = Task(
            id=task_id,
            type=task_type,
            prompt=prompt,
        )
        state.results[task_id] = TaskResult(
            task_id=task_id,
            output=output,
            score=0.9,
            model_used=Model.DEEPSEEK_CHAT,
            status=TaskStatus.COMPLETED,
        )
        state.execution_order.append(task_id)
    return state


# ── Basic assembly ────────────────────────────────────────────────────────────

def test_assemble_writes_files_with_target_paths(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN,  "Write main.py", "```python\nprint('hello')\n```"),
        ("task_002", TaskType.WRITING,   "Write README", "# MyApp\n\nA test app."),
    ])
    task_paths = {"task_001": "src/main.py", "task_002": "README.md"}

    result = assemble_project(state, tmp_path / "app", task_paths=task_paths)

    assert result.success
    assert "src/main.py" in result.files_written
    assert "README.md" in result.files_written
    assert (tmp_path / "app" / "src" / "main.py").exists()
    assert (tmp_path / "app" / "README.md").exists()


def test_assemble_creates_subdirectories(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Write auth", "```python\ndef login(): pass\n```"),
    ])
    result = assemble_project(
        state, tmp_path / "out",
        task_paths={"task_001": "src/api/auth.py"}
    )
    assert (tmp_path / "out" / "src" / "api" / "auth.py").exists()


def test_assemble_manifest_written(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "main", "print('x')"),
    ])
    result = assemble_project(
        state, tmp_path / "out",
        task_paths={"task_001": "main.py"}
    )
    manifest_path = tmp_path / "out" / "assembly-manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "files_written" in manifest
    assert "main.py" in manifest["files_written"]


# ── Fallback naming ───────────────────────────────────────────────────────────

def test_assemble_fallback_to_flat_name_when_no_target_path(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Write something", "```python\nx = 1\n```"),
    ])
    # No task_paths provided → flat name
    result = assemble_project(state, tmp_path / "out")
    assert result.files_written
    # Flat name should contain the task_id
    assert any("task_001" in f for f in result.files_written)


def test_assemble_task_with_target_path_field(tmp_path):
    """Task.target_path field is respected when no task_paths dict is given."""
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Write models", "```python\nclass User: pass\n```"),
    ])
    # Set target_path on the Task object directly
    state.tasks["task_001"].target_path = "app/models.py"

    result = assemble_project(state, tmp_path / "out")
    assert (tmp_path / "out" / "app" / "models.py").exists()
    assert "app/models.py" in result.files_written


# ── Skipping ──────────────────────────────────────────────────────────────────

def test_assemble_skips_tasks_with_no_output(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Generate code", "```python\npass\n```"),
    ])
    # Add a result with empty output
    state.tasks["task_002"] = Task(id="task_002", type=TaskType.WRITING, prompt="Write")
    state.results["task_002"] = TaskResult(
        task_id="task_002", output="", score=0.0,
        model_used=Model.DEEPSEEK_CHAT, status=TaskStatus.FAILED,
    )
    state.execution_order.append("task_002")

    result = assemble_project(state, tmp_path / "out",
                              task_paths={"task_001": "main.py"})
    assert "task_002" in result.files_skipped
    assert len(result.files_written) == 1


# ── Overwrite ─────────────────────────────────────────────────────────────────

def test_assemble_no_overwrite_skips_existing(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Write code", "```python\nx = 1\n```"),
    ])
    out = tmp_path / "out"
    # Pre-create the file with different content
    (out / "src").mkdir(parents=True)
    existing = out / "src" / "main.py"
    existing.write_text("ORIGINAL", encoding="utf-8")

    result = assemble_project(
        state, out,
        task_paths={"task_001": "src/main.py"},
        overwrite=False,
    )
    # File still tracked as written (idempotent)
    assert "src/main.py" in result.files_written
    # Content must NOT be overwritten
    assert existing.read_text(encoding="utf-8") == "ORIGINAL"


def test_assemble_overwrite_true_replaces_file(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "Write code", "```python\nx = 42\n```"),
    ])
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "main.py"
    existing.write_text("OLD", encoding="utf-8")

    assemble_project(state, out, task_paths={"task_001": "main.py"}, overwrite=True)
    # Content should be replaced
    assert existing.read_text(encoding="utf-8") != "OLD"


# ── Content rendering ─────────────────────────────────────────────────────────

def test_assemble_strips_fences_from_python_output(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code",
         "```python\ndef hello():\n    return 'hi'\n```"),
    ])
    assemble_project(state, tmp_path / "out", task_paths={"task_001": "hello.py"})
    content = (tmp_path / "out" / "hello.py").read_text(encoding="utf-8")
    assert "```" not in content
    assert "def hello" in content


def test_assemble_markdown_written_as_is(tmp_path):
    state = _make_state([
        ("task_001", TaskType.WRITING, "README", "# Title\n\nSome **bold** text."),
    ])
    assemble_project(state, tmp_path / "out", task_paths={"task_001": "README.md"})
    content = (tmp_path / "out" / "README.md").read_text(encoding="utf-8")
    assert "# Title" in content
    assert "**bold**" in content


def test_assemble_typescript_file_strips_fences(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "React component",
         "```typescript\nconst App = () => <div />;\n```"),
    ])
    assemble_project(state, tmp_path / "out", task_paths={"task_001": "src/App.tsx"})
    content = (tmp_path / "out" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "```" not in content
    assert "const App" in content


# ── Result object ─────────────────────────────────────────────────────────────

def test_assembly_result_success_when_no_errors(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code", "x = 1"),
    ])
    result = assemble_project(state, tmp_path / "out", task_paths={"task_001": "x.py"})
    assert result.success
    assert result.errors == []
    assert result.verify_returncode is None


def test_assembly_result_output_dir_is_absolute(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code", "x = 1"),
    ])
    result = assemble_project(state, tmp_path / "out")
    assert result.output_dir.is_absolute()


# ── ProjectAssembler class API ────────────────────────────────────────────────

def test_project_assembler_class_direct_usage(tmp_path):
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code", "```python\npass\n```"),
        ("task_002", TaskType.WRITING,  "docs", "# Docs"),
    ])
    assembler = ProjectAssembler(
        state,
        task_paths={"task_001": "src/app.py", "task_002": "docs/index.md"},
    )
    result = assembler.assemble(tmp_path / "project")
    assert (tmp_path / "project" / "src" / "app.py").exists()
    assert (tmp_path / "project" / "docs" / "index.md").exists()
    assert result.success


def test_project_assembler_empty_state(tmp_path):
    """Assembler should not crash on an empty state."""
    state = ProjectState(
        project_description="empty",
        success_criteria="n/a",
        budget=Budget(max_usd=1.0),
        status=ProjectStatus.SUCCESS,
    )
    result = assemble_project(state, tmp_path / "out")
    assert result.success
    assert result.files_written == []


# ── Verify command ────────────────────────────────────────────────────────────

def test_assemble_verify_cmd_success(tmp_path):
    """A trivially successful verify_cmd (echo) records returncode=0."""
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code", "x = 1"),
    ])
    result = assemble_project(
        state, tmp_path / "out",
        task_paths={"task_001": "x.py"},
        verify_cmd="echo OK",
    )
    assert result.verify_returncode == 0
    assert result.success


def test_assemble_verify_cmd_failure_marks_result(tmp_path):
    """A failing verify_cmd sets success=False."""
    state = _make_state([
        ("task_001", TaskType.CODE_GEN, "code", "x = 1"),
    ])
    result = assemble_project(
        state, tmp_path / "out",
        task_paths={"task_001": "x.py"},
        # 'exit 1' fails on all platforms; 'false' on Unix
        verify_cmd="python -c \"import sys; sys.exit(1)\"",
    )
    assert result.verify_returncode == 1
    assert not result.success
    assert result.errors  # error recorded
