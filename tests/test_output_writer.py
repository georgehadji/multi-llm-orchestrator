"""
Tests for orchestrator/output_writer.py
========================================
Pure unit tests — no API calls, no SQLite, no LLM.

Run with:
    python -m pytest tests/test_output_writer.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from orchestrator.models import (
    Budget, Model, ProjectState, ProjectStatus,
    Task, TaskResult, TaskStatus, TaskType,
)
from orchestrator.output_writer import (
    write_output_dir,
    _extract_python,
    _ext_for,
    _render_content,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_simple_state(
    task_type: TaskType = TaskType.CODE_GEN,
    output: str = '```python\ndef hello():\n    return "Hello!"\n```',
    status: TaskStatus = TaskStatus.COMPLETED,
    score: float = 0.95,
) -> tuple[ProjectState, str]:
    """Create a minimal one-task ProjectState for testing."""
    task = Task(
        id="task_001",
        type=task_type,
        prompt="Write a hello world function",
    )
    result = TaskResult(
        task_id="task_001",
        output=output,
        score=score,
        model_used=Model.CLAUDE_SONNET,
        status=status,
        iterations=1,
        cost_usd=0.002,
    )
    state = ProjectState(
        project_description="Test project",
        success_criteria="Say hello",
        budget=Budget(max_usd=8.0),
        tasks={"task_001": task},
        results={"task_001": result},
        status=ProjectStatus.SUCCESS,
        execution_order=["task_001"],
    )
    return state, "test-project-001"


# ── Directory creation ────────────────────────────────────────────────────────

def test_creates_output_directory():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "results"
        assert not out.exists()
        write_output_dir(state, out, project_id=pid)
        assert out.exists()


def test_returns_resolved_path():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        result_path = write_output_dir(state, tmp, project_id=pid)
        assert result_path.is_absolute()
        assert result_path.exists()


# ── File extension mapping ────────────────────────────────────────────────────

@pytest.mark.parametrize("task_type,expected_ext", [
    (TaskType.CODE_GEN,     ".py"),
    (TaskType.CODE_REVIEW,  ".md"),
    (TaskType.REASONING,    ".md"),
    (TaskType.WRITING,      ".md"),
    (TaskType.SUMMARIZE,    ".md"),
    (TaskType.EVALUATE,     ".md"),
])
def test_extension_mapping(task_type, expected_ext):
    assert _ext_for(task_type, "some output") == expected_ext


def test_data_extract_json_when_valid_json():
    assert _ext_for(TaskType.DATA_EXTRACT, '{"key": "value"}') == ".json"


def test_data_extract_fallback_to_md_when_not_json():
    assert _ext_for(TaskType.DATA_EXTRACT, "This is just prose.") == ".md"


def test_data_extract_fallback_with_fenced_invalid_json():
    assert _ext_for(TaskType.DATA_EXTRACT, "```\nnot json\n```") == ".md"


# ── Python extraction ─────────────────────────────────────────────────────────

def test_extract_python_from_fenced_block():
    text = '```python\ndef foo():\n    pass\n```'
    result = _extract_python(text)
    assert "def foo():" in result
    assert "```" not in result


def test_extract_python_from_generic_fence():
    text = '```\ndef foo():\n    pass\n```'
    result = _extract_python(text)
    assert "def foo():" in result
    assert "```" not in result


def test_extract_python_heuristic_from_prose():
    text = "Here is the implementation:\n\nimport re\nclass Foo:\n    pass"
    result = _extract_python(text)
    assert result.startswith("import re")


def test_extract_python_fallback_for_pure_prose():
    text = "This is just a description with no code."
    result = _extract_python(text)
    assert result == text  # returns as-is


# ── .py file writing ──────────────────────────────────────────────────────────

def test_code_gen_writes_py_file():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        py_file = Path(tmp) / "task_001_code_generation.py"
        assert py_file.exists()
        content = py_file.read_text(encoding="utf-8")
        assert 'def hello():' in content
        assert "```" not in content  # fences must be stripped


def test_code_gen_empty_output_not_written():
    state, pid = _make_simple_state(output="")
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        py_file = Path(tmp) / "task_001_code_generation.py"
        assert not py_file.exists()


# ── .md file writing ──────────────────────────────────────────────────────────

def test_code_review_writes_md_file():
    state, pid = _make_simple_state(
        task_type=TaskType.CODE_REVIEW,
        output="The code looks good. Consider adding type hints.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        md_file = Path(tmp) / "task_001_code_review.md"
        assert md_file.exists()
        assert "type hints" in md_file.read_text(encoding="utf-8")


# ── .json file writing ────────────────────────────────────────────────────────

def test_data_extract_writes_json_file_when_valid():
    state, pid = _make_simple_state(
        task_type=TaskType.DATA_EXTRACT,
        output='{"users": [{"id": 1, "name": "Alice"}]}',
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        json_file = Path(tmp) / "task_001_data_extraction.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["users"][0]["name"] == "Alice"


def test_data_extract_fallback_md_when_prose():
    state, pid = _make_simple_state(
        task_type=TaskType.DATA_EXTRACT,
        output="Found 3 items in the dataset.",
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        assert (Path(tmp) / "task_001_data_extraction.md").exists()
        assert not (Path(tmp) / "task_001_data_extraction.json").exists()


# ── summary.json ──────────────────────────────────────────────────────────────

def test_summary_json_exists():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        assert (Path(tmp) / "summary.json").exists()


def test_summary_json_structure():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        summary = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
        assert summary["project_id"] == pid
        assert summary["status"] == "SUCCESS"
        assert summary["project_description"] == "Test project"
        assert len(summary["tasks"]) == 1
        assert summary["tasks"][0]["task_id"] == "task_001"
        assert summary["tasks"][0]["output_file"] == "task_001_code_generation.py"
        assert summary["tasks"][0]["score"] == 0.95
        assert summary["totals"]["tasks_completed"] == 1
        assert summary["totals"]["tasks_failed"] == 0
        assert summary["totals"]["tasks_degraded"] == 0
        assert "generated_at" in summary
        assert "budget" in summary


def test_summary_json_includes_raw_output():
    output = '```python\ndef foo(): pass\n```'
    state, pid = _make_simple_state(output=output)
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        summary = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
        assert summary["tasks"][0]["output"] == output


def test_summary_counts_failed_and_degraded():
    task_ok  = Task(id="task_001", type=TaskType.CODE_GEN,    prompt="p1")
    task_fail= Task(id="task_002", type=TaskType.CODE_REVIEW, prompt="p2")
    task_deg = Task(id="task_003", type=TaskType.SUMMARIZE,   prompt="p3")
    state = ProjectState(
        project_description="Multi",
        success_criteria="all",
        budget=Budget(),
        tasks={"task_001": task_ok, "task_002": task_fail, "task_003": task_deg},
        results={
            "task_001": TaskResult("task_001", "def f(): pass", 0.9, Model.GPT_4O,
                                   status=TaskStatus.COMPLETED),
            "task_002": TaskResult("task_002", "",             0.0, Model.GPT_4O_MINI,
                                   status=TaskStatus.FAILED),
            "task_003": TaskResult("task_003", "summary text", 0.6, Model.GEMINI_FLASH,
                                   status=TaskStatus.DEGRADED),
        },
        status=ProjectStatus.PARTIAL_SUCCESS,
        execution_order=["task_001", "task_002", "task_003"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id="multi")
        s = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
        assert s["totals"]["tasks_completed"] == 1
        assert s["totals"]["tasks_failed"] == 1
        assert s["totals"]["tasks_degraded"] == 1


# ── README.md ─────────────────────────────────────────────────────────────────

def test_readme_exists():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        assert (Path(tmp) / "README.md").exists()


def test_readme_contains_key_info():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        readme = (Path(tmp) / "README.md").read_text(encoding="utf-8")
        assert "Test project" in readme
        assert "SUCCESS" in readme
        assert pid in readme
        assert "task_001_code_generation.py" in readme
        assert "summary.json" in readme


def test_readme_table_row_per_task():
    state, pid = _make_simple_state()
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id=pid)
        readme = (Path(tmp) / "README.md").read_text(encoding="utf-8")
        # Table row should contain the filename and score
        assert "task_001_code_generation.py" in readme
        assert "0.950" in readme


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_state_writes_only_summary_and_readme():
    state = ProjectState(
        project_description="Empty",
        success_criteria="nothing",
        budget=Budget(),
        tasks={},
        results={},
        status=ProjectStatus.SYSTEM_FAILURE,
        execution_order=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id="empty")
        files = list(Path(tmp).iterdir())
        names = {f.name for f in files}
        assert "summary.json" in names
        assert "README.md" in names
        # No task files
        assert len(names) == 2


def test_task_missing_from_tasks_dict_is_skipped():
    """Result exists but Task object missing (e.g. skipped dependency)."""
    result = TaskResult("task_999", "", 0.0, Model.GPT_4O_MINI,
                        status=TaskStatus.FAILED)
    state = ProjectState(
        project_description="Test",
        success_criteria="ok",
        budget=Budget(),
        tasks={},           # task_999 not in tasks dict
        results={"task_999": result},
        status=ProjectStatus.PARTIAL_SUCCESS,
        execution_order=["task_999"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_output_dir(state, tmp, project_id="test")
        # Should not crash, just write summary + readme
        assert (Path(tmp) / "summary.json").exists()
