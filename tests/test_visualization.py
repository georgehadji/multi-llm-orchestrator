# tests/test_visualization.py
from orchestrator.visualization import DagRenderer
from orchestrator.models import Task, TaskType


def _make_task(tid, deps=None, task_type=TaskType.CODE_GEN):
    return Task(
        id=tid,
        type=task_type,
        prompt=f"Prompt for {tid}",
        dependencies=deps or [],
        hard_validators=[],
    )


def test_mermaid_output_contains_all_nodes():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t1"]),
        "t4": _make_task("t4", deps=["t2", "t3"]),
    }
    renderer = DagRenderer(tasks)
    mermaid = renderer.to_mermaid()
    assert "t1" in mermaid
    assert "t2" in mermaid
    assert "t3" in mermaid
    assert "t4" in mermaid
    assert "graph TD" in mermaid or "flowchart TD" in mermaid


def test_mermaid_output_contains_edges():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    renderer = DagRenderer(tasks)
    mermaid = renderer.to_mermaid()
    assert "t1" in mermaid and "t2" in mermaid
    assert "-->" in mermaid


def test_critical_path_single_chain():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t2"]),
    }
    renderer = DagRenderer(tasks)
    path = renderer.critical_path()
    assert path == ["t1", "t2", "t3"]


def test_critical_path_diamond():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
        "t3": _make_task("t3", deps=["t1"]),
        "t4": _make_task("t4", deps=["t2", "t3"]),
    }
    renderer = DagRenderer(tasks)
    path = renderer.critical_path()
    assert path[0] == "t1"
    assert path[-1] == "t4"
    assert len(path) == 3


def test_ascii_output_has_rows_for_each_task():
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    renderer = DagRenderer(tasks)
    ascii_out = renderer.to_ascii()
    assert "t1" in ascii_out
    assert "t2" in ascii_out


def test_dependency_report_includes_context_size():
    from orchestrator.models import TaskResult, Model, TaskStatus
    tasks = {
        "t1": _make_task("t1"),
        "t2": _make_task("t2", deps=["t1"]),
    }
    results = {
        "t1": TaskResult(
            task_id="t1",
            output="x" * 5000,
            score=0.9,
            model_used=Model.DEEPSEEK_CHAT,
            status=TaskStatus.COMPLETED,
        )
    }
    renderer = DagRenderer(tasks, results=results)
    report = renderer.dependency_report()
    assert "t2" in report
    assert "5000" in report or "5,000" in report or "chars" in report.lower()


def test_single_node_critical_path():
    tasks = {"t1": _make_task("t1")}
    renderer = DagRenderer(tasks)
    path = renderer.critical_path()
    assert path == ["t1"]


def test_empty_dag_critical_path():
    renderer = DagRenderer({})
    path = renderer.critical_path()
    assert path == []
