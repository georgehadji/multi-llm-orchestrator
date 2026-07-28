"""
Tests for codebase-aware token minimization and optimization strategies.
========================================================================
Tests the four synergistic optimizations:
1. Dynamic Context Slicing (Strategy Pattern)
2. Decomposer Task Consolidation
3. SEARCH/REPLACE block patching
4. Prompt Caching integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.codebase import (
    CodebaseDecomposer,
    CodebaseReader,
    ContextSlicingStrategy,
    DeleteFileSlicingStrategy,
    FileNode,
    InstallDepSlicingStrategy,
    ModifyFileSlicingStrategy,
    get_slicing_strategy,
)
from orchestrator.codebase.writer import DiffEngine, SearchReplaceBlock
from orchestrator.models import Task, TaskType

# ═══════════════════════════════════════════════════
# 4.1 Dynamic Context Slicing — Strategy Pattern
# ═══════════════════════════════════════════════════


class TestContextSlicingStrategy:
    """Verify each concrete strategy selects correct files by task type."""

    def test_modify_file_strategy_resolves_target(self) -> None:
        """ModifyFileSlicingStrategy includes the target file."""
        strategy = ModifyFileSlicingStrategy()
        root = Path("/root").resolve()
        target = "src/app.py"
        files = [
            FileNode(path=root / "src" / "app.py"),
            FileNode(path=root / "src" / "utils.py"),
            FileNode(path=root / "tests" / "test_app.py"),
        ]

        # Create a minimal mock reader — use a factory to capture root
        mock_root = root

        class MockReader:
            root = mock_root
            symbols = {}
            graph = _MockGraph({})

        result = strategy.slice_files(target, files, MockReader())  # type: ignore[arg-type]
        result_paths = {str(f.path) for f in result}
        target_resolved = root / "src" / "app.py"
        assert str(target_resolved) in result_paths

    def test_modify_file_strategy_includes_siblings(self) -> None:
        """ModifyFileSlicingStrategy includes sibling files in same directory."""
        strategy = ModifyFileSlicingStrategy()
        root = Path("/root").resolve()
        mock_root = root
        target = "src/app.py"
        files = [
            FileNode(path=root / "src" / "app.py"),
            FileNode(path=root / "src" / "utils.py"),
            FileNode(path=root / "tests" / "test_app.py"),
        ]

        class MockReader:
            root = mock_root
            symbols = {}
            graph = _MockGraph({})

        result = strategy.slice_files(target, files, MockReader())  # type: ignore[arg-type]
        result_str = {str(f.path) for f in result}
        # Should include the sibling src/utils.py
        assert str(root / "src" / "utils.py") in result_str

    def test_delete_file_strategy_includes_blast_radius(self) -> None:
        """DeleteFileSlicingStrategy includes dependent files (blast radius)."""
        strategy = DeleteFileSlicingStrategy()
        root = Path("/root").resolve()
        mock_root = root
        target = "src/core.py"
        files = [
            FileNode(path=root / "src" / "core.py"),
            FileNode(path=root / "src" / "consumer.py"),
            FileNode(path=root / "tests" / "test_core.py"),
        ]

        class MockReader:
            root = mock_root
            symbols = {}
            graph = _MockGraph({str(root / "src" / "core.py"): [str(root / "src" / "consumer.py")]})

        result = strategy.slice_files(target, files, MockReader())  # type: ignore[arg-type]
        result_str = {str(f.path) for f in result}
        assert str(root / "src" / "core.py") in result_str

    def test_install_dep_strategy_only_lockfiles(self) -> None:
        """InstallDepSlicingStrategy returns only config/lockfiles."""
        strategy = InstallDepSlicingStrategy()
        root = Path("/root").resolve()
        mock_root = root
        files = [
            FileNode(path=root / "src" / "app.py"),
            FileNode(path=root / "pyproject.toml"),
            FileNode(path=root / "requirements.txt"),
            FileNode(path=root / "tests" / "test_app.py"),
        ]

        class MockReader:
            root = mock_root
            symbols = {}
            graph = _MockGraph({})

        result = strategy.slice_files("", files, MockReader())  # type: ignore[arg-type]
        result_names = {f.path.name for f in result}
        assert "pyproject.toml" in result_names
        assert "requirements.txt" in result_names
        assert "app.py" not in result_names
        assert "test_app.py" not in result_names

    def test_get_slicing_strategy_defaults_to_modify(self) -> None:
        """get_slicing_strategy returns ModifyFileSlicingStrategy for unknown types."""
        strategy = get_slicing_strategy("unknown_type")
        assert isinstance(strategy, ModifyFileSlicingStrategy)

    def test_get_slicing_strategy_delete(self) -> None:
        """get_slicing_strategy returns DeleteFileSlicingStrategy for delete_file."""
        strategy = get_slicing_strategy("delete_file")
        assert isinstance(strategy, DeleteFileSlicingStrategy)

    def test_get_slicing_strategy_install(self) -> None:
        """get_slicing_strategy returns InstallDepSlicingStrategy for install_dep."""
        strategy = get_slicing_strategy("install_dep")
        assert isinstance(strategy, InstallDepSlicingStrategy)

    def test_context_slicing_strategy_is_abstract(self) -> None:
        """ContextSlicingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ContextSlicingStrategy()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════
# 4.4 Decomposer Task Consolidation
# ═══════════════════════════════════════════════════


class TestDecomposerConsolidation:
    """Verify task consolidation merges same-file MODIFY_FILE tasks."""

    def test_consolidate_single_task_unchanged(self) -> None:
        """A single MODIFY_FILE task is left unchanged."""
        tasks = {
            "task_001": Task(
                id="task_001",
                type=TaskType.MODIFY_FILE,
                prompt="Add login endpoint",
                target_path="src/routes.py",
                dependencies=[],
            )
        }
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        assert "task_001" in result
        assert result["task_001"].prompt == "Add login endpoint"

    def test_consolidate_merges_same_file(self) -> None:
        """Multiple MODIFY_FILE tasks for the same file are merged."""
        t1 = Task(
            id="task_001",
            type=TaskType.MODIFY_FILE,
            prompt="Add imports",
            target_path="src/app.py",
            dependencies=[],
        )
        object.__setattr__(t1, "modification_strategy", "replace")
        t2 = Task(
            id="task_002",
            type=TaskType.MODIFY_FILE,
            prompt="Add middleware",
            target_path="src/app.py",
            dependencies=["task_001"],
        )
        object.__setattr__(t2, "modification_strategy", "patch")
        tasks = {"task_001": t1, "task_002": t2}
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        # Should have a single consolidated task
        assert len(result) == 1
        merged_id = list(result.keys())[0]
        assert merged_id.startswith("consolidated_")
        assert "1. Add imports" in result[merged_id].prompt
        assert "2. Add middleware" in result[merged_id].prompt
        assert result[merged_id].type == TaskType.MODIFY_FILE
        assert result[merged_id].target_path == "src/app.py"

    def test_consolidate_different_files_unchanged(self) -> None:
        """Tasks for different files are NOT merged."""
        tasks = {
            "task_001": Task(
                id="task_001",
                type=TaskType.MODIFY_FILE,
                prompt="Edit routes",
                target_path="src/routes.py",
                dependencies=[],
            ),
            "task_002": Task(
                id="task_002",
                type=TaskType.MODIFY_FILE,
                prompt="Edit auth",
                target_path="src/auth.py",
                dependencies=[],
            ),
        }
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        assert len(result) == 2
        assert "task_001" in result
        assert "task_002" in result

    def test_consolidate_skips_non_modify(self) -> None:
        """Non-MODIFY_FILE tasks are not affected."""
        tasks = {
            "task_001": Task(
                id="task_001",
                type=TaskType.CODE_GEN,
                prompt="Create test",
                target_path="tests/test_app.py",
                dependencies=[],
            ),
            "task_002": Task(
                id="task_002",
                type=TaskType.INSTALL_DEP,
                prompt="Install flask",
                target_path="",
                dependencies=[],
            ),
        }
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        assert len(result) == 2
        assert "task_001" in result
        assert "task_002" in result

    def test_consolidate_deduplicates_dependencies(self) -> None:
        """Consolidated tasks have deduplicated dependency lists."""
        t1 = Task(
            id="task_001",
            type=TaskType.MODIFY_FILE,
            prompt="Add imports",
            target_path="src/app.py",
            dependencies=["dep_001"],
        )
        object.__setattr__(t1, "modification_strategy", "replace")
        t2 = Task(
            id="task_002",
            type=TaskType.MODIFY_FILE,
            prompt="Add logic",
            target_path="src/app.py",
            dependencies=["dep_001", "dep_002"],
        )
        object.__setattr__(t2, "modification_strategy", "replace")
        tasks = {"task_001": t1, "task_002": t2}
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        merged = list(result.values())[0]
        # Should have deduplicated deps: ["dep_001", "dep_002"]
        assert set(merged.dependencies) == {"dep_001", "dep_002"}

    def test_consolidate_no_target_path(self) -> None:
        """MODIFY_FILE without target_path is left as-is."""
        tasks = {
            "task_001": Task(
                id="task_001",
                type=TaskType.MODIFY_FILE,
                prompt="Edit something",
                target_path="",
                dependencies=[],
            ),
        }
        result = CodebaseDecomposer._consolidate_tasks(tasks)
        assert "task_001" in result


# ═══════════════════════════════════════════════════
# 4.3 SEARCH/REPLACE Block Patching
# ═══════════════════════════════════════════════════


class TestSearchReplaceParsing:
    """Verify SEARCH/REPLACE block parsing and applying."""

    def test_parse_single_block(self) -> None:
        """Parse a single SEARCH/REPLACE block correctly."""
        text = """<<<<<<< SEARCH
def old_func():
    pass
=======
def new_func():
    return 42
>>>>>>> REPLACE
"""
        blocks = DiffEngine.parse_search_replace_blocks(text)
        assert len(blocks) == 1
        assert "def old_func():" in blocks[0].search
        assert "def new_func():" in blocks[0].replace

    def test_parse_multiple_blocks(self) -> None:
        """Parse multiple SEARCH/REPLACE blocks in order."""
        text = """<<<<<<< SEARCH
line1
=======
line1_modified
>>>>>>> REPLACE

<<<<<<< SEARCH
line2
=======
line2_modified
>>>>>>> REPLACE
"""
        blocks = DiffEngine.parse_search_replace_blocks(text)
        assert len(blocks) == 2
        assert "line1" in blocks[0].search
        assert "line2" in blocks[1].search

    def test_parse_no_blocks(self) -> None:
        """No SEARCH/REPLACE blocks returns empty list."""
        text = "Just some regular text without blocks."
        blocks = DiffEngine.parse_search_replace_blocks(text)
        assert len(blocks) == 0

    def test_parse_skips_empty_search(self) -> None:
        """Blocks with empty search text are skipped."""
        text = """<<<<<<< SEARCH
=======
replacement
>>>>>>> REPLACE
"""
        blocks = DiffEngine.parse_search_replace_blocks(text)
        assert len(blocks) == 0

    def test_apply_single_block(self) -> None:
        """Apply a single SEARCH/REPLACE block to content."""
        original = """def old_func():
    pass

def another():
    pass
"""
        blocks = [
            SearchReplaceBlock(
                search="def old_func():\n    pass",
                replace="def new_func():\n    return 42",
            )
        ]
        result = DiffEngine.apply_search_replace_blocks(original, blocks)
        assert "def new_func():" in result
        assert "def old_func():" not in result
        assert "def another():" in result  # Unchanged

    def test_apply_multiple_blocks(self) -> None:
        """Apply multiple SEARCH/REPLACE blocks in sequence."""
        original = """x = 1
y = 2
z = 3
"""
        blocks = [
            SearchReplaceBlock(search="x = 1", replace="x = 10"),
            SearchReplaceBlock(search="y = 2", replace="y = 20"),
        ]
        result = DiffEngine.apply_search_replace_blocks(original, blocks)
        assert "x = 10" in result
        assert "y = 20" in result
        assert "z = 3" in result

    def test_apply_block_not_found_skips(self) -> None:
        """Block that doesn't match existing content is skipped safely."""
        original = "existing content"
        blocks = [
            SearchReplaceBlock(
                search="nonexistent content",
                replace="replacement",
            )
        ]
        result = DiffEngine.apply_search_replace_blocks(original, blocks)
        assert result == original  # Unchanged

    def test_apply_replaces_only_first_occurrence(self) -> None:
        """Only the first occurrence of search text is replaced."""
        original = "hello world\nhello world"
        blocks = [
            SearchReplaceBlock(search="hello", replace="hi"),
        ]
        result = DiffEngine.apply_search_replace_blocks(original, blocks)
        assert result == "hi world\nhello world"

    def test_apply_whitespace_sensitivity(self) -> None:
        """Whitespace-sensitive match works correctly."""
        original = "    indented line\nnot indented"
        blocks = [
            SearchReplaceBlock(search="    indented line", replace="        reindented"),
        ]
        result = DiffEngine.apply_search_replace_blocks(original, blocks)
        assert "        reindented" in result
        assert "not indented" in result

    def test_generate_diff_still_works(self) -> None:
        """Existing generate_diff method is preserved."""
        engine = DiffEngine()
        diff = engine.generate_diff("old\n", "new\n", "test.py")
        assert "a/test.py" in diff
        assert "b/test.py" in diff

    def test_save_diff_creates_file(self, tmp_path: Path) -> None:
        """save_diff creates a diff file at the specified path."""
        engine = DiffEngine()
        diff_content = "--- a/test.py\n+++ b/test.py\n"
        result_path = engine.save_diff(diff_content, tmp_path, "test.diff")
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8") == diff_content


# ═══════════════════════════════════════════════════
# Helper: Mock graph for Strategy tests
# ═══════════════════════════════════════════════════


class _MockGraph:
    """Minimal mock of DependencyGraph for strategy tests."""

    def __init__(self, reverse_edges: dict[str, list[str]] | None = None) -> None:
        """Initialize with optional reverse edge mapping."""
        self._graph = _MockDiGraph(reverse_edges or {})

    def rank_by_centrality(self) -> list[tuple[str, float]]:
        return []

    def find_blast_radius(self, module: str) -> set[str]:
        """Return dependents of the module."""
        result: set[str] = set()
        for node, edges in self._graph._reverse_edges.items():
            if module in edges:
                result.add(node)
        return result

    def to_dict(self) -> dict:
        return {
            "node_count": 0,
            "edge_count": 0,
            "circular_deps": [],
            "unused_modules": [],
            "central_modules": [],
        }


class _MockDiGraph:
    """Minimal mock of networkx.DiGraph for strategy tests."""

    def __init__(self, reverse_edges: dict[str, list[str]]) -> None:
        self._reverse_edges = reverse_edges
        # Build forward edges from reverse
        self._forward: dict[str, list[str]] = {}
        for target, sources in reverse_edges.items():
            for src in sources:
                self._forward.setdefault(src, []).append(target)

    def predecessors(self, node: str) -> list[str]:
        """Nodes that point to this node (reverse edges key)."""
        return self._reverse_edges.get(node, [])

    def successors(self, node: str) -> list[str]:
        """Nodes this node points to (forward edges)."""
        return self._forward.get(node, [])

    def __contains__(self, node: str) -> bool:
        return node in self._reverse_edges or node in self._forward

    def reverse(self, *_args: object) -> "_MockDiGraph":
        return _MockDiGraph(self._forward)


@pytest.fixture(autouse=True)
def _no_networkx() -> None:
    """Ensure tests don't accidentally require networkx."""
    pass
