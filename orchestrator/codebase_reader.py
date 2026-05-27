"""
CodebaseReader — File system walker, AST indexer, dependency graph, project profiler
====================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 1 of the Codebase-Aware Orchestrator enhancement.
Reads an existing codebase and produces structured representations:
- File inventory (respecting .gitignore)
- AST-level symbol index (classes, functions, imports)
- Dependency graph (module-level)
- Project profile (language, framework, test gaps)
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger("orchestrator.codebase_reader")

# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────


@dataclass
class FileNode:
    """A single source file found during walk."""

    path: Path
    language: str = ""
    size: int = 0
    lines: int = 0


@dataclass
class Symbol:
    """A named symbol (class, function, method, variable) defined in source code."""

    name: str
    type: str  # "class", "function", "method", "variable"
    file: Path
    line_start: int
    line_end: int
    docstring: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ProjectProfile:
    """High-level project metadata."""

    languages: list[str] = field(default_factory=list)
    framework: str | None = None
    package_manager: str | None = None
    test_framework: str | None = None
    entry_points: list[Path] = field(default_factory=list)
    has_docker: bool = False
    has_ci: bool = False
    loc: int = 0
    file_count: int = 0
    test_count: int = 0


# ─────────────────────────────────────────────
# 1.1 FileSystemWalker
# ─────────────────────────────────────────────

_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".php": "php",
}


class FileSystemWalker:
    """Walk a directory tree, respect .gitignore, collect source files.

    Args:
        root: Root directory to walk.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._ignored = self._load_gitignore()

    def _load_gitignore(self) -> Any:
        """Load .gitignore patterns using pathspec if available."""
        try:
            import pathspec
        except ImportError:
            return None

        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.exists():
            return None

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                spec_text = f.read()
            return pathspec.PathSpec.from_lines(
                pathspec.patterns.GitWildMatchPattern, spec_text.splitlines()
            )
        except Exception as exc:
            logger.warning("Failed to load .gitignore: %s", exc)
            return None

    def _is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        if self._ignored is None:
            return False
        try:
            rel = str(path.relative_to(self.root))
            return self._ignored.match_file(rel) if rel != "." else False
        except ValueError:
            return False

    def walk(
        self,
        extensions: set[str] | None = None,
        include_hidden: bool = False,
    ) -> list[FileNode]:
        """Walk the directory tree and return matching source files.

        Args:
            extensions: Set of file extensions to include. Defaults to Python/Js/Ts.
            include_hidden: If True, include dot-files and dot-directories.

        Returns:
            List of FileNode objects for matching source files.
        """
        if extensions is None:
            extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java"}

        results: list[FileNode] = []
        for root_dir, dirs, files in os.walk(self.root):
            root_path = Path(root_dir)

            # Early skip hidden directories unless explicitly included
            if not include_hidden:
                # Remove hidden directories from traversal (in-place)
                dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(root_path / d)]

            for filename in files:
                filepath = root_path / filename
                if self._is_ignored(filepath):
                    continue
                if not include_hidden and filename.startswith("."):
                    continue

                ext = filepath.suffix.lower()
                if ext not in extensions:
                    continue
                if filepath.is_symlink():
                    continue

                try:
                    stat = filepath.stat()
                    with open(filepath, "r", encoding="utf-8") as fh:
                        line_count = sum(1 for _ in fh)
                except (OSError, UnicodeDecodeError):
                    continue

                results.append(
                    FileNode(
                        path=filepath,
                        language=_LANGUAGE_MAP.get(ext, ext.lstrip(".")),
                        size=stat.st_size,
                        lines=line_count,
                    )
                )

        results.sort(key=lambda n: str(n.path))
        return results


# ─────────────────────────────────────────────
# 1.2 ASTIndexer
# ─────────────────────────────────────────────


class ASTIndexer:
    """Parse source files and extract symbol definitions and dependencies.

    Uses Python's built-in ``ast`` module for Python files.
    For other languages, falls back to regex-based extraction.
    """

    def index_file(self, path: Path) -> list[Symbol]:
        """Parse a single file and return its symbols.

        Args:
            path: Path to a source file.

        Returns:
            List of Symbol objects defined in this file.
        """
        ext = path.suffix.lower()
        if ext == ".py":
            return self._index_python(path)
        # Fallback: try generic regex-based extraction
        return self._index_generic(path)

    def index_files(self, files: list[FileNode]) -> dict[Path, list[Symbol]]:
        """Index multiple files in parallel.

        Args:
            files: List of FileNode objects from FileSystemWalker.

        Returns:
            Dict mapping file path -> list of Symbol.
        """
        result: dict[Path, list[Symbol]] = {}
        for fn in files:
            try:
                symbols = self.index_file(fn.path)
                if symbols:
                    result[fn.path] = symbols
            except Exception as exc:
                logger.debug("Failed to index %s: %s", fn.path, exc)
        return result

    def _index_python(self, path: Path) -> list[Symbol]:
        """Parse Python source with the ``ast`` module."""
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            logger.debug("Syntax error in %s — skipping AST index", path)
            return []

        symbols: list[Symbol] = []
        imports: list[str] = []

        for node in ast.walk(tree):
            # Collect imports for dependency tracking
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full = f"{module}.{alias.name}" if module else alias.name
                    imports.append(full)

            # Class definitions
            if isinstance(node, ast.ClassDef):
                deps = self._extract_dependencies(node)
                docstring = ast.get_docstring(node) or ""
                symbols.append(
                    Symbol(
                        name=node.name,
                        type="class",
                        file=path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=docstring,
                        dependencies=deps,
                    )
                )

            # Function definitions (top-level, not methods)
            if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                deps = self._extract_dependencies(node)
                docstring = ast.get_docstring(node) or ""
                symbols.append(
                    Symbol(
                        name=node.name,
                        type="function",
                        file=path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=docstring,
                        dependencies=deps,
                    )
                )

        # If no symbols found but there are imports, create a module-level symbol
        if not symbols and imports:
            symbols.append(
                Symbol(
                    name=path.stem,
                    type="module",
                    file=path,
                    line_start=1,
                    line_end=1,
                    docstring="",
                    dependencies=imports,
                )
            )

        return symbols

    def _is_method(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if a function is a method (inside a class body)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for child in parent.body:
                    if child is node:
                        return True
        return False

    def _extract_dependencies(self, node: ast.AST) -> list[str]:
        """Extract function calls and name references from a node."""
        deps: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                deps.append(child.func.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                deps.append(child.func.attr)
        return list(set(deps))

    def _index_generic(self, path: Path) -> list[Symbol]:
        """Generic regex-based symbol extraction for non-Python files."""
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        symbols: list[Symbol] = []

        # Class-like patterns (language-agnostic)
        for match in re.finditer(
            r"(?:class|struct|interface|trait|type)\s+(\w+)",
            source,
        ):
            symbols.append(
                Symbol(
                    name=match.group(1),
                    type="class",
                    file=path,
                    line_start=source[: match.start()].count("\n") + 1,
                    line_end=source[: match.start()].count("\n") + 1,
                )
            )

        # Function-like patterns
        for match in re.finditer(
            r"(?:def|function|fn|func|async def)\s+(\w+)\s*\(",
            source,
        ):
            symbols.append(
                Symbol(
                    name=match.group(1),
                    type="function",
                    file=path,
                    line_start=source[: match.start()].count("\n") + 1,
                    line_end=source[: match.start()].count("\n") + 1,
                )
            )

        return symbols


# ─────────────────────────────────────────────
# 1.3 DependencyGraph
# ─────────────────────────────────────────────


class DependencyGraph:
    """Directed graph of module-level dependencies using networkx.

    Nodes are file paths, edges represent "depends on" relationships.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_module(self, path: Path, dependencies: list[str]) -> None:
        """Add a module and its dependency edges.

        Args:
            path: Path to the module file.
            dependencies: List of symbol names this module depends on.
        """
        str_path = str(path)
        if str_path not in self._graph:
            self._graph.add_node(str_path)

        for dep in dependencies:
            dep_str = self._resolve_dependency(dep)
            if dep_str:
                self._graph.add_edge(str_path, dep_str)

    def _resolve_dependency(self, name: str) -> str | None:
        """Resolve a dependency name to a graph node key."""
        if not name:
            return None
        # Map common imports to nodes that might exist
        # This is best-effort — nodes that don't exist are still tracked
        return name.replace(".", "/")

    def find_unused_modules(self) -> list[str]:
        """Find modules with zero dependents (excluding entry points).

        Returns:
            List of module node keys with no incoming edges.
        """
        candidates = []
        for node in self._graph.nodes():
            if self._graph.in_degree(node) == 0:
                candidates.append(node)
        return candidates

    def find_circular_dependencies(self) -> list[list[str]]:
        """Detect circular import chains.

        Returns:
            List of cycles, each a list of node keys.
        """
        try:
            cycles = list(nx.simple_cycles(self._graph))
            return cycles
        except nx.NetworkXNoCycle:
            return []

    def find_blast_radius(self, module: str) -> set[str]:
        """Find all transitive dependents of a module.

        Args:
            module: The module node key to check.

        Returns:
            Set of node keys that would be affected.
        """
        if module not in self._graph:
            return set()
        # Descendants in reverse graph = everything that depends on this module
        descendants = nx.descendants(self._graph.reverse(), module)
        return set(descendants)

    def rank_by_centrality(self) -> list[tuple[str, float]]:
        """Rank modules by PageRank centrality.

        Returns:
            List of (module_key, score) sorted by score descending.
        """
        if self._graph.number_of_nodes() == 0:
            return []
        pr = nx.pagerank(self._graph, alpha=0.85)
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to a dict for debugging/reporting."""
        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "unused_modules": self.find_unused_modules(),
            "circular_deps": self.find_circular_dependencies(),
            "central_modules": [
                {"module": m, "score": round(s, 4)} for m, s in self.rank_by_centrality()[:10]
            ],
        }

    def export_graphml(self, path: Path) -> None:
        """Export graph to GraphML format for visualization."""
        nx.write_graphml(self._graph, str(path))


# ─────────────────────────────────────────────
# 1.4 ProjectProfiler
# ─────────────────────────────────────────────

# Config file patterns for framework / language detection
_CONFIG_PATTERNS: dict[str, tuple[str, str]] = {
    "pyproject.toml": ("python", "pip"),
    "setup.py": ("python", "pip"),
    "setup.cfg": ("python", "pip"),
    "Pipfile": ("python", "pipenv"),
    "poetry.lock": ("python", "poetry"),
    "Cargo.toml": ("rust", "cargo"),
    "go.mod": ("go", "go"),
    "package.json": ("javascript", "npm"),
    "yarn.lock": ("javascript", "yarn"),
    "pnpm-lock.yaml": ("javascript", "pnpm"),
    "pnpm-lock.yml": ("javascript", "pnpm"),
    "Gemfile": ("ruby", "bundler"),
    "composer.json": ("php", "composer"),
    "build.gradle": ("kotlin", "gradle"),
    "pom.xml": ("java", "maven"),
}

_FRAMEWORK_DETECTORS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"from\s+fastapi\s+import|FastAPI\("), "python", "fastapi"),
    (re.compile(r"from\s+django\s+|django\.urls|django\.db"), "python", "django"),
    (re.compile(r"import\s+flask|from\s+flask\s+import"), "python", "flask"),
    (re.compile(r"from\s+react\s+|import\s+React\s+from"), "javascript", "react"),
    (re.compile(r"from\s+vue\s+|import\s+Vue\s+from|createApp"), "javascript", "vue"),
    (re.compile(r"@nestjs\|NestFactory"), "typescript", "nestjs"),
    (re.compile(r"from\s+express\s+|const\s+express\s*="), "javascript", "express"),
    (re.compile(r"use\s+actix|actix_web"), "rust", "actix"),
]

_TEST_FILE_PATTERNS = re.compile(
    r"(test_.*\.py$|.*_test\.py$|.*\.spec\.(ts|js|tsx|jsx)$|"
    r".*\.test\.(ts|js|tsx|jsx)$|.*_test\.go$|.*test\.rs$)"
)


class ProjectProfiler:
    """Detect project type, framework, and structure from config files."""

    def profile(self, root: Path, files: list[FileNode] | None = None) -> ProjectProfile:
        """Profile a project directory.

        Args:
            root: The project root directory.
            files: Optional pre-computed file list (from FileSystemWalker).

        Returns:
            A ProjectProfile describing the project.
        """
        profile = ProjectProfile()

        # Detect language and package manager from config files
        for config_file, (lang, pkg_mgr) in _CONFIG_PATTERNS.items():
            if (root / config_file).exists():
                profile.languages.append(lang)
                if pkg_mgr and not profile.package_manager:
                    profile.package_manager = pkg_mgr

        # Detect from file contents if not found yet
        if files:
            for fn in files:
                if fn.language not in profile.languages:
                    profile.languages.append(fn.language)

        # Detect framework from source file content
        framework_found = set()
        if files:
            for fn in list(files)[:50]:  # Check first 50 files
                try:
                    content = fn.path.read_text(encoding="utf-8", errors="replace")[:2000]
                except OSError:
                    continue
                for pattern, lang, fw in _FRAMEWORK_DETECTORS:
                    if (
                        lang in profile.languages or not profile.languages
                    ) and fw not in framework_found:
                        if pattern.search(content):
                            profile.framework = fw
                            framework_found.add(fw)

        # Detect entry points
        for ep_name in ("main.py", "app.py", "index.ts", "index.js", "main.ts", "main.rs"):
            ep = root / ep_name
            if ep.exists():
                profile.entry_points.append(ep)

        # Detect test framework
        if files:
            for fn in files:
                if "pytest" in str(fn.path) or fn.path.suffix == ".py":
                    try:
                        content = fn.path.read_text(encoding="utf-8", errors="replace")[:1000]
                        if "pytest" in content or "from pytest" in content:
                            profile.test_framework = "pytest"
                            break
                    except OSError:
                        continue

        if profile.languages:
            # Check for jest via package.json
            pkg_json = root / "package.json"
            if pkg_json.exists():
                try:
                    import json

                    pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                    dev_deps = pkg.get("devDependencies", {})
                    if "jest" in dev_deps:
                        profile.test_framework = "jest"
                    elif "vitest" in dev_deps:
                        profile.test_framework = "vitest"
                except (json.JSONDecodeError, OSError):
                    pass

        # Check for Docker and CI
        profile.has_docker = (root / "Dockerfile").exists() or (
            root / "docker-compose.yml"
        ).exists()
        profile.has_ci = (root / ".github" / "workflows").exists()

        # Count files and tests
        if files:
            profile.file_count = len(files)
            profile.loc = sum(f.lines for f in files)
            profile.test_count = sum(1 for f in files if _TEST_FILE_PATTERNS.search(f.path.name))

        profile.languages = list(set(profile.languages))
        return profile

    def find_coverage_gaps(self, root: Path, files: list[FileNode]) -> list[str]:
        """Find Python modules without corresponding test files.

        Args:
            root: Project root directory.
            files: List of files from FileSystemWalker.

        Returns:
            List of module paths (relative) that lack test coverage.
        """
        gaps: list[str] = []
        source_files = [f for f in files if f.language == "python"]
        test_files = {
            f.path.stem
            for f in files
            if _TEST_FILE_PATTERNS.search(f.path.name) and f.language == "python"
        }

        for sf in source_files:
            try:
                rel = sf.path.relative_to(root)
            except ValueError:
                continue
            # Check if there's a corresponding test file
            module_name = rel.with_suffix("").name
            test_name = f"test_{module_name}"
            if test_name not in test_files:
                # Check also _test suffix
                test_name2 = f"{module_name}_test"
                if test_name2 not in test_files:
                    gaps.append(str(rel))

        return gaps


# ─────────────────────────────────────────────
# 1.5 CodebaseReader (Assembly)
# ─────────────────────────────────────────────


class CodebaseReader:
    """Top-level reader that orchestrates walk → index → graph → profile.

    Usage::

        reader = CodebaseReader("/path/to/project")
        await reader.read()
        print(reader.profile)
        print(f"Found {len(reader.symbols)} symbols across {len(reader.files)} files")
        print(f"Dependency graph: {reader.graph.rank_by_centrality()[:5]}")
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.files: list[FileNode] = []
        self.symbols: dict[Path, list[Symbol]] = {}
        self.graph: DependencyGraph = DependencyGraph()
        self.profile: ProjectProfile | None = None

    async def read(self, quiet: bool = False) -> "CodebaseReader":
        """Execute the full read pipeline: walk → index → graph → profile.

        Args:
            quiet: If True, suppress logging.

        Returns:
            Self for chaining.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Root path does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {self.root}")

        # 1. Walk
        walker = FileSystemWalker(self.root)
        self.files = walker.walk()
        if not quiet:
            logger.info("CodebaseReader: found %d source files in %s", len(self.files), self.root)

        # 2. Index
        indexer = ASTIndexer()
        self.symbols = indexer.index_files(self.files)
        total_symbols = sum(len(syms) for syms in self.symbols.values())
        if not quiet:
            logger.info(
                "CodebaseReader: indexed %d symbols across %d files",
                total_symbols,
                len(self.symbols),
            )

        # 3. Build dependency graph
        for sym_path, symbols in self.symbols.items():
            for sym in symbols:
                self.graph.add_module(sym_path, sym.dependencies)
        if not quiet:
            stats = self.graph.to_dict()
            logger.info(
                "CodebaseReader: dependency graph has %d nodes, %d edges",
                stats["node_count"],
                stats["edge_count"],
            )
            if stats["circular_deps"]:
                logger.warning(
                    "CodebaseReader: found %d circular dependencies", len(stats["circular_deps"])
                )

        # 4. Profile
        profiler = ProjectProfiler()
        self.profile = profiler.profile(self.root, self.files)
        if not quiet:
            logger.info(
                "CodebaseReader: project profile — languages=%s, framework=%s",
                self.profile.languages,
                self.profile.framework,
            )
            if self.profile.test_count == 0:
                logger.warning("CodebaseReader: no test files detected")

        return self

    def find_symbol(self, name: str) -> list[Symbol]:
        """Find all symbols with a given name across the codebase.

        Args:
            name: Symbol name to search for.

        Returns:
            List of matching Symbol objects.
        """
        results = []
        for symbols in self.symbols.values():
            for sym in symbols:
                if sym.name == name:
                    results.append(sym)
        return results

    def find_file(self, pattern: str) -> list[Path]:
        """Find files matching a regex pattern.

        Args:
            pattern: Regex pattern to match against file paths.

        Returns:
            List of matching Path objects.
        """
        import re

        compiled = re.compile(pattern)
        return [f.path for f in self.files if compiled.search(str(f.path))]
