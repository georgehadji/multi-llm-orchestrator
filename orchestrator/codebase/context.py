"""
CodebaseContext — Relevance ranker, LLM context builder, quality analyzer
==========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 2 of the Codebase-Aware Orchestrator enhancement.
Builds on CodebaseReader (Phase 1) to produce:
- Relevance-ranked files/symbols for a given objective
- Compressed LLM context string within token budget
- Quality analysis findings from static analysis tools

Optimization: Dynamic Context Slicing (4.1)
- Uses Strategy Pattern to prune context based on task type.
- Modifications: target file + AST neighbors + immediate dependents
- Deletions: target path + its imports + dependents from graph
- Dependency installs: only lockfiles and config files
"""

from __future__ import annotations

import abc
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reader import CodebaseReader, FileNode, ProjectProfiler, Symbol

logger = logging.getLogger("orchestrator.codebase_context")

# ─────────────────────────────────────────────
# 4.1 Dynamic Context Slicing — Strategy Pattern
# ─────────────────────────────────────────────


class ContextSlicingStrategy(abc.ABC):
    """Abstract strategy for pruning the file list based on task type.

    Each concrete strategy receives the full file list and reader
    and returns a subset of files relevant for its task type.
    """

    @abc.abstractmethod
    def slice_files(
        self,
        target_path: str,
        files: list[FileNode],
        reader: CodebaseReader,
    ) -> list[FileNode]:
        """Return a pruned list of FileNodes relevant for the task."""
        ...

    def _resolve_path(self, target_path: str, root: Path) -> Path:
        """Resolve a potentially-relative target path against the root."""
        p = Path(target_path)
        if not p.is_absolute():
            p = root / p
        return p.resolve()

    def _neighbors(
        self, resolved: Path, symbols: dict[Path, list[Symbol]], graph: Any, files: list[FileNode]
    ) -> set[Path]:
        """Collect AST-level neighbor files (import siblings + dependents)."""
        neighbor_paths: set[Path] = set()
        str_resolved = str(resolved)

        # 1. Modules that import from the target (reverse graph)
        if str_resolved in graph._graph:
            for pred in graph._graph.predecessors(str_resolved):
                p = Path(pred)
                if p.exists():
                    neighbor_paths.add(p)

        # 2. Module's own import targets (forward graph)
        if str_resolved in graph._graph:
            for succ in graph._graph.successors(str_resolved):
                p = Path(succ)
                if p.exists():
                    neighbor_paths.add(p)

        # 3. Sibling files in the same directory
        parent = resolved.parent
        for fn in files:
            if fn.path.parent == parent and fn.path != resolved:
                neighbor_paths.add(fn.path)

        return neighbor_paths


class ModifyFileSlicingStrategy(ContextSlicingStrategy):
    """Slice context around a MODIFY_FILE task.

    Returns:
    - The target file itself
    - Its AST import neighbors (forward graph)
    - Its immediate dependents (reverse graph)
    - Sibling files in the same directory
    """

    def slice_files(
        self,
        target_path: str,
        files: list[FileNode],
        reader: CodebaseReader,
    ) -> list[FileNode]:
        resolved = self._resolve_path(target_path, reader.root)
        included: set[Path] = {resolved}

        # Collect AST neighbors
        neighbor_paths = self._neighbors(resolved, reader.symbols, reader.graph, files)
        included.update(neighbor_paths)

        # Filter and return
        result = [fn for fn in files if fn.path in included]
        if not result:
            # Fallback: return the target file only
            matching = [fn for fn in files if fn.path == resolved]
            if matching:
                return matching
        return result


class DeleteFileSlicingStrategy(ContextSlicingStrategy):
    """Slice context around a DELETE_FILE task.

    Returns:
    - The target file itself
    - Its direct import dependencies
    - All transitive dependents (blast radius)
    """

    def slice_files(
        self,
        target_path: str,
        files: list[FileNode],
        reader: CodebaseReader,
    ) -> list[FileNode]:
        resolved = self._resolve_path(target_path, reader.root)
        included: set[Path] = {resolved}

        str_resolved = str(resolved)

        # Blast radius: all transitive dependents
        if str_resolved in reader.graph._graph:
            try:
                blast = reader.graph.find_blast_radius(str_resolved)
                for node_key in blast:
                    p = Path(node_key)
                    if p.exists():
                        included.add(p)
            except Exception:
                pass

        # Also include forward imports
        if str_resolved in reader.graph._graph:
            for succ in reader.graph._graph.successors(str_resolved):
                p = Path(succ)
                if p.exists():
                    included.add(p)

        result = [fn for fn in files if fn.path in included]
        if not result:
            matching = [fn for fn in files if fn.path == resolved]
            if matching:
                return matching
        return result


class InstallDepSlicingStrategy(ContextSlicingStrategy):
    """Slice context around an INSTALL_DEP task.

    Returns:
    - Only lockfiles and config files (pyproject.toml, requirements.txt, etc.)
    """

    _LOCKFILE_NAMES = {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "Pipfile.lock",
        "poetry.lock",
        "Cargo.toml",
        "Cargo.lock",
        "go.mod",
        "go.sum",
        "package.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Gemfile",
        "Gemfile.lock",
        "composer.json",
        "composer.lock",
        "build.gradle",
        "pom.xml",
    }

    def slice_files(
        self,
        target_path: str,
        files: list[FileNode],
        reader: CodebaseReader,
    ) -> list[FileNode]:
        return [fn for fn in files if fn.path.name.lower() in self._LOCKFILE_NAMES]


# Strategy registry — maps TaskType (string) -> strategy instance
_SLICING_STRATEGIES: dict[str, ContextSlicingStrategy] = {}


def get_slicing_strategy(task_type: str) -> ContextSlicingStrategy:
    """Get the appropriate slicing strategy for a task type."""
    if not _SLICING_STRATEGIES:
        _SLICING_STRATEGIES["modify_file"] = ModifyFileSlicingStrategy()
        _SLICING_STRATEGIES["delete_file"] = DeleteFileSlicingStrategy()
        _SLICING_STRATEGIES["install_dep"] = InstallDepSlicingStrategy()
    return _SLICING_STRATEGIES.get(task_type, ModifyFileSlicingStrategy())


# ─────────────────────────────────────────────
# 2.1 RelevanceRanker
# ─────────────────────────────────────────────


@dataclass
class RelevanceScore:
    """Relevance score for a file or symbol."""

    path: str
    score: float
    reason: str  # "keyword_match", "graph_centrality", "name_match"


class RelevanceRanker:
    """Rank files and symbols by relevance to a given objective.

    Uses three signals combined into a weighted score:
    1. Keyword matching — objective keywords vs file/symbol names
    2. Graph centrality — PageRank from DependencyGraph
    3. Name proximity — fuzzy matches between objective and symbol names
    """

    def __init__(
        self,
        keyword_weight: float = 0.5,
        centrality_weight: float = 0.3,
        name_weight: float = 0.2,
    ) -> None:
        self._kw_weight = keyword_weight
        self._central_weight = centrality_weight
        self._name_weight = name_weight

    def rank_files(
        self,
        objective: str,
        files: list[FileNode],
        reader: CodebaseReader | None = None,
    ) -> list[tuple[Path, float]]:
        """Rank files by relevance to the objective.

        Args:
            objective: The task objective (e.g. "Add authentication").
            files: List of FileNode objects from the walker.
            reader: Optional CodebaseReader for graph centrality data.

        Returns:
            List of (file_path, score) sorted by score descending.
        """
        keywords = self._extract_keywords(objective)
        centrality: dict[str, float] = {}
        if reader is not None:
            centrality = dict(reader.graph.rank_by_centrality())

        scored: list[tuple[Path, float]] = []
        for fn in files:
            score = 0.0
            reasons: list[str] = []

            # Keyword matching on path and filename
            path_lower = str(fn.path).lower()
            name_lower = fn.path.stem.lower()
            for kw in keywords:
                if kw in path_lower:
                    score += self._kw_weight
                    reasons.append("keyword_match")
                    break

            # Graph centrality bonus
            str_path = str(fn.path)
            if str_path in centrality:
                cent_score = centrality[str_path]
                score += self._central_weight * min(cent_score * 10, 1.0)
                reasons.append("graph_centrality")

            # Name proximity — check if objective words appear in filename
            obj_words = set(objective.lower().split())
            name_words = set(re.split(r"[_\-.\\/]", name_lower))
            overlap = obj_words & name_words
            if overlap:
                score += self._name_weight * min(len(overlap) / max(len(name_words), 1), 1.0)
                reasons.append("name_match")

            if score > 0 or len(scored) < 100:  # keep at least 100 files
                scored.append((fn.path, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_symbols(
        self,
        objective: str,
        symbols: dict[Path, list[Symbol]],
        top_n: int = 50,
    ) -> list[tuple[Symbol, float]]:
        """Rank symbols by relevance to the objective.

        Args:
            objective: The task objective.
            symbols: Dict mapping file path -> symbols from ASTIndexer.
            top_n: Maximum number of symbols to return.

        Returns:
            List of (symbol, score) sorted by score descending.
        """
        keywords = self._extract_keywords(objective)
        obj_words = set(objective.lower().split())

        scored: list[tuple[Symbol, float]] = []
        for file_path, syms in symbols.items():
            for sym in syms:
                score = 0.0
                name_lower = sym.name.lower()

                # Exact keyword match in symbol name
                for kw in keywords:
                    if kw in name_lower:
                        score += 1.0
                        break

                # Docstring keyword match
                if sym.docstring:
                    for kw in keywords:
                        if kw in sym.docstring.lower():
                            score += 0.5
                            break

                # Name proximity
                sym_words = set(re.split(r"_|(?=[A-Z])", name_lower))
                overlap = obj_words & sym_words
                if overlap:
                    score += 0.3 * min(len(overlap) / max(len(sym_words), 1), 1.0)

                if score > 0:
                    scored.append((sym, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from an objective string.

        Strips common stop words, splits on spaces/camelCase/snake_case.
        """
        stop_words = {
            "a",
            "an",
            "the",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "add",
            "new",
            "create",
            "make",
            "implement",
            "update",
            "fix",
        }

        words: set[str] = set()
        # Split on whitespace, camelCase, and snake_case
        for token in re.split(r"[\s_]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", text):
            token = token.strip().lower()
            if len(token) > 2 and token not in stop_words:
                words.add(token)

        # Also add the full objective as a single phrase match
        return sorted(words)


# ─────────────────────────────────────────────
# 2.2 CodebaseContext
# ─────────────────────────────────────────────


class CodebaseContext:
    """Build a token-budgeted LLM context string from CodebaseReader output.

    The context follows a structured format:

    1. Project overview (profile summary)
    2. Top-N most relevant files (ranked by objective)
    3. Key symbols near the objective
    4. Relevant test files
    5. Coverage gaps and quality findings (if any)
    6. Dependency graph structure (compressed)
    """

    def __init__(
        self,
        reader: CodebaseReader,
        max_tokens: int = 8192,
    ) -> None:
        self._reader = reader
        self._max_chars = max_tokens * 4  # rough estimate: 4 chars per token
        self._ranker = RelevanceRanker()

    def to_llm_prompt(
        self,
        objective: str = "",
        include_quality: bool = False,
        findings: list[AnalysisFinding] | None = None,
    ) -> str:
        """Build a structured context string for the LLM.

        Args:
            objective: The task objective. If empty, returns a general project overview.
            include_quality: If True, include quality analysis findings.
            findings: Pre-computed analysis findings from QualityAnalyzer.

        Returns:
            A string within the token budget, suitable for LLM system prompt.
        """
        parts: list[str] = []
        char_budget = self._max_chars
        used = 0

        # 1. Project overview
        overview = self._build_overview()
        parts.append(overview)
        used += len(overview)

        # 2. Ranked files (most relevant first)
        if objective:
            ranked = self._ranker.rank_files(objective, self._reader.files, self._reader)
            file_section = self._build_file_section(ranked)
            remaining = char_budget - used - 500  # reserve space for other sections
            if len(file_section) > remaining:
                file_section = self._truncate_section(file_section, remaining)
            parts.append(file_section)
            used += len(file_section)

            # 3. Key symbols
            ranked_syms = self._ranker.rank_symbols(objective, self._reader.symbols)
            if ranked_syms:
                sym_section = self._build_symbol_section(ranked_syms)
                remaining = char_budget - used - 300
                if len(sym_section) > remaining:
                    sym_section = self._truncate_section(sym_section, remaining)
                parts.append(sym_section)
                used += len(sym_section)

        # 4. Test gaps
        if self._reader.profile:
            gaps = ProjectProfiler().find_coverage_gaps(self._reader.root, self._reader.files)
            if gaps:
                gap_section = self._build_gap_section(gaps)
                remaining = char_budget - used - 200
                if len(gap_section) > remaining:
                    gap_section = self._truncate_section(gap_section, remaining)
                parts.append(gap_section)
                used += len(gap_section)

        # 5. Quality findings
        if include_quality and findings:
            quality_section = self._build_quality_section(findings)
            remaining = char_budget - used - 200
            if len(quality_section) > remaining:
                quality_section = self._truncate_section(quality_section, remaining)
            parts.append(quality_section)
            used += len(quality_section)

        # 6. Dependency graph summary
        graph_stats = self._reader.graph.to_dict()
        if graph_stats["node_count"] > 0:
            graph_section = self._build_graph_section(graph_stats)
            remaining = char_budget - used - 100
            if len(graph_section) > remaining:
                graph_section = self._truncate_section(graph_section, remaining)
            parts.append(graph_section)

        result = "\n\n".join(parts)
        return result.strip()

    def slice_to_context(
        self,
        target_path: str,
        task_type: str,
        objective: str = "",
    ) -> str:
        """Build a task-specialized context string using a slicing strategy.

        This is the entry point for Dynamic Context Slicing (Optimization 4.1).
        Instead of including all files, it uses the Strategy Pattern to select
        only files relevant to the given task type, reducing token usage by up to 70%.

        Args:
            target_path: The file path targeted by the task.
            task_type: One of "modify_file", "delete_file", "install_dep".
            objective: Optional objective string for relevance ranking within the slice.

        Returns:
            A context string within the token budget, pruned to the task type.
        """
        strategy = get_slicing_strategy(task_type)
        sliced_files = strategy.slice_files(target_path, self._reader.files, self._reader)

        original_files = self._reader.files[:]
        self._reader.files = sliced_files

        try:
            result = self.to_llm_prompt(objective=objective)
        finally:
            self._reader.files = original_files

        return result

    def _build_overview(self) -> str:
        """Build the project overview section."""
        profile = self._reader.profile
        if profile is None:
            return "# Project Overview\n\n(Profile not available — run reader.read() first)"

        lines = [
            "# Project Overview",
            "",
            f"- **Languages:** {', '.join(profile.languages)}",
            f"- **Framework:** {profile.framework or 'unknown'}",
            f"- **Package Manager:** {profile.package_manager or 'unknown'}",
            f"- **Source Files:** {profile.file_count}",
            f"- **Total Lines:** {profile.loc:,}",
            f"- **Test Files:** {profile.test_count}",
        ]

        if profile.entry_points:
            eps = ", ".join(str(p.relative_to(self._reader.root)) for p in profile.entry_points)
            lines.append(f"- **Entry Points:** {eps}")

        lines.extend(
            [
                f"- **Has Docker:** {profile.has_docker}",
                f"- **Has CI:** {profile.has_ci}",
                "",
            ]
        )

        return "\n".join(lines)

    def _build_file_section(self, ranked: list[tuple[Path, float]]) -> str:
        """Build the ranked files section."""
        lines = ["## Relevant Files", ""]
        for path, score in ranked[:30]:  # top 30 files
            try:
                rel = path.relative_to(self._reader.root)
            except ValueError:
                rel = path
            lines.append(f"- **{rel}** (relevance: {score})")
        lines.append("")
        return "\n".join(lines)

    def _build_symbol_section(self, ranked: list[tuple[Symbol, float]]) -> str:
        """Build the ranked symbols section."""
        lines = ["## Key Symbols", ""]
        for sym, score in ranked[:20]:  # top 20 symbols
            try:
                rel = sym.file.relative_to(self._reader.root)
            except ValueError:
                rel = sym.file
            lines.append(
                f"- **{sym.name}** ({sym.type}) — `{rel}` L{sym.line_start} "
                f"— relevance: {score}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_gap_section(self, gaps: list[str]) -> str:
        """Build the coverage gaps section."""
        lines = ["## Coverage Gaps", ""]
        for gap in gaps[:15]:  # top 15 gaps
            lines.append(f"- `{gap}` — no corresponding test file")
        if len(gaps) > 15:
            lines.append(f"- ... and {len(gaps) - 15} more")
        lines.append("")
        return "\n".join(lines)

    def _build_quality_section(self, findings: list[AnalysisFinding]) -> str:
        """Build the quality findings section."""
        lines = ["## Quality Analysis", ""]
        # Group by severity
        by_severity: dict[str, list[AnalysisFinding]] = {}
        for f in findings:
            by_severity.setdefault(f.severity, []).append(f)

        for severity in ("error", "warning", "info"):
            items = by_severity.get(severity, [])
            for item in items[:10]:
                rel = str(item.file)[:60]
                lines.append(f"- [{severity.upper()}] `{rel}` L{item.line}: {item.message}")
        if sum(len(v) for v in by_severity.values()) > 30:
            lines.append("- ... (truncated)")
        lines.append("")
        return "\n".join(lines)

    def _build_graph_section(self, stats: dict[str, Any]) -> str:
        """Build the dependency graph summary section."""
        lines = ["## Dependency Graph", ""]
        lines.append(f"- **Modules:** {stats['node_count']}")
        lines.append(f"- **Dependencies:** {stats['edge_count']}")
        if stats["circular_deps"]:
            lines.append(f"- **Circular Dependencies:** {len(stats['circular_deps'])}")
        if stats["unused_modules"]:
            lines.append(f"- **Unused Modules:** {len(stats['unused_modules'])}")
        if stats["central_modules"]:
            lines.append("- **Most Central Modules:**")
            for item in stats["central_modules"][:5]:
                lines.append(f"  - {item['module']} (score: {item['score']})")
        lines.append("")
        return "\n".join(lines)

    def _truncate_section(self, text: str, max_chars: int) -> str:
        """Truncate a section to fit within max_chars."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"


# ───────────────────────────────────────────────
# Helper for ProjectProfiler in this module
# ───────────────────────────────────────────────


# ─────────────────────────────────────────────
# 2.3 QualityAnalyzer
# ─────────────────────────────────────────────


@dataclass
class AnalysisFinding:
    """A single quality analysis finding."""

    severity: str  # "error", "warning", "info"
    category: str  # "bug", "security", "performance", "style", "complexity", "type"
    file: Path
    line: int
    message: str
    suggestion: str | None = None


class QualityAnalyzer:
    """Run static analysis tools and collect findings.

    Uses subprocess to call:
    - bandit (security scanning)
    - radon (complexity metrics)
    - mypy (type checking, if pyproject.toml present)
    - ruff (linting, if config present)

    All tools are optional — missing tools are skipped gracefully.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    async def analyze(self) -> list[AnalysisFinding]:
        """Run all available analysis tools and return combined findings.

        Returns:
            List of AnalysisFinding objects. Empty if no tools available.
        """
        findings: list[AnalysisFinding] = []

        # Try each tool — failures are logged, not raised
        try:
            findings.extend(self._run_bandit())
        except Exception as exc:
            logger.debug("bandit analysis failed: %s", exc)

        try:
            findings.extend(self._run_radon())
        except Exception as exc:
            logger.debug("radon analysis failed: %s", exc)

        try:
            findings.extend(self._run_mypy())
        except Exception as exc:
            logger.debug("mypy analysis failed: %s", exc)

        try:
            findings.extend(self._run_ruff())
        except Exception as exc:
            logger.debug("ruff analysis failed: %s", exc)

        return findings

    def _run_bandit(self) -> list[AnalysisFinding]:
        """Run bandit security scanner."""
        return self._run_tool(
            ["python", "-m", "bandit", "-r", str(self.root), "-f", "json"],
            parse_fn=self._parse_bandit_output,
        )

    def _run_radon(self) -> list[AnalysisFinding]:
        """Run radon complexity analyzer."""
        findings: list[AnalysisFinding] = []
        # radon cc — cyclomatic complexity
        cc_result = self._run_tool(
            ["python", "-m", "radon", "cc", str(self.root), "-s", "--json"],
            parse_fn=self._parse_radon_cc_output,
        )
        findings.extend(cc_result)

        # radon mi — maintainability index
        mi_result = self._run_tool(
            ["python", "-m", "radon", "mi", str(self.root), "-s", "--json"],
            parse_fn=self._parse_radon_mi_output,
        )
        findings.extend(mi_result)
        return findings

    def _run_mypy(self) -> list[AnalysisFinding]:
        """Run mypy type checker if pyproject.toml exists."""
        if (
            not (self.root / "pyproject.toml").exists()
            and not (self.root / "mypy.ini").exists()
            and not (self.root / ".mypy.ini").exists()
        ):
            return []

        return self._run_tool(
            ["python", "-m", "mypy", str(self.root), "--show-error-codes", "--no-error-summary"],
            parse_fn=self._parse_mypy_output,
        )

    def _run_ruff(self) -> list[AnalysisFinding]:
        """Run ruff linter if config exists."""
        if (
            not (self.root / "pyproject.toml").exists()
            and not (self.root / "ruff.toml").exists()
            and not (self.root / ".ruff.toml").exists()
        ):
            return []

        return self._run_tool(
            ["python", "-m", "ruff", "check", str(self.root), "--format", "json"],
            parse_fn=self._parse_ruff_output,
        )

    def _run_tool(
        self,
        cmd: list[str],
        parse_fn: callable,
    ) -> list[AnalysisFinding]:
        """Run a CLI tool and parse its output.

        Args:
            cmd: Command list for subprocess.
            parse_fn: Function to parse stdout into list[AnalysisFinding].

        Returns:
            List of findings. Empty if tool unavailable or failed.
        """
        import subprocess

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.root,
            )
            if result.returncode in (0, 1):  # tools often return 1 for findings
                return parse_fn(result.stdout)
            return []
        except FileNotFoundError:
            logger.debug("Tool not found: %s", cmd[0])
            return []
        except subprocess.TimeoutExpired:
            logger.debug("Tool timed out: %s", cmd[0])
            return []

    def _parse_bandit_output(self, stdout: str) -> list[AnalysisFinding]:
        """Parse bandit JSON output."""
        findings: list[AnalysisFinding] = []
        try:
            data = json.loads(stdout)
            for result in data.get("results", []):
                findings.append(
                    AnalysisFinding(
                        severity=result.get("issue_severity", "warning").lower(),
                        category="security",
                        file=Path(result.get("filename", "")),
                        line=int(result.get("line_number", 0)),
                        message=result.get("issue_text", ""),
                        suggestion=result.get("code", ""),
                    )
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return findings

    def _parse_radon_cc_output(self, stdout: str) -> list[AnalysisFinding]:
        """Parse radon cyclomatic complexity JSON output."""
        findings: list[AnalysisFinding] = []
        try:
            data = json.loads(stdout)
            for filepath, blocks in data.items():
                for block in blocks:
                    complexity = block.get("complexity", 0)
                    if isinstance(complexity, (int, float)) and complexity > 10:
                        findings.append(
                            AnalysisFinding(
                                severity="warning" if complexity > 15 else "info",
                                category="complexity",
                                file=Path(filepath),
                                line=int(block.get("lineno", 0)),
                                message=(
                                    f"{block.get('name', 'unknown')} has "
                                    f"cyclomatic complexity {complexity} "
                                    f"(threshold: 10)"
                                ),
                                suggestion=f"Consider refactoring into smaller functions",
                            )
                        )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return findings

    def _parse_radon_mi_output(self, stdout: str) -> list[AnalysisFinding]:
        """Parse radon maintainability index JSON output."""
        findings: list[AnalysisFinding] = []
        try:
            data = json.loads(stdout)
            for filepath, mi in data.items():
                if isinstance(mi, dict):
                    score = mi.get("mi", 100)
                    if isinstance(score, (int, float)) and score < 65:
                        findings.append(
                            AnalysisFinding(
                                severity="warning" if score < 40 else "info",
                                category="complexity",
                                file=Path(filepath),
                                line=1,
                                message=f"Maintainability index: {score:.1f} (threshold: 65)",
                                suggestion="Consider refactoring to improve maintainability",
                            )
                        )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return findings

    def _parse_mypy_output(self, stdout: str) -> list[AnalysisFinding]:
        """Parse mypy text output."""
        findings: list[AnalysisFinding] = []
        for line in stdout.splitlines():
            # Format: file:line:error: message
            match = re.match(r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+)$", line)
            if match:
                findings.append(
                    AnalysisFinding(
                        severity=match.group(3).lower(),
                        category="type",
                        file=Path(match.group(1)),
                        line=int(match.group(2)),
                        message=match.group(4),
                    )
                )
        return findings

    def _parse_ruff_output(self, stdout: str) -> list[AnalysisFinding]:
        """Parse ruff JSON output."""
        findings: list[AnalysisFinding] = []
        try:
            data = json.loads(stdout)
            for result in data:
                findings.append(
                    AnalysisFinding(
                        severity="warning",
                        category="style",
                        file=Path(result.get("filename", "")),
                        line=int(result.get("location", {}).get("row", 0)),
                        message=result.get("message", ""),
                        suggestion=result.get("code", ""),
                    )
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return findings

    async def find_coverage_gaps(self) -> list[str]:
        """Find Python modules without corresponding test files.

        Returns:
            List of module paths (relative to root) missing test coverage.
        """
        from .reader import ProjectProfiler, FileSystemWalker

        walker = FileSystemWalker(self.root)
        files = walker.walk(extensions={".py"})
        profiler = ProjectProfiler()
        return profiler.find_coverage_gaps(self.root, files)
