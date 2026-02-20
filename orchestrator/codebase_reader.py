"""
Codebase Reader — Scans a directory and builds an LLM-ready context string
===========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads source files from a directory tree, respects ignore patterns,
and produces a compact, token-efficient context representation suitable
for passing to LLMs for analysis.

Features:
- Auto-detects language from extension
- Skips binaries, generated files, and common noise directories
- Respects .gitignore patterns (if present)
- Enforces a max-token budget to keep context LLM-friendly
- Produces a file tree summary + per-file content blocks
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator.reader")

# ─────────────────────────────────────────────
# Default ignore patterns (always skipped)
# ─────────────────────────────────────────────

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".next", ".nuxt", "dist", "build", "out",
    ".venv", "venv", "env", ".env",
    ".idea", ".vscode",
    "coverage", ".coverage",
    "eggs", "*.egg-info",
}

DEFAULT_IGNORE_FILES = {
    "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dll", "*.dylib",
    "*.class", "*.jar",
    "*.min.js", "*.min.css",
    "*.lock",           # package-lock.json, yarn.lock, Pipfile.lock
    "*.map",            # source maps
    "*.log",
    "*.db", "*.sqlite", "*.sqlite3",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg", "*.webp",
    "*.pdf", "*.zip", "*.tar", "*.gz", "*.7z",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    ".DS_Store", "Thumbs.db",
}

# Extension → language name (for syntax context in output)
EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "jsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".less": "less",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown", ".rst": "rst",
    ".tex": "latex",
    ".r": "r", ".R": "r",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell",
    ".tf": "terraform", ".tfvars": "terraform",
    ".dockerfile": "dockerfile",
}

# Approximate chars-per-token for budget estimation
CHARS_PER_TOKEN = 4


@dataclass
class FileEntry:
    """A single source file's metadata and content."""
    path: Path          # absolute path
    rel_path: str       # path relative to the scanned root
    language: str       # detected language (e.g. "python")
    size_bytes: int
    line_count: int
    content: str        # full text content


@dataclass
class CodebaseContext:
    """
    Fully assembled context ready for LLM consumption.

    Attributes
    ----------
    root           : Scanned root directory
    files          : All FileEntry objects (ordered by relative path)
    file_tree      : ASCII tree of the directory structure
    context_text   : Full assembled text to pass to LLM
    total_files    : Total files discovered (before budget trimming)
    included_files : Files actually included in context_text
    skipped_files  : Files skipped due to token budget
    estimated_tokens : Rough token count of context_text
    languages      : Set of detected programming languages
    """
    root: Path
    files: list[FileEntry] = field(default_factory=list)
    file_tree: str = ""
    context_text: str = ""
    total_files: int = 0
    included_files: int = 0
    skipped_files: int = 0
    estimated_tokens: int = 0
    languages: set[str] = field(default_factory=set)


class CodebaseReader:
    """
    Scans a directory tree and produces an LLM-ready CodebaseContext.

    Parameters
    ----------
    max_tokens      : Approximate token budget for the context string.
                      Files are included largest-first until budget is hit.
                      Default: 80_000 (~320KB of text).
    include_exts    : If set, only files with these extensions are included.
                      Example: {".py", ".ts"}
    extra_ignore_dirs  : Additional directory names to skip.
    extra_ignore_files : Additional glob patterns for files to skip.
    max_file_tokens : Max tokens to include per individual file
                      (large files are truncated with a notice).
    """

    def __init__(
        self,
        max_tokens: int = 80_000,
        include_exts: Optional[set[str]] = None,
        extra_ignore_dirs: Optional[set[str]] = None,
        extra_ignore_files: Optional[set[str]] = None,
        max_file_tokens: int = 8_000,
    ):
        self.max_tokens = max_tokens
        self.include_exts = include_exts
        self.ignore_dirs = DEFAULT_IGNORE_DIRS | (extra_ignore_dirs or set())
        self.ignore_file_patterns = DEFAULT_IGNORE_FILES | (extra_ignore_files or set())
        self.max_file_tokens = max_file_tokens

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def read(self, root: str | Path) -> CodebaseContext:
        """
        Scan `root` and return a CodebaseContext.

        Steps:
        1. Walk directory tree, collect eligible files
        2. Load & parse .gitignore (optional)
        3. Sort files by size descending (largest first, most informative)
        4. Include files until token budget is exhausted
        5. Build file tree and assemble context_text
        """
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root}")

        gitignore_patterns = self._load_gitignore(root)
        all_files = self._scan(root, gitignore_patterns)

        ctx = CodebaseContext(root=root)
        ctx.total_files = len(all_files)
        ctx.languages = {f.language for f in all_files if f.language}

        # Sort: prioritise key config files first, then by size descending
        all_files.sort(key=lambda f: (self._priority(f.rel_path), -f.size_bytes))

        # Fill token budget
        budget_chars = self.max_tokens * CHARS_PER_TOKEN
        used_chars = 0
        included: list[FileEntry] = []
        skipped: list[FileEntry] = []

        for fe in all_files:
            file_chars = len(fe.content)
            if used_chars + file_chars > budget_chars and included:
                skipped.append(fe)
            else:
                # Truncate individual file if over per-file limit
                max_file_chars = self.max_file_tokens * CHARS_PER_TOKEN
                if len(fe.content) > max_file_chars:
                    fe.content = (
                        fe.content[:max_file_chars]
                        + f"\n\n... [TRUNCATED — {fe.line_count} lines total, "
                        f"showing first {max_file_chars // CHARS_PER_TOKEN} tokens] ..."
                    )
                included.append(fe)
                used_chars += len(fe.content)

        ctx.files = included
        ctx.included_files = len(included)
        ctx.skipped_files = len(skipped)
        ctx.file_tree = self._build_tree(root, all_files)
        ctx.context_text = self._assemble(root, included, skipped)
        ctx.estimated_tokens = len(ctx.context_text) // CHARS_PER_TOKEN

        logger.info(
            f"Codebase scan: {ctx.total_files} files found, "
            f"{ctx.included_files} included, {ctx.skipped_files} skipped. "
            f"~{ctx.estimated_tokens:,} tokens. "
            f"Languages: {', '.join(sorted(ctx.languages)) or 'unknown'}"
        )
        return ctx

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _scan(self, root: Path, gitignore: set[str]) -> list[FileEntry]:
        """Recursively walk root and return FileEntry objects for included files."""
        entries: list[FileEntry] = []

        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            rel_dir = current.relative_to(root)

            # Prune ignored directories (modifying dirnames in-place stops os.walk descent)
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore_dir(d, str(rel_dir / d), gitignore)
            ]
            dirnames.sort()  # deterministic order

            for fname in sorted(filenames):
                fpath = current / fname
                rel_path = str(fpath.relative_to(root)).replace("\\", "/")

                if self._should_ignore_file(fname, rel_path, gitignore):
                    continue

                ext = Path(fname).suffix.lower()
                if self.include_exts and ext not in self.include_exts:
                    continue

                lang = EXT_LANG.get(ext, "")
                # Special case: Dockerfile has no extension
                if not lang and fname.lower() in ("dockerfile", "containerfile"):
                    lang = "dockerfile"

                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    size = fpath.stat().st_size
                    line_count = content.count("\n") + 1
                    entries.append(FileEntry(
                        path=fpath,
                        rel_path=rel_path,
                        language=lang,
                        size_bytes=size,
                        line_count=line_count,
                        content=content,
                    ))
                except (OSError, PermissionError) as e:
                    logger.debug(f"Skipping {rel_path}: {e}")

        return entries

    def _should_ignore_dir(self, name: str, rel: str, gitignore: set[str]) -> bool:
        if name in self.ignore_dirs:
            return True
        for pat in gitignore:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _should_ignore_file(self, name: str, rel: str, gitignore: set[str]) -> bool:
        for pat in self.ignore_file_patterns:
            if fnmatch.fnmatch(name, pat):
                return True
        for pat in gitignore:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _load_gitignore(self, root: Path) -> set[str]:
        """Parse .gitignore at root and return a set of glob patterns."""
        gi = root / ".gitignore"
        if not gi.exists():
            return set()
        patterns: set[str] = set()
        try:
            for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Strip leading slash (gitignore-specific)
                    patterns.add(line.lstrip("/"))
        except OSError:
            pass
        return patterns

    def _priority(self, rel_path: str) -> int:
        """
        Return a sort priority (lower = shown first).
        Key config/entrypoint files get priority 0, everything else gets 1.
        """
        name = rel_path.split("/")[-1].lower()
        priority_names = {
            "readme.md", "readme.rst", "readme.txt",
            "pyproject.toml", "setup.py", "setup.cfg",
            "package.json", "tsconfig.json",
            "cargo.toml", "go.mod",
            "makefile", "dockerfile",
            "requirements.txt", "requirements-dev.txt",
            "main.py", "app.py", "index.py", "server.py",
            "main.ts", "index.ts", "app.ts",
            "main.go", "main.rs",
        }
        return 0 if name in priority_names else 1

    def _build_tree(self, root: Path, files: list[FileEntry]) -> str:
        """Build a compact ASCII directory tree from the file list."""
        tree_lines: list[str] = [f"{root.name}/"]
        # Collect unique directories
        dirs_seen: set[str] = set()
        items: list[str] = []
        for fe in files:
            parts = fe.rel_path.split("/")
            # Add intermediate directories
            for i in range(len(parts) - 1):
                d = "/".join(parts[:i + 1])
                if d not in dirs_seen:
                    dirs_seen.add(d)
                    items.append(("dir", d))
            items.append(("file", fe.rel_path))

        seen: set[str] = set()
        for kind, path in items:
            if path in seen:
                continue
            seen.add(path)
            depth = path.count("/")
            indent = "  " * depth
            name = path.split("/")[-1]
            prefix = "📁 " if kind == "dir" else "📄 "
            tree_lines.append(f"{indent}{prefix}{name}")

        return "\n".join(tree_lines)

    def _assemble(
        self,
        root: Path,
        included: list[FileEntry],
        skipped: list[FileEntry],
    ) -> str:
        """Assemble the final context string for LLM consumption."""
        parts: list[str] = []

        # Header
        parts.append(f"# Codebase: {root.name}")
        parts.append(f"Root: {root}")
        languages = sorted({f.language for f in included if f.language})
        parts.append(f"Languages: {', '.join(languages) or 'mixed'}")
        parts.append(f"Files included: {len(included)} | Skipped (budget): {len(skipped)}")
        parts.append("")

        # File tree
        parts.append("## Directory Structure")
        parts.append(self._build_tree(root, included))
        parts.append("")

        if skipped:
            parts.append(
                f"## Note: {len(skipped)} file(s) omitted due to token budget "
                f"({', '.join(f.rel_path for f in skipped[:5])}"
                + (f" and {len(skipped)-5} more" if len(skipped) > 5 else "")
                + ")"
            )
            parts.append("")

        # File contents
        parts.append("## Source Files")
        parts.append("")
        for fe in included:
            lang_tag = fe.language or ""
            parts.append(f"### {fe.rel_path}")
            parts.append(f"*{fe.line_count} lines | {fe.size_bytes:,} bytes*")
            parts.append("")
            parts.append(f"```{lang_tag}")
            parts.append(fe.content)
            parts.append("```")
            parts.append("")

        return "\n".join(parts)
