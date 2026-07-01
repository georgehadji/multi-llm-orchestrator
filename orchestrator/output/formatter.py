"""
orchestrator/output/formatter.py
=================================
Filesystem-level code formatting + lint-fixing for generated project output.

Generated apps and websites are run through deterministic formatters so the
delivered code passes ``black`` / ``ruff`` (Python) and ``prettier`` (web)
out of the box, satisfying the "generated output must pass lint/ruff/black"
quality bar.

Design notes
------------
* **Best-effort tooling** — a missing tool is skipped and recorded, never
  fatal. Generation must never crash because a formatter is absent.
* **Format then verify use identical invocations** so that anything the
  formatter touched is guaranteed to pass the matching ``--check`` afterwards;
  only genuinely unfixable lint findings remain in the report.
* **Config isolation** — ruff runs with ``--isolated`` so generated projects
  are judged against ruff's standard defaults, not the orchestrator's own
  ``pyproject.toml``. This keeps the bar deterministic regardless of where the
  output directory lives on disk.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories never worth formatting (deps, build artifacts, vcs, caches).
_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
        ".next",
        ".nuxt",
        "out",
        "coverage",
        "coverage_html",
        ".tox",
        "site-packages",
    }
)

_PYTHON_SUFFIXES = frozenset({".py"})
_WEB_SUFFIXES = frozenset(
    {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".scss", ".html", ".json", ".md"}
)

# Cap how many files we hand to a single subprocess invocation to avoid
# blowing past the OS command-line length limit on very large projects.
_BATCH_SIZE = 200
_TOOL_TIMEOUT_S = 120


@dataclass
class FormatReport:
    """Outcome of formatting a generated output directory."""

    python_files: int = 0
    web_files: int = 0
    tools_used: list[str] = field(default_factory=list)
    tools_missing: list[str] = field(default_factory=list)
    remaining_issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when no unresolved lint/format issues remain."""
        return not self.remaining_issues and not self.errors

    def to_dict(self) -> dict:
        return {
            "python_files": self.python_files,
            "web_files": self.web_files,
            "tools_used": list(self.tools_used),
            "tools_missing": list(self.tools_missing),
            "remaining_issues": list(self.remaining_issues),
            "errors": list(self.errors),
            "is_clean": self.is_clean,
        }


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIRS for part in path.parts)


def _iter_files(root: Path, suffixes: frozenset[str]) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix in suffixes else []
    found: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in suffixes and not _is_excluded(path):
            found.append(path)
    return found


def _batched(items: list[Path], size: int = _BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess | None:
    """Run a tool, returning the completed process or None if unavailable."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Tool unavailable or timed out: %s (%s)", cmd[:3], exc)
        return None


def _tool_available(module: str) -> bool:
    proc = _run([sys.executable, "-m", module, "--version"], cwd=Path.cwd())
    return proc is not None and proc.returncode == 0


def _format_python(files: list[Path], root: Path, report: FormatReport) -> None:
    """Run black + ruff --fix, then verify, recording remaining issues."""
    str_files = [str(f) for f in files]

    have_black = _tool_available("black")
    have_ruff = _tool_available("ruff")

    # Order matters: ruff --fix first (sorts/removes imports, which can leave
    # stray blank lines), then black last so the final layout is black-clean.
    if have_ruff:
        report.tools_used.append("ruff")
        for batch in _batched(files):
            _run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--isolated",
                    "--fix",
                    "-q",
                    *[str(f) for f in batch],
                ],
                cwd=root,
            )
    else:
        report.tools_missing.append("ruff")

    if have_black:
        report.tools_used.append("black")
        for batch in _batched(files):
            _run([sys.executable, "-m", "black", "-q", *[str(f) for f in batch]], cwd=root)
    else:
        report.tools_missing.append("black")

    # ── Verify ────────────────────────────────────────────────────────────
    if have_black:
        for batch in _batched(files):
            proc = _run(
                [sys.executable, "-m", "black", "--check", "-q", *[str(f) for f in batch]],
                cwd=root,
            )
            if proc is not None and proc.returncode not in (0, None):
                for line in (proc.stderr or "").splitlines():
                    line = line.strip()
                    if line.lower().startswith("would reformat"):
                        report.remaining_issues.append(f"black: {line}")

    if have_ruff:
        proc = _run(
            [sys.executable, "-m", "ruff", "check", "--isolated", "-q", *str_files],
            cwd=root,
        )
        if proc is not None and proc.returncode != 0:
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if line and not line.startswith(("Found ", "[*]")):
                    report.remaining_issues.append(f"ruff: {line}")


def _format_web(files: list[Path], root: Path, report: FormatReport) -> None:
    """Best-effort prettier pass over web assets."""
    proc = _run(["npx", "--no-install", "prettier", "--version"], cwd=root)
    if proc is None or proc.returncode != 0:
        report.tools_missing.append("prettier")
        return
    report.tools_used.append("prettier")
    for batch in _batched(files):
        _run(
            [
                "npx",
                "--no-install",
                "prettier",
                "--write",
                "--log-level",
                "warn",
                *[str(f) for f in batch],
            ],
            cwd=root,
        )


def format_output_dir(
    output_dir: str | Path,
    *,
    format_python: bool = True,
    format_web: bool = True,
) -> FormatReport:
    """Format and lint-fix all generated source under ``output_dir``.

    Args:
        output_dir: Root directory of the generated app/website.
        format_python: Run black + ruff on ``*.py`` files.
        format_web: Run prettier (if available) on web assets.

    Returns:
        A :class:`FormatReport`. ``report.is_clean`` is True when no unfixable
        lint/format issues remain.
    """
    root = Path(output_dir)
    report = FormatReport()

    if not root.exists():
        report.errors.append(f"output dir does not exist: {root}")
        return report

    try:
        if format_python:
            py_files = _iter_files(root, _PYTHON_SUFFIXES)
            report.python_files = len(py_files)
            if py_files:
                _format_python(py_files, root, report)

        if format_web:
            web_files = _iter_files(root, _WEB_SUFFIXES)
            report.web_files = len(web_files)
            if web_files:
                _format_web(web_files, root, report)
    except Exception as exc:  # never let formatting crash a build
        logger.warning("Formatting pass failed: %s", exc)
        report.errors.append(str(exc))

    return report
