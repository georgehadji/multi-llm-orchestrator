"""
Project Assembler
=================
Author: Georgios-Chrysovalantis Chatzivantsidis

Transforms flat task outputs into a real project directory tree by
placing each task's result at its declared ``target_path``.

Usage
-----
When a YAML project file includes a ``tasks:`` section with per-task
``target_path`` fields the assembler writes each output to the correct
location inside *output_dir*, creating subdirectories as needed.

Example YAML fragment::

    tasks:
      - id: task_001
        target_path: src/main.py
      - id: task_002
        target_path: tests/test_main.py
      - id: task_003
        target_path: README.md

If a task has no ``target_path`` the assembler falls back to the flat
``task_NNN_<type><ext>`` naming used by ``output_writer``.

Post-assembly verification
--------------------------
Pass ``verify_cmd`` (e.g. ``"pytest"`` or ``"npm run build"``) to run a
shell command inside the assembled directory.  The result is captured and
returned in ``AssemblyResult.verify_output``.

All filesystem operations use ``pathlib.Path`` and are safe to call from
sync or async code (blocking I/O, but small files only).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import ProjectState, TaskType
from .output_writer import _ext_for, _render_content, _strip_fences

logger = logging.getLogger("orchestrator.assembler")


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class AssemblyResult:
    """Summary of a completed assembly run."""
    output_dir: Path
    files_written: list[str] = field(default_factory=list)   # relative paths
    files_skipped: list[str] = field(default_factory=list)   # task_ids with no output
    verify_output: str = ""                                    # stdout+stderr of verify_cmd
    verify_returncode: Optional[int] = None                   # None = not run
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors and (
            self.verify_returncode is None or self.verify_returncode == 0
        )


# ── Main class ────────────────────────────────────────────────────────────────

class ProjectAssembler:
    """
    Assembles task outputs into a real project directory tree.

    Parameters
    ----------
    state:
        Completed ``ProjectState`` with populated ``results`` and ``tasks``.
    task_paths:
        Optional mapping of ``task_id -> target_path`` that overrides or
        supplements the ``Task.target_path`` field.  Useful when the YAML
        parser supplies per-task paths that the engine does not store in
        ``Task`` objects yet.
    """

    def __init__(
        self,
        state: ProjectState,
        task_paths: Optional[dict[str, str]] = None,
    ) -> None:
        self._state = state
        self._task_paths: dict[str, str] = task_paths or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def assemble(
        self,
        output_dir: str | Path,
        verify_cmd: str = "",
        overwrite: bool = True,
    ) -> AssemblyResult:
        """
        Write task outputs into *output_dir* using declared target paths.

        Parameters
        ----------
        output_dir:
            Root directory for the assembled project.  Created if absent.
        verify_cmd:
            Shell command to run inside *output_dir* after assembly
            (e.g. ``"pytest"`` or ``"npm run build"``).  Skipped if empty.
        overwrite:
            When ``False``, skip files that already exist (idempotent mode).

        Returns
        -------
        AssemblyResult
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        result = AssemblyResult(output_dir=out.resolve())

        order = self._state.execution_order or list(self._state.results.keys())
        for task_id in order:
            task_result = self._state.results.get(task_id)
            task = self._state.tasks.get(task_id)
            if not task_result or not task:
                result.files_skipped.append(task_id)
                continue
            if not task_result.output:
                result.files_skipped.append(task_id)
                continue

            # Resolve destination path
            target = (
                self._task_paths.get(task_id)
                or (task.target_path if hasattr(task, "target_path") else "")
            )
            if target:
                dest = out / target
            else:
                # Fallback: flat name identical to output_writer
                ext = _ext_for(task.type, task_result.output)
                dest = out / f"{task_id}_{task.type.value}{ext}"

            if dest.exists() and not overwrite:
                logger.debug("Skipping existing file: %s", dest)
                result.files_written.append(dest.relative_to(out).as_posix())
                continue

            # Create parent dirs
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Render content
            ext = dest.suffix or _ext_for(task.type, task_result.output)
            content = _render_content_for_ext(task.type, task_result.output, ext)

            try:
                dest.write_text(content, encoding="utf-8")
                # Always use forward slashes so paths are portable in test assertions
                rel = dest.relative_to(out).as_posix()
                result.files_written.append(rel)
                logger.info("Assembled %s -> %s (%d chars)", task_id, rel, len(content))
            except OSError as exc:
                msg = f"Failed to write {dest}: {exc}"
                logger.error(msg)
                result.errors.append(msg)

        # Write summary.json + README.md (same as output_writer for compatibility)
        self._write_manifests(out, result)

        # Optional post-assembly verification
        if verify_cmd:
            result.verify_output, result.verify_returncode = _run_verify(
                verify_cmd, cwd=out
            )
            if result.verify_returncode != 0:
                result.errors.append(
                    f"Verification failed (exit {result.verify_returncode}): {verify_cmd}"
                )

        logger.info(
            "Assembly complete: %d files written, %d skipped, %d errors",
            len(result.files_written),
            len(result.files_skipped),
            len(result.errors),
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_manifests(self, out: Path, result: AssemblyResult) -> None:
        """Write assembly-manifest.json listing all assembled files."""
        manifest = {
            "output_dir": str(result.output_dir),
            "files_written": result.files_written,
            "files_skipped": result.files_skipped,
            "task_count": len(self._state.results),
            "project_status": self._state.status.value,
        }
        dest = out / "assembly-manifest.json"
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote assembly-manifest.json")


# ── Content rendering ─────────────────────────────────────────────────────────

def _render_content_for_ext(task_type: TaskType, raw_output: str, ext: str) -> str:
    """
    Render content for an arbitrary file extension.

    Strategy:
    - ``.json``               → parse+pretty-print, fall back to raw
    - Recognised code exts    → strip any markdown code fence (language-agnostic),
                                return bare code.  For ``.py`` specifically we also
                                run the Python-aware extractor via output_writer.
    - Everything else         → return prose as-is (markdown, txt, rst, …)
    """
    _CODE_EXTS = {".py", ".ts", ".tsx", ".jsx", ".js", ".rs", ".go",
                  ".sh", ".bash", ".sql", ".proto", ".toml", ".yaml",
                  ".yml", ".xml", ".ini", ".dockerfile", ".rb", ".java",
                  ".cs", ".cpp", ".c", ".h", ".kt", ".swift"}

    ext_lower = ext.lower()

    if ext_lower == ".json":
        return _render_content(task_type, raw_output, ".json")

    if ext_lower == ".py":
        # Use the full Python-aware extractor
        return _render_content(task_type, raw_output, ".py")

    if ext_lower in _CODE_EXTS:
        # Generic fence stripping — handles ```typescript, ```rust, etc.
        stripped = _strip_fences(raw_output)
        return stripped if stripped else raw_output

    # Prose (md, txt, rst, …) — return as-is
    return raw_output


# ── Verification helper ───────────────────────────────────────────────────────

def _run_verify(cmd: str, cwd: Path) -> tuple[str, int]:
    """Run *cmd* in *cwd*, return (combined_output, returncode)."""
    logger.info("Running verification: %s", cmd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        logger.info("Verification exit code: %d", proc.returncode)
        return combined, proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning("Verification command timed out: %s", cmd)
        return "TIMEOUT", 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Verification command error: %s", exc)
        return str(exc), 1


# ── Convenience function ──────────────────────────────────────────────────────

def assemble_project(
    state: ProjectState,
    output_dir: str | Path,
    task_paths: Optional[dict[str, str]] = None,
    verify_cmd: str = "",
    overwrite: bool = True,
) -> AssemblyResult:
    """
    One-call convenience wrapper around ``ProjectAssembler``.

    Parameters
    ----------
    state:
        Completed project state.
    output_dir:
        Root of the assembled project.
    task_paths:
        Optional ``{task_id: target_path}`` override map.
    verify_cmd:
        Shell command to run for post-assembly verification.
    overwrite:
        Overwrite existing files (default ``True``).
    """
    assembler = ProjectAssembler(state, task_paths=task_paths)
    return assembler.assemble(output_dir, verify_cmd=verify_cmd, overwrite=overwrite)
