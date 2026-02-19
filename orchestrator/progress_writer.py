"""
ProgressWriter — incremental per-task output writer (Improvement 13)
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

After each task completes, immediately:
  1. Write the task output file to output_dir (same naming as write_output_dir)
  2. Append one JSON line to PROGRESS.jsonl
  3. Rewrite summary.json with the current partial state

This lets users see results as they stream in, and ensures partial output
is available even if the run is interrupted mid-way.

PROGRESS.jsonl format (one JSON object per line):
  {"task_id": "...", "status": "completed", "score": 0.9,
   "cost_usd": 0.003, "model_used": "gpt-4o", "timestamp_iso": "...",
   "output_file": "task_001_code_generation.py"}
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ProjectState, Task, TaskResult

logger = logging.getLogger("orchestrator.progress_writer")

# Reuse output_writer helpers so naming stays consistent
from .output_writer import _ext_for, _render_content, _write_summary_json


@dataclass
class ProgressEntry:
    task_id: str
    status: str
    score: float
    cost_usd: float
    model_used: str
    timestamp_iso: str
    output_file: str


class ProgressWriter:
    """
    Writes per-task output files and PROGRESS.jsonl incrementally.

    Safe for concurrent callers — uses asyncio.Lock for the shared files
    (PROGRESS.jsonl and summary.json). Individual task files are unique
    per task so they need no locking.

    Usage (engine calls this after every task):
        pw = ProgressWriter(output_dir, state)
        await pw.task_completed(task_id, result, task)
    """

    def __init__(self, output_dir: Path, state: "ProjectState") -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._state = state          # shared reference; .results grows as tasks finish
        self._lock = asyncio.Lock()  # protects PROGRESS.jsonl and summary.json
        self._file_map: dict[str, str] = {}   # task_id -> written filename

    async def task_completed(
        self,
        task_id: str,
        result: "TaskResult",
        task: "Task",
    ) -> None:
        """
        Called from _run_one immediately after self.results[task_id] is set.
        Writes output file, appends progress line, updates summary.json.
        """
        if not result.output:
            return  # skip empty/failed tasks with no content

        # Write the task output file (no lock needed — unique filename per task)
        filename = self._write_task_file(task_id, result, task)

        # Build the progress entry
        entry = ProgressEntry(
            task_id=task_id,
            status=result.status.value,
            score=round(result.score, 4),
            cost_usd=round(result.cost_usd, 6),
            model_used=result.model_used.value,
            timestamp_iso=datetime.now().isoformat(timespec="seconds"),
            output_file=filename,
        )

        # Shared file writes under lock
        async with self._lock:
            self._file_map[task_id] = filename
            await self._append_progress_line(entry)
            await self._update_summary()

    def _write_task_file(
        self, task_id: str, result: "TaskResult", task: "Task"
    ) -> str:
        """Write task output file. Returns the filename (relative to output_dir)."""
        from .models import TaskType
        ext = _ext_for(task.type, result.output)
        filename = f"{task_id}_{task.type.value}{ext}"
        dest = self._out / filename
        content = _render_content(task.type, result.output, ext)
        dest.write_text(content, encoding="utf-8")
        logger.debug(
            "ProgressWriter: wrote %s (%d chars, score=%.3f)",
            filename, len(content), result.score,
        )

        # Extract named files for CODE_GEN tasks (same as write_output_dir does)
        if task.type == TaskType.CODE_GEN:
            from .output_writer import extract_named_files, write_extracted_files
            named = extract_named_files(result.output)
            if named:
                app_dir = self._out / "app"
                write_extracted_files(named, app_dir)
                logger.debug(
                    "ProgressWriter: extracted %d files from %s", len(named), task_id
                )

        return filename

    async def _append_progress_line(self, entry: ProgressEntry) -> None:
        """Append one JSON line to PROGRESS.jsonl (called under self._lock)."""
        progress_path = self._out / "PROGRESS.jsonl"
        line = json.dumps({
            "task_id": entry.task_id,
            "status": entry.status,
            "score": entry.score,
            "cost_usd": entry.cost_usd,
            "model_used": entry.model_used,
            "timestamp_iso": entry.timestamp_iso,
            "output_file": entry.output_file,
        }, ensure_ascii=False) + "\n"
        # Appending a single short JSON line is effectively atomic on all major
        # OS file systems for writes < 4KB (POSIX O_APPEND guarantee).
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line)

    async def _update_summary(self) -> None:
        """Rewrite summary.json with the current partial results (called under self._lock)."""
        _write_summary_json(
            state=self._state,
            out=self._out,
            file_map=self._file_map,
            project_id=getattr(self._state, "_project_id", ""),
        )
