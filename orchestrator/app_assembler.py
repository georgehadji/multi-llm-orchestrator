"""
AppAssembler — writes TaskResult outputs to target files inside output_dir.

Steps:
1. For each TaskResult, write result.output to task.target_path inside output_dir
2. Skip tasks with empty output or empty target_path (record in files_skipped)
3. Run ImportFixer: ensure __init__.py exists for every Python package directory
4. Return AssemblyReport(files_written, files_skipped, import_issues)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AssemblyReport:
    """Report of what the AppAssembler wrote and what it skipped."""

    files_written: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    import_issues: list[str] = field(default_factory=list)


class AppAssembler:
    """
    Writes TaskResult outputs to the correct file paths inside output_dir.

    Usage:
        assembler = AppAssembler()
        report = assembler.assemble(results, tasks, scaffold, output_dir)
    """

    def assemble(
        self,
        results: dict,      # task_id -> TaskResult
        tasks: dict,        # task_id -> Task
        scaffold: dict,     # rel_path -> content (from ScaffoldEngine)
        output_dir: Path,
    ) -> AssemblyReport:
        """
        Write each TaskResult.output to its task.target_path.

        Parameters
        ----------
        results:    dict[task_id, TaskResult]
        tasks:      dict[task_id, Task]
        scaffold:   dict[rel_path, content] — scaffold files already written to disk by ScaffoldEngine;
                    any that were not overwritten by a task are recorded in files_written
        output_dir: root directory for all written files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = AssemblyReport()

        for task_id, result in results.items():
            task = tasks.get(task_id)

            # Skip if task not found or has no target path
            if task is None or not task.target_path:
                report.files_skipped.append(f"{task_id} (no target_path)")
                continue

            # Skip if output is empty or whitespace-only
            if not result.output or not result.output.strip():
                report.files_skipped.append(f"{task_id} (empty output)")
                logger.warning("Task %s produced empty output; skipping %s", task_id, task.target_path)
                continue

            dest = output_dir / task.target_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result.output, encoding="utf-8")
            report.files_written.append(task.target_path)
            logger.debug("Assembled: %s", task.target_path)

        # Record scaffold files that exist on disk (written earlier by ScaffoldEngine)
        # and were not overwritten by any task
        task_paths = set(report.files_written)
        for rel_path in scaffold:
            if rel_path not in task_paths and (output_dir / rel_path).exists():
                report.files_written.append(rel_path)
                logger.debug("Scaffold file kept: %s", rel_path)

        # Run ImportFixer — ensure __init__.py exists for every Python package dir
        self._ensure_init_files(output_dir, report)

        return report

    def _ensure_init_files(self, output_dir: Path, report: AssemblyReport) -> None:
        """
        For every .py file written, ensure all parent directories that are
        Python packages have an __init__.py file.

        Modifies report.import_issues if any directory cannot be processed.
        """
        for rel_path_str in list(report.files_written):
            py_file = output_dir / rel_path_str
            if not py_file.suffix == ".py":
                continue

            # Walk up the directory tree from the file's parent to output_dir
            current = py_file.parent
            while current != output_dir and current != current.parent:
                init_file = current / "__init__.py"
                if not init_file.exists():
                    try:
                        init_file.write_text("", encoding="utf-8")
                        logger.debug("Created __init__.py: %s", init_file.relative_to(output_dir))
                    except OSError as exc:
                        issue = f"Could not create {init_file}: {exc}"
                        report.import_issues.append(issue)
                        logger.warning(issue)
                current = current.parent
