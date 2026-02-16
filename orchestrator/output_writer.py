"""
Output Directory Writer
=======================
Writes structured output from a completed ProjectState to a directory.

Directory layout:
    <output_dir>/
    ├── task_001_code_generation.py    # code tasks  → .py
    ├── task_002_code_review.md        # prose tasks → .md
    ├── task_003_data_extraction.json  # data tasks  → .json (fallback .md)
    ...
    ├── summary.json                   # full machine-readable results
    └── README.md                      # human-readable summary

Called from cli.py after run_project() returns.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ProjectState, TaskType

logger = logging.getLogger("orchestrator.output_writer")

# ── Extension mapping per TaskType ───────────────────────────────────────────
_EXT: dict[TaskType, str] = {
    TaskType.CODE_GEN:     ".py",
    TaskType.CODE_REVIEW:  ".md",
    TaskType.REASONING:    ".md",
    TaskType.WRITING:      ".md",
    TaskType.DATA_EXTRACT: ".json",   # falls back to .md if output is not valid JSON
    TaskType.SUMMARIZE:    ".md",
    TaskType.EVALUATE:     ".md",
}


def write_output_dir(
    state: ProjectState,
    output_dir: str | Path,
    project_id: str = "",
) -> Path:
    """
    Write all task outputs + summary files to output_dir.
    Creates the directory (and parents) if it does not exist.
    Returns the resolved absolute Path of the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, str] = {}  # task_id -> filename written

    order = state.execution_order or list(state.results.keys())
    for task_id in order:
        result = state.results.get(task_id)
        task = state.tasks.get(task_id)
        if not result or not task:
            continue
        if not result.output:
            continue  # skip skipped/empty tasks

        ext = _ext_for(task.type, result.output)
        filename = f"{task_id}_{task.type.value}{ext}"
        dest = out / filename

        content = _render_content(task.type, result.output, ext)
        dest.write_text(content, encoding="utf-8")
        file_map[task_id] = filename
        logger.info(f"Wrote {filename} ({len(content)} chars, score={result.score:.3f})")

    _write_summary_json(state, out, file_map, project_id)
    _write_readme(state, out, file_map, project_id)

    resolved = out.resolve()
    logger.info(f"Output written to: {resolved}")
    return resolved


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ext_for(task_type: TaskType, output: str) -> str:
    """
    Determine the file extension for a task.
    DATA_EXTRACT falls back to .md if the output is not valid JSON.
    """
    ext = _EXT.get(task_type, ".md")
    if ext == ".json":
        try:
            json.loads(_strip_fences(output).strip())
        except (json.JSONDecodeError, ValueError):
            ext = ".md"
    return ext


def _render_content(task_type: TaskType, raw_output: str, ext: str) -> str:
    """
    Render the file content for a task output.

    .py   → strip markdown fences, return first Python code block only
    .json → strip fences, parse and pretty-print JSON
    .md   → return raw output as-is (LLM prose is already markdown-compatible)
    """
    if ext == ".py":
        return _extract_python(raw_output)
    if ext == ".json":
        try:
            text = _strip_fences(raw_output).strip()
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            return raw_output
    # .md and everything else: return as-is
    return raw_output


def _extract_python(text: str) -> str:
    """
    Extract the first Python code block from markdown-fenced LLM output.
    Mirrors the logic in validators._extract_code_block() without importing it
    (validators has subprocess/tempfile side-effects we don't want here).
    """
    # 1. Fenced with explicit python tag
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    # 2. Generic fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    # 3. Heuristic: first top-level (column-0) Python statement
    m = re.search(
        r"^(import |from \w|def |class |@\w|if __name__|async def )",
        text, re.MULTILINE
    )
    if m:
        return text[m.start():]
    # 4. Fallback: return as-is
    return text


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _write_summary_json(
    state: ProjectState,
    out: Path,
    file_map: dict[str, str],
    project_id: str,
) -> None:
    """Write summary.json with full task list, outputs, and aggregate totals."""
    tasks_list = []
    total_cost = 0.0
    scores: list[float] = []
    completed = failed = degraded = 0

    order = state.execution_order or list(state.results.keys())
    for task_id in order:
        result = state.results.get(task_id)
        task = state.tasks.get(task_id)
        if not result:
            continue

        total_cost += result.cost_usd
        if result.score > 0:
            scores.append(result.score)

        s = result.status.value
        if s == "completed":
            completed += 1
        elif s == "failed":
            failed += 1
        else:
            degraded += 1

        tasks_list.append({
            "task_id": task_id,
            "task_type": task.type.value if task else "unknown",
            "prompt": task.prompt if task else "",
            "status": s,
            "score": round(result.score, 4),
            "model_used": result.model_used.value,
            "reviewer_model": result.reviewer_model.value if result.reviewer_model else None,
            "iterations": result.iterations,
            "cost_usd": round(result.cost_usd, 6),
            "deterministic_check_passed": result.deterministic_check_passed,
            "degraded_fallback_count": result.degraded_fallback_count,
            "output_file": file_map.get(task_id, ""),
            "output": result.output,
        })

    summary = {
        "project_id": project_id,
        "project_description": state.project_description,
        "success_criteria": state.success_criteria,
        "status": state.status.value,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "budget": state.budget.to_dict(),
        "tasks": tasks_list,
        "totals": {
            "tasks_completed": completed,
            "tasks_failed": failed,
            "tasks_degraded": degraded,
            "total_cost_usd": round(total_cost, 6),
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        },
    }

    dest = out / "summary.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote summary.json")


def _write_readme(
    state: ProjectState,
    out: Path,
    file_map: dict[str, str],
    project_id: str,
) -> None:
    """Write a human-readable README.md summarizing the project run."""
    b = state.budget
    budget_pct = (b.spent_usd / b.max_usd * 100) if b.max_usd > 0 else 0.0

    lines = [
        f"# Project: {state.project_description[:80]}",
        "",
        f"**Project ID**: `{project_id}`  ",
        f"**Status**: `{state.status.value}`  ",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Budget used**: ${b.spent_usd:.4f} / ${b.max_usd} ({budget_pct:.1f}%)  ",
        f"**Time elapsed**: {b.elapsed_seconds:.1f}s  ",
        "",
        "## Success Criteria",
        "",
        state.success_criteria,
        "",
        "## Task Results",
        "",
        "| File | Task Type | Score | Model | Status |",
        "|------|-----------|-------|-------|--------|",
    ]

    order = state.execution_order or list(state.results.keys())
    for task_id in order:
        result = state.results.get(task_id)
        task = state.tasks.get(task_id)
        if not result:
            continue
        filename = file_map.get(task_id, task_id)
        task_type = task.type.value if task else "unknown"
        # Shorten model name for table readability
        model_parts = result.model_used.value.split("-")
        model_display = "-".join(model_parts[:2])
        status_icon = {"completed": "✅", "failed": "❌", "degraded": "⚠️"}.get(
            result.status.value, "~"
        )
        lines.append(
            f"| `{filename}` | {task_type} | {result.score:.3f} "
            f"| {model_display} | {status_icon} {result.status.value} |"
        )

    lines += [
        "",
        "## Files Generated",
        "",
    ]
    for task_id, filename in file_map.items():
        task = state.tasks.get(task_id)
        desc = (task.prompt[:70].replace("\n", " ") + "...") if task else ""
        lines.append(f"- `{filename}` — {desc}")
    lines += [
        "- `summary.json` — Full machine-readable results (includes raw outputs)",
        "- `README.md` — This file",
        "",
    ]

    dest = out / "README.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote README.md")
