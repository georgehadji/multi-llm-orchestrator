"""
Output Directory Writer
=======================
Author: Georgios-Chrysovalantis Chatzivantsidis
Writes structured output from a completed ProjectState to a directory.

Directory layout:
    <output_dir>/
    ├── task_001_code_generation.py    # code tasks  → .py
    ├── task_002_code_review.md        # prose tasks → .md
    ├── task_003_data_extraction.json  # data tasks  → .json (fallback .md)
    ...
    ├── app/                           # extracted multi-file code (if detected)
    │   ├── src/App.tsx
    │   ├── src/store/appStore.ts
    │   └── ...
    ├── summary.json                   # full machine-readable results
    └── README.md                      # human-readable summary

Called from cli.py after run_project() returns.

Improvement 7 — Code Extractor:
    When a task output contains multiple named code blocks (e.g. **src/App.tsx**
    followed by a fenced block), extract each file and write it to output_dir/app/.
    This turns "here is all the code" LLM prose into a runnable project structure.
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


# ── Code Extractor (Improvement 7) ───────────────────────────────────────────

# Matches: **path/to/file.ext** or **`path/to/file.ext`** (bold filename header)
# Also matches dotfiles like **.gitignore** (leading dot, no extension)
_FILENAME_HEADER = re.compile(
    r"\*\*`?(\.?[a-zA-Z0-9_@-][a-zA-Z0-9_./ @-]*\.[a-zA-Z0-9]{1,6}|"
    r"\.[a-zA-Z0-9_-]+)`?\*\*"
)

# Matches a fenced code block (``` ... ```) with optional language tag
_CODE_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)

# Extensions we consider "source code" worth extracting
_SOURCE_EXTS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".css", ".scss", ".html", ".json", ".toml",
    ".yaml", ".yml", ".env", ".md", ".sh", ".sql",
    ".gitignore", ".eslintrc", ".prettierrc",
}

# Min characters to bother writing (skip near-empty stubs)
_MIN_CONTENT_LEN = 20


def extract_named_files(text: str) -> dict[str, str]:
    """
    Extract named code files from LLM output.

    Looks for the pattern:
        **src/store/appStore.ts**
        ```typescript
        <code content>
        ```

    Returns a dict of {relative_path: code_content}.
    Paths are normalised (no leading slash, no ``..``).
    """
    files: dict[str, str] = {}
    # Split on filename headers; keep delimiters so we can pair them with blocks
    parts = _FILENAME_HEADER.split(text)
    # parts alternates: [prose, filename, prose_after, filename2, prose_after2, ...]
    # Index 0: text before first header; odd indices: filenames; even indices > 0: text after
    i = 1
    while i < len(parts):
        candidate_path = parts[i].strip()
        following_text = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        # Only extract recognised source extensions
        suffix = Path(candidate_path).suffix.lower()
        if suffix not in _SOURCE_EXTS and not candidate_path.startswith("."):
            continue

        # Sanitise path: strip leading slashes, reject traversal
        clean_path = candidate_path.lstrip("/\\").replace("\\", "/")
        if ".." in clean_path.split("/"):
            continue

        # Find the first code fence in the text immediately after the header
        # (look only in the first ~3000 chars to avoid grabbing a distant block)
        window = following_text[:3000]
        fence_match = _CODE_FENCE.search(window)
        if not fence_match:
            continue

        content = fence_match.group(1)
        if len(content.strip()) < _MIN_CONTENT_LEN:
            continue

        # Last-write-wins if the same path appears twice
        files[clean_path] = content

    return files


def write_extracted_files(
    files: dict[str, str],
    output_dir: Path,
) -> list[str]:
    """
    Write extracted {path: content} files under output_dir.
    Creates parent directories as needed.
    Returns list of written relative paths.
    """
    written: list[str] = []
    for rel_path, content in files.items():
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(rel_path)
        logger.debug("Extracted: %s (%d chars)", rel_path, len(content))
    return written

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

    Improvement 7 — Code Extractor:
        For CODE_GEN tasks that contain multiple named file blocks (e.g. **src/App.tsx**),
        each file is extracted and written to output_dir/app/<path>.
        A single task can produce an entire project tree this way.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, str] = {}   # task_id -> raw task filename
    extracted_total: dict[str, str] = {}  # rel_path -> content (across all tasks)

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

        # Improvement 7: extract named files for CODE_GEN tasks
        if task.type == TaskType.CODE_GEN:
            named = extract_named_files(result.output)
            if named:
                extracted_total.update(named)
                logger.info(
                    "  → Extracted %d named files from %s: %s",
                    len(named), task_id,
                    ", ".join(list(named.keys())[:5]) + ("…" if len(named) > 5 else ""),
                )

    # Write all extracted files under output_dir/app/
    if extracted_total:
        app_dir = out / "app"
        written = write_extracted_files(extracted_total, app_dir)
        logger.info("Code Extractor: wrote %d files to %s/app/", len(written), out.name)
        _write_app_readme(app_dir, written, state, project_id)

    _write_summary_json(state, out, file_map, project_id)
    _write_readme(state, out, file_map, project_id, extracted_count=len(extracted_total))

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


def _write_app_readme(
    app_dir: Path,
    written: list[str],
    state: ProjectState,
    project_id: str,
) -> None:
    """Write app/README.md listing all extracted source files."""
    lines = [
        "# Extracted App Files",
        "",
        f"Auto-extracted from project: `{project_id}`  ",
        f"Source project: {state.project_description[:80]}  ",
        "",
        "## Files",
        "",
    ]
    for rel_path in sorted(written):
        lines.append(f"- `{rel_path}`")
    lines += [
        "",
        "> Generated by Code Extractor (Improvement 7).",
        "> Each file was parsed from named code blocks in LLM task outputs.",
        "",
    ]
    dest = app_dir / "README.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Wrote app/README.md (%d files listed)", len(written))


def _write_readme(
    state: ProjectState,
    out: Path,
    file_map: dict[str, str],
    project_id: str,
    extracted_count: int = 0,
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

    if extracted_count > 0:
        lines += [
            "## Extracted App Files",
            "",
            f"The Code Extractor found **{extracted_count}** named source file(s) in the "
            "task outputs and wrote them to `app/`.  ",
            "See `app/README.md` for the full file list.",
            "",
        ]

    dest = out / "README.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote README.md")
