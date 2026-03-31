"""
Export Manager - Multi-format export for orchestrator outputs
===============================================================
Supports exporting project outputs in multiple formats:
- Markdown (.md) - Human-readable documentation
- JSON (.json) - Machine-readable structured data
- YAML (.yaml) - Configuration-friendly format

Features:
- Export individual tasks or full projects
- Template-based markdown generation
- Schema-compliant JSON output
- Git-friendly YAML with comments
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


if TYPE_CHECKING:
    from .models import Task

logger = logging.getLogger("orchestrator.export")


class ExportFormat(Enum):
    """Supported export formats."""

    MARKDOWN = "md"
    JSON = "json"
    YAML = "yaml"


class ExportManager:
    """Manages multi-format exports for orchestrator outputs."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # Markdown Export
    # ─────────────────────────────────────────

    def export_markdown(
        self,
        project_name: str,
        tasks: list[Task],
        results: dict[str, Any],
        total_cost: float,
        elapsed_time: float,
    ) -> Path:
        """Export project as comprehensive Markdown document."""

        lines = [
            f"# {project_name}\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## 📊 Project Summary\n\n",
            "| Metric | Value |\n",
            "|--------|-------|\n",
            f"| **Total Cost** | ${total_cost:.4f} |\n",
            f"| **Elapsed Time** | {elapsed_time:.1f}s |\n",
            f"| **Tasks** | {len(tasks)} |\n",
            f"| **Completed** | {sum(1 for r in results.values() if r.get('status') == 'completed')} |\n\n",
            "## 📋 Task Breakdown\n\n",
        ]

        for i, task in enumerate(tasks, 1):
            result = results.get(task.id, {})
            status = result.get("status", "pending")
            score = result.get("score", 0.0)
            model = result.get("model_used", "N/A")

            # Status emoji
            status_emoji = {
                "completed": "✅",
                "degraded": "⚠️",
                "failed": "❌",
                "pending": "⏳",
            }.get(status, "❓")

            lines.append(f"### {i}. {task.id} {status_emoji}\n\n")
            lines.append(f"**Type:** `{task.type.value if task.type else 'unknown'}`\n\n")
            lines.append(f"**Status:** {status}\n\n")

            if score > 0:
                lines.append(f"**Score:** {score:.2f}/1.0\n\n")
            if model != "N/A":
                lines.append(f"**Model:** `{model}`\n\n")

            lines.append(f"**Description:**\n{task.prompt[:300]}...\n\n")

            if task.dependencies:
                lines.append(f"**Dependencies:** {', '.join(task.dependencies)}\n\n")

            # Add output preview if available
            output = result.get("output", "")
            if output:
                lines.append("**Output Preview:**\n")
                lines.append("```\n")
                lines.append(output[:500])
                if len(output) > 500:
                    lines.append(f"\n... ({len(output) - 500} more characters)")
                lines.append("\n```\n\n")

            lines.append("---\n\n")

        # Add cost breakdown
        lines.append("## 💰 Cost Breakdown\n\n")
        lines.append("| Task | Model | Cost | Tokens |\n")
        lines.append("|------|-------|------|--------|\n")

        for task in tasks:
            result = results.get(task.id, {})
            model = result.get("model_used", "N/A")
            cost = result.get("cost_usd", 0.0)
            tokens = result.get("tokens_used", {})
            total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
            lines.append(f"| {task.id} | {model} | ${cost:.4f} | {total_tokens:,} |\n")

        lines.append(f"\n**Total:** ${total_cost:.4f}\n\n")

        # Add model usage statistics
        lines.append("## 🤖 Model Usage\n\n")
        model_usage: dict[str, dict[str, Any]] = {}
        for result in results.values():
            model = result.get("model_used", "unknown")
            if model not in model_usage:
                model_usage[model] = {"count": 0, "cost": 0.0, "tokens": 0}
            model_usage[model]["count"] += 1
            model_usage[model]["cost"] += result.get("cost_usd", 0.0)
            tokens = result.get("tokens_used", {})
            model_usage[model]["tokens"] += tokens.get("input", 0) + tokens.get("output", 0)

        lines.append("| Model | Tasks | Cost | Tokens |\n")
        lines.append("|-------|-------|------|--------|\n")
        for model, stats in sorted(model_usage.items(), key=lambda x: x[1]["cost"], reverse=True):
            lines.append(
                f"| {model} | {stats['count']} | ${stats['cost']:.4f} | {stats['tokens']:,} |\n"
            )

        # Write file
        output_path = (
            self.exports_dir
            / f"{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
        )
        output_path.write_text("".join(lines), encoding="utf-8")

        logger.info(f"Exported Markdown to {output_path}")
        return output_path

    # ─────────────────────────────────────────
    # JSON Export
    # ─────────────────────────────────────────

    def export_json(
        self,
        project_name: str,
        project_desc: str,
        success_criteria: str,
        tasks: list[Task],
        results: dict[str, Any],
        total_cost: float,
        elapsed_time: float,
        metadata: dict | None = None,
    ) -> Path:
        """Export project as structured JSON."""

        export_data = {
            "schema_version": "1.0",
            "export_format": "json",
            "generated_at": datetime.now().isoformat(),
            "project": {
                "name": project_name,
                "description": project_desc,
                "success_criteria": success_criteria,
            },
            "summary": {
                "total_tasks": len(tasks),
                "completed_tasks": sum(
                    1 for r in results.values() if r.get("status") == "completed"
                ),
                "failed_tasks": sum(1 for r in results.values() if r.get("status") == "failed"),
                "total_cost_usd": round(total_cost, 6),
                "elapsed_time_seconds": round(elapsed_time, 2),
            },
            "tasks": [],
            "model_usage": {},
            "metadata": metadata or {},
        }

        # Add task details
        for task in tasks:
            result = results.get(task.id, {})
            task_data = {
                "id": task.id,
                "type": task.type.value if task.type else None,
                "prompt": task.prompt,
                "dependencies": task.dependencies,
                "acceptance_threshold": task.acceptance_threshold,
                "max_iterations": task.max_iterations,
                "result": (
                    {
                        "status": result.get("status", "pending"),
                        "score": result.get("score"),
                        "model_used": result.get("model_used"),
                        "cost_usd": result.get("cost_usd"),
                        "tokens_used": result.get("tokens_used"),
                        "iterations": result.get("iterations"),
                        "output": result.get("output"),
                    }
                    if result
                    else None
                ),
            }
            export_data["tasks"].append(task_data)

        # Add model usage stats
        model_stats: dict[str, dict[str, Any]] = {}
        for result in results.values():
            model = result.get("model_used", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "tasks": 0,
                    "total_cost_usd": 0.0,
                    "total_tokens": {"input": 0, "output": 0},
                }
            model_stats[model]["tasks"] += 1
            model_stats[model]["total_cost_usd"] += result.get("cost_usd", 0.0)
            tokens = result.get("tokens_used", {})
            model_stats[model]["total_tokens"]["input"] += tokens.get("input", 0)
            model_stats[model]["total_tokens"]["output"] += tokens.get("output", 0)

        export_data["model_usage"] = model_stats

        # Write file
        output_path = (
            self.exports_dir
            / f"{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        output_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")

        logger.info(f"Exported JSON to {output_path}")
        return output_path

    # ─────────────────────────────────────────
    # YAML Export
    # ─────────────────────────────────────────

    def export_yaml(
        self,
        project_name: str,
        project_desc: str,
        success_criteria: str,
        tasks: list[Task],
        results: dict[str, Any],
        total_cost: float,
        elapsed_time: float,
    ) -> Path:
        """Export project as YAML (git-friendly format)."""

        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed. Install with: pip install pyyaml")
            # Fallback to JSON
            return self.export_json(
                project_name,
                project_desc,
                success_criteria,
                tasks,
                results,
                total_cost,
                elapsed_time,
            )

        export_data = {
            "# Multi-LLM Orchestrator Export": None,
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "project": {
                "name": project_name,
                "description": project_desc,
                "success_criteria": success_criteria,
            },
            "summary": {
                "total_tasks": len(tasks),
                "completed": sum(1 for r in results.values() if r.get("status") == "completed"),
                "failed": sum(1 for r in results.values() if r.get("status") == "failed"),
                "cost_usd": round(total_cost, 6),
                "time_seconds": round(elapsed_time, 2),
            },
            "tasks": [],
        }

        # Add tasks in YAML-friendly format
        for task in tasks:
            result = results.get(task.id, {})
            task_yaml = {
                "id": task.id,
                "type": task.type.value if task.type else "unknown",
                "prompt": task.prompt[:200] + "..." if len(task.prompt) > 200 else task.prompt,
                "dependencies": task.dependencies or [],
                "result": (
                    {
                        "status": result.get("status", "pending"),
                        "score": result.get("score"),
                        "model": result.get("model_used"),
                        "cost": result.get("cost_usd"),
                    }
                    if result
                    else None
                ),
            }
            export_data["tasks"].append(task_yaml)

        # Custom YAML representer for better formatting
        def str_representer(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer)

        # Write file
        output_path = (
            self.exports_dir
            / f"{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.yaml"
        )

        yaml_content = yaml.dump(
            export_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

        # Add header comment
        header = f"""# {project_name}
# Generated by Multi-LLM Orchestrator
# Total Cost: ${total_cost:.4f} | Tasks: {len(tasks)} | Time: {elapsed_time:.1f}s
#
# This file is git-friendly and human-readable.
# Edit with care - changes may affect reproducibility.

"""
        output_path.write_text(header + yaml_content, encoding="utf-8")

        logger.info(f"Exported YAML to {output_path}")
        return output_path

    # ─────────────────────────────────────────
    # Batch Export
    # ─────────────────────────────────────────

    def export_all(
        self,
        project_name: str,
        project_desc: str,
        success_criteria: str,
        tasks: list[Task],
        results: dict[str, Any],
        total_cost: float,
        elapsed_time: float,
    ) -> dict[str, Path]:
        """Export in all available formats."""

        exported = {}

        try:
            exported["markdown"] = self.export_markdown(
                project_name, tasks, results, total_cost, elapsed_time
            )
        except Exception as e:
            logger.error(f"Markdown export failed: {e}")

        try:
            exported["json"] = self.export_json(
                project_name,
                project_desc,
                success_criteria,
                tasks,
                results,
                total_cost,
                elapsed_time,
            )
        except Exception as e:
            logger.error(f"JSON export failed: {e}")

        try:
            exported["yaml"] = self.export_yaml(
                project_name,
                project_desc,
                success_criteria,
                tasks,
                results,
                total_cost,
                elapsed_time,
            )
        except Exception as e:
            logger.error(f"YAML export failed: {e}")

        return exported

    # ─────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────

    def list_exports(self) -> list[Path]:
        """List all exported files."""
        if not self.exports_dir.exists():
            return []
        return sorted(self.exports_dir.iterdir())

    def get_latest_export(self, format: ExportFormat) -> Path | None:
        """Get the most recent export of a specific format."""
        files = list(self.exports_dir.glob(f"*.{format.value}"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
