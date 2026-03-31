"""
Progressive Output Manager - bmalph-style organized outputs
===========================================================
Organizes orchestrator outputs in numbered folders:

outputs/
├── 001-analysis/           # Product analysis & briefs
├── 002-architecture/       # Architecture rules & design
├── 003-decomposition/      # Task breakdown
├── 004-task-001/          # Individual task outputs
├── 005-task-002/
├── ...
└── final/                 # Final assembled output
    ├── code/
    ├── tests/
    └── docs/

Features:
- Automatic folder numbering
- Spec changelog tracking
- Incremental development support
- Resume capability
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .models import Task, TaskStatus, TaskType

logger = logging.getLogger("orchestrator.output")


@dataclass
class OutputEntry:
    """Single output entry in the progressive structure."""

    sequence: int
    phase: str
    task_id: str | None
    model: str
    timestamp: str
    files: list[str]
    cost_usd: float
    tokens_input: int
    tokens_output: int
    score: float = 0.0
    status: str = "pending"


@dataclass
class SpecChange:
    """Change record for spec changelog."""

    timestamp: str
    change_type: str  # 'added', 'modified', 'completed', 'removed'
    description: str
    task_id: str | None
    diff_hash: str


class ProgressiveOutputManager:
    """
    Manages progressive output structure similar to bmalph's _bmad-output/.
    """

    def __init__(self, base_dir: Path, project_name: str = "project"):
        self.base_dir = Path(base_dir)
        self.project_name = project_name
        self.output_dir = self.base_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self._sequence_counter = self._find_last_sequence()
        self._entries: list[OutputEntry] = []
        self._changelog: list[SpecChange] = []

        # Load existing state if resuming
        self._load_state()

    def _find_last_sequence(self) -> int:
        """Find the highest sequence number in existing outputs."""
        max_seq = 0
        for item in self.output_dir.iterdir():
            if item.is_dir():
                # Parse 001-name format
                parts = item.name.split("-", 1)
                if parts[0].isdigit():
                    max_seq = max(max_seq, int(parts[0]))
        return max_seq

    def _load_state(self):
        """Load previous output state for resume support."""
        state_file = self.output_dir / ".progressive_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self._entries = [OutputEntry(**e) for e in data.get("entries", [])]
                self._changelog = [SpecChange(**c) for c in data.get("changelog", [])]
                logger.info(f"Loaded {len(self._entries)} previous output entries")
            except Exception as e:
                logger.warning(f"Failed to load progressive state: {e}")

    def _save_state(self):
        """Save current output state."""
        state_file = self.output_dir / ".progressive_state.json"
        state = {
            "project_name": self.project_name,
            "last_sequence": self._sequence_counter,
            "entries": [asdict(e) for e in self._entries],
            "changelog": [asdict(c) for c in self._changelog],
            "updated_at": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence_counter += 1
        return self._sequence_counter

    def _create_folder(self, phase: str, task_id: str | None = None) -> Path:
        """Create numbered output folder."""
        seq = self._next_sequence()

        folder_name = f"{seq:03d}-{task_id}" if task_id else f"{seq:03d}-{phase}"

        folder_path = self.output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "sequence": seq,
            "phase": phase,
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
        }
        (folder_path / ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        return folder_path

    # ─────────────────────────────────────────
    # Phase Outputs
    # ─────────────────────────────────────────

    def save_analysis(self, content: str, analysis_type: str = "product-brief") -> Path:
        """Save analysis phase output."""
        folder = self._create_folder("analysis")

        # Save main content
        if analysis_type == "product-brief":
            file_path = folder / "product-brief.md"
        elif analysis_type == "market-research":
            file_path = folder / "market-research.md"
        elif analysis_type == "domain-research":
            file_path = folder / "domain-research.md"
        else:
            file_path = folder / f"{analysis_type}.md"

        file_path.write_text(content, encoding="utf-8")

        # Record entry
        entry = OutputEntry(
            sequence=self._sequence_counter,
            phase="analysis",
            task_id=None,
            model="N/A",
            timestamp=datetime.now().isoformat(),
            files=[str(file_path.relative_to(self.output_dir))],
            cost_usd=0.0,
            tokens_input=0,
            tokens_output=0,
        )
        self._entries.append(entry)
        self._save_state()

        # Add changelog entry
        self._add_changelog("added", f"Analysis: {analysis_type}", None, content)

        logger.info(f"Saved analysis to {file_path}")
        return file_path

    def save_architecture(self, content: str, arch_type: str = "architecture") -> Path:
        """Save architecture phase output."""
        folder = self._create_folder("architecture")

        file_path = folder / f"{arch_type}.md"
        file_path.write_text(content, encoding="utf-8")

        entry = OutputEntry(
            sequence=self._sequence_counter,
            phase="architecture",
            task_id=None,
            model="N/A",
            timestamp=datetime.now().isoformat(),
            files=[str(file_path.relative_to(self.output_dir))],
            cost_usd=0.0,
            tokens_input=0,
            tokens_output=0,
        )
        self._entries.append(entry)
        self._save_state()

        self._add_changelog("added", f"Architecture: {arch_type}", None, content)

        logger.info(f"Saved architecture to {file_path}")
        return file_path

    def save_decomposition(self, tasks: list[Task]) -> Path:
        """Save task decomposition."""
        folder = self._create_folder("decomposition")

        # Save as JSON
        tasks_data = []
        for task in tasks:
            tasks_data.append(
                {
                    "id": task.id,
                    "type": task.type.value if task.type else None,
                    "prompt": task.prompt,
                    "dependencies": task.dependencies,
                    "acceptance_threshold": task.acceptance_threshold,
                    "max_iterations": task.max_iterations,
                }
            )

        file_path = folder / "tasks.json"
        file_path.write_text(json.dumps(tasks_data, indent=2), encoding="utf-8")

        # Also save as markdown for readability
        md_path = folder / "tasks.md"
        md_lines = ["# Task Decomposition\n"]
        for task in tasks_data:
            md_lines.append(f"\n## {task['id']}\n")
            md_lines.append(f"- **Type:** {task['type']}\n")
            md_lines.append(f"- **Dependencies:** {', '.join(task['dependencies']) or 'None'}\n")
            md_lines.append(f"- **Threshold:** {task['acceptance_threshold']}\n")
            md_lines.append(f"\n**Prompt:**\n{task['prompt'][:200]}...\n")
        md_path.write_text("".join(md_lines), encoding="utf-8")

        entry = OutputEntry(
            sequence=self._sequence_counter,
            phase="decomposition",
            task_id=None,
            model="N/A",
            timestamp=datetime.now().isoformat(),
            files=[
                str(file_path.relative_to(self.output_dir)),
                str(md_path.relative_to(self.output_dir)),
            ],
            cost_usd=0.0,
            tokens_input=0,
            tokens_output=0,
        )
        self._entries.append(entry)
        self._save_state()

        self._add_changelog(
            "added", f"Decomposition: {len(tasks)} tasks", None, json.dumps(tasks_data)
        )

        logger.info(f"Saved decomposition ({len(tasks)} tasks) to {folder}")
        return folder

    def save_task_output(
        self,
        task: Task,
        output: str,
        model: str,
        cost_usd: float,
        tokens_input: int,
        tokens_output: int,
        score: float,
        status: TaskStatus,
    ) -> Path:
        """Save individual task execution output."""
        folder = self._create_folder("task", task.id)

        # Save output
        ext = self._get_extension_for_task(task.type)
        file_path = folder / f"output{ext}"
        file_path.write_text(output, encoding="utf-8")

        # Save metadata
        meta = {
            "task_id": task.id,
            "task_type": task.type.value if task.type else None,
            "model": model,
            "cost_usd": cost_usd,
            "tokens": {"input": tokens_input, "output": tokens_output},
            "score": score,
            "status": status.value,
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = folder / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        entry = OutputEntry(
            sequence=self._sequence_counter,
            phase="execution",
            task_id=task.id,
            model=model,
            timestamp=datetime.now().isoformat(),
            files=[str(file_path.relative_to(self.output_dir))],
            cost_usd=cost_usd,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            score=score,
            status=status.value,
        )
        self._entries.append(entry)
        self._save_state()

        change_type = "completed" if status == TaskStatus.COMPLETED else "modified"
        self._add_changelog(change_type, f"Task {task.id}: {status.value}", task.id, output)

        logger.info(f"Saved task {task.id} output to {folder}")
        return folder

    def _get_extension_for_task(self, task_type: TaskType | None) -> str:
        """Get file extension based on task type."""
        if task_type == TaskType.CODE_GEN:
            return ".py"
        elif task_type == TaskType.CODE_REVIEW:
            return ".md"
        elif task_type == TaskType.DATA_EXTRACT:
            return ".json"
        elif task_type == TaskType.SUMMARIZE:
            return ".md"
        else:
            return ".txt"

    # ─────────────────────────────────────────
    # Final Assembly
    # ─────────────────────────────────────────

    def create_final_output(self, code_files: dict[str, str], docs: dict[str, str]) -> Path:
        """Create final assembled output."""
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Code directory
        code_dir = final_dir / "code"
        code_dir.mkdir(exist_ok=True)
        for name, content in code_files.items():
            (code_dir / name).write_text(content, encoding="utf-8")

        # Docs directory
        docs_dir = final_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        for name, content in docs.items():
            (docs_dir / name).write_text(content, encoding="utf-8")

        # Summary
        summary = self._generate_summary()
        summary_path = final_dir / "SUMMARY.md"
        summary_path.write_text(summary, encoding="utf-8")

        logger.info(f"Created final output in {final_dir}")
        return final_dir

    def _generate_summary(self) -> str:
        """Generate project summary."""
        total_cost = sum(e.cost_usd for e in self._entries)
        total_input = sum(e.tokens_input for e in self._entries)
        total_output = sum(e.tokens_output for e in self._entries)
        completed = sum(1 for e in self._entries if e.status == "completed")

        lines = [
            f"# Project Summary: {self.project_name}\n",
            f"**Generated:** {datetime.now().isoformat()}\n\n",
            "## Statistics\n",
            f"- **Total Cost:** ${total_cost:.4f}\n",
            f"- **Total Tokens:** {total_input + total_output:,} ({total_input:,} in, {total_output:,} out)\n",
            f"- **Completed Tasks:** {completed}\n",
            f"- **Output Entries:** {len(self._entries)}\n\n",
            "## Timeline\n",
        ]

        for entry in sorted(self._entries, key=lambda e: e.sequence):
            lines.append(f"{entry.sequence:03d}. **{entry.phase}**")
            if entry.task_id:
                lines.append(f" ({entry.task_id})")
            lines.append(f" - {entry.model} - ${entry.cost_usd:.4f}")
            if entry.score > 0:
                lines.append(f" - Score: {entry.score:.2f}")
            lines.append("\n")

        return "".join(lines)

    # ─────────────────────────────────────────
    # Spec Changelog
    # ─────────────────────────────────────────

    def _add_changelog(
        self,
        change_type: str,
        description: str,
        task_id: str | None,
        content: str,
    ):
        """Add entry to spec changelog."""
        change = SpecChange(
            timestamp=datetime.now().isoformat(),
            change_type=change_type,
            description=description,
            task_id=task_id,
            diff_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        )
        self._changelog.append(change)
        self._write_changelog()

    def _write_changelog(self):
        """Write changelog to file."""
        changelog_path = self.output_dir / "SPECS_CHANGELOG.md"

        lines = ["# Spec Changelog\n\n"]
        lines.append("Track changes to specifications and outputs over time.\n\n")

        for change in reversed(self._changelog[-50:]):  # Last 50 entries
            emoji = {
                "added": "📝",
                "modified": "✏️",
                "completed": "✅",
                "removed": "🗑️",
            }.get(change.change_type, "•")

            lines.append(f"{emoji} **{change.timestamp[:10]}** - {change.description}")
            if change.task_id:
                lines.append(f" (`{change.task_id}`)")
            lines.append(f" [{change.diff_hash}]\n")

        changelog_path.write_text("".join(lines), encoding="utf-8")

    def get_changelog(self) -> list[SpecChange]:
        """Get spec changelog entries."""
        return self._changelog.copy()

    # ─────────────────────────────────────────
    # Resume Support
    # ─────────────────────────────────────────

    def get_completed_task_ids(self) -> list[str]:
        """Get list of completed task IDs for resume."""
        return [
            e.task_id
            for e in self._entries
            if e.phase == "execution" and e.status == "completed" and e.task_id
        ]

    def get_task_output_path(self, task_id: str) -> Path | None:
        """Get output path for a specific task."""
        for entry in self._entries:
            if entry.task_id == task_id and entry.files:
                return self.output_dir / entry.files[0]
        return None
