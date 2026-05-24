"""
DocsGenerator — Auto-generates documentation from knowledge entries
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Pillar 3: Generates ARCHITECTURE.md, DECISIONS.md from KnowledgeBase.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..knowledge.knowledge_base import KnowledgeBase

logger = logging.getLogger("orchestrator.knowledge.docs_generator")


class DocsGenerator:
    """Generate markdown documentation from a KnowledgeBase."""

    def generate_architecture_doc(self, kb: KnowledgeBase) -> str:
        """Generate ARCHITECTURE.md from architecture decisions."""
        lines = ["# Architecture Decision Record", ""]
        arch_entries = [e for e in kb.entries if e.category == "architecture"]
        for entry in arch_entries:
            lines.append(f"## {entry.id}")
            lines.append(f"**Date:** {entry.date}")
            lines.append(f"**Tags:** {', '.join(entry.tags)}")
            lines.append("")
            lines.append(entry.content)
            lines.append("")
        return "\n".join(lines)

    def generate_decisions_log(self, kb: KnowledgeBase) -> str:
        """Generate DECISIONS.md — chronological decision log."""
        lines = ["# Decision Log", ""]
        sorted_by_date = sorted(kb.entries, key=lambda e: e.date)
        for entry in sorted_by_date:
            lines.append(f"- **{entry.date}**: [{entry.category}] {entry.content[:200]}")
        return "\n".join(lines)

    def export(self, kb: KnowledgeBase, output_dir: Path) -> list[Path]:
        """Write all docs to output_dir. Returns list of created files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        created = []
        arch = output_dir / "ARCHITECTURE.md"
        arch.write_text(self.generate_architecture_doc(kb), encoding="utf-8")
        created.append(arch)
        dec = output_dir / "DECISIONS.md"
        dec.write_text(self.generate_decisions_log(kb), encoding="utf-8")
        created.append(dec)
        logger.info("DocsGenerator: wrote %d files to %s", len(created), output_dir)
        return created
