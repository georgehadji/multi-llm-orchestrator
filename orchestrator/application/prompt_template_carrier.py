"""
PromptTemplateCarrier — Versioned Prompt Deltas (Phase 4.3)
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Manages versioned prompt-template deltas that can be applied to the
base worker, critique, and decomposition prompts. Integrates with the
``MechanismRegistry`` for lifecycle management, backup/restore, and
A/B testing compatibility.

Each delta is a structured overlay that modifies specific sections of
the base prompt (system prefix, instructions, output format, examples).
Deltas are versioned and can be rolled back independently.

Usage:
    carrier = PromptTemplateCarrier()
    delta = PromptDelta(
        version="2.0",
        target="worker",
        system_prefix="You are an expert Python developer...",
    )
    carrier.apply(delta)
    prompt = carrier.render("worker", base_prompt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .mechanism_registry import MechanismRegistry

logger = logging.getLogger("orchestrator.bilevel.prompt_carrier")


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class PromptDelta:
    """A versioned prompt-template delta.

    Each delta targets a specific prompt type (``target``) and provides
    optional overrides for the system prefix, instructions, output format,
    and few-shot examples.

    Args:
        version: Semantic version string (e.g. "2.0").
        target: Which prompt type to modify ("worker", "critique", "decomposition").
        system_prefix: Override for the system-level prefix.
        instructions: Additional or replacement instructions.
        output_format: Override for the expected output format.
        examples: Few-shot examples to append.
        description: Human-readable description of what this delta changes.
    """

    version: str
    target: str
    system_prefix: str = ""
    instructions: str = ""
    output_format: str = ""
    examples: list[str] = field(default_factory=list)
    description: str = ""

    def apply_to(self, base_prompt: str) -> str:
        """Apply this delta to a base prompt string.

        Currently this replaces the base prompt when overrides are present.
        In future this could do structured section-level merging.

        Args:
            base_prompt: The original prompt text.

        Returns:
            The modified prompt with deltas applied.
        """
        result = base_prompt
        if self.system_prefix:
            result = f"{self.system_prefix}\n\n{result}"
        if self.instructions:
            result = f"{result}\n\n---\n{self.instructions}"
        if self.output_format:
            result = f"{result}\n\nOutput format:\n{self.output_format}"
        if self.examples:
            examples_text = "\n\nExamples:\n" + "\n---\n".join(self.examples)
            result = f"{result}{examples_text}"
        return result


# ── Carrier ───────────────────────────────────────────────────────────────


class PromptTemplateCarrier:
    """Manage versioned prompt-template deltas.

    Each prompt type (worker, critique, decomposition) can have an
    active delta applied on top of the base prompt. Deltas are stored
    in a versioned history and can be rolled back.

    Integrates with ``MechanismRegistry`` so prompt changes go through
    the same lifecycle (activate/deactivate/rollback) as other mechanisms.

    Args:
        registry: Optional ``MechanismRegistry`` for lifecycle tracking.
    """

    def __init__(
        self,
        registry: MechanismRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._active_deltas: dict[str, PromptDelta] = {}  # target -> active delta
        self._delta_history: dict[str, list[PromptDelta]] = {}  # target -> history

    def apply(self, delta: PromptDelta) -> None:
        """Apply a prompt delta, making it the active version for its target.

        Args:
            delta: The ``PromptDelta`` to apply.
        """
        target = delta.target
        # Store previous in history
        if target in self._active_deltas:
            self._delta_history.setdefault(target, []).append(self._active_deltas[target])

        # Set new active delta
        self._active_deltas[target] = delta

        # Register in mechanism registry if available
        if self._registry is not None:
            mech_name = f"prompt_delta_{target}_v{delta.version}"
            try:
                self._registry.register(
                    name=mech_name,
                    version=delta.version,
                    description=f"Prompt delta for {target}: {delta.description}",
                )
                self._registry.activate(mech_name)
            except ValueError:
                # Already registered — update instead
                pass

        logger.info(
            "Applied prompt delta v%s for '%s': %s",
            delta.version,
            target,
            delta.description or "no description",
        )

    def rollback(self, target: str) -> PromptDelta | None:
        """Roll back the active delta for a target, restoring the previous one.

        Args:
            target: The prompt type to roll back.

        Returns:
            The previously-active delta if one was restored, or ``None``.
        """
        if target not in self._delta_history or not self._delta_history[target]:
            # No history — clear the active delta
            old = self._active_deltas.pop(target, None)
            logger.info("Rolled back prompt delta for '%s' (no history)", target)
            return old

        previous = self._delta_history[target].pop()
        self._active_deltas[target] = previous
        logger.info("Restored previous prompt delta for '%s' (v%s)", target, previous.version)
        return previous

    def render(self, target: str, base_prompt: str) -> str:
        """Render the final prompt by applying any active delta to the base.

        Args:
            target: The prompt type ("worker", "critique", "decomposition").
            base_prompt: The original base prompt text.

        Returns:
            The prompt with any active delta applied, or the original if none.
        """
        delta = self._active_deltas.get(target)
        if delta is None:
            return base_prompt
        return delta.apply_to(base_prompt)

    def get_active(self, target: str) -> PromptDelta | None:
        """Get the active delta for a target, or ``None``."""
        return self._active_deltas.get(target)

    def get_history(self, target: str) -> list[PromptDelta]:
        """Get the version history for a target."""
        return list(self._delta_history.get(target, []))

    @property
    def active_targets(self) -> list[str]:
        """List of targets with active deltas."""
        return list(self._active_deltas.keys())

    @property
    def delta_count(self) -> int:
        """Total number of active deltas."""
        return len(self._active_deltas)
