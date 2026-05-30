"""
SkillOptimizer — Per-TaskType text-space optimization loop
===========================================================
Implements the SkillOpt algorithm:

  1. Accumulate N trajectories (task + output + score + critique)
  2. Split into train/val
  3. Ask the optimizer LLM to propose minimal patches to the skill doc
  4. Enforce an edit budget (token_cost sum ≤ edit_budget)
  5. Apply patches and estimate val-set score improvement
  6. Accept if strictly better; otherwise store in negative-feedback buffer
  7. Every K accepted epochs, rebuild the protected ## Guidance block

Key design decisions
--------------------
- Validation scoring is a cheap proxy (weighted mean of val trajectory scores +
  content-quality heuristic) — no extra LLM calls on the hot path.
- The ``## Guidance`` block in the skill doc is protected by a regex guard and
  rebuilt only on slow-update epochs (every slow_update_every=5 epochs).
- All failures are caught and logged; a crashed epoch must never abort task execution.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import TaskType
from ..models_skill import SkillPatch, SkillUpdateResult, Trajectory

if TYPE_CHECKING:
    from ..domain.ports import LLMClient, SkillStorePort

logger = logging.getLogger("orchestrator.skill_optimizer")

# Path to the optimizer meta-skill (shipped static, never updated by training)
_META_SKILL_PATH = Path(__file__).parent / "skill_optimizer_meta.md"

# Protected heading — regex that matches the Guidance block header
_GUIDANCE_HEADER_RE = re.compile(r"^##\s+Guidance\s*$", re.MULTILINE)

# Starter skill docs directory
_STARTER_SKILLS_DIR = Path(__file__).parent / "skills"


def _load_meta_skill() -> str:
    try:
        return _META_SKILL_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _load_starter_skill(task_type: TaskType) -> str:
    """Return the bundled starter skill doc for task_type, or a minimal default."""
    # File name: code_generation.md for CODE_GEN, etc.
    fname = f"{task_type.value.replace(' ', '_').lower()}.md"
    path = _STARTER_SKILLS_DIR / fname
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            f"# {task_type.value} Skill\n\n## Core Guidance\n\n## Guidance\n<!-- auto-updated -->\n"
        )


class SkillOptimizer:
    """Runs one optimization epoch for a single TaskType.

    Instantiated once per TaskType by SkillManager.
    """

    def __init__(
        self,
        task_type: TaskType,
        optimizer_client: "LLMClient",
        skill_store: "SkillStorePort",
        edit_budget: int = 150,
        validation_fraction: float = 0.2,
        min_trajectories: int = 5,
        slow_update_every: int = 5,
    ) -> None:
        self._task_type = task_type
        self._client = optimizer_client
        self._store = skill_store
        self._edit_budget = edit_budget
        self._validation_fraction = validation_fraction
        self._min_trajectories = min_trajectories
        self._slow_update_every = slow_update_every
        self._meta_skill = _load_meta_skill()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_epoch(self, trajectories: list[Trajectory]) -> SkillUpdateResult:
        """Run one optimization epoch; return the outcome."""
        if len(trajectories) < self._min_trajectories:
            logger.debug(
                "SkillOptimizer[%s]: only %d trajectories, need %d — skipping epoch",
                self._task_type.value,
                len(trajectories),
                self._min_trajectories,
            )
            return SkillUpdateResult(
                task_type=self._task_type,
                epoch=0,
                accepted=False,
                score_before=0.0,
                score_after=0.0,
                rejection_reason="not enough trajectories",
            )

        # ── 1. Train/val split ────────────────────────────────────────
        split = max(1, int(len(trajectories) * self._validation_fraction))
        train, val = trajectories[:-split], trajectories[-split:]

        # ── 2. Load current best skill ───────────────────────────────
        loaded = await self._store.load_best_skill(self._task_type)
        if loaded is None:
            current_skill = _load_starter_skill(self._task_type)
            current_score = 0.0
            epoch_num = 0
            # Persist starter skill so history is consistent
            await self._store.save_skill(self._task_type, current_skill, current_score, 0)
        else:
            current_skill, current_score, epoch_num = loaded

        # ── 3. Negative-feedback buffer ──────────────────────────────
        neg_feedback = await self._store.load_negative_feedback(self._task_type)

        # ── 4. Propose patches ───────────────────────────────────────
        prompt = self._build_optimizer_prompt(train, current_skill, neg_feedback)
        try:
            patches = await self._propose_patches(prompt)
        except Exception as exc:
            logger.warning(
                "SkillOptimizer[%s]: patch proposal failed: %s", self._task_type.value, exc
            )
            return SkillUpdateResult(
                task_type=self._task_type,
                epoch=epoch_num + 1,
                accepted=False,
                score_before=current_score,
                score_after=current_score,
                rejection_reason=f"patch proposal error: {exc}",
            )

        if not patches:
            logger.info("SkillOptimizer[%s]: no patches proposed", self._task_type.value)
            return SkillUpdateResult(
                task_type=self._task_type,
                epoch=epoch_num + 1,
                accepted=False,
                score_before=current_score,
                score_after=current_score,
                rejection_reason="no patches proposed",
            )

        # ── 5. Enforce edit budget ───────────────────────────────────
        patches = self._apply_budget(patches)

        # ── 6. Apply patches ─────────────────────────────────────────
        try:
            candidate_skill = self._apply_patches(current_skill, patches)
        except Exception as exc:
            logger.warning(
                "SkillOptimizer[%s]: patch application failed: %s", self._task_type.value, exc
            )
            return SkillUpdateResult(
                task_type=self._task_type,
                epoch=epoch_num + 1,
                accepted=False,
                score_before=current_score,
                score_after=current_score,
                rejection_reason=f"patch apply error: {exc}",
            )

        # ── 7. Estimate val score ─────────────────────────────────────
        val_score = self._estimate_val_score(val, current_score, patches)

        await self._store.save_patches(self._task_type, epoch_num + 1, patches, accepted=False)  # type: ignore[attr-defined]

        # ── 8. Validation gate ────────────────────────────────────────
        if val_score <= current_score:
            reason = f"no improvement: {val_score:.4f} vs {current_score:.4f}"
            logger.info(
                "SkillOptimizer[%s]: epoch %d rejected — %s",
                self._task_type.value,
                epoch_num + 1,
                reason,
            )
            await self._store.save_negative_feedback(self._task_type, patches, reason)
            return SkillUpdateResult(
                task_type=self._task_type,
                epoch=epoch_num + 1,
                accepted=False,
                score_before=current_score,
                score_after=val_score,
                patches_applied=patches,
                rejection_reason=reason,
            )

        # ── 9. Slow update of Guidance block ─────────────────────────
        if (epoch_num + 1) % self._slow_update_every == 0:
            try:
                candidate_skill = await self._update_guidance_block(candidate_skill, train)
            except Exception as exc:
                logger.warning(
                    "SkillOptimizer[%s]: guidance block update failed: %s",
                    self._task_type.value,
                    exc,
                )

        # ── 10. Persist ───────────────────────────────────────────────
        await self._store.save_skill(self._task_type, candidate_skill, val_score, epoch_num + 1)
        await self._store.save_patches(self._task_type, epoch_num + 1, patches, accepted=True)  # type: ignore[attr-defined]

        logger.info(
            "SkillOptimizer[%s]: epoch %d accepted — score %.4f → %.4f",
            self._task_type.value,
            epoch_num + 1,
            current_score,
            val_score,
        )
        return SkillUpdateResult(
            task_type=self._task_type,
            epoch=epoch_num + 1,
            accepted=True,
            score_before=current_score,
            score_after=val_score,
            patches_applied=patches,
        )

    async def best_skill(self) -> str | None:
        """Return the current best skill document, or None if not yet trained."""
        result = await self._store.load_best_skill(self._task_type)
        return result[0] if result else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_optimizer_prompt(
        self,
        train: list[Trajectory],
        current_skill: str,
        neg_feedback: list[dict],  # type: ignore[type-arg]
    ) -> str:
        """Construct the prompt sent to the optimizer model."""
        traj_lines: list[str] = []
        for i, t in enumerate(train[-10:], 1):  # cap at last 10 to keep context small
            traj_lines.append(
                f"[{i}] score={t.score:.3f} model={t.model_used}\n"
                f"  prompt: {t.prompt[:200]}\n"
                f"  critique: {t.critique_text[:300]}"
            )
        traj_block = "\n".join(traj_lines)

        nfb_lines: list[str] = []
        for fb in neg_feedback[:5]:
            nfb_lines.append(f"  reason: {fb['rejection_reason']}")
        nfb_block = "\n".join(nfb_lines) if nfb_lines else "  (none)"

        return (
            f"{self._meta_skill}\n\n"
            f"## Task type: {self._task_type.value}\n\n"
            f"## Current skill document\n```markdown\n{current_skill}\n```\n\n"
            f"## Recent trajectories (train split)\n{traj_block}\n\n"
            f"## Previously rejected patches (reasons)\n{nfb_block}\n\n"
            f"## Instructions\n"
            f"Propose patches to improve the skill. "
            f"Edit budget: {self._edit_budget} tokens total. "
            f"Output ONLY a JSON array of patch objects."
        )

    async def _propose_patches(self, prompt: str) -> list[SkillPatch]:
        """Call the optimizer LLM and parse the returned patch list."""
        from ..models import Model

        # Use REASONING model for the optimizer
        try:
            from ..models import ROUTING_TABLE

            models = ROUTING_TABLE.get(TaskType.REASONING, [])
            opt_model = models[0] if models else Model.GPT_4O_MINI
        except Exception:
            opt_model = Model.GPT_4O_MINI

        response = await self._client.call(  # type: ignore[no-untyped-call]
            model=opt_model,
            prompt=prompt,
            system=self._meta_skill,
            max_tokens=512,
            temperature=0.3,
        )
        raw_text = response.text if hasattr(response, "text") else str(response)

        return self._parse_patches(raw_text)

    def _parse_patches(self, raw: str) -> list[SkillPatch]:
        """Extract a JSON patch array from a possibly noisy LLM response."""
        # Try to find JSON array anywhere in the response
        match = re.search(r"\[[\s\S]*?\]", raw)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            logger.debug("SkillOptimizer: failed to parse patches JSON")
            return []

        patches: list[SkillPatch] = []
        for item in items:
            try:
                op = item.get("op", "append")
                if op not in ("append", "insert_after", "replace", "delete"):
                    continue
                patches.append(
                    SkillPatch(
                        op=op,
                        anchor=str(item.get("anchor", "")),
                        content=str(item.get("content", "")),
                        token_cost=int(item.get("token_cost", 10)),
                    )
                )
            except Exception:
                continue
        return patches

    def _apply_budget(self, patches: list[SkillPatch]) -> list[SkillPatch]:
        """Drop patches that push the total token_cost over edit_budget.

        Patches are kept in order; once the budget is exhausted, remaining
        patches are dropped.
        """
        kept: list[SkillPatch] = []
        total = 0
        for p in patches:
            cost = max(1, p.token_cost)
            if total + cost > self._edit_budget:
                break
            kept.append(p)
            total += cost
        return kept

    def _apply_patches(self, skill_doc: str, patches: list[SkillPatch]) -> str:
        """Apply a list of SkillPatch ops to the skill document text.

        The ``## Guidance`` block is protected — patches targeting it are silently skipped.
        """
        doc = skill_doc
        for patch in patches:
            # Protect the Guidance block
            if _GUIDANCE_HEADER_RE.search(patch.anchor or ""):
                continue
            try:
                doc = self._apply_single_patch(doc, patch)
            except Exception as exc:
                logger.debug("SkillOptimizer: skipping malformed patch (%s): %s", patch.op, exc)
        return doc

    def _apply_single_patch(self, doc: str, patch: SkillPatch) -> str:
        if patch.op == "append":
            return doc.rstrip() + "\n\n" + patch.content

        if patch.op == "delete":
            if patch.anchor and patch.anchor in doc:
                return doc.replace(patch.anchor, "", 1).strip()
            return doc

        if patch.op == "insert_after":
            if patch.anchor and patch.anchor in doc:
                idx = doc.index(patch.anchor) + len(patch.anchor)
                return doc[:idx] + "\n" + patch.content + doc[idx:]
            # Fallback to append
            return doc.rstrip() + "\n\n" + patch.content

        if patch.op == "replace":
            if patch.anchor and patch.anchor in doc:
                return doc.replace(patch.anchor, patch.content, 1)
            return doc

        return doc  # type: ignore[unreachable]

    def _estimate_val_score(
        self,
        val: list[Trajectory],
        current_score: float,
        patches: list[SkillPatch],
    ) -> float:
        """Cheap proxy for val-set improvement without extra LLM calls.

        Strategy:
        - Base = mean of val trajectory scores
        - Bonus = fraction of total budget used × 0.01 (rewards substantive edits)
        - Guard: if patches only add content (append/insert), give slight benefit of
          the doubt proportional to content length
        """
        if not val:
            return current_score

        base = sum(t.score for t in val) / len(val)

        budget_used = sum(max(1, p.token_cost) for p in patches)
        budget_fraction = min(1.0, budget_used / max(1, self._edit_budget))

        # Small bonus for substantive patches
        content_bonus = 0.0
        for p in patches:
            if p.op in ("append", "insert_after", "replace") and len(p.content) > 20:
                content_bonus += 0.005

        # Cap bonus at 0.02 so it never dominates
        bonus = min(0.02, budget_fraction * 0.01 + content_bonus)

        return min(1.0, base + bonus)

    async def _update_guidance_block(self, skill_doc: str, train: list[Trajectory]) -> str:
        """Rebuild the protected ## Guidance block from accepted epoch lessons.

        Makes one extra LLM call but only every slow_update_every epochs.
        """
        top_critiques = sorted(train, key=lambda t: t.score, reverse=True)[:5]
        critique_block = "\n".join(
            f"- score={t.score:.3f}: {t.critique_text[:200]}" for t in top_critiques
        )
        prompt = (
            f"Based on the following high-scoring task critiques for task type "
            f"'{self._task_type.value}', write 3-5 concise durable lessons that should "
            f"guide future task execution. Output ONLY the bullet-point lessons, "
            f"no preamble:\n\n{critique_block}"
        )
        try:
            from ..models import Model

            response = await self._client.call(  # type: ignore[no-untyped-call]
                model=Model.GPT_4O_MINI,
                prompt=prompt,
                max_tokens=256,
                temperature=0.2,
            )
            new_lessons = response.text.strip() if hasattr(response, "text") else ""
        except Exception:
            return skill_doc

        if not new_lessons:
            return skill_doc

        # Replace or append the ## Guidance block
        new_guidance = f"## Guidance\n{new_lessons}\n"
        if _GUIDANCE_HEADER_RE.search(skill_doc):
            # Replace everything from ## Guidance to the next ## heading (or end)
            skill_doc = re.sub(
                r"##\s+Guidance\s*\n[\s\S]*?(?=\n##|\Z)",
                new_guidance,
                skill_doc,
            )
        else:
            skill_doc = skill_doc.rstrip() + "\n\n" + new_guidance

        return skill_doc
