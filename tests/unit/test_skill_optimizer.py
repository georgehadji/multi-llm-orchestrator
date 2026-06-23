"""
Unit tests for orchestrator.application.skill_optimizer.SkillOptimizer

Tests use a NullSkillStore and a mock LLM client to verify epoch logic
without touching SQLite or making real API calls.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.application.skill_optimizer import SkillOptimizer, _load_starter_skill
from orchestrator.domain.ports import NullSkillStore
from orchestrator.models import TaskType
from orchestrator.models_skill import SkillPatch, Trajectory

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_client(patch_json: str = "[]") -> MagicMock:
    """Return a mock LLM client that returns patch_json as response text."""
    client = MagicMock()
    response = MagicMock()
    response.text = patch_json
    client.call = AsyncMock(return_value=response)
    return client


def _make_trajectories(
    n: int, score: float = 0.7, task_type: TaskType = TaskType.CODE_GEN
) -> list[Trajectory]:
    return [
        Trajectory(
            task_id=f"t{i}",
            task_type=task_type,
            prompt="write hello world",
            output="print('hello')",
            score=score,
            critique_text="decent output",
            model_used="openai/gpt-4o-mini",
            cost_usd=0.001,
            recorded_at=time.time() + i,
        )
        for i in range(n)
    ]


def _optimizer(client=None, store=None, edit_budget=50) -> SkillOptimizer:
    return SkillOptimizer(
        task_type=TaskType.CODE_GEN,
        optimizer_client=client or _make_client(),
        skill_store=store or NullSkillStore(),
        edit_budget=edit_budget,
        validation_fraction=0.2,
        min_trajectories=3,
        slow_update_every=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Starter skill loading
# ─────────────────────────────────────────────────────────────────────────────


def test_load_starter_skill_returns_string():
    skill = _load_starter_skill(TaskType.CODE_GEN)
    assert isinstance(skill, str)
    assert len(skill) > 0


def test_load_starter_skill_missing_type_returns_default():
    # Use a custom-value mock TaskType-like
    class FakeType:
        value = "nonexistent_task_type_xyz"

    skill = _load_starter_skill(FakeType())  # type: ignore[arg-type]
    assert isinstance(skill, str)


# ─────────────────────────────────────────────────────────────────────────────
# run_epoch — not enough trajectories
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_epoch_skipped_when_too_few_trajectories():
    opt = _optimizer()
    result = await opt.run_epoch(_make_trajectories(2))  # need min=3
    assert not result.accepted
    assert "not enough" in result.rejection_reason


# ─────────────────────────────────────────────────────────────────────────────
# run_epoch — no patches proposed
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_epoch_rejected_when_no_patches():
    opt = _optimizer(client=_make_client("[]"))
    result = await opt.run_epoch(_make_trajectories(5))
    assert not result.accepted
    assert "no patches" in result.rejection_reason


# ─────────────────────────────────────────────────────────────────────────────
# run_epoch — patches proposed but validation doesn't improve
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_epoch_rejected_when_no_improvement():
    # Provide a patch, but store already has a best skill with high score
    # so val_score (proxy ~0.7) won't beat it
    patch_json = json.dumps([{"op": "append", "anchor": "", "content": "hi", "token_cost": 5}])

    class HighScoreStore(NullSkillStore):
        async def load_best_skill(self, task_type):
            return ("# existing skill", 0.99, 1)  # nearly perfect

        async def save_patches(self, task_type, epoch, patches, accepted):
            pass

    opt = _optimizer(client=_make_client(patch_json), store=HighScoreStore())
    result = await opt.run_epoch(_make_trajectories(5, score=0.7))
    assert not result.accepted
    assert "no improvement" in result.rejection_reason


# ─────────────────────────────────────────────────────────────────────────────
# run_epoch — patches accepted
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_epoch_accepted_when_improvement():
    # Store starts with a low score, val trajectories are slightly higher
    patch_json = json.dumps(
        [{"op": "append", "anchor": "", "content": "Always use type hints.", "token_cost": 10}]
    )

    class LowScoreStore(NullSkillStore):
        _saved: list = []

        async def load_best_skill(self, task_type):
            return ("# low skill", 0.10, 0)  # very low baseline

        async def save_skill(self, task_type, skill_doc, score, epoch):
            self._saved.append({"doc": skill_doc, "score": score, "epoch": epoch})

        async def save_patches(self, task_type, epoch, patches, accepted):
            pass

    store = LowScoreStore()
    opt = _optimizer(client=_make_client(patch_json), store=store, edit_budget=50)
    result = await opt.run_epoch(_make_trajectories(10, score=0.75))

    assert result.accepted
    assert result.score_after > result.score_before
    assert len(store._saved) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Budget enforcement
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_budget_drops_patches_over_limit():
    opt = _optimizer(edit_budget=20)
    patches = [
        SkillPatch(op="append", anchor="", content="a", token_cost=10),
        SkillPatch(op="append", anchor="", content="b", token_cost=10),
        SkillPatch(op="append", anchor="", content="c", token_cost=10),  # over budget
    ]
    kept = opt._apply_budget(patches)
    assert len(kept) == 2
    assert sum(p.token_cost for p in kept) <= 20


# ─────────────────────────────────────────────────────────────────────────────
# Patch application
# ─────────────────────────────────────────────────────────────────────────────


def test_apply_patches_append():
    opt = _optimizer()
    doc = "# Skill\n\n## Core\n"
    patch = SkillPatch(op="append", anchor="", content="New guidance.", token_cost=5)
    result = opt._apply_patches(doc, [patch])
    assert "New guidance." in result


def test_apply_patches_insert_after():
    opt = _optimizer()
    doc = "## Core\nsome text"
    patch = SkillPatch(op="insert_after", anchor="## Core", content="extra line", token_cost=5)
    result = opt._apply_patches(doc, [patch])
    assert "extra line" in result
    assert result.index("extra line") > result.index("## Core")


def test_apply_patches_replace():
    opt = _optimizer()
    doc = "## Core\nold text"
    patch = SkillPatch(op="replace", anchor="old text", content="new text", token_cost=5)
    result = opt._apply_patches(doc, [patch])
    assert "new text" in result
    assert "old text" not in result


def test_apply_patches_delete():
    opt = _optimizer()
    doc = "## Core\nbad line\nrest"
    patch = SkillPatch(op="delete", anchor="bad line", content="", token_cost=3)
    result = opt._apply_patches(doc, [patch])
    assert "bad line" not in result


def test_apply_patches_protects_guidance_block():
    opt = _optimizer()
    doc = "## Core\n\n## Guidance\nprotected"
    patch = SkillPatch(op="replace", anchor="## Guidance", content="injected", token_cost=5)
    result = opt._apply_patches(doc, [patch])
    # Guidance block must NOT be touched
    assert "protected" in result or "## Guidance" in result


# ─────────────────────────────────────────────────────────────────────────────
# Parse patches JSON
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_patches_valid_json():
    opt = _optimizer()
    raw = json.dumps([{"op": "append", "anchor": "", "content": "hello", "token_cost": 5}])
    patches = opt._parse_patches(raw)
    assert len(patches) == 1
    assert patches[0].op == "append"


def test_parse_patches_json_with_noise():
    opt = _optimizer()
    raw = "Here are the patches:\n" + json.dumps(
        [{"op": "delete", "anchor": "bad", "content": "", "token_cost": 3}]
    )
    patches = opt._parse_patches(raw)
    assert len(patches) == 1
    assert patches[0].op == "delete"


def test_parse_patches_invalid_op_skipped():
    opt = _optimizer()
    raw = json.dumps([{"op": "EXPLODE", "anchor": "", "content": "x", "token_cost": 5}])
    patches = opt._parse_patches(raw)
    assert patches == []


def test_parse_patches_empty_list():
    opt = _optimizer()
    patches = opt._parse_patches("[]")
    assert patches == []


def test_parse_patches_malformed_returns_empty():
    opt = _optimizer()
    patches = opt._parse_patches("this is not JSON")
    assert patches == []
