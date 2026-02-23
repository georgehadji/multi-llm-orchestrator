"""
Tests for orchestrator/optimization.py
=======================================
Covers GreedyBackend, WeightedSumBackend, ParetoBackend, and
the ConstraintPlanner.set_backend() integration.

Pattern: reuses _make_profile() from test_constraint_planner.py style.
No pytest-asyncio; all tests are synchronous.
"""
from __future__ import annotations

import pytest

from orchestrator.models import Model, TaskType
from orchestrator.policy import ModelProfile
from orchestrator.policy_engine import PolicyEngine
from orchestrator.planner import ConstraintPlanner
from orchestrator.optimization import (
    GreedyBackend,
    OptimizationBackend,
    ParetoBackend,
    WeightedSumBackend,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(
    model: Model,
    provider: str,
    cost_in: float = 1.0,
    cost_out: float = 5.0,
    task_types: dict | None = None,
    quality: float = 0.8,
    trust: float = 1.0,
    latency: float = 2000.0,
) -> ModelProfile:
    return ModelProfile(
        model=model,
        provider=provider,
        cost_per_1m_input=cost_in,
        cost_per_1m_output=cost_out,
        capable_task_types=task_types or {TaskType.CODE_GEN: 0},
        quality_score=quality,
        trust_factor=trust,
        avg_latency_ms=latency,
    )


def _cost_fn(profile: ModelProfile, task_type: TaskType) -> float:
    """Simplified cost estimate: input_tokens=500, output_tokens=500."""
    return profile.estimate_cost(500, 500)


def _make_planner(
    profiles: dict[Model, ModelProfile],
    backend=None,
    all_healthy: bool = True,
) -> ConstraintPlanner:
    engine = PolicyEngine()
    health = {m: all_healthy for m in profiles}
    return ConstraintPlanner(profiles, engine, health, backend=backend)


# ── GreedyBackend ─────────────────────────────────────────────────────────────

class TestGreedyBackend:

    def test_empty_candidates_returns_none(self):
        backend = GreedyBackend()
        result = backend.select([], {}, TaskType.CODE_GEN, _cost_fn)
        assert result is None

    def test_single_candidate_returned(self):
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai")}
        backend = GreedyBackend()
        result = backend.select([Model.GPT_4O], profiles, TaskType.CODE_GEN, _cost_fn)
        assert result == Model.GPT_4O

    def test_higher_quality_wins_with_equal_cost(self):
        """Model with quality 0.99 should beat model with quality 0.5 at equal cost."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",      cost_in=1.0, quality=0.99),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai", cost_in=1.0, quality=0.50),
        }
        backend = GreedyBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O

    def test_lower_cost_wins_when_quality_equal(self):
        """With equal quality, cheaper model scores higher (quality/cost formula)."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",      cost_in=10.0, quality=0.8),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai", cost_in=0.1,  quality=0.8),
        }
        backend = GreedyBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O_MINI

    def test_low_trust_degrades_score_despite_quality(self):
        """High quality (0.99) with near-zero trust (0.01) loses to average model."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",      quality=0.99, trust=0.01),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai", quality=0.80, trust=1.0),
        }
        backend = GreedyBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O_MINI

    def test_priority_rank_tiebreaker(self):
        """When quality×trust/cost is equal, lower priority_rank (rank 0) wins."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              task_types={TaskType.CODE_GEN: 0},  # rank 0
                                              quality=0.8, trust=1.0, cost_in=1.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              task_types={TaskType.CODE_GEN: 1},  # rank 1
                                              quality=0.8, trust=1.0, cost_in=1.0),
        }
        backend = GreedyBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O


# ── WeightedSumBackend ────────────────────────────────────────────────────────

class TestWeightedSumBackend:

    def test_empty_candidates_returns_none(self):
        backend = WeightedSumBackend()
        assert backend.select([], {}, TaskType.CODE_GEN, _cost_fn) is None

    def test_latency_dominates_when_beta_one(self):
        """With β=1.0 (α=0.0), lower latency wins even at slightly higher cost."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=0.1, quality=0.8, latency=500.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=0.05, quality=0.8, latency=5000.0),
        }
        backend = WeightedSumBackend(alpha=0.0, beta=1.0)
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        # GPT_4O has lower latency → should win with β=1.0
        assert result == Model.GPT_4O

    def test_cost_dominates_when_alpha_one(self):
        """With α=1.0 (β=0.0), lower cost wins even at higher latency."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=10.0, quality=0.8, latency=500.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=0.1,  quality=0.8, latency=5000.0),
        }
        backend = WeightedSumBackend(alpha=1.0, beta=0.0)
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        # GPT_4O_MINI has lower cost → should win with α=1.0
        assert result == Model.GPT_4O_MINI

    def test_balanced_weights_default(self):
        """With α=β=0.5 (default), intermediate behaviour between pure-cost and pure-latency."""
        backend = WeightedSumBackend()  # defaults: α=0.5, β=0.5
        profiles = {
            Model.GPT_4O: _make_profile(Model.GPT_4O, "openai",
                                         cost_in=1.0, quality=0.9, latency=1000.0),
        }
        result = backend.select([Model.GPT_4O], profiles, TaskType.CODE_GEN, _cost_fn)
        assert result == Model.GPT_4O

    def test_custom_latency_scale(self):
        """Latency scale affects normalisation; low scale_ms makes latency penalise more."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=1.0, quality=0.8, latency=2000.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=1.0, quality=0.8, latency=100.0),
        }
        # With very small scale_ms (100ms), latency=2000ms is heavily penalised
        backend = WeightedSumBackend(alpha=0.5, beta=0.5, latency_scale_ms=100.0)
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O_MINI


# ── ParetoBackend ─────────────────────────────────────────────────────────────

class TestParetoBackend:

    def test_empty_candidates_returns_none(self):
        backend = ParetoBackend()
        assert backend.select([], {}, TaskType.CODE_GEN, _cost_fn) is None

    def test_single_candidate_returned(self):
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai")}
        backend = ParetoBackend()
        result = backend.select([Model.GPT_4O], profiles, TaskType.CODE_GEN, _cost_fn)
        assert result == Model.GPT_4O

    def test_dominated_model_excluded(self):
        """
        GPT_4O: cost=10, latency=5000 — dominated by GPT_4O_MINI on both axes.
        GPT_4O_MINI: cost=1, latency=1000 — Pareto-optimal.
        Only GPT_4O_MINI should survive.
        """
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=10.0, quality=0.9, latency=5000.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=1.0,  quality=0.7, latency=1000.0),
        }
        backend = ParetoBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        # GPT_4O is dominated on BOTH objectives → excluded from front
        assert result == Model.GPT_4O_MINI

    def test_both_non_dominated_quality_wins(self):
        """
        GPT_4O:      cost=1, latency=5000 — cheaper
        GPT_4O_MINI: cost=10, latency=500  — faster
        Neither dominates the other → both on Pareto front.
        Among Pareto-front, quality × trust picks the winner.
        """
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=1.0, quality=0.95, latency=5000.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=10.0, quality=0.70, latency=500.0),
        }
        backend = ParetoBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        # Both on Pareto front; GPT_4O has higher quality → wins
        assert result == Model.GPT_4O

    def test_all_equal_falls_back_to_quality(self):
        """When all models have identical cost/latency, quality × trust decides."""
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O, "openai",
                                              cost_in=1.0, quality=0.5, latency=2000.0),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai",
                                              cost_in=1.0, quality=0.9, latency=2000.0),
        }
        backend = ParetoBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI], profiles, TaskType.CODE_GEN, _cost_fn
        )
        assert result == Model.GPT_4O_MINI

    def test_three_candidates_one_dominated(self):
        """
        A: cost=5, latency=2000
        B: cost=2, latency=1000  — dominates A (both better)
        C: cost=10, latency=500  — neither dominates B nor is dominated by B
        Pareto front = {B, C}; among those, quality picks winner.
        """
        # Use GPT_4O, GPT_4O_MINI, GEMINI_FLASH as our three models
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O, "openai",
                                               cost_in=5.0, quality=0.9, latency=2000.0),
            Model.GPT_4O_MINI:  _make_profile(Model.GPT_4O_MINI, "openai",
                                               cost_in=2.0, quality=0.7, latency=1000.0),
            Model.GEMINI_FLASH: _make_profile(Model.GEMINI_FLASH, "google",
                                               cost_in=10.0, quality=0.85, latency=500.0),
        }
        backend = ParetoBackend()
        result = backend.select(
            [Model.GPT_4O, Model.GPT_4O_MINI, Model.GEMINI_FLASH],
            profiles, TaskType.CODE_GEN, _cost_fn
        )
        # GPT_4O is dominated by GPT_4O_MINI → excluded
        # Pareto front = {GPT_4O_MINI (0.7), GEMINI_FLASH (0.85)}
        # quality picks GEMINI_FLASH (0.85 > 0.7)
        assert result == Model.GEMINI_FLASH


# ── Backend swap in ConstraintPlanner ────────────────────────────────────────

class TestBackendSwapInPlanner:

    def _make_two_model_profiles(self) -> dict[Model, ModelProfile]:
        """
        GPT_4O:      cheap ($0.1/M), slow (5000ms), quality=0.8
        GPT_4O_MINI: expensive ($10/M), fast (500ms), quality=0.7
        """
        return {
            Model.GPT_4O: _make_profile(
                Model.GPT_4O, "openai",
                cost_in=0.1, quality=0.8, latency=5000.0,
                task_types={TaskType.CODE_GEN: 0},
            ),
            Model.GPT_4O_MINI: _make_profile(
                Model.GPT_4O_MINI, "openai",
                cost_in=10.0, quality=0.7, latency=500.0,
                task_types={TaskType.CODE_GEN: 1},
            ),
        }

    def test_greedy_picks_cheap_model(self):
        """GreedyBackend: quality/cost → GPT_4O wins (low cost boosts score)."""
        profiles = self._make_two_model_profiles()
        planner = _make_planner(profiles, backend=GreedyBackend())
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O

    def test_weighted_latency_picks_fast_model(self):
        """WeightedSumBackend with β=1: GPT_4O_MINI wins (500ms vs 5000ms)."""
        profiles = self._make_two_model_profiles()
        planner = _make_planner(profiles, backend=WeightedSumBackend(alpha=0.0, beta=1.0))
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O_MINI

    def test_set_backend_changes_selection(self):
        """set_backend() mid-lifecycle switches strategy."""
        profiles = self._make_two_model_profiles()
        planner = _make_planner(profiles, backend=GreedyBackend())

        first = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert first == Model.GPT_4O

        planner.set_backend(WeightedSumBackend(alpha=0.0, beta=1.0))
        second = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert second == Model.GPT_4O_MINI

    def test_pareto_backend_in_planner(self):
        """ParetoBackend wired into planner selects from Pareto front."""
        profiles = self._make_two_model_profiles()
        planner = _make_planner(profiles, backend=ParetoBackend())
        # GPT_4O (cheap+slow) and GPT_4O_MINI (costly+fast) — neither dominates
        # Pareto front = both; quality picks GPT_4O (0.8 > 0.7)
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O

    def test_backend_is_abstract_interface(self):
        """OptimizationBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OptimizationBackend()  # type: ignore[abstract]

    def test_default_backend_is_greedy(self):
        """Planner constructed without explicit backend uses GreedyBackend."""
        profiles = self._make_two_model_profiles()
        planner = _make_planner(profiles)  # no backend arg
        # Should behave identically to GreedyBackend
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O
