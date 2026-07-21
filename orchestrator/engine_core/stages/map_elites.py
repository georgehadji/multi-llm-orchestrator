"""
MAPElitesPipeline — Quality-diversity evolutionary code optimization.
ARA Method #21: creates a population of code variants across a 3x3 feature grid
(complexity, performance), evolves them over generations, and returns the best.
"""

from __future__ import annotations

import json
import logging
import random
import threading

from ...ara_pipelines import BasePipeline, PipelineState, ReasoningMethod
from ...crosscutting.config import flags
from ...models import Model, ProbabilityFormat, Task, TaskResult, TaskStatus, VSConfig

# Lazy import for VerbalizedSampler (avoids circular dep at module level)
_VerbalizedSampler = None
_VerbalizedSampler_lock = threading.Lock()


def _get_vs_sampler(client):
    global _VerbalizedSampler
    if _VerbalizedSampler is None:
        with _VerbalizedSampler_lock:
            if _VerbalizedSampler is None:
                from ...application.verbalized_sampling import VerbalizedSampler

                _VerbalizedSampler = VerbalizedSampler
    return _VerbalizedSampler(client=client)


logger = logging.getLogger("orchestrator.engine_core.stages.map_elites")


_INITIALIZE_SYSTEM = (
    "You are a code diversity generator. Given a codebase and optimization goal, "
    "generate diverse implementation variants that explore the feature space. "
    "Each variant should differ in complexity and performance characteristics. "
)

_MUTATE_SYSTEM = (
    "You are a code evolution engine. Given a high-performing code variant, "
    "generate a derived variant that improves performance while managing complexity. "
    "Use diff-based changes where possible."
)


class MAPElitesPipeline(BasePipeline):
    """MAP-Elites quality-diversity evolutionary pipeline.

    Grid: 3x3 (complexity x performance), 3 generations.
    Selection: 20% elite (highest score), 30% diverse (novel feature combos),
    50% exploratory (random grid cells).
    """

    # Pipeline stage ordering — lower values run first
    priority: int = 800
    @classmethod
    def build_kwargs(cls, **deps):
        return {}

    def __init__(self, grid_rows: int = 3, grid_cols: int = 3, generations: int = 3) -> None:
        self._grid: list[list[dict | None]] = [[None] * grid_cols for _ in range(grid_rows)]
        self._rows = grid_rows
        self._cols = grid_cols
        self._generations = generations

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.MAP_ELITES

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.metadata["problem"] = task.prompt

        # 1. Initialize population
        population = await self._initialize(task.prompt)
        logger.info("MAP-Elites: initialized %d variants", len(population))

        for gen in range(self._generations):
            logger.info("MAP-Elites: generation %d/%d", gen + 1, self._generations)

            # 2. Place each variant into the grid
            scored = await self._evaluate(population)
            for code, score in scored:
                self._place_in_grid(code, score)

            # 3. Select elites for next generation
            elites = self._select_elites()
            if not elites:
                break

            # 4. Mutate elites
            population = await self._mutate(elites, task.prompt)

        # 5. Return best variant
        best = self._best_variant()
        state.final_output = best["code"] if best else ""
        state.final_score = best.get("score", 0.0) if best else 0.0
        return self._build_result(state)

    async def _initialize(self, prompt: str, count: int = 9) -> list[str]:
        """Generate count diverse code variants.

        When ORCH_VS_MAP_ELITES_SEEDING=true, uses VerbalizedSampler with
        tail-threshold sampling to deliberately seed unconventional grid cells
        from the low-probability tail of the distribution.
        """
        if self.client is None:
            return [f"# MAP-Elites variant: {prompt[:50]}"] * count

        # CodeWhale Phase 2: VS-tail seeding for diversity
        if flags.vs_map_elites_seeding:
            sampler = _get_vs_sampler(self.client)
            candidates = await sampler.sample(
                prompt=(
                    f"Goal: {prompt}\n\n"
                    f"Generate {count} diverse code implementations. "
                    f"Each should differ in complexity (simple/medium/advanced) "
                    f"and performance (slow/fast/optimal)."
                ),
                model=self._get_model(prompt),
                cfg=VSConfig(
                    k=count,
                    probability_threshold=0.10,  # Tail: sample unconventional approaches
                    temperature=0.9,
                    fmt=ProbabilityFormat.EXPLICIT,
                ),
                max_tokens=4096,
                timeout=120,
            )
            if candidates:
                logger.info(
                    "MAP-Elites VS-seeded: %d candidates from tail distribution",
                    len(candidates),
                )
                return [c.text for c in candidates]

            logger.warning("MAP-Elites VS returned no candidates — falling back to direct call")

        # Legacy path (flag off or VS returned nothing)
        resp = await self.client.call(
            model=self._get_model(prompt),
            system=_INITIALIZE_SYSTEM,
            prompt=(
                f"Goal: {prompt}\n\n"
                f"Generate {count} diverse code implementations. "
                f"Each should differ in complexity (simple/medium/advanced) "
                f"and performance (slow/fast/optimal).\n\n"
                f'Return JSON: {{"variants": ["<code1>", "<code2>", ...]}}'
            ),
            max_tokens=4096,
            temperature=0.8,
        )
        data = self._extract_json(resp.text) or {}
        variants = data.get("variants", [])
        return variants if len(variants) >= 3 else [f"# Variant {i}" for i in range(count)]

    async def _evaluate(self, variants: list[str]) -> list[tuple[str, dict]]:
        """Score each variant. Returns [(code, {complexity, performance, score})]."""
        scored = []
        for v in variants:
            if v.strip():
                # Simple heuristic scoring based on code characteristics
                score = self._heuristic_score(v)
                scored.append((v, score))
        return scored

    def _heuristic_score(self, code: str) -> dict:
        """Score code variant based on features."""
        lines = code.splitlines()
        line_count = len(lines)
        avg_line_len = sum(len(l) for l in lines) / max(line_count, 1)

        complexity = min(line_count / 50, 1.0)  # 0-50 lines maps to 0-1
        performance = 1.0 - min(avg_line_len / 80, 1.0)  # shorter avg line = faster
        score = complexity * 0.3 + performance * 0.7

        return {
            "complexity": round(complexity, 2),
            "performance": round(performance, 2),
            "score": round(score, 2),
        }

    def _place_in_grid(self, code: str, features: dict) -> None:
        """Place a variant into the grid based on its feature dimensions."""
        row = int(features["complexity"] * (self._rows - 1))
        col = int(features["performance"] * (self._cols - 1))
        row = min(row, self._rows - 1)
        col = min(col, self._cols - 1)

        current = self._grid[row][col]
        if current is None or features["score"] > current["score"]:
            self._grid[row][col] = {"code": code, **features}

    def _select_elites(self) -> list[str]:
        """Select 20% elite, 30% diverse, 50% exploratory from the grid."""
        cells = [c for row in self._grid for c in row if c is not None]
        if not cells:
            return []

        # 20% elite (highest score)
        elite_count = max(1, len(cells) // 5)
        elite = sorted(cells, key=lambda c: c["score"], reverse=True)[:elite_count]

        # 30% diverse (furthest feature distance)
        remaining = [c for c in cells if c not in elite]
        diverse_count = max(1, len(remaining) // 3)
        diverse = sorted(
            remaining, key=lambda c: abs(c["complexity"] - c["performance"]), reverse=True
        )[:diverse_count]

        # 50% exploratory (random)
        exploratory_remaining = [c for c in remaining if c not in diverse]
        exp_count = min(len(exploratory_remaining), max(1, len(cells) // 2))
        exploratory = (
            random.sample(exploratory_remaining, exp_count) if exploratory_remaining else []
        )

        return [c["code"] for c in elite + diverse + exploratory]

    async def _mutate(self, elites: list[str], prompt: str) -> list[str]:
        """Generate new variants from elites via LLM mutation."""
        if self.client is None or not elites:
            return [f"# Mutant of {e[:40]}" for e in elites]

        resp = await self.client.call(
            model=self._get_model(prompt),
            system=_MUTATE_SYSTEM,
            prompt=(
                f"Goal: {prompt}\n\n"
                f"Parent code:\n{elites[0][:2000]}\n\n"
                f"Generate {len(elites) * 3} mutated variants. "
                f"Some should be small mutations, some should explore new approaches. "
                f'Return JSON: {{"mutations": ["<code1>", "<code2>", ...]}}'
            ),
            max_tokens=4096,
            temperature=0.9,
        )
        data = self._extract_json(resp.text) or {}
        mutations = data.get("mutations", [])
        return mutations if mutations else [f"# Mutant {i}" for i in range(len(elites) * 3)]

    def _best_variant(self) -> dict | None:
        """Return the highest-scoring variant from the entire grid."""
        best = None
        for row in self._grid:
            for cell in row:
                if cell is not None and (best is None or cell["score"] > best["score"]):
                    best = cell
        return best

    def _get_model(self, prompt: str) -> Model | None:
        """Select model for MAP-Elites operations."""
        from ...models import Model as M

        return M.GPT_4O_MINI

    def _extract_json(self, text: str) -> dict:
        import re

        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=self._get_model(state.task.prompt),
            status=TaskStatus.COMPLETED if state.final_score > 0 else TaskStatus.DEGRADED,
            metadata={"method": self.get_method().value},
        )
