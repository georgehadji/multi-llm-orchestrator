"""
Tests for the Planning module (Capability 3).
"""

import pytest


class TestGoalDecomposer:
    """Recursive planning."""

    @pytest.mark.asyncio
    async def test_simple_goal_is_atomic(self):
        from orchestrator.planning.decomposer import GoalDecomposer

        d = GoalDecomposer()
        plan = await d.decompose("Write a function")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].is_atomic is True

    @pytest.mark.asyncio
    async def test_compound_goal_splits(self):
        from orchestrator.planning.decomposer import GoalDecomposer

        d = GoalDecomposer()
        plan = await d.decompose("Design the API and write tests")
        assert len(plan.tasks) >= 2

    @pytest.mark.asyncio
    async def test_depth_limit(self):
        from orchestrator.planning.decomposer import GoalDecomposer, MAX_DEPTH

        d = GoalDecomposer()
        # Create a deep recursion by splitting on "and" repeatedly
        goal = " and ".join([f"task {i}" for i in range(10)])
        plan = await d.decompose(goal)
        assert len(plan.tasks) > 0

    def test_split_by_bullets(self):
        from orchestrator.planning.decomposer import GoalDecomposer

        d = GoalDecomposer()
        result = d._split_goal("- item 1\n- item 2\n- item 3")
        assert len(result) >= 2

    def test_split_by_and(self):
        from orchestrator.planning.decomposer import GoalDecomposer

        d = GoalDecomposer()
        result = d._split_goal("Build frontend and write tests")
        assert len(result) == 2

    def test_single_goal(self):
        from orchestrator.planning.decomposer import GoalDecomposer

        d = GoalDecomposer()
        result = d._split_goal("Just one task")
        assert len(result) == 1

    def test_plan_merge(self):
        from orchestrator.planning.decomposer import Plan
        from orchestrator.planning.goal import SubGoal

        p1 = Plan(tasks=[SubGoal(id="a", description="A")])
        p2 = Plan(tasks=[SubGoal(id="b", description="B")])
        merged = Plan.merge([p1, p2])
        assert len(merged.tasks) == 2

    def test_plan_merge_dedup(self):
        from orchestrator.planning.decomposer import Plan
        from orchestrator.planning.goal import SubGoal

        p1 = Plan(tasks=[SubGoal(id="a", description="A")])
        p2 = Plan(tasks=[SubGoal(id="a", description="A")])
        merged = Plan.merge([p1, p2])
        assert len(merged.tasks) == 1
