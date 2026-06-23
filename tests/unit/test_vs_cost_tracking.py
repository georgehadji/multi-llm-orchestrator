"""
Regression tests for VS-first cost tracking bug:
  - GenerateStage passed budget=None to VS sampler → cost invisible to budget
"""

import pytest

from orchestrator.engine_core.stages.generate import _get_vs_sampler


@pytest.mark.unit
def test_vs_sampler_receives_budget():
    """
    Reproducer: _get_vs_sampler was called without budget → sampler._budget=None.
    After fix:  budget is forwarded to the sampler.
    """

    class _FakeClient:
        async def call(self, **kw):
            pass

    class _FakeBudget:
        pass

    budget = _FakeBudget()
    sampler = _get_vs_sampler(_FakeClient(), budget=budget)
    assert (
        sampler._budget is budget
    ), "VS sampler must hold the budget reference so charges are applied"


@pytest.mark.unit
def test_vs_sampler_without_budget_still_constructs():
    """When budget is None (default), sampler constructs without error."""

    class _FakeClient:
        async def call(self, **kw):
            pass

    sampler = _get_vs_sampler(_FakeClient())
    assert sampler._budget is None


@pytest.mark.unit
async def test_vs_sampler_charges_budget_on_call():
    """
    When budget is wired, VerbalizedSampler.sample() charges the budget.
    """
    from orchestrator.application.verbalized_sampling import VerbalizedSampler
    from orchestrator.models import Model, VSConfig, ProbabilityFormat

    charges = []

    class _FakeBudget:
        async def charge(self, cost, label):
            charges.append((cost, label))

    class _FakeResponse:
        text = '{"responses": [{"text": "hello", "probability": 0.9}]}'
        cost_usd = 0.05

    class _FakeClient:
        async def call(self, **kw):
            return _FakeResponse()

    model = next(iter(Model))  # any model
    budget = _FakeBudget()
    sampler = VerbalizedSampler(client=_FakeClient(), budget=budget)
    cfg = VSConfig(k=1, temperature=0.3, fmt=ProbabilityFormat.CONFIDENCE)

    candidates = await sampler.sample(prompt="test", model=model, cfg=cfg)

    assert candidates, "Should have parsed one candidate"
    assert len(charges) == 1, "Budget should have been charged once"
    assert charges[0][0] == pytest.approx(0.05), "Charged cost should match response cost"
    assert charges[0][1] == "verbalized_sampling"
