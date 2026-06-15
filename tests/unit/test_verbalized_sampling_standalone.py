"""
Standalone tests for VerbalizedSampler — bypasses broken package __init__.

TODO: When agent_model_registry.py is fixed, remove this file and
rename test_verbalized_sampling.py to test_verbalized_sampling.py (overwrite).
"""

import json
import sys
import unittest
from pathlib import Path

# ── Bypass broken package import by loading modules directly ────────────────

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib.util

# Load models.py
spec = importlib.util.spec_from_file_location(
    "orchestrator.models",
    "orchestrator/models.py",
    submodule_search_locations=[],
)
models_mod = importlib.util.module_from_spec(spec)

# Stub out the broken circular imports that models.py triggers
import types as _types
_budget_stub = _types.ModuleType("orchestrator.budget")
_budget_stub.Budget = object
sys.modules["orchestrator.budget"] = _budget_stub

spec.loader.exec_module(models_mod)

Model = models_mod.Model
VSConfig = models_mod.VSConfig
ProbabilityFormat = models_mod.ProbabilityFormat
TaskType = models_mod.TaskType

# Load verbalized_sampling.py
spec2 = importlib.util.spec_from_file_location(
    "orchestrator.application.verbalized_sampling",
    "orchestrator/application/verbalized_sampling.py",
)
vs_mod = importlib.util.module_from_spec(spec2)
# Wire up dependencies the module expects
vs_mod.Model = Model
vs_mod.VSConfig = VSConfig
vs_mod.ProbabilityFormat = ProbabilityFormat
vs_mod.TaskType = TaskType
# Stub domain.ports
_dp = _types.ModuleType("orchestrator.domain.ports")
sys.modules["orchestrator.domain.ports"] = _dp
spec2.loader.exec_module(vs_mod)

VSCandidate = vs_mod.VSCandidate
VerbalizedSampler = vs_mod.VerbalizedSampler


# ── Fake client ──────────────────────────────────────────────────────────────

class FakeAPIResponse:
    def __init__(self, text):
        self.text = text
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0


class FakeLLMClient:
    def __init__(self, text=""):
        self.response_text = text
        self.last_call = {}

    async def call(self, **kwargs):
        self.last_call = kwargs
        return FakeAPIResponse(text=self.response_text)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBuildSystem(unittest.TestCase):
    """VerbalizedSampler._build_system()"""

    def test_injects_k(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(k=5), "")
        self.assertIn("Generate 5 possible responses", system)

    def test_injects_k_1(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(k=1), "")
        self.assertIn("Generate 1 possible responses", system)

    def test_explicit_format(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(fmt=ProbabilityFormat.EXPLICIT), "")
        self.assertIn("estimated probability", system)
        self.assertIn("relative to the full distribution", system)

    def test_confidence_format(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(fmt=ProbabilityFormat.CONFIDENCE), "")
        self.assertIn("likelihood score", system)
        self.assertIn("representative or typical", system)

    def test_threshold_tail(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(probability_threshold=0.10), "")
        self.assertIn("below 0.1", system)

    def test_no_threshold_no_tail(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(probability_threshold=None), "")
        self.assertNotIn("below", system)

    def test_extra_prepended(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(), "CUSTOM_PREFIX")
        self.assertTrue(system.startswith("CUSTOM_PREFIX"))

    def test_json_format_in_prompt(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(), "")
        self.assertIn('"responses"', system)
        self.assertIn('"text"', system)
        self.assertIn('"probability"', system)


class TestParse(unittest.TestCase):
    """VerbalizedSampler._parse() — robustness"""

    def setUp(self):
        self.s = VerbalizedSampler(client=FakeLLMClient())

    def test_clean_json(self):
        text = json.dumps({
            "responses": [
                {"text": "A", "probability": 0.5},
                {"text": "B", "probability": 0.3},
            ]
        })
        candidates = self.s._parse(text, 2)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].text, "A")
        self.assertEqual(candidates[0].probability, 0.5)

    def test_fenced_json(self):
        text = (
            "```json\n"
            + json.dumps({"responses": [{"text": "fenced", "probability": 0.7}]})
            + "\n```"
        )
        candidates = self.s._parse(text, 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "fenced")

    def test_fence_no_tag(self):
        text = "```\n" + json.dumps({"responses": [{"text": "x", "probability": 0.5}]}) + "\n```"
        candidates = self.s._parse(text, 1)
        self.assertEqual(len(candidates), 1)

    def test_missing_prob_uniform(self):
        text = json.dumps({"responses": [{"text": "no prob"}]})
        candidates = self.s._parse(text, 1)
        self.assertEqual(candidates[0].probability, 1.0)  # 1/k = 1/1

    def test_multiple_missing_prob(self):
        text = json.dumps({"responses": [{"text": "a"}, {"text": "b"}]})
        candidates = self.s._parse(text, 2)
        self.assertEqual(candidates[0].probability, 0.5)  # 1/2
        self.assertEqual(candidates[1].probability, 0.5)

    def test_prob_clamped_high(self):
        text = json.dumps({"responses": [{"text": "high", "probability": 5.0}]})
        candidates = self.s._parse(text, 1)
        self.assertEqual(candidates[0].probability, 1.0)

    def test_prob_clamped_low(self):
        text = json.dumps({"responses": [{"text": "low", "probability": -1.0}]})
        candidates = self.s._parse(text, 1)
        self.assertEqual(candidates[0].probability, 0.0)

    def test_prob_as_string(self):
        text = json.dumps({"responses": [{"text": "str", "probability": "0.75"}]})
        candidates = self.s._parse(text, 1)
        self.assertEqual(candidates[0].probability, 0.75)

    def test_prob_invalid_string(self):
        text = json.dumps({"responses": [{"text": "bad", "probability": "abc"}]})
        candidates = self.s._parse(text, 1)
        self.assertEqual(candidates[0].probability, 1.0)  # uniform default

    def test_empty_response(self):
        self.assertEqual(self.s._parse("", 5), [])

    def test_garbage_non_json(self):
        self.assertEqual(self.s._parse("This is not JSON at all.", 3), [])

    def test_respects_k_limit(self):
        items = [{"text": f"r{i}", "probability": 0.1} for i in range(10)]
        text = json.dumps({"responses": items})
        candidates = self.s._parse(text, 3)
        self.assertEqual(len(candidates), 3)

    def test_skips_empty_text(self):
        text = json.dumps({
            "responses": [
                {"text": "", "probability": 0.5},
                {"text": "real", "probability": 0.5},
            ]
        })
        candidates = self.s._parse(text, 2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "real")

    def test_bare_array_format(self):
        text = json.dumps([{"text": "bare", "probability": 0.4}])
        candidates = self.s._parse(text, 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "bare")

    def test_partial_object_recovery(self):
        """JSON truncated mid-object — should still recover via regex."""
        text = '{"responses": [{"text": "partial", "probability": 0.6}'
        candidates = self.s._parse(text, 2)
        self.assertGreaterEqual(len(candidates), 1)

    def test_partial_array_recovery_brace(self):
        """Truncated before final close — recoverable."""
        text = '{"responses": [{"text": "A", "probability": 0.5}'
        candidates = self.s._parse(text, 1)
        self.assertGreaterEqual(len(candidates), 1)

    def test_extra_fields_ignored(self):
        text = json.dumps({
            "responses": [{"text": "real", "probability": 0.5, "extra_field": "ignored"}]
        })
        candidates = self.s._parse(text, 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "real")


class TestSample(unittest.TestCase):
    """VerbalizedSampler.sample() — integration shape"""

    def setUp(self):
        self.fake = FakeLLMClient(
            text=json.dumps({
                "responses": [
                    {"text": "R1", "probability": 0.6},
                    {"text": "R2", "probability": 0.4},
                ]
            })
        )
        self.s = VerbalizedSampler(client=self.fake)

    def _run(self, **kwargs):
        """Helper: run sample synchronously via asyncio.run."""
        import asyncio
        return asyncio.run(
            self.s.sample(prompt="test prompt", model=Model.GPT_4O_MINI, **kwargs)
        )

    def test_returns_vscandidates(self):
        candidates = self._run(cfg=VSConfig(k=2))
        self.assertEqual(len(candidates), 2)
        self.assertIsInstance(candidates[0], VSCandidate)
        self.assertEqual(candidates[0].text, "R1")

    def test_passes_model_and_temperature(self):
        self._run(cfg=VSConfig(k=2, temperature=0.7))
        self.assertEqual(self.fake.last_call.get("model"), Model.GPT_4O_MINI)
        self.assertEqual(self.fake.last_call.get("temperature"), 0.7)

    def test_passes_system_prompt(self):
        self._run(cfg=VSConfig(k=2), system_extra="CUSTOM_SYS")
        system = self.fake.last_call.get("system", "")
        self.assertIn("CUSTOM_SYS", system)
        self.assertIn("Generate 2", system)

    def test_passes_timeout_max_tokens(self):
        self._run(cfg=VSConfig(k=1), max_tokens=2048, timeout=90)
        self.assertEqual(self.fake.last_call.get("max_tokens"), 2048)
        self.assertEqual(self.fake.last_call.get("timeout"), 90)

    def test_passes_task_type(self):
        self._run(cfg=VSConfig(k=1), task_type=TaskType.CODE_GEN)
        self.assertEqual(self.fake.last_call.get("task_type"), TaskType.CODE_GEN)

    def test_empty_response_returns_empty(self):
        self.fake.response_text = ""
        candidates = self._run(cfg=VSConfig(k=3))
        self.assertEqual(candidates, [])

    def test_client_error_returns_empty(self):
        class BrokenClient:
            async def call(self, **kwargs):
                raise RuntimeError("API down")
        s = VerbalizedSampler(client=BrokenClient())
        import asyncio
        candidates = asyncio.run(s.sample(prompt="test", model=Model.GPT_4O_MINI))
        self.assertEqual(candidates, [])

    def test_tail_config_in_system(self):
        self._run(cfg=VSConfig(k=3, probability_threshold=0.10))
        system = self.fake.last_call.get("system", "")
        self.assertIn("below 0.1", system)


if __name__ == "__main__":
    unittest.main(verbosity=2)
