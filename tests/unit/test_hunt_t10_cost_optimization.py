"""
T10 (cost_optimization/ remainder) proof-of-defect and no-regression tests.

Four VERIFIED DEFECTs from docs/hunts/t10-cost-optimization/inventory.md:

C1 — orchestrator/token_budget.py was a byte-for-byte independent duplicate
     (not a shim) of orchestrator/infrastructure/token_budget.py — the same
     unshimmed-duplicate-pair shape every prior tier (T1/T2/T3/T5/T7/T9) has
     found silently diverging once one copy gets a fix the other doesn't.
C2 — orchestrator/provisioned_throughput.py was likewise an unshimmed,
     byte-for-byte independent duplicate of
     orchestrator/operations/provisioned_throughput.py (the module
     orchestrator/infrastructure/provisioned_throughput.py already shims to).
C3 — cost_optimization/cost_optimization_integration.py's
     Tier1OptimizationMixin used a same-package-depth relative import
     (`from .log_config import get_logger`) one dot short for its actual
     depth (orchestrator/cost_optimization/), raising ModuleNotFoundError
     unconditionally at import time. The module was deleted outright in
     docs/plans/2026-09-07-patterns-convergence-and-wire-or-delete.md S6
     (TEST-ONLY, and the batch client it wrapped could never provide real
     batching over the OpenRouter adapter) — its C3 regression test below
     went with it.
C4 — cost_optimization/docker_sandbox.py::DockerSandbox.execute() wrote
     caller-supplied `code_files` filenames straight into the sandbox
     workspace with no path-containment check — an absolute path or a
     "../" traversal filename could write outside the temporary sandbox
     directory onto the host filesystem.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_root_token_budget_is_canonical():
    from orchestrator.infrastructure.token_budget import TokenBudgetManager as canonical
    from orchestrator.token_budget import TokenBudgetManager as via_root

    assert via_root is canonical


# --- C2 -----------------------------------------------------------------


def test_c2_root_provisioned_throughput_is_canonical():
    from orchestrator.operations.provisioned_throughput import (
        ProvisionedThroughputManager as canonical,
    )
    from orchestrator.provisioned_throughput import (
        ProvisionedThroughputManager as via_root,
    )

    assert via_root is canonical


# --- C4 -----------------------------------------------------------------


async def _fake_check_docker(self) -> bool:
    return True


class _FakeDockerModule:
    def from_env(self):
        # Never actually reached before the containment check fires.
        return object()


@pytest.mark.asyncio
async def test_c4_docker_sandbox_rejects_path_traversal_filename(monkeypatch):
    import sys

    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    monkeypatch.setitem(sys.modules, "docker", _FakeDockerModule())
    monkeypatch.setattr(DockerSandbox, "_check_docker", _fake_check_docker)

    sandbox = DockerSandbox()
    result = await sandbox.execute(
        code_files={"../../../etc/evil.txt": "malicious content"},
        command="true",
    )

    assert result.return_code == -1
    assert (
        "escapes the sandbox" in result.error
    ), f"expected a containment error, got: {result.error!r}"


@pytest.mark.asyncio
async def test_c4_docker_sandbox_accepts_normal_filename(monkeypatch):
    """Confirms the fix doesn't break the legitimate case."""
    import sys

    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    monkeypatch.setitem(sys.modules, "docker", _FakeDockerModule())
    monkeypatch.setattr(DockerSandbox, "_check_docker", _fake_check_docker)

    sandbox = DockerSandbox()
    result = await sandbox.execute(
        code_files={"main.py": "print('hello')"},
        command="true",
    )

    # Still fails (the fake docker client's from_env() object has no real
    # .containers.run()), but NOT on the containment check — confirms a
    # normal filename passes through it untouched.
    assert "escapes the sandbox" not in (result.error or "")


# --- T1B3 (V3 precision defect audit, docs/audits/v3/T1/batch3) ------------
#
# A separate audit campaign (docs/PRECISION_DEFECT_AUDIT_PLAN.md) found ten
# more defects in cost_optimization/{tier3_quality,prompt_cache,
# streaming_validator,structured_output}.py — all confirmed DEAD CODE (no
# caller anywhere in the repo), fixed anyway since they're public API that
# invites direct construction.


@pytest.mark.asyncio
async def test_t1b3_01_record_failure_empty_eval_scores(tmp_path):
    """Fires T1B3-01 without the fix; passes with it. Violated property:
    `avg` must be bound before use regardless of whether eval_scores is
    empty (EvalDatasetBuilder.record_failure raised UnboundLocalError)."""
    from orchestrator.cost_optimization.tier3_quality import EvalDatasetBuilder

    builder = EvalDatasetBuilder(dataset_path=str(tmp_path / "eval.jsonl"))

    try:
        await builder.record_failure(
            task_prompt="p",
            generated_code="c",
            errors=["SomeError: boom"],
            eval_scores={},  # falsy -> triggers the original UnboundLocalError
            model="deepseek/deepseek-v4-flash",
            task_type="code_generation",
        )
    except UnboundLocalError:
        pytest.fail("defect still present: avg referenced before assignment")


@pytest.mark.asyncio
async def test_t1b3_02_clear_cache_counts_evictions():
    """Fires T1B3-02 without the fix; passes with it. Violated property:
    evictions must reflect entries actually removed by clear_cache
    (PromptCacher counted the size of the dict AFTER clearing it)."""
    import time

    from orchestrator.cost_optimization.prompt_cache import CacheEntry, PromptCacher

    cacher = PromptCacher()
    cacher._cache_entries["k1"] = CacheEntry(
        key="k1", created_at=time.time(), last_accessed=time.time()
    )
    cacher._cache_entries["k2"] = CacheEntry(
        key="k2", created_at=time.time(), last_accessed=time.time()
    )

    await cacher.clear_cache()

    assert cacher.metrics.evictions == 2


@pytest.mark.asyncio
async def test_t1b3_03_first_call_is_not_counted_as_a_hit():
    """Fires T1B3-03 without the fix; passes with it. Violated property:
    `hits` must only count an actual provider cache read, not every
    successful call (which includes the necessarily-a-miss first call)."""
    from dataclasses import dataclass

    from orchestrator.cost_optimization.prompt_cache import PromptCacher

    @dataclass
    class _Usage:
        cache_read_input_tokens: int = 0

    @dataclass
    class _Response:
        usage: _Usage

    class _FirstTimeAnthropicClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                return _Response(usage=_Usage(cache_read_input_tokens=0))

    cacher = PromptCacher(client=_FirstTimeAnthropicClient())

    await cacher.call_with_cache(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="sys",
    )

    assert cacher.metrics.hits == 0, "first-time cache creation must not count as a hit"
    assert cacher.metrics.misses == 1


@pytest.mark.asyncio
async def test_t1b3_04_call_with_cache_preserves_user_messages():
    """Fires T1B3-04 without the fix; passes with it. Violated property:
    the user's `messages` content must reach the model, not be replaced by
    the caching system prompt (the fallback branch passed system_prompt
    positionally into UnifiedClient.call's `prompt` slot, dropping the
    actual conversation)."""
    from orchestrator.cost_optimization.prompt_cache import PromptCacher

    class _FakeUnifiedClient:
        def __init__(self):
            self.calls = []

        async def call(self, model, prompt, system=None, **kwargs):
            self.calls.append({"model": model, "prompt": prompt, "system": system})
            return {"text": "ok"}

    client = _FakeUnifiedClient()
    cacher = PromptCacher(client=client)

    await cacher.call_with_cache(
        model="deepseek/deepseek-v4-flash",
        messages=[{"role": "user", "content": "USER_TASK_CONTENT"}],
        system_prompt="SYSTEM_CACHE_PROMPT",
    )

    assert len(client.calls) == 1
    assert (
        "USER_TASK_CONTENT" in client.calls[0]["prompt"]
    ), "user messages were dropped from the outbound call"
    assert client.calls[0]["system"] == "SYSTEM_CACHE_PROMPT"


def test_t1b3_05_ellipsis_pattern_detects_literal_ellipsis():
    """Fires T1B3-05 without the fix; passes with it. Violated property:
    the 'Ellipsis in code' pattern must match a literal '...' placeholder
    (the raw string was double-escaped: r"\\.\\.\\." matches "backslash,
    any-char" x3 in regex syntax, not three literal dots)."""
    from orchestrator.cost_optimization.streaming_validator import StreamingValidator

    validator = StreamingValidator()
    text = "def foo():\n    ...\n" + ("x" * 250)

    failure = validator._detect_early_failure(text, task_type="code_generation")

    assert failure is not None, "ellipsis placeholder was not detected"


@pytest.mark.asyncio
async def test_t1b3_06_early_abort_window_matches_documented_default():
    """Fires T1B3-06 without the fix; passes with it. Violated property:
    early_abort_tokens=500 (default) must check the first ~500 estimated
    tokens, not ~125 (the check divided an already-token-scaled threshold
    by 4 a second time)."""
    from orchestrator.cost_optimization.streaming_validator import StreamingValidator
    from orchestrator.models import Model

    class _SlowFailureClient:
        async def stream(self, model, prompt, **kwargs):
            yield "x" * 600  # ~150 estimated tokens of filler
            yield "i cannot help with that"

    validator = StreamingValidator(client=_SlowFailureClient())

    result = await validator.stream_and_validate(
        model=Model.CLAUDE_SONNET_5,
        prompt="do something",
        early_abort_tokens=500,
    )

    assert (
        result.early_aborted
    ), "failure marker at ~150 estimated tokens was missed by a too-narrow window"


@pytest.mark.asyncio
async def test_t1b3_07_cost_uses_token_estimate_not_char_count():
    """Fires T1B3-07 without the fix; passes with it. Violated property:
    .cost must be derived from the same token estimate as .total_tokens,
    not from the raw character length of the response (~4x inflation)."""
    from orchestrator.cost_optimization.streaming_validator import StreamingValidator
    from orchestrator.models import Model

    class _FixedTextClient:
        async def stream(self, model, prompt, **kwargs):
            yield "y" * 4000  # 4000 chars == ~1000 estimated tokens

    validator = StreamingValidator(client=_FixedTextClient())

    result = await validator.stream_and_validate(
        model=Model.CLAUDE_SONNET_5, prompt="p", early_abort_tokens=0
    )

    expected_cost = validator._estimate_cost(Model.CLAUDE_SONNET_5, result.total_tokens)
    assert abs(result.cost - expected_cost) < 1e-9, (
        f"cost {result.cost} does not match the token-based estimate {expected_cost} "
        "-- likely computed from character count instead"
    )


def test_t1b3_08_model_costs_matches_canonical_cost_table():
    """Fires T1B3-08 without the fix; passes with it. Violated property:
    the local MODEL_COSTS duplicate must match orchestrator.models.COST_TABLE
    for every model it lists (was stale by 1.5x-14x per model)."""
    from orchestrator.cost_optimization.streaming_validator import StreamingValidator
    from orchestrator.models import COST_TABLE

    for model, local_prices in StreamingValidator.MODEL_COSTS.items():
        canonical = COST_TABLE[model]
        assert local_prices["input"] == canonical["input"], (
            f"{model}: stale input price {local_prices['input']} != "
            f"canonical {canonical['input']}"
        )
        assert local_prices["output"] == canonical["output"], (
            f"{model}: stale output price {local_prices['output']} != "
            f"canonical {canonical['output']}"
        )


def test_t1b3_09_fallback_relative_to_failed_model():
    """Fires T1B3-09 without the fix; passes with it. Violated property:
    the retry model must be chosen relative to the model that just failed,
    not by raw loop-attempt index into the global FALLBACK_CHAIN (which
    could escalate the cheapest model straight to the most expensive)."""
    from orchestrator.cost_optimization.streaming_validator import StreamingValidator
    from orchestrator.models import Model

    validator = StreamingValidator()

    next_model = validator._next_fallback_model(Model.DEEPSEEK_V4_FLASH)
    assert next_model is None, (
        "expected fallback to be exhausted after the cheapest/last model, "
        f"got escalation to {next_model!r}"
    )

    assert validator._next_fallback_model(Model.CLAUDE_OPUS_4_8) == Model.GPT_4O


def test_t1b3_10_extracts_balanced_json_past_leading_brace_prose():
    """Fires T1B3-10 without the fix; passes with it. Violated property:
    JSON extraction must return the balanced object, not everything from
    the first '{' to the last '}' in the whole response."""
    from orchestrator.cost_optimization.structured_output import StructuredOutputEnforcer

    text = (
        "Sure, {like this} is an example. Here is the real answer:\n"
        '{"summary": "ok", "key_points": [], "tokens_saved": 0}'
    )

    extracted = StructuredOutputEnforcer._extract_json_object(text)

    assert (
        extracted == '{"summary": "ok", "key_points": [], "tokens_saved": 0}'
    ), f"expected only the real JSON object, got: {extracted!r}"


# --- T1 batch 4 (V3 precision defect audit, docs/audits/v3/T1/batch4) ------
#
# model_cascading.py, github_push.py, speculative_gen.py -- all UNKNOWN
# reachability (no confirmed caller found by static tracing), fixed anyway.


class _AlwaysFailsClient:
    async def call(self, model: str, **kwargs: object) -> str:
        raise RuntimeError(f"{model} is unavailable")


@pytest.mark.asyncio
async def test_casc_01_all_tiers_failing_raises_cleanly_not_crashes():
    """Fires CASC-01 without the fix (IndexError/UnboundLocalError leaks
    out); passes with it (a clear RuntimeError instead). Violated property:
    the post-loop fallback may only run once at least one tier has actually
    produced a response."""
    from orchestrator.cost_optimization.model_cascading import ModelCascader

    cascader = ModelCascader(client=_AlwaysFailsClient())
    with pytest.raises(RuntimeError, match="cascade"):
        await cascader.cascading_generate(task_prompt="write code", task_type="code_generation")


@pytest.mark.asyncio
async def test_casc_02_cache_key_includes_task_type(monkeypatch):
    """Fires CASC-02 without the fix (second call wrongly returns the first
    call's cached score); passes with it. Violated property: a memoization
    key must include every input that affects the computed value."""
    from orchestrator.cost_optimization.model_cascading import ModelCascader
    from orchestrator.crosscutting.config import flags
    from orchestrator.models import TaskType, Verdict

    class _StubVerifier:
        async def verify(self, *, prompt, response, task_type):
            score = 0.9 if task_type == TaskType.CODE_GEN else 0.1
            return Verdict(passed=score >= 0.5, score=score)

    monkeypatch.setattr(flags, "use_objective_verifiers", True, raising=False)

    cascader = ModelCascader(verifier=_StubVerifier())
    prompt, response, model = "same prompt", "same response", "test-model"

    score_a = await cascader._quick_evaluate(prompt, response, model, task_type="code_generation")
    score_b = await cascader._quick_evaluate(prompt, response, model, task_type="evaluation")

    assert score_a == pytest.approx(0.9)
    assert score_b == pytest.approx(0.1), (
        "got the code_generation score back for an 'evaluation' lookup -- "
        "the cache key does not vary with task_type"
    )


def test_spec_01_default_cheap_model_id_matches_cost_table():
    """Fires SPEC-01 without the fix (id mismatch); passes with it.
    Violated property: a model id used for generation must be a valid
    MODEL_COSTS lookup key, or cost accounting silently uses the wrong
    (more expensive, generic-default) price."""
    from orchestrator.cost_optimization.speculative_gen import SpeculativeGenerator

    gen = SpeculativeGenerator()
    for task_type, pair in gen.DEFAULT_MODEL_PAIRS.items():
        cheap_id = pair["cheap"]
        assert (
            cheap_id.lower() in gen.MODEL_COSTS
        ), f"{task_type}: cheap model id {cheap_id!r} has no MODEL_COSTS entry"


@pytest.mark.asyncio
async def test_spec_02_empty_cheap_result_is_not_treated_as_absent():
    """Fires SPEC-02 without the fix (raises RuntimeError instead of
    returning the valid empty cheap result); passes with it. Violated
    property: 'falsy' must not be conflated with 'missing' when deciding
    fallback eligibility."""
    from orchestrator.cost_optimization.speculative_gen import SpeculativeGenerator

    class _FlakyClient:
        async def call(self, model: str, **kwargs: object):
            if "premium" in model:
                raise RuntimeError("simulated premium outage")
            return type("Resp", (), {"text": ""})()

    gen = SpeculativeGenerator(client=_FlakyClient())
    result = await gen.speculative_generate(
        prompt="x",
        cheap_model="cheap-model",
        premium_model="premium-model",
        threshold=0.99,  # forces the low empty-string score to be rejected -> awaits premium
    )
    assert result.response == ""
    assert result.model_used == "cheap-model"


@pytest.mark.asyncio
async def test_spec_03_premium_failure_does_not_leak_unretrieved_exception():
    """Fires SPEC-03 without the fix (asyncio logs 'exception was never
    retrieved' for the raced premium task); passes with it. Violated
    property: every asyncio Task's outcome must be retrieved exactly once."""
    import asyncio
    import gc

    from orchestrator.cost_optimization.speculative_gen import SpeculativeGenerator

    class _RacyClient:
        async def call(self, model: str, **kwargs: object):
            if "premium" in model:
                raise RuntimeError("premium blew up before cancellation")
            await asyncio.sleep(0.01)
            return type("Resp", (), {"text": "```python\ndef f():\n    return 1\n```\n" * 20})()

    loop = asyncio.get_event_loop()
    unretrieved = []

    def handler(loop, context):
        if "never retrieved" in context.get("message", ""):
            unretrieved.append(context.get("exception"))

    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(handler)
    try:
        gen = SpeculativeGenerator(client=_RacyClient())
        await gen.speculative_generate(
            prompt="x", cheap_model="cheap-model", premium_model="premium-model", threshold=0.5
        )
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(old_handler)

    assert not unretrieved, f"premium task exception leaked: {unretrieved}"


def _fake_completed_process(cmd, **kw):
    """subprocess.run stub: succeeds for everything except git push."""
    import subprocess

    text = bool(kw.get("text", False))
    if cmd[:2] == ["git", "rev-parse"]:
        return subprocess.CompletedProcess(
            cmd, 0, "abc123\n" if text else b"abc123\n", "" if text else b""
        )
    return subprocess.CompletedProcess(cmd, 0, "" if text else b"", "" if text else b"")


@pytest.mark.asyncio
async def test_push_01_reports_failure_when_remote_push_fails(tmp_path):
    """Fires PUSH-01 without the fix (success=True despite git push
    failing); passes with it (success=False, honest error). Violated
    property: PushResult.success / GitHubMetrics.successful_pushes must
    reflect whether the code actually reached the remote."""
    import subprocess
    from unittest.mock import patch

    from orchestrator.cost_optimization.github_push import GitHubIntegration

    (tmp_path / "app.py").write_text("print('hi')\n")

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "push"]:
            raise subprocess.CalledProcessError(
                1, cmd, output=b"", stderr=b"fatal: no configured push destination\n"
            )
        return _fake_completed_process(cmd, **kw)

    gh = GitHubIntegration(token="fake-token", owner="o", repo="r")
    with patch("subprocess.run", side_effect=fake_run):
        result = await gh.push_results(output_dir=tmp_path, project_id="p1", summary="test")

    assert result.success is False, "push_results() reported success=True although git push failed"


def test_push_02_pr_api_uses_api_host_not_web_host():
    """Fires PUSH-02 without the fix (api_base_url attribute doesn't exist /
    would equal the web host); passes with it. Violated property: REST
    calls must target the API host, matching the convention this codebase
    already uses in orchestrator/vcs/service.py."""
    from orchestrator.cost_optimization.github_push import GitHubIntegration

    gh = GitHubIntegration(token="t", owner="o", repo="r")
    assert gh.api_base_url == "https://api.github.com"


def test_push_03_git_add_uses_separator_before_path():
    """Fires PUSH-03 without the fix (no '--' present); passes with it.
    Violated property: external (generation-authored) filenames reaching a
    CLI argument position must be disambiguated from options."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2]
        / "orchestrator"
        / "cost_optimization"
        / "github_push.py"
    ).read_text()
    assert '"git", "add", "--", str(rel_path)' in src


# --- T1 batch 2 (V3 precision defect audit, docs/audits/v3/T1/batch2) ------
#
# docker_sandbox.py, dependency_context.py -- the 4 fix bodies that survived
# a notification-truncation gap (see docs/audits/v3/T1/batch2 for the other
# 7 findings from this batch, whose fix bodies were lost).


@pytest.mark.asyncio
async def test_f3_avg_execution_time_tracks_unavailable_path(monkeypatch):
    """Fires F3 without the fix; passes with it. Violated property:
    avg_execution_time reflects exactly the calls counted by
    total_executions (error/timeout/unavailable exit paths incremented the
    denominator without contributing to the numerator)."""
    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    sandbox = DockerSandbox()
    monkeypatch.setattr(sandbox, "_check_docker", lambda: _false_coro())

    await sandbox.execute(code_files={"a.py": "pass"}, command="python a.py")

    if sandbox.metrics.avg_execution_time == 0.0:
        pytest.fail(
            "defect still present: total_executions advanced without a "
            "matching avg_execution_time contribution"
        )


async def _false_coro():
    return False


@pytest.mark.asyncio
async def test_f4_log_message_matches_fail_closed_behavior(caplog, monkeypatch):
    """Fires F4 without the fix; passes with it. Violated property: log
    output accurately reflects that there is no subprocess fallback
    (FIX-OPT-001a removed it, but the log message still claimed one)."""
    import sys

    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    sandbox = DockerSandbox()
    monkeypatch.setitem(sys.modules, "docker", None)
    with caplog.at_level("WARNING"):
        await sandbox._check_docker()

    assert not any("falling back to subprocess" in rec.message for rec in caplog.records)


def test_f5_explicit_zero_cpu_quota_is_honored():
    """Fires F5 without the fix; passes with it. Violated property: an
    explicitly-passed 0 is not silently replaced by the class default (0 is
    a valid Docker cpu_quota meaning "unlimited"; `x or DEFAULT` treats 0
    as falsy)."""
    from orchestrator.cost_optimization.docker_sandbox import DockerSandbox

    sandbox = DockerSandbox(cpu_quota=0)
    assert sandbox.cpu_quota == 0


@pytest.mark.asyncio
async def test_f11_avg_context_size_ignores_no_op_calls():
    """Fires F11 without the fix; passes with it. Violated property:
    avg_context_size averages only over calls that actually injected
    context (early-return no-op calls still incremented the denominator)."""
    from orchestrator.cost_optimization.dependency_context import DependencyContextInjector

    injector = DependencyContextInjector()

    class _Dep:
        output = "def f(): pass"
        task_type = "code_generation"

    completed = {"dep1": _Dep()}
    await injector.inject_context(
        task_prompt="p",
        task_type="code_generation",
        completed_tasks=completed,
        dependencies=["dep1"],
    )
    first_avg = injector.metrics.avg_context_size

    await injector.inject_context(
        task_prompt="p",
        task_type="code_generation",
        completed_tasks=completed,
        dependencies=None,
    )
    await injector.inject_context(
        task_prompt="p",
        task_type="code_generation",
        completed_tasks=completed,
        dependencies=["dep1"],
    )
    second_avg = injector.metrics.avg_context_size

    assert second_avg == pytest.approx(
        first_avg
    ), f"no-op call diluted avg_context_size ({first_avg} -> {second_avg})"
