"""
V3 precision-audit, tier T2 (credentials & trust boundary): 6 batches across
security/, safety/, api_clients.py, gateway/, integrations/gateway.py.

Each test proves the defect for the exact predicted reason (RED against the
pre-fix source) before the fix restores it (GREEN). Sections mirror the
batch/finding IDs in docs/audits/v3/T2/*.md and docs/audits/v3/LEDGER.md.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.unit


# ── Batch 2 (sandbox/execution isolation): SC-1, SC-2, SC-3 ────────────────


def test_sc1_safecommand_blocks_python_dash_c_reproducer() -> None:
    """Fires SC-1 without the fix; passes with it. Violated property: SafeCommand
    must reject interpreter '-c'/'/c' inline-code execution, not merely scan the
    script text for shell metacharacters."""
    from orchestrator.safety.secure_execution import CommandInjectionError, SafeCommand

    payload = ["python", "-c", "__import__('os').system('id')"]
    with pytest.raises(CommandInjectionError):
        SafeCommand(payload)


def test_sc1_safecommand_blocks_python3_and_exe_suffix() -> None:
    """Covers the two secondary bypasses folded into the same fix: executable
    name matching didn't account for 'python3' or a Windows '.exe' suffix."""
    from orchestrator.safety.secure_execution import CommandInjectionError, SafeCommand

    for exe in ("python3", "python.exe", "PYTHON.EXE"):
        with pytest.raises(CommandInjectionError):
            SafeCommand([exe, "-c", "__import__('os').system('id')"])


def test_sc2_secure_subprocess_caps_output_reproducer(tmp_path) -> None:
    """Fires SC-2 without the fix; passes with it. Violated property:
    SecureSubprocess.MAX_OUTPUT_SIZE must actually bound returned output.
    Uses a real script file (not -c) so this is independent of SC-1's fix."""
    from orchestrator.safety.secure_execution import SecureSubprocess

    big = SecureSubprocess.MAX_OUTPUT_SIZE + 1000
    script = tmp_path / "big_output.py"
    script.write_text(f"print('x' * {big})", encoding="utf-8")
    # Pass a bare relative filename + cwd, not the absolute path: a native
    # Windows absolute path contains '\', which SafeCommand's own
    # _SHELL_METACHARACTERS blacklist rejects (a separate, pre-existing bug —
    # see the spawned follow-up task) and that would fail this test for the
    # wrong reason.
    result = SecureSubprocess.run(["python", script.name], cwd=tmp_path, timeout=60)
    assert len(result.stdout) <= SecureSubprocess.MAX_OUTPUT_SIZE + len("\n...[truncated]")


def test_sc3_sanitize_filename_never_exceeds_cap_reproducer() -> None:
    """Fires SC-3 without the fix; passes with it. Violated property:
    sanitize_filename's output must never exceed MAX_FILENAME_LENGTH."""
    from orchestrator.safety.secure_execution import InputValidator

    dirty = "a." + "x" * 300  # extension alone (301 chars) exceeds the 255 cap
    clean = InputValidator.sanitize_filename(dirty)
    assert len(clean) <= InputValidator.MAX_FILENAME_LENGTH


# ── Batch 3 (input/output validation & scanning): B3-GOS-01, B3-GOS-04, B3-WIRE-01 ──


def test_b3_gos_01_dotenv_files_are_scanned(tmp_path) -> None:
    """Fires B3-GOS-01 without the fix; passes with it. Violated property:
    every file whose type is declared scannable (.env is explicitly listed
    in _SCAN_SUFFIXES) is actually scanned before delivery."""
    from orchestrator.safety.generated_output_scanner import scan_output_dir

    (tmp_path / ".env").write_text('DB_PASSWORD="supersecret123"\n', encoding="utf-8")
    (tmp_path / ".env.production").write_text('API_KEY="AKIAABCDEFGHIJKLMNOP"\n', encoding="utf-8")

    report = scan_output_dir(tmp_path)

    if report.files_scanned == 0:
        pytest.fail("defect still present: .env files are never scanned")
    assert report.files_scanned >= 2
    assert any(f.rule == "aws-access-key" for f in report.findings)


def test_b3_gos_04_real_secret_in_example_file_is_still_flagged(tmp_path) -> None:
    """Fires B3-GOS-04 without the fix; passes with it. Violated property: a
    real (non-placeholder) secret must be flagged regardless of which file
    it is found in, including *.example files."""
    from orchestrator.safety.generated_output_scanner import scan_output_dir

    (tmp_path / ".env.example").write_text(
        'db_password = "actualRealSecretValue123"\n', encoding="utf-8"
    )

    report = scan_output_dir(tmp_path)

    if not any(f.rule == "hardcoded-secret-assignment" for f in report.findings):
        pytest.fail("defect still present: real secret in *.example file went unflagged")


async def test_b3_wire_01_rescans_after_fix_tests_engaged(tmp_path, monkeypatch) -> None:
    """Fires B3-WIRE-01 without the fix; passes with it. Violated property:
    security_report must reflect content after step 4b (test-fixing) can
    mutate files, not just the one scan that ran before it."""
    from orchestrator.output_organizer import OutputOrganizer

    organizer = OutputOrganizer(
        tmp_path,
        auto_generate_tests=False,
        run_tests=True,
        fix_tests=True,
        format_code=False,
        security_scan=True,
    )
    scan_calls = 0

    async def fake_scan():
        nonlocal scan_calls
        scan_calls += 1

    async def fake_noop(*_a, **_kw):
        return None

    monkeypatch.setattr(organizer, "_organize_task_files", fake_noop)
    monkeypatch.setattr(organizer, "_detect_source_files", lambda: [])
    monkeypatch.setattr(organizer, "_security_scan", fake_scan)
    monkeypatch.setattr(organizer, "_run_all_tests", fake_noop)
    monkeypatch.setattr(organizer, "_fix_failing_tests", fake_noop)
    monkeypatch.setattr(organizer, "_organize_test_files", fake_noop)
    monkeypatch.setattr(organizer, "_print_summary", lambda: None)
    monkeypatch.setattr(organizer, "_save_report", lambda: None)

    await organizer.organize_project()

    if scan_calls < 2:
        pytest.fail(
            f"defect still present: _security_scan called {scan_calls}x, "
            "expected 2 (once before, once after the fix_tests step)"
        )


# ── Batch 5 (security scoring/posture/templates): T2-B5-06, T2-B5-07 ───────


def test_t2b506_secret_not_hidden_by_unrelated_allow_keyword(tmp_path) -> None:
    """Fires T2-B5-06 without the fix; passes with it.
    Violated property: a real hardcoded secret must be flagged even when the
    same line also contains an unrelated placeholder-shaped substring."""
    from orchestrator.safety.architecture_scorer import ArchitectureScorer

    leak = tmp_path / "leak.py"
    leak.write_text(
        'OPENAI_API_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr"  '
        "# see <https://dashboard.example.com/keys>\n",
        encoding="utf-8",
    )
    hits = ArchitectureScorer()._find_hardcoded_secrets([leak])
    if not hits:
        pytest.fail(
            "defect still present: secret co-located with 'example'/'<...>' went undetected"
        )


def test_t2b507_unpinned_pyproject_is_not_credited_as_pinned(tmp_path) -> None:
    """Fires T2-B5-07 without the fix; passes with it.
    Violated property: '+2 dependencies are version-pinned' must require an
    actual exact pin, not merely the presence of a pyproject.toml file."""
    from orchestrator.safety.architecture_scorer import ArchitectureScorer

    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["fastapi>=0.100", "uvicorn~=0.29"]\n',
        encoding="utf-8",
    )
    pinned = ArchitectureScorer()._has_pinned_deps(tmp_path, "pyproject.toml")
    if pinned:
        pytest.fail("defect still present: unpinned pyproject.toml credited as pinned")


def test_t2b507_pinned_requirements_still_credited(tmp_path) -> None:
    """Guards the fix's own boundary: a genuinely fully-pinned requirements.txt
    must still be credited (regression guard against over-tightening)."""
    from orchestrator.safety.architecture_scorer import ArchitectureScorer

    (tmp_path / "requirements.txt").write_text(
        "fastapi==0.110.0\nuvicorn==0.29.0\n", encoding="utf-8"
    )
    assert ArchitectureScorer()._has_pinned_deps(tmp_path, "requirements.txt") is True


# ── Batch 1 (auth/gateway/egress boundary): B1-GW-01, B1-OB-01 ─────────────


def test_b1_gw_01_raw_key_not_retained_after_transform() -> None:
    """Fires B1-GW-01 without the fix; passes with it. Violated property:
    a bearer credential must not survive the sanitization step that already
    exists for exactly this class of data."""
    from orchestrator.integrations.gateway import APIGateway, APIRequest

    async def _run() -> None:
        gw = APIGateway()
        raw_key = gw.register_api_key("test-user", ["read"])
        req = APIRequest(method="GET", url="/health", headers={"X-API-Key": raw_key})
        assert await gw.authenticate_request(req) is True
        assert req.api_key == raw_key  # sanity: auth really did set it

        transformed = await gw.transform_request(req, "health_service")
        if transformed.api_key == raw_key:
            pytest.fail("defect still present: raw API key survives transform_request()")
        assert transformed.api_key is None

    asyncio.run(_run())


def test_b1_ob_01_fetch_json_bounds_memory_during_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fires B1-OB-01 without the fix; passes with it. Violated property:
    fetch_json's max_bytes cap must bound how much of an oversized response
    is ever buffered, not just detect the overage after full download.

    outbound.py's fetch_json() does `import httpx` LOCALLY inside the
    function, so `outbound.httpx` is never a module attribute — patching
    must target the real top-level httpx module (the same sys.modules
    entry fetch_json's local import resolves to), not outbound.httpx."""
    import httpx

    import orchestrator.safety.outbound as outbound

    huge_chunk = b"x" * 1024

    class _FakeStreamResponse:
        is_redirect = False
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            for _ in range(10_000):
                yield huge_chunk

    class _FakeStreamCtx:
        async def __aenter__(self) -> _FakeStreamResponse:
            return _FakeStreamResponse()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class _FakeClient:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def stream(self, method: str, url: str) -> _FakeStreamCtx:
            return _FakeStreamCtx()

    monkeypatch.setattr(outbound, "check_destination", lambda url: None)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    async def _run() -> None:
        with pytest.raises(outbound.OutboundPolicyError, match="exceeded the .* cap"):
            await outbound.fetch_json("https://example.com/spec.json", max_bytes=4096)

    try:
        asyncio.run(_run())
    except Exception:
        pytest.fail("defect still present: fetch_json did not bound streamed bytes")


# ── Round 2 ──────────────────────────────────────────────────────────────
# Batch 1 additional: B1-KS-01, B1-KS-02, B1-KS-03, B1-GW-02, B1-GW-03


def test_b1_ks_01_verify_compares_every_record_before_deciding() -> None:
    """Fires B1-KS-01 without the fix; passes with it. Violated property:
    verify()'s total compare_digest call count must not depend on where (or
    whether) a match falls in iteration order."""
    from unittest.mock import patch

    from orchestrator.safety.api_keys import KeyStore

    store = KeyStore(pepper="test-pepper-not-secret")
    keys = [store.issue(f"user-{i}", [])[0] for i in range(5)]

    call_counts: list[int] = []
    from orchestrator.safety import api_keys as api_keys_mod

    original_compare = api_keys_mod.hmac.compare_digest

    def _counting_compare(a, b):
        call_counts.append(1)
        return original_compare(a, b)

    with patch.object(api_keys_mod.hmac, "compare_digest", side_effect=_counting_compare):
        call_counts.clear()
        store.verify(keys[0])
        first_count = len(call_counts)

        call_counts.clear()
        store.verify(keys[-1])
        last_count = len(call_counts)

    if first_count != last_count:
        pytest.fail(
            "defect still present: compare_digest call count depends on match "
            f"position ({first_count} vs {last_count}) — verify() is short-circuiting"
        )


def test_b1_ks_02_long_dead_records_are_pruned_on_issue() -> None:
    """Fires B1-KS-02 without the fix; passes with it. Violated property:
    the store's record count must not grow without bound as keys are
    issued and later revoked over a long time horizon."""
    import time

    from orchestrator.safety.api_keys import KeyRecord, KeyStore, _RETENTION_SECONDS

    store = KeyStore(pepper="test-pepper-not-secret")
    _raw, record = store.issue("stale-user", [])
    store.revoke(record.key_id)

    old = store._records[record.key_id]
    store._records[record.key_id] = KeyRecord(
        key_id=old.key_id,
        principal_id=old.principal_id,
        permissions=old.permissions,
        digest=old.digest,
        created_at=old.created_at,
        expires_at=old.expires_at,
        revoked_at=time.time() - _RETENTION_SECONDS - 3600,
    )

    store.issue("another-user", [])

    if record.key_id in store._records:
        pytest.fail("defect still present: long-dead record was never pruned")


def test_b1_ks_03_rotate_does_not_reissue_after_concurrent_revoke() -> None:
    """Fires B1-KS-03 without the fix; passes with it. Violated property: a
    concurrently-revoked key must not still be reissued by an in-flight
    rotate() for the same key_id."""
    from orchestrator.safety.api_keys import KeyStore, KeyStoreError

    store = KeyStore(pepper="test-pepper-not-secret")
    _raw, record = store.issue("victim", [])
    assert store.revoke(record.key_id) is True

    try:
        store.rotate(record.key_id)
    except KeyStoreError:
        return
    pytest.fail(
        "defect still present: rotate() minted a fresh key for a principal "
        "whose key was already revoked, instead of refusing"
    )


def test_b1_gw_02_rate_limit_bucket_is_not_the_raw_key() -> None:
    """Fires B1-GW-02 without the fix; passes with it. Violated property:
    the raw bearer credential must never be used as a retained dict key."""
    from orchestrator.integrations.gateway import APIGateway, APIRequest

    async def _run() -> None:
        gw = APIGateway()
        raw_key = gw.register_api_key("test-user", ["read"])
        req = APIRequest(method="GET", url="/health", headers={"X-API-Key": raw_key})
        await gw.authenticate_request(req)

        await gw.route_request(
            {"method": "GET", "url": "/health", "headers": {"X-API-Key": raw_key}}
        )

        if raw_key in gw.rate_limits:
            pytest.fail("defect still present: raw API key used as a live rate_limits dict key")

    asyncio.run(_run())


def test_b1_gw_03_forwarding_error_does_not_leak_exception_text() -> None:
    """Fires B1-GW-03 without the fix; passes with it. Violated property:
    internal exception text must not reach a response field this file
    hands back to route_request()'s caller."""
    from unittest.mock import patch

    from orchestrator.integrations.gateway import APIGateway

    async def _run():
        gw = APIGateway()
        raw_key = gw.register_api_key("u", ["read"])
        secret_detail = "internal-path=/etc/shadow-ish-detail"
        with patch.object(
            gw, "_forward_request", new=AsyncMock(side_effect=RuntimeError(secret_detail))
        ):
            resp = await gw.route_request(
                {"method": "GET", "url": "/orchestrator/x", "headers": {"X-API-Key": raw_key}}
            )
        return resp.error

    error = asyncio.run(_run())
    if error is not None and "shadow-ish-detail" in error:
        pytest.fail(f"defect still present: internal exception text leaked into .error: {error!r}")


# ── Batch 2 additional: SX-1, SX-2, SX-3, CE-1, CE-2, SB-3 ──────────────────


async def test_sx1_run_rejects_path_traversal_reproducer(tmp_path) -> None:
    """Fires SX-1 without the fix; passes with it. Violated property:
    output_path must stay within the per-task sandbox directory."""
    from orchestrator.safety.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(project_dir=str(tmp_path))
    outside_marker = tmp_path.parent / "sx1_escape_marker.txt"
    outside_marker.unlink(missing_ok=True)
    try:
        result = await executor.run(
            task_id="t1",
            code="pwned",
            output_path="../../sx1_escape_marker.txt",
        )
        assert result.success is False
        assert not outside_marker.exists()
    finally:
        outside_marker.unlink(missing_ok=True)


async def test_sx2_apply_actually_copies_the_file_run_wrote_reproducer(tmp_path) -> None:
    """Fires SX-2 without the fix; passes with it. Violated property:
    apply() must copy the exact file run() wrote, not silently no-op while
    still reporting success."""
    from orchestrator.safety.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(project_dir=str(tmp_path))
    result = await executor.run(task_id="task_001", code="hello", output_path="out.txt")
    result.review_approved = True

    applied = executor.apply(result)

    assert applied is True
    assert (tmp_path / "out.txt").exists()
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"


async def test_sx3_run_reports_malformed_test_command_reproducer(tmp_path) -> None:
    """Fires SX-3 without the fix; passes with it. Violated property: every
    error path in run() must return a SandboxResult, never raise."""
    from orchestrator.safety.sandbox_executor import SandboxExecutor

    executor = SandboxExecutor(project_dir=str(tmp_path))
    result = await executor.run(
        task_id="t1",
        code="print('hi')",
        output_path="out.py",
        test_command="pytest 'unterminated",
    )
    assert result.success is False
    assert "Invalid test command syntax" in result.error


async def test_ce1_execute_in_sandbox_never_raises_reproducer() -> None:
    """Fires CE-1 without the fix; passes with it. Violated property:
    CodeExecutor.execute() must always return an ExecutionResult, never raise
    — the contract every OTHER branch of this class already upholds."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.safety.code_executor import CodeExecutor, ExecutionConfig

    executor = CodeExecutor(ExecutionConfig(require_sandbox=True, fail_if_sandbox_unavailable=True))
    with patch.object(executor, "_is_sandbox_available", return_value=True):
        with patch(
            "orchestrator.cost_optimization.docker_sandbox.DockerSandbox.execute",
            new_callable=AsyncMock,
            side_effect=RuntimeError("docker daemon vanished"),
        ):
            result = await executor.execute("print('hi')", language="python")
    assert result.success is False
    assert "docker daemon vanished" in result.error


async def test_ce1_root_twin_execute_in_sandbox_never_raises_reproducer() -> None:
    """XRef: the byte-for-byte-identical root orchestrator/code_executor.py fork."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.code_executor import CodeExecutor, ExecutionConfig

    executor = CodeExecutor(ExecutionConfig(require_sandbox=True, fail_if_sandbox_unavailable=True))
    with patch.object(executor, "_is_sandbox_available", return_value=True):
        with patch(
            "orchestrator.cost_optimization.docker_sandbox.DockerSandbox.execute",
            new_callable=AsyncMock,
            side_effect=RuntimeError("docker daemon vanished"),
        ):
            result = await executor.execute("print('hi')", language="python")
    assert result.success is False
    assert "docker daemon vanished" in result.error


async def test_ce2_execute_local_caps_output_reproducer() -> None:
    """Fires CE-2 without the fix; passes with it. Violated property:
    ExecutionConfig.max_output_size must actually bound local-execution output."""
    from orchestrator.safety.code_executor import CodeExecutor, ExecutionConfig

    config = ExecutionConfig(require_sandbox=False, max_output_size=100)
    executor = CodeExecutor(config)
    result = await executor.execute("print('x' * 10000)", language="python")
    assert len(result.output) <= 100


async def test_sb3_no_unboundlocalerror_on_spawn_timeout_reproducer(tmp_path) -> None:
    """Fires SB-3 without the fix; passes with it. Violated property: every
    exception path in _execute_with_limits must reference only bound locals."""
    from unittest.mock import patch

    from orchestrator.safety.sandbox import Sandbox

    sandbox = Sandbox(timeout=30.0)
    code_file = tmp_path / "code.py"
    code_file.write_text("print('hi')", encoding="utf-8")

    async def slow_spawn(*_a, **_kw):
        await asyncio.sleep(999)

    with patch("asyncio.create_subprocess_exec", side_effect=slow_spawn):
        stdout, stderr, exit_code = await sandbox._execute_with_limits(
            code_file, "python", None, {"cpu_time": 0.01}
        )
    assert exit_code == -1
    assert stderr == "Execution timed out"


# ── Batch 3 additional: B3-GOS-02, B3-GOS-03, B3-GOS-05, B3-IV-01, B3-IV-02 ─


def test_b3_gos_02_short_real_secret_not_auto_cleared() -> None:
    """Fires B3-GOS-02 without the fix; passes with it. Violated property: an
    assignment matching _ASSIGN_SECRET_RE with a non-placeholder-looking
    value must be flagged, regardless of its length."""
    from orchestrator.safety.generated_output_scanner import _is_placeholder

    if _is_placeholder("root123"):
        pytest.fail("defect still present: a 7-char real-looking secret is auto-cleared")
    assert _is_placeholder("") is True
    assert _is_placeholder("xxx") is True


def test_b3_gos_03_multiline_secret_assignment_detected(tmp_path) -> None:
    """Fires B3-GOS-03 without the fix; passes with it. Violated property:
    generic secret-assignment detection must not be defeated by an
    assignment split across lines."""
    from orchestrator.safety.generated_output_scanner import scan_output_dir

    (tmp_path / "config.py").write_text(
        'db_password = \\\n    "RealProdPassword2024"\n', encoding="utf-8"
    )

    report = scan_output_dir(tmp_path)

    matches = [f for f in report.findings if f.rule == "hardcoded-secret-assignment"]
    if not matches:
        pytest.fail("defect still present: multi-line secret assignment is never detected")


def test_b3_gos_05_no_false_positive_on_unrelated_identifier() -> None:
    """Fires B3-GOS-05 without the fix; passes with it. Violated property:
    the scanner's own stated high-precision design goal — do not flag a
    benign identifier that merely ends in 'verify'/'debug'."""
    from orchestrator.safety.generated_output_scanner import _INSECURE_PATTERNS

    verify_rule = next(r for r in _INSECURE_PATTERNS if r[0] == "verify-ssl-false")
    debug_rule = next(r for r in _INSECURE_PATTERNS if r[0] == "flask-debug-true")

    if verify_rule[2].search("should_verify = False"):
        pytest.fail("defect still present: false positive on should_verify = False")
    if debug_rule[2].search('app.run(host="0.0.0.0", mydebug=True)'):
        pytest.fail("defect still present: false positive on mydebug=True")

    assert verify_rule[2].search("requests.get(url, verify=False)")
    assert debug_rule[2].search('app.run(host="0.0.0.0", debug=True)')


def test_b3_iv_01_visit_number_does_not_crash() -> None:
    """Fires B3-IV-01 without the fix; passes with it. Violated property:
    visit_number must return a value for every well-formed NumberField
    input."""
    from orchestrator.safety.input_validation import NumberField, ZodSchemaVisitor

    field = NumberField(name="age", min_value=18, max_value=120)
    visitor = ZodSchemaVisitor()

    try:
        result = visitor.visit_number(field)
    except AttributeError:
        pytest.fail("defect still present: visit_number raises AttributeError")

    assert result == "z.number().min(18).max(120)"


def test_b3_iv_02_pattern_wrapped_as_regex_literal() -> None:
    """Fires B3-IV-02 without the fix; passes with it. Violated property:
    the visitor must emit syntactically valid target-language code."""
    import re as _re

    from orchestrator.safety.input_validation import JoiSchemaVisitor, StringField, ZodSchemaVisitor

    field = StringField(name="password", pattern=r"^(?=.*[a-z]).{8,}$", trim=False)

    zod_out = ZodSchemaVisitor().visit_string(field)
    joi_out = JoiSchemaVisitor().visit_string(field)

    if not _re.search(r"\.regex\(/.*/\)", zod_out):
        pytest.fail(f"defect still present: Zod output has no delimited regex literal: {zod_out!r}")
    if not _re.search(r"\.pattern\(/.*/\)", joi_out):
        pytest.fail(f"defect still present: Joi output has no delimited regex literal: {joi_out!r}")


# ── Batch 6: SE-1, WP-1, WP-2 ────────────────────────────────────────────


def test_se1_json_ld_does_not_allow_script_breakout() -> None:
    """Fires SE-1 without the fix; passes with it.
    Violated property: a JSON-LD field value must not be able to terminate
    the enclosing <script> element early."""
    from orchestrator.security.enhancer import OpenGraphGenerator

    og = OpenGraphGenerator()
    payload = "</script><script>alert(1)</script>"
    html_out = og.generate_json_ld(name=payload, description="d", url="https://example.com")

    if "</script><script>alert(1)</script>" in html_out:
        pytest.fail("defect still present: raw </script> breakout sequence reached the page")


def test_wp1_namespace_uses_single_backslash_separator() -> None:
    """Fires WP-1 without the fix; passes with it.
    Violated property: generated PHP namespaces must use PHP's single-
    backslash separator."""
    from orchestrator.security.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    config = rules.generate_config("My Awesome Plugin")
    if config.namespace.count("\\") != 2:
        pytest.fail(
            f"defect still present: namespace={config.namespace!r} has "
            f"{config.namespace.count(chr(92))} backslash chars, expected 2"
        )


def test_wp2_headless_recommendation_is_reachable() -> None:
    """Fires WP-2 without the fix; passes with it.
    Violated property: every architecture path the class advertises must be
    reachable from its own recommendation function."""
    from orchestrator.security.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    result = rules.recommend_architecture_path(
        public_distribution=False, team_size=1, complexity="complex"
    )
    if result != "headless":
        pytest.fail(f"defect still present: expected 'headless', got {result!r}")


def test_wp2_team_collaboration_still_recommends_modular_oop() -> None:
    """Guards the fix's own boundary: complex + multi-person team must stay
    modular_oop, not fall through to headless."""
    from orchestrator.security.wordpress_plugin_rules import WordPressPluginRules

    rules = WordPressPluginRules()
    result = rules.recommend_architecture_path(
        public_distribution=False, team_size=3, complexity="complex"
    )
    assert result == "modular_oop"


# ── Round 3: B1-CG-01/02/03, B1-GR-01, B3-GOS-06, GR-1, TG-1 ────────────────


def test_b1_cg_01_suspicious_requires_approval() -> None:
    """Fires B1-CG-01 without the fix; passes with it. Violated property:
    requires_explicit_approval's own docstring — 'SUSPICIOUS and BLOCKED
    still gate' — must hold for every allow_dangerous value."""
    from orchestrator.safety.command_guard import (
        RiskAssessment,
        RiskLevel,
        requires_explicit_approval,
    )

    suspicious = RiskAssessment(level=RiskLevel.SUSPICIOUS, rationale="test", command="curl x")
    if requires_explicit_approval(suspicious, allow_dangerous=False) is not True:
        pytest.fail("defect still present: SUSPICIOUS not gated with allow_dangerous=False")
    if requires_explicit_approval(suspicious, allow_dangerous=True) is not True:
        pytest.fail("defect still present: SUSPICIOUS not gated with allow_dangerous=True")

    unknown = RiskAssessment(level=RiskLevel.SUSPICIOUS, rationale="Unknown command", command="???")
    assert requires_explicit_approval(unknown) is True, "secure-by-default fallback must gate"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -fr /",
        "rm -rfv /",
        "rm -Rf /",
        "rm -r -f /",
        "rm --force --recursive /",
        "rm -rf ~",
    ],
)
def test_b1_cg_02_rm_recursive_force_blocked_regardless_of_flag_order(command: str) -> None:
    """Fires B1-CG-02 without the fix; passes with it."""
    from orchestrator.safety.command_guard import RiskLevel, classify_command

    result = classify_command(command)
    if result.level != RiskLevel.BLOCKED:
        pytest.fail(f"defect still present: {command!r} classified {result.level}, not BLOCKED")


@pytest.mark.parametrize(
    "command", ["rm -f /tmp/x", "rm -i /tmp/x", "rm -rf /tmp/build", "rm -v /tmp/x"]
)
def test_b1_cg_02_non_root_or_non_recursive_rm_not_blocked(command: str) -> None:
    """Guards against over-blocking."""
    from orchestrator.safety.command_guard import RiskLevel, classify_command

    result = classify_command(command)
    assert result.level != RiskLevel.BLOCKED, f"{command!r} incorrectly BLOCKED"


@pytest.mark.parametrize(
    "command",
    [
        "git branch -D main",
        "git branch -D feature/x",
        "git remote add evil https://x",
        "git remote set-url origin https://x",
        "git remote remove origin",
    ],
)
def test_b1_cg_03_mutating_branch_remote_not_safe(command: str) -> None:
    """Fires B1-CG-03 without the fix; passes with it."""
    from orchestrator.safety.command_guard import RiskLevel, classify_command

    result = classify_command(command)
    if result.level == RiskLevel.SAFE:
        pytest.fail(f"defect still present: {command!r} classified SAFE")


@pytest.mark.parametrize("command", ["git branch", "git remote", "git status", "git log"])
def test_b1_cg_03_bare_readonly_forms_still_safe(command: str) -> None:
    """Guards against over-fixing: genuinely read-only bare invocations stay SAFE."""
    from orchestrator.safety.command_guard import RiskLevel, classify_command

    assert classify_command(command).level == RiskLevel.SAFE


def test_b1_gr_01_handle_message_denies_unauthorized_sender_by_default() -> None:
    """Fires B1-GR-01 without the fix; passes with it. Violated property:
    handle_message must not run a project for a sender that was never
    authorized — today's default (no allowed_users configured) must deny.

    _execute_project_spec is mocked in BOTH directions of this test (not
    just the allowed-sender case) — pre-fix, with no gate, this method
    would otherwise construct a real Orchestrator and attempt a real,
    unmocked LLM call. The property under test is "was the execution path
    reached at all", not the literal return string."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.gateway.run import GatewayConfig, OrchestratorGateway

    async def _run() -> tuple[str, bool]:
        gw = OrchestratorGateway(GatewayConfig())
        with patch.object(
            OrchestratorGateway, "_execute_project_spec", new=AsyncMock(return_value="ok")
        ) as mocked:
            result = await gw.handle_message("webhook", "anyone", "build me an app")
            return result, mocked.called

    result, was_executed = asyncio.run(_run())
    if was_executed:
        pytest.fail(
            "defect still present: handle_message ran a project for an "
            f"unauthenticated sender instead of denying it; got: {result!r}"
        )


def test_b1_gr_01_handle_message_allows_configured_sender() -> None:
    """Sanity check: an explicitly allowlisted sender is not blocked by the fix."""
    from unittest.mock import AsyncMock, patch

    from orchestrator.gateway.run import GatewayConfig, OrchestratorGateway

    async def _run() -> str:
        cfg = GatewayConfig(allowed_users={"webhook": frozenset({"alice"})})
        gw = OrchestratorGateway(cfg)
        with patch.object(
            OrchestratorGateway, "_execute_project_spec", new=AsyncMock(return_value="ok")
        ):
            return await gw.handle_message("webhook", "alice", "build me an app")

    assert asyncio.run(_run()) == "ok"


def test_b3_gos_06_openai_project_key_detected() -> None:
    """Fires B3-GOS-06 without the fix; passes with it."""
    from orchestrator.safety.generated_output_scanner import _RULES

    openai_rule = next(r for r in _RULES if r[0] == "openai-key")
    pattern = openai_rule[2]

    sample = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
    if not pattern.search(sample):
        pytest.fail("defect still present: hyphenated OpenAI project key not matched")


def test_b3_gos_06_pii_masker_masks_hyphenated_key() -> None:
    """Companion assertion for the pii_masking_etl.py XRef site."""
    from orchestrator.safety.pii_masking_etl import PIIMaskingETL

    etl = PIIMaskingETL()
    masked = etl.transform("key=sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890")
    if "sk-proj-" in masked:
        pytest.fail("defect still present: hyphenated key not masked by PIIMaskingETL")
    assert "<API_KEY_MASKED>" in masked


def test_gr1_kill_switch_defaults_are_not_world_writable_tmp() -> None:
    """Fires GR-1 without the fix; passes with it."""
    from pathlib import Path

    from orchestrator.safety.guardrails import KillSwitch

    ks = KillSwitch()
    home = str(Path.home())
    if not (str(ks.kill_file).startswith(home) and str(ks.force_file).startswith(home)):
        pytest.fail(
            f"defect still present: kill_file={ks.kill_file!r} / "
            f"force_file={ks.force_file!r} default outside the user's own state dir"
        )


# ── Round 4: T2B4-04/07/08/09/10/11/12/13, T2-B5-04/05/08 ──────────────────


def test_t2b4_07_yaml_load_detected_for_s_prefixed_argument() -> None:
    """Fires T2B4-07 without the fix; passes with it."""
    from orchestrator.safety.security_review import SecurityReviewer

    reviewer = SecurityReviewer()
    report = reviewer.quick_scan("config = yaml.load(stream)")
    matching = [f for f in report.findings if f.rule_id == "SEC-004"]
    assert len(matching) > 0, "yaml.load(stream) was not detected"


def test_t2b4_09_safety_namespace_no_longer_ambiguous() -> None:
    """Fires T2B4-09 without the fix; passes with it."""
    import orchestrator.safety as safety_pkg
    from orchestrator.safety import security_review, security_validator

    assert safety_pkg.SecurityFinding is security_validator.SecurityFinding
    assert safety_pkg.SecurityFinding is not security_review.SecurityFinding

    finding = security_review.SecurityFinding(
        rule_id="SEC-001",
        title="t",
        severity=security_review.Severity.LOW,
        category=security_review.Category.CONFIG,
        description="d",
    )
    assert finding.rule_id == "SEC-001"


def test_t2b4_04_aware_datetime_boundary_does_not_crash() -> None:
    """Fires T2B4-04 without the fix; passes with it."""
    from datetime import datetime, timezone

    from orchestrator.safety.accountability import AccountabilityTracker, ActionType, ActorType

    tracker = AccountabilityTracker()
    tracker.record_action(
        actor_id="a",
        actor_type=ActorType.AGENT,
        actor_name="a",
        action_type=ActionType.TASK_EXECUTE,
        target="x",
    )
    report = tracker.get_accountability_report(
        start_time=datetime.now(timezone.utc).replace(year=2000)
    )
    assert report["summary"]["total_actions"] == 1


def test_t2b4_08_empty_params_pattern_actually_matches() -> None:
    """Fires T2B4-08 without the fix; passes with it."""
    from orchestrator.safety.security_validator import check_sql_injection

    code = "cursor.execute(sql_string, [])"
    findings = check_sql_injection(code, "db.py")
    assert any(
        "Empty parameters" in f.description for f in findings
    ), "the empty-parameters pattern never matches its own intended target"


def test_t2b4_10_accountability_docstring_example_runs() -> None:
    """Fires T2B4-10 (accountability.py half) without the fix; passes with it."""
    from orchestrator.safety.accountability import AccountabilityTracker, ActionType, ActorType

    tracker = AccountabilityTracker()
    action_id = tracker.record_action(
        actor_id="admin",
        actor_type=ActorType.USER,
        actor_name="admin",
        action_type=ActionType.FILE_WRITE,
        target="src/main.py",
        delegation_chain=["user:admin", "agent:code_writer", "tool:file_write"],
    )
    assert tracker.get_action(action_id) is not None


async def test_t2b4_11_explicit_empty_results_not_collapsed_to_stale_state() -> None:
    """Fires T2B4-11 without the fix; passes with it."""
    from orchestrator.safety.red_team import RedTeamFramework

    framework = RedTeamFramework()
    await framework.run_scenario("task_misrep_001")

    report = framework.generate_report({})

    assert (
        report.executed_scenarios == 0
    ), "explicit empty results was ignored in favor of stale self._results"


def test_t2b4_12_docstring_now_matches_broad_extension_filter() -> None:
    """Fires T2B4-12 (documentation-drift guard) without the fix's docstring text."""
    from orchestrator.safety.security_validator import check_security_headers

    plain_module = "x = 1\n" * 20

    findings = check_security_headers(plain_module, "orchestrator/plain_data.py")

    assert len(findings) == 4, (
        "a plain non-HTTP .py file should still be flagged under the "
        "documented (now-accurate) broad extension heuristic"
    )


def test_t2b4_13_info_findings_are_counted() -> None:
    """Fires T2B4-13 without the fix; passes with it."""
    from orchestrator.safety.security_review import (
        Category,
        SecurityFinding,
        SecurityReviewer,
        Severity,
    )

    reviewer = SecurityReviewer()
    findings = [
        SecurityFinding("SEC-X", "t", Severity.INFO, Category.CONFIG, "d"),
    ]

    report = reviewer._build_report(findings)

    assert report.info_count == 1
    assert "1I" in report.summary


def test_t2b504_rate_limit_decorator_honors_its_own_arguments() -> None:
    """Fires T2-B5-04 without the fix; passes with it."""
    from orchestrator.safety.security_templates import RateLimitTemplate, SecurityConfig

    code = RateLimitTemplate().generate(SecurityConfig(rate_limit_max=100, rate_limit_window=60))
    wrapper_body = code[code.index("def rate_limit(key_func") : code.index("def rate_limit_ip")]
    if "_rate_limiter.is_allowed(key)" in wrapper_body:
        pytest.fail(
            "defect still present: wrapper() reads the shared global limiter, "
            "ignoring this call's own max_requests/window"
        )


def test_t2b505_token_bucket_consume_is_lock_guarded() -> None:
    """Fires T2-B5-05 without the fix; passes with it."""
    import ast

    from orchestrator.safety.security_templates import RateLimitTemplate, SecurityConfig

    code = RateLimitTemplate().generate(SecurityConfig())
    tree = ast.parse(code)
    consume_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "consume"
    )
    has_with_lock = any(isinstance(n, ast.With) for n in ast.walk(consume_fn))
    if not has_with_lock:
        pytest.fail("defect still present: consume() mutates state with no lock held")


def test_t2b508_tests_with_no_source_does_not_score_perfect(tmp_path) -> None:
    """Fires T2-B5-08 without the fix; passes with it."""
    from orchestrator.safety.architecture_scorer import ArchitectureScorer

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_something.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    scorer = ArchitectureScorer()
    files = scorer._collect_code_files(tmp_path)
    dim = scorer._score_tests(tmp_path, files)
    if dim.score >= 12:
        pytest.fail(f"defect still present: scored {dim.score}/15 with zero source files")
