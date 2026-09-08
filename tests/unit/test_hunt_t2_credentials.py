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
