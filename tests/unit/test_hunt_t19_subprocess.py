"""
Hunt T19 — subprocess/exec argument construction.

`nexus_search/server_manager.py` built its docker-compose invocation as an
f-string and handed it to `asyncio.create_subprocess_shell`:

    cmd = f"{self._docker_compose_cmd} -f {self.compose_file} up -d"

`_docker_compose_cmd` is hardcoded, but `compose_file` is the constructor's
caller-supplied parameter. Under a shell, a path containing merely a *space*
splits into two arguments and breaks the command; a path containing `;` runs
whatever follows as a separate command. Both were demonstrated empirically:
`shlex.split` shreds such a path, and a RED run against the unmodified
production code made docker itself report the argument it received as
`"compose projects/docker-compose.yml"` while the injected `touch` created its
marker file.

`scripts/utils/push_to_github.py` had the same shape with the current branch
name, and `git check-ref-format` accepts `;`, `$()`, backticks, `&&` and `|`
in branch names.

These tests pin the property that matters: the caller-supplied value reaches
the OS as exactly one argv element, never as shell text.
"""

from __future__ import annotations

import asyncio
import shlex
from unittest.mock import patch

import pytest

from orchestrator.nexus_search.server_manager import NexusServerManager

pytestmark = pytest.mark.unit

# A path hostile in two independent ways: a space (breaks any shell invocation
# by accident) and a `;` (injects on purpose). The injected command is inert on
# purpose — if this fix is ever reverted, the RED run of these tests reaches a
# real shell, and it should not be able to touch the filesystem to prove it.
HOSTILE_PATH = "/tmp/my projects/docker-compose.yml; echo t19"


class _FakeProc:
    returncode = 0

    async def communicate(self):
        return (b"", b"")


def _manager() -> NexusServerManager:
    """A manager with only the attributes start()/stop() read before exec."""
    m = NexusServerManager.__new__(NexusServerManager)
    m._docker_compose_cmd = "docker compose"
    m._server_started = False
    m.compose_file = HOSTILE_PATH
    m.port = 8080
    m.health_check_interval = 0
    m._health_check_task = None
    m._shutdown_event = asyncio.Event()
    m._restart_count = 0
    return m


async def _capture(coro_factory) -> list[str]:
    """Run the manager method with the exec syscall stubbed; return argv."""
    captured: list[str] = []

    async def fake_exec(*argv, **_kw):
        captured.extend(argv)
        return _FakeProc()

    async def always_healthy(_self):
        return True

    with (
        patch("asyncio.create_subprocess_exec", fake_exec),
        patch.object(NexusServerManager, "_wait_for_healthy", always_healthy),
    ):
        await coro_factory()
    return captured


@pytest.mark.asyncio
async def test_start_passes_the_compose_path_as_one_argument():
    m = _manager()
    argv = await _capture(m.start)

    assert argv, "start() never reached the subprocess call"
    assert HOSTILE_PATH in argv, (
        "the compose path must arrive as a single argv element; got " f"{argv!r}"
    )
    assert argv[:3] == ["docker", "compose", "-f"]
    assert argv[-2:] == ["up", "-d"]


@pytest.mark.asyncio
async def test_stop_passes_the_compose_path_as_one_argument():
    m = _manager()
    m._server_started = True
    argv = await _capture(m.stop)

    assert argv, "stop() never reached the subprocess call"
    assert HOSTILE_PATH in argv, f"the compose path must survive intact; got {argv!r}"
    assert argv[-1] == "down"


def test_the_pre_fix_shell_string_would_have_shredded_that_path():
    """Documents why argv matters — no shell is spawned here."""
    pre_fix = f"docker compose -f {HOSTILE_PATH} up -d"
    tokens = shlex.split(pre_fix)

    assert HOSTILE_PATH not in tokens, "sanity: the shell must not keep it whole"
    assert "echo" in tokens, "the injected command becomes its own shell word"


@pytest.mark.asyncio
async def test_server_manager_uses_no_shell_interpreting_api():
    """create_subprocess_shell must not reappear in this module."""
    import inspect

    from orchestrator.nexus_search import server_manager

    source = inspect.getsource(server_manager)
    assert "create_subprocess_shell" not in source
    assert "shell=True" not in source


def test_push_script_pushes_the_branch_as_argv():
    """A branch name may legally contain `;`, `$()`, backticks, `&&`, `|`."""
    from pathlib import Path

    source = Path("scripts/utils/push_to_github.py").read_text(encoding="utf-8")
    assert 'run(f"git push origin {branch}")' not in source
    assert 'run(["git", "push", "origin", branch])' in source
