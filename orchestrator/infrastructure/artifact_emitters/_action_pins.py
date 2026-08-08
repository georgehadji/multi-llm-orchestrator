"""GitHub Actions SHA-pin resolution for the CI/CD emitter (Phase 7, P-5).

Resolution shells out to ``git ls-remote`` against the public GitHub mirror
of an action. A failure here — no git, no network, unknown ref — must
degrade to an explicit unresolved result; never a fabricated 40-hex string
wearing the shape of a real commit.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionPinResult:
    sha: str | None
    ok: bool
    log: str


def _parse_ls_remote(stdout: str, ref: str) -> str | None:
    direct = f"refs/tags/{ref}"
    peeled = f"refs/tags/{ref}^{{}}"
    shas_by_ref = {}
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sha, refname = parts
        shas_by_ref[refname] = sha
    return shas_by_ref.get(peeled) or shas_by_ref.get(direct)


def resolve_action_sha(owner_repo: str, ref: str, timeout_s: float = 15.0) -> ActionPinResult:
    if shutil.which("git") is None:
        return ActionPinResult(None, False, "git not available on PATH")
    url = f"https://github.com/{owner_repo}.git"
    try:
        proc = subprocess.run(
            ["git", "ls-remote", url, f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ActionPinResult(None, False, str(exc))
    if proc.returncode != 0:
        return ActionPinResult(None, False, (proc.stderr or "git ls-remote failed").strip()[:500])
    sha = _parse_ls_remote(proc.stdout, ref)
    if not sha:
        return ActionPinResult(None, False, f"ref not found: {owner_repo}@{ref}")
    return ActionPinResult(sha, True, "")


__all__ = ["ActionPinResult", "resolve_action_sha"]
