"""Base-image digest resolution for the Docker emitter (Phase 7, P-4).

Resolution requires talking to a registry (``docker manifest inspect``).
A failure here — no Docker CLI, no network, unparseable output — must
degrade to an explicit unresolved result, never to a silently-invented
digest wearing a mutable tag as if it were pinned.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DigestResult:
    digest: str | None
    ok: bool
    log: str


def _extract_digest(data) -> str | None:
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        descriptor = data.get("Descriptor")
        if isinstance(descriptor, dict):
            digest = descriptor.get("digest")
            if isinstance(digest, str):
                return digest
    return None


def resolve_digest(image_ref: str, timeout_s: float = 15.0) -> DigestResult:
    if shutil.which("docker") is None:
        return DigestResult(None, False, "docker CLI not available")
    try:
        proc = subprocess.run(
            ["docker", "manifest", "inspect", image_ref, "--verbose"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DigestResult(None, False, str(exc))
    if proc.returncode != 0:
        return DigestResult(None, False, (proc.stderr or "manifest inspect failed").strip()[:500])
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return DigestResult(None, False, f"unparseable manifest output: {exc}")
    digest = _extract_digest(data)
    if not digest:
        return DigestResult(None, False, "no digest field in manifest output")
    return DigestResult(digest, True, "")


__all__ = ["DigestResult", "resolve_digest"]
