"""Lock-file resolution for the Docker emitter (Phase 7, P-4).

Tries ``uv lock`` first, falls back to ``pip-compile --generate-hashes``.
Never invents a lock file when neither tool is available — ``ok=False``
with an explanatory log is preferred over silently skipping.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

_REQUIREMENTS_LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)")


@dataclass(frozen=True)
class LockFileResult:
    filename: str | None
    tool: str
    ok: bool
    log: str
    components: tuple[tuple[str, str], ...] = ()


def _components_from_uv_lock(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ()
    packages = data.get("package", [])
    return tuple(
        (pkg["name"], pkg["version"])
        for pkg in packages
        if isinstance(pkg, dict) and "name" in pkg and "version" in pkg
    )


def _components_from_requirements(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    components = []
    for line in text.splitlines():
        m = _REQUIREMENTS_LINE_RE.match(line.strip())
        if m:
            components.append((m.group(1), m.group(2)))
    return tuple(components)


def resolve_lockfile(root: Path, timeout_s: float = 60.0) -> LockFileResult:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return LockFileResult(None, "none", False, "no pyproject.toml to lock")

    if shutil.which("uv") is not None:
        try:
            proc = subprocess.run(
                ["uv", "lock"], cwd=str(root), capture_output=True, text=True, timeout=timeout_s
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LockFileResult(None, "uv", False, str(exc))
        lock_path = root / "uv.lock"
        if proc.returncode == 0 and lock_path.is_file():
            return LockFileResult("uv.lock", "uv", True, "", _components_from_uv_lock(lock_path))
        # uv present but resolution failed (e.g. no network) — fall through to pip-compile.

    if shutil.which("pip-compile") is not None:
        try:
            proc = subprocess.run(
                [
                    "pip-compile",
                    "--generate-hashes",
                    "--quiet",
                    "pyproject.toml",
                    "-o",
                    "requirements.txt",
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LockFileResult(None, "pip-compile", False, str(exc))
        req_path = root / "requirements.txt"
        if proc.returncode == 0 and req_path.is_file():
            return LockFileResult(
                "requirements.txt",
                "pip-compile",
                True,
                "",
                _components_from_requirements(req_path),
            )
        return LockFileResult(
            None, "pip-compile", False, (proc.stderr or "pip-compile failed").strip()[:500]
        )

    return LockFileResult(None, "none", False, "neither uv nor pip-compile available on PATH")


__all__ = ["LockFileResult", "resolve_lockfile"]
