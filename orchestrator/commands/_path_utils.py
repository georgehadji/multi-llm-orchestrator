"""Shared path validation helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path

_BLOCKED_PREFIXES = (
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
    Path("/etc"),
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/root"),
    Path("/sys"),
    Path("/dev"),
    Path("/proc"),
)


def resolve_allowed_path(path_str: str, outputs_root: Path | None = None) -> Path:
    """Resolve and validate a user-supplied CLI path.

    Paths inside ``outputs_root`` are accepted. Paths outside must not target
    blocked system directories.

    Raises:
        ValueError: if the path is empty or targets a system directory.
    """
    if not path_str:
        raise ValueError("Path must not be empty")

    raw = Path(path_str).resolve()

    if outputs_root is not None:
        try:
            raw.relative_to(outputs_root)
            return raw
        except ValueError:
            pass

    for blocked in _BLOCKED_PREFIXES:
        try:
            raw.relative_to(blocked.resolve())
        except ValueError:
            continue
        raise ValueError(
            f"Path {path_str!r} targets a system directory. "
            "Use a path inside the project outputs/ folder."
        )

    return raw
