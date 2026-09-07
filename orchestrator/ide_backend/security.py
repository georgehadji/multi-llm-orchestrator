"""Bind and CORS policy for the IDE service (SEC-001).

The IDE exposes session creation, file access and orchestrator control. Before
this module it bound to ``0.0.0.0`` by default and mounted CORS with
``allow_origins=["*"]`` and ``allow_credentials=True``, so anything that could
reach the port — or any origin a browser visited — had an unauthenticated
control plane.

The rules here are deliberately small and fail closed:

* loopback is the default and needs no ceremony;
* any other bind target requires an explicit ``ORCHESTRATOR_IDE_ALLOW_REMOTE``
  opt-in *and* authentication;
* origins are an explicit list, never ``*`` while credentials are enabled.
"""

from __future__ import annotations

import ipaddress
import os

__all__ = [
    "InsecureBindError",
    "allowed_origins",
    "is_loopback",
    "remote_allowed",
    "validate_bind_target",
]

#: Origins the local dev setup actually uses (Vite dev server + the IDE port).
_DEFAULT_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://localhost:8765",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8765",
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

ALLOW_REMOTE_ENV = "ORCHESTRATOR_IDE_ALLOW_REMOTE"
ALLOWED_ORIGINS_ENV = "ORCHESTRATOR_IDE_ALLOWED_ORIGINS"


class InsecureBindError(RuntimeError):
    """The requested configuration would expose the IDE unsafely.

    Raised at startup rather than logged as a warning: a warning on a service
    that is already listening does not un-expose it.
    """


def is_loopback(host: str) -> bool:
    """True if `host` can only be reached from this machine.

    ``localhost`` is accepted by name; everything else must parse as a loopback
    IP. A hostname that is not ``localhost`` is treated as non-loopback because
    resolving it here would be a DNS call whose answer can change.
    """
    if not host:
        return False
    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def remote_allowed() -> bool:
    """True if the operator explicitly opted in to non-loopback binding."""
    return os.getenv(ALLOW_REMOTE_ENV, "").strip().lower() in _TRUTHY


def allowed_origins() -> list[str]:
    """The CORS origin allowlist.

    Defaults to the loopback dev origins. A literal ``*`` is refused outright
    because the app sends credentials, and wildcard-plus-credentials is the
    combination that makes every visited site a same-origin client.
    """
    raw = os.getenv(ALLOWED_ORIGINS_ENV, "").strip()
    if not raw:
        return list(_DEFAULT_ORIGINS)

    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise InsecureBindError(
            f"wildcard origin '*' in {ALLOWED_ORIGINS_ENV} cannot be combined with "
            "credentialed CORS; list the exact origins instead"
        )
    return origins or list(_DEFAULT_ORIGINS)


def validate_bind_target(host: str, *, allow_remote: bool, auth_required: bool) -> None:
    """Raise `InsecureBindError` if binding `host` would expose the IDE unsafely.

    Args:
        host: the interface the server is about to bind.
        allow_remote: operator opt-in, normally `remote_allowed()`.
        auth_required: whether the service authenticates requests.
    """
    if is_loopback(host):
        return

    if not allow_remote:
        raise InsecureBindError(
            f"refusing to bind the IDE server to non-loopback host {host!r}. "
            f"Set {ALLOW_REMOTE_ENV}=true to opt in explicitly."
        )

    if not auth_required:
        raise InsecureBindError(
            f"refusing to bind the IDE server to {host!r} without authentication. "
            "A remotely reachable IDE must authenticate every request (SEC-001)."
        )
