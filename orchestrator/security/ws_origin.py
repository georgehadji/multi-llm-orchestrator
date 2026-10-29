"""WebSocket Origin check (cross-site WebSocket hijacking guard).

Browsers do not apply CORS to WebSocket handshakes, so any page the user visits
can open a socket to a local service unless the server checks ``Origin`` itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def check_ws_origin(headers: Mapping[str, str], allowed: Iterable[str] | None = None) -> bool:
    """True if the handshake may be accepted.

    No ``Origin`` header means a non-browser client, which a web page cannot
    forge, so it is allowed. With ``allowed`` the origin must match exactly;
    without it, only loopback host names pass. Comparing the name (not the
    resolved address) is what stops DNS rebinding.
    """
    origin = headers.get("origin")
    if origin is None:
        return True
    if allowed is not None:
        return origin in set(allowed)
    return urlsplit(origin).hostname in _LOOPBACK_HOSTS
