"""One authentication path for the IDE: REST and WebSocket both use it (T7).

Every route took a caller-supplied ``session_id`` and acted on it. There was no
authentication and no notion of who owned a session, so anyone who could reach
the port could read another session's chat history and generated files, edit
them, retarget its tasks or delete it.

Two rules here, and they are deliberately in one module so the REST handlers
and the WebSocket handshake cannot drift into two different policies:

* who is calling — `authenticate`
* whether they may touch this session — `require_owner`

Authentication is off by default because the server binds loopback (SEC-001a)
and requiring a token to run your own IDE locally is friction people route
around — which is precisely how the insecure standalone server came to be the
shipped launcher. Off does not mean absent: requests get `LOCAL_PRINCIPAL`, so
ownership is still evaluated on every call and the path is exercised in dev.
Binding a non-loopback host already demands authentication before the socket
opens, in `security.validate_bind_target`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from orchestrator.domain.security import Permission, Principal
from orchestrator.safety.api_keys import get_key_store

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "AUTH_REQUIRED_ENV",
    "LOCAL_PRINCIPAL",
    "AuthError",
    "authenticate",
    "auth_required",
    "bearer_token",
    "owns",
    "require_owner",
]

AUTH_REQUIRED_ENV = "ORCHESTRATOR_IDE_AUTH_REQUIRED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The principal every request gets when authentication is off. A real id, so
#: sessions created in dev have a real owner rather than an empty one.
LOCAL_PRINCIPAL = Principal(
    id="local",
    key_id="local",
    permissions=frozenset(Permission),
)


class AuthError(Exception):
    """Refused. Carries a stable code, never an exception message.

    Handlers turn this into the HTTP status; the ``code`` is what reaches the
    client, so a caller cannot mine internal state from error text.
    """

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def auth_required() -> bool:
    return os.getenv(AUTH_REQUIRED_ENV, "").strip().lower() in _TRUTHY


def bearer_token(headers: Mapping[str, str]) -> str | None:
    """Pull a bearer token out of an Authorization header, if present."""
    raw = headers.get("Authorization") or headers.get("authorization")
    if not raw:
        return None
    prefix, _, token = raw.partition(" ")
    if prefix.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def authenticate(headers: Mapping[str, str]) -> Principal:
    """Identify the caller, or raise `AuthError`.

    With auth off every caller is `LOCAL_PRINCIPAL`. With auth on, a valid key
    is mandatory — expiry and revocation are enforced by the key store (T8).
    """
    token = bearer_token(headers)

    if not auth_required():
        # A valid token still identifies you, so a session created with one
        # stays yours if authentication is switched on later.
        if token:
            principal = get_key_store().verify(token)
            if principal is not None:
                return principal
        return LOCAL_PRINCIPAL

    if not token:
        raise AuthError("unauthorized", 401)

    principal = get_key_store().verify(token)
    if principal is None:
        raise AuthError("unauthorized", 401)
    return principal


def owns(principal: Principal, owner_id: str | None) -> bool:
    """Whether `principal` may act on something owned by `owner_id`."""
    if principal.has(Permission.ADMIN):
        return True
    if not owner_id:
        # Sessions predating ownership. Only the local principal may adopt
        # them, so an authenticated deployment does not inherit a free-for-all.
        return principal.id == LOCAL_PRINCIPAL.id
    return principal.id == owner_id


def require_owner(principal: Principal, owner_id: str | None) -> None:
    """Raise `AuthError("not_found", 404)` unless `principal` owns this.

    404 rather than 403 on purpose: 403 confirms the session exists, which
    turns any session-scoped route into an existence oracle. Callers already
    answer 404 for a session that is genuinely absent, so the two cases are
    indistinguishable from outside.
    """
    if not owns(principal, owner_id):
        raise AuthError("not_found", 404)
