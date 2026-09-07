"""Authentication and authorization primitives (SEC-006).

Pure data: no I/O, no framework imports, no persistence. The API server
authenticates once and then evaluates a typed `Permission` against the
`Principal` it recovered.

Background: API keys were registered with a ``permissions`` list, but
verification only answered "does this key exist?" and returned a bool. Every
valid key therefore had every capability, including execution and supervisor
access, whatever it was registered with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "PERMISSION_IMPLIES",
    "Permission",
    "Principal",
    "parse_permissions",
    "principal_has_permission",
]


class Permission(str, Enum):
    """A capability a principal may hold.

    `str` mixin so stored permission lists (plain JSON strings) compare
    directly against members without conversion at every call site.
    """

    READ = "read"
    EXECUTE = "execute"
    PROJECT_CANCEL = "project:cancel"
    SUPERVISOR_READ = "supervisor:read"
    SUPERVISOR_EXECUTE = "supervisor:execute"
    KEY_REGISTER = "key:register"
    ADMIN = "admin"


#: Permissions each granted permission also confers.
#:
#: Keys registered before this change defaulted to ``["read", "execute"]``, so
#: `execute` implies `project:cancel` — cancelling a run you started is part of
#: running it, and denying it would break existing callers for no security
#: gain. `supervisor:*` is deliberately NOT implied by `read` or `execute`:
#: supervisor routes expose cross-project state and directive control, which is
#: exactly the escalation this finding is about. Existing keys must be
#: re-registered with `supervisor:read` to keep that access.
PERMISSION_IMPLIES: dict[Permission, frozenset[Permission]] = {
    Permission.ADMIN: frozenset(Permission) - {Permission.ADMIN},
    Permission.EXECUTE: frozenset({Permission.PROJECT_CANCEL}),
    Permission.SUPERVISOR_EXECUTE: frozenset({Permission.SUPERVISOR_READ}),
}


@dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    Immutable so a handler cannot widen its own authority mid-request.
    """

    id: str
    key_id: str
    permissions: frozenset[Permission] = field(default_factory=frozenset)

    def has(self, permission: Permission) -> bool:
        """True if this principal holds `permission`, directly or by implication."""
        return principal_has_permission(self.permissions, permission)


def principal_has_permission(
    held: frozenset[Permission] | set[Permission] | list[Permission],
    required: Permission,
) -> bool:
    """Evaluate `required` against the `held` permission set.

    Fails closed: an empty or unrecognised set grants nothing.
    """
    held_set = set(held)
    if required in held_set:
        return True
    return any(required in PERMISSION_IMPLIES.get(granted, frozenset()) for granted in held_set)


def parse_permissions(raw: object) -> frozenset[Permission]:
    """Convert a stored permission list into `Permission` members.

    Unknown strings are dropped rather than accepted — an unrecognised
    permission must never widen access (default-deny).
    """
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    known = {member.value: member for member in Permission}
    return frozenset(known[item] for item in raw if isinstance(item, str) and item in known)
