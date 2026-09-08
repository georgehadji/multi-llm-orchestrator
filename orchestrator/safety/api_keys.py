"""API key lifecycle: issue, verify, expire, revoke, rotate (T8).

`api_server.py` stored `sha256(raw_key)` in a dict. That is three problems:

* A bare SHA-256 of a token is a fast offline target. API keys are
  high-entropy, so this is far less dire than a password hash, but the store
  should not be self-sufficient — a leaked file alone must not yield working
  keys. Hence HMAC under a server-side pepper that lives outside the file.
* No expiry and no revocation: a leaked key was valid forever.
* A dict dies with the process, so every restart invalidated every key and
  nobody could tell that apart from a compromise.

Deliberately not bcrypt/argon2: verification happens on every request, and a
work factor there is a self-inflicted denial of service. The security comes
from 256 bits of token entropy plus a pepper the attacker does not get from
the file.

Threading: guarded by a plain `threading.Lock`. Callers are aiohttp and
FastAPI handlers, and the critical sections are microseconds of dict work.
# ponytail: one process-wide lock; shard by principal only if key verification
# ever shows up in a profile.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.domain.security import Permission, Principal, parse_permissions

__all__ = [
    "MAX_KEYS_PER_PRINCIPAL",
    "PEPPER_ENV",
    "STORE_PATH_ENV",
    "KeyRecord",
    "KeyStore",
    "KeyStoreError",
    "get_key_store",
    "reset_key_store",
]

logger = logging.getLogger("orchestrator.safety.api_keys")

PEPPER_ENV = "ORCHESTRATOR_API_KEY_PEPPER"
STORE_PATH_ENV = "ORCHESTRATOR_API_KEY_STORE"

#: Enough that automation does not fight the limit, few enough that a
#: compromised issuing path cannot quietly mint thousands.
MAX_KEYS_PER_PRINCIPAL = 10

#: How long a revoked/expired record is kept around (for audit/debugging)
#: before being pruned. Bounds store growth over the process/deployment's life.
_RETENTION_SECONDS = 30 * 24 * 3600

KEY_PREFIX = "orchestrator_"
_TOKEN_BYTES = 32


class KeyStoreError(RuntimeError):
    """The key store refused an operation."""


@dataclass(frozen=True)
class KeyRecord:
    """One issued key. Never holds the raw token."""

    key_id: str
    principal_id: str
    permissions: frozenset[Permission]
    digest: str
    created_at: float
    expires_at: float | None = None
    revoked_at: float | None = None

    def is_active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        if self.revoked_at is not None:
            return False
        return self.expires_at is None or now < self.expires_at

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "principal_id": self.principal_id,
            "permissions": sorted(p.value for p in self.permissions),
            "digest": self.digest,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
        }

    def to_public_dict(self) -> dict:
        """The same record with the digest withheld — safe to return over HTTP."""
        public = self.to_dict()
        public.pop("digest")
        return public

    @classmethod
    def from_dict(cls, data: dict) -> KeyRecord:
        expires_at = data.get("expires_at")
        revoked_at = data.get("revoked_at")
        return cls(
            key_id=str(data["key_id"]),
            principal_id=str(data["principal_id"]),
            permissions=parse_permissions(data.get("permissions")),
            digest=str(data["digest"]),
            created_at=float(data["created_at"]),
            expires_at=None if expires_at is None else float(expires_at),
            revoked_at=None if revoked_at is None else float(revoked_at),
        )


class KeyStore:
    """Keys held in memory, optionally mirrored to a JSON file.

    `path=None` is the in-memory adapter: tests and single-process dev runs get
    a working store without provisioning a secret. A persistent store must have
    a real pepper, because that file will outlive the process that wrote it.
    """

    def __init__(self, path: Path | None = None, pepper: str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: dict[str, KeyRecord] = {}
        self._lock = threading.Lock()

        resolved = pepper if pepper is not None else os.getenv(PEPPER_ENV, "").strip()
        if not resolved:
            if self.path is not None:
                raise KeyStoreError(
                    f"a persistent key store needs a pepper — set {PEPPER_ENV}. "
                    "Without it the store file alone is enough to forge keys."
                )
            # In-memory only: an ephemeral pepper is exactly as durable as the
            # keys it protects, which is the whole process lifetime.
            resolved = secrets.token_urlsafe(32)
            logger.info("No %s set; using an ephemeral pepper (in-memory store).", PEPPER_ENV)
        self._pepper = resolved.encode("utf-8")

        if self.path is not None:
            self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            records = [KeyRecord.from_dict(item) for item in raw]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Fail closed. A store we cannot parse is a store we cannot
            # enforce revocation from.
            raise KeyStoreError(f"unreadable key store at {self.path}: {exc}") from exc
        self._records = {record.key_id: record for record in records}

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([r.to_dict() for r in self._records.values()], indent=2)
        # Write-then-replace so a crash cannot truncate the store to nothing.
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            # Windows and some filesystems do not honour this; the pepper,
            # not the file mode, is what makes the file insufficient alone.
            pass

    # ── digest ───────────────────────────────────────────────────────────

    def _digest(self, raw_key: str) -> str:
        return hmac.new(self._pepper, raw_key.encode("utf-8"), "sha256").hexdigest()

    # ── operations ───────────────────────────────────────────────────────

    def issue(
        self,
        principal_id: str,
        permissions: object,
        ttl_seconds: float | None = None,
    ) -> tuple[str, KeyRecord]:
        """Mint a key. The raw value is returned once and never stored."""
        if not principal_id or not principal_id.strip():
            raise KeyStoreError("principal_id is required")

        parsed = parse_permissions(permissions)
        raw_key = f"{KEY_PREFIX}{secrets.token_urlsafe(_TOKEN_BYTES)}"
        now = time.time()

        record = KeyRecord(
            key_id=secrets.token_hex(8),
            principal_id=principal_id,
            permissions=parsed,
            digest=self._digest(raw_key),
            created_at=now,
            expires_at=None if ttl_seconds is None else now + ttl_seconds,
        )

        with self._lock:
            self._prune_expired(now)
            active = sum(
                1
                for r in self._records.values()
                if r.principal_id == principal_id and r.is_active(now)
            )
            if active >= MAX_KEYS_PER_PRINCIPAL:
                raise KeyStoreError(
                    f"principal {principal_id!r} already holds too many active keys "
                    f"({active} >= {MAX_KEYS_PER_PRINCIPAL}); revoke one first"
                )
            self._records[record.key_id] = record
            self._flush()

        logger.info("Issued key %s for principal %s", record.key_id, principal_id)
        return raw_key, record

    def _prune_expired(self, now: float) -> None:
        """Drop records inactive for over `_RETENTION_SECONDS`.

        Caller must already hold `self._lock`. Without this, `_records` (and
        the on-disk store) grow once per issued key for the life of the
        process, and every issue()/revoke() rewrites the whole file.
        """
        cutoff = now - _RETENTION_SECONDS
        stale = [
            key_id
            for key_id, r in self._records.items()
            if not r.is_active(now) and max(r.revoked_at or 0.0, r.expires_at or 0.0) < cutoff
        ]
        for key_id in stale:
            del self._records[key_id]

    def verify(self, raw_key: str) -> Principal | None:
        """Resolve `raw_key` to a `Principal`, or None if it is not usable."""
        if not raw_key or not isinstance(raw_key, str) or not raw_key.strip():
            return None

        candidate = self._digest(raw_key)
        now = time.time()

        with self._lock:
            # Compare against every record before deciding: stopping at the
            # first match makes verify() take less time the earlier a match
            # falls in iteration order, leaking match-existence/position.
            matched: KeyRecord | None = None
            for record in self._records.values():
                if hmac.compare_digest(record.digest, candidate):
                    matched = record

            if matched is None or not matched.is_active(now):
                return None
            return Principal(
                id=matched.principal_id,
                key_id=matched.key_id,
                permissions=matched.permissions,
            )

    def revoke(self, key_id: str) -> bool:
        """Revoke a key. False if it was unknown or already revoked."""
        with self._lock:
            record = self._records.get(key_id)
            if record is None or record.revoked_at is not None:
                return False
            self._records[key_id] = KeyRecord(
                key_id=record.key_id,
                principal_id=record.principal_id,
                permissions=record.permissions,
                digest=record.digest,
                created_at=record.created_at,
                expires_at=record.expires_at,
                revoked_at=time.time(),
            )
            self._flush()

        logger.info("Revoked key %s", key_id)
        return True

    def rotate(self, key_id: str) -> tuple[str, KeyRecord]:
        """Issue a replacement with the same principal and permissions."""
        with self._lock:
            record = self._records.get(key_id)
        if record is None:
            raise KeyStoreError(f"unknown key {key_id!r}")

        if not self.revoke(key_id):
            # Concurrently revoked (or already revoked) between the read
            # above and here — do not mint a fresh key under an identity
            # someone else just killed.
            raise KeyStoreError(f"key {key_id!r} was concurrently revoked; rotate aborted")
        return self.issue(record.principal_id, record.permissions)

    def list_for(self, principal_id: str) -> list[KeyRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.principal_id == principal_id]

    def active_key_count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for r in self._records.values() if r.is_active(now))


_store: KeyStore | None = None
_store_lock = threading.Lock()


def get_key_store() -> KeyStore:
    """The process-wide store. Persistent when `STORE_PATH_ENV` is set."""
    global _store
    with _store_lock:
        if _store is None:
            configured = os.getenv(STORE_PATH_ENV, "").strip()
            _store = KeyStore(path=Path(configured) if configured else None)
        return _store


def reset_key_store() -> None:
    """Drop the process-wide store (tests)."""
    global _store
    with _store_lock:
        _store = None
