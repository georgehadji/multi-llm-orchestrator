"""T8 — API keys get a lifecycle instead of a dict that dies with the process.

`api_server.py` kept `self.api_keys[sha256(raw)] = {...}` in memory. Three
problems: a bare SHA-256 of a token is brute-forceable offline if the store
leaks (no pepper, no work factor), there is no revocation or expiry, and every
restart silently invalidates every key.

This pins the replacement: HMAC with a server-side pepper, records that can
expire and be revoked, a bounded number of keys per principal, and raw tokens
that exist exactly once — at issue time.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.domain.security import Permission
from orchestrator.safety import api_keys as keys_mod
from orchestrator.safety.api_keys import KeyStore, KeyStoreError

pytestmark = pytest.mark.unit

PEPPER = "test-pepper-not-a-real-secret"


@pytest.fixture
def store(tmp_path, monkeypatch) -> KeyStore:
    monkeypatch.setenv(keys_mod.PEPPER_ENV, PEPPER)
    return KeyStore(path=tmp_path / "keys.json")


class TestIssue:
    def test_issue_returns_a_raw_key_and_a_record(self, store: KeyStore) -> None:
        raw, record = store.issue("alice", {Permission.READ})
        assert raw
        assert record.principal_id == "alice"
        assert record.permissions == frozenset({Permission.READ})

    def test_raw_key_is_never_stored(self, store: KeyStore) -> None:
        raw, record = store.issue("alice", {Permission.READ})
        assert raw not in json.dumps(record.to_dict())
        assert raw not in store.path.read_text(encoding="utf-8")

    def test_two_issues_differ(self, store: KeyStore) -> None:
        first, _ = store.issue("alice", {Permission.READ})
        second, _ = store.issue("alice", {Permission.READ})
        assert first != second

    def test_key_count_per_principal_is_bounded(self, store: KeyStore) -> None:
        for _ in range(keys_mod.MAX_KEYS_PER_PRINCIPAL):
            store.issue("alice", {Permission.READ})
        with pytest.raises(KeyStoreError, match="too many"):
            store.issue("alice", {Permission.READ})

    def test_revoked_keys_do_not_count_against_the_limit(self, store: KeyStore) -> None:
        _, first = store.issue("alice", {Permission.READ})
        for _ in range(keys_mod.MAX_KEYS_PER_PRINCIPAL - 1):
            store.issue("alice", {Permission.READ})
        store.revoke(first.key_id)
        store.issue("alice", {Permission.READ})  # room freed


class TestVerify:
    def test_valid_key_returns_its_principal(self, store: KeyStore) -> None:
        raw, record = store.issue("alice", {Permission.READ, Permission.EXECUTE})
        principal = store.verify(raw)
        assert principal is not None
        assert principal.id == "alice"
        assert principal.key_id == record.key_id
        assert principal.has(Permission.EXECUTE)

    @pytest.mark.parametrize("bogus", ["", "   ", "not-a-key", "orchestrator_" + "a" * 43])
    def test_unknown_key_is_rejected(self, store: KeyStore, bogus: str) -> None:
        store.issue("alice", {Permission.READ})
        assert store.verify(bogus) is None

    def test_revoked_key_is_rejected(self, store: KeyStore) -> None:
        raw, record = store.issue("alice", {Permission.READ})
        assert store.revoke(record.key_id) is True
        assert store.verify(raw) is None

    def test_expired_key_is_rejected(self, store: KeyStore) -> None:
        raw, _ = store.issue("alice", {Permission.READ}, ttl_seconds=-1)
        assert store.verify(raw) is None

    def test_a_different_pepper_invalidates_every_key(self, tmp_path, monkeypatch) -> None:
        """The pepper is the point: the file alone must not be enough."""
        monkeypatch.setenv(keys_mod.PEPPER_ENV, PEPPER)
        path = tmp_path / "keys.json"
        raw, _ = KeyStore(path=path).issue("alice", {Permission.READ})

        monkeypatch.setenv(keys_mod.PEPPER_ENV, "a-different-pepper")
        assert KeyStore(path=path).verify(raw) is None


class TestRevokeAndRotate:
    def test_revoking_an_unknown_key_is_false_not_an_error(self, store: KeyStore) -> None:
        assert store.revoke("nope") is False

    def test_revoke_is_idempotent(self, store: KeyStore) -> None:
        _, record = store.issue("alice", {Permission.READ})
        assert store.revoke(record.key_id) is True
        assert store.revoke(record.key_id) is False

    def test_rotate_replaces_the_key_and_keeps_the_permissions(self, store: KeyStore) -> None:
        old_raw, old = store.issue("alice", {Permission.READ, Permission.EXECUTE})
        new_raw, new = store.rotate(old.key_id)

        assert new_raw != old_raw
        assert store.verify(old_raw) is None
        assert store.verify(new_raw) is not None
        assert new.permissions == old.permissions
        assert new.principal_id == old.principal_id

    def test_rotating_an_unknown_key_raises(self, store: KeyStore) -> None:
        with pytest.raises(KeyStoreError):
            store.rotate("nope")


class TestPersistence:
    def test_keys_survive_a_restart(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(keys_mod.PEPPER_ENV, PEPPER)
        path = tmp_path / "keys.json"

        raw, _ = KeyStore(path=path).issue("alice", {Permission.READ})

        assert KeyStore(path=path).verify(raw) is not None

    def test_list_for_reports_records_without_digests(self, store: KeyStore) -> None:
        _, record = store.issue("alice", {Permission.READ})
        store.issue("bob", {Permission.READ})

        listed = store.list_for("alice")
        assert [r.key_id for r in listed] == [record.key_id]
        assert "digest" not in listed[0].to_public_dict()

    def test_corrupt_store_fails_closed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(keys_mod.PEPPER_ENV, PEPPER)
        path = tmp_path / "keys.json"
        path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(KeyStoreError):
            KeyStore(path=path)


class TestPepperPolicy:
    def test_persistent_store_without_a_pepper_is_refused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv(keys_mod.PEPPER_ENV, raising=False)
        with pytest.raises(KeyStoreError, match="pepper"):
            KeyStore(path=tmp_path / "keys.json")

    def test_memory_only_store_generates_an_ephemeral_pepper(self, monkeypatch) -> None:
        """Tests and dev runs should not need secret management to start."""
        monkeypatch.delenv(keys_mod.PEPPER_ENV, raising=False)
        store = KeyStore(path=None)
        raw, _ = store.issue("alice", {Permission.READ})
        assert store.verify(raw) is not None
