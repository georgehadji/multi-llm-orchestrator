"""SEC-004 — outbound fetches must not reach back inside the perimeter.

The OpenAPI importer checked ``spec_url.startswith("http")`` and then issued an
unrestricted GET: no allowlist, no address validation, no redirect policy, no
size cap. These tests pin the replacement policy.

No real DNS or network I/O happens here — `socket.getaddrinfo` is stubbed.
"""

from __future__ import annotations

import socket

import pytest

from orchestrator.safety import outbound
from orchestrator.safety.outbound import (
    OutboundPolicyError,
    check_destination,
    resolve_and_validate_host,
    validate_outbound_url,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(outbound.ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.delenv(outbound.ALLOW_HTTP_ENV, raising=False)


@pytest.fixture
def resolves_to(monkeypatch):
    """Force `getaddrinfo` to return a chosen address."""

    def _apply(address: str):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(family, socket.SOCK_STREAM, 6, "", (address, 443))],
        )

    return _apply


class TestSchemeAndCredentials:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://evil.example/",
            "ftp://evil.example/spec.json",
            "httpx://evil.example/",
            "javascript:alert(1)",
        ],
    )
    def test_non_http_schemes_refused(self, url: str) -> None:
        with pytest.raises(OutboundPolicyError):
            validate_outbound_url(url)

    def test_plain_http_refused_by_default(self) -> None:
        with pytest.raises(OutboundPolicyError, match="scheme"):
            validate_outbound_url("http://example.com/spec.json")

    def test_plain_http_allowed_with_optin(self, monkeypatch) -> None:
        monkeypatch.setenv(outbound.ALLOW_HTTP_ENV, "true")
        assert validate_outbound_url("http://example.com/spec.json").scheme == "http"

    def test_userinfo_refused(self) -> None:
        with pytest.raises(OutboundPolicyError, match="credentials"):
            validate_outbound_url("https://user:pw@example.com/spec.json")

    def test_blocked_port_refused(self) -> None:
        with pytest.raises(OutboundPolicyError, match="port"):
            validate_outbound_url("https://example.com:6379/spec.json")

    def test_empty_url_refused(self) -> None:
        with pytest.raises(OutboundPolicyError):
            validate_outbound_url("")


class TestHostAllowlist:
    def test_host_off_allowlist_refused(self, monkeypatch) -> None:
        monkeypatch.setenv(outbound.ALLOWED_HOSTS_ENV, "specs.example.com")
        with pytest.raises(OutboundPolicyError, match="not on"):
            validate_outbound_url("https://evil.example/spec.json")

    def test_host_on_allowlist_passes(self, monkeypatch) -> None:
        monkeypatch.setenv(outbound.ALLOWED_HOSTS_ENV, "specs.example.com, other.example")
        assert validate_outbound_url("https://specs.example.com/spec.json")


class TestAddressValidation:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",  # loopback
            "0.0.0.0",  # unspecified
            "169.254.169.254",  # cloud metadata
            "10.1.2.3",  # private
            "192.168.1.1",  # private
            "172.16.0.1",  # private
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fd00::1",  # IPv6 unique-local
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "224.0.0.1",  # multicast
        ],
    )
    def test_internal_literals_refused(self, address: str) -> None:
        with pytest.raises(OutboundPolicyError):
            resolve_and_validate_host(address)

    def test_public_literal_allowed(self) -> None:
        assert resolve_and_validate_host("8.8.8.8") == ["8.8.8.8"]

    @pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "10.0.0.1"])
    def test_hostname_resolving_internally_refused(self, resolves_to, address: str) -> None:
        resolves_to(address)
        with pytest.raises(OutboundPolicyError, match="resolves to"):
            resolve_and_validate_host("evil.example")

    def test_hostname_resolving_publicly_allowed(self, resolves_to) -> None:
        resolves_to("93.184.216.34")
        assert resolve_and_validate_host("example.com") == ["93.184.216.34"]


class TestCheckDestination:
    def test_metadata_endpoint_blocked_end_to_end(self) -> None:
        # The canonical SSRF target.
        with pytest.raises(OutboundPolicyError):
            check_destination("https://169.254.169.254/latest/meta-data/")

    def test_localhost_blocked_end_to_end(self, resolves_to) -> None:
        resolves_to("127.0.0.1")
        with pytest.raises(OutboundPolicyError):
            check_destination("https://localhost/admin")

    def test_public_destination_passes(self, resolves_to) -> None:
        resolves_to("93.184.216.34")
        assert check_destination("https://example.com/spec.json").hostname == "example.com"


class TestImporterUsesThePolicy:
    @pytest.mark.asyncio
    async def test_openapi_import_refuses_internal_url(self) -> None:
        from orchestrator.api_builder import APIIntegrationBuilder

        builder = APIIntegrationBuilder()
        with pytest.raises(OutboundPolicyError):
            await builder._load_openapi_spec("https://169.254.169.254/latest/meta-data/")

    @pytest.mark.asyncio
    async def test_openapi_import_refuses_file_scheme(self) -> None:
        from orchestrator.api_builder import APIIntegrationBuilder

        builder = APIIntegrationBuilder()
        with pytest.raises(OutboundPolicyError):
            await builder._load_openapi_spec("file:///etc/passwd")
