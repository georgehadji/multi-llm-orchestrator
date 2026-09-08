"""SSRF-resistant outbound fetching (SEC-004).

The OpenAPI importer accepted an arbitrary ``spec_url``, checked it with
``startswith("http")`` and handed it to ``httpx.AsyncClient.get``. A caller who
could trigger an import made the host fetch cloud metadata, loopback admin
services or anything else on the internal network.

``startswith("http")`` is not a URL check — it passes ``httpx://``,
``http://user:pw@…`` and ``httpevil.example``. This module parses the URL,
resolves it, and rejects destinations that point back inside the perimeter,
re-checking after every redirect.

Residual risk, deliberately not papered over: between the DNS answer we
validate and the connection httpx opens, a hostile resolver can return a
different address (DNS rebinding). Closing that fully means connecting to the
validated IP with the original Host header and matching TLS SNI. The narrower
mitigation here is the host allowlist — use it for anything untrusted.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import ParseResult, urlparse

__all__ = [
    "DEFAULT_MAX_BYTES",
    "OutboundPolicyError",
    "allowed_hosts",
    "check_destination",
    "fetch_json",
    "resolve_and_validate_host",
    "validate_outbound_url",
]

ALLOWED_HOSTS_ENV = "ORCHESTRATOR_OUTBOUND_ALLOWED_HOSTS"
ALLOW_HTTP_ENV = "ORCHESTRATOR_OUTBOUND_ALLOW_HTTP"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: 10 MiB. An OpenAPI document larger than this is not a document we want.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_REDIRECTS = 3

#: Ports that are plainly not web endpoints. Not exhaustive — the address
#: checks below do the real work — but it blocks the obvious pivots.
_BLOCKED_PORTS = frozenset({22, 23, 25, 445, 3306, 5432, 6379, 9200, 11211, 27017})


class OutboundPolicyError(RuntimeError):
    """A destination was refused by the outbound policy."""


def allowed_hosts() -> frozenset[str]:
    """Explicit host allowlist, empty when unset (meaning "no allowlist")."""
    raw = os.getenv(ALLOWED_HOSTS_ENV, "").strip()
    if not raw:
        return frozenset()
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def _http_allowed() -> bool:
    return os.getenv(ALLOW_HTTP_ENV, "").strip().lower() in _TRUTHY


def validate_outbound_url(url: str) -> ParseResult:
    """Parse and vet `url`, or raise `OutboundPolicyError`.

    Checks the URL itself; `resolve_and_validate_host` checks where it points.
    """
    if not url or not isinstance(url, str):
        raise OutboundPolicyError("empty URL")

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise OutboundPolicyError(f"unparseable URL: {exc}") from exc

    schemes = {"https", "http"} if _http_allowed() else {"https"}
    if parsed.scheme not in schemes:
        raise OutboundPolicyError(
            f"scheme {parsed.scheme!r} is not permitted (allowed: {sorted(schemes)})"
        )

    if parsed.username or parsed.password:
        raise OutboundPolicyError("credentials in URL are not permitted")

    if not parsed.hostname:
        raise OutboundPolicyError("URL has no host")

    try:
        port = parsed.port
    except ValueError as exc:
        raise OutboundPolicyError(f"invalid port: {exc}") from exc
    if port is not None and port in _BLOCKED_PORTS:
        raise OutboundPolicyError(f"port {port} is not permitted")

    permitted = allowed_hosts()
    if permitted and parsed.hostname.lower() not in permitted:
        raise OutboundPolicyError(f"host {parsed.hostname!r} is not on {ALLOWED_HOSTS_ENV}")

    return parsed


def _reject_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a rejection reason for `ip`, or None if it is acceptable."""
    # An IPv4-mapped IPv6 address hides a v4 address that must be re-checked.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return _reject_address(mapped)
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        # Covers 169.254.169.254, the cloud metadata endpoint.
        return "link-local"
    if ip.is_private:
        return "private"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_unspecified:
        return "unspecified"
    return None


def resolve_and_validate_host(host: str, port: int = 443) -> list[str]:
    """Resolve `host` and reject any address inside the perimeter.

    Returns the resolved addresses so a caller can log where it actually went.
    """
    # A literal IP still has to pass the address checks.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        reason = _reject_address(literal)
        if reason:
            raise OutboundPolicyError(f"destination {host} is {reason}")
        return [host]

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise OutboundPolicyError(f"could not resolve {host!r}: {exc}") from exc

    if not infos:
        raise OutboundPolicyError(f"no addresses for {host!r}")

    addresses: list[str] = []
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise OutboundPolicyError(f"unparseable address {address!r}") from None
        reason = _reject_address(ip)
        if reason:
            # Every answer must be acceptable: one bad record is enough for a
            # rebinding resolver to win the race.
            raise OutboundPolicyError(f"{host} resolves to a {reason} address")
        addresses.append(address)
    return addresses


def check_destination(url: str) -> ParseResult:
    """Run the full policy chain against `url`."""
    parsed = validate_outbound_url(url)
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port if parsed.port is not None else default_port
    resolve_and_validate_host(parsed.hostname or "", port)
    return parsed


async def fetch_json(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> dict:
    """Fetch and decode JSON from `url` under the outbound policy.

    Redirects are followed manually so each hop is re-validated; httpx's own
    redirect handling would follow a 302 into the perimeter unchecked.
    """
    import json

    import httpx

    current = url
    for _ in range(max_redirects + 1):
        check_destination(current)

        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise OutboundPolicyError("redirect without a Location header")
                    # Resolve relative redirects against the current URL, then
                    # re-run the whole policy on the next iteration.
                    current = str(httpx.URL(current).join(location))
                    continue

                response.raise_for_status()

                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                    raise OutboundPolicyError(
                        f"response declares {declared} bytes, over the {max_bytes} cap"
                    )

                # Stream and stop as soon as the cap is crossed — a declared or
                # absent content-length can still lie; this bounds memory even
                # against a dishonest or chunked/unbounded response.
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise OutboundPolicyError(
                            f"response exceeded the {max_bytes}-byte cap while streaming"
                        )
                    chunks.append(chunk)

                return json.loads(b"".join(chunks))

    raise OutboundPolicyError(f"exceeded {max_redirects} redirects")
