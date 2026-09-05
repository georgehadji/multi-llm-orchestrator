"""What the auditor holds in hand when it decides a check.

``SiteEvidence.available`` is the honesty mechanism. It is computed from what
was actually collected, never declared by a caller, and the auditor refuses to
run any check whose ``requires`` is not a subset of it. That is what makes
"we read a folder of HTML, therefore TLS is fine" unrepresentable rather than
merely discouraged.

Two collection modes:

``from_directory``
    A build on disk: markup, styles, scripts, asset sizes. Everything a static
    read can establish, and nothing else. No HTTP, ever — a ``_headers`` file
    states an intention, and an intention is not a served response.

``from_url``
    A live site: the above plus real responses, status codes and headers. The
    fetcher is injected so the crawl is testable without a socket, and so a
    caller can supply a session with its own auth, proxy or rate limiting.

Neither mode can produce ``FIELD_DATA``, ``BROWSER`` or ``HUMAN`` evidence.
Those arrive from a RUM export, a rendering engine and a person respectively,
and the checks that need them stay outstanding until they do.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from .standard import Evidence

# Files a static host serves as-is and an auditor reads as text.
_TEXT_SUFFIXES = {".txt", ".xml", ".json", ".webmanifest", ".htaccess", ".md"}
_MARKUP_SUFFIXES = {".html", ".htm"}
_STYLE_SUFFIXES = {".css"}
_SCRIPT_SUFFIXES = {".js", ".mjs"}

# Directories that are build inputs or tooling, not the site being served.
_SKIP_DIRS = {".git", "node_modules", ".next", ".svelte-kit", "__pycache__", ".cache"}

_STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
_SCRIPT_BLOCK = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_HREF = re.compile(r"""<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.I)
_LINK_CSS = re.compile(
    r"""<link\b[^>]*\brel\s*=\s*["']?stylesheet["']?[^>]*\bhref\s*=\s*["']([^"']+)["']""", re.I
)
_SCRIPT_SRC = re.compile(r"""<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.I)
_IMG_SRC = re.compile(r"""<img\b[^>]*\bsrc\s*=\s*["']([^"']+)["']""", re.I)


@dataclass(frozen=True)
class BusinessRecord:
    """The facts the site is checked *against*, supplied by the client.

    Local-SEO and contact checks compare what the page claims with what the
    business actually is. Without this the auditor can confirm a phone number
    is formatted as a tel: link but not that it is the right number — which is
    the difference between an ASSISTED check passing and staying outstanding.

    Never holds credentials. Admin authentication is verified in the hosting
    account, not from a file in the repository.
    """

    name: str = ""
    street: str = ""
    postcode: str = ""
    city: str = ""
    country: str = ""
    phone: str = ""
    email: str = ""
    opening_hours: tuple[str, ...] = ()
    google_business_profile: str = ""
    search_console_verified: bool | None = None
    admin_2fa_enabled: bool | None = None
    canonical_domain: str = ""

    @property
    def address_parts(self) -> tuple[str, ...]:
        return tuple(p for p in (self.street, self.postcode, self.city) if p.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "street": self.street,
            "postcode": self.postcode,
            "city": self.city,
            "country": self.country,
            "phone": self.phone,
            "email": self.email,
            "opening_hours": list(self.opening_hours),
            "google_business_profile": self.google_business_profile,
            "search_console_verified": self.search_console_verified,
            "admin_2fa_enabled": self.admin_2fa_enabled,
            "canonical_domain": self.canonical_domain,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BusinessRecord:
        hours = data.get("opening_hours") or ()
        if isinstance(hours, str):
            hours = (hours,)
        return cls(
            name=str(data.get("name", "") or ""),
            street=str(data.get("street", "") or ""),
            postcode=str(data.get("postcode", "") or ""),
            city=str(data.get("city", "") or ""),
            country=str(data.get("country", "") or ""),
            phone=str(data.get("phone", "") or ""),
            email=str(data.get("email", "") or ""),
            opening_hours=tuple(str(h) for h in hours),
            google_business_profile=str(data.get("google_business_profile", "") or ""),
            search_console_verified=_tri(data.get("search_console_verified")),
            admin_2fa_enabled=_tri(data.get("admin_2fa_enabled")),
            canonical_domain=str(data.get("canonical_domain", "") or ""),
        )


def _tri(value: object) -> bool | None:
    """None means "not stated", which is different from stated-as-false."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1", "y"}


@dataclass(frozen=True)
class Page:
    """One HTML document, plus the derivations every check would repeat."""

    path: str
    markup: str
    url: str = ""

    @property
    def text(self) -> str:
        """Visible text: markup, script bodies and style bodies removed."""
        stripped = _STYLE_BLOCK.sub(" ", self.markup)
        stripped = re.sub(r"<script[^>]*>.*?</script>", " ", stripped, flags=re.S | re.I)
        stripped = re.sub(r"<!--.*?-->", " ", stripped, flags=re.S)
        return re.sub(r"\s+", " ", _TAG.sub(" ", stripped)).strip()

    @property
    def links(self) -> tuple[str, ...]:
        return tuple(_HREF.findall(self.markup))

    @property
    def is_error_page(self) -> bool:
        """Whether this is an HTTP error document (404, 410, 500...).

        Error pages are legitimately noindex, legitimately without a canonical,
        and legitimately linked from nowhere. Judging them by the rules for
        content pages turns three pieces of correct practice into three defects
        and, worse, into a launch-blocking "accidental" noindex.
        """
        name = self.path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        return name in {"404", "403", "410", "500", "error", "not-found", "notfound"}

    @property
    def is_home(self) -> bool:
        name = self.path.rsplit("/", 1)[-1].lower()
        return name in {"index.html", "index.htm", ""} or self.path in {"/", ""}


@dataclass(frozen=True)
class HttpResponse:
    """One response, as the auditor received it."""

    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str = ""

    def header(self, name: str) -> str:
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return ""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass
class HttpEvidence:
    """Every response the crawl collected, plus the plain-HTTP redirect probe."""

    responses: dict[str, HttpResponse] = field(default_factory=dict)
    entry_url: str = ""
    plain_http: HttpResponse | None = None

    def header(self, url: str, name: str) -> str:
        response = self.responses.get(url)
        return response.header(name) if response else ""

    def entry(self) -> HttpResponse | None:
        return self.responses.get(self.entry_url)

    def statuses(self) -> dict[str, int]:
        return {url: r.status for url, r in self.responses.items()}


Fetcher = Callable[..., HttpResponse]


@dataclass
class SiteEvidence:
    """Everything the auditor collected, and an honest account of what it is."""

    pages: tuple[Page, ...] = ()
    styles: str = ""
    scripts: str = ""
    assets: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    http: HttpEvidence | None = None
    record: BusinessRecord | None = None
    root: Path | None = None
    source: str = ""
    mode: str = ""

    @property
    def available(self) -> frozenset[Evidence]:
        """Evidence kinds actually held. Derived, never declared."""
        kinds: set[Evidence] = set()
        if self.pages:
            kinds.add(Evidence.MARKUP)
        if self.styles.strip():
            kinds.add(Evidence.STYLES)
        if self.scripts.strip():
            kinds.add(Evidence.SCRIPTS)
        if self.assets or self.files:
            kinds.add(Evidence.ASSETS)
        if self.http is not None and self.http.responses:
            kinds.add(Evidence.HTTP)
        if self.record is not None:
            kinds.add(Evidence.RECORD)
        # BROWSER, FIELD_DATA and HUMAN are never produced here. A rendering
        # engine, a RUM export and a person are the only sources, and none of
        # them is a file on disk or a response header.
        return frozenset(kinds)

    def page(self, path: str) -> Page:
        for page in self.pages:
            if page.path == path or page.url == path:
                return page
        raise KeyError(f"no page {path!r} in this site")

    @property
    def markup(self) -> str:
        """Every page's markup concatenated — for site-wide scans."""
        return "\n".join(p.markup for p in self.pages)

    def file(self, name: str) -> str:
        """A served text file by name, matched case-insensitively at any depth."""
        lowered = name.lower()
        for path, content in self.files.items():
            if path.lower() == lowered or path.lower().endswith("/" + lowered):
                return content
        return ""

    def has_file(self, name: str) -> bool:
        lowered = name.lower()
        candidates = list(self.files) + list(self.assets)
        return any(p.lower() == lowered or p.lower().endswith("/" + lowered) for p in candidates)

    # ── directory mode ───────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, path: str | Path, record: BusinessRecord | None = None) -> SiteEvidence:
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory")

        pages: list[Page] = []
        styles: list[str] = []
        scripts: list[str] = []
        assets: dict[str, int] = {}
        files: dict[str, str] = {}

        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            if any(part in _SKIP_DIRS for part in item.parts):
                continue
            rel = item.relative_to(root).as_posix()
            try:
                assets[rel] = item.stat().st_size
            except OSError:
                continue

            suffix = item.suffix.lower()
            if suffix in _MARKUP_SUFFIXES:
                markup = _read(item)
                pages.append(Page(path=rel, markup=markup))
                styles.extend(_STYLE_BLOCK.findall(markup))
                scripts.extend(_SCRIPT_BLOCK.findall(markup))
            elif suffix in _STYLE_SUFFIXES:
                styles.append(_read(item))
            elif suffix in _SCRIPT_SUFFIXES:
                scripts.append(_read(item))
            elif suffix in _TEXT_SUFFIXES or item.name.startswith("_") or item.name.startswith("."):
                files[rel] = _read(item)

        return cls(
            pages=tuple(pages),
            styles="\n".join(styles),
            scripts="\n".join(scripts),
            assets=assets,
            files=files,
            record=record,
            root=root,
            source=str(root),
            mode="directory",
        )

    # ── url mode ─────────────────────────────────────────────────────────────

    @classmethod
    def from_url(
        cls,
        url: str,
        fetch: Fetcher | None = None,
        record: BusinessRecord | None = None,
        max_pages: int = 25,
    ) -> SiteEvidence:
        fetcher = fetch or _urllib_fetch
        entry = fetcher(url)
        if entry.status >= 400:
            raise ValueError(f"{url} answered {entry.status}: nothing to audit")

        origin = _origin(url)
        responses: dict[str, HttpResponse] = {url: entry}
        pages: list[Page] = [Page(path=_path_of(url), markup=entry.text, url=url)]
        seen = {_normalise(url)}
        queue = [url]
        styles: list[str] = []
        scripts: list[str] = []
        assets: dict[str, int] = {}
        files: dict[str, str] = {}

        while queue and len(pages) < max_pages:
            current = queue.pop(0)
            markup = responses[current].text
            styles.extend(_STYLE_BLOCK.findall(markup))
            scripts.extend(_SCRIPT_BLOCK.findall(markup))

            for raw in _HREF.findall(markup):
                if len(pages) >= max_pages:
                    break
                target = urljoin(current, raw.split("#")[0])
                if not target or _origin(target) != origin:
                    continue
                key = _normalise(target)
                if key in seen:
                    continue
                seen.add(key)
                response = fetcher(target)
                responses[target] = response
                if response.status >= 400:
                    continue
                if not _looks_like_a_page(target, response):
                    continue
                pages.append(Page(path=_path_of(target), markup=response.text, url=target))
                queue.append(target)

        # Subresources: fetched for their real transferred size, which is the
        # number the performance budget is actually about.
        for page in pages:
            for raw in (
                _LINK_CSS.findall(page.markup)
                + _SCRIPT_SRC.findall(page.markup)
                + _IMG_SRC.findall(page.markup)
            ):
                target = urljoin(page.url or url, raw)
                if _origin(target) != origin or target in responses:
                    continue
                response = fetcher(target)
                responses[target] = response
                if response.status >= 400:
                    continue
                rel = _path_of(target).lstrip("/")
                assets[rel] = len(response.body)
                lowered = rel.lower()
                if lowered.endswith(tuple(_STYLE_SUFFIXES)):
                    styles.append(response.text)
                elif lowered.endswith(tuple(_SCRIPT_SUFFIXES)):
                    scripts.append(response.text)

        for name in ("robots.txt", "sitemap.xml", ".well-known/security.txt"):
            target = urljoin(origin + "/", name)
            response = fetcher(target)
            responses[target] = response
            if response.status < 400:
                files[name] = response.text

        plain = None
        if url.lower().startswith("https://"):
            try:
                plain = fetcher("http://" + url.split("://", 1)[1], method="GET")
            except Exception:  # noqa: BLE001 - a failed probe is not an audit failure
                plain = None

        return cls(
            pages=tuple(pages),
            styles="\n".join(styles),
            scripts="\n".join(scripts),
            assets=assets,
            files=files,
            http=HttpEvidence(responses=responses, entry_url=url, plain_http=plain),
            record=record,
            source=url,
            mode="url",
        )


# Extensions a link points at when it is a download, not a page to crawl into.
_NOT_A_PAGE = (
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".dmg",
    ".exe",
    ".apk",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".avif",
    ".svg",
    ".ico",
    ".mp4",
    ".webm",
    ".mov",
    ".mp3",
    ".wav",
    ".ogg",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".xml",
    ".txt",
    ".csv",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
)


def _looks_like_a_page(url: str, response: HttpResponse) -> bool:
    """Whether to treat a crawled URL as a page.

    Excluding known non-pages beats requiring an .html suffix: clean URLs are
    the norm (/services/implants, not /services/implants.html), and demanding
    an extension silently reduced a whole site to its home page.
    """
    if _path_of(url).lower().endswith(_NOT_A_PAGE):
        return False
    content_type = response.header("content-type").lower()
    if content_type and "html" not in content_type and "xml" not in content_type:
        return False
    return True


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""


def _path_of(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or "/"


def _normalise(url: str) -> str:
    return url.split("#")[0].rstrip("/") or url


def _urllib_fetch(url: str, method: str = "GET", timeout: float = 15.0) -> HttpResponse:
    """The real fetcher. Kept thin so the crawl above stays testable offline."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(  # noqa: S310 - scheme is validated below
        url, method=method, headers={"User-Agent": "WF-100-Auditor/1.0"}
    )
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"refusing to fetch non-HTTP url: {url!r}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpResponse(
                url=url,
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
                final_url=response.geturl(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            url=url, status=exc.code, headers=dict(exc.headers.items()), body=exc.read() or b""
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return HttpResponse(url=url, status=0, headers={}, body=str(exc).encode("utf-8"))


__all__ = [
    "BusinessRecord",
    "HttpEvidence",
    "HttpResponse",
    "Page",
    "SiteEvidence",
]
