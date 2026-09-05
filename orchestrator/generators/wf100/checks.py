"""The WF-100 check implementations.

Each function decides exactly one check from the evidence it was given and
returns a ``Finding`` that says what it saw. Three rules hold throughout:

1. A check never guesses. If the evidence cannot settle it, the check is not
   implemented here at all and the auditor reports it outstanding — which is
   why categories B, C, D and G have gaps in this file rather than optimistic
   stand-ins. Core Web Vitals are the clearest case: there is no offline proxy
   for what a real visitor's phone experienced, so there is no B1 here.
2. A check reports what it measured, not that it ran. ``detail`` carries the
   number, the filename, or the offending snippet.
3. ``NOT_APPLICABLE`` is used only where the check genuinely has nothing to
   judge, and always says why — it is the one status that raises the score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlparse

from .evidence import Page, SiteEvidence
from .report import Finding, Status
from .standard import get

Implementation = Callable[[SiteEvidence], Finding]

_IMPLEMENTATIONS: dict[str, Implementation] = {}


def implements(check_id: str) -> Callable[[Implementation], Implementation]:
    """Register a check implementation. Unregistered checks stay outstanding."""

    def decorate(fn: Implementation) -> Implementation:
        if check_id in _IMPLEMENTATIONS:
            raise RuntimeError(f"two implementations registered for {check_id}")
        get(check_id)  # fail loudly on a typo rather than silently never running
        _IMPLEMENTATIONS[check_id] = fn
        return fn

    return decorate


def implementation_for(check_id: str) -> Implementation | None:
    return _IMPLEMENTATIONS.get(check_id)


def implemented_ids() -> frozenset[str]:
    return frozenset(_IMPLEMENTATIONS)


# ── finding constructors ─────────────────────────────────────────────────────


def _ok(cid: str, detail: str, evidence: tuple[str, ...] = ()) -> Finding:
    return Finding(check=get(cid), status=Status.PASS, detail=detail, evidence=evidence)


def _no(cid: str, detail: str, evidence: tuple[str, ...] = ()) -> Finding:
    return Finding(check=get(cid), status=Status.FAIL, detail=detail, evidence=evidence)


def _na(cid: str, why: str) -> Finding:
    return Finding(check=get(cid), status=Status.NOT_APPLICABLE, detail=why)


def _out(cid: str, why: str) -> Finding:
    return Finding(check=get(cid), status=Status.OUTSTANDING, detail=why)


def _judge(cid: str, passed: bool, good: str, bad: str, evidence: tuple[str, ...] = ()) -> Finding:
    return _ok(cid, good, evidence) if passed else _no(cid, bad, evidence)


# ── markup helpers ───────────────────────────────────────────────────────────

_ATTR = re.compile(r"""(\w[\w:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")


def _tags(markup: str, name: str) -> list[str]:
    """Every opening tag of one element, as raw strings."""
    return re.findall(rf"<{name}\b[^>]*>", markup, re.I)


def _attr(tag: str, name: str) -> str | None:
    """One attribute's value, or None when absent. '' means present-but-empty."""
    for key, dq, sq, bare in _ATTR.findall(tag):
        if key.lower() == name.lower():
            return dq or sq or bare or ""
    return None


def _has(tag: str, name: str) -> bool:
    """Whether an attribute is present, valued or bare.

    HTML boolean attributes are written without a value — `<script src=... defer>`
    — so an `=`-based parser reports them absent and every deferred script reads
    as render-blocking. Presence is the question here, not the value.
    """
    if _attr(tag, name) is not None:
        return True
    return bool(re.search(rf"<[^>]*\s{re.escape(name)}(?=[\s/>])", tag, re.I))


def _headings(markup: str) -> list[tuple[int, str]]:
    found = re.findall(r"<h([1-6])\b[^>]*>(.*?)</h\1>", markup, re.S | re.I)
    return [(int(level), _text(body)) for level, body in found]


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment)).strip()


def _links(markup: str) -> list[tuple[str, str]]:
    """(href, visible text) for every anchor."""
    found = re.findall(r"""<a\b([^>]*)>(.*?)</a>""", markup, re.S | re.I)
    out: list[tuple[str, str]] = []
    for attrs, body in found:
        href = _attr("<a " + attrs + ">", "href")
        out.append((href or "", _text(body)))
    return out


def _is_external(url: str) -> bool:
    return bool(urlparse(url).netloc)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc


def _plural(n: int, one: str, many: str = "") -> str:
    return one if n == 1 else (many or one + "s")


def _sample(items: list[str], limit: int = 5) -> tuple[str, ...]:
    return tuple(items[:limit])


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".svg")
_MODERN_SUFFIXES = (".webp", ".avif", ".svg")


def _content_pages(ev: SiteEvidence) -> list[Page]:
    """Pages the SEO rules are about. Error documents are not among them."""
    return [p for p in ev.pages if not p.is_error_page]


def _images(page: Page) -> list[str]:
    return _tags(page.markup, "img")


# ─────────────────────────────────────────────────────────────────────────────
# A. Architecture & Code
# ─────────────────────────────────────────────────────────────────────────────


@implements("A1")
def _a1_semantic_structure(ev: SiteEvidence) -> Finding:
    missing_main = [p.path for p in ev.pages if not re.search(r"<main\b", p.markup, re.I)]
    landmarks = {
        tag
        for tag in ("header", "nav", "main", "footer", "section", "article", "aside")
        if re.search(rf"<{tag}\b", ev.markup, re.I)
    }
    if missing_main:
        return _no(
            "A1",
            f"{len(missing_main)} {_plural(len(missing_main), 'page')} without a <main> landmark",
            _sample(missing_main),
        )
    if len(landmarks) < 4:
        return _no(
            "A1",
            f"only {len(landmarks)} distinct landmarks in use ({', '.join(sorted(landmarks))})",
        )
    return _ok("A1", f"landmarks in use: {', '.join(sorted(landmarks))}")


@implements("A2")
def _a2_single_h1(ev: SiteEvidence) -> Finding:
    offenders = []
    for page in ev.pages:
        count = len(re.findall(r"<h1\b", page.markup, re.I))
        if count != 1:
            offenders.append(f"{page.path}: {count} h1")
    return _judge(
        "A2",
        not offenders,
        f"exactly one <h1> on each of {len(ev.pages)} {_plural(len(ev.pages), 'page')}",
        f"{len(offenders)} {_plural(len(offenders), 'page')} without exactly one <h1>",
        _sample(offenders),
    )


@implements("A3")
def _a3_heading_hierarchy(ev: SiteEvidence) -> Finding:
    offenders = []
    for page in ev.pages:
        previous = 0
        for level, text in _headings(page.markup):
            if previous and level > previous + 1:
                offenders.append(f"{page.path}: h{previous} -> h{level} ({text[:40]})")
            previous = level
    return _judge(
        "A3",
        not offenders,
        "heading outline descends one level at a time on every page",
        f"{len(offenders)} skipped heading {_plural(len(offenders), 'level')}",
        _sample(offenders),
    )


_OPAQUE_SLUG = re.compile(r"^(?:[0-9]+|p[0-9]+|page[0-9]*|[0-9a-f]{6,}|untitled|new-page)$", re.I)


@implements("A4")
def _a4_readable_urls(ev: SiteEvidence) -> Finding:
    offenders = []
    for page in ev.pages:
        path = page.path
        if "?" in path:
            offenders.append(f"{path} (query string)")
            continue
        slug = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if not slug or slug in {"index", "404"}:
            continue
        if _OPAQUE_SLUG.match(slug) or "_" in slug or slug != slug.lower():
            offenders.append(path)
    return _judge(
        "A4",
        not offenders,
        f"all {len(ev.pages)} URL {_plural(len(ev.pages), 'path')} are lowercase readable slugs",
        f"{len(offenders)} unreadable {_plural(len(offenders), 'URL')}",
        _sample(offenders),
    )


@implements("A5")
def _a5_no_copy_paste(ev: SiteEvidence) -> Finding:
    """Identical *body* blocks across pages. Shared chrome is excluded: a header
    repeating on every page is componentisation working, not copy-paste."""
    if len(ev.pages) < 2:
        return _na("A5", "a single-page site has nothing to duplicate across pages")

    seen: dict[str, list[str]] = {}
    for page in ev.pages:
        body = re.sub(r"<(header|nav|footer)\b.*?</\1>", " ", page.markup, flags=re.S | re.I)
        for block in re.findall(
            r"<(?:section|article|div)\b[^>]*>(.*?)</(?:section|article|div)>", body, re.S | re.I
        ):
            text = _text(block)
            if len(text) < 180:
                continue
            seen.setdefault(text, []).append(page.path)

    duplicates = {text: pages for text, pages in seen.items() if len(set(pages)) > 1}
    return _judge(
        "A5",
        not duplicates,
        f"no body block repeats verbatim across the {len(ev.pages)} pages",
        f"{len(duplicates)} content {_plural(len(duplicates), 'block')} duplicated across pages",
        _sample([f"{sorted(set(p))}: {t[:60]}..." for t, p in duplicates.items()]),
    )


# Libraries whose presence is easy to justify and easy to forget to remove.
_LIB_GLOBALS = {
    "jquery": (r"\$\(|jQuery\b",),
    "lodash": (r"\b_\.",),
    "moment": (r"\bmoment\(",),
    "bootstrap": (r"data-bs-|\bbootstrap\b",),
    "gsap": (r"\bgsap\b|TweenMax",),
    "three": (r"\bTHREE\b",),
    "alpine": (r"\bx-data\b",),
    "swiper": (r"\bSwiper\b|swiper-",),
}


@implements("A6")
def _a6_unused_dependencies(ev: SiteEvidence) -> Finding:
    haystack = ev.markup + "\n" + ev.scripts
    loaded = []
    unused = []
    for tag in _tags(ev.markup, "script"):
        src = (_attr(tag, "src") or "").lower()
        if not src:
            continue
        for lib, patterns in _LIB_GLOBALS.items():
            if lib in src:
                loaded.append(lib)
                if not any(re.search(p, haystack) for p in patterns):
                    unused.append(f"{lib} ({src})")
    if not loaded:
        return _ok("A6", "no third-party library bundles loaded")
    return _judge(
        "A6",
        not unused,
        f"every loaded library is used: {', '.join(sorted(set(loaded)))}",
        f"{len(unused)} library {_plural(len(unused), 'bundle')} loaded but never called",
        _sample(unused),
    )


@implements("A8")
def _a8_internal_links(ev: SiteEvidence) -> Finding:
    known = {p.path.lstrip("./") for p in ev.pages} | {k.lstrip("./") for k in ev.assets}
    known |= {p.lstrip("./") for p in ev.files}
    broken = []
    for page in ev.pages:
        base = page.path.rsplit("/", 1)[0] if "/" in page.path else ""
        for href, _label in _links(page.markup):
            if not href or _is_external(href):
                continue
            if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
                continue
            target = href.split("#")[0].split("?")[0]
            if not target:
                continue
            if target == "/":
                resolved = "index.html"
            elif target.startswith("/"):
                resolved = target.lstrip("/")
            else:
                resolved = f"{base}/{target}" if base else target
            resolved = re.sub(r"[^/]+/\.\./", "", resolved).lstrip("./")
            candidates = {
                resolved,
                resolved.rstrip("/") + "/index.html",
                resolved + ".html",
                resolved.rstrip("/") + ".html",
            }
            if not (candidates & known):
                broken.append(f"{page.path} -> {href}")
    return _judge(
        "A8",
        not broken,
        "every internal link resolves to a file in the build",
        f"{len(broken)} broken internal {_plural(len(broken), 'link')}",
        _sample(broken),
    )


@implements("A9")
def _a9_not_found_page(ev: SiteEvidence) -> Finding:
    if ev.http is not None:
        probe = f"{ev.source.rstrip('/')}/wf100-probe-nonexistent-path"
        response = ev.http.responses.get(probe)
        if response is not None:
            return _judge(
                "A9",
                response.status == 404 and len(response.body) > 200,
                f"a request for a missing path answered 404 with {len(response.body)}B of content",
                f"a missing path answered {response.status} with {len(response.body)}B",
            )
    has404 = ev.has_file("404.html") or "404" in ev.file("_redirects")
    if not has404:
        return _no("A9", "no 404.html in the build and no 404 rule in _redirects")
    markup = next((p.markup for p in ev.pages if p.path.endswith("404.html")), "")
    if markup and not _links(markup):
        return _no("A9", "the 404 page offers no link back into the site")
    return _ok("A9", "a 404 page is present and links back into the site")


_DEBUG_MARKERS = (
    (r"\bconsole\.(log|debug|table|dir)\s*\(", "console logging"),
    (r"\bdebugger\b", "debugger statement"),
    (r"\b(TODO|FIXME|XXX|HACK)\b", "unfinished-work marker"),
    (r"\balert\s*\(", "alert() call"),
)


@implements("A10")
def _a10_debug_artifacts(ev: SiteEvidence) -> Finding:
    hits = []
    for pattern, label in _DEBUG_MARKERS:
        found = re.findall(pattern, ev.scripts + "\n" + ev.markup, re.I)
        if found:
            hits.append(f"{label} x{len(found)}")
    commented = len(re.findall(r"<!--(?!\s*\[if|\s*/?(?:repeat|section))", ev.markup))
    if commented > 12:
        hits.append(f"{commented} HTML comments left in the build")
    return _judge(
        "A10",
        not hits,
        "no debug statements, unfinished-work markers or stray comment blocks",
        "debug artifacts shipped: " + "; ".join(hits),
        _sample(hits),
    )


# ─────────────────────────────────────────────────────────────────────────────
# B. Performance
#
# B1-B3 (LCP, INP, CLS) are deliberately absent: they are field metrics, and
# nothing in a build directory or a single response is a substitute for what a
# real visitor's device measured. They stay outstanding until a CrUX or RUM
# export is supplied. B12-B14 are likewise absent from directory mode; they are
# properties of the server, and the auditor gets them only from live headers.
# ─────────────────────────────────────────────────────────────────────────────

_MAX_IMAGE_BYTES = 300 * 1024
_MAX_IMAGE_TOTAL = 1_500 * 1024
_MAX_CODE_BYTES = 300 * 1024


def _image_assets(ev: SiteEvidence) -> dict[str, int]:
    return {
        path: size for path, size in ev.assets.items() if path.lower().endswith(_IMAGE_SUFFIXES)
    }


@implements("B4")
def _b4_image_budget(ev: SiteEvidence) -> Finding:
    images = _image_assets(ev)
    if not images:
        return _na("B4", "the build ships no raster or vector images to weigh")
    oversized = [f"{p} ({s // 1024}KB)" for p, s in images.items() if s > _MAX_IMAGE_BYTES]
    total = sum(images.values())
    if oversized:
        return _no(
            "B4",
            f"{len(oversized)} {_plural(len(oversized), 'image')} over "
            f"{_MAX_IMAGE_BYTES // 1024}KB",
            _sample(sorted(oversized, reverse=True)),
        )
    if total > _MAX_IMAGE_TOTAL:
        return _no(
            "B4", f"{total // 1024}KB of imagery, over the {_MAX_IMAGE_TOTAL // 1024}KB budget"
        )
    return _ok(
        "B4",
        f"{len(images)} images, {total // 1024}KB total, largest {max(images.values()) // 1024}KB",
    )


@implements("B5")
def _b5_modern_formats(ev: SiteEvidence) -> Finding:
    images = _image_assets(ev)
    if not images:
        return _na("B5", "no images to convert")
    legacy = [p for p in images if p.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]
    if not legacy:
        return _ok("B5", f"all {len(images)} images are WebP, AVIF or SVG")
    picture_sources = len(_tags(ev.markup, "source"))
    if picture_sources >= len(legacy):
        return _ok("B5", f"{len(legacy)} legacy images, each with a <picture> source alternative")
    return _no(
        "B5",
        f"{len(legacy)} of {len(images)} images are legacy formats with no modern alternative",
        _sample(sorted(legacy)),
    )


@implements("B6")
def _b6_responsive_images(ev: SiteEvidence) -> Finding:
    unsized = []
    for page in ev.pages:
        for tag in _images(page):
            src = _attr(tag, "src") or ""
            if src.startswith("data:") or src.lower().endswith(".svg"):
                continue
            sized = (_has(tag, "width") and _has(tag, "height")) or _has(tag, "srcset")
            if not sized:
                unsized.append(f"{page.path}: {src[:60]}")
    total = sum(len(_images(p)) for p in ev.pages)
    if not total:
        return _na("B6", "no <img> elements to size")
    return _judge(
        "B6",
        not unsized,
        f"all {total} images carry explicit dimensions or a srcset",
        f"{len(unsized)} of {total} images have neither width/height nor srcset",
        _sample(unsized),
    )


@implements("B7")
def _b7_lazy_loading(ev: SiteEvidence) -> Finding:
    late_eager = []
    total = 0
    for page in ev.pages:
        tags = _images(page)
        total += len(tags)
        # The first image is the LCP candidate and should stay eager; the rest
        # are below the fold on a phone and should not block it.
        for tag in tags[1:]:
            if (_attr(tag, "loading") or "").lower() != "lazy":
                late_eager.append(f"{page.path}: {(_attr(tag, 'src') or '')[:60]}")
    if total <= 1:
        return _na("B7", "at most one image per page, so there is nothing below the fold to defer")
    return _judge(
        "B7",
        not late_eager,
        f"every image after the first is lazy-loaded ({total} images)",
        f"{len(late_eager)} below-the-fold {_plural(len(late_eager), 'image')} load eagerly",
        _sample(late_eager),
    )


@implements("B8")
def _b8_critical_assets(ev: SiteEvidence) -> Finding:
    problems = []
    for page in ev.pages:
        tags = _images(page)
        if tags and (_attr(tags[0], "loading") or "").lower() == "lazy":
            problems.append(f"{page.path}: the first image is lazy-loaded, delaying the LCP")
    # Only resources the browser actually fetches sit on the critical path.
    # rel=canonical, rel=alternate and their kin are metadata: demanding a
    # preconnect for a canonical URL's own origin measures the wrong thing.
    fetching_rels = {"stylesheet", "preload", "modulepreload", "prefetch", "icon", "manifest"}
    external_origins = {
        _origin(_attr(t, "href") or "")
        for t in _tags(ev.markup, "link")
        if (_attr(t, "rel") or "").lower() in fetching_rels
    }
    external_origins |= {_origin(_attr(t, "src") or "") for t in _tags(ev.markup, "script")}
    external_origins -= {""}
    preconnects = {
        _origin(_attr(t, "href") or "")
        for t in _tags(ev.markup, "link")
        if (_attr(t, "rel") or "").lower() in {"preconnect", "dns-prefetch"}
    }
    unconnected = external_origins - preconnects
    if unconnected:
        problems.append(
            f"{len(unconnected)} external {_plural(len(unconnected), 'origin')} on the critical "
            f"path with no preconnect: {', '.join(sorted(unconnected))}"
        )
    return _judge(
        "B8",
        not problems,
        "the first image loads eagerly and every external origin is preconnected",
        "; ".join(problems),
        _sample(problems),
    )


@implements("B9")
def _b9_font_payload(ev: SiteEvidence) -> Finding:
    families = set(
        re.findall(r"@font-face\s*{[^}]*?font-family\s*:\s*['\"]?([^;'\"}]+)", ev.styles, re.I)
    )
    for tag in _tags(ev.markup, "link"):
        href = _attr(tag, "href") or ""
        if "fonts.googleapis.com" in href:
            families |= set(re.findall(r"family=([^:&]+)", href))
    if not families:
        return _ok("B9", "no webfonts loaded — the system font stack costs nothing")
    if len(families) > 2:
        return _no(
            "B9",
            f"{len(families)} font families loaded: {', '.join(sorted(families))}",
        )
    if "font-display" not in ev.styles.lower() and not any(
        "display=swap" in (_attr(t, "href") or "") for t in _tags(ev.markup, "link")
    ):
        return _no("B9", f"{len(families)} font families load without font-display: swap")
    return _ok("B9", f"{len(families)} font {_plural(len(families), 'family')} with display: swap")


@implements("B10")
def _b10_third_party_scripts(ev: SiteEvidence) -> Finding:
    origins = {
        _origin(_attr(t, "src") or "") for t in _tags(ev.markup, "script") if _attr(t, "src")
    } - {""}
    if len(origins) > 3:
        return _no(
            "B10",
            f"{len(origins)} third-party script origins: {', '.join(sorted(origins))}",
        )
    return _ok(
        "B10",
        f"{len(origins)} third-party script {_plural(len(origins), 'origin')}"
        + (f": {', '.join(sorted(origins))}" if origins else ""),
    )


@implements("B11")
def _b11_code_weight(ev: SiteEvidence) -> Finding:
    code = {
        path: size
        for path, size in ev.assets.items()
        if path.lower().endswith((".css", ".js", ".mjs"))
    }
    total = sum(code.values())
    if total > _MAX_CODE_BYTES:
        biggest = sorted(code.items(), key=lambda kv: -kv[1])[:3]
        return _no(
            "B11",
            f"{total // 1024}KB of CSS and JS, over the {_MAX_CODE_BYTES // 1024}KB budget",
            tuple(f"{p} ({s // 1024}KB)" for p, s in biggest),
        )
    return _ok("B11", f"{total // 1024}KB of CSS and JS across {len(code)} files")


@implements("B15")
def _b15_render_blocking(ev: SiteEvidence) -> Finding:
    blocking = []
    for page in ev.pages:
        head = re.search(r"<head\b[^>]*>(.*?)</head>", page.markup, re.S | re.I)
        if not head:
            continue
        for tag in _tags(head.group(1), "script"):
            src = _attr(tag, "src")
            if not src:
                continue
            if not (
                _has(tag, "defer") or _has(tag, "async") or (_attr(tag, "type") or "") == "module"
            ):
                blocking.append(f"{page.path}: {src[:60]}")
    return _judge(
        "B15",
        not blocking,
        "no render-blocking scripts in any <head>",
        f"{len(blocking)} blocking {_plural(len(blocking), 'script')} in <head>",
        _sample(blocking),
    )


__all__ = ["implementation_for", "implemented_ids", "implements"]


# ─────────────────────────────────────────────────────────────────────────────
# C. Accessibility
#
# C2 (keyboard navigation), C10 (colour as the only signal) and C13 (keyboard
# traps) have no implementation and never will: each is defined by what happens
# when a person tries to use the site, and a parser cannot try.
# ─────────────────────────────────────────────────────────────────────────────


@implements("C1")
def _c1_wcag_baseline(ev: SiteEvidence) -> Finding:
    """The machine-checkable violations no other C check owns."""
    problems = []
    for page in ev.pages:
        ids = re.findall(r"""\bid\s*=\s*["']([^"']+)["']""", page.markup)
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            problems.append(f"{page.path}: duplicate id {sorted(duplicates)[:3]}")
        for tag in _tags(page.markup, "iframe"):
            if not (_attr(tag, "title") or "").strip():
                problems.append(f"{page.path}: <iframe> without a title")
        if re.search(r"<table\b", page.markup, re.I) and not re.search(r"<th\b", page.markup, re.I):
            problems.append(f"{page.path}: <table> without header cells")
        if not re.search(r"<title\b[^>]*>\s*\S", page.markup, re.I):
            problems.append(f"{page.path}: empty or missing <title>")
    return _judge(
        "C1",
        not problems,
        f"no machine-detectable WCAG violations across {len(ev.pages)} pages",
        f"{len(problems)} machine-detectable WCAG {_plural(len(problems), 'violation')}",
        _sample(problems),
    )


@implements("C3")
def _c3_focus_visible(ev: SiteEvidence) -> Finding:
    css = ev.styles
    focus_rules = re.findall(r"([^{}]*:focus(?:-visible|-within)?[^{}]*)\{([^}]*)\}", css, re.I)
    if not focus_rules:
        return _no(
            "C3",
            "no :focus or :focus-visible styling at all — keyboard users cannot see where they are",
        )
    stripped = []
    for selector, body in focus_rules:
        removes = re.search(r"outline\s*:\s*(none|0)\b", body, re.I)
        replaces = re.search(r"box-shadow|border|outline\s*:\s*(?!none|0)", body, re.I)
        if removes and not replaces:
            stripped.append(selector.strip()[:60])
    return _judge(
        "C3",
        not stripped,
        f"{len(focus_rules)} focus {_plural(len(focus_rules), 'rule')}, none removing the "
        "indicator without a replacement",
        f"{len(stripped)} {_plural(len(stripped), 'rule')} remove the focus outline "
        "without replacing it",
        _sample(stripped),
    )


@implements("C4")
def _c4_tab_order(ev: SiteEvidence) -> Finding:
    positive = []
    for page in ev.pages:
        for value in re.findall(r"""\btabindex\s*=\s*["']?(\d+)["']?""", page.markup, re.I):
            if int(value) > 0:
                positive.append(f"{page.path}: tabindex={value}")
    return _judge(
        "C4",
        not positive,
        "tab order follows the DOM: no positive tabindex anywhere",
        f"{len(positive)} positive tabindex {_plural(len(positive), 'value')} override the DOM order",
        _sample(positive),
    )


@implements("C5")
def _c5_nav_and_forms(ev: SiteEvidence) -> Finding:
    problems = []
    if not re.search(r"<nav\b", ev.markup, re.I):
        problems.append("no <nav> landmark anywhere on the site")
    for page in ev.pages:
        for form in re.findall(r"<form\b.*?</form>", page.markup, re.S | re.I):
            has_submit = re.search(
                r"""<button\b(?![^>]*\btype\s*=\s*["']?(?:button|reset))|"""
                r"""<input\b[^>]*\btype\s*=\s*["']?submit""",
                form,
                re.I,
            )
            if not has_submit:
                problems.append(f"{page.path}: a <form> with no submit control")
    return _judge(
        "C5",
        not problems,
        "navigation is a landmark and every form has a submit control",
        "; ".join(problems),
        _sample(problems),
    )


_LABELLESS_TYPES = {"hidden", "submit", "button", "reset", "image"}


@implements("C6")
def _c6_input_labels(ev: SiteEvidence) -> Finding:
    unlabelled = []
    total = 0
    for page in ev.pages:
        label_targets = set(
            re.findall(r"""<label\b[^>]*\bfor\s*=\s*["']([^"']+)["']""", page.markup, re.I)
        )
        wrapped = re.findall(r"<label\b[^>]*>(.*?)</label>", page.markup, re.S | re.I)
        for element in ("input", "select", "textarea"):
            for tag in _tags(page.markup, element):
                if (_attr(tag, "type") or "").lower() in _LABELLESS_TYPES:
                    continue
                total += 1
                field_id = _attr(tag, "id") or ""
                labelled = (
                    (field_id and field_id in label_targets)
                    or (_attr(tag, "aria-label") or "").strip()
                    or (_attr(tag, "aria-labelledby") or "").strip()
                    or any(tag in block for block in wrapped)
                )
                if not labelled:
                    unlabelled.append(f"{page.path}: {tag[:70]}")
    if not total:
        return _na("C6", "the site has no form fields to label")
    return _judge(
        "C6",
        not unlabelled,
        f"all {total} form fields have an associated label",
        f"{len(unlabelled)} of {total} form fields have no label",
        _sample(unlabelled),
    )


_USELESS_ALT = {"image", "photo", "picture", "img", "graphic", "icon", "logo image", "untitled"}


@implements("C7")
def _c7_useful_alt(ev: SiteEvidence) -> Finding:
    poor = []
    informative = 0
    for page in ev.pages:
        for tag in _images(page):
            alt = _attr(tag, "alt")
            if alt is None or not alt.strip():
                continue  # missing/decorative is C8's business
            informative += 1
            cleaned = alt.strip().lower()
            src = (_attr(tag, "src") or "").rsplit("/", 1)[-1].lower()
            if cleaned in _USELESS_ALT or cleaned == src or len(cleaned) < 4:
                poor.append(f"{page.path}: alt={alt!r}")
    if not informative:
        return _na("C7", "no informative images on the site — every image is marked decorative")
    return _judge(
        "C7",
        not poor,
        f"all {informative} informative images carry descriptive alt text",
        f"{len(poor)} of {informative} alt texts are filenames or generic words",
        _sample(poor),
    )


@implements("C8")
def _c8_decorative_images(ev: SiteEvidence) -> Finding:
    missing = []
    total = 0
    for page in ev.pages:
        for tag in _images(page):
            total += 1
            if _attr(tag, "alt") is None and (_attr(tag, "role") or "") != "presentation":
                missing.append(f"{page.path}: {(_attr(tag, 'src') or '')[:60]}")
    if not total:
        return _na("C8", "the site has no <img> elements")
    return _judge(
        "C8",
        not missing,
        f"all {total} images carry an alt attribute, decorative ones empty",
        f"{len(missing)} of {total} images have no alt attribute at all",
        _sample(missing),
    )


_HEX = re.compile(r"#(?:[0-9a-f]{3}|[0-9a-f]{6})\b", re.I)


@implements("C9")
def _c9_contrast(ev: SiteEvidence) -> Finding:
    from ..website_palette import contrast_ratio

    css = ev.styles
    pairs: list[tuple[str, str, str]] = []
    for selector, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        fg = re.search(r"(?<!-)\bcolor\s*:\s*(#[0-9a-fA-F]{3,6})", body)
        bg = re.search(r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6})", body)
        if fg and bg:
            pairs.append((selector.strip()[:50], fg.group(1), bg.group(1)))

    # A page's body pair is the one that matters most and is often split across
    # rules, so reconstruct it from the declared tokens when it is not colocated.
    root = re.search(r":root\s*\{([^}]*)\}", css)
    if root:
        ink = re.search(r"--(?:ink|text|fg|color-text)\s*:\s*(#[0-9a-fA-F]{3,6})", root.group(1))
        paper = re.search(
            r"--(?:paper|bg|background|surface)\s*:\s*(#[0-9a-fA-F]{3,6})", root.group(1)
        )
        if ink and paper:
            pairs.append(("body tokens", ink.group(1), paper.group(1)))

    if not pairs:
        return _out(
            "C9",
            "no colour pair could be resolved from the stylesheet — contrast must be "
            "measured in a browser against the computed styles",
        )
    failures = [
        f"{sel}: {fg} on {bg} = {contrast_ratio(fg, bg):.2f}:1"
        for sel, fg, bg in pairs
        if contrast_ratio(fg, bg) < 4.5
    ]
    return _judge(
        "C9",
        not failures,
        f"all {len(pairs)} resolvable colour pairs meet 4.5:1 "
        f"(lowest {min(contrast_ratio(f, b) for _, f, b in pairs):.2f}:1)",
        f"{len(failures)} of {len(pairs)} colour pairs fall below 4.5:1",
        _sample(failures),
    )


@implements("C11")
def _c11_button_names(ev: SiteEvidence) -> Finding:
    nameless = []
    total = 0
    for page in ev.pages:
        for attrs, body in re.findall(r"<button\b([^>]*)>(.*?)</button>", page.markup, re.S | re.I):
            total += 1
            tag = "<button " + attrs + ">"
            named = (
                _text(body)
                or (_attr(tag, "aria-label") or "").strip()
                or (_attr(tag, "aria-labelledby") or "").strip()
                or (_attr(tag, "title") or "").strip()
            )
            if not named:
                nameless.append(f"{page.path}: {tag[:70]}")
    if not total:
        return _na("C11", "the site has no <button> elements")
    return _judge(
        "C11",
        not nameless,
        f"all {total} buttons have an accessible name",
        f"{len(nameless)} of {total} buttons have no accessible name",
        _sample(nameless),
    )


# Trailing punctuation and affordance glyphs a designer adds to a link label:
# "read more →" is the same vague label as "read more".
_TRAILING_CHROME = re.compile(r"[\s.!\u2192>\u00bb\u203a)]+$")

_VAGUE_LINKS = {
    "click here",
    "here",
    "read more",
    "more",
    "link",
    "this",
    "learn more",
    "continue",
    "go",
    "details",
    "info",
    "download",
}


@implements("C12")
def _c12_link_text(ev: SiteEvidence) -> Finding:
    vague = []
    total = 0
    for page in ev.pages:
        for href, label in _links(page.markup):
            if not href or href.startswith(("#", "javascript:")):
                continue
            total += 1
            if _TRAILING_CHROME.sub("", label.strip().lower()) in _VAGUE_LINKS:
                vague.append(f"{page.path}: {label!r} -> {href[:40]}")
    if not total:
        return _na("C12", "the site has no navigable links")
    return _judge(
        "C12",
        not vague,
        f"all {total} link labels name their destination",
        f"{len(vague)} of {total} links are labelled with a generic phrase",
        _sample(vague),
    )


@implements("C14")
def _c14_reduced_motion(ev: SiteEvidence) -> Finding:
    css = ev.styles
    moves = re.search(r"\b(transition|animation|transform)\s*:", css, re.I)
    if not moves:
        return _na("C14", "the stylesheet declares no motion, so there is nothing to reduce")
    if not re.search(r"prefers-reduced-motion", css, re.I):
        return _no(
            "C14",
            "the stylesheet animates but never checks prefers-reduced-motion",
        )
    return _ok("C14", "motion is gated behind a prefers-reduced-motion query")


@implements("C15")
def _c15_page_language(ev: SiteEvidence) -> Finding:
    missing = []
    for page in ev.pages:
        html = re.search(r"<html\b[^>]*>", page.markup, re.I)
        lang = _attr(html.group(0), "lang") if html else None
        if not (lang or "").strip():
            missing.append(page.path)
    languages = {
        (_attr(re.search(r"<html\b[^>]*>", p.markup, re.I).group(0), "lang") or "")
        for p in ev.pages
        if re.search(r"<html\b[^>]*>", p.markup, re.I)
    } - {""}
    return _judge(
        "C15",
        not missing,
        f"every page declares a language ({', '.join(sorted(languages)) or 'none'})",
        f"{len(missing)} {_plural(len(missing), 'page')} without <html lang>",
        _sample(missing),
    )


# ─────────────────────────────────────────────────────────────────────────────
# D. SEO
#
# D4 (does the content match search intent) has no implementation. Intent is a
# claim about what people wanted, and no amount of markup settles it.
# ─────────────────────────────────────────────────────────────────────────────


def _meta(markup: str, name: str, attr: str = "name") -> str:
    for tag in _tags(markup, "meta"):
        if (_attr(tag, attr) or "").lower() == name.lower():
            return (_attr(tag, "content") or "").strip()
    return ""


def _title_of(markup: str) -> str:
    found = re.search(r"<title[^>]*>(.*?)</title>", markup, re.S | re.I)
    return _text(found.group(1)) if found else ""


@implements("D1")
def _d1_titles(ev: SiteEvidence) -> Finding:
    # Error pages carry no SERP snippet — they are noindex by design — so their
    # title and description are held to the site's own needs, not Google's.
    titles: dict[str, str] = {p.path: _title_of(p.markup) for p in _content_pages(ev)}
    problems = []
    for path, title in titles.items():
        if not title:
            problems.append(f"{path}: no <title>")
        elif not 20 <= len(title) <= 65:
            problems.append(f"{path}: {len(title)} chars ({title[:45]!r})")
    duplicates = {t for t in titles.values() if t and list(titles.values()).count(t) > 1}
    for title in sorted(duplicates):
        problems.append(
            f"duplicate title on {sum(1 for v in titles.values() if v == title)} pages: {title[:45]!r}"
        )
    return _judge(
        "D1",
        not problems,
        f"{len(titles)} unique titles, each 20-65 characters",
        f"{len(problems)} title {_plural(len(problems), 'problem')}",
        _sample(problems),
    )


@implements("D2")
def _d2_meta_descriptions(ev: SiteEvidence) -> Finding:
    descriptions = {p.path: _meta(p.markup, "description") for p in _content_pages(ev)}
    problems = []
    for path, text in descriptions.items():
        if not text:
            problems.append(f"{path}: no meta description")
        elif not 70 <= len(text) <= 160:
            problems.append(f"{path}: {len(text)} chars")
    values = [d for d in descriptions.values() if d]
    duplicates = {d for d in values if values.count(d) > 1}
    for text in sorted(duplicates):
        problems.append(f"duplicate description: {text[:45]!r}")
    return _judge(
        "D2",
        not problems,
        f"{len(descriptions)} unique meta descriptions, each 70-160 characters",
        f"{len(problems)} meta description {_plural(len(problems), 'problem')}",
        _sample(problems),
    )


@implements("D3")
def _d3_h1(ev: SiteEvidence) -> Finding:
    h1s = {}
    problems = []
    for page in _content_pages(ev):
        headings = [text for level, text in _headings(page.markup) if level == 1]
        if not headings:
            problems.append(f"{page.path}: no H1")
            continue
        h1s[page.path] = headings[0]
        if len(headings[0]) < 3:
            problems.append(f"{page.path}: H1 is {headings[0]!r}")
    values = list(h1s.values())
    for text in {v for v in values if values.count(v) > 1}:
        problems.append(f"H1 {text[:40]!r} repeats on {values.count(text)} pages")
    return _judge(
        "D3",
        not problems,
        f"{len(h1s)} pages, each with one specific and distinct H1",
        f"{len(problems)} H1 {_plural(len(problems), 'problem')}",
        _sample(problems),
    )


@implements("D5")
def _d5_canonical(ev: SiteEvidence) -> Finding:
    missing = []
    relative = []
    for page in _content_pages(ev):
        href = ""
        for tag in _tags(page.markup, "link"):
            if (_attr(tag, "rel") or "").lower() == "canonical":
                href = _attr(tag, "href") or ""
        if not href:
            missing.append(page.path)
        elif not _is_external(href):
            relative.append(f"{page.path}: {href}")
    if missing:
        return _no(
            "D5",
            f"{len(missing)} {_plural(len(missing), 'page')} without rel=canonical",
            _sample(missing),
        )
    if relative:
        return _no(
            "D5", f"{len(relative)} canonical URLs are relative, not absolute", _sample(relative)
        )
    return _ok(
        "D5", f"all {len(_content_pages(ev))} content pages declare an absolute canonical URL"
    )


@implements("D6")
def _d6_sitemap(ev: SiteEvidence) -> Finding:
    sitemap = ev.file("sitemap.xml")
    if not sitemap:
        return _no("D6", "no sitemap.xml in the build")
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", sitemap)
    if not urls:
        return _no("D6", "sitemap.xml contains no <loc> entries")
    if "<urlset" not in sitemap and "<sitemapindex" not in sitemap:
        return _no("D6", "sitemap.xml has no <urlset> or <sitemapindex> root element")
    return _ok("D6", f"sitemap.xml lists {len(urls)} {_plural(len(urls), 'URL')}")


@implements("D7")
def _d7_robots(ev: SiteEvidence) -> Finding:
    robots = ev.file("robots.txt")
    if not robots:
        return _no("D7", "no robots.txt in the build")
    if not re.search(r"^\s*user-agent\s*:", robots, re.I | re.M):
        return _no("D7", "robots.txt declares no User-agent line")
    if not re.search(r"^\s*sitemap\s*:", robots, re.I | re.M):
        return _no("D7", "robots.txt does not point crawlers at the sitemap")
    return _ok("D7", "robots.txt declares a User-agent and references the sitemap")


@implements("D8")
def _d8_crawlable(ev: SiteEvidence) -> Finding:
    robots = ev.file("robots.txt")
    if not robots:
        return _na("D8", "no robots.txt, so nothing is disallowed by one")
    blanket = re.search(r"^\s*disallow\s*:\s*/\s*$", robots, re.I | re.M)
    return _judge(
        "D8",
        not blanket,
        "robots.txt disallows nothing site-wide",
        "robots.txt contains `Disallow: /` — the whole site is closed to crawlers",
    )


@implements("D9")
def _d9_no_accidental_noindex(ev: SiteEvidence) -> Finding:
    blocked = []
    for page in _content_pages(ev):
        robots_meta = _meta(page.markup, "robots") or _meta(page.markup, "googlebot")
        if "noindex" in robots_meta.lower():
            blocked.append(f"{page.path}: meta robots={robots_meta!r}")
    if ev.http is not None:
        for url, response in ev.http.responses.items():
            if "noindex" in response.header("x-robots-tag").lower():
                blocked.append(f"{url}: X-Robots-Tag")
    return _judge(
        "D9",
        not blocked,
        f"no content page carries a noindex directive "
        f"({len(_content_pages(ev))} checked; error pages excluded)",
        f"{len(blocked)} {_plural(len(blocked), 'page')} are set to noindex",
        _sample(blocked),
    )


@implements("D10")
def _d10_internal_linking(ev: SiteEvidence) -> Finding:
    if len(ev.pages) < 2:
        return _na("D10", "a single-page site has no internal link graph")
    linked: set[str] = set()
    for page in ev.pages:
        for href, _label in _links(page.markup):
            if href and not _is_external(href) and not href.startswith(("#", "mailto:", "tel:")):
                linked.add(href.split("#")[0].split("?")[0].lstrip("./").lstrip("/"))
    orphans = [
        p.path
        for p in _content_pages(ev)
        if not p.is_home
        and p.path.lstrip("./") not in linked
        and p.path.rsplit("/", 1)[-1] not in linked
    ]
    return _judge(
        "D10",
        not orphans,
        f"every one of {len(ev.pages)} pages is linked from within the site",
        f"{len(orphans)} orphaned {_plural(len(orphans), 'page')} nothing links to",
        _sample(orphans),
    )


@implements("D11")
def _d11_anchor_text(ev: SiteEvidence) -> Finding:
    # C12 owns the accessibility reading of this; here it is the SEO one, so
    # bare URLs as anchor text count against it too.
    poor = []
    total = 0
    for page in ev.pages:
        for href, label in _links(page.markup):
            if not href or href.startswith(("#", "javascript:", "tel:", "mailto:")):
                continue
            total += 1
            cleaned = label.strip().lower()
            if cleaned in _VAGUE_LINKS or cleaned.startswith(("http://", "https://", "www.")):
                poor.append(f"{page.path}: {label[:40]!r}")
    if not total:
        return _na("D11", "the site has no navigable links to describe")
    return _judge(
        "D11",
        not poor,
        f"all {total} anchors use descriptive text",
        f"{len(poor)} of {total} anchors use generic text or a bare URL",
        _sample(poor),
    )


_OG_REQUIRED = ("og:title", "og:description", "og:image", "og:url", "og:type")


@implements("D12")
def _d12_social_cards(ev: SiteEvidence) -> Finding:
    problems = []
    for page in ev.pages:
        missing = [p for p in _OG_REQUIRED if not _meta(page.markup, p, attr="property")]
        if missing:
            problems.append(f"{page.path}: missing {', '.join(missing)}")
        if not (
            _meta(page.markup, "twitter:card") or _meta(page.markup, "twitter:card", "property")
        ):
            problems.append(f"{page.path}: no twitter:card")
    return _judge(
        "D12",
        not problems,
        f"all {len(ev.pages)} pages carry complete Open Graph and Twitter card metadata",
        f"{len(problems)} social-card {_plural(len(problems), 'gap')}",
        _sample(problems),
    )


@implements("D13")
def _d13_structured_data(ev: SiteEvidence) -> Finding:
    import json

    blocks = re.findall(
        r"""<script\b[^>]*type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
        ev.markup,
        re.S | re.I,
    )
    if not blocks:
        return _no("D13", "no JSON-LD structured data anywhere on the site")
    types = []
    for raw in blocks:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return _no("D13", f"JSON-LD does not parse: {exc}", (raw.strip()[:120],))
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                return _no("D13", "a JSON-LD block is not an object")
            if "@context" not in node:
                return _no("D13", "a JSON-LD block declares no @context")
            types.append(str(node.get("@type", "")))
    if not any(types):
        return _no("D13", f"{len(blocks)} JSON-LD blocks with no @type")
    return _ok(
        "D13", f"{len(blocks)} valid JSON-LD {_plural(len(blocks), 'block')}: {', '.join(types)}"
    )


_OPAQUE_FILENAME = re.compile(
    r"^(?:img|image|photo|dsc|pxl|screenshot|untitled)?[\W_]*\d{3,}", re.I
)


@implements("D14")
def _d14_image_naming(ev: SiteEvidence) -> Finding:
    problems = []
    total = 0
    for page in ev.pages:
        for tag in _images(page):
            src = _attr(tag, "src") or ""
            if not src or src.startswith("data:"):
                continue
            total += 1
            name = src.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if _OPAQUE_FILENAME.match(name):
                problems.append(f"{page.path}: {src.rsplit('/', 1)[-1]}")
    if not total:
        return _na("D14", "the site references no image files")
    return _judge(
        "D14",
        not problems,
        f"all {total} image filenames are descriptive",
        f"{len(problems)} of {total} images use camera or placeholder filenames",
        _sample(problems),
    )


@implements("D15")
def _d15_search_console(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or record.search_console_verified is None:
        return _out(
            "D15",
            "Search Console verification lives in a Google account; state it in the "
            "business record or confirm it in the console",
        )
    return _judge(
        "D15",
        bool(record.search_console_verified),
        "the business record states the Search Console property is verified",
        "the business record states Search Console is not yet verified",
    )


# ─────────────────────────────────────────────────────────────────────────────
# E. Local SEO
#
# E1, E2, E3 and E5 compare the site against the business record. Without a
# record the auditor can see that a phone number is formatted as a tel: link
# but not that it is the right number, so those checks require RECORD evidence
# and stay outstanding until one is supplied.
# ─────────────────────────────────────────────────────────────────────────────

_DIGITS = re.compile(r"\D+")


def _digits(value: str) -> str:
    return _DIGITS.sub("", value)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


@implements("E1")
def _e1_business_name(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or not record.name.strip():
        return _out("E1", "the business record states no registered name to compare against")
    wanted = _normalise_text(record.name)
    pages = [p.path for p in ev.pages if wanted in _normalise_text(p.text)]
    return _judge(
        "E1",
        len(pages) == len(ev.pages),
        f"the registered name {record.name!r} appears on all {len(ev.pages)} pages",
        f"the registered name {record.name!r} is missing from "
        f"{len(ev.pages) - len(pages)} of {len(ev.pages)} pages",
        _sample([p.path for p in ev.pages if p.path not in pages]),
    )


@implements("E2")
def _e2_address(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or not record.address_parts:
        return _out("E2", "the business record states no address to compare against")
    haystack = _normalise_text(" ".join(p.text for p in ev.pages))
    missing = [part for part in record.address_parts if _normalise_text(part) not in haystack]
    return _judge(
        "E2",
        not missing,
        f"the full address is published: {', '.join(record.address_parts)}",
        f"{len(missing)} address {_plural(len(missing), 'part')} never appear on the site",
        tuple(missing),
    )


@implements("E3")
def _e3_phone(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or not record.phone.strip():
        return _out("E3", "the business record states no phone number to compare against")
    wanted = _digits(record.phone)
    tel_links = [
        _attr(t, "href") or ""
        for page in ev.pages
        for t in _tags(page.markup, "a")
        if (_attr(t, "href") or "").lower().startswith("tel:")
    ]
    matching = [href for href in tel_links if wanted.endswith(_digits(href)[-9:] or "\0")]
    if not tel_links:
        return _no("E3", f"the number {record.phone} is never published as a tel: link")
    if not matching:
        return _no(
            "E3",
            f"tel: links do not match the registered number {record.phone}",
            _sample(sorted(set(tel_links))),
        )
    return _ok("E3", f"{len(matching)} tel: {_plural(len(matching), 'link')} match {record.phone}")


@implements("E4")
def _e4_opening_hours(ev: SiteEvidence) -> Finding:
    structured = re.search(r'"openingHours(?:Specification)?"', ev.markup, re.I)
    visible = re.search(
        r"\b(?:mon|tue|wed|thu|fri|sat|sun|δευ|τρι|τετ|πεμ|παρ|σαβ|κυρ)\w*\b[^<]{0,40}"
        r"\d{1,2}[:.]\d{2}",
        " ".join(p.text for p in ev.pages),
        re.I,
    )
    if not visible:
        return _no("E4", "no opening hours are published in the page content")
    if not structured:
        return _no("E4", "opening hours appear in the copy but not in structured data")
    return _ok("E4", "opening hours are published in the copy and in openingHours structured data")


@implements("E5")
def _e5_google_business_profile(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or not record.google_business_profile.strip():
        return _out(
            "E5",
            "claim status lives in the Google Business Profile account; record the "
            "profile URL once it is claimed",
        )
    linked = record.google_business_profile.split("://")[-1][:40] in ev.markup
    return _judge(
        "E5",
        linked,
        "the claimed Google Business Profile is linked from the site",
        f"the record names a profile ({record.google_business_profile}) that the site never links",
    )


@implements("E6")
def _e6_map(ev: SiteEvidence) -> Finding:
    embedded = re.search(
        r"<iframe\b[^>]*\b(?:google\.com/maps|openstreetmap|maps\.)", ev.markup, re.I
    )
    linked = re.search(
        r"""href\s*=\s*["'][^"']*(?:google\.[a-z.]+/maps|maps\.app|openstreetmap)""",
        ev.markup,
        re.I,
    )
    return _judge(
        "E6",
        bool(embedded or linked),
        "a map to the location is " + ("embedded" if embedded else "linked"),
        "no embedded or linked map — a visitor cannot start navigating in one tap",
    )


@implements("E7")
def _e7_service_area(ev: SiteEvidence) -> Finding:
    text = " ".join(p.text for p in ev.pages)
    structured = re.search(r'"areaServed"', ev.markup, re.I)
    stated = re.search(
        r"\b(?:areas? served|serving|we serve|catchment|περιοχ|εξυπηρετ)\w*", text, re.I
    )
    return _judge(
        "E7",
        bool(structured or stated),
        "the site states which areas it serves" + (" (areaServed in schema)" if structured else ""),
        "the site never says which areas or neighbourhoods it serves",
    )


@implements("E8")
def _e8_keyword_stuffing(ev: SiteEvidence) -> Finding:
    record = ev.record
    city = (record.city if record else "").strip()
    if not city:
        # Fall back to the structured data, so the check still runs without a
        # record. The business name comes from the same place for the same
        # reason: reading the locality but not the name is what made a practice
        # called "Thessaloniki Dental" look like it was stuffing "Thessaloniki".
        found = re.search(r'"addressLocality"\s*:\s*"([^"]+)"', ev.markup)
        city = found.group(1) if found else ""
    if not city:
        return _out("E8", "no locality is known, so density cannot be measured against one")
    worst = ("", 0.0, 0)
    measured = False
    for page in ev.pages:
        words = page.text.split()
        if len(words) < 50:
            continue
        measured = True
        # Occurrences inside the business's own name are branding. A practice
        # called "Thessaloniki Dental" repeating its name in header and footer
        # is not stuffing the locality, and counting it as such punishes every
        # business named after where it is.
        body = page.text
        brand = (record.name if record else "").strip()
        if not brand:
            named = re.search(r'"name"\s*:\s*"([^"]+)"', ev.markup)
            brand = named.group(1) if named else ""
        if brand and city.casefold() in brand.casefold():
            body = re.sub(re.escape(brand), " ", body, flags=re.I)
        hits = len(re.findall(re.escape(city), body, re.I))
        density = 100.0 * hits / len(words)
        if density > worst[1]:
            worst = (page.path, density, hits)
    if not measured:
        return _na("E8", "no page carries enough copy to measure keyword density")
    if not worst[0]:
        # Measured, and the locality never appears often enough to register.
        return _ok("E8", f"{city!r} is used sparingly — no page shows locality repetition")
    return _judge(
        "E8",
        worst[1] <= 2.5,
        f"peak locality density is {worst[1]:.2f}% ({worst[2]}x on {worst[0]}) — reads naturally",
        f"{city!r} appears {worst[2]} times on {worst[0]} ({worst[1]:.2f}% of words) — stuffing",
    )


@implements("E9")
def _e9_nap_consistency(ev: SiteEvidence) -> Finding:
    phones = set()
    for page in ev.pages:
        for tag in _tags(page.markup, "a"):
            href = _attr(tag, "href") or ""
            if href.lower().startswith("tel:"):
                phones.add(_digits(href))
    schema_phones = {_digits(p) for p in re.findall(r'"telephone"\s*:\s*"([^"]+)"', ev.markup)}
    all_phones = {p for p in phones | schema_phones if p}
    if not all_phones:
        return _no("E9", "no phone number is published anywhere, in copy or in schema")
    if len(all_phones) > 1:
        return _no(
            "E9",
            f"{len(all_phones)} different phone numbers across the site and its schema",
            _sample(sorted(all_phones)),
        )
    localities = {
        _normalise_text(v) for v in re.findall(r'"addressLocality"\s*:\s*"([^"]+)"', ev.markup)
    }
    if len(localities) > 1:
        return _no(
            "E9",
            f"{len(localities)} different localities in structured data",
            _sample(sorted(localities)),
        )
    return _ok("E9", "one phone number and one locality, consistent across copy and schema")


@implements("E10")
def _e10_local_structured_data(ev: SiteEvidence) -> Finding:
    import json

    blocks = re.findall(
        r"""<script\b[^>]*type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
        ev.markup,
        re.S | re.I,
    )
    for raw in blocks:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("@type", ""))
            if "LocalBusiness" not in node_type and node_type not in {
                "Dentist",
                "MedicalClinic",
                "Physician",
                "Store",
                "Restaurant",
                "ProfessionalService",
            }:
                continue
            missing = [key for key in ("address", "telephone", "geo") if not node.get(key)]
            return _judge(
                "E10",
                not missing,
                f"{node_type} JSON-LD with address, telephone and geo coordinates",
                f"{node_type} JSON-LD is missing {', '.join(missing)}",
            )
    return _no("E10", "no LocalBusiness-family JSON-LD on the site")


# ─────────────────────────────────────────────────────────────────────────────
# F. UX & Conversion
#
# F7 (do the forms actually deliver) has no implementation. A form that renders
# is not a form that arrives; someone has to send one and watch it land.
# ─────────────────────────────────────────────────────────────────────────────

_CTA_WORDS = re.compile(
    r"\b(book|appointment|call|contact|enquir|inquir|quote|schedule|"
    r"ραντεβού|ραντεβου|κλείσ|καλέστε|επικοινων)\w*",
    re.I,
)


def _home(ev: SiteEvidence) -> Page | None:
    for page in ev.pages:
        if page.is_home:
            return page
    return ev.pages[0] if ev.pages else None


@implements("F1")
def _f1_value_proposition(ev: SiteEvidence) -> Finding:
    page = _home(ev)
    if page is None:
        return _out("F1", "no page to read")
    above = page.markup[: page.markup.find("</section>") + 1] or page.markup[:4000]
    heading = next((t for level, t in _headings(above) if level <= 2), "")
    supporting = len(_text(above)) > len(heading) + 40
    return _judge(
        "F1",
        bool(heading) and supporting,
        f"the first screen leads with {heading[:60]!r} and supporting copy",
        "the first screen carries no heading and supporting sentence a visitor can read in one pass",
    )


@implements("F2")
def _f2_primary_cta(ev: SiteEvidence) -> Finding:
    page = _home(ev)
    if page is None:
        return _out("F2", "no page to read")
    ctas = [
        label
        for href, label in _links(page.markup)
        if _CTA_WORDS.search(label) or _CTA_WORDS.search(href)
    ]
    ctas += [
        _text(b)
        for b in re.findall(r"<button\b[^>]*>(.*?)</button>", page.markup, re.S | re.I)
        if _CTA_WORDS.search(_text(b))
    ]
    styled = re.search(r"\.(?:btn|button|cta)[\w-]*\s*(?:,|\{)", ev.styles, re.I)
    if not ctas:
        return _no("F2", "no call to action on the home page")
    if not styled:
        return _no("F2", f"{len(ctas)} calls to action, none styled as a distinct button class")
    return _ok("F2", f"{len(ctas)} calls to action, styled distinctly: {ctas[0][:40]!r}")


_MIN_TAP_TARGET_PX = 44


@implements("F3")
def _f3_mobile_cta(ev: SiteEvidence) -> Finding:
    css = ev.styles
    sizes = [
        float(v)
        for v in re.findall(
            r"\.(?:btn|button|cta)[^{}]*\{[^}]*?min-height\s*:\s*([\d.]+)px", css, re.I
        )
    ]
    rem_sizes = [
        float(v) * 16
        for v in re.findall(
            r"\.(?:btn|button|cta)[^{}]*\{[^}]*?min-height\s*:\s*([\d.]+)rem", css, re.I
        )
    ]
    padded = re.search(r"\.(?:btn|button|cta)[^{}]*\{[^}]*?padding\s*:", css, re.I)
    all_sizes = sizes + rem_sizes
    if all_sizes:
        smallest = min(all_sizes)
        return _judge(
            "F3",
            smallest >= _MIN_TAP_TARGET_PX,
            f"the primary action is at least {smallest:.0f}px tall on a phone",
            f"the primary action is {smallest:.0f}px tall, under the {_MIN_TAP_TARGET_PX}px "
            "minimum tap target",
        )
    if padded:
        return _out(
            "F3",
            "button height comes from padding, whose computed value only a browser "
            "can resolve — measure the tap target on a device",
        )
    return _no("F3", "no button sizing at all: tap targets are whatever the text happens to be")


@implements("F4")
def _f4_tap_to_call(ev: SiteEvidence) -> Finding:
    tel_links = [
        t
        for page in ev.pages
        for t in _tags(page.markup, "a")
        if (_attr(t, "href") or "").lower().startswith("tel:")
    ]
    plain = re.findall(
        r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)", " ".join(p.text for p in ev.pages)
    )
    if not tel_links:
        return _no(
            "F4",
            f"no tel: links; {len(plain)} phone-shaped {_plural(len(plain), 'string')} in the copy "
            "would have to be retyped by hand",
        )
    return _ok("F4", f"{len(tel_links)} tap-to-call {_plural(len(tel_links), 'link')}")


@implements("F5")
def _f5_booking_route(ev: SiteEvidence) -> Finding:
    page = _home(ev)
    if page is None:
        return _out("F5", "no page to read")
    routes = [
        (href, label)
        for href, label in _links(page.markup)
        if _CTA_WORDS.search(label) or _CTA_WORDS.search(href)
    ]
    has_form = re.search(r"<form\b", ev.markup, re.I)
    if not routes and not has_form:
        return _no("F5", "no booking route: no booking link and no form anywhere on the site")
    return _ok(
        "F5",
        f"{len(routes)} booking {_plural(len(routes), 'route')} from the home page"
        + (" plus an on-site form" if has_form else ""),
    )


@implements("F6")
def _f6_contact_everywhere(ev: SiteEvidence) -> Finding:
    missing = []
    for page in ev.pages:
        footer = re.search(r"<footer\b.*?</footer>", page.markup, re.S | re.I)
        scope = footer.group(0) if footer else page.markup
        has_phone = "tel:" in scope.lower()
        has_contact = bool(_CTA_WORDS.search(scope)) or "mailto:" in scope.lower()
        if not (has_phone and has_contact):
            missing.append(page.path)
    return _judge(
        "F6",
        not missing,
        f"every one of {len(ev.pages)} pages carries a phone number and a contact route",
        f"{len(missing)} {_plural(len(missing), 'page')} lack a phone number or contact route",
        _sample(missing),
    )


@implements("F8")
def _f8_form_errors(ev: SiteEvidence) -> Finding:
    if not re.search(r"<form\b", ev.markup, re.I):
        return _na("F8", "the site has no forms, so there are no error messages to write")
    friendly = re.search(
        r"(?:setCustomValidity|aria-(?:invalid|describedby|errormessage)|"
        r"class\s*=\s*[\"'][^\"']*error)",
        ev.markup + ev.scripts,
        re.I,
    )
    native_only = re.search(r"\brequired\b", ev.markup, re.I)
    if friendly:
        return _ok("F8", "form errors are surfaced through custom messaging and ARIA")
    if native_only:
        return _no(
            "F8",
            "validation relies entirely on the browser's default messages, which say "
            "'invalid input' rather than what to do",
        )
    return _no("F8", "forms have no validation or error messaging at all")


@implements("F9")
def _f9_submission_confirmed(ev: SiteEvidence) -> Finding:
    if not re.search(r"<form\b", ev.markup, re.I):
        return _na("F9", "the site has no forms to confirm")
    confirmation = re.search(
        r"(?:thank|success|received|confirm|ευχαριστ|επιτυχ|λάβαμε)", ev.markup + ev.scripts, re.I
    )
    redirect = re.search(r"""<form\b[^>]*\baction\s*=\s*["'][^"']+""", ev.markup, re.I)
    return _judge(
        "F9",
        bool(confirmation),
        "a success state is present for form submission",
        "no success state: the form "
        + ("posts away with nothing confirming receipt" if redirect else "does nothing visible"),
    )


@implements("F10")
def _f10_trust_signals(ev: SiteEvidence) -> Finding:
    text = " ".join(p.text for p in ev.pages)
    signals = []
    # Quoted praise is structural: <blockquote> with a <cite> is a testimonial
    # whether or not the heading above it uses the word. Keyword-only detection
    # missed a real testimonials section headed "What patients say".
    if re.search(r"<blockquote\b", ev.markup, re.I) or re.search(r"<cite\b", ev.markup, re.I):
        signals.append("quoted testimonials")
    elif re.search(r"(?:review|testimonial|μαρτυρ|κριτικ)\w*", text, re.I):
        signals.append("reviews")
    if re.search(r"(?:registration|licence|license|ΑΦΜ|αρ\.?\s*μητρώου|reg\.? no)", text, re.I):
        signals.append("registration details")
    if any(
        "team" in (_attr(t, "src") or "").lower()
        or "staff" in (_attr(t, "src") or "").lower()
        or "dr" in (_attr(t, "alt") or "").lower()
        for p in ev.pages
        for t in _images(p)
    ):
        signals.append("photographs of the people")
    # "Fifteen years placing implants" and "practising since 2009" are both
    # experience claims. Requiring the literal word "experience" next to "years"
    # missed every natural way of writing one.
    if re.search(
        r"\b(?:\d+\+?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\s+"
        r"(?:years?|έτη|χρόνια)\b|\b(?:since|από το)\s+(?:19|20)\d{2}\b",
        text,
        re.I,
    ):
        signals.append("stated experience")
    return _judge(
        "F10",
        len(signals) >= 2,
        f"trust signals present: {', '.join(signals)}",
        f"only {len(signals)} trust signal{'s' if len(signals) != 1 else ''} "
        f"({', '.join(signals) or 'none'}) — a cautious visitor has little to go on",
    )


@implements("F11")
def _f11_credentials_findable(ev: SiteEvidence) -> Finding:
    text = " ".join(p.text for p in ev.pages)
    found = re.search(
        r"(?:DDS|DMD|BDS|MSc|PhD|MD|specialist|qualifi\w+|graduat\w+|"
        r"πτυχ\w+|ειδικ\w+|μεταπτυχ\w+)",
        text,
        re.I,
    )
    if not found:
        return _no("F11", "no qualifications or registrations stated anywhere a visitor would look")
    return _ok("F11", f"professional credentials are stated in the copy ({found.group(0)!r})")


@implements("F12")
def _f12_services_structured(ev: SiteEvidence) -> Finding:
    headings = [t for level, t in _headings(ev.markup) if level in (2, 3)]
    lists = len(_tags(ev.markup, "li"))
    if len(headings) < 3 and lists < 4:
        return _no(
            "F12",
            f"services are not broken up: {len(headings)} subheadings and {lists} list items "
            "across the whole site",
        )
    return _ok(
        "F12", f"{len(headings)} subheadings and {lists} list items make the services scannable"
    )


@implements("F13")
def _f13_pricing(ev: SiteEvidence) -> Finding:
    text = " ".join(p.text for p in ev.pages)
    money = re.search(r"(?:€|EUR|\$|£)\s?\d|(?:\d+\s?(?:€|ευρώ))", text, re.I)
    policy = re.search(
        r"(?:pricing|price list|τιμ(?:ή|ές|οκατάλογος)|κόστος|free (?:consultation|quote)|"
        r"δωρεάν)\w*",
        text,
        re.I,
    )
    return _judge(
        "F13",
        bool(money or policy),
        "pricing is addressed" + (" with figures" if money else " as a stated policy"),
        "pricing is never addressed — the most common reason a visitor leaves to call a competitor",
    )


@implements("F14")
def _f14_faq(ev: SiteEvidence) -> Finding:
    has_schema = re.search(r'"@type"\s*:\s*"FAQPage"', ev.markup, re.I)
    heading = re.search(r"\b(?:FAQ|frequently asked|συχνές ερωτήσεις)\b", ev.markup, re.I)
    questions = len(re.findall(r"<(?:summary|dt|h[23])\b[^>]*>[^<]*\?", ev.markup, re.I))
    if not (has_schema or heading or questions >= 3):
        return _no("F14", "no FAQ section: the questions visitors phone to ask go unanswered")
    return _ok(
        "F14",
        f"an FAQ is present ({questions} question-shaped headings"
        + (", marked up as FAQPage" if has_schema else "")
        + ")",
    )


@implements("F15")
def _f15_conversion_friction(ev: SiteEvidence) -> Finding:
    forms = re.findall(r"<form\b.*?</form>", ev.markup, re.S | re.I)
    if not forms:
        return _na("F15", "the site has no forms, so there are no fields to trim")
    worst = 0
    for form in forms:
        required = len(
            [
                t
                for t in _tags(form, "input") + _tags(form, "select") + _tags(form, "textarea")
                if _has(t, "required") and (_attr(t, "type") or "").lower() not in _LABELLESS_TYPES
            ]
        )
        worst = max(worst, required)
    return _judge(
        "F15",
        worst <= 4,
        f"the longest form asks for {worst} required fields",
        f"the longest form demands {worst} required fields before it will accept an enquiry",
    )


# ─────────────────────────────────────────────────────────────────────────────
# G. Security & Privacy
#
# G1-G3 need a live response and are implemented in the HTTP section below:
# a _headers file in a build directory declares an intention, and an intention
# is not a served header. G5 (admin 2FA) is answered from the business record,
# never from the repository — credentials do not belong in code. G10 (analytics
# firing only after consent) needs a browser to watch what actually fired.
# ─────────────────────────────────────────────────────────────────────────────

_ADMIN_PATHS = re.compile(
    r"^(?:admin|wp-admin|administrator|dashboard|cms|staging|backup|\.env|"
    r"config\.(?:php|json|yaml|yml))",
    re.I,
)


@implements("G4")
def _g4_admin_areas(ev: SiteEvidence) -> Finding:
    exposed = [
        path
        for path in list(ev.assets) + list(ev.files)
        if _ADMIN_PATHS.match(path.split("/")[0]) or _ADMIN_PATHS.match(path)
    ]
    secrets = [
        path
        for path in ev.files
        if re.search(r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S{8,}", ev.files[path], re.I)
    ]
    if secrets:
        return _no(
            "G4",
            f"{len(secrets)} shipped {_plural(len(secrets), 'file')} contain credential-shaped values",
            _sample(secrets),
        )
    if exposed:
        return _no(
            "G4",
            f"{len(exposed)} administrative or configuration {_plural(len(exposed), 'path')} "
            "are part of the published build",
            _sample(exposed),
        )
    return _ok("G4", "no administrative paths, config files or credentials in the published build")


@implements("G5")
def _g5_admin_authentication(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None or record.admin_2fa_enabled is None:
        return _out(
            "G5",
            "administrative authentication lives in the hosting and CMS accounts; confirm "
            "2FA there and state it in the business record — never store credentials in the repo",
        )
    return _judge(
        "G5",
        bool(record.admin_2fa_enabled),
        "the business record states 2FA is enabled on administrative accounts",
        "the business record states administrative accounts have no 2FA",
    )


@implements("G6")
def _g6_third_party_integrity(ev: SiteEvidence) -> Finding:
    problems = []
    for tag in _tags(ev.markup, "script"):
        src = _attr(tag, "src") or ""
        if not src or not _is_external(src):
            continue
        if not _has(tag, "integrity"):
            problems.append(f"no SRI: {src[:70]}")
        elif not _has(tag, "crossorigin"):
            problems.append(f"integrity without crossorigin (ignored by browsers): {src[:60]}")
        if re.search(r"/(?:latest|main|master|v?\d+)(?:/|\.min\.js)", src) and "@" not in src:
            problems.append(f"unpinned version: {src[:70]}")
    for tag in _tags(ev.markup, "link"):
        href = _attr(tag, "href") or ""
        if (_attr(tag, "rel") or "").lower() == "stylesheet" and _is_external(href):
            if not _has(tag, "integrity") and "fonts.googleapis.com" not in href:
                problems.append(f"no SRI on external stylesheet: {href[:60]}")
    externals = sum(1 for t in _tags(ev.markup, "script") if _is_external(_attr(t, "src") or ""))
    if not externals and not problems:
        return _ok("G6", "no third-party code is loaded at runtime — nothing to pin")
    return _judge(
        "G6",
        not problems,
        f"all {externals} third-party resources are pinned with subresource integrity",
        f"{len(problems)} third-party supply-chain {_plural(len(problems), 'gap')}",
        _sample(problems),
    )


@implements("G7")
def _g7_form_protection(ev: SiteEvidence) -> Finding:
    if not re.search(r"<form\b", ev.markup, re.I):
        return _na("G7", "the site has no public forms to protect")
    combined = ev.markup + ev.scripts
    protections = []
    if re.search(r"(?:recaptcha|hcaptcha|turnstile|friendly-?challenge)", combined, re.I):
        protections.append("captcha")
    if re.search(r"""(?:name\s*=\s*["'](?:_?honey|_gotcha|bot-field)|honeypot)""", combined, re.I):
        protections.append("honeypot")
    if re.search(r"(?:csrf|authenticity_token|_token)", combined, re.I):
        protections.append("CSRF token")
    if re.search(r"rate[_-]?limit", combined, re.I):
        protections.append("rate limiting")
    return _judge(
        "G7",
        bool(protections),
        f"form abuse protection in place: {', '.join(protections)}",
        "public forms with no captcha, honeypot or rate limiting — they will be found by bots",
    )


@implements("G8")
def _g8_privacy_policy(ev: SiteEvidence) -> Finding:
    linked = [
        (href, label)
        for page in ev.pages
        for href, label in _links(page.markup)
        if re.search(r"(?:privacy|gdpr|απόρρητ|προσωπικ[άώ]ν δεδομ)", href + " " + label, re.I)
    ]
    if not linked:
        return _no("G8", "no privacy policy is linked from anywhere on the site")
    target = linked[0][0].split("#")[0].lstrip("./").lstrip("/")
    exists = any(
        target and (target in p.path or p.path.endswith(target)) for p in ev.pages
    ) or _is_external(linked[0][0])
    return _judge(
        "G8",
        exists,
        f"a privacy policy is linked from {len({p for p in ev.pages if _links(p.markup)})} pages",
        f"the privacy link points at {linked[0][0]!r}, which is not in the build",
    )


@implements("G9")
def _g9_cookie_consent(ev: SiteEvidence) -> Finding:
    combined = ev.markup + ev.scripts
    trackers = re.findall(
        r"(?:googletagmanager|google-analytics|gtag\(|fbq\(|hotjar|clarity\.ms|"
        r"matomo|plausible|segment\.com|doubleclick)",
        combined,
        re.I,
    )
    sets_cookies = re.search(r"document\.cookie\s*=", combined, re.I)
    if not trackers and not sets_cookies:
        return _na(
            "G9",
            "the site loads no analytics or advertising script and sets no cookies in the "
            "page, so no consent is required",
        )
    consent = re.search(
        r"(?:cookie[-_]?(?:consent|banner|notice)|consent(?:Mode|Manager)|cookiebot|"
        r"osano|klaro|didomi|συγκατάθεση)",
        combined,
        re.I,
    )
    return _judge(
        "G9",
        bool(consent),
        f"a consent mechanism is present alongside {len(set(trackers))} tracking integrations",
        f"{len(set(trackers))} tracking {_plural(len(set(trackers)), 'integration')} load with "
        "no consent mechanism anywhere on the site",
        _sample(sorted(set(trackers))),
    )


# ─────────────────────────────────────────────────────────────────────────────
# H. Content & Professional Quality
#
# Only H1 is implementable. H2-H4 and H6-H10 ask whether something is TRUE —
# whether a photograph is of this practice, whether a testimonial was written
# by a real patient, whether a credential was earned. A tool can read the claim
# and cannot verify it, and a tool that scored these as passes would be
# manufacturing exactly the false assurance the standard exists to prevent.
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDERS = (
    (r"\blorem ipsum\b", "lorem ipsum"),
    (r"\bdolor sit amet\b", "lorem ipsum"),
    (r"\b(?:TODO|FIXME|TBD|TBC)\b", "unfinished-work marker"),
    (r"\bYour (?:Name|Company|Business|Text) Here\b", "template placeholder"),
    (r"\[(?:insert|placeholder|your)[^\]]{0,40}\]", "bracketed placeholder"),
    (r"\bXXX+\b", "placeholder marker"),
    (r"\bexample\.com\b", "example.com"),
    (r"\b(?:123[- ]?456[- ]?7890|555[- ]?\d{4})\b", "placeholder phone number"),
)

_UNRESOLVED_TEMPLATE = re.compile(
    r"\$\{?[a-z_][a-z0-9_]{2,}\}?|\{\{\s*[a-z_][a-z0-9_.]*\s*\}\}", re.I
)


@implements("H1")
def _h1_no_placeholder_text(ev: SiteEvidence) -> Finding:
    hits = []
    visible = " ".join(p.text for p in ev.pages)
    for pattern, label in _PLACEHOLDERS:
        found = re.findall(pattern, visible, re.I)
        if found:
            hits.append(f"{label} x{len(found)} ({found[0][:40]!r})")
    unresolved = _UNRESOLVED_TEMPLATE.findall(visible)
    if unresolved:
        hits.append(
            f"{len(unresolved)} unresolved template "
            f"{_plural(len(unresolved), 'variable')}: {', '.join(sorted(set(unresolved))[:4])}"
        )
    return _judge(
        "H1",
        not hits,
        f"no placeholder text in {len(visible.split())} words of visible copy",
        "placeholder text is still in the published copy: " + "; ".join(hits),
        _sample(hits),
    )


@implements("H5")
def _h5_contact_verified(ev: SiteEvidence) -> Finding:
    record = ev.record
    if record is None:
        return _out("H5", "no business record to check the published contact details against")
    published_phones = {
        _digits(_attr(t, "href") or "")
        for page in ev.pages
        for t in _tags(page.markup, "a")
        if (_attr(t, "href") or "").lower().startswith("tel:")
    } - {""}
    published_emails = {
        (_attr(t, "href") or "")[7:].lower()
        for page in ev.pages
        for t in _tags(page.markup, "a")
        if (_attr(t, "href") or "").lower().startswith("mailto:")
    } - {""}
    problems = []
    if record.phone and not any(_digits(record.phone).endswith(p[-9:]) for p in published_phones):
        problems.append(f"published phone(s) {sorted(published_phones)} != record {record.phone}")
    if record.email and record.email.lower() not in published_emails:
        problems.append(f"published email(s) {sorted(published_emails)} != record {record.email}")
    if not problems and not (record.phone or record.email):
        return _out("H5", "the business record states no phone or email to verify against")
    return _judge(
        "H5",
        not problems,
        "published phone and email match the business record",
        "; ".join(problems),
        _sample(problems),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Live-response checks. These exist only in URL mode, by construction.
# ─────────────────────────────────────────────────────────────────────────────


def _entry_response(ev: SiteEvidence):
    return ev.http.entry() if ev.http is not None else None


@implements("B12")
def _b12_compression(ev: SiteEvidence) -> Finding:
    response = _entry_response(ev)
    if response is None:
        return _out("B12", "no live response to inspect")
    encoding = response.header("content-encoding").lower()
    return _judge(
        "B12",
        encoding in {"br", "gzip", "deflate", "zstd"},
        f"the server compresses HTML with {encoding}",
        f"HTML is served uncompressed (content-encoding: {encoding or 'absent'})",
    )


@implements("B13")
def _b13_caching(ev: SiteEvidence) -> Finding:
    if ev.http is None:
        return _out("B13", "no live responses to inspect")
    static = [
        (url, r)
        for url, r in ev.http.responses.items()
        if url.lower().endswith((".css", ".js", ".webp", ".avif", ".png", ".jpg", ".woff2"))
        and r.status < 400
    ]
    if not static:
        return _out("B13", "no static asset responses were collected to read cache headers from")
    uncached = [
        url
        for url, r in static
        if "max-age" not in r.header("cache-control").lower() and not r.header("etag")
    ]
    return _judge(
        "B13",
        not uncached,
        f"all {len(static)} static assets carry Cache-Control or an ETag",
        f"{len(uncached)} of {len(static)} static assets are served with no caching headers",
        _sample(uncached),
    )


@implements("B14")
def _b14_cdn(ev: SiteEvidence) -> Finding:
    response = _entry_response(ev)
    if response is None:
        return _out("B14", "no live response to inspect")
    markers = [
        name
        for name in (
            "cf-ray",
            "x-vercel-id",
            "x-amz-cf-id",
            "x-served-by",
            "x-cache",
            "fly-request-id",
        )
        if response.header(name)
    ]
    server = response.header("server").lower()
    if markers or any(
        v in server for v in ("cloudflare", "cloudfront", "fastly", "netlify", "vercel")
    ):
        return _ok("B14", f"served from an edge network ({', '.join(markers) or server})")
    return _no(
        "B14",
        f"no CDN or edge-cache headers on the response (server: {server or 'not stated'})",
    )


_REQUIRED_HEADERS = {
    "strict-transport-security": "HSTS",
    "x-content-type-options": "MIME-sniffing protection",
    "referrer-policy": "referrer policy",
    "content-security-policy": "CSP",
}


@implements("G1")
def _g1_https(ev: SiteEvidence) -> Finding:
    response = _entry_response(ev)
    if response is None:
        return _out("G1", "no live response to inspect")
    url = response.final_url or response.url
    if not url.lower().startswith("https://"):
        return _no("G1", f"the site answered over plain HTTP at {url}")
    if response.status >= 400:
        return _no("G1", f"the HTTPS entry point answered {response.status}")
    return _ok("G1", f"served over HTTPS with status {response.status}")


@implements("G2")
def _g2_https_redirect(ev: SiteEvidence) -> Finding:
    if ev.http is None:
        return _out("G2", "no live responses to inspect")
    probe = ev.http.plain_http
    if probe is None:
        return _out("G2", "the plain-HTTP origin was not probed, so the redirect is unconfirmed")
    if probe.status == 0:
        return _ok("G2", "the plain-HTTP origin refuses connections entirely")
    redirected = (probe.final_url or "").lower().startswith("https://")
    return _judge(
        "G2",
        redirected or probe.status in {301, 308},
        f"plain HTTP answers {probe.status} and lands on HTTPS",
        f"plain HTTP answers {probe.status} without redirecting to HTTPS",
    )


@implements("G3")
def _g3_security_headers(ev: SiteEvidence) -> Finding:
    response = _entry_response(ev)
    if response is None:
        return _out("G3", "no live response to inspect")
    missing = [label for header, label in _REQUIRED_HEADERS.items() if not response.header(header)]
    return _judge(
        "G3",
        not missing,
        f"all {len(_REQUIRED_HEADERS)} security headers are served",
        f"{len(missing)} security {_plural(len(missing), 'header')} not served: "
        + ", ".join(missing),
        tuple(missing),
    )
