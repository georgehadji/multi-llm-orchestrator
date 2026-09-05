"""
Reusable site templates — build a vertical once, ship it per client.

The factory generates every site from scratch, so the second dentist costs the
same as the first: same spend, same latency, and no guarantee the two look
related. A template separates what is *the same for the vertical* (structure,
layout, styling, section order) from what is *different per client* (brand,
copy, contact details, palette).

Two directions, both deterministic and free:

    extract   a site you already built  ->  a template + starting client data
    apply     a template + client data  ->  a new site

Substitution is stdlib ``string.Template`` (``$name``) plus one repeat-block
convention for lists::

    <!-- repeat: services -->
    <article><h3>$title</h3><p>$body</p></article>
    <!-- /repeat -->

That is the whole engine. A real template language (Jinja2 and friends) would
be a new dependency to render what is name/value replacement with one loop.

Template layout on disk::

    templates/dentist/
      template.yaml            name, description, sections order, required, defaults
      sections/hero.html       one file per section — this is the modular unit
      sections/services.html
      styles.css               mobile-first
      script.js                optional
      client.example.yaml      written by extract; the file you copy per client
"""

from __future__ import annotations

import html as _html
import re
import shutil
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

__all__ = [
    "SiteTemplate",
    "apply_template",
    "extract_template",
    "list_templates",
    "load_template",
    "render_section",
]

_REPEAT = re.compile(
    r"[ \t]*<!--\s*repeat:\s*(?P<key>[a-z_][a-z0-9_]*)\s*-->\n?"
    r"(?P<body>.*?)"
    r"[ \t]*<!--\s*/repeat\s*-->\n?",
    re.S | re.I,
)

# Where the shipped templates live.
TEMPLATE_ROOT = Path(__file__).resolve().parent.parent.parent / "templates" / "websites"


class _KeepUnknown(dict[str, str]):
    """Render a missing placeholder visibly instead of raising mid-batch.

    A KeyError here would abort a 50-site run over one unset field. A literal
    ``$phone`` left in the page is loud, greppable, and caught by the quality
    gate's content check.
    """

    def __missing__(self, key: str) -> str:
        return "$" + key


def render_section(source: str, values: dict[str, Any]) -> str:
    """Render one section: expand repeat blocks, then substitute ``$name``."""

    def _expand(match: re.Match[str]) -> str:
        key = match.group("key")
        body = match.group("body")
        items = values.get(key) or []
        if not isinstance(items, (list, tuple)):
            items = [items]
        rendered = []
        for item in items:
            scope = dict(values)
            if isinstance(item, dict):
                scope.update(item)
            else:
                scope["item"] = item
            rendered.append(Template(body).safe_substitute(_KeepUnknown(scope)))
        return "".join(rendered)

    expanded = _REPEAT.sub(_expand, source)
    return Template(expanded).safe_substitute(_KeepUnknown(values))


@dataclass
class SiteTemplate:
    """A loaded template: metadata plus the raw source of each section."""

    name: str
    description: str = ""
    sections: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)  # section name -> html
    root: Path | None = None


def load_template(path: str | Path) -> SiteTemplate:
    """Load a template directory. Raises ValueError with a specific reason."""
    import yaml

    root = Path(path)
    manifest = root / "template.yaml"
    if not manifest.is_file():
        raise ValueError(f"{root}: no template.yaml — not a site template")

    meta = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    sections = list(meta.get("sections") or [])
    if not sections:
        raise ValueError(f"{root}: template.yaml declares no sections")

    source: dict[str, str] = {}
    for section in sections:
        section_file = root / "sections" / f"{section}.html"
        if not section_file.is_file():
            raise ValueError(
                f"{root}: section '{section}' is declared in template.yaml but "
                f"sections/{section}.html does not exist"
            )
        source[section] = section_file.read_text(encoding="utf-8")

    return SiteTemplate(
        name=str(meta.get("name") or root.name),
        description=str(meta.get("description") or ""),
        sections=sections,
        required=list(meta.get("required") or []),
        defaults=dict(meta.get("defaults") or {}),
        source=source,
        root=root,
    )


def list_templates(root: str | Path | None = None) -> list[SiteTemplate]:
    """Every template under *root* (defaults to the shipped template directory)."""
    base = Path(root) if root is not None else TEMPLATE_ROOT
    if not base.is_dir():
        return []
    found = []
    for candidate in sorted(base.iterdir()):
        if (candidate / "template.yaml").is_file():
            try:
                found.append(load_template(candidate))
            except ValueError:
                continue
    return found


_PAGE = Template("""<!DOCTYPE html>
<html lang="$lang">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$page_title</title>
  <meta name="description" content="$meta_description">
  <meta name="robots" content="index, follow, max-image-preview:large">
$canonical$geo_meta
  <meta property="og:type" content="website">
  <meta property="og:locale" content="$og_locale">
  <meta property="og:site_name" content="$brand_name">
  <meta property="og:title" content="$page_title">
  <meta property="og:description" content="$meta_description">
  <meta property="og:image" content="$og_image_url">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="$brand_name — $tagline">
$og_url
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="$page_title">
  <meta name="twitter:description" content="$meta_description">
  <meta name="twitter:image" content="$og_image_url">
  <link rel="stylesheet" href="styles.css">
  <script type="application/ld+json">
$json_ld
  </script>
</head>
<body>
$body
$script
</body>
</html>
""")


_OG_IMAGE = Template(
    """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-label="$brand_name">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="$accent"/>
      <stop offset="1" stop-color="$secondary"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="$paper"/>
  <rect width="1200" height="12" fill="url(#g)"/>
  <circle cx="1040" cy="520" r="220" fill="$accent" opacity="0.08"/>
  <text x="80" y="250" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        font-size="78" font-weight="700" fill="$ink">$brand_name</text>
  <text x="80" y="330" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        font-size="38" fill="$ink_soft">$tagline</text>
  <text x="80" y="susp" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        font-size="30" fill="$accent">$city  ·  $phone</text>
</svg>
"""
)


_HEADERS = Template("""/*
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: geolocation=(), microphone=(), camera=(), interest-cohort=()
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  Cross-Origin-Opener-Policy: same-origin
""")


def apply_template(
    template: SiteTemplate,
    client: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Render *template* for *client* into *output_dir*. Returns the directory."""
    missing = [key for key in template.required if not client.get(key)]
    if missing:
        raise ValueError(
            f"template '{template.name}' requires {missing} — "
            f"add them to the client file and re-run"
        )

    values: dict[str, Any] = {**template.defaults, **client}
    # Values may themselves contain placeholders — extraction stores
    # `page_title: "$brand_name — Implants"` so that renaming the clinic is ONE
    # edit, not one per field that mentions it. Resolve those first.
    for _ in range(2):  # two passes is enough for one level of indirection
        values = {
            key: (
                Template(val).safe_substitute(_KeepUnknown(values))
                if isinstance(val, str) and "$" in val
                else val
            )
            for key, val in values.items()
        }
    values.setdefault("lang", "en")
    values.setdefault("brand_name", template.name.replace("-", " ").title())
    # A title of just the brand name is 19 characters and wastes the whole
    # result snippet; a description of just the tagline is too short to say
    # anything. Both defaults are composed from the client's own facts — never
    # invented copy — and an explicit value in the client file always wins.
    # A directions link derived from the address the client already gave us.
    # Nothing is invented: it is their own address handed to a map search.
    if not values.get("map_url"):
        where = ", ".join(
            str(values.get(k, "")).strip() for k in ("address", "city", "country") if values.get(k)
        )
        if where:
            values["map_url"] = (
                "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote_plus(where)
            )
    values.setdefault("page_title", _compose_title(values))
    values.setdefault("meta_description", _compose_description(values))

    # Colour theory: one brand colour in, a full contrast-checked palette out.
    # Explicit `palette:` entries win over the derived values.
    from .website_palette import derive_palette, palette_to_css

    palette: dict[str, str] = {}
    accent = values.get("accent")
    if accent:
        try:
            palette = derive_palette(str(accent), str(values.get("scheme") or "analogous"))
        except ValueError:
            palette = {}
    palette.update(values.get("palette") or {})
    values["_palette"] = palette

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sections = [s for s in template.sections if _section_has_content(s, template, values)]
    body = _assemble_body(sections, template, values)

    script_tag = ""
    if template.root and (template.root / "script.js").is_file():
        shutil.copyfile(template.root / "script.js", out / "script.js")
        script_tag = '<script src="script.js" defer></script>'

    seo = _build_seo(values)
    page = _PAGE.safe_substitute(
        _KeepUnknown({**values, **seo, "body": body, "script": script_tag})
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    _write_supporting_pages(out, template, values, seo)
    _write_discovery_files(out, values, seo)

    if template.root:
        css = template.root / "styles.css"
        if css.is_file():
            # The derived palette rides on top of the template's own tokens, so
            # a client is one colour rather than a forked stylesheet.
            extra = ""
            if palette:
                extra = "\n/* client palette — derived from `accent`, WCAG-checked */\n"
                extra += palette_to_css(palette)
            (out / "styles.css").write_text(
                css.read_text(encoding="utf-8") + extra, encoding="utf-8"
            )
        assets = template.root / "public"
        if assets.is_dir():
            shutil.copytree(assets, out / "public", dirs_exist_ok=True)

    return out


# ── Extraction ───────────────────────────────────────────────────────────────

_SECTION = re.compile(r'<section\b[^>]*\bid="(?P<id>[a-z0-9_-]+)"[^>]*>.*?</section>', re.S | re.I)
# A tel: href is the only unambiguous phone signal in a page; prefer it.
_TEL_HREF = re.compile(r'href=["\']tel:([^"\']+)["\']', re.I)
# Fallback: international form, or grouped digits. Deliberately strict — a loose
# pattern matched the cache-busting timestamp "1606811841689-23" as a phone.
_PHONE = re.compile(r"\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_TAG = re.compile(r"<[^>]+>")


def _text(markup: str) -> str:
    """Plain text of an HTML fragment: tags out, entities decoded, spaces tidy."""
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub(" ", markup))).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "section"


def extract_template(
    site_dir: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Turn a built site into a reusable template plus starting client data.

    Best-effort and deliberately conservative: it parameterises the things that
    are unambiguously per-client (brand name, page title, meta description,
    phone, email) and leaves everything else as-is. The result is a starting
    point to edit, not a finished abstraction — extraction cannot know which
    paragraph is boilerplate and which is this client's story.
    """
    import yaml

    site = Path(site_dir)
    index = site / "index.html"
    if not index.is_file():
        raise ValueError(f"{site}: no index.html to extract from")

    markup = index.read_text(encoding="utf-8", errors="ignore")
    out = Path(output_dir)
    (out / "sections").mkdir(parents=True, exist_ok=True)

    title = _first(r"<title[^>]*>(.*?)</title>", markup)
    description = _first(r'<meta[^>]+name="description"[^>]+content="([^"]*)"', markup)
    heading = _first(r"<h1[^>]*>(.*?)</h1>", markup)
    # _text, not raw: an <h1> containing <br> and <span> otherwise becomes the
    # brand name verbatim, markup and all.
    brand = _text(heading) or _text(title) or site.name
    # "Brand — tagline" is the common shape; the brand is the part before the dash.
    brand = re.split(r"\s+[—–|]\s+", brand)[0].strip()

    tel = _TEL_HREF.search(markup)
    phone = tel.group(1).strip() if tel else _search(_PHONE, _text(markup))
    email = _search(_EMAIL, markup)

    replacements: list[tuple[str, str]] = []
    for literal, placeholder in (
        (brand, "$brand_name"),
        (phone, "$phone"),
        (email, "$email"),
    ):
        if literal:
            replacements.append((literal, placeholder))

    sections: list[str] = []
    for match in _SECTION.finditer(markup):
        section_id = _slug(match.group("id"))
        body = match.group(0)
        for literal, placeholder in replacements:
            body = body.replace(literal, placeholder)
        (out / "sections" / f"{section_id}.html").write_text(body + "\n", encoding="utf-8")
        sections.append(section_id)

    if not sections:
        raise ValueError(
            f"{site}: found no <section id=...> elements to split into modules. "
            f"Templates are built from identifiable sections."
        )

    for asset in ("styles.css", "script.js"):
        source = site / asset
        if source.is_file():
            shutil.copyfile(source, out / asset)
    if (site / "public").is_dir():
        shutil.copytree(site / "public", out / "public", dirs_exist_ok=True)

    template_name = name or _slug(site.name)
    (out / "template.yaml").write_text(
        yaml.safe_dump(
            {
                "name": template_name,
                "description": f"Extracted from {site.name}",
                "sections": sections,
                "required": [k for k in ("brand_name", "phone") if dict(replacements or []) or k],
                "defaults": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    client: dict[str, Any] = {"brand_name": brand}
    if phone:
        client["phone"] = phone
    if email:
        client["email"] = email

    def _parameterise(text: str) -> str:
        """Swap the brand for $brand_name so a rename propagates everywhere."""
        clean = _html.unescape(text.strip())
        return clean.replace(brand, "$brand_name") if brand else clean

    if title:
        client["page_title"] = _parameterise(title)
    if description:
        client["meta_description"] = _parameterise(description)
    (out / "client.example.yaml").write_text(
        "# Copy this per client and edit. Then:\n"
        f"#   python -m orchestrator website-template apply {out} --client <file> -o <site>\n"
        + yaml.safe_dump(client, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out


def _first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.S | re.I)
    return match.group(1).strip() if match else ""


def _search(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


# ── SEO / GEO / social / security ────────────────────────────────────────────


def _esc(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def _build_seo(values: dict[str, Any]) -> dict[str, str]:
    """Structured data, geo meta and social tags derived from the client data.

    Everything here comes from fields a client already has to supply (address,
    phone, city). Nothing is invented: a missing latitude means no geo block
    rather than a zeroed one, because wrong coordinates are worse than none.
    """
    import json

    site_url = str(values.get("site_url") or "").rstrip("/")
    brand = str(values.get("brand_name", ""))

    schema: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": str(values.get("business_type") or "Dentist"),
        "name": brand,
        "description": str(values.get("meta_description") or values.get("tagline") or ""),
        "telephone": str(values.get("phone", "")),
        "email": str(values.get("email", "")),
        "address": {
            "@type": "PostalAddress",
            "streetAddress": str(values.get("address", "")),
            "addressLocality": str(values.get("city", "")),
            "postalCode": str(values.get("postal_code", "")),
            "addressCountry": str(values.get("country") or "GR"),
        },
    }
    if values.get("hours"):
        schema["openingHours"] = str(values["hours"])
    if values.get("price_range"):
        schema["priceRange"] = str(values["price_range"])
    if site_url:
        schema["url"] = site_url
        schema["image"] = f"{site_url}/public/og-image.svg"
    if values.get("area_served") or values.get("city"):
        schema["areaServed"] = str(values.get("area_served") or values["city"])

    lat, lon = values.get("latitude"), values.get("longitude")
    geo_meta = ""
    if lat not in (None, "") and lon not in (None, ""):
        schema["geo"] = {
            "@type": "GeoCoordinates",
            "latitude": float(lat),
            "longitude": float(lon),
        }
        geo_meta = (
            f'\n  <meta name="geo.position" content="{lat};{lon}">'
            f'\n  <meta name="ICBM" content="{lat}, {lon}">'
        )
    if values.get("city"):
        geo_meta = (
            f'\n  <meta name="geo.placename" content="{_esc(values["city"])}">'
            f'\n  <meta name="geo.region" content="{_esc(values.get("country") or "GR")}">'
            + geo_meta
        )

    canonical = f'\n  <link rel="canonical" href="{_esc(site_url)}/">' if site_url else ""
    og_url = f'  <meta property="og:url" content="{_esc(site_url)}/">' if site_url else ""
    og_image_url = f"{site_url}/public/og-image.svg" if site_url else "public/og-image.svg"

    # An FAQ the client actually wrote gets marked up as one, so the answers can
    # surface in search. Nothing is emitted when there are no questions: an empty
    # FAQPage is a structured-data lie.
    graph: Any = schema
    faq = values.get("faq") or []
    entries = [
        {
            "@type": "Question",
            "name": str(item.get("question", "")),
            "acceptedAnswer": {"@type": "Answer", "text": str(item.get("answer", ""))},
        }
        for item in faq
        if isinstance(item, dict) and item.get("question") and item.get("answer")
    ]
    if entries:
        graph = [
            schema,
            {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entries},
        ]

    return {
        "json_ld": json.dumps(graph, indent=4, ensure_ascii=False),
        "geo_meta": geo_meta,
        "canonical": canonical,
        "og_url": og_url,
        "og_image_url": og_image_url,
        "og_locale": str(values.get("og_locale") or "en_GB"),
    }


# Sections that frame the page rather than carry its content. Everything else
# goes inside <main>, which is what makes the skip link land somewhere and what
# a screen reader uses to jump past the navigation.
_CHROME_SECTIONS = ("header", "nav", "footer", "banner")

_REPEAT_KEY = re.compile(r"<!--\s*repeat:\s*([a-z_][a-z0-9_]*)\s*-->", re.I)


def _section_has_content(name: str, template: SiteTemplate, values: dict[str, Any]) -> bool:
    """Whether a section has anything to say once its data is filled in.

    A section is dropped only when it is built ENTIRELY around a repeat block
    whose list is empty — an FAQ heading over nothing is a placeholder, and a
    launch must not ship placeholders. A section that merely happens to share a
    name with an empty value keeps rendering: its static copy is still content.
    """
    source = template.source.get(name, "")
    keys = _REPEAT_KEY.findall(source)
    if not keys:
        return True
    if any(values.get(key) for key in keys):
        return True
    # Every repeat in this section is empty. Keep it only if it carries prose of
    # its own outside the repeat blocks.
    without_repeats = _REPEAT.sub(" ", source)
    stripped = re.sub(r"<[^>]+>", " ", without_repeats)
    return len(re.sub(r"\s+", " ", stripped).strip()) > 60


def _assemble_body(sections: list[str], template: SiteTemplate, values: dict[str, Any]) -> str:
    """Concatenate the sections, wrapping the content ones in a <main> landmark."""
    rendered = [(name, render_section(template.source[name], values)) for name in sections]
    before: list[str] = []
    content: list[str] = []
    after: list[str] = []
    seen_content = False
    for name, html in rendered:
        if name in _CHROME_SECTIONS:
            (after if seen_content else before).append(html)
        else:
            seen_content = True
            content.append(html)
    parts = before
    if content:
        parts = parts + ['<main id="main">', *content, "</main>"]
    return "\n".join(parts + after)


def _compose_title(values: dict[str, Any]) -> str:
    """Brand plus what it does plus where — from the client's own facts."""
    brand = str(values.get("brand_name", "")).strip()
    tagline = str(values.get("tagline", "")).strip().rstrip(".")
    city = str(values.get("city", "")).strip()
    title = brand
    if tagline and len(f"{brand} — {tagline}") <= 60:
        title = f"{brand} — {tagline}"
    if city and city.lower() not in title.lower() and len(f"{title} | {city}") <= 60:
        title = f"{title} | {city}"
    return title


def _compose_description(values: dict[str, Any]) -> str:
    """A search snippet assembled from stated facts, never invented copy."""
    explicit = str(values.get("meta_description", "")).strip()
    if explicit:
        return explicit
    brand = str(values.get("brand_name", "")).strip()
    parts = [str(values.get("tagline", "")).strip().rstrip(".")]
    where = ", ".join(
        p
        for p in (str(values.get("address", "")).strip(), str(values.get("city", "")).strip())
        if p
    )
    if where:
        parts.append(f"{brand} is at {where}")
    hours = str(values.get("hours", "")).strip()
    if hours:
        parts.append(f"Open {hours}")
    phone = str(values.get("phone", "")).strip()
    if phone and len(". ".join(p for p in parts if p)) < 120:
        parts.append(f"Call {phone}")
    return ". ".join(p for p in parts if p).strip().rstrip(".") + "."


_NOT_FOUND_BODY = """<main id="main">
  <section class="wrap notfound">
    <h1>That page is not here</h1>
    <p>The link may be out of date, or the page may have moved.</p>
    <p><a class="btn" href="/">Go to the home page</a></p>
  </section>
</main>"""


def _write_supporting_pages(
    out: Path, template: SiteTemplate, values: dict[str, Any], seo: dict[str, str]
) -> None:
    """A 404 that keeps the navigation, and a privacy page if one was supplied.

    The privacy policy is deliberately NOT generated. It is a legal document
    about how this practice handles patient data, and a plausible-looking one
    written by a template would be exactly the fabricated professional content
    the quality standard forbids. Supply `privacy_body` (or `privacy_url` for an
    externally hosted policy) and it is published; supply neither and the audit
    reports the gap rather than the tool papering over it.
    """
    chrome = {
        name: render_section(template.source[name], values)
        for name in template.sections
        if name in _CHROME_SECTIONS and name in template.source
    }
    head = chrome.get("header", "")
    foot = chrome.get("footer", "")

    not_found = _PAGE.safe_substitute(
        _KeepUnknown(
            {
                **values,
                **seo,
                "page_title": f"Page not found — {values.get('brand_name', '')}".strip(" —"),
                "meta_description": "This page could not be found.",
                # A 404 must never be indexed as a page in its own right.
                "canonical": "",
                "body": f"{head}\n{_NOT_FOUND_BODY}\n{foot}",
                "script": "",
            }
        )
    ).replace(
        '<meta name="robots" content="index, follow, max-image-preview:large">',
        '<meta name="robots" content="noindex, follow">',
    )
    (out / "404.html").write_text(not_found, encoding="utf-8")

    body = str(values.get("privacy_body", "")).strip()
    if not body:
        return
    paragraphs = "\n".join(
        f"      <p>{line.strip()}</p>" for line in body.splitlines() if line.strip()
    )
    privacy = _PAGE.safe_substitute(
        _KeepUnknown(
            {
                **values,
                **seo,
                "page_title": f"Privacy policy — {values.get('brand_name', '')}".strip(" —"),
                "meta_description": (
                    f"How {values.get('brand_name', 'we')} collects, uses and stores "
                    "personal data, and the rights you have over it."
                ),
                "body": (
                    f'{head}\n<main id="main">\n    <section class="wrap prose">\n'
                    f"      <h1>Privacy policy</h1>\n{paragraphs}\n    </section>\n  </main>\n{foot}"
                ),
                "script": "",
            }
        )
    )
    (out / "privacy.html").write_text(privacy, encoding="utf-8")


def _write_discovery_files(out: Path, values: dict[str, Any], seo: dict[str, str]) -> None:
    """robots.txt, sitemap.xml, security headers, security.txt and the OG image."""
    site_url = str(values.get("site_url") or "").rstrip("/")

    robots = "User-agent: *\nAllow: /\n"
    if site_url:
        robots += f"Sitemap: {site_url}/sitemap.xml\n"
    (out / "robots.txt").write_text(robots, encoding="utf-8")

    loc = f"{site_url}/" if site_url else "/"
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{_esc(loc)}</loc><changefreq>monthly</changefreq>"
        "<priority>1.0</priority></url>\n"
        "</urlset>\n",
        encoding="utf-8",
    )

    # Deployed as-is by Netlify/Cloudflare Pages; a starting point elsewhere.
    (out / "_headers").write_text(_HEADERS.substitute(), encoding="utf-8")

    well_known = out / ".well-known"
    well_known.mkdir(parents=True, exist_ok=True)
    contact = values.get("security_contact") or values.get("email") or ""
    (well_known / "security.txt").write_text(
        f"Contact: mailto:{contact}\nPreferred-Languages: en\n", encoding="utf-8"
    )

    palette = values.get("_palette") or {}
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "public" / "og-image.svg").write_text(
        _OG_IMAGE.safe_substitute(
            _KeepUnknown(
                {
                    "brand_name": _esc(values.get("brand_name", "")),
                    "tagline": _esc(values.get("tagline", "")),
                    "city": _esc(values.get("city", "")),
                    "phone": _esc(values.get("phone", "")),
                    "accent": palette.get("accent", "#0f5d4a"),
                    "secondary": palette.get("secondary", "#0f5d4a"),
                    "paper": palette.get("paper", "#fbfaf7"),
                    "ink": palette.get("ink", "#16241f"),
                    "ink_soft": palette.get("ink_soft", "#4b5f58"),
                    "susp": "410",
                }
            )
        ),
        encoding="utf-8",
    )
