"""The WF-100 catalogue: a hundred checks, one point each.

The catalogue is *data*. It knows what must be true before a site launches and
what evidence deciding that requires — it knows nothing about how to gather
that evidence or how to score it. Keeping it inert is what lets the honesty
invariants in ``tests/unit/test_wf100_standard.py`` be asserted at all.

Reading a ``Check``:

``requires``
    What the auditor must hold in hand to decide this check. If any required
    evidence is missing, the check is OUTSTANDING — not passed, not failed.
    This single field is what stops the tool from claiming a live TLS
    configuration is fine after reading a folder of HTML off a disk.

``level``
    The user-facing tier from the standard: AUTOMATED (a machine decides),
    ASSISTED (a machine observes, a person confirms against a record), HUMAN
    (only a person can judge). Level tracks *who decides*; ``requires`` tracks
    *what it takes*. They agree by construction and the tests enforce it.

``critical``
    Failing this blocks launch at any score. An unresolved critical check also
    blocks launch — an unread smoke alarm is not an absence of fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    """The eight WF-100 categories and their id prefixes."""

    ARCHITECTURE = ("A", "Architecture & Code")
    PERFORMANCE = ("B", "Performance")
    ACCESSIBILITY = ("C", "Accessibility")
    SEO = ("D", "SEO")
    LOCAL_SEO = ("E", "Local SEO")
    UX_CONVERSION = ("F", "UX & Conversion")
    SECURITY_PRIVACY = ("G", "Security & Privacy")
    CONTENT = ("H", "Content & Professional Quality")

    def __init__(self, prefix: str, label: str) -> None:
        self.prefix = prefix
        self.label = label


CATEGORY_WEIGHTS: dict[Category, int] = {
    Category.ARCHITECTURE: 10,
    Category.PERFORMANCE: 15,
    Category.ACCESSIBILITY: 15,
    Category.SEO: 15,
    Category.LOCAL_SEO: 10,
    Category.UX_CONVERSION: 15,
    Category.SECURITY_PRIVACY: 10,
    Category.CONTENT: 10,
}


class Level(Enum):
    """Verification tier. Determines who is able to close the check out."""

    AUTOMATED = "automated"
    ASSISTED = "assisted"
    HUMAN = "human"


class Evidence(Enum):
    """What deciding a check requires the auditor to actually possess.

    The distinction that matters is between what a static read of a build can
    establish and what it cannot. ``MARKUP`` is on disk. ``HTTP`` requires the
    site to be served and answering. ``FIELD_DATA`` requires real visitors.
    ``HUMAN`` requires a person. Conflating the first with the rest is the
    failure mode this enum exists to make impossible.
    """

    MARKUP = "markup"  # HTML source of the site's pages
    STYLES = "styles"  # CSS, whether linked or inline
    SCRIPTS = "scripts"  # JavaScript shipped with the page
    ASSETS = "assets"  # files on disk, with their byte sizes
    HTTP = "http"  # a live response: status, headers, redirects, TLS
    BROWSER = "browser"  # a rendering engine: runtime errors, focus, layout
    RECORD = "record"  # declared business facts to check the site against
    FIELD_DATA = "field_data"  # Core Web Vitals from real users (CrUX/RUM)
    HUMAN = "human"  # a person's judgement


@dataclass(frozen=True)
class Check:
    """One WF-100 check. Worth one point; decidable only with ``requires``."""

    id: str
    category: Category
    title: str
    remedy: str
    level: Level
    requires: frozenset[Evidence]
    critical: bool = False
    points: int = 1
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.id} {self.title}"


def _c(
    cid: str,
    category: Category,
    title: str,
    remedy: str,
    level: Level,
    requires: tuple[Evidence, ...],
    *,
    critical: bool = False,
    notes: str = "",
) -> Check:
    return Check(
        id=cid,
        category=category,
        title=title,
        remedy=remedy,
        level=level,
        requires=frozenset(requires),
        critical=critical,
        notes=notes,
    )


_A = Category.ARCHITECTURE
_B = Category.PERFORMANCE
_C = Category.ACCESSIBILITY
_D = Category.SEO
_E = Category.LOCAL_SEO
_F = Category.UX_CONVERSION
_G = Category.SECURITY_PRIVACY
_H = Category.CONTENT

_AUTO = Level.AUTOMATED
_ASSIST = Level.ASSISTED
_HUMAN = Level.HUMAN

_M = Evidence.MARKUP
_S = Evidence.STYLES
_J = Evidence.SCRIPTS
_AS = Evidence.ASSETS
_HTTP = Evidence.HTTP
_BR = Evidence.BROWSER
_REC = Evidence.RECORD
_FIELD = Evidence.FIELD_DATA
_HU = Evidence.HUMAN


# ── A. Architecture & Code — 10 ──────────────────────────────────────────────

_ARCHITECTURE: list[Check] = [
    _c(
        "A1",
        _A,
        "Semantic HTML5 structure",
        "Wrap the page in header/nav/main/footer landmarks instead of nested divs.",
        _AUTO,
        (_M,),
    ),
    _c(
        "A2",
        _A,
        "Exactly one logical H1 per page",
        "Keep a single H1 naming the page's subject; demote the others to H2.",
        _AUTO,
        (_M,),
    ),
    _c(
        "A3",
        _A,
        "Heading hierarchy without skipped levels",
        "Do not jump H2 to H4; headings are an outline, not a type scale.",
        _AUTO,
        (_M,),
    ),
    _c(
        "A4",
        _A,
        "Descriptive, readable URLs",
        "Use /services/implants, not /page?id=42 or /p/3f9a.",
        _AUTO,
        (_M,),
    ),
    _c(
        "A5",
        _A,
        "No copy-pasted content blocks across pages",
        "Two pages sharing a body block verbatim is copy-paste; extract it or cut it.",
        _AUTO,
        (_M,),
        notes=(
            "Judged on the build, where shared chrome is expected to repeat. Whether the "
            "SOURCE is componentised is a code review, not an audit of the output."
        ),
    ),
    _c(
        "A6",
        _A,
        "No unused critical dependencies shipped",
        "Drop libraries the pages never call; every KB is paid for on mobile data.",
        _AUTO,
        (_M, _J),
    ),
    _c(
        "A7",
        _A,
        "No console errors on load",
        "Open the console on every template and clear what it reports.",
        _ASSIST,
        (_BR,),
        notes="Runtime behaviour. A static read cannot see an exception that has not been thrown.",
    ),
    _c(
        "A8",
        _A,
        "No broken internal links",
        "Fix or remove links whose target does not exist in the build.",
        _AUTO,
        (_M, _AS),
    ),
    _c(
        "A9",
        _A,
        "A working 404 page",
        "Ship a 404 that keeps the site's navigation and offers a way back.",
        _AUTO,
        (_M, _AS),
    ),
    _c(
        "A10",
        _A,
        "Production build free of debug artifacts",
        "Strip console.log, TODO markers, debugger statements and commented-out blocks.",
        _AUTO,
        (_M, _J),
    ),
]


# ── B. Performance — 15 ──────────────────────────────────────────────────────

_PERFORMANCE: list[Check] = [
    _c(
        "B1",
        _B,
        "LCP at or under 2.5s for real users",
        "Measure the field p75 in CrUX or your RUM tool, then fix the largest element's load path.",
        _ASSIST,
        (_FIELD,),
        notes="Field metric. A lab score on a fast desktop is a different measurement, not this one.",
    ),
    _c(
        "B2",
        _B,
        "INP at or under 200ms for real users",
        "Measure the field p75, then break up the long tasks blocking the main thread.",
        _ASSIST,
        (_FIELD,),
        notes="Field metric. Requires real interactions from real devices.",
    ),
    _c(
        "B3",
        _B,
        "CLS at or under 0.1 for real users",
        "Measure the field p75; reserve space for media, ads and late-loading fonts.",
        _ASSIST,
        (_FIELD,),
        notes="Field metric. Reserving space is necessary but does not prove the score.",
    ),
    _c(
        "B4",
        _B,
        "Images within a sane weight budget",
        "Compress and resize; no single image over ~300KB, no page over ~1.5MB of imagery.",
        _AUTO,
        (_AS,),
    ),
    _c(
        "B5",
        _B,
        "Modern image formats served",
        "Serve WebP or AVIF with a legacy fallback via <picture>.",
        _AUTO,
        (_M, _AS),
    ),
    _c(
        "B6",
        _B,
        "Responsive image sizing",
        "Give images srcset/sizes and explicit width/height so phones do not fetch desktop pixels.",
        _AUTO,
        (_M,),
    ),
    _c(
        "B7",
        _B,
        "Below-the-fold media lazy-loaded",
        'Add loading="lazy" below the fold and leave the LCP image eager.',
        _AUTO,
        (_M,),
    ),
    _c(
        "B8",
        _B,
        "Critical assets prioritised",
        "Preload the LCP image and the first font; preconnect to any origin on the critical path.",
        _AUTO,
        (_M,),
    ),
    _c(
        "B9",
        _B,
        "Font payload minimised",
        "Two families at most, subset to the glyphs used, with font-display: swap.",
        _AUTO,
        (_M, _S),
    ),
    _c(
        "B10",
        _B,
        "Third-party scripts minimised",
        "Every third-party tag is someone else's performance budget spending yours.",
        _AUTO,
        (_M,),
    ),
    _c(
        "B11",
        _B,
        "CSS and JS minified and small",
        "Minify and split; a brochure site does not need 300KB of JavaScript.",
        _AUTO,
        (_AS,),
    ),
    _c(
        "B12",
        _B,
        "Compression enabled on the server",
        "Enable Brotli or gzip for HTML, CSS, JS and SVG at the edge.",
        _ASSIST,
        (_HTTP,),
        notes="A property of the server, not the build. Needs a live response to confirm.",
    ),
    _c(
        "B13",
        _B,
        "Browser caching configured",
        "Long max-age with content hashes for static assets; short for HTML.",
        _ASSIST,
        (_HTTP,),
        notes="A _headers file states the intent; only a response header proves it is honoured.",
    ),
    _c(
        "B14",
        _B,
        "CDN in front of static assets where justified",
        "Serve assets from an edge network when visitors are not all next to the origin.",
        _ASSIST,
        (_HTTP,),
        notes="Judged from response headers and origin geography, not from the build.",
    ),
    _c(
        "B15",
        _B,
        "No unnecessary render-blocking resources",
        "Defer non-critical JS and inline the CSS the first paint actually needs.",
        _AUTO,
        (_M,),
    ),
]


# ── C. Accessibility — 15 ────────────────────────────────────────────────────

_ACCESSIBILITY: list[Check] = [
    _c(
        "C1",
        _C,
        "WCAG 2.2 AA baseline: no machine-detectable violations",
        "Clear every violation an automated scan can prove before booking human testing.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C2",
        _C,
        "Every function reachable by keyboard",
        "Tab through the whole site; anything a mouse can do, a keyboard must do.",
        _HUMAN,
        (_HU,),
        notes="Requires someone to actually tab through the site.",
    ),
    _c(
        "C3",
        _C,
        "Focus indicator always visible",
        "Never remove the outline without replacing it; style :focus-visible deliberately.",
        _AUTO,
        (_S,),
    ),
    _c(
        "C4",
        _C,
        "Logical tab order",
        "Let the DOM order carry the tab order; positive tabindex values break it.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C5",
        _C,
        "Navigation and forms exposed to assistive technology",
        "Give nav a landmark and an accessible name; group related fields.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C6",
        _C,
        "Every input has an associated label",
        "Bind label[for] to input[id]; a placeholder is not a label.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C7",
        _C,
        "Informative images carry useful alt text",
        "Describe what the image conveys, not that it is an image.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C8",
        _C,
        "Decorative images hidden from assistive technology",
        'Give purely decorative images alt="" so screen readers skip them.',
        _AUTO,
        (_M,),
    ),
    _c(
        "C9",
        _C,
        "Sufficient colour contrast",
        "4.5:1 for body text, 3:1 for large text and meaningful UI boundaries.",
        _AUTO,
        (_M, _S),
    ),
    _c(
        "C10",
        _C,
        "Information never conveyed by colour alone",
        "Pair every colour signal with text, shape or an icon.",
        _HUMAN,
        (_HU,),
        notes="Requires reading the page's meaning, not its markup.",
    ),
    _c(
        "C11",
        _C,
        "Buttons have accessible names",
        "An icon-only button needs aria-label or visually hidden text.",
        _AUTO,
        (_M,),
    ),
    _c(
        "C12",
        _C,
        "Link text is meaningful out of context",
        '"Read more" tells a screen-reader user nothing; name the destination.',
        _AUTO,
        (_M,),
    ),
    _c(
        "C13",
        _C,
        "No keyboard traps",
        "Every dialog, menu and embed must be escapable by keyboard.",
        _HUMAN,
        (_HU,),
        notes="A trap is defined by what happens when you try to leave.",
    ),
    _c(
        "C14",
        _C,
        "Reduced-motion preference respected",
        "Gate transforms and parallax behind prefers-reduced-motion.",
        _AUTO,
        (_S,),
    ),
    _c(
        "C15",
        _C,
        "Page language declared",
        'Set <html lang="el"> so screen readers pronounce the content correctly.',
        _AUTO,
        (_M,),
        notes="WCAG 3.1.1, required for AA conformance. Added to complete this category to 15.",
    ),
]


# ── D. SEO — 15 ──────────────────────────────────────────────────────────────

_SEO: list[Check] = [
    _c(
        "D1",
        _D,
        "Unique, well-sized title on every page",
        "One title per page, roughly 30-60 characters, subject first.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D2",
        _D,
        "Unique, well-sized meta description on every page",
        "Roughly 70-160 characters, written for a human deciding whether to click.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D3",
        _D,
        "H1 present, unique and specific",
        "One H1 per page naming that page's subject, distinct from every other page.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D4",
        _D,
        "Content matches search intent",
        "Read the queries you want to win and check the page actually answers them.",
        _HUMAN,
        (_HU,),
        notes="Intent is a judgement about people, not a property of markup.",
    ),
    _c(
        "D5",
        _D,
        "Canonical URL declared",
        "Point rel=canonical at the preferred absolute URL of each page.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D6",
        _D,
        "XML sitemap present and well-formed",
        "Ship sitemap.xml listing the canonical URLs and reference it from robots.txt.",
        _AUTO,
        (_AS,),
    ),
    _c(
        "D7",
        _D,
        "robots.txt present and sane",
        "Allow crawling of what should rank and point to the sitemap.",
        _AUTO,
        (_AS,),
    ),
    _c(
        "D8",
        _D,
        "Site is crawlable",
        "No Disallow: / and no crawl rule hiding the pages meant to rank.",
        _AUTO,
        (_AS,),
    ),
    _c(
        "D9",
        _D,
        "No accidental noindex",
        "Remove the staging noindex before launch; check meta robots and X-Robots-Tag.",
        _AUTO,
        (_M,),
        critical=True,
    ),
    _c(
        "D10",
        _D,
        "Internal linking connects the site",
        "Every page reachable from the navigation; no orphans.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D11",
        _D,
        "Descriptive anchor text",
        'Link "implant treatment in Kalamaria", not "here".',
        _AUTO,
        (_M,),
    ),
    _c(
        "D12",
        _D,
        "Open Graph and Twitter cards complete",
        "og:title, og:description, og:image, og:url and twitter:card on every page.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D13",
        _D,
        "Schema.org structured data present and valid",
        "Ship JSON-LD that parses and uses real types and required properties.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D14",
        _D,
        "Images have alt text and descriptive filenames",
        "hero-implant-clinic.webp, not IMG_2931.jpg.",
        _AUTO,
        (_M,),
    ),
    _c(
        "D15",
        _D,
        "Google Search Console verified and reporting",
        "Verify the property, submit the sitemap and check coverage after launch.",
        _ASSIST,
        (_REC,),
        notes="Lives in an external account. The site cannot prove its own registration.",
    ),
]


# ── E. Local SEO — 10 ────────────────────────────────────────────────────────

_LOCAL_SEO: list[Check] = [
    _c(
        "E1",
        _E,
        "Business name exactly as registered",
        "Match the legal and Google Business Profile name character for character.",
        _ASSIST,
        (_M, _REC),
    ),
    _c(
        "E2",
        _E,
        "Address correct and complete",
        "Street, number, postcode and city, matching the profile exactly.",
        _ASSIST,
        (_M, _REC),
    ),
    _c(
        "E3",
        _E,
        "Phone number correct and tappable",
        "Publish the real number as a tel: link, in the same format everywhere.",
        _ASSIST,
        (_M, _REC),
    ),
    _c(
        "E4",
        _E,
        "Opening hours published and machine-readable",
        "State the hours on the page and mirror them in openingHours structured data.",
        _AUTO,
        (_M,),
    ),
    _c(
        "E5",
        _E,
        "Google Business Profile claimed and linked",
        "Claim the profile, complete it, and link it from the site.",
        _ASSIST,
        (_M, _REC),
        notes="Claim status lives in Google's account, not in the page.",
    ),
    _c(
        "E6",
        _E,
        "Map to the location embedded",
        "Embed a map or link one, so a visitor can start navigating in one tap.",
        _AUTO,
        (_M,),
    ),
    _c(
        "E7",
        _E,
        "Local service information stated",
        "Say which areas you serve and what a local visitor gets.",
        _AUTO,
        (_M,),
    ),
    _c(
        "E8",
        _E,
        "Location relevance without keyword stuffing",
        "Name the city where it reads naturally; repetition reads as spam to both sides.",
        _AUTO,
        (_M,),
    ),
    _c(
        "E9",
        _E,
        "NAP consistent across the whole site",
        "One name, one address, one phone, identical in header, footer and schema.",
        _AUTO,
        (_M,),
    ),
    _c(
        "E10",
        _E,
        "Local structured data complete",
        "LocalBusiness JSON-LD with address, geo coordinates and telephone.",
        _AUTO,
        (_M,),
    ),
]


# ── F. UX & Conversion — 15 ──────────────────────────────────────────────────

_UX: list[Check] = [
    _c(
        "F1",
        _F,
        "Value proposition visible above the fold",
        "A visitor should know what this is and where it is without scrolling.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F2",
        _F,
        "Primary CTA obvious",
        "One clear primary action per page, styled distinctly from everything else.",
        _AUTO,
        (_M, _S),
    ),
    _c(
        "F3",
        _F,
        "CTA reachable on mobile without hunting",
        "Keep the primary action within thumb reach and at least 44px tall.",
        _AUTO,
        (_M, _S),
    ),
    _c(
        "F4",
        _F,
        "Tap-to-call works",
        "Phone numbers are tel: links, not plain text a visitor must retype.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F5",
        _F,
        "Booking route is obvious",
        "The way to book is visible from the first screen and never more than one tap away.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F6",
        _F,
        "Contact routes reachable from every page",
        "Phone, address and a contact link in the footer of every page.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F7",
        _F,
        "Forms tested end to end",
        "Submit every form and confirm the message actually arrives at the client.",
        _HUMAN,
        (_HU,),
        critical=True,
        notes="A form that renders is not a form that delivers. Someone must send one.",
    ),
    _c(
        "F8",
        _F,
        "Form errors are understandable",
        'Say "enter a phone number we can reach you on", not "invalid input".',
        _AUTO,
        (_M, _J),
    ),
    _c(
        "F9",
        _F,
        "Successful submission is confirmed",
        "Show a clear success state; silence reads as failure and the visitor leaves.",
        _AUTO,
        (_M, _J),
    ),
    _c(
        "F10",
        _F,
        "Trust signals present",
        "Real photos, real names, registration numbers, genuine reviews.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F11",
        _F,
        "Professional credentials easy to find",
        "State qualifications and registrations where a cautious visitor will look.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F12",
        _F,
        "Services structured and scannable",
        "Group services with headings a visitor can skim, not one wall of prose.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F13",
        _F,
        "Pricing information addressed",
        "Publish prices, ranges, or a plain statement of how pricing works.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F14",
        _F,
        "FAQ answers the real questions",
        "Answer what people actually ask before they call: cost, pain, time, parking.",
        _AUTO,
        (_M,),
    ),
    _c(
        "F15",
        _F,
        "No unnecessary friction in the conversion path",
        "Ask for the fewest fields that let you call the person back.",
        _AUTO,
        (_M,),
    ),
]


# ── G. Security & Privacy — 10 ───────────────────────────────────────────────

_SECURITY: list[Check] = [
    _c(
        "G1",
        _G,
        "HTTPS served with a valid certificate",
        "Serve everything over TLS with a certificate that is valid and not near expiry.",
        _ASSIST,
        (_HTTP,),
        critical=True,
        notes="A property of the deployment. Cannot be established from a build directory.",
    ),
    _c(
        "G2",
        _G,
        "HTTP redirects to HTTPS",
        "Answer plain HTTP with a 301 to the HTTPS URL, then enable HSTS.",
        _ASSIST,
        (_HTTP,),
        notes="Requires a request to the plain-HTTP origin to confirm.",
    ),
    _c(
        "G3",
        _G,
        "Security headers set",
        "CSP, HSTS, X-Content-Type-Options, Referrer-Policy and Permissions-Policy.",
        _ASSIST,
        (_HTTP,),
        notes="A _headers or vhost file declares intent; only a response proves delivery.",
    ),
    _c(
        "G4",
        _G,
        "Administrative areas protected",
        "No admin, staging or upload path reachable without authentication.",
        _AUTO,
        (_M, _AS),
        critical=True,
    ),
    _c(
        "G5",
        _G,
        "Strong administrative authentication with 2FA",
        "Unique passwords in a manager, 2FA on every admin account, least privilege.",
        _ASSIST,
        (_REC,),
        notes="Lives in the hosting and CMS accounts. Never in the repository.",
    ),
    _c(
        "G6",
        _G,
        "Third-party code pinned and integrity-checked",
        "Pin external script versions and add subresource integrity, or self-host them.",
        _AUTO,
        (_M,),
    ),
    _c(
        "G7",
        _G,
        "Forms protected against spam and abuse",
        "Honeypot or captcha plus server-side rate limiting on every public form.",
        _AUTO,
        (_M,),
    ),
    _c(
        "G8",
        _G,
        "Privacy policy present and linked",
        "Publish a real policy naming what is collected, why, and for how long.",
        _AUTO,
        (_M,),
    ),
    _c(
        "G9",
        _G,
        "Cookie consent where non-essential cookies are used",
        "Ask before setting anything beyond strictly necessary cookies.",
        _AUTO,
        (_M, _J),
    ),
    _c(
        "G10",
        _G,
        "Analytics fire only after consent",
        "Confirm in the network tab that nothing tracks before the visitor agrees.",
        _ASSIST,
        (_BR,),
        critical=True,
        notes="Runtime behaviour. Only a browser can show what fired and when.",
    ),
]


# ── H. Content & Professional Quality — 10 ───────────────────────────────────

_CONTENT: list[Check] = [
    _c(
        "H1",
        _H,
        "No placeholder text anywhere",
        "Remove lorem ipsum, TODO, and unresolved template variables before launch.",
        _AUTO,
        (_M,),
        critical=True,
    ),
    _c(
        "H2",
        _H,
        "No stock photography presented as the real practice",
        "Label stock imagery or replace it with photographs of the actual place and people.",
        _HUMAN,
        (_HU,),
        notes="Whether a photograph is of this practice cannot be read from the file.",
    ),
    _c(
        "H3",
        _H,
        "No fabricated testimonials",
        "Publish only reviews a real patient actually left, with their consent.",
        _HUMAN,
        (_HU,),
        critical=True,
        notes="Invented testimonials are fraud. This check is never satisfied by a tool.",
    ),
    _c(
        "H4",
        _H,
        "No fabricated qualifications or credentials",
        "State only degrees, registrations and memberships that can be verified.",
        _HUMAN,
        (_HU,),
        critical=True,
        notes="Invented credentials are fraud and, for a clinician, a regulatory matter.",
    ),
    _c(
        "H5",
        _H,
        "Contact information verified against the client",
        "Call the number and send to the address before launch, not after.",
        _ASSIST,
        (_M, _REC),
        critical=True,
    ),
    _c(
        "H6",
        _H,
        "All imagery properly licensed",
        "Keep the licence or release for every photograph and illustration on the site.",
        _HUMAN,
        (_HU,),
        notes="Provenance lives in paperwork, not in the image.",
    ),
    _c(
        "H7",
        _H,
        "Health and medical claims reviewed by the practitioner",
        "The clinician signs off every clinical statement before it is published.",
        _HUMAN,
        (_HU,),
        critical=True,
        notes="Regulated speech. Only the responsible professional can approve it.",
    ),
    _c(
        "H8",
        _H,
        "Copy proofread",
        "Read every page aloud; spelling and grammar are read as competence.",
        _HUMAN,
        (_HU,),
    ),
    _c(
        "H9",
        _H,
        "Content reviewed on a real phone",
        "Read the whole site on a mid-range handset, not a desktop window resized.",
        _HUMAN,
        (_HU,),
    ),
    _c(
        "H10",
        _H,
        "Client has approved the content",
        "Get explicit written approval of the final copy before launch.",
        _HUMAN,
        (_HU,),
    ),
]


STANDARD: tuple[Check, ...] = tuple(
    _ARCHITECTURE + _PERFORMANCE + _ACCESSIBILITY + _SEO + _LOCAL_SEO + _UX + _SECURITY + _CONTENT
)

BY_ID: dict[str, Check] = {c.id: c for c in STANDARD}

#: Score at or above this, with no critical failure, and the site may launch.
LAUNCH_THRESHOLD = 90.0


def checks_for(category: Category) -> tuple[Check, ...]:
    """Every check in one category, in catalogue order."""
    return tuple(c for c in STANDARD if c.category is category)


def get(check_id: str) -> Check:
    """Look up one check. Raises KeyError with the id, not a bare KeyError."""
    try:
        return BY_ID[check_id]
    except KeyError:
        raise KeyError(f"no WF-100 check with id {check_id!r}") from None


_CATEGORY_ORDER: tuple[Category, ...] = tuple(Category)


def category_order() -> tuple[Category, ...]:
    """Categories in the order the standard publishes them (A through H)."""
    return _CATEGORY_ORDER


__all__ = [
    "BY_ID",
    "CATEGORY_WEIGHTS",
    "LAUNCH_THRESHOLD",
    "STANDARD",
    "Category",
    "Check",
    "Evidence",
    "Level",
    "category_order",
    "checks_for",
    "get",
]
