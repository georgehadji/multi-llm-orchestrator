"""
Component Archetype Catalog — 50 component shapes with variation knobs.
=======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Each archetype is a component shape with 2–3 variation knobs so the same
macrostructure doesn't produce identical output twice.

Source: Hallmark design skill (references/component-cookbook.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Archetype:
    """A component archetype — shape + variation knobs."""

    code: str
    category: Literal[
        "hero",
        "section_head",
        "feature",
        "cta",
        "testimonial",
        "footer",
        "nav",
    ]
    name: str
    description: str
    knobs: dict[str, list[str]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Heroes (H1–H9)
# ═══════════════════════════════════════════════════════════════════════════════

ARCHETYPES: dict[str, Archetype] = {
    # Heroes
    "H1": Archetype(
        code="H1",
        category="hero",
        name="Marquee",
        description="A single statement fills the fold. No subhead, no CTA in view.",
        knobs={
            "display_size": ["xxl", "xl"],
            "alignment": ["left-bias", "centred", "right-bias"],
            "underlay": ["none", "single-rule-above", "single-rule-below"],
        },
    ),
    "H2": Archetype(
        code="H2",
        category="hero",
        name="Split Diptych",
        description="Headline + lede on one side, image or product capture on the other.",
        knobs={
            "ratio": ["7/5", "6/6", "5/7"],
            "right_side": ["photo", "proof-column", "pull-quote"],
            "divider": ["hairline", "negative-space", "vertical-rule"],
        },
    ),
    "H3": Archetype(
        code="H3",
        category="hero",
        name="Quote Led",
        description="A pull-quote with attribution is the hero. Headline is borrowed credibility.",
        knobs={
            "quote_weight": ["italic-display", "roman-display", "roman-body-large"],
            "attribution_position": ["under-quote", "margin-aligned", "right-flush"],
            "length": ["≤80-chars", "80–160-chars"],
        },
    ),
    "H4": Archetype(
        code="H4",
        category="hero",
        name="Stat Led",
        description="A giant number or metric is the hero. Small qualifier line below.",
        knobs={
            "number_style": ["tabular-display", "italic-display", "monospace"],
            "qualifier_position": ["below", "inline-right", "stacked-above"],
            "secondary_stats": ["none", "two-below", "row-of-four"],
        },
    ),
    "H5": Archetype(
        code="H5",
        category="hero",
        name="Letter Hero",
        description="First-person opening — 'Dear reader,'. No buttons in fold.",
        knobs={
            "salutation": ["greeting", "dear-x", "time-stamp"],
            "body_length": ["1-paragraph", "2-paragraphs", "3-paragraphs"],
            "signoff": ["typed-name", "drawn-signature-svg", "initials"],
        },
    ),
    "H6": Archetype(
        code="H6",
        category="hero",
        name="Photographic Fold",
        description="Single full-bleed image fills the viewport. Caption sits in a corner.",
        knobs={
            "image_area": ["full-bleed", "16/7", "4/3", "1/1-square"],
            "caption_position": ["lower-left", "upper-right", "margin"],
            "text_placement": ["below", "overlaid"],
        },
    ),
    "H7": Archetype(
        code="H7",
        category="hero",
        name="Demo Video Clipped",
        description="Display headline left, demo video right, ~10–20% cut off by viewport edge.",
        knobs={
            "clip_side": ["right", "left", "both"],
            "aspect_ratio": ["16/10", "16/9", "4/3"],
            "frame": ["hairline", "browser-chrome", "none"],
        },
    ),
    "H8": Archetype(
        code="H8",
        category="hero",
        name="Mockup Split Browser Framed",
        description="Headline left, browser-frame mockup right, tilted 1–3° for life.",
        knobs={
            "frame_style": [
                "browser-chrome",
                "macOS-toolbar",
                "minimal-hairline",
                "floating-no-frame",
            ],
            "tilt": ["0deg", "1.5deg", "3deg"],
            "screenshot_count": ["1", "stack-of-3", "orbit-of-3"],
        },
    ),
    "H9": Archetype(
        code="H9",
        category="hero",
        name="Custom Illustration Centerpiece",
        description="A single hand-built SVG or pure-CSS shape sitting on the hero as one illustrative element.",
        knobs={
            "build_method": [
                "tier-a-pure-css",
                "tier-b-hand-svg",
                "tier-c-generated",
                "tier-d-library",
            ],
            "animation": ["none", "loop", "scroll-linked"],
            "scale": ["small-accent", "dominant"],
        },
    ),
    # Section Heads (S1–S5)
    "S1": Archetype(
        code="S1",
        category="section_head",
        name="Left Margin Numbered",
        description="A narrow left column holds '01 — LABEL.'; the wide right column holds heading and content.",
        knobs={
            "width": ["12ch", "16ch", "20ch"],
            "number_format": ["01", "I", "01."],
            "divider": ["hairline", "negative-space", "vertical-rule"],
        },
    ),
    "S2": Archetype(
        code="S2",
        category="section_head",
        name="Hanging",
        description="Heading floats above the section in negative space; no border, no rule.",
        knobs={
            "spacing": ["tight", "default", "spacious"],
            "alignment": ["left", "centred", "right"],
            "decoration": ["none", "hairline-above", "hairline-below"],
        },
    ),
    "S3": Archetype(
        code="S3",
        category="section_head",
        name="Sticky Pinned",
        description="Heading remains in viewport while content scrolls beneath. Orientation aid.",
        knobs={
            "pin_side": ["left", "right"],
            "width": ["25%", "33%", "40%"],
            "behavior": ["sticky", "fixed"],
        },
    ),
    "S4": Archetype(
        code="S4",
        category="section_head",
        name="Inline No Break",
        description="The heading is a small caps phrase that emerges inside the body flow; no spatial break.",
        knobs={
            "style": ["small-caps", "mono-label", "bold-inline"],
            "separator": ["none", "hairline", "dot"],
            "spacing": ["tight", "default"],
        },
    ),
    "S5": Archetype(
        code="S5",
        category="section_head",
        name="Bottom Anchored",
        description="The label or heading sits below the section's content. Inverts hierarchy.",
        knobs={
            "alignment": ["left", "centred", "right"],
            "style": ["display", "body-large", "mono-label"],
            "spacing": ["tight", "default", "spacious"],
        },
    ),
    # Features (F1–F6)
    "F1": Archetype(
        code="F1",
        category="feature",
        name="Bento Grid",
        description="Asymmetric grid of 8–15 tiles in mixed spans (1×1, 2×1, 1×2, 2×2).",
        knobs={
            "tiles": ["4", "6", "7", "9"],
            "spans": ["regular", "irregular", "mosaic"],
            "border": ["hairline-all", "accent-corners", "none"],
        },
    ),
    "F2": Archetype(
        code="F2",
        category="feature",
        name="Sticky Scroll Stack",
        description="Sticky left pane, scrolling right pane that cycles through related screenshots.",
        knobs={
            "pinned_side": ["left", "right"],
            "right_pane_content": ["code", "screenshot", "diagram"],
            "pin_steps": ["3", "4", "5"],
        },
    ),
    "F3": Archetype(
        code="F3",
        category="feature",
        name="Tabular Spec Sheet",
        description="Each row is a feature; columns hold name, value, footnote. Hairline rules between rows.",
        knobs={
            "columns": ["2-key-val", "3-key-val-unit", "4-with-footnote"],
            "rule_density": ["every-row", "groups-of-3", "headers-only"],
            "numbers": ["tabular", "proportional"],
        },
    ),
    "F4": Archetype(
        code="F4",
        category="feature",
        name="Step Sequence",
        description="Numbered stages (1.0 → 2.0 → 3.0) flow vertically. Each stage has heading, paragraph, small visual.",
        knobs={
            "numbering": ["I-II-III", "01-02-03", "1.0-2.0-3.0"],
            "layout": ["vertical-stack", "horizontal-flow", "diagonal"],
            "connector": ["line", "arrow", "none"],
        },
    ),
    "F5": Archetype(
        code="F5",
        category="feature",
        name="Annotated Screenshot",
        description="A product capture sits centre-stage with arrows or short labels pointing to UI details.",
        knobs={
            "callouts": ["numbered-pins", "margin-labels", "inline-arrows"],
            "frame": ["device", "plain", "floating"],
            "anchor": ["image-led", "text-led"],
        },
    ),
    "F6": Archetype(
        code="F6",
        category="feature",
        name="Product Card Grid",
        description="Each card is a product, not a feature. Image · name · price · one micro-action.",
        knobs={
            "card_ratio": ["3/4-portrait", "1/1-square", "4/3-landscape"],
            "density": ["3-up", "4-up", "5-up"],
            "micro_action": ["add", "save", "view-arrow", "none"],
        },
    ),
    # CTAs (C1–C4)
    "C1": Archetype(
        code="C1",
        category="cta",
        name="Outlined Chip",
        description="A bordered, transparent button with a typographic verb ('Save changes').",
        knobs={
            "shape": ["rectangular", "pill", "slab"],
            "density": ["spacious", "compact"],
            "adornment": ["arrow", "plus", "none"],
        },
    ),
    "C2": Archetype(
        code="C2",
        category="cta",
        name="Inline Form as CTA",
        description="The CTA IS the form — a single email input with a 'Submit →' beside it.",
        knobs={
            "field_count": ["1", "2", "3"],
            "submit_position": ["end-of-row", "separate-line", "embedded-button"],
            "helper": ["above", "below", "none"],
        },
    ),
    "C3": Archetype(
        code="C3",
        category="cta",
        name="Typographic Link",
        description="Just a word, an arrow, and a 1-px underline. No box, no fill.",
        knobs={
            "underline": ["solid", "dashed", "double", "none"],
            "hover_behaviour": ["thicken", "slide", "colour-shift"],
            "arrow": ["→", "↗", "none"],
        },
    ),
    "C4": Archetype(
        code="C4",
        category="cta",
        name="Sticky Bottom Bar",
        description="A horizontal bar pinned to the viewport bottom, holding a CTA + brief reassurance line.",
        knobs={
            "reveal": ["always", "scroll-up", "after-fold"],
            "anchored": ["viewport-bottom", "viewport-top", "inline-at-bottom"],
            "shadow": ["hairline", "none", "subtle"],
        },
    ),
    # Testimonials (T1–T4)
    "T1": Archetype(
        code="T1",
        category="testimonial",
        name="Pull Quote with Marginalia",
        description="A quote sits in the wide column; attribution floats in the narrow margin column.",
        knobs={
            "quote_treatment": ["italic-display", "roman-large", "serif-italic"],
            "attribution": ["signed", "stamped", "timestamped"],
            "marginalia": ["none", "timeline", "1-footnote"],
        },
    ),
    "T2": Archetype(
        code="T2",
        category="testimonial",
        name="Logo Wall Hairline",
        description="A row of customer logos, monochromatic, separated by hairline rules. No card boxes.",
        knobs={
            "layout": ["single-row", "2-rows", "grid-3xN"],
            "logo_treatment": ["monochrome-ink", "brand-colour", "ghosted"],
            "divider": ["hairline-cells", "none"],
        },
    ),
    "T3": Archetype(
        code="T3",
        category="testimonial",
        name="Single Huge Quote",
        description="One quote, set big, centered, taking a whole section. Attribution is small caps beneath.",
        knobs={
            "quote_face": ["serif-italic", "roman-display", "italic-mono"],
            "width": ["full-bleed", "60ch", "40ch"],
            "attribution_position": ["same-line", "separate-band"],
        },
    ),
    "T4": Archetype(
        code="T4",
        category="testimonial",
        name="Numbered Stat Strip",
        description="A horizontal strip of 3–5 stats (count + qualifier) running across one row. Tabular nums.",
        knobs={
            "layout": ["3-up", "4-up", "5-up", "6-up"],
            "number_weight": ["display", "body-large"],
            "qualifier_position": ["under", "inline", "above"],
        },
    ),
    # Footers (Ft1–Ft8)
    "Ft1": Archetype(
        code="Ft1",
        category="footer",
        name="Mast Headed",
        description="A wordmark and tagline anchor a single horizontal band. Two or three small links beside.",
        knobs={
            "wordmark_size": ["display-3xl", "display-2xl", "xl"],
            "tagline": ["italic-serif", "roman-body", "none"],
            "links_row": ["inline", "2-line-stack"],
        },
    ),
    "Ft2": Archetype(
        code="Ft2",
        category="footer",
        name="Inline Rule Single Line",
        description="A single horizontal line of credits, address, copyright. Hairline rule above.",
        knobs={
            "order": ["wordmark-links-credit", "credit-wordmark-links"],
            "separator": ["middot", "pipe", "em-dash", "vertical-rule"],
            "density": ["dense", "spaced"],
        },
    ),
    "Ft3": Archetype(
        code="Ft3",
        category="footer",
        name="Index Style Category List",
        description="Three or four short columns, each headed by a category in small caps, holding 4–6 links each.",
        knobs={
            "columns": ["3", "4", "5"],
            "heading_style": ["small-caps", "italic", "monospace"],
            "bullet": ["hairline", "none"],
        },
    ),
    "Ft4": Archetype(
        code="Ft4",
        category="footer",
        name="Dense Typographic",
        description="One large block of text — credits, references, licence, address — in small monospace, fully justified.",
        knobs={
            "family": ["monospace", "serif", "sans"],
            "layout": ["single-block", "paragraphs", "log-style"],
            "includes": ["build-hash", "date", "attribution"],
        },
    ),
    "Ft5": Archetype(
        code="Ft5",
        category="footer",
        name="Statement",
        description="One large display sentence dominates the footer — a closing line, not a sitemap.",
        knobs={
            "sentence_width": ["28ch", "38ch", "50ch"],
            "wordmark_position": ["under-sentence", "top-right", "none"],
            "rule_above_meta": ["hairline", "double", "none"],
        },
    ),
    "Ft6": Archetype(
        code="Ft6",
        category="footer",
        name="Letter Close",
        description="Closes the page like a letter — 'Yours, the team. 2026.' Optional postscript beneath.",
        knobs={
            "signoff": ["italic", "roman", "monogram"],
            "postscript": ["yes", "no"],
            "width": ["40ch", "60ch", "80ch"],
        },
    ),
    "Ft7": Archetype(
        code="Ft7",
        category="footer",
        name="Newsletter First",
        description="The form (label + input + submit) is the primary element; everything else is muted small type beneath.",
        knobs={
            "layout": ["stacked", "inline", "split"],
            "submit_style": ["filled", "outline", "arrow-link"],
            "privacy_line": ["yes", "no"],
        },
    ),
    "Ft8": Archetype(
        code="Ft8",
        category="footer",
        name="Marquee Scroll",
        description="A horizontal infinite-scroll line of repeating tagline + dot separator.",
        knobs={
            "speed": ["24s", "32s", "48s"],
            "direction": ["left", "right", "alternate"],
            "glyph": ["middot", "em-dash", "slash"],
        },
    ),
    # Navigation (N1–N13)
    "N1": Archetype(
        code="N1",
        category="nav",
        name="Wordmark 2 Links",
        description="Top-of-page bar: wordmark left, two text links right. Minimal variant.",
        knobs={
            "position": ["left-right-split", "centred", "right-flush"],
            "links": ["text", "text+icon", "pill"],
            "sticky": ["yes", "no"],
        },
    ),
    "N1b": Archetype(
        code="N1b",
        category="nav",
        name="Canonical SaaS Three-Section",
        description="Wordmark-left · centred 4–6-link cluster · sign-in + filled CTA right. The dominant modern nav.",
        knobs={
            "centre_links": ["3", "4", "5–6"],
            "dropdowns": ["none", "1", "2"],
            "scroll": ["frost-on-scroll", "always-solid", "transparent-fixed"],
        },
    ),
    "N2": Archetype(
        code="N2",
        category="nav",
        name="Floating Chip",
        description="A small fixed chip in a corner — wordmark + a single action. Doesn't sit in document flow.",
        knobs={
            "anchor": ["top", "bottom", "top-right", "bottom-left"],
            "content": ["theme-picker", "search", "navigation"],
            "backdrop": ["blur", "solid", "none"],
        },
    ),
    "N3": Archetype(
        code="N3",
        category="nav",
        name="Side Rail",
        description="A thin vertical strip on the left edge — wordmark rotated, plus 2–3 dot-indicators for sections.",
        knobs={
            "side": ["left", "right"],
            "width": ["12ch", "16ch", "20ch"],
            "indicator": ["filled-bar", "text-only", "numbered"],
        },
    ),
    "N4": Archetype(
        code="N4",
        category="nav",
        name="Hidden Behind ⌘K",
        description="No visible nav. User opens a command palette via ⌘K to get anywhere.",
        knobs={
            "trigger": ["button", "keyboard-only", "both"],
            "surface": ["modal", "sheet", "spotlight"],
            "recents": ["shown", "hidden"],
        },
    ),
    "N5": Archetype(
        code="N5",
        category="nav",
        name="Floating Pill",
        description="A rounded full-pill nav, visibly detached from page edges, sitting ~var(--space-md) from top.",
        knobs={
            "width": ["content-sized", "max-720", "max-560"],
            "backdrop": ["blur+saturate", "solid", "subtle-gradient"],
            "anchor": ["top-centred", "top-right", "top-left"],
        },
    ),
    "N6": Archetype(
        code="N6",
        category="nav",
        name="Newspaper Masthead",
        description="Full-width header, large centred wordmark, thin issue/date line, double-rule below.",
        knobs={
            "issue_line": ["above-wordmark", "below-wordmark", "none"],
            "wordmark_size": ["3xl", "2xl", "xl"],
            "rule": ["double", "single", "none"],
        },
    ),
    "N7": Archetype(
        code="N7",
        category="nav",
        name="Brutal Slab",
        description="A heavy, full-width nav with a 2px solid border-bottom, all-caps wordmark, tracked uppercase links.",
        knobs={
            "border_weight": ["2px", "3px", "4px"],
            "letter_spacing": ["tracked-uppercase", "normal"],
            "cta": ["filled-slab", "outline-block", "text-only"],
        },
    ),
    "N8": Archetype(
        code="N8",
        category="nav",
        name="Terminal Command",
        description="A nav formatted as a CLI prompt: > studio --catalog --voice --get▮",
        knobs={
            "prompt": [">", "$", "~/$"],
            "cursor": ["in-line-at-end", "after-final-flag", "none"],
            "width": ["full-bleed", "content", "80ch"],
        },
    ),
    "N9": Archetype(
        code="N9",
        category="nav",
        name="Edge Aligned Minimal",
        description="Wordmark hard-left, single CTA hard-right, vast empty space between. No link row at all.",
        knobs={
            "cta_shape": ["outlined", "filled-pill", "text+arrow"],
            "wordmark": ["serif-italic", "sans", "monospace"],
            "padding_block": ["tight", "default", "spacious"],
        },
    ),
    "N10": Archetype(
        code="N10",
        category="nav",
        name="Floating on Scroll Morph",
        description="A sticky bar that morphs into a floating pill as the user scrolls past a threshold.",
        knobs={
            "morph_trigger": ["scroll-100px", "scroll-200px", "past-hero"],
            "end_state": ["floating-pill", "compact-bar", "minimal-chip"],
            "transition": ["smooth", "snap"],
        },
    ),
    "N11": Archetype(
        code="N11",
        category="nav",
        name="Mega-Menu Panel",
        description="Top bar triggers open a full-width multi-column panel with icon·title·description per item.",
        knobs={
            "columns": ["2", "3", "4"],
            "feature_cell": ["none", "promo-card", "code-sample"],
            "scrim": ["dim+blur", "dim-only", "none"],
        },
    ),
    "N12": Archetype(
        code="N12",
        category="nav",
        name="Banner + Retract",
        description="A coloured promo banner stacked above one real nav; banner retracts on scroll-down.",
        knobs={
            "banner_fill": ["solid", "gradient", "tint+ink"],
            "dismiss": ["yes", "none"],
            "bar_scroll": ["sticky", "also-frosts"],
        },
    ),
    "N13": Archetype(
        code="N13",
        category="nav",
        name="Inline ⌘K Search Pill",
        description="A visible search pill in the bar opening a spotlight modal with grouped, keyboard-navigable results.",
        knobs={
            "pill_placement": ["centred", "right-of-brand"],
            "result_groups": ["flat", "grouped"],
            "footer_hints": ["shown", "hidden"],
        },
    ),
}

ALL_ARCHETYPE_CODES: list[str] = list(ARCHETYPES.keys())
