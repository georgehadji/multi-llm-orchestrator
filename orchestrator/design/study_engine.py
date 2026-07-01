"""
Study Engine — Extract design DNA from screenshots or URLs.
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Analyses a visual design (screenshot or live URL) and extracts its DNA:
macrostructure, type-pairing, colour anchor, spacing scale, motion stance.

Can optionally emit a portable design.md for handoff to other AI tools.

Usage:
    engine = StudyEngine(client)
    dna = await engine.from_screenshot(Path("screenshot.png"))
    design_md = engine.emit_design_md(dna)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api_clients import UnifiedClient

logger = logging.getLogger(__name__)


@dataclass
class DesignDNA:
    """Extracted design DNA from a visual source."""

    source: str  # "screenshot:<path>" or "url:<url>"
    macrostructure: str = ""
    archetypes: list[str] = field(default_factory=list)
    type_pairing: str = ""  # e.g. "Space Grotesk + Inter"
    colour_anchor: str = ""  # e.g. "oklch(58% 0.20 256) cobalt"
    spacing_scale: str = ""  # e.g. "4-pt grid"
    motion_stance: str = "motion-cut"
    genre: str = ""
    notes: str = ""


class StudyEngine:
    """Extract design DNA from visual sources."""

    def __init__(self, client: "UnifiedClient | None" = None) -> None:
        self._client = client

    async def from_screenshot(self, image_path: Path) -> DesignDNA:
        """Extract DNA from a screenshot using a vision model.

        Args:
            image_path: Path to the screenshot image.

        Returns:
            DesignDNA with extracted properties.
        """
        if self._client is None:
            logger.warning("study_engine: no client available; returning empty DNA")
            return DesignDNA(source=f"screenshot:{image_path}")

        image_data = self._encode_image(image_path)
        prompt = (
            "You are a design DNA extractor. Analyse this screenshot and extract:\n"
            "1. Macrostructure (page shape): Bento Grid, Long Document, Marquee Hero, "
            "Stat-Led, Workbench, Manifesto, Photographic, etc.\n"
            "2. Type pairing (display + body fonts)\n"
            "3. Colour anchor (dominant accent hue, e.g. 'cobalt blue ~250°')\n"
            "4. Spacing scale (4-pt, 8-pt, or custom)\n"
            "5. Motion stance (motion-on or motion-cut)\n"
            "6. Genre (editorial, modern-minimal, atmospheric, playful, terminal)\n"
            "\n"
            'Return ONLY JSON: {"macrostructure": "...", "type_pairing": "...", '
            '"colour_anchor": "...", "spacing_scale": "...", "motion_stance": "...", '
            '"genre": "...", "notes": "..."}'
        )

        try:
            response = await self._client.call(
                model="anthropic/claude-sonnet-5",  # vision-capable
                prompt=prompt,
                system="You are a precise design analyst. Extract only what's visible.",
                max_tokens=500,
                temperature=0.1,
            )
            import json

            data = json.loads(response.text)
            return DesignDNA(
                source=f"screenshot:{image_path}",
                macrostructure=data.get("macrostructure", ""),
                type_pairing=data.get("type_pairing", ""),
                colour_anchor=data.get("colour_anchor", ""),
                spacing_scale=data.get("spacing_scale", ""),
                motion_stance=data.get("motion_stance", "motion-cut"),
                genre=data.get("genre", ""),
                notes=data.get("notes", ""),
            )
        except Exception as exc:
            logger.error("study_engine: vision analysis failed: %s", exc)
            return DesignDNA(source=f"screenshot:{image_path}")

    async def from_url(self, url: str) -> DesignDNA:
        """Extract DNA from a live URL by fetching HTML/CSS.

        Args:
            url: The live page URL.

        Returns:
            DesignDNA with extracted properties.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    html = await resp.text()
        except Exception as exc:
            logger.error("study_engine: failed to fetch %s: %s", url, exc)
            return DesignDNA(source=f"url:{url}")

        # Extract CSS from the page
        css = self._extract_css(html)

        # Analyse with LLM
        if self._client is not None:
            prompt = (
                f"Analyse this HTML/CSS and extract design DNA:\n\n"
                f"HTML (first 2000 chars):\n{html[:2000]}\n\n"
                f"CSS (first 2000 chars):\n{css[:2000]}\n\n"
                "Extract: macrostructure, type_pairing, colour_anchor, spacing_scale, "
                "motion_stance, genre.\n"
                "Return ONLY JSON."
            )
            try:
                response = await self._client.call(
                    model="anthropic/claude-sonnet-5",
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.1,
                )
                import json

                data = json.loads(response.text)
                return DesignDNA(
                    source=f"url:{url}",
                    macrostructure=data.get("macrostructure", ""),
                    type_pairing=data.get("type_pairing", ""),
                    colour_anchor=data.get("colour_anchor", ""),
                    spacing_scale=data.get("spacing_scale", ""),
                    motion_stance=data.get("motion_stance", "motion-cut"),
                    genre=data.get("genre", ""),
                    notes=data.get("notes", ""),
                )
            except Exception as exc:
                logger.error("study_engine: URL analysis failed: %s", exc)

        return DesignDNA(source=f"url:{url}")

    def emit_design_md(self, dna: DesignDNA) -> str:
        """Emit a portable design.md from extracted DNA.

        Args:
            dna: The extracted DesignDNA.

        Returns:
            design.md content as string.
        """
        return f"""# Design System — Extracted DNA

> Source: {dna.source}
> Extracted by Hallmark Study Engine

## Tokens

- **Paper**: (extract from source)
- **Ink**: (extract from source)
- **Accent**: {dna.colour_anchor}

## Typography

- **Display**: {dna.type_pairing.split("+")[0].strip() if "+" in dna.type_pairing else dna.type_pairing}
- **Body**: {dna.type_pairing.split("+")[1].strip() if "+" in dna.type_pairing else "Inter"}

## Structure

- **Macrostructure**: {dna.macrostructure}
- **Genre**: {dna.genre}
- **Motion stance**: {dna.motion_stance}
- **Spacing scale**: {dna.spacing_scale}

## Notes

{dna.notes}

## Usage

This file is a locked design system. Subsequent pages MUST defer to it.
Do not override tokens, fonts, or genre without explicit user confirmation.
"""

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """Encode an image to base64 for vision model."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _extract_css(html: str) -> str:
        """Extract inline CSS from HTML."""
        import re

        css_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL)
        css_links = re.findall(r'<link[^>]*href="([^"]*\.css)"', html, re.IGNORECASE)
        return "\n".join(css_blocks) + "\n/* linked: " + ", ".join(css_links) + " */"
