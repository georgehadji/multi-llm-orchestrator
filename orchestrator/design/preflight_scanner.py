"""
Pre-flight Scanner — Reads existing project files before generating.
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Scans a project directory for existing design system signals:
1. design.md / DESIGN.md — locked design system (highest priority)
2. Font stack — package.json, HTML links, Tailwind config
3. Palette — :root blocks, Tailwind theme, tokens.json
4. Motion stance — framer-motion, gsap, motion, lenis in deps
5. Spacing scale — Tailwind theme.extend.spacing
6. Framework — Next.js, Astro, Vue, Svelte, Remix, vanilla

Source: Hallmark design skill (pre-flight scan section)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class PreflightReport:
    """Results of a pre-flight project scan."""

    font_stack: str = ""
    palette_source: str = ""
    motion_stance: Literal["motion-on", "motion-cut"] = "motion-cut"
    spacing_scale: str = ""
    framework: str = ""
    design_md_present: bool = False
    design_md_content: str = ""
    findings: list[str] = field(default_factory=list)
    cached: bool = False
    conflicts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Pre-flight findings:"]
        if self.design_md_present:
            lines.append("· design.md detected — this is a system-managed project")
        if self.font_stack:
            lines.append(f"· Font stack: {self.font_stack}")
        if self.palette_source:
            lines.append(f"· Palette: {self.palette_source}")
        if self.motion_stance == "motion-on":
            lines.append("· Motion: motion-on (framer-motion / gsap / lenis detected)")
        else:
            lines.append("· Motion: motion-cut (no motion library detected)")
        if self.spacing_scale:
            lines.append(f"· Spacing: {self.spacing_scale}")
        if self.framework:
            lines.append(f"· Framework: {self.framework}")
        if self.conflicts:
            lines.append(f"· Conflicts: {', '.join(self.conflicts)}")
        return "\n".join(lines)


class PreflightScanner:
    """Scan a project directory for existing design system signals."""

    _MOTION_LIBS: frozenset[str] = frozenset({
        "framer-motion", "motion", "gsap", "lenis", "lottie-react",
        "@react-spring", "auto-animate",
    })

    _FRAMEWORKS: dict[str, str] = {
        "next": "Next.js",
        "astro": "Astro",
        "vue": "Vue",
        "svelte": "Svelte",
        "@sveltejs/kit": "SvelteKit",
        "@remix-run": "Remix",
    }

    async def scan(self, project_dir: Path | str) -> PreflightReport:
        """Scan *project_dir* for design system signals."""
        proj = Path(project_dir)
        report = PreflightReport()

        # 0. Check for design.md
        design_md = self._find_design_md(proj)
        if design_md:
            report.design_md_present = True
            try:
                report.design_md_content = design_md.read_text(encoding="utf-8")[:2000]
                report.findings.append(f"design.md found at {design_md}")
            except OSError as exc:
                report.findings.append(f"design.md found but unreadable: {exc}")

        # 1. Read package.json
        pkg = self._read_package_json(proj)
        if pkg:
            report.font_stack = self._detect_fonts(pkg, proj)
            report.motion_stance = self._detect_motion(pkg)
            report.framework = self._detect_framework(pkg)
            report.findings.append(f"package.json parsed")

        # 2. Detect palette from CSS / Tailwind
        report.palette_source = self._detect_palette(proj)

        # 3. Detect spacing scale
        report.spacing_scale = self._detect_spacing(proj)

        # 4. Detect conflicts
        report.conflicts = self._detect_conflicts(proj, pkg)

        logger.debug("preflight_scanner: %s", report.summary().replace("\n", " | "))
        return report

    def _find_design_md(self, proj: Path) -> Path | None:
        for name in ("design.md", "DESIGN.md", "Design.md"):
            p = proj / name
            if p.is_file():
                return p
        return None

    def _read_package_json(self, proj: Path) -> dict | None:
        p = proj / "package.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _detect_fonts(self, pkg: dict, proj: Path) -> str:
        """Detect font stack from package.json and project files."""
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        fonts: list[str] = []

        # next/font
        if "next" in deps:
            fonts.append("next/font")
        # @fontsource
        for dep in deps:
            if dep.startswith("@fontsource/"):
                fonts.append(dep.replace("@fontsource/", ""))
        # geist
        if "geist" in deps:
            fonts.append("Geist")

        # Search CSS/HTML for Google Fonts links
        for pattern in ("**/*.css", "**/*.html", "**/*.tsx", "**/*.jsx", "**/*.vue"):
            for fp in proj.glob(pattern):
                try:
                    text = fp.read_text(encoding="utf-8")
                    if "fonts.googleapis.com" in text:
                        # Extract font names
                        matches = re.findall(
                            r"family=([^:&]+)", text
                        )
                        for m in matches:
                            fonts.append(m.replace("+", " "))
                    if "@font-face" in text:
                        family_matches = re.findall(
                            r"font-family:\s*['\"]?([^;'\"]+)['\"]?", text
                        )
                        fonts.extend(f.strip() for f in family_matches)
                except (OSError, UnicodeDecodeError):
                    continue
                if len(fonts) >= 3:
                    break
            if len(fonts) >= 3:
                break

        return ", ".join(dict.fromkeys(fonts)) if fonts else ""

    def _detect_motion(self, pkg: dict) -> Literal["motion-on", "motion-cut"]:
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        if any(lib in deps for lib in self._MOTION_LIBS):
            return "motion-on"
        return "motion-cut"

    def _detect_framework(self, pkg: dict) -> str:
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        for key, name in self._FRAMEWORKS.items():
            if key in deps:
                return name
        if "react" in deps:
            return "React"
        return "vanilla HTML"

    def _detect_palette(self, proj: Path) -> str:
        """Detect palette from CSS custom properties, Tailwind config, or tokens.json."""
        # Check tokens.json / design-tokens.json
        for name in ("tokens.json", "design-tokens.json", "design.tokens.json"):
            p = proj / name
            if p.is_file():
                return f"{name} at project root"

        # Check CSS files for :root blocks
        for fp in proj.glob("**/*.css"):
            try:
                text = fp.read_text(encoding="utf-8")
                if ":root" in text and ("--color" in text or "oklch" in text or "#" in text):
                    return f"CSS custom properties ({fp.relative_to(proj)})"
            except (OSError, UnicodeDecodeError):
                continue

        # Check tailwind config
        for pattern in ("tailwind.config.js", "tailwind.config.ts", "tailwind.config.mjs"):
            p = proj / pattern
            if p.is_file():
                return f"Tailwind config ({pattern})"

        return ""

    def _detect_spacing(self, proj: Path) -> str:
        """Detect spacing scale from Tailwind config or CSS."""
        for pattern in ("tailwind.config.js", "tailwind.config.ts"):
            p = proj / pattern
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8")
                    if "spacing" in text:
                        return f"Tailwind extend.spacing ({pattern})"
                except (OSError, UnicodeDecodeError):
                    continue

        for fp in proj.glob("**/*.css"):
            try:
                text = fp.read_text(encoding="utf-8")
                if "--space-" in text or "spacing" in text:
                    return f"CSS spacing tokens ({fp.relative_to(proj)})"
            except (OSError, UnicodeDecodeError):
                continue

        return ""

    def _detect_conflicts(self, proj: Path, pkg: dict | None) -> list[str]:
        """Detect conflicting signals (e.g. Geist imported but Inter hard-coded)."""
        conflicts: list[str] = []
        if pkg is None:
            return conflicts

        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

        # Check for conflicting font declarations in CSS
        if "geist" in deps:
            for fp in proj.glob("**/*.css"):
                try:
                    text = fp.read_text(encoding="utf-8")
                    if 'font-family: "Inter"' in text or "font-family: Inter" in text:
                        conflicts.append(
                            f"Geist imported but Inter hard-coded in {fp.relative_to(proj)}"
                        )
                        break
                except (OSError, UnicodeDecodeError):
                    continue

        # Check for motion lib installed but not used
        motion_installed = any(lib in deps for lib in self._MOTION_LIBS)
        motion_used = False
        for fp in proj.glob("**/*.{tsx,jsx,vue,svelte}"):
            try:
                text = fp.read_text(encoding="utf-8")
                if "motion." in text or "framer-motion" in text or "gsap" in text:
                    motion_used = True
                    break
            except (OSError, UnicodeDecodeError):
                continue
        if motion_installed and not motion_used:
            conflicts.append("Motion library installed but no usage found in source files")

        return conflicts
