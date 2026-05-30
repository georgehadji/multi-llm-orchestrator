"""
Breakpoint Configuration — Specification + Builder Pattern
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Breakpoint configuration using Specification Pattern for matching rules
and Builder Pattern for fluent configuration.

Paradigm: OOP with Functional utilities
Patterns: Specification, Builder, Immutable Data

Usage:
    from orchestrator.breakpoints import BreakpointBuilder, BreakpointSpecification

    breakpoints = (BreakpointBuilder()
        .add_standard()
        .add_custom("ultrawide", 1920)
        .build())

    spec = BreakpointSpecification(lambda bp: bp.min_width >= 768)
    matches = spec.is_satisfied_by(breakpoints[2])  # md breakpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BreakpointConfig:
    """
    Immutable breakpoint configuration.

    Attributes:
        name: Breakpoint name (xs, sm, md, lg, xl, 2xl)
        min_width: Minimum width in pixels
        max_width: Maximum width in pixels (None = unlimited)
        container_max: Maximum container width at this breakpoint
        description: Human-readable description
    """

    name: str
    min_width: int
    max_width: Optional[int] = None
    container_max: Optional[int] = None
    description: str = ""

    def __post_init__(self):
        """Validate breakpoint configuration."""
        if self.min_width < 0:
            object.__setattr__(self, "min_width", 0)
        if self.max_width is not None and self.max_width <= self.min_width:
            object.__setattr__(self, "max_width", None)

    def to_css_media_query(self) -> str:
        """
        Convert to CSS media query.

        Returns:
            CSS media query string
        """
        if self.max_width:
            return f"@media (min-width: {self.min_width}px) and (max-width: {self.max_width}px)"
        else:
            return f"@media (min-width: {self.min_width}px)"

    def to_tailwind_prefix(self) -> str:
        """
        Convert to Tailwind CSS prefix.

        Returns:
            Tailwind prefix (e.g., "md:", "lg:")
        """
        prefix_map = {
            "xs": "",
            "sm": "sm:",
            "md": "md:",
            "lg": "lg:",
            "xl": "xl:",
            "2xl": "2xl:",
        }
        return prefix_map.get(self.name, "")

    def matches_width(self, width: int) -> bool:
        """
        Check if width matches this breakpoint.

        Args:
            width: Width to check in pixels

        Returns:
            True if width matches
        """
        if self.max_width:
            return self.min_width <= width < self.max_width
        else:
            return width >= self.min_width


# ═══════════════════════════════════════════════════════════════════
# SPECIFICATION PATTERN — BREAKPOINT SPECIFICATION
# ═══════════════════════════════════════════════════════════════════


class BreakpointSpecification:
    """
    Specification Pattern for breakpoint rules.

    Encapsulates business rules for breakpoint matching.

    Usage:
        # Create specifications
        tablet_spec = BreakpointSpecification(lambda bp: bp.min_width >= 768)
        desktop_spec = BreakpointSpecification(lambda bp: bp.name in ["lg", "xl", "2xl"])

        # Combine specifications
        large_screen_spec = tablet_spec.and_(desktop_spec)

        # Test against breakpoints
        matches = large_screen_spec.is_satisfied_by(md_breakpoint)
    """

    def __init__(self, predicate: Callable[[BreakpointConfig], bool]):
        """
        Initialize specification.

        Args:
            predicate: Function that tests breakpoint
        """
        self._predicate = predicate

    def is_satisfied_by(self, breakpoint: BreakpointConfig) -> bool:
        """
        Check if breakpoint satisfies specification.

        Args:
            breakpoint: Breakpoint to test

        Returns:
            True if satisfied
        """
        return self._predicate(breakpoint)

    def and_(self, other: "BreakpointSpecification") -> "BreakpointSpecification":
        """
        Combine specifications with AND.

        Args:
            other: Other specification

        Returns:
            Combined specification
        """
        return BreakpointSpecification(
            lambda bp: self.is_satisfied_by(bp) and other.is_satisfied_by(bp)
        )

    def or_(self, other: "BreakpointSpecification") -> "BreakpointSpecification":
        """
        Combine specifications with OR.

        Args:
            other: Other specification

        Returns:
            Combined specification
        """
        return BreakpointSpecification(
            lambda bp: self.is_satisfied_by(bp) or other.is_satisfied_by(bp)
        )

    def not_(self) -> "BreakpointSpecification":
        """
        Negate specification.

        Returns:
            Negated specification
        """
        return BreakpointSpecification(lambda bp: not self.is_satisfied_by(bp))

    @staticmethod
    def is_mobile() -> "BreakpointSpecification":
        """
        Create mobile breakpoint specification.

        Returns:
            Mobile specification
        """
        return BreakpointSpecification(lambda bp: bp.name in ["xs", "sm"])

    @staticmethod
    def is_tablet() -> "BreakpointSpecification":
        """
        Create tablet breakpoint specification.

        Returns:
            Tablet specification
        """
        return BreakpointSpecification(lambda bp: bp.name in ["md"])

    @staticmethod
    def is_desktop() -> "BreakpointSpecification":
        """
        Create desktop breakpoint specification.

        Returns:
            Desktop specification
        """
        return BreakpointSpecification(lambda bp: bp.name in ["lg", "xl", "2xl"])

    @staticmethod
    def min_width(width: int) -> "BreakpointSpecification":
        """
        Create minimum width specification.

        Args:
            width: Minimum width in pixels

        Returns:
            Width specification
        """
        return BreakpointSpecification(lambda bp: bp.min_width >= width)

    @staticmethod
    def max_width(width: int) -> "BreakpointSpecification":
        """
        Create maximum width specification.

        Args:
            width: Maximum width in pixels

        Returns:
            Width specification
        """
        return BreakpointSpecification(lambda bp: bp.max_width is None or bp.max_width <= width)


# ═══════════════════════════════════════════════════════════════════
# BUILDER PATTERN — BREAKPOINT BUILDER
# ═══════════════════════════════════════════════════════════════════


class BreakpointBuilder:
    """
    Builder Pattern for breakpoint configurations.

    Fluent interface for constructing breakpoint sets.

    Usage:
        breakpoints = (BreakpointBuilder()
            .add_standard()
            .add_custom("ultrawide", 1920, 2560)
            .build())
    """

    def __init__(self):
        """Initialize breakpoint builder."""
        self._breakpoints: List[BreakpointConfig] = []

    def _add(self, breakpoint: BreakpointConfig) -> "BreakpointBuilder":
        """Add breakpoint internally."""
        self._breakpoints.append(breakpoint)
        return self

    def add_standard(self) -> "BreakpointBuilder":
        """
        Add standard breakpoints.

        Standard breakpoints:
        - xs: 0px (mobile)
        - sm: 640px (small tablets)
        - md: 768px (tablets)
        - lg: 1024px (laptops)
        - xl: 1280px (desktops)
        - 2xl: 1536px (large desktops)

        Returns:
            Self for fluent interface
        """
        self._breakpoints.extend(
            [
                BreakpointConfig(
                    name="xs",
                    min_width=0,
                    max_width=639,
                    container_max=100,
                    description="Mobile devices (<640px)",
                ),
                BreakpointConfig(
                    name="sm",
                    min_width=640,
                    max_width=767,
                    container_max=640,
                    description="Small tablets (≥640px)",
                ),
                BreakpointConfig(
                    name="md",
                    min_width=768,
                    max_width=1023,
                    container_max=768,
                    description="Tablets (≥768px)",
                ),
                BreakpointConfig(
                    name="lg",
                    min_width=1024,
                    max_width=1279,
                    container_max=1024,
                    description="Laptops (≥1024px)",
                ),
                BreakpointConfig(
                    name="xl",
                    min_width=1280,
                    max_width=1535,
                    container_max=1280,
                    description="Desktops (≥1280px)",
                ),
                BreakpointConfig(
                    name="2xl",
                    min_width=1536,
                    max_width=None,
                    container_max=1536,
                    description="Large desktops (≥1536px)",
                ),
            ]
        )
        return self

    def add_minimal(self) -> "BreakpointBuilder":
        """
        Add minimal breakpoint set.

        Minimal breakpoints:
        - sm: 640px
        - md: 768px
        - lg: 1024px

        Returns:
            Self for fluent interface
        """
        self._breakpoints.extend(
            [
                BreakpointConfig(name="sm", min_width=640, max_width=767),
                BreakpointConfig(name="md", min_width=768, max_width=1023),
                BreakpointConfig(name="lg", min_width=1024, max_width=None),
            ]
        )
        return self

    def add_custom(
        self,
        name: str,
        min_width: int,
        max_width: int = None,
        container_max: int = None,
        description: str = "",
    ) -> "BreakpointBuilder":
        """
        Add custom breakpoint.

        Args:
            name: Breakpoint name
            min_width: Minimum width
            max_width: Maximum width (optional)
            container_max: Container max width (optional)
            description: Description

        Returns:
            Self for fluent interface
        """
        return self._add(
            BreakpointConfig(
                name=name,
                min_width=min_width,
                max_width=max_width,
                container_max=container_max,
                description=description,
            )
        )

    def add_mobile_first(self) -> "BreakpointBuilder":
        """
        Add mobile-first breakpoints.

        Optimized for mobile-first CSS.

        Returns:
            Self for fluent interface
        """
        return (
            self._add(
                BreakpointConfig(
                    name="mobile",
                    min_width=0,
                    max_width=767,
                    description="Mobile first (default)",
                )
            )
            ._add(
                BreakpointConfig(
                    name="tablet",
                    min_width=768,
                    max_width=1023,
                    description="Tablet",
                )
            )
            ._add(
                BreakpointConfig(
                    name="desktop",
                    min_width=1024,
                    max_width=None,
                    description="Desktop",
                )
            )
        )

    def add_container_queries(self) -> "BreakpointBuilder":
        """
        Add container query breakpoints.

        For CSS container queries (@container).

        Returns:
            Self for fluent interface
        """
        self._breakpoints.extend(
            [
                BreakpointConfig(name="narrow", min_width=0, max_width=399),
                BreakpointConfig(name="normal", min_width=400, max_width=599),
                BreakpointConfig(name="wide", min_width=600, max_width=899),
                BreakpointConfig(name="wider", min_width=900, max_width=None),
            ]
        )
        return self

    def build(self) -> List[BreakpointConfig]:
        """
        Build breakpoint configuration.

        Returns:
            List of breakpoint configurations
        """
        # Sort by min_width
        return sorted(self._breakpoints, key=lambda bp: bp.min_width)

    def build_dict(self) -> Dict[str, BreakpointConfig]:
        """
        Build breakpoint dictionary.

        Returns:
            Dictionary mapping name to config
        """
        return {bp.name: bp for bp in self.build()}

    def build_css(self) -> str:
        """
        Build CSS media queries.

        Returns:
            CSS string with media queries
        """
        css_parts = []

        for bp in self.build():
            css_parts.append(f"/* {bp.name}: {bp.description} */")
            css_parts.append(f"{bp.to_css_media_query()} {{ /* styles */ }}")
            css_parts.append("")

        return "\n".join(css_parts)

    def build_tailwind_config(self) -> str:
        """
        Build Tailwind CSS config.

        Returns:
            Tailwind config string
        """
        config_parts = ["module.exports = {", "  theme: {", "    screens: {"]

        for bp in self.build():
            if bp.max_width:
                config_parts.append(
                    f"      '{bp.name}': {{'min': '{bp.min_width}px', 'max': '{bp.max_width}px'}},"
                )
            else:
                config_parts.append(f"      '{bp.name}': '{bp.min_width}px',")

        config_parts.extend(["    }", "  }", "}"])

        return "\n".join(config_parts)


# ═══════════════════════════════════════════════════════════════════
# CONTAINER QUERY BUILDER
# ═══════════════════════════════════════════════════════════════════


class ContainerQueryBuilder:
    """
    Builder for CSS container queries.

    Container queries respond to container size, not viewport.
    """

    def __init__(self):
        """Initialize container query builder."""
        self._queries: List[str] = []

    def inline(self, min_width: int) -> "ContainerQueryBuilder":
        """Add inline-size query."""
        self._queries.append(f"@container (inline-size >= {min_width}px)")
        return self

    def block(self, min_height: int) -> "ContainerQueryBuilder":
        """Add block-size query."""
        self._queries.append(f"@container (block-size >= {min_height}px)")
        return self

    def style(self, property: str, value: str) -> "ContainerQueryBuilder":
        """Add style query."""
        self._queries.append(f"@container style({property}: {value})")
        return self

    def build(self, styles: str) -> str:
        """
        Build container query CSS.

        Args:
            styles: CSS styles to apply

        Returns:
            Complete container query CSS
        """
        if not self._queries:
            return f"@container {{ {styles} }}"

        return " ".join(self._queries) + f" {{ {styles} }}"


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def create_standard_breakpoints() -> List[BreakpointConfig]:
    """
    Create standard breakpoint set.

    Returns:
        Standard breakpoints
    """
    return BreakpointBuilder().add_standard().build()


def create_minimal_breakpoints() -> List[BreakpointConfig]:
    """
    Create minimal breakpoint set.

    Returns:
        Minimal breakpoints
    """
    return BreakpointBuilder().add_minimal().build()


def create_mobile_first_breakpoints() -> List[BreakpointConfig]:
    """
    Create mobile-first breakpoint set.

    Returns:
        Mobile-first breakpoints
    """
    return BreakpointBuilder().add_mobile_first().build()


def get_breakpoint_by_name(
    breakpoints: List[BreakpointConfig],
    name: str,
) -> Optional[BreakpointConfig]:
    """
    Get breakpoint by name.

    Args:
        breakpoints: Breakpoint list
        name: Breakpoint name

    Returns:
        Breakpoint config or None
    """
    for bp in breakpoints:
        if bp.name == name:
            return bp
    return None


def width_to_breakpoint(
    breakpoints: List[BreakpointConfig],
    width: int,
) -> Optional[BreakpointConfig]:
    """
    Find breakpoint for given width.

    Args:
        breakpoints: Breakpoint list
        width: Width in pixels

    Returns:
        Matching breakpoint or None
    """
    for bp in breakpoints:
        if bp.matches_width(width):
            return bp
    return None


def generate_responsive_css(
    property: str,
    values: Dict[str, str],
    breakpoints: List[BreakpointConfig] = None,
) -> str:
    """
    Generate responsive CSS.

    Args:
        property: CSS property
        values: Breakpoint → value mapping
        breakpoints: Breakpoint list (uses standard if None)

    Returns:
        Responsive CSS string
    """
    if breakpoints is None:
        breakpoints = create_standard_breakpoints()

    css_parts = []

    # Base value (mobile)
    if "xs" in values or "base" in values:
        base_value = values.get("xs", values.get("base"))
        css_parts.append(f"{property}: {base_value};")

    # Responsive values
    breakpoint_map = {bp.name: bp for bp in breakpoints}

    for name in ["sm", "md", "lg", "xl", "2xl"]:
        if name in values and name in breakpoint_map:
            bp = breakpoint_map[name]
            css_parts.append(f"{bp.to_css_media_query()} {{ {property}: {values[name]}; }}")

    return "\n".join(css_parts)