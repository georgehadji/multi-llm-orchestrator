"""
Responsive Layout Generator — Composite + Strategy Pattern
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Responsive layout generation using Composite Pattern for nested components
and Strategy Pattern for different layout types (Grid, Flex, Box).

Paradigm: OOP with Functional utilities
Patterns: Composite, Strategy, Builder, Immutable Data

Usage:
    from orchestrator.responsive_layouts import ContainerComponent, GridLayoutComponent

    layout = (ContainerComponent()
        .add_child(GridLayoutComponent(columns="12"))
        .add_child(FlexLayoutComponent(direction="row"))
        .render())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


class Breakpoint(str, Enum):
    """Responsive breakpoints (mobile-first)."""

    XS = "xs"  # Default (mobile, <640px)
    SM = "sm"  # Small tablets (≥640px)
    MD = "md"  # Tablets (≥768px)
    LG = "lg"  # Laptops (≥1024px)
    XL = "xl"  # Desktops (≥1280px)
    XXL = "2xl"  # Large desktops (≥1536px)


@dataclass(frozen=True)
class ResponsiveValue:
    """
    Immutable responsive value (mobile-first).

    Attributes:
        xs: Default value (mobile)
        sm: Small tablets
        md: Tablets
        lg: Laptops
        xl: Desktops
        xxl: Large desktops

    Example:
        ResponsiveValue(xs="1rem", md="2rem", lg="3rem")
        # Mobile: 1rem, Tablet: 2rem, Desktop: 3rem
    """

    xs: Any
    sm: Optional[Any] = None
    md: Optional[Any] = None
    lg: Optional[Any] = None
    xl: Optional[Any] = None
    xxl: Optional[Any] = None

    def to_css(self, property_name: str) -> str:
        """
        Convert to CSS media queries.

        Args:
            property_name: CSS property name

        Returns:
            CSS string with media queries
        """
        breakpoints = {
            Breakpoint.SM: "640px",
            Breakpoint.MD: "768px",
            Breakpoint.LG: "1024px",
            Breakpoint.XL: "1280px",
            Breakpoint.XXL: "1536px",
        }

        css_parts = []

        # Base value (mobile-first)
        if self.xs is not None:
            css_parts.append(f"{property_name}: {self.xs};")

        # Media queries for larger breakpoints
        for bp_name, bp_value in breakpoints.items():
            value = getattr(self, bp_name.value, None)
            if value is not None:
                css_parts.append(f"@media (min-width: {bp_value}) {{ {property_name}: {value}; }}")

        return " ".join(css_parts)

    def to_tailwind(self, property_prefix: str) -> str:
        """
        Convert to Tailwind CSS classes.

        Args:
            property_prefix: Tailwind property prefix (e.g., "p" for padding)

        Returns:
            Tailwind class string
        """
        classes = []

        # Base value (mobile)
        if self.xs is not None:
            classes.append(f"{property_prefix}-{self.xs}")

        # Responsive prefixes
        if self.sm is not None:
            classes.append(f"sm:{property_prefix}-{self.sm}")
        if self.md is not None:
            classes.append(f"md:{property_prefix}-{self.md}")
        if self.lg is not None:
            classes.append(f"lg:{property_prefix}-{self.lg}")
        if self.xl is not None:
            classes.append(f"xl:{property_prefix}-{self.xl}")
        if self.xxl is not None:
            classes.append(f"2xl:{property_prefix}-{self.xxl}")

        return " ".join(classes)


@dataclass(frozen=True)
class Spacing:
    """Immutable spacing configuration."""

    top: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("0"))
    right: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("0"))
    bottom: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("0"))
    left: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("0"))

    @classmethod
    def all(cls, value: Any) -> "Spacing":
        """Create uniform spacing."""
        rv = value if isinstance(value, ResponsiveValue) else ResponsiveValue(xs=value)
        return cls(top=rv, right=rv, bottom=rv, left=rv)

    @classmethod
    def symmetric(cls, vertical: Any, horizontal: Any) -> "Spacing":
        """Create symmetric spacing."""
        v = vertical if isinstance(vertical, ResponsiveValue) else ResponsiveValue(xs=vertical)
        h = (
            horizontal
            if isinstance(horizontal, ResponsiveValue)
            else ResponsiveValue(xs=horizontal)
        )
        return cls(top=v, bottom=v, left=h, right=h)


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE PATTERN — LAYOUT COMPONENT INTERFACE
# ═══════════════════════════════════════════════════════════════════


class LayoutComponent(ABC):
    """
    Composite Pattern for layout components.

    Treats individual components and compositions uniformly.
    """

    @abstractmethod
    def render(self) -> str:
        """
        Render component to HTML/CSS string.

        Returns:
            HTML/CSS string
        """
        pass

    @abstractmethod
    def add_child(self, child: "LayoutComponent") -> "LayoutComponent":
        """
        Add child component.

        Args:
            child: Child component

        Returns:
            Self for fluent interface
        """
        pass

    @abstractmethod
    def get_children(self) -> List["LayoutComponent"]:
        """
        Get child components.

        Returns:
            List of child components
        """
        pass


# ═══════════════════════════════════════════════════════════════════
# CONCRETE COMPONENTS
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ContainerComponent(LayoutComponent):
    """
    Composite container component.

    Attributes:
        class_name: CSS class name
        max_width: Maximum container width
        padding: Container padding
        children: Child components
    """

    class_name: str = "container"
    max_width: Optional[str] = None
    padding: Spacing = field(default_factory=lambda: Spacing.all("1rem"))
    children: List[LayoutComponent] = field(default_factory=list)

    def render(self) -> str:
        """Render container with all children."""
        # Build style attribute
        styles = []

        if self.max_width:
            styles.append(f"max-width: {self.max_width}")
        if self.padding:
            styles.append(
                f"padding: {self.padding.top.xs} {self.padding.right.xs} {self.padding.bottom.xs} {self.padding.left.xs}"
            )

        style_attr = f' style="{"; ".join(styles)}"' if styles else ""

        # Render children
        children_html = "".join(child.render() for child in self.children)

        return f'<div class="{self.class_name}"{style_attr}>{children_html}</div>'

    def add_child(self, child: LayoutComponent) -> "ContainerComponent":
        """Add child component (fluent interface)."""
        self.children.append(child)
        return self

    def get_children(self) -> List[LayoutComponent]:
        """Get child components."""
        return self.children

    # Fluent interface methods
    def with_class(self, class_name: str) -> "ContainerComponent":
        """Set CSS class (fluent interface)."""
        object.__setattr__(self, "class_name", class_name)
        return self

    def with_max_width(self, max_width: str) -> "ContainerComponent":
        """Set maximum width (fluent interface)."""
        object.__setattr__(self, "max_width", max_width)
        return self

    def with_padding(self, padding: Spacing) -> "ContainerComponent":
        """Set padding (fluent interface)."""
        object.__setattr__(self, "padding", padding)
        return self


@dataclass
class GridLayoutComponent(LayoutComponent):
    """
    Strategy: Grid layout component.

    Attributes:
        columns: Number of columns or template
        gap: Grid gap
        children: Grid items
        responsive: Enable responsive columns
    """

    columns: int | str = 12
    gap: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("1rem"))
    children: List[LayoutComponent] = field(default_factory=list)
    responsive: bool = True

    def render(self) -> str:
        """Render grid layout."""
        # Build grid style
        if isinstance(self.columns, int):
            grid_template = f"repeat({self.columns}, 1fr)"
        else:
            grid_template = self.columns

        styles = [
            f"display: grid",
            f"grid-template-columns: {grid_template}",
            f"gap: {self.gap.xs}",
        ]

        # Add responsive breakpoints
        if self.responsive and isinstance(self.columns, int):
            styles.append(self.gap.to_css("gap"))

        style_attr = f' style="{"; ".join(styles)}"'

        # Render children
        children_html = "".join(
            f'<div class="grid-item">{child.render()}</div>' for child in self.children
        )

        return f'<div class="grid-layout"{style_attr}>{children_html}</div>'

    def add_child(self, child: LayoutComponent) -> "GridLayoutComponent":
        """Add child component (fluent interface)."""
        self.children.append(child)
        return self

    def get_children(self) -> List[LayoutComponent]:
        """Get child components."""
        return self.children


@dataclass
class FlexLayoutComponent(LayoutComponent):
    """
    Strategy: Flexbox layout component.

    Attributes:
        direction: Flex direction (row, column, etc.)
        justify: Justify content
        align: Align items
        wrap: Flex wrap
        gap: Flex gap
        children: Flex items
    """

    direction: str = "row"
    justify: str = "flex-start"
    align: str = "stretch"
    wrap: str = "nowrap"
    gap: ResponsiveValue = field(default_factory=lambda: ResponsiveValue("0.5rem"))
    children: List[LayoutComponent] = field(default_factory=list)

    def render(self) -> str:
        """Render flexbox layout."""
        styles = [
            f"display: flex",
            f"flex-direction: {self.direction}",
            f"justify-content: {self.justify}",
            f"align-items: {self.align}",
            f"flex-wrap: {self.wrap}",
            f"gap: {self.gap.xs}",
        ]

        style_attr = f' style="{"; ".join(styles)}"'

        # Render children
        children_html = "".join(
            f'<div class="flex-item">{child.render()}</div>' for child in self.children
        )

        return f'<div class="flex-layout"{style_attr}>{children_html}</div>'

    def add_child(self, child: LayoutComponent) -> "FlexLayoutComponent":
        """Add child component (fluent interface)."""
        self.children.append(child)
        return self

    def get_children(self) -> List[LayoutComponent]:
        """Get child components."""
        return self.children


@dataclass
class BoxComponent(LayoutComponent):
    """
    Simple box component (leaf in composite tree).

    Attributes:
        content: HTML content
        class_name: CSS class name
        styles: Inline styles
    """

    content: str = ""
    class_name: str = "box"
    styles: Dict[str, str] = field(default_factory=dict)
    children: List[LayoutComponent] = field(default_factory=list)

    def render(self) -> str:
        """Render box component."""
        style_attr = ""
        if self.styles:
            style_str = "; ".join(f"{k}: {v}" for k, v in self.styles.items())
            style_attr = f' style="{style_str}"'

        children_html = "".join(child.render() for child in self.children)

        return f'<div class="{self.class_name}"{style_attr}>{self.content}{children_html}</div>'

    def add_child(self, child: LayoutComponent) -> "BoxComponent":
        """Add child component (fluent interface)."""
        self.children.append(child)
        return self

    def get_children(self) -> List[LayoutComponent]:
        """Get child components."""
        return self.children


@dataclass
class ResponsiveNavigationComponent(LayoutComponent):
    """
    Responsive navigation with mobile hamburger menu.

    Attributes:
        brand: Brand/logo text
        links: Navigation links
        breakpoint: Mobile breakpoint
    """

    brand: str = "Brand"
    links: List[Dict[str, str]] = field(default_factory=list)
    breakpoint: str = "md"
    children: List[LayoutComponent] = field(default_factory=list)

    def render(self) -> str:
        """Render responsive navigation."""
        links_html = "".join(
            f'<a href="{link.get("href", "#")}" class="nav-link">{link.get("text", "Link")}</a>'
            for link in self.links
        )

        return f"""
<nav class="responsive-nav" data-breakpoint="{self.breakpoint}">
  <div class="nav-container">
    <div class="nav-brand">{self.brand}</div>
    <button class="nav-toggle" aria-label="Toggle navigation">
      <span class="hamburger"></span>
    </button>
    <div class="nav-menu">
      {links_html}
    </div>
  </div>
  <style>
    .responsive-nav {{ background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .nav-container {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem; }}
    .nav-brand {{ font-size: 1.5rem; font-weight: bold; }}
    .nav-menu {{ display: flex; gap: 1rem; }}
    .nav-toggle {{ display: none; background: none; border: none; cursor: pointer; }}
    .hamburger {{ display: block; width: 24px; height: 2px; background: #333; position: relative; }}
    .hamburger::before, .hamburger::after {{ content: ""; position: absolute; width: 24px; height: 2px; background: #333; }}
    .hamburger::before {{ top: -8px; }}
    .hamburger::after {{ top: 8px; }}
    
    @media (max-width: 768px) {{
      .nav-toggle {{ display: block; }}
      .nav-menu {{ display: none; flex-direction: column; position: absolute; top: 100%; left: 0; right: 0; background: #fff; padding: 1rem; }}
      .nav-menu.active {{ display: flex; }}
    }}
  </style>
</nav>
"""

    def add_child(self, child: LayoutComponent) -> "ResponsiveNavigationComponent":
        """Add child component (fluent interface)."""
        self.children.append(child)
        return self

    def get_children(self) -> List[LayoutComponent]:
        """Get child components."""
        return self.children


# ═══════════════════════════════════════════════════════════════════
# LAYOUT BUILDER
# ═══════════════════════════════════════════════════════════════════


class LayoutBuilder:
    """
    Builder Pattern for constructing layouts.

    Fluent interface for building complex responsive layouts.

    Usage:
        layout = (LayoutBuilder()
            .container(max_width="1200px")
            .grid(columns=12)
            .box("Content")
            .build())
    """

    def __init__(self):
        """Initialize layout builder."""
        self._root: Optional[LayoutComponent] = None
        self._current: Optional[LayoutComponent] = None
        self._stack: List[LayoutComponent] = []

    def container(
        self,
        max_width: str = None,
        padding: str = "1rem",
        class_name: str = "container",
    ) -> "LayoutBuilder":
        """Add container (fluent interface)."""
        container = ContainerComponent(
            max_width=max_width,
            padding=Spacing.all(padding),
            class_name=class_name,
        )

        if self._root is None:
            self._root = container
            self._current = container
        else:
            self._current.add_child(container)
            self._stack.append(self._current)
            self._current = container

        return self

    def grid(
        self,
        columns: int = 12,
        gap: str = "1rem",
        responsive: bool = True,
    ) -> "LayoutBuilder":
        """Add grid layout (fluent interface)."""
        grid = GridLayoutComponent(
            columns=columns,
            gap=ResponsiveValue(xs=gap),
            responsive=responsive,
        )

        if self._root is None:
            self._root = grid
            self._current = grid
        else:
            self._current.add_child(grid)
            self._stack.append(self._current)
            self._current = grid

        return self

    def flex(
        self,
        direction: str = "row",
        justify: str = "center",
        align: str = "center",
        gap: str = "0.5rem",
    ) -> "LayoutBuilder":
        """Add flexbox layout (fluent interface)."""
        flex = FlexLayoutComponent(
            direction=direction,
            justify=justify,
            align=align,
            gap=ResponsiveValue(xs=gap),
        )

        if self._root is None:
            self._root = flex
            self._current = flex
        else:
            self._current.add_child(flex)
            self._stack.append(self._current)
            self._current = flex

        return self

    def box(
        self,
        content: str = "",
        class_name: str = "box",
        styles: Dict[str, str] = None,
    ) -> "LayoutBuilder":
        """Add box component (fluent interface)."""
        box = BoxComponent(
            content=content,
            class_name=class_name,
            styles=styles or {},
        )

        if self._root is None:
            self._root = box
            self._current = box
        else:
            self._current.add_child(box)

        return self

    def nav(
        self,
        brand: str = "Brand",
        links: List[Dict[str, str]] = None,
        breakpoint: str = "md",
    ) -> "LayoutBuilder":
        """Add responsive navigation (fluent interface)."""
        nav = ResponsiveNavigationComponent(
            brand=brand,
            links=links or [],
            breakpoint=breakpoint,
        )

        if self._root is None:
            self._root = nav
            self._current = nav
        else:
            self._current.add_child(nav)

        return self

    def end(self) -> "LayoutBuilder":
        """End current component and return to parent (fluent interface)."""
        if self._stack:
            self._current = self._stack.pop()
        return self

    def build(self) -> str:
        """
        Build final layout.

        Returns:
            Rendered HTML string
        """
        if self._root is None:
            return ""
        return self._root.render()


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def create_responsive_container(
    max_width: str = "1200px",
    padding: str = "1rem",
) -> ContainerComponent:
    """
    Create responsive container.

    Args:
        max_width: Maximum container width
        padding: Container padding

    Returns:
        ContainerComponent
    """
    return ContainerComponent(
        max_width=max_width,
        padding=Spacing.all(padding),
    )


def create_responsive_grid(
    columns: int = 12,
    gap: str = "1rem",
    responsive: bool = True,
) -> GridLayoutComponent:
    """
    Create responsive grid.

    Args:
        columns: Number of columns
        gap: Grid gap
        responsive: Enable responsive columns

    Returns:
        GridLayoutComponent
    """
    return GridLayoutComponent(
        columns=columns,
        gap=ResponsiveValue(xs=gap),
        responsive=responsive,
    )


def create_responsive_navigation(
    brand: str = "Brand",
    links: List[Dict[str, str]] = None,
) -> ResponsiveNavigationComponent:
    """
    Create responsive navigation.

    Args:
        brand: Brand text
        links: Navigation links

    Returns:
        ResponsiveNavigationComponent
    """
    return ResponsiveNavigationComponent(
        brand=brand,
        links=links or [],
    )


def create_mobile_first_layout() -> LayoutBuilder:
    """
    Create mobile-first layout builder.

    Returns:
        LayoutBuilder instance
    """
    return LayoutBuilder()