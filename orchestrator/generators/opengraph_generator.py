"""
OpenGraph Generator — Builder Pattern for Metadata
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Builder pattern for constructing OpenGraph and SEO metadata.
Fluent API for chainable configuration.

Paradigm: Builder Pattern (fluent interface, method chaining)

Usage:
    from orchestrator.generators.opengraph_generator import OpenGraphBuilder

    metadata = (OpenGraphBuilder()
        .with_title("My App")
        .with_description("Best app ever")
        .with_image("https://example.com/og.jpg")
        .with_url("https://example.com")
        .with_twitter("@myapp")
        .build())

    html = metadata.to_html()
    json_ld = metadata.to_json_ld()
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json


@dataclass
class OpenGraphMetadata:
    """
    Immutable OpenGraph metadata container.

    Built using OpenGraphBuilder (builder pattern).

    Attributes:
        title: Page title (<60 chars for SEO)
        description: Page description (<160 chars for SEO)
        image: OG image URL (1200x630 minimum)
        url: Canonical URL
        type: OG type (website, article, product)
        site_name: Site name
        locale: Locale (default: en_US)
        keywords: SEO keywords
        author: Content author
        twitter_site: Twitter handle (@username)
        twitter_creator: Creator Twitter handle
        robots: Robots directive
        schema_type: Schema.org type
    """

    title: str
    description: str
    image: str
    url: str
    type: str = "website"
    site_name: str = ""
    locale: str = "en_US"
    keywords: list[str] = field(default_factory=list)
    author: str = ""
    twitter_site: str = ""
    twitter_creator: str = ""
    robots: str = "index, follow"
    schema_type: str = "WebSite"

    def __post_init__(self):
        """Validate metadata after initialization."""
        # Title length check
        if len(self.title) > 60:
            raise ValueError(f"Title too long ({len(self.title)} chars, max 60)")

        # Description length check
        if len(self.description) > 160:
            raise ValueError(f"Description too long ({len(self.description)} chars, max 160)")

        # URL validation
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")

        # Image validation
        if not self.image.startswith(("http://", "https://")):
            raise ValueError("Image URL must be absolute (start with http:// or https://)")

    def to_html(self) -> str:
        """
        Generate HTML meta tags.

        Returns:
            HTML string with all meta tags
        """
        tags = []

        # Basic meta tags
        tags.append('  <meta charset="UTF-8">')
        tags.append('  <meta name="viewport" content="width=device-width, initial-scale=1.0">')

        # SEO meta tags
        tags.append(f"  <title>{self._escape(self.title)}</title>")
        tags.append(f'  <meta name="description" content="{self._escape(self.description)}">')

        if self.keywords:
            tags.append(
                f'  <meta name="keywords" content="{self._escape(", ".join(self.keywords))}">'
            )

        if self.author:
            tags.append(f'  <meta name="author" content="{self._escape(self.author)}">')

        tags.append(f'  <meta name="robots" content="{self.robots}">')

        # Canonical URL
        tags.append(f'  <link rel="canonical" href="{self._escape(self.url)}">')

        # OpenGraph meta tags
        tags.append(f'  <meta property="og:title" content="{self._escape(self.title)}">')
        tags.append(
            f'  <meta property="og:description" content="{self._escape(self.description)}">'
        )
        tags.append(f'  <meta property="og:image" content="{self._escape(self.image)}">')
        tags.append(f'  <meta property="og:url" content="{self._escape(self.url)}">')
        tags.append(f'  <meta property="og:type" content="{self.type}">')

        if self.site_name:
            tags.append(
                f'  <meta property="og:site_name" content="{self._escape(self.site_name)}">'
            )

        if self.locale:
            tags.append(f'  <meta property="og:locale" content="{self._escape(self.locale)}">')

        # Twitter Card meta tags
        tags.append(f'  <meta name="twitter:card" content="summary_large_image">')
        tags.append(f'  <meta name="twitter:title" content="{self._escape(self.title)}">')
        tags.append(
            f'  <meta name="twitter:description" content="{self._escape(self.description)}">'
        )
        tags.append(f'  <meta name="twitter:image" content="{self._escape(self.image)}">')

        if self.twitter_site:
            twitter_handle = self.twitter_site.lstrip("@")
            tags.append(f'  <meta name="twitter:site" content="@{twitter_handle}">')

        if self.twitter_creator:
            creator_handle = self.twitter_creator.lstrip("@")
            tags.append(f'  <meta name="twitter:creator" content="@{creator_handle}">')

        # Favicons
        tags.append('  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">')
        tags.append('  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">')
        tags.append('  <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">')

        return "\n".join(tags)

    def to_json_ld(self) -> str:
        """
        Generate Schema.org JSON-LD structured data.

        Returns:
            JSON-LD script tag
        """
        schema = {
            "@context": "https://schema.org",
            "@type": self.schema_type,
            "name": self.title,
            "description": self.description,
            "url": self.url,
        }

        if self.site_name:
            schema["publisher"] = {
                "@type": "Organization",
                "name": self.site_name,
            }

        if self.author:
            schema["author"] = {
                "@type": "Person" if not self.site_name else "Organization",
                "name": self.author,
            }

        # Add image if present
        if self.image:
            schema["image"] = self.image

        return f'<script type="application/ld+json">\n{json.dumps(schema, indent=2)}\n</script>'

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "image": self.image,
            "url": self.url,
            "type": self.type,
            "site_name": self.site_name,
            "locale": self.locale,
            "keywords": self.keywords,
            "author": self.author,
            "twitter_site": self.twitter_site,
            "twitter_creator": self.twitter_creator,
            "robots": self.robots,
            "schema_type": self.schema_type,
        }

    def _escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


class OpenGraphBuilder:
    """
    Builder for OpenGraphMetadata (fluent interface).

    Usage:
        metadata = (OpenGraphBuilder()
            .with_title("My App")
            .with_description("Best app ever")
            .with_image("https://example.com/og.jpg")
            .with_url("https://example.com")
            .with_twitter("@myapp")
            .build())
    """

    def __init__(self):
        """Initialize builder with defaults."""
        self._title = ""
        self._description = ""
        self._image = ""
        self._url = ""
        self._type = "website"
        self._site_name = ""
        self._locale = "en_US"
        self._keywords: list[str] = []
        self._author = ""
        self._twitter_site = ""
        self._twitter_creator = ""
        self._robots = "index, follow"
        self._schema_type = "WebSite"

    def with_title(self, title: str) -> OpenGraphBuilder:
        """
        Set page title.

        Args:
            title: Page title (max 60 chars for SEO)

        Returns:
            Self for method chaining
        """
        self._title = title
        return self

    def with_description(self, description: str) -> OpenGraphBuilder:
        """
        Set page description.

        Args:
            description: Page description (max 160 chars for SEO)

        Returns:
            Self for method chaining
        """
        self._description = description
        return self

    def with_image(self, image: str) -> OpenGraphBuilder:
        """
        Set OpenGraph image.

        Args:
            image: Absolute URL to image (1200x630 minimum recommended)

        Returns:
            Self for method chaining
        """
        self._image = image
        return self

    def with_url(self, url: str) -> OpenGraphBuilder:
        """
        Set canonical URL.

        Args:
            url: Absolute URL (must start with http:// or https://)

        Returns:
            Self for method chaining
        """
        self._url = url
        return self

    def with_type(self, og_type: str) -> OpenGraphBuilder:
        """
        Set OpenGraph type.

        Args:
            og_type: Type (website, article, product, etc.)

        Returns:
            Self for method chaining
        """
        self._type = og_type
        return self

    def with_site_name(self, site_name: str) -> OpenGraphBuilder:
        """
        Set site name.

        Args:
            site_name: Name of the website

        Returns:
            Self for method chaining
        """
        self._site_name = site_name
        return self

    def with_locale(self, locale: str) -> OpenGraphBuilder:
        """
        Set locale.

        Args:
            locale: Locale code (e.g., "en_US", "el_GR")

        Returns:
            Self for method chaining
        """
        self._locale = locale
        return self

    def with_keywords(self, *keywords: str) -> OpenGraphBuilder:
        """
        Set SEO keywords.

        Args:
            *keywords: Variable number of keyword strings

        Returns:
            Self for method chaining
        """
        self._keywords = list(keywords)
        return self

    def with_author(self, author: str) -> OpenGraphBuilder:
        """
        Set content author.

        Args:
            author: Author name

        Returns:
            Self for method chaining
        """
        self._author = author
        return self

    def with_twitter(self, twitter_site: str, twitter_creator: str = "") -> OpenGraphBuilder:
        """
        Set Twitter handles.

        Args:
            twitter_site: Site Twitter handle (@username or username)
            twitter_creator: Creator Twitter handle (optional)

        Returns:
            Self for method chaining
        """
        self._twitter_site = twitter_site.lstrip("@")
        self._twitter_creator = twitter_creator.lstrip("@")
        return self

    def with_robots(self, robots: str) -> OpenGraphBuilder:
        """
        Set robots directive.

        Args:
            robots: Robots directive (e.g., "index, follow" or "noindex, nofollow")

        Returns:
            Self for method chaining
        """
        self._robots = robots
        return self

    def with_schema_type(self, schema_type: str) -> OpenGraphBuilder:
        """
        Set Schema.org type.

        Args:
            schema_type: Schema type (WebSite, Article, Product, etc.)

        Returns:
            Self for method chaining
        """
        self._schema_type = schema_type
        return self

    def build(self) -> OpenGraphMetadata:
        """
        Build immutable OpenGraphMetadata.

        Returns:
            Immutable OpenGraphMetadata instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if not self._title:
            raise ValueError("Title is required")
        if not self._description:
            raise ValueError("Description is required")
        if not self._image:
            raise ValueError("Image is required")
        if not self._url:
            raise ValueError("URL is required")

        return OpenGraphMetadata(
            title=self._title,
            description=self._description,
            image=self._image,
            url=self._url,
            type=self._type,
            site_name=self._site_name,
            locale=self._locale,
            keywords=self._keywords,
            author=self._author,
            twitter_site=self._twitter_site,
            twitter_creator=self._twitter_creator,
            robots=self._robots,
            schema_type=self._schema_type,
        )


# ═══════════════════════════════════════════════════════
# PRESET BUILDERS (Common Templates)
# ═══════════════════════════════════════════════════════


class WebsiteBuilder(OpenGraphBuilder):
    """Builder preset for general websites."""

    def __init__(self):
        super().__init__()
        self.with_type("website")
        self.with_schema_type("WebSite")


class ArticleBuilder(OpenGraphBuilder):
    """Builder preset for blog posts/articles."""

    def __init__(self):
        super().__init__()
        self.with_type("article")
        self.with_schema_type("Article")

    def with_published_time(self, published_time: str) -> ArticleBuilder:
        """Add article published time."""
        # Would add to metadata if needed
        return self

    def with_modified_time(self, modified_time: str) -> ArticleBuilder:
        """Add article modified time."""
        # Would add to metadata if needed
        return self


class ProductBuilder(OpenGraphBuilder):
    """Builder preset for e-commerce products."""

    def __init__(self):
        super().__init__()
        self.with_type("product")
        self.with_schema_type("Product")

    def with_price(self, price: float, currency: str = "USD") -> ProductBuilder:
        """Add product price."""
        # Would add to metadata if needed
        return self

    def with_availability(self, availability: str) -> ProductBuilder:
        """Add product availability."""
        # Would add to metadata if needed
        return self


# ═══════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════


def generate_opengraph(
    title: str, description: str, image: str, url: str, **kwargs
) -> OpenGraphMetadata:
    """
    Convenience function to generate OpenGraph metadata.

    Args:
        title: Page title
        description: Page description
        image: OG image URL
        url: Canonical URL
        **kwargs: Additional options (site_name, twitter, etc.)

    Returns:
        OpenGraphMetadata instance
    """
    builder = OpenGraphBuilder()
    builder.with_title(title)
    builder.with_description(description)
    builder.with_image(image)
    builder.with_url(url)

    # Apply optional kwargs
    if "site_name" in kwargs:
        builder.with_site_name(kwargs["site_name"])
    if "twitter" in kwargs:
        builder.with_twitter(kwargs["twitter"])
    if "author" in kwargs:
        builder.with_author(kwargs["author"])
    if "keywords" in kwargs:
        builder.with_keywords(*kwargs["keywords"])
    if "type" in kwargs:
        builder.with_type(kwargs["type"])

    return builder.build()
