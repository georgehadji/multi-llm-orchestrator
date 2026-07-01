"""
Copy Generator — Template Method + Memento Pattern
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Copy generation using Template Method for structure and Memento for versioning.

Paradigm: OOP with Functional utilities
Patterns: Template Method, Memento, Factory Method, Builder

Usage:
    from orchestrator.generators.copy_generator import CopyGenerator, HeroCopyGenerator

    generator = HeroCopyGenerator()
    copy = generator.generate()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CopyMemento:
    """
    Memento for copy versioning.

    Attributes:
        content: Copy content
        version: Version number
        timestamp: Creation timestamp
        metadata: Additional metadata
    """

    content: str
    version: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CopyConfig:
    """
    Copy generation configuration.

    Note: This dataclass is frozen (immutable) to ensure proper Memento
    pattern behavior. When config is stored in memento metadata, it must
    remain unchanged to allow accurate version restoration.

    Attributes:
        tone: Copy tone (professional, casual, friendly, etc.)
        language: Language code
        target_audience: Target audience description
        word_count_min: Minimum word count
        word_count_max: Maximum word count
        include_cta: Include call-to-action
        seo_keywords: SEO keywords to include
    """

    tone: str = "professional"
    language: str = "en"
    target_audience: str = "general"
    word_count_min: int = 50
    word_count_max: int = 300
    include_cta: bool = True
    seo_keywords: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE METHOD PATTERN — COPY GENERATOR BASE
# ═══════════════════════════════════════════════════════════════════


class CopyGenerator(ABC):
    """
    Template Method Pattern for copy generation.

    Defines the skeleton of copy generation, allowing subclasses
    to customize specific steps.
    """

    def __init__(self):
        """Initialize copy generator."""
        self._mementos: List[CopyMemento] = []
        self._current_version = 0
        self._config = CopyConfig()

    def set_config(self, config: CopyConfig) -> "CopyGenerator":
        """Set configuration (fluent interface)."""
        self._config = config
        return self

    @abstractmethod
    def _define_structure(self) -> Dict[str, Any]:
        """
        Hook: Define copy structure.

        Subclasses implement to define their specific structure.

        Returns:
            Structure dictionary
        """
        pass

    @abstractmethod
    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """
        Hook: Generate content from structure.

        Subclasses implement content generation logic.

        Args:
            structure: Copy structure

        Returns:
            Generated content
        """
        pass

    @abstractmethod
    def _validate(self, content: str) -> bool:
        """
        Hook: Validate generated content.

        Subclasses implement validation logic.

        Args:
            content: Generated content

        Returns:
            True if valid
        """
        pass

    def generate(self) -> str:
        """
        Template Method: Final algorithm for copy generation.

        Steps:
        1. Define structure
        2. Generate content
        3. Validate
        4. Save memento
        5. Return content

        Returns:
            Generated copy
        """
        # Step 1: Define structure
        structure = self._define_structure()

        # Step 2: Generate content
        content = self._generate_content(structure)

        # Step 3: Validate
        if not self._validate(content):
            raise ValueError("Generated content failed validation")

        # Step 4: Save memento
        self._save_memento(content)

        # Step 5: Return
        return content

    def _save_memento(self, content: str) -> None:
        """
        Save state to memento.

        Note: Uses deep copy of config to ensure memento immutability.
        Without deep copy, changes to self._config would affect all
        historical mementos since they'd reference the same object.
        """
        self._current_version += 1
        import copy

        self._mementos.append(
            CopyMemento(
                content=content,
                version=self._current_version,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"config": copy.deepcopy(self._config)},
            )
        )

    def restore(self, version: int) -> Optional[str]:
        """
        Restore to previous version.

        Args:
            version: Version number to restore

        Returns:
            Content or None if version not found
        """
        for memento in self._mementos:
            if memento.version == version:
                return memento.content
        return None

    def get_versions(self) -> List[int]:
        """
        Get list of available versions.

        Returns:
            List of version numbers
        """
        return [m.version for m in self._mementos]

    def get_current_version(self) -> int:
        """Get current version number."""
        return self._current_version


# ═══════════════════════════════════════════════════════════════════
# CONCRETE COPY GENERATORS
# ═══════════════════════════════════════════════════════════════════


class HeroCopyGenerator(CopyGenerator):
    """
    Hero section copy generator.

    Generates compelling hero headlines and subheadlines.
    """

    def _define_structure(self) -> Dict[str, Any]:
        """Define hero copy structure."""
        return {
            "headline": {
                "max_length": 60,
                "tone": self._config.tone,
                "include_keywords": (
                    self._config.seo_keywords[:3] if self._config.seo_keywords else []
                ),
            },
            "subheadline": {
                "max_length": 120,
                "tone": self._config.tone,
                "support_headline": True,
            },
            "cta": {
                "max_length": 30,
                "action_oriented": self._config.include_cta,
            },
        }

    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """Generate hero copy."""
        # Placeholder implementation
        # In production, this would use LLM to generate copy

        headline_templates = {
            "professional": "Transform Your Business with {keyword}",
            "casual": "Hey There! Ready to {keyword}?",
            "friendly": "Welcome to the Future of {keyword}",
            "bold": "Revolutionize Your {keyword} Today",
        }

        tone = self._config.tone
        keyword = self._config.seo_keywords[0] if self._config.seo_keywords else "Business"

        headline = headline_templates.get(tone, headline_templates["professional"]).format(
            keyword=keyword
        )
        subheadline = f"Discover how our solution helps you achieve more with less effort."
        cta = "Get Started Free" if self._config.include_cta else ""

        return f"""# {headline}

{subheadline}

{cta}"""

    def _validate(self, content: str) -> bool:
        """Validate hero copy."""
        # Check word count
        words = content.split()
        word_count = len(words)

        if word_count < self._config.word_count_min:
            return False
        if word_count > self._config.word_count_max:
            return False

        # Check for headline (should have #)
        if "#" not in content:
            return False

        return True


class AboutCopyGenerator(CopyGenerator):
    """
    About page copy generator.

    Generates compelling about page content.
    """

    def _define_structure(self) -> Dict[str, Any]:
        """Define about copy structure."""
        return {
            "introduction": {
                "max_length": 100,
                "tone": self._config.tone,
            },
            "story": {
                "max_length": 300,
                "tone": self._config.tone,
                "include_mission": True,
            },
            "values": {
                "count": 3,
                "tone": self._config.tone,
            },
            "cta": {
                "max_length": 50,
                "action_oriented": self._config.include_cta,
            },
        }

    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """Generate about copy."""
        return f"""# About Us

## Our Story

We're passionate about delivering exceptional value to our customers. Our mission is to make {self._config.seo_keywords[0] if self._config.seo_keywords else "business"} easier and more efficient.

## Our Values

1. **Customer First** - Your success is our success
2. **Innovation** - Always pushing boundaries
3. **Integrity** - Honest and transparent in everything we do

{f"Ready to get started? Contact us today!" if self._config.include_cta else ""}"""

    def _validate(self, content: str) -> bool:
        """Validate about copy."""
        words = content.split()
        return len(words) >= self._config.word_count_min


class ProductCopyGenerator(CopyGenerator):
    """
    Product description copy generator.

    Generates compelling product descriptions.
    """

    def __init__(self, product_name: str = "Product"):
        """Initialize with product name."""
        super().__init__()
        self._product_name = product_name

    def _define_structure(self) -> Dict[str, Any]:
        """Define product copy structure."""
        return {
            "title": {
                "max_length": 80,
                "include_product_name": True,
            },
            "short_description": {
                "max_length": 100,
                "highlight_benefits": True,
            },
            "features": {
                "count": 3,
                "format": "bullet",
            },
            "cta": {
                "max_length": 30,
                "action_oriented": True,
            },
        }

    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """Generate product copy."""
        return f"""# {self._product_name}

{f"Experience the ultimate solution for {self._config.seo_keywords[0] if self._config.seo_keywords else 'your needs'}."}

## Features

- Feature 1: Benefit-focused description
- Feature 2: Another great benefit
- Feature 3: One more compelling benefit

{f"Buy Now - Only $99" if self._config.include_cta else ""}"""

    def _validate(self, content: str) -> bool:
        """Validate product copy."""
        return self._product_name in content and "-" in content  # Has bullet points


class FAQCopyGenerator(CopyGenerator):
    """
    FAQ page copy generator.

    Generates frequently asked questions and answers.
    """

    def _define_structure(self) -> Dict[str, Any]:
        """Define FAQ copy structure."""
        return {
            "questions": {
                "count": 5,
                "tone": self._config.tone,
                "categories": ["general", "pricing", "support", "technical"],
            },
        }

    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """Generate FAQ copy."""
        faqs = [
            (
                "What is your product?",
                f"Our product helps you {self._config.seo_keywords[0] if self._config.seo_keywords else 'achieve your goals'} efficiently.",
            ),
            ("How much does it cost?", "We offer flexible pricing plans starting at $9/month."),
            ("Do you offer support?", "Yes! We provide 24/7 customer support via email and chat."),
            ("Can I cancel anytime?", "Absolutely! No contracts, cancel anytime."),
            ("Is there a free trial?", "Yes, we offer a 14-day free trial with full access."),
        ]

        content = "# Frequently Asked Questions\n\n"
        for question, answer in faqs:
            content += f"### {question}\n\n{answer}\n\n"

        return content

    def _validate(self, content: str) -> bool:
        """Validate FAQ copy."""
        return "###" in content and "?" in content  # Has questions


class TestimonialCopyGenerator(CopyGenerator):
    """
    Testimonial copy generator.

    Generates realistic testimonial templates.
    """

    def _define_structure(self) -> Dict[str, Any]:
        """Define testimonial structure."""
        return {
            "testimonials": {
                "count": 3,
                "tone": "authentic",
                "include_rating": True,
            },
        }

    def _generate_content(self, structure: Dict[str, Any]) -> str:
        """Generate testimonial copy."""
        testimonials = [
            (
                "John D.",
                "CEO, Tech Corp",
                "⭐⭐⭐⭐⭐",
                "This product transformed our business. Highly recommended!",
            ),
            (
                "Sarah M.",
                "Marketing Director",
                "⭐⭐⭐⭐⭐",
                "Amazing support and incredible results.",
            ),
            ("Mike R.", "Startup Founder", "⭐⭐⭐⭐⭐", "Best investment we made this year."),
        ]

        content = "# What Our Customers Say\n\n"
        for name, role, rating, text in testimonials:
            content += f"""
> "{text}"
> 
> **— {name}, {role}**
> {rating}
"""

        return content

    def _validate(self, content: str) -> bool:
        """Validate testimonial copy."""
        return ">" in content and "⭐" in content  # Has quotes and ratings


# ═══════════════════════════════════════════════════════════════════
# COPY GENERATOR FACTORY
# ═══════════════════════════════════════════════════════════════════


class CopyGeneratorFactory:
    """
    Factory Method for copy generators.

    Usage:
        factory = CopyGeneratorFactory()
        hero_generator = factory.create("hero")
        about_generator = factory.create("about")
    """

    _generators = {
        "hero": HeroCopyGenerator,
        "about": AboutCopyGenerator,
        "product": ProductCopyGenerator,
        "faq": FAQCopyGenerator,
        "testimonial": TestimonialCopyGenerator,
    }

    def create(self, copy_type: str, **kwargs) -> CopyGenerator:
        """
        Create copy generator.

        Args:
            copy_type: Type of copy ("hero", "about", "product", etc.)
            **kwargs: Additional arguments for generator

        Returns:
            CopyGenerator instance
        """
        if copy_type not in self._generators:
            raise ValueError(f"Unknown copy type: {copy_type}")

        generator_class = self._generators[copy_type]

        if copy_type == "product":
            return generator_class(kwargs.get("product_name", "Product"))
        else:
            return generator_class()


# ═══════════════════════════════════════════════════════════════════
# SEO COPY OPTIMIZER
# ═══════════════════════════════════════════════════════════════════


class SEOCopyOptimizer:
    """
    SEO optimization for copy.

    Decorator pattern for enhancing copy with SEO.
    """

    def __init__(self, keywords: List[str]):
        """
        Initialize SEO optimizer.

        Args:
            keywords: SEO keywords to optimize for
        """
        self._keywords = keywords

    def optimize(self, content: str) -> str:
        """
        Optimize content for SEO.

        Args:
            content: Original content

        Returns:
            Optimized content
        """
        # Placeholder implementation
        # In production, this would use NLP to optimize keyword density

        optimized = content

        # Ensure primary keyword appears in first 100 words
        if self._keywords and self._keywords[0] not in content[:100]:
            optimized = f"{self._keywords[0]}: {content}"

        return optimized

    def calculate_readability_score(self, content: str) -> float:
        """
        Calculate readability score.

        Args:
            content: Content to analyze

        Returns:
            Readability score (0-100)
        """
        # Simple Flesch-Kincaid approximation
        sentences = content.count(".") + content.count("!") + content.count("?")
        words = len(content.split())
        syllables = sum(self._count_syllables(word) for word in content.split())

        if sentences == 0 or words == 0:
            return 0.0

        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0.0, min(100.0, score))

    def _count_syllables(self, word: str) -> int:
        """Count syllables in word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        return max(1, count)  # Every word has at least 1 syllable


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def generate_hero_copy(
    tone: str = "professional",
    keywords: List[str] = None,
    include_cta: bool = True,
) -> str:
    """
    Generate hero copy.

    Args:
        tone: Copy tone
        keywords: SEO keywords
        include_cta: Include call-to-action

    Returns:
        Generated hero copy
    """
    config = CopyConfig(
        tone=tone,
        seo_keywords=keywords or [],
        include_cta=include_cta,
    )

    generator = HeroCopyGenerator()
    generator.set_config(config)
    return generator.generate()


def generate_about_copy(
    tone: str = "professional",
    keywords: List[str] = None,
) -> str:
    """
    Generate about page copy.

    Args:
        tone: Copy tone
        keywords: SEO keywords

    Returns:
        Generated about copy
    """
    config = CopyConfig(
        tone=tone,
        seo_keywords=keywords or [],
    )

    generator = AboutCopyGenerator()
    generator.set_config(config)
    return generator.generate()


def generate_product_copy(
    product_name: str,
    tone: str = "professional",
    keywords: List[str] = None,
) -> str:
    """
    Generate product description.

    Args:
        product_name: Product name
        tone: Copy tone
        keywords: SEO keywords

    Returns:
        Generated product copy
    """
    config = CopyConfig(
        tone=tone,
        seo_keywords=keywords or [],
    )

    generator = ProductCopyGenerator(product_name)
    generator.set_config(config)
    return generator.generate()


def generate_faq_copy() -> str:
    """
    Generate FAQ copy.

    Returns:
        Generated FAQ copy
    """
    generator = FAQCopyGenerator()
    return generator.generate()


def generate_testimonial_copy() -> str:
    """
    Generate testimonial copy.

    Returns:
        Generated testimonial copy
    """
    generator = TestimonialCopyGenerator()
    return generator.generate()
