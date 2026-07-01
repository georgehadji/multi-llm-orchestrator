"""
Image Optimizer — Pipeline + Strategy Pattern
==============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Image optimization using Pipeline Pattern for processing chains
and Strategy Pattern for different optimization algorithms.

Paradigm: OOP with Functional utilities
Patterns: Pipeline, Strategy, Factory Method, Immutable Data

Usage:
    from orchestrator.generators.image_optimizer import ImageOptimizer, WebPOptimizer

    optimizer = ImageOptimizer([WebPOptimizer(), AVIFOptimizer()])
    optimized = optimizer.optimize("input.jpg", output_format="webp")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import base64

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


class ImageFormat(str, Enum):
    """Image format enumeration."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    AVIF = "avif"
    GIF = "gif"
    SVG = "svg"


class OptimizationLevel(str, Enum):
    """Optimization compression level."""

    LOSSLESS = "lossless"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Immutable optimization configuration.

    Attributes:
        quality: Quality level (0-100)
        level: Optimization level
        max_width: Maximum width (resize if larger)
        max_height: Maximum height (resize if larger)
        strip_metadata: Remove EXIF metadata
        progressive: Progressive encoding
        optimize_colors: Optimize color palette
    """

    quality: int = 85
    level: OptimizationLevel = OptimizationLevel.MEDIUM
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    strip_metadata: bool = True
    progressive: bool = True
    optimize_colors: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.quality < 0 or self.quality > 100:
            object.__setattr__(self, "quality", 85)


@dataclass(frozen=True)
class OptimizedImage:
    """
    Immutable optimized image result.

    Attributes:
        data: Image data (bytes or base64)
        format: Output format
        width: Image width
        height: Image height
        original_size: Original size in bytes
        optimized_size: Optimized size in bytes
        compression_ratio: Compression ratio
        config: Optimization configuration
    """

    data: bytes
    format: ImageFormat
    width: int
    height: int
    original_size: int
    optimized_size: int
    config: OptimizationConfig

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.optimized_size / self.original_size)

    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage."""
        return self.compression_ratio * 100

    def to_base64(self) -> str:
        """Convert to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")

    def to_data_uri(self) -> str:
        """Convert to data URI."""
        mime_type = f"image/{self.format.value}"
        return f"data:{mime_type};base64,{self.to_base64()}"


# ═══════════════════════════════════════════════════════════════════
# STRATEGY PATTERN — OPTIMIZER INTERFACE
# ═══════════════════════════════════════════════════════════════════


class ImageOptimizerStrategy(ABC):
    """
    Strategy Pattern for image optimization algorithms.

    Subclasses implement different optimization strategies.
    """

    @abstractmethod
    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """
        Optimize image.

        Args:
            image_data: Input image data
            config: Optimization configuration

        Returns:
            OptimizedImage
        """
        pass

    @abstractmethod
    def get_format(self) -> ImageFormat:
        """
        Get output format.

        Returns:
            Image format
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if optimizer is available.

        Returns:
            True if available
        """
        pass


# ═══════════════════════════════════════════════════════════════════
# CONCRETE OPTIMIZERS
# ═══════════════════════════════════════════════════════════════════


class WebPOptimizer(ImageOptimizerStrategy):
    """
    WebP optimization strategy.

    Google's modern image format with superior compression.
    """

    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """Optimize to WebP format."""
        # Placeholder implementation
        # In production, use Pillow or similar:
        # from PIL import Image
        # img = Image.open(io.BytesIO(image_data))
        # img.save(output, 'WEBP', quality=config.quality)

        # Simulate optimization
        optimized_size = int(len(image_data) * (1 - config.compression_ratio))

        return OptimizedImage(
            data=image_data,  # In production, actual optimized data
            format=ImageFormat.WEBP,
            width=800,
            height=600,
            original_size=len(image_data),
            optimized_size=optimized_size,
            config=config,
        )

    def get_format(self) -> ImageFormat:
        """Get output format."""
        return ImageFormat.WEBP

    def is_available(self) -> bool:
        """Check if WebP is available."""
        return True  # Widely supported


class AVIFOptimizer(ImageOptimizerStrategy):
    """
    AVIF optimization strategy.

    Next-gen format with best compression.
    """

    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """Optimize to AVIF format."""
        # Placeholder - in production use avifenc or similar
        optimized_size = int(len(image_data) * 0.5)  # AVIF typically 50% smaller

        return OptimizedImage(
            data=image_data,
            format=ImageFormat.AVIF,
            width=800,
            height=600,
            original_size=len(image_data),
            optimized_size=optimized_size,
            config=config,
        )

    def get_format(self) -> ImageFormat:
        """Get output format."""
        return ImageFormat.AVIF

    def is_available(self) -> bool:
        """Check if AVIF is available."""
        return True  # Growing support


class JPEGOptimizer(ImageOptimizerStrategy):
    """
    JPEG optimization strategy.

    Classic format with good compression.
    """

    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """Optimize JPEG."""
        # Placeholder - in production use Pillow
        optimized_size = int(len(image_data) * (config.quality / 100))

        return OptimizedImage(
            data=image_data,
            format=ImageFormat.JPEG,
            width=800,
            height=600,
            original_size=len(image_data),
            optimized_size=optimized_size,
            config=config,
        )

    def get_format(self) -> ImageFormat:
        """Get output format."""
        return ImageFormat.JPEG

    def is_available(self) -> bool:
        """Check if JPEG is available."""
        return True  # Universal support


class PNGOptimizer(ImageOptimizerStrategy):
    """
    PNG optimization strategy.

    Lossless format for graphics.
    """

    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """Optimize PNG."""
        # Placeholder - in production use pngquant or optipng
        if config.level == OptimizationLevel.LOSSLESS:
            optimized_size = int(len(image_data) * 0.9)
        else:
            optimized_size = int(len(image_data) * (config.quality / 100))

        return OptimizedImage(
            data=image_data,
            format=ImageFormat.PNG,
            width=800,
            height=600,
            original_size=len(image_data),
            optimized_size=optimized_size,
            config=config,
        )

    def get_format(self) -> ImageFormat:
        """Get output format."""
        return ImageFormat.PNG

    def is_available(self) -> bool:
        """Check if PNG is available."""
        return True  # Universal support


class SVGOptimizer(ImageOptimizerStrategy):
    """
    SVG optimization strategy.

    Vector format optimization.
    """

    def optimize(self, image_data: bytes, config: OptimizationConfig) -> OptimizedImage:
        """Optimize SVG."""
        # Placeholder - in production use svgo
        # Remove whitespace, comments, unused definitions
        optimized_data = image_data

        return OptimizedImage(
            data=optimized_data,
            format=ImageFormat.SVG,
            width=800,
            height=600,
            original_size=len(image_data),
            optimized_size=len(optimized_data),
            config=config,
        )

    def get_format(self) -> ImageFormat:
        """Get output format."""
        return ImageFormat.SVG

    def is_available(self) -> bool:
        """Check if SVG optimizer is available."""
        return True


# ═══════════════════════════════════════════════════════════════════
# PIPELINE PATTERN — IMAGE OPTIMIZATION PIPELINE
# ═══════════════════════════════════════════════════════════════════


class ImageOptimizationPipeline:
    """
    Pipeline Pattern for image processing.

    Chains multiple optimization steps.

    Usage:
        pipeline = (ImageOptimizationPipeline()
            .add_step(ResizeStep())
            .add_step(CompressStep())
            .add_step(MetadataStripStep()))

        result = pipeline.execute(image_data, config)
    """

    def __init__(self):
        """Initialize pipeline."""
        self._steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> "ImageOptimizationPipeline":
        """
        Add pipeline step.

        Args:
            step: Pipeline step

        Returns:
            Self for fluent interface
        """
        self._steps.append(step)
        return self

    def execute(
        self,
        image_data: bytes,
        config: OptimizationConfig,
    ) -> OptimizedImage:
        """
        Execute pipeline.

        Args:
            image_data: Input image data
            config: Optimization configuration

        Returns:
            OptimizedImage
        """
        current_data = image_data

        for step in self._steps:
            current_data = step.process(current_data, config)

        # Final optimization with selected strategy
        optimizer = self._select_optimizer(config)
        return optimizer.optimize(current_data, config)

    def _select_optimizer(self, config: OptimizationConfig) -> ImageOptimizerStrategy:
        """Select optimizer based on config."""
        # Default to WebP for best balance
        return WebPOptimizer()


class PipelineStep(ABC):
    """Abstract pipeline step."""

    @abstractmethod
    def process(self, image_data: bytes, config: OptimizationConfig) -> bytes:
        """
        Process image data.

        Args:
            image_data: Input image data
            config: Optimization configuration

        Returns:
            Processed image data
        """
        pass


class ResizeStep(PipelineStep):
    """Resize pipeline step."""

    def process(self, image_data: bytes, config: OptimizationConfig) -> bytes:
        """Resize image if needed."""
        # Placeholder - in production use Pillow
        # img = Image.open(io.BytesIO(image_data))
        # if config.max_width and img.width > config.max_width:
        #     img = img.resize((config.max_width, ...))
        return image_data


class CompressStep(PipelineStep):
    """Compression pipeline step."""

    def process(self, image_data: bytes, config: OptimizationConfig) -> bytes:
        """Compress image."""
        # Placeholder - in production apply compression
        return image_data


class MetadataStripStep(PipelineStep):
    """Metadata removal step."""

    def process(self, image_data: bytes, config: OptimizationConfig) -> bytes:
        """Strip EXIF metadata."""
        if not config.strip_metadata:
            return image_data

        # Placeholder - in production remove EXIF
        return image_data


# ═══════════════════════════════════════════════════════════════════
# FACADE — IMAGE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════


class ImageOptimizer:
    """
    Facade for image optimization.

    Simplifies image optimization with sensible defaults.

    Usage:
        optimizer = ImageOptimizer()
        optimized = optimizer.optimize("input.jpg", "output.webp")
    """

    def __init__(self, strategies: List[ImageOptimizerStrategy] = None):
        """
        Initialize optimizer.

        Args:
            strategies: List of optimization strategies
        """
        self._strategies = strategies or [
            WebPOptimizer(),
            AVIFOptimizer(),
            JPEGOptimizer(),
            PNGOptimizer(),
            SVGOptimizer(),
        ]
        self._pipeline = ImageOptimizationPipeline()

    def optimize(
        self,
        image_data: bytes,
        output_format: str = "webp",
        config: OptimizationConfig = None,
    ) -> OptimizedImage:
        """
        Optimize image.

        Args:
            image_data: Input image data
            output_format: Output format
            config: Optimization configuration

        Returns:
            OptimizedImage
        """
        if config is None:
            config = OptimizationConfig()

        # Select strategy
        strategy = self._get_strategy(output_format)

        if not strategy:
            raise ValueError(f"Unsupported format: {output_format}")

        return strategy.optimize(image_data, config)

    def optimize_file(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "webp",
        config: OptimizationConfig = None,
    ) -> OptimizedImage:
        """
        Optimize image file.

        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Output format
            config: Optimization configuration

        Returns:
            OptimizedImage
        """
        # Read input file
        with open(input_path, "rb") as f:
            image_data = f.read()

        # Optimize
        optimized = self.optimize(image_data, output_format, config)

        # Write output file
        with open(output_path, "wb") as f:
            f.write(optimized.data)

        return optimized

    def _get_strategy(self, format: str) -> Optional[ImageOptimizerStrategy]:
        """Get optimizer strategy by format."""
        format_map = {
            "webp": WebPOptimizer,
            "avif": AVIFOptimizer,
            "jpeg": JPEGOptimizer,
            "jpg": JPEGOptimizer,
            "png": PNGOptimizer,
            "svg": SVGOptimizer,
        }

        strategy_class = format_map.get(format.lower())

        if strategy_class:
            return strategy_class()

        return None

    def get_supported_formats(self) -> List[str]:
        """
        Get supported output formats.

        Returns:
            List of format names
        """
        return ["webp", "avif", "jpeg", "png", "svg"]

    def batch_optimize(
        self,
        images: List[bytes],
        output_format: str = "webp",
        config: OptimizationConfig = None,
    ) -> List[OptimizedImage]:
        """
        Batch optimize multiple images.

        Args:
            images: List of image data
            output_format: Output format
            config: Optimization configuration

        Returns:
            List of OptimizedImage
        """
        return [self.optimize(image, output_format, config) for image in images]


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def optimize_image(
    image_data: bytes,
    output_format: str = "webp",
    quality: int = 85,
) -> OptimizedImage:
    """
    Optimize image with defaults.

    Args:
        image_data: Input image data
        output_format: Output format
        quality: Quality level (0-100)

    Returns:
        OptimizedImage
    """
    optimizer = ImageOptimizer()
    config = OptimizationConfig(quality=quality)
    return optimizer.optimize(image_data, output_format, config)


def optimize_image_file(
    input_path: str,
    output_path: str,
    output_format: str = "webp",
    quality: int = 85,
) -> OptimizedImage:
    """
    Optimize image file.

    Args:
        input_path: Input file path
        output_path: Output file path
        output_format: Output format
        quality: Quality level

    Returns:
        OptimizedImage
    """
    optimizer = ImageOptimizer()
    config = OptimizationConfig(quality=quality)
    return optimizer.optimize_file(input_path, output_path, output_format, config)


def convert_to_webp(
    image_data: bytes,
    quality: int = 85,
) -> OptimizedImage:
    """
    Convert image to WebP.

    Args:
        image_data: Input image data
        quality: Quality level

    Returns:
        OptimizedImage
    """
    return optimize_image(image_data, "webp", quality)


def convert_to_avif(
    image_data: bytes,
    quality: int = 85,
) -> OptimizedImage:
    """
    Convert image to AVIF.

    Args:
        image_data: Input image data
        quality: Quality level

    Returns:
        OptimizedImage
    """
    return optimize_image(image_data, "avif", quality)
