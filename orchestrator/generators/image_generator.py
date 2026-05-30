"""
Image Generator — Abstract Factory + Chain of Responsibility
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

AI image generation using Abstract Factory for different providers
and Chain of Responsibility for processing pipelines.

Paradigm: OOP with Functional utilities
Patterns: Abstract Factory, Chain of Responsibility, Strategy, Immutable Data

Usage:
    from orchestrator.image_generator import ImageGenerator, FluxImageFactory

    factory = FluxImageFactory()
    generator = factory.create_generator()
    image_url = generator.generate("A beautiful sunset", style="realistic")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


class ImageStyle(str, Enum):
    """Image style enumeration."""

    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    MINIMAL = "minimal"
    VINTAGE = "vintage"
    CYBERPUNK = "cyberpunk"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    SKETCH = "sketch"


class ImageFormat(str, Enum):
    """Image format enumeration."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    AVIF = "avif"


@dataclass(frozen=True)
class ImageGenerationRequest:
    """
    Immutable image generation request.

    Attributes:
        prompt: Text prompt for image generation
        style: Image style
        width: Image width in pixels
        height: Image height in pixels
        negative_prompt: What to exclude from image
        seed: Random seed for reproducibility
        steps: Number of diffusion steps
        guidance_scale: CFG scale
    """

    prompt: str
    style: ImageStyle = ImageStyle.REALISTIC
    width: int = 1024
    height: int = 1024
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: int = 30
    guidance_scale: float = 7.5

    def __post_init__(self):
        """Validate request."""
        if self.width < 256 or self.width > 2048:
            object.__setattr__(self, "width", 1024)
        if self.height < 256 or self.height > 2048:
            object.__setattr__(self, "height", 1024)
        if self.steps < 10 or self.steps > 100:
            object.__setattr__(self, "steps", 30)


@dataclass(frozen=True)
class GeneratedImage:
    """
    Immutable generated image result.

    Attributes:
        url: Image URL
        base64: Base64 encoded image (optional)
        width: Image width
        height: Image height
        format: Image format
        prompt: Prompt used
        style: Style applied
        generation_time: Time taken in seconds
        model: Model used for generation
    """

    url: str
    width: int
    height: int
    format: ImageFormat
    prompt: str
    style: ImageStyle
    generation_time: float
    model: str
    base64: Optional[str] = None
    seed: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════
# CHAIN OF RESPONSIBILITY — PROCESSING HANDLERS
# ═══════════════════════════════════════════════════════════════════


class ImageProcessingHandler(ABC):
    """
    Chain of Responsibility for image processing.

    Each handler processes the request, then passes to next handler.
    """

    def __init__(self):
        self._next_handler: Optional[ImageProcessingHandler] = None

    def set_next(self, handler: "ImageProcessingHandler") -> "ImageProcessingHandler":
        """Set next handler in chain (fluent interface)."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request: ImageGenerationRequest) -> GeneratedImage:
        """
        Handle request, pass to next if can't handle.

        Args:
            request: Image generation request

        Returns:
            Generated image
        """
        pass

    def _pass_to_next(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Pass request to next handler."""
        if self._next_handler:
            return self._next_handler.handle(request)
        raise NotImplementedError("No handler in chain could process request")


class PromptEnhancementHandler(ImageProcessingHandler):
    """
    First handler: Enhance prompt for better results.

    Adds style-specific keywords to prompt.
    """

    STYLE_KEYWORDS = {
        ImageStyle.REALISTIC: "photorealistic, highly detailed, 8k, professional photography",
        ImageStyle.ARTISTIC: "artistic, creative, stylized, masterpiece",
        ImageStyle.MINIMAL: "minimalist, clean, simple, elegant",
        ImageStyle.VINTAGE: "vintage, retro, nostalgic, film grain",
        ImageStyle.CYBERPUNK: "cyberpunk, futuristic, neon, sci-fi",
        ImageStyle.WATERCOLOR: "watercolor, painting, artistic, soft colors",
        ImageStyle.OIL_PAINTING: "oil painting, textured, classical art",
        ImageStyle.SKETCH: "sketch, pencil drawing, monochrome, artistic",
    }

    def handle(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Enhance prompt then pass to next handler."""
        # Enhance prompt with style keywords
        style_keywords = self.STYLE_KEYWORDS.get(request.style, "")
        enhanced_prompt = f"{request.prompt}, {style_keywords}"

        # Create enhanced request
        enhanced_request = ImageGenerationRequest(
            prompt=enhanced_prompt,
            style=request.style,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
        )

        return self._pass_to_next(enhanced_request)


class ImageGenerationHandler(ImageProcessingHandler):
    """
    Handler: Generate image using AI model.

    This is where actual image generation happens.
    """

    def __init__(self, generator: "ImageGenerator"):
        """
        Initialize with generator.

        Args:
            generator: Image generator instance
        """
        super().__init__()
        self._generator = generator

    def handle(self, request: ImageGenerationRequest) -> GeneratedImage:
        """Generate image then pass to next handler."""
        import time

        start_time = time.time()

        # Generate image
        image_data = self._generator._generate_image(request)

        generation_time = time.time() - start_time

        # Create result
        result = GeneratedImage(
            url=image_data.get("url", ""),
            base64=image_data.get("base64"),
            width=request.width,
            height=request.height,
            format=ImageFormat.PNG,
            prompt=request.prompt,
            style=request.style,
            generation_time=generation_time,
            model=self._generator.get_model_name(),
            seed=request.seed,
        )

        return self._pass_to_next(result)


class ImageOptimizationHandler(ImageProcessingHandler):
    """
    Handler: Optimize generated image.

    Converts to optimal format, compresses, etc.

    Note: This handler receives GeneratedImage from ImageGenerationHandler,
    not ImageGenerationRequest. The Chain of Responsibility pattern requires
    each handler to accept the output type of the previous handler.
    """

    def handle(self, image: GeneratedImage) -> GeneratedImage:
        """
        Pass through for now (optimization implemented separately).

        Args:
            image: Generated image from previous handler

        Returns:
            GeneratedImage (passed to next handler or returned)
        """
        # In production, apply optimization here:
        # - Format conversion (WebP/AVIF)
        # - Compression
        # - Resize if needed
        return self._pass_to_next(image)


# ═══════════════════════════════════════════════════════════════════
# ABSTRACT FACTORY — IMAGE PROVIDER FACTORIES
# ═══════════════════════════════════════════════════════════════════


class ImageProviderFactory(ABC):
    """
    Abstract Factory for image providers.

    Subclasses implement specific AI image providers.
    """

    @abstractmethod
    def create_generator(self) -> "ImageGenerator":
        """
        Create image generator.

        Returns:
            ImageGenerator instance
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider capabilities.

        Returns:
            Capabilities dictionary
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available.

        Returns:
            True if available
        """
        pass


class FluxImageFactory(ImageProviderFactory):
    """
    Factory for FLUX.2 image generation.

    FLUX.2: High-quality open-source image model.
    Cost: ~$0.014 per megapixel
    """

    def create_generator(self) -> "ImageGenerator":
        """Create FLUX.2 generator."""
        return ImageGenerator(
            model_id="black-forest-labs/flux-2",
            api_endpoint="https://api.flux.ai/v1/generate",
            cost_per_mp=0.014,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get FLUX.2 capabilities."""
        return {
            "max_resolution": "2048x2048",
            "styles": ["realistic", "artistic", "minimal", "cyberpunk"],
            "cost_per_image": 0.014,
            "avg_generation_time": "5-10s",
            "supports_negative_prompt": True,
        }

    def is_available(self) -> bool:
        """Check if FLUX.2 is available."""
        return True  # Via OpenRouter


class RiverflowImageFactory(ImageProviderFactory):
    """
    Factory for Riverflow V2 image generation.

    Riverflow V2: High-quality multi-style model.
    Cost: $0.02-0.33 per image
    """

    def create_generator(self) -> "ImageGenerator":
        """Create Riverflow generator."""
        return ImageGenerator(
            model_id="riverflow/riverflow-v2",
            api_endpoint="https://api.riverflow.ai/v1/generate",
            cost_per_mp=0.05,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Riverflow capabilities."""
        return {
            "max_resolution": "1536x1536",
            "styles": ["realistic", "artistic", "watercolor", "oil_painting", "sketch"],
            "cost_per_image": 0.05,
            "avg_generation_time": "3-8s",
            "supports_negative_prompt": True,
        }

    def is_available(self) -> bool:
        """Check if Riverflow is available."""
        return True  # Via OpenRouter


class OpenRouterImageFactory(ImageProviderFactory):
    """
    Factory for OpenRouter image generation.

    Access to multiple image models via OpenRouter.
    """

    def create_generator(self, model_id: str = None) -> "ImageGenerator":
        """
        Create OpenRouter generator.

        Args:
            model_id: Specific model ID (optional)

        Returns:
            ImageGenerator instance
        """
        return ImageGenerator(
            model_id=model_id or "recraft/recraftv3",
            api_endpoint="https://openrouter.ai/api/v1/images/generate",
            cost_per_mp=0.03,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get OpenRouter capabilities."""
        return {
            "models": [
                "recraft/recraftv3",
                "stability-ai/sdxl",
                "midjourney/midjourney-v6",
            ],
            "max_resolution": "2048x2048",
            "styles": ["all"],
            "cost_range": "$0.01-0.50 per image",
        }

    def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        import os

        return bool(os.getenv("OPENROUTER_API_KEY"))


# ═══════════════════════════════════════════════════════════════════
# IMAGE GENERATOR — MAIN CLASS
# ═══════════════════════════════════════════════════════════════════


class ImageGenerator:
    """
    Main image generator class.

    Uses processing chain for generation pipeline.

    Usage:
        generator = ImageGenerator(model_id="flux-2")
        chain = (PromptEnhancementHandler()
            .set_next(ImageGenerationHandler(generator))
            .set_next(ImageOptimizationHandler()))

        result = chain.handle(request)
    """

    def __init__(
        self,
        model_id: str,
        api_endpoint: str,
        cost_per_mp: float,
    ):
        """
        Initialize image generator.

        Args:
            model_id: Model identifier
            api_endpoint: API endpoint URL
            cost_per_mp: Cost per megapixel
        """
        self._model_id = model_id
        self._api_endpoint = api_endpoint
        self._cost_per_mp = cost_per_mp

    def get_model_name(self) -> str:
        """Get model name."""
        return self._model_id

    def get_cost(self, width: int, height: int) -> float:
        """
        Calculate generation cost.

        Args:
            width: Image width
            height: Image height

        Returns:
            Cost in USD
        """
        megapixels = (width * height) / 1_000_000
        return megapixels * self._cost_per_mp

    def _generate_image(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        """
        Generate image (internal implementation).

        Args:
            request: Image generation request

        Returns:
            Image data dictionary
        """
        # Placeholder implementation
        # In production, this would call the actual API

        return {
            "url": f"https://example.com/generated/{request.seed or 12345}.png",
            "base64": None,
            "width": request.width,
            "height": request.height,
        }

    async def generate_async(
        self,
        prompt: str,
        style: ImageStyle = ImageStyle.REALISTIC,
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> GeneratedImage:
        """
        Generate image asynchronously.

        Args:
            prompt: Text prompt
            style: Image style
            width: Image width
            height: Image height
            **kwargs: Additional parameters

        Returns:
            Generated image
        """
        request = ImageGenerationRequest(
            prompt=prompt,
            style=style,
            width=width,
            height=height,
            **kwargs,
        )

        # Build processing chain
        chain = (
            PromptEnhancementHandler()
            .set_next(ImageGenerationHandler(self))
            .set_next(ImageOptimizationHandler())
        )

        return chain.handle(request)

    def generate(
        self,
        prompt: str,
        style: ImageStyle = ImageStyle.REALISTIC,
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> GeneratedImage:
        """
        Generate image synchronously.

        Args:
            prompt: Text prompt
            style: Image style
            width: Image width
            height: Image height
            **kwargs: Additional parameters

        Returns:
            Generated image
        """
        import asyncio

        return asyncio.run(self.generate_async(prompt, style, width, height, **kwargs))


# ═══════════════════════════════════════════════════════════════════
# IMAGE GENERATOR FACADE
# ═══════════════════════════════════════════════════════════════════


class ImageGeneratorFacade:
    """
    Facade for image generation.

    Simplifies image generation with sensible defaults.

    Usage:
        facade = ImageGeneratorFacade()
        image = facade.generate("A sunset", style="realistic")
    """

    def __init__(self, provider: str = "flux"):
        """
        Initialize facade.

        Args:
            provider: Provider name ("flux", "riverflow", "openrouter")
        """
        self._provider = self._create_provider(provider)

    def _create_provider(self, provider: str) -> ImageProviderFactory:
        """Create provider factory."""
        factories = {
            "flux": FluxImageFactory(),
            "riverflow": RiverflowImageFactory(),
            "openrouter": OpenRouterImageFactory(),
        }
        return factories.get(provider, FluxImageFactory())

    def generate(
        self,
        prompt: str,
        style: str = "realistic",
        width: int = 1024,
        height: int = 1024,
    ) -> GeneratedImage:
        """
        Generate image.

        Args:
            prompt: Text prompt
            style: Image style
            width: Image width
            height: Image height

        Returns:
            Generated image
        """
        generator = self._provider.create_generator()

        return generator.generate(
            prompt=prompt,
            style=ImageStyle(style),
            width=width,
            height=height,
        )

    def generate_batch(
        self,
        prompts: List[str],
        style: str = "realistic",
        width: int = 1024,
        height: int = 1024,
    ) -> List[GeneratedImage]:
        """
        Generate multiple images.

        Args:
            prompts: List of prompts
            style: Image style
            width: Image width
            height: Image height

        Returns:
            List of generated images
        """
        return [self.generate(prompt, style, width, height) for prompt in prompts]


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def generate_image(
    prompt: str,
    style: str = "realistic",
    provider: str = "flux",
    width: int = 1024,
    height: int = 1024,
) -> GeneratedImage:
    """
    Generate image with defaults.

    Args:
        prompt: Text prompt
        style: Image style
        provider: Provider name
        width: Image width
        height: Image height

    Returns:
        Generated image
    """
    facade = ImageGeneratorFacade(provider)
    return facade.generate(prompt, style, width, height)


def generate_images(
    prompts: List[str],
    style: str = "realistic",
    provider: str = "flux",
    width: int = 1024,
    height: int = 1024,
) -> List[GeneratedImage]:
    """
    Generate multiple images.

    Args:
        prompts: List of prompts
        style: Image style
        provider: Provider name
        width: Image width
        height: Image height

    Returns:
        List of generated images
    """
    facade = ImageGeneratorFacade(provider)
    return facade.generate_batch(prompts, style, width, height)