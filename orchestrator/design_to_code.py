"""
Design-to-Code Pipeline (Multi-Modal Input)
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Accept Figma/screenshot → Extract UI spec → Generate code

Current State: Text input only
Future State: Multi-modal (image → spec → code)

Benefits:
- Opens designer market (non-developers who want implementation)
- Figma → Orchestrator → deployable app pipeline
- Differentiator: no other orchestration tool does this

Usage:
    from orchestrator.design_to_code import DesignToCodePipeline

    pipeline = DesignToCodePipeline(client)
    spec = await pipeline.process_image("screenshot.png")
    result = await pipeline.generate_code(spec)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .log_config import get_logger
from .models import Model

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


@dataclass
class UIComponent:
    """Extracted UI component from design."""

    type: str  # button, form, list, card, nav, etc.
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    children: list[UIComponent] = field(default_factory=list)
    position: dict[str, Any] = field(default_factory=dict)
    styles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "properties": self.properties,
            "children": [c.to_dict() for c in self.children],
            "position": self.position,
            "styles": self.styles,
        }


@dataclass
class DesignSpec:
    """Complete design specification from image analysis."""

    components: list[UIComponent]
    layout_structure: str  # grid, flexbox, etc.
    color_palette: dict[str, str]
    typography: dict[str, Any]
    interactive_elements: list[dict[str, Any]]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "layout_structure": self.layout_structure,
            "color_palette": self.color_palette,
            "typography": self.typography,
            "interactive_elements": self.interactive_elements,
            "notes": self.notes,
        }

    def to_project_spec(self) -> str:
        """Convert to project specification for orchestrator."""
        spec_text = "# UI Design Specification\n\n"

        spec_text += "## Layout Structure\n"
        spec_text += f"Use {self.layout_structure} for main layout.\n\n"

        spec_text += "## Color Palette\n"
        for name, color in self.color_palette.items():
            spec_text += f"- {name}: `{color}`\n"
        spec_text += "\n"

        spec_text += "## Typography\n"
        for key, value in self.typography.items():
            spec_text += f"- {key}: {value}\n"
        spec_text += "\n"

        spec_text += "## Components\n"
        for component in self.components:
            spec_text += f"### {component.name} ({component.type})\n"
            if component.properties:
                spec_text += f"Properties: {json.dumps(component.properties, indent=2)}\n"
            if component.styles:
                spec_text += f"Styles: {json.dumps(component.styles, indent=2)}\n"
            spec_text += "\n"

        if self.interactive_elements:
            spec_text += "## Interactive Elements\n"
            for elem in self.interactive_elements:
                spec_text += (
                    f"- {elem.get('name', 'Element')}: {elem.get('behavior', 'Click action')}\n"
                )

        if self.notes:
            spec_text += f"\n## Notes\n{self.notes}\n"

        return spec_text


@dataclass
class GeneratedCode:
    """Generated code from design spec."""

    files: dict[str, str]
    framework: str
    dependencies: list[str]
    notes: str = ""


class DesignToCodePipeline:
    """
    Convert visual designs to code specifications.

    Pipeline:
    1. Send screenshot to vision model
    2. Extract UI components, layout, colors, typography
    3. Generate structured design spec
    4. Generate code from spec
    """

    # Supported vision models
    VISION_MODELS = {
        "claude-sonnet-4.6": {"provider": "anthropic", "strength": "strong"},
        "gpt-4o": {"provider": "openai", "strength": "strong"},
        "gemini-2.0-flash": {"provider": "google", "strength": "good"},
    }

    # Supported frameworks
    FRAMEWORKS = ["react", "vue", "fastapi", "flask", "nextjs"]

    def __init__(self, client, default_model: Model = Model.CLAUDE_SONNET_4_6):
        """
        Initialize design-to-code pipeline.

        Args:
            client: LLM client with vision support
            default_model: Default vision model to use
        """
        self.client = client
        self.default_model = default_model

        logger.info(f"Design-to-Code pipeline initialized with {default_model.value}")

    async def process_image(
        self,
        image_path: Path,
        framework: str = "react",
    ) -> DesignSpec:
        """
        Process image and extract UI specification.

        Args:
            image_path: Path to screenshot/image
            framework: Target framework (react, vue, fastapi, etc.)

        Returns:
            DesignSpec with extracted components and styles
        """
        logger.info(f"Processing image: {image_path} for {framework}")

        # Encode image
        image_data = self._encode_image(image_path)

        # Send to vision model
        analysis = await self._analyze_with_vision(image_data, framework)

        # Parse analysis into structured spec
        spec = self._parse_analysis(analysis, framework)

        logger.info(
            f"Extracted {len(spec.components)} components, "
            f"{len(spec.color_palette)} colors, "
            f"{len(spec.typography)} typography styles"
        )

        return spec

    async def generate_code(
        self,
        spec: DesignSpec,
        framework: str = "react",
    ) -> GeneratedCode:
        """
        Generate code from design specification.

        Args:
            spec: Design specification
            framework: Target framework

        Returns:
            GeneratedCode with files and dependencies
        """
        logger.info(f"Generating {framework} code from design spec")

        # Build code generation prompt
        prompt = self._build_code_prompt(spec, framework)

        # Generate code
        response = await self.client.call(
            model=self.default_model,
            prompt=prompt,
            system=f"You are an expert {framework} developer. "
            f"Generate clean, production-ready code from design specifications. "
            f"Output complete, runnable code files.",
            max_tokens=8000,
            temperature=0.2,
            timeout=300,
        )

        # Parse generated code
        files = self._parse_generated_code(response.text, framework)

        # Get dependencies
        dependencies = self._extract_dependencies(spec, framework)

        return GeneratedCode(
            files=files,
            framework=framework,
            dependencies=dependencies,
            notes=f"Generated from design spec with {len(spec.components)} components",
        )

    async def process_and_generate(
        self,
        image_path: Path,
        framework: str = "react",
    ) -> tuple[DesignSpec, GeneratedCode]:
        """
        Complete pipeline: image → spec → code.

        Args:
            image_path: Path to screenshot/image
            framework: Target framework

        Returns:
            Tuple of (DesignSpec, GeneratedCode)
        """
        spec = await self.process_image(image_path, framework)
        code = await self.generate_code(spec, framework)
        return spec, code

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        return base64.b64encode(image_bytes).decode("utf-8")

    async def _analyze_with_vision(
        self,
        image_data: str,
        framework: str,
    ) -> str:
        """
        Analyze image with vision model.

        Args:
            image_data: Base64 encoded image
            framework: Target framework

        Returns:
            Analysis text from vision model
        """
        # Determine media type
        media_type = "image/png"  # Default, could detect from extension

        # Build vision prompt
        prompt = self._build_vision_prompt(framework)

        # Check model provider and build appropriate request
        model_name = self.default_model.value

        if "claude" in model_name:
            # Anthropic format
            response = await self.client.call(
                model=self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=4000,
                temperature=0.2,
                timeout=300,
            )
        elif "gpt" in model_name:
            # OpenAI format
            response = await self.client.call(
                model=self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=4000,
                temperature=0.2,
                timeout=300,
            )
        else:
            # Fallback to text-only with description
            logger.warning(f"Model {model_name} may not support vision, using fallback")
            response = await self.client.call(
                model=self.default_model,
                prompt=prompt,
                max_tokens=4000,
                temperature=0.2,
                timeout=300,
            )

        return response.text

    def _build_vision_prompt(self, framework: str) -> str:
        """Build prompt for vision model analysis."""
        return (
            f"Analyze this UI screenshot and extract a complete specification for implementing it in {framework}.\n\n"
            f"Extract the following:\n\n"
            f"1. **UI Components**: Identify all UI components (buttons, forms, lists, cards, navigation, inputs, etc.)\n"
            f"   - For each component, note: type, name, properties, styles, position\n\n"
            f"2. **Layout Structure**: Describe the layout (grid, flexbox, positioning, responsive behavior)\n\n"
            f"3. **Color Palette**: Extract all colors with hex codes\n"
            f"   - Primary, secondary, accent colors\n"
            f"   - Background colors\n"
            f"   - Text colors\n\n"
            f"4. **Typography**: Extract font information\n"
            f"   - Font families\n"
            f"   - Font sizes (headings, body, captions)\n"
            f"   - Font weights\n"
            f"   - Line heights\n\n"
            f"5. **Interactive Elements**: Identify interactive behaviors\n"
            f"   - Buttons and their actions\n"
            f"   - Form submissions\n"
            f"   - Hover states\n"
            f"   - Animations\n\n"
            f"Output as structured JSON with these keys:\n"
            f"{{\n"
            f'  "components": [...],\n'
            f'  "layout": "...",\n'
            f'  "colors": {{"primary": "#...", ...}},\n'
            f'  "typography": {{...}},\n'
            f'  "interactive": [...],\n'
            f'  "notes": "..."\n'
            f"}}\n\n"
            f"Be thorough and precise. This will be used to generate production code."
        )

    def _build_code_prompt(self, spec: DesignSpec, framework: str) -> str:
        """Build prompt for code generation."""
        return (
            f"Generate complete {framework} code that implements this UI design.\n\n"
            f"## Design Specification\n\n"
            f"{spec.to_project_spec()}\n\n"
            f"## Requirements\n\n"
            f"1. Create all necessary component files\n"
            f"2. Implement the exact layout structure\n"
            f"3. Use the exact color palette provided\n"
            f"4. Implement all interactive behaviors\n"
            f"5. Make it responsive (mobile-friendly)\n"
            f"6. Include proper accessibility attributes\n"
            f"7. Add comprehensive comments\n\n"
            f"## Output Format\n\n"
            f"For each file, output:\n"
            f"```\n"
            f"### filename.ext\n"
            f"[file content here]\n"
            f"```\n\n"
            f"Generate all files needed for a complete, runnable application."
        )

    def _parse_analysis(self, analysis: str, framework: str) -> DesignSpec:
        """
        Parse vision model analysis into structured spec.

        Args:
            analysis: Analysis text from vision model
            framework: Target framework

        Returns:
            DesignSpec
        """
        # Try to extract JSON from response
        json_start = analysis.find("{")
        json_end = analysis.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(analysis[json_start:json_end])

                # Parse components
                components = []
                for comp_data in data.get("components", []):
                    components.append(
                        UIComponent(
                            type=comp_data.get("type", "unknown"),
                            name=comp_data.get("name", "unnamed"),
                            properties=comp_data.get("properties", {}),
                            children=[],
                            position=comp_data.get("position", {}),
                            styles=comp_data.get("styles", {}),
                        )
                    )

                return DesignSpec(
                    components=components,
                    layout_structure=data.get("layout", "flexbox"),
                    color_palette=data.get("colors", {}),
                    typography=data.get("typography", {}),
                    interactive_elements=data.get("interactive", []),
                    notes=data.get("notes", ""),
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")

        # Fallback: create minimal spec
        return DesignSpec(
            components=[],
            layout_structure="flexbox",
            color_palette={},
            typography={},
            interactive_elements=[],
            notes="Failed to parse vision model output, using fallback",
        )

    def _parse_generated_code(self, code_text: str, framework: str) -> dict[str, str]:
        """
        Parse generated code text into files.

        Args:
            code_text: Generated code with file markers
            framework: Target framework

        Returns:
            Dict of filename → content
        """
        files = {}

        # Look for file markers: ### filename.ext
        import re

        pattern = r"###\s+(\S+)\n(.*?)(?=###\s+\S+|$)"
        matches = re.findall(pattern, code_text, re.DOTALL)

        for filename, content in matches:
            files[filename] = content.strip()

        # If no markers found, assume single file
        if not files:
            if framework == "react":
                files["App.jsx"] = code_text.strip()
            elif framework == "vue":
                files["App.vue"] = code_text.strip()
            elif framework == "fastapi":
                files["main.py"] = code_text.strip()
            else:
                files["main.py"] = code_text.strip()

        return files

    def _extract_dependencies(self, spec: DesignSpec, framework: str) -> list[str]:
        """
        Extract dependencies from spec and framework.

        Args:
            spec: Design specification
            framework: Target framework

        Returns:
            List of dependencies
        """
        deps = []

        # Framework-specific base dependencies
        if framework == "react":
            deps = ["react", "react-dom"]
        elif framework == "vue":
            deps = ["vue@latest"]
        elif framework == "nextjs":
            deps = ["next", "react", "react-dom"]
        elif framework == "fastapi":
            deps = ["fastapi", "uvicorn"]
        elif framework == "flask":
            deps = ["flask"]

        # Add dependencies based on components
        for component in spec.components:
            comp_type = component.type.lower()

            if "chart" in comp_type or "graph" in comp_type:
                if framework == "react":
                    deps.append("recharts")
                else:
                    deps.append("chart.js")

            if "date" in comp_type or "calendar" in comp_type:
                deps.append("date-fns")

            if "icon" in comp_type:
                deps.append("lucide-react" if framework == "react" else "@lucide/icon")

        # Add styling dependencies
        if (spec.color_palette or spec.styles) and framework in ["react", "nextjs"]:
            deps.append("tailwindcss")

        return list(set(deps))  # Remove duplicates


__all__ = [
    "DesignToCodePipeline",
    "UIComponent",
    "DesignSpec",
    "GeneratedCode",
]
