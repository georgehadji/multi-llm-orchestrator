"""
Tests for Multi-Platform Generator
===================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_multi_platform_generator.py -v
"""

import pytest
import asyncio
from pathlib import Path

from orchestrator.multi_platform_generator import (
    MultiPlatformGenerator,
    OutputTarget,
    ProjectOutputConfig,
    PlatformOutput,
    GeneratedFile,
    MultiPlatformResult,
    generate_multi_platform,
)


class TestOutputTarget:
    """Test OutputTarget enum."""

    def test_output_target_values(self):
        """Test all output target values."""
        assert OutputTarget.PYTHON_LIBRARY.value == "python"
        assert OutputTarget.REACT_WEB_APP.value == "react"
        assert OutputTarget.REACT_NATIVE_MOBILE.value == "react_native"
        assert OutputTarget.SWIFTUI_IOS.value == "swiftui"
        assert OutputTarget.KOTLIN_ANDROID.value == "kotlin"
        assert OutputTarget.FASTAPI_BACKEND.value == "fastapi"
        assert OutputTarget.FLASK_BACKEND.value == "flask"
        assert OutputTarget.FULL_STACK.value == "full_stack"
        assert OutputTarget.PWA.value == "pwa"


class TestProjectOutputConfig:
    """Test ProjectOutputConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ProjectOutputConfig()
        assert config.targets == []
        assert config.ios_deployment is False
        assert config.include_privacy_policy is True
        assert config.include_tests is True
        assert config.use_typescript is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProjectOutputConfig(
            targets=[OutputTarget.REACT_WEB_APP, OutputTarget.FASTAPI_BACKEND],
            ios_deployment=True,
            android_deployment=True,
            include_tests=False,
            use_typescript=False,
        )
        assert len(config.targets) == 2
        assert config.ios_deployment is True
        assert config.android_deployment is True
        assert config.include_tests is False
        assert config.use_typescript is False

    def test_config_to_dict(self):
        """Test config serialization."""
        config = ProjectOutputConfig(
            targets=[OutputTarget.PYTHON_LIBRARY],
            database_type="postgresql",
        )
        config_dict = config.to_dict()

        assert config_dict["targets"] == ["python"]
        assert config_dict["database_type"] == "postgresql"
        assert config_dict["include_privacy_policy"] is True


class TestGeneratedFile:
    """Test GeneratedFile dataclass."""

    def test_generated_file_creation(self):
        """Test creating a generated file."""
        file = GeneratedFile(
            path="src/main.py",
            content="print('hello')",
            description="Main entry point",
        )

        assert file.path == "src/main.py"
        assert file.content == "print('hello')"
        assert file.description == "Main entry point"
        assert file.is_binary is False


class TestPlatformOutput:
    """Test PlatformOutput dataclass."""

    def test_platform_output_creation(self):
        """Test creating platform output."""
        output = PlatformOutput(target=OutputTarget.PYTHON_LIBRARY)

        assert output.target == OutputTarget.PYTHON_LIBRARY
        assert len(output.files) == 0
        assert len(output.dependencies) == 0

    def test_add_file(self):
        """Test adding files to platform output."""
        output = PlatformOutput(target=OutputTarget.REACT_WEB_APP)
        output.add_file("app/page.tsx", "export default function Page() {{}}")

        assert len(output.files) == 1
        assert output.files[0].path == "app/page.tsx"

    def test_to_dict(self):
        """Test platform output serialization."""
        output = PlatformOutput(target=OutputTarget.FASTAPI_BACKEND)
        output.add_file("main.py", "from fastapi import FastAPI")
        output.dependencies = ["fastapi", "uvicorn"]
        output.scripts = {"dev": "uvicorn main:app --reload"}

        output_dict = output.to_dict()

        assert output_dict["target"] == "fastapi"
        assert output_dict["file_count"] == 1
        assert "fastapi" in output_dict["dependencies"]


class TestMultiPlatformResult:
    """Test MultiPlatformResult dataclass."""

    def test_result_creation(self):
        """Test creating multi-platform result."""
        result = MultiPlatformResult(
            project_name="test_app",
            project_description="Test project",
        )

        assert result.project_name == "test_app"
        assert result.total_files == 0

    def test_result_with_outputs(self):
        """Test result with platform outputs."""
        result = MultiPlatformResult(
            project_name="test_app",
            project_description="Test project",
        )

        react_output = PlatformOutput(target=OutputTarget.REACT_WEB_APP)
        react_output.add_file("app/page.tsx", "content")
        result.outputs[OutputTarget.REACT_WEB_APP] = react_output

        assert result.total_files == 1
        assert OutputTarget.REACT_WEB_APP in result.outputs

    def test_result_summary(self):
        """Test result summary string."""
        result = MultiPlatformResult(
            project_name="my_app",
            project_description="My test project",
        )

        summary = result.summary()

        assert "my_app" in summary
        assert "Total files: 0" in summary


class TestMultiPlatformGenerator:
    """Test MultiPlatformGenerator class."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = MultiPlatformGenerator()

        assert generator.output_dir == Path("./generated")
        assert len(generator._generators) == 9  # All platform generators

    def test_generator_custom_output_dir(self):
        """Test generator with custom output directory."""
        generator = MultiPlatformGenerator(output_dir=Path("/custom/output"))

        assert generator.output_dir == Path("/custom/output")

    @pytest.mark.asyncio
    async def test_generate_python_only(self):
        """Test generating Python library only."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.PYTHON_LIBRARY],
            include_tests=True,
        )

        result = await generator.generate(
            project_description="Build a Python utility library",
            config=config,
            project_name="test_python",
        )

        assert result.project_name == "test_python"
        assert OutputTarget.PYTHON_LIBRARY in result.outputs
        assert result.total_files > 0

    @pytest.mark.asyncio
    async def test_generate_react_web_app(self):
        """Test generating React web app."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.REACT_WEB_APP],
            use_typescript=True,
            include_tests=True,
        )

        result = await generator.generate(
            project_description="Build a React web application",
            config=config,
            project_name="test_react",
        )

        assert result.project_name == "test_react"
        assert OutputTarget.REACT_WEB_APP in result.outputs

        react_output = result.outputs[OutputTarget.REACT_WEB_APP]
        assert len(react_output.files) > 0
        assert "package.json" in [f.path for f in react_output.files]

    @pytest.mark.asyncio
    async def test_generate_react_native(self):
        """Test generating React Native mobile app."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.REACT_NATIVE_MOBILE],
            use_typescript=True,
        )

        result = await generator.generate(
            project_description="Build a React Native mobile app",
            config=config,
            project_name="test_rn",
        )

        assert result.project_name == "test_rn"
        assert OutputTarget.REACT_NATIVE_MOBILE in result.outputs

        rn_output = result.outputs[OutputTarget.REACT_NATIVE_MOBILE]
        assert len(rn_output.files) > 0
        assert any("App." in f.path for f in rn_output.files)

    @pytest.mark.asyncio
    async def test_generate_swiftui_ios(self):
        """Test generating SwiftUI iOS app."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.SWIFTUI_IOS],
            ios_deployment=True,
            include_privacy_policy=True,
        )

        result = await generator.generate(
            project_description="Build an iOS app",
            config=config,
            project_name="test_ios",
        )

        assert result.project_name == "test_ios"
        assert OutputTarget.SWIFTUI_IOS in result.outputs

        ios_output = result.outputs[OutputTarget.SWIFTUI_IOS]
        assert len(ios_output.files) > 0
        assert any("App.swift" in f.path for f in ios_output.files)

    @pytest.mark.asyncio
    async def test_generate_fastapi_backend(self):
        """Test generating FastAPI backend."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.FASTAPI_BACKEND],
            include_database=True,
            include_auth=True,
            include_tests=True,
        )

        result = await generator.generate(
            project_description="Build a REST API backend",
            config=config,
            project_name="test_api",
        )

        assert result.project_name == "test_api"
        assert OutputTarget.FASTAPI_BACKEND in result.outputs

        api_output = result.outputs[OutputTarget.FASTAPI_BACKEND]
        assert len(api_output.files) > 0
        assert "main.py" in [f.path for f in api_output.files]
        assert "requirements.txt" in [f.path for f in api_output.files]

    @pytest.mark.asyncio
    async def test_generate_full_stack(self):
        """Test generating full-stack application."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.FULL_STACK],
            include_database=True,
            include_tests=True,
        )

        result = await generator.generate(
            project_description="Build a full-stack web application",
            config=config,
            project_name="test_fullstack",
        )

        assert result.project_name == "test_fullstack"
        assert OutputTarget.FULL_STACK in result.outputs

        fullstack_output = result.outputs[OutputTarget.FULL_STACK]
        assert len(fullstack_output.files) > 0
        assert "docker-compose.yml" in [f.path for f in fullstack_output.files]

    @pytest.mark.asyncio
    async def test_generate_multi_platform(self):
        """Test generating multiple platforms at once."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[
                OutputTarget.REACT_WEB_APP,
                OutputTarget.FASTAPI_BACKEND,
                OutputTarget.SWIFTUI_IOS,
            ],
            include_tests=True,
        )

        result = await generator.generate(
            project_description="Build a multi-platform app",
            config=config,
            project_name="test_multi",
        )

        assert result.project_name == "test_multi"
        assert len(result.outputs) == 3
        assert OutputTarget.REACT_WEB_APP in result.outputs
        assert OutputTarget.FASTAPI_BACKEND in result.outputs
        assert OutputTarget.SWIFTUI_IOS in result.outputs
        assert result.total_files > 0

    @pytest.mark.asyncio
    async def test_generate_shared_files(self):
        """Test that shared files are generated."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.PYTHON_LIBRARY],
            include_privacy_policy=True,
            include_ci_cd=True,
        )

        result = await generator.generate(
            project_description="Test project",
            config=config,
            project_name="test_shared",
        )

        shared_paths = [f.path for f in result.shared_files]

        assert "README.md" in shared_paths
        assert "LICENSE" in shared_paths
        assert "PRIVACY_POLICY.md" in shared_paths
        assert ".gitignore" in shared_paths
        assert ".github/workflows/ci.yml" in shared_paths

    @pytest.mark.asyncio
    async def test_generate_with_default_config(self):
        """Test generation with default (None) config."""
        generator = MultiPlatformGenerator()

        result = await generator.generate(
            project_description="Test with defaults",
            config=None,
            project_name="test_defaults",
        )

        # Should use default config with Python library
        assert result.config is not None
        assert OutputTarget.PYTHON_LIBRARY in result.outputs

    @pytest.mark.asyncio
    async def test_infer_project_name(self):
        """Test project name inference from description."""
        generator = MultiPlatformGenerator()

        name = generator._infer_project_name("Build a todo application")
        assert "todo" in name.lower()

        name = generator._infer_project_name("E-commerce platform with React")
        assert len(name) <= 50  # Limited length


class TestConvenienceFunction:
    """Test generate_multi_platform convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience function."""
        result = await generate_multi_platform(
            project_description="Test convenience",
            config=ProjectOutputConfig(
                targets=[OutputTarget.PYTHON_LIBRARY],
            ),
            project_name="test_convenience",
        )

        assert isinstance(result, MultiPlatformResult)
        assert result.project_name == "test_convenience"


class TestPlatformSpecificContent:
    """Test platform-specific generated content."""

    @pytest.mark.asyncio
    async def test_react_typescript(self):
        """Test React app with TypeScript."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.REACT_WEB_APP],
            use_typescript=True,
        )

        result = await generator.generate(
            project_description="TS React app",
            config=config,
            project_name="ts_app",
        )

        react_output = result.outputs[OutputTarget.REACT_WEB_APP]
        file_paths = [f.path for f in react_output.files]

        # Should have TypeScript files
        assert any(".tsx" in p or ".ts" in p for p in file_paths)
        assert "tsconfig.json" in file_paths

    @pytest.mark.asyncio
    async def test_react_javascript(self):
        """Test React app without TypeScript."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.REACT_WEB_APP],
            use_typescript=False,
        )

        result = await generator.generate(
            project_description="JS React app",
            config=config,
            project_name="js_app",
        )

        react_output = result.outputs[OutputTarget.REACT_WEB_APP]
        file_paths = [f.path for f in react_output.files]

        # Should have JavaScript files (no tsconfig)
        assert "tsconfig.json" not in file_paths

    @pytest.mark.asyncio
    async def test_fastapi_with_auth(self):
        """Test FastAPI with authentication."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.FASTAPI_BACKEND],
            include_auth=True,
        )

        result = await generator.generate(
            project_description="API with auth",
            config=config,
            project_name="auth_api",
        )

        api_output = result.outputs[OutputTarget.FASTAPI_BACKEND]
        main_file = next((f for f in api_output.files if f.path == "main.py"), None)

        assert main_file is not None
        assert "OAuth2" in main_file.content

    @pytest.mark.asyncio
    async def test_ios_privacy_policy(self):
        """Test iOS app includes privacy policy."""
        generator = MultiPlatformGenerator()
        config = ProjectOutputConfig(
            targets=[OutputTarget.SWIFTUI_IOS],
            include_privacy_policy=True,
        )

        result = await generator.generate(
            project_description="iOS app",
            config=config,
            project_name="privacy_ios",
        )

        ios_output = result.outputs[OutputTarget.SWIFTUI_IOS]
        file_paths = [f.path for f in ios_output.files]

        assert any("Privacy" in p for p in file_paths)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
