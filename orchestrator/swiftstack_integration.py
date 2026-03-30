"""
SwiftStack Integration — Unified interface for all SwiftStack-inspired features
==========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Unified integration layer that combines all SwiftStack-inspired features:
- Component Library
- Full-Stack Generator
- Deployment Service
- GitHub Sync
- API Builder
- Preview Server
- Database Generator

USAGE:
    from orchestrator.swiftstack_integration import SwiftStackIntegration, SwiftStackConfig

    config = SwiftStackConfig(
        enable_all=True,
        component_library_enabled=True,
        deployment_enabled=True,
    )

    integration = SwiftStackIntegration(config)

    # Generate and deploy complete app
    app = await integration.generate_fullstack(
        "Build a task management app",
        deploy=True,
        target="vercel",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .api_builder import APIIntegration, APIIntegrationBuilder
from .component_library import Component, ComponentLibrary, ComponentType
from .database_generator import DatabaseSchemaGenerator, DatabaseType
from .deployment_service import DeploymentService, DeploymentTarget
from .fullstack_generator import FullStackApp, FullStackGenerator, GenerationOptions
from .github_sync import GitHubSync
from .preview_server import PreviewConfig, PreviewServer

logger = logging.getLogger("orchestrator.swiftstack_integration")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class SwiftStackConfig:
    """SwiftStack features configuration."""

    # Feature flags
    component_library_enabled: bool = True
    deployment_enabled: bool = True
    github_sync_enabled: bool = True
    fullstack_generator_enabled: bool = True
    api_builder_enabled: bool = True
    preview_server_enabled: bool = True
    database_generator_enabled: bool = True

    # Convenience: enable all features
    enable_all: bool = False

    # GitHub configuration
    github_token: str | None = None

    # Deployment configuration
    vercel_token: str | None = None
    netlify_token: str | None = None

    # Preview configuration
    preview_port: int = 3000
    preview_hot_reload: bool = True

    def __post_init__(self):
        """Post-initialization processing."""
        if self.enable_all:
            self.component_library_enabled = True
            self.deployment_enabled = True
            self.github_sync_enabled = True
            self.fullstack_generator_enabled = True
            self.api_builder_enabled = True
            self.preview_server_enabled = True
            self.database_generator_enabled = True


# ─────────────────────────────────────────────
# SwiftStack Integration
# ─────────────────────────────────────────────

class SwiftStackIntegration:
    """
    Unified integration for all SwiftStack-inspired features.

    Provides a single interface for generating, building, and deploying
    complete full-stack applications.
    """

    def __init__(self, config: SwiftStackConfig | None = None):
        """
        Initialize SwiftStack integration.

        Args:
            config: Integration configuration
        """
        self.config = config or SwiftStackConfig()

        # Initialize enabled components
        self.component_library: ComponentLibrary | None = None
        self.deployment_service: DeploymentService | None = None
        self.github_sync: GitHubSync | None = None
        self.fullstack_generator: FullStackGenerator | None = None
        self.api_builder: APIIntegrationBuilder | None = None
        self.preview_server: PreviewServer | None = None
        self.database_generator: DatabaseSchemaGenerator | None = None

        self._initialize_components()

        # Statistics
        self._total_projects = 0
        self._total_deployments = 0

        logger.info("SwiftStackIntegration initialized")

    def _initialize_components(self):
        """Initialize enabled components."""
        if self.config.component_library_enabled:
            self.component_library = ComponentLibrary()
            logger.info("Component Library enabled")

        if self.config.deployment_enabled:
            self.deployment_service = DeploymentService()
            logger.info("Deployment Service enabled")

        if self.config.github_sync_enabled:
            self.github_sync = GitHubSync(token=self.config.github_token)
            logger.info("GitHub Sync enabled")

        if self.config.fullstack_generator_enabled:
            self.fullstack_generator = FullStackGenerator(
                component_library=self.component_library,
                deployment_service=self.deployment_service,
            )
            logger.info("Full-Stack Generator enabled")

        if self.config.api_builder_enabled:
            self.api_builder = APIIntegrationBuilder()
            logger.info("API Builder enabled")

        if self.config.preview_server_enabled:
            self.preview_server = PreviewServer()
            logger.info("Preview Server enabled")

        if self.config.database_generator_enabled:
            self.database_generator = DatabaseSchemaGenerator()
            logger.info("Database Generator enabled")

    async def generate_fullstack(
        self,
        description: str,
        options: dict | None = None,
        deploy: bool = False,
        target: str = "vercel",
        sync_github: bool = False,
        github_repo: str | None = None,
        start_preview: bool = False,
    ) -> FullStackApp:
        """
        Generate complete full-stack application.

        Args:
            description: Project description
            options: Generation options
            deploy: Deploy after generation
            target: Deployment target
            sync_github: Sync with GitHub
            github_repo: GitHub repository URL
            start_preview: Start preview server

        Returns:
            Generated application
        """
        if not self.fullstack_generator:
            raise RuntimeError("Full-Stack Generator not enabled")

        # Parse options
        gen_options = None
        if options:
            gen_options = GenerationOptions(**options)

        # Generate application
        logger.info(f"Generating full-stack app: {description[:50]}...")
        app = await self.fullstack_generator.generate(description, gen_options)

        self._total_projects += 1

        # Sync with GitHub
        if sync_github and self.github_sync and github_repo:
            logger.info(f"Syncing with GitHub: {github_repo}")
            await self.github_sync.connect(github_repo)
            # In a real implementation, would push generated files

        # Deploy
        if deploy and self.deployment_service:
            logger.info(f"Deploying to {target}")
            result = await self.deployment_service.deploy(
                project_path=f"./{app.name}",
                target=DeploymentTarget(target),
                config={
                    "vercel_token": self.config.vercel_token,
                    "netlify_token": self.config.netlify_token,
                },
            )

            if result.success:
                self._total_deployments += 1
                app.deployment_config["deployment_url"] = result.url

        # Start preview
        if start_preview and self.preview_server:
            logger.info("Starting preview server")
            await self.preview_server.start(
                f"./{app.name}",
                PreviewConfig(
                    port=self.config.preview_port,
                    hot_reload=self.config.preview_hot_reload,
                ),
            )

        logger.info(f"Generated {app.name}: {app.total_files} files, {app.total_lines} lines")

        return app

    async def import_api(
        self,
        spec_url: str,
        name: str | None = None,
        auth_type: str = "none",
        auth_credentials: dict | None = None,
        generate_client: bool = True,
        language: str = "python",
    ) -> APIIntegration:
        """
        Import API integration.

        Args:
            spec_url: OpenAPI spec URL or path
            name: API name
            auth_type: Authentication type
            auth_credentials: Authentication credentials
            generate_client: Generate client code
            language: Client language

        Returns:
            API integration
        """
        if not self.api_builder:
            raise RuntimeError("API Builder not enabled")

        # Import from spec
        logger.info(f"Importing API from: {spec_url}")
        integration = await self.api_builder.from_openapi(spec_url, name)

        # Configure auth
        if auth_type != "none" and auth_credentials:
            integration = self.api_builder.configure_auth(
                integration,
                auth_type,
                auth_credentials,
            )

        # Generate client
        if generate_client:
            logger.info(f"Generating {language} client")
            integration.client_code = self.api_builder.generate_client(
                integration,
                language,
            )

        return integration

    async def generate_database(
        self,
        description: str,
        db_type: str = "postgresql",
        generate_migrations: bool = True,
        generate_models: bool = True,
        orm: str = "sqlalchemy",
    ) -> dict[str, Any]:
        """
        Generate database schema and models.

        Args:
            description: Project description
            db_type: Database type
            generate_migrations: Generate migration files
            generate_models: Generate ORM models
            orm: ORM type

        Returns:
            Generated database artifacts
        """
        if not self.database_generator:
            raise RuntimeError("Database Generator not enabled")

        result = {}

        # Generate schema
        logger.info(f"Generating database schema for: {description[:50]}...")
        schema = await self.database_generator.from_description(
            description,
            DatabaseType(db_type),
        )
        result["schema"] = schema

        # Generate migrations
        if generate_migrations:
            logger.info("Generating migrations")
            result["migrations"] = await self.database_generator.generate_migrations(
                schema,
                DatabaseType(db_type),
            )

        # Generate models
        if generate_models:
            logger.info(f"Generating {orm} models")
            from .database_generator import ORMType
            result["models"] = self.database_generator.generate_models(
                schema,
                ORMType(orm),
            )

        return result

    def get_component(self, type: str, variant: str = "default") -> Component:
        """
        Get pre-built component.

        Args:
            type: Component type
            variant: Component variant

        Returns:
            Component
        """
        if not self.component_library:
            raise RuntimeError("Component Library not enabled")

        return self.component_library.get(ComponentType(type), variant)

    async def start_preview(
        self,
        project_path: str,
        port: int | None = None,
    ) -> str:
        """
        Start preview server.

        Args:
            project_path: Project path
            port: Server port

        Returns:
            Preview URL
        """
        if not self.preview_server:
            raise RuntimeError("Preview Server not enabled")

        config = PreviewConfig(
            port=port or self.config.preview_port,
            hot_reload=self.config.preview_hot_reload,
        )

        return await self.preview_server.start(project_path, config)

    def get_stats(self) -> dict[str, Any]:
        """Get integration statistics."""
        stats = {
            "total_projects": self._total_projects,
            "total_deployments": self._total_deployments,
            "enabled_features": [],
        }

        if self.component_library:
            stats["enabled_features"].append("component_library")
            stats["component_library"] = self.component_library.get_stats()

        if self.deployment_service:
            stats["enabled_features"].append("deployment")
            stats["deployment"] = self.deployment_service.get_stats()

        if self.github_sync:
            stats["enabled_features"].append("github_sync")
            stats["github_sync"] = self.github_sync.get_stats()

        if self.fullstack_generator:
            stats["enabled_features"].append("fullstack_generator")
            stats["fullstack_generator"] = self.fullstack_generator.get_stats()

        if self.api_builder:
            stats["enabled_features"].append("api_builder")
            stats["api_builder"] = self.api_builder.get_stats()

        if self.preview_server:
            stats["enabled_features"].append("preview_server")
            stats["preview_server"] = self.preview_server.get_stats()

        if self.database_generator:
            stats["enabled_features"].append("database_generator")
            stats["database_generator"] = self.database_generator.get_stats()

        return stats


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_integration: SwiftStackIntegration | None = None


def get_swiftstack_integration(
    config: SwiftStackConfig | None = None,
) -> SwiftStackIntegration:
    """Get or create default SwiftStack integration."""
    global _default_integration
    if _default_integration is None:
        _default_integration = SwiftStackIntegration(config)
    return _default_integration


def reset_swiftstack_integration() -> None:
    """Reset default integration (for testing)."""
    global _default_integration
    _default_integration = None


async def generate_app(
    description: str,
    deploy: bool = False,
    target: str = "vercel",
) -> FullStackApp:
    """
    Quick app generation.

    Args:
        description: Project description
        deploy: Deploy after generation
        target: Deployment target

    Returns:
        Generated app
    """
    integration = get_swiftstack_integration(SwiftStackConfig(enable_all=True))
    return await integration.generate_fullstack(
        description,
        deploy=deploy,
        target=target,
    )
