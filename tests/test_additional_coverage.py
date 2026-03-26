"""
Additional Unit Tests — Increase coverage to 60%+
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Comprehensive unit tests for uncovered modules:
- Component Library
- Full-Stack Generator
- API Builder
- Deployment Service
- GitHub Sync
- Preview Server
- Database Generator
- SwiftStack Integration

USAGE:
    pytest tests/test_additional_coverage.py -v --cov=orchestrator --cov-report=term-missing
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

# Component Library
from orchestrator.component_library import (
    ComponentLibrary, Component, ComponentType, Framework,
    get_component_library, get_component, assemble_ui,
)

# Full-Stack Generator
from orchestrator.fullstack_generator import (
    FullStackGenerator, FullStackApp, GenerationOptions,
    FrontendFramework, BackendFramework, DatabaseType, AuthType,
    get_fullstack_generator, generate_fullstack_app,
)

# API Builder
from orchestrator.api_builder import (
    APIIntegrationBuilder, APIIntegration, APIEndpoint,
    AuthType as APIAuthType, HTTPMethod,
    get_api_builder, import_from_openapi, import_from_postman,
)

# Deployment Service
from orchestrator.deployment_service import (
    DeploymentService, DeploymentTarget, DeploymentStatus,
    DeploymentResult, DeploymentConfig,
    get_deployment_service, deploy_to_vercel, deploy_to_netlify, deploy_to_docker,
)

# GitHub Sync
from orchestrator.github_sync import (
    GitHubSync, SyncConfig, SyncDirection, Change, ChangeOperation,
    get_github_sync, sync_with_github,
)

# Preview Server
from orchestrator.preview_server import (
    PreviewServer, PreviewConfig, PreviewSession,
    get_preview_server, start_preview, stop_preview,
)

# Database Generator
from orchestrator.database_generator import (
    DatabaseSchemaGenerator, DatabaseSchema, Table, Column,
    DatabaseType as DBType, ORMType,
    get_database_generator, generate_database_schema,
)

# SwiftStack Integration
from orchestrator.swiftstack_integration import (
    SwiftStackIntegration, SwiftStackConfig,
    get_swiftstack_integration, generate_app,
)


# ─────────────────────────────────────────────
# Component Library Tests
# ─────────────────────────────────────────────

class TestComponentLibrary:
    """Test component library functionality."""

    def test_component_creation(self):
        """Test component creation."""
        component = Component(
            name="TestButton",
            type=ComponentType.BUTTON,
            variant="primary",
            props={"text": "Click me"},
        )
        
        assert component.name == "TestButton"
        assert component.type == ComponentType.BUTTON
        assert component.props["text"] == "Click me"

    def test_component_render_react(self):
        """Test React component rendering."""
        library = ComponentLibrary(framework=Framework.REACT)
        button = library.get(ComponentType.BUTTON, "primary")
        
        rendered = button.render()
        assert "button" in rendered.lower()
        assert "React" in rendered or "export default" in rendered

    def test_component_render_vue(self):
        """Test Vue component rendering."""
        library = ComponentLibrary(framework=Framework.VUE)
        form = library.get(ComponentType.FORM, "login")
        
        rendered = form.render()
        assert "<template>" in rendered or "Vue" in rendered

    def test_component_assemble(self):
        """Test component assembly."""
        library = ComponentLibrary()
        
        components = [
            library.get(ComponentType.NAVIGATION, "navbar"),
            library.get(ComponentType.FORM, "login"),
        ]
        
        assembled = library.assemble(components, layout="vertical")
        assert len(assembled) > 0

    def test_component_stats(self):
        """Test component statistics."""
        library = ComponentLibrary()
        
        # Use some components
        library.get(ComponentType.FORM, "login")
        library.get(ComponentType.BUTTON, "primary")
        
        stats = library.get_stats()
        assert stats["total_uses"] >= 2
        assert stats["total_tokens_saved"] > 0

    def test_get_component_library_singleton(self):
        """Test component library singleton."""
        lib1 = get_component_library()
        lib2 = get_component_library()
        assert lib1 is lib2

    def test_get_component_convenience(self):
        """Test convenience function."""
        component = get_component(ComponentType.FORM, "login")
        assert component is not None
        assert component.type == ComponentType.FORM


# ─────────────────────────────────────────────
# Full-Stack Generator Tests
# ─────────────────────────────────────────────

class TestFullStackGenerator:
    """Test full-stack generator functionality."""

    def test_generation_options(self):
        """Test generation options."""
        options = GenerationOptions(
            frontend=FrontendFramework.REACT,
            backend=BackendFramework.FASTAPI,
            database=DatabaseType.POSTGRESQL,
            auth=AuthType.JWT,
            include_tests=True,
        )
        
        opts_dict = options.to_dict()
        assert opts_dict["frontend"] == "react"
        assert opts_dict["backend"] == "fastapi"

    def test_fullstack_app_creation(self):
        """Test full-stack app creation."""
        app = FullStackApp(
            name="TestApp",
            description="A test application",
            frontend_framework=FrontendFramework.REACT,
            backend_framework=BackendFramework.FASTAPI,
        )
        
        assert app.name == "TestApp"
        assert app.frontend_framework == FrontendFramework.REACT

    def test_fullstack_app_to_dict(self):
        """Test full-stack app serialization."""
        app = FullStackApp(
            name="TestApp",
            description="Test",
        )
        
        app_dict = app.to_dict()
        assert app_dict["name"] == "TestApp"
        assert app_dict["frontend_framework"] == "react"

    def test_generator_stats(self):
        """Test generator statistics."""
        generator = FullStackGenerator()
        stats = generator.get_stats()
        
        assert "total_generations" in stats
        assert "component_library_stats" in stats


# ─────────────────────────────────────────────
# API Builder Tests
# ─────────────────────────────────────────────

class TestAPIBuilder:
    """Test API builder functionality."""

    def test_api_endpoint_creation(self):
        """Test API endpoint creation."""
        endpoint = APIEndpoint(
            path="/users",
            method=HTTPMethod.GET,
            summary="Get all users",
        )
        
        assert endpoint.path == "/users"
        assert endpoint.method == HTTPMethod.GET

    def test_api_integration_creation(self):
        """Test API integration creation."""
        integration = APIIntegration(
            name="Test API",
            base_url="https://api.example.com",
            auth_type=APIAuthType.BEARER,
        )
        
        assert integration.name == "Test API"
        assert integration.base_url == "https://api.example.com"

    def test_api_integration_to_dict(self):
        """Test API integration serialization."""
        integration = APIIntegration(
            name="Test API",
            base_url="https://api.example.com",
        )
        
        int_dict = integration.to_dict()
        assert int_dict["name"] == "Test API"
        assert int_dict["base_url"] == "https://api.example.com"

    def test_api_integration_from_dict(self):
        """Test API integration deserialization."""
        data = {
            "name": "Test API",
            "base_url": "https://api.example.com",
            "endpoints": [
                {
                    "path": "/users",
                    "method": "get",
                    "summary": "Get users",
                }
            ],
        }
        
        integration = APIIntegration.from_dict(data)
        assert integration.name == "Test API"
        assert len(integration.endpoints) == 1

    def test_builder_stats(self):
        """Test builder statistics."""
        builder = APIIntegrationBuilder()
        stats = builder.get_stats()
        
        assert "total_imports" in stats
        assert "total_endpoints" in stats


# ─────────────────────────────────────────────
# Deployment Service Tests
# ─────────────────────────────────────────────

class TestDeploymentService:
    """Test deployment service functionality."""

    def test_deployment_target_enum(self):
        """Test deployment target enum."""
        assert DeploymentTarget.VERCEL.value == "vercel"
        assert DeploymentTarget.NETLIFY.value == "netlify"
        assert DeploymentTarget.DOCKER.value == "docker"

    def test_deployment_status_enum(self):
        """Test deployment status enum."""
        assert DeploymentStatus.SUCCESS.value == "success"
        assert DeploymentStatus.FAILED.value == "failed"

    def test_deployment_result(self):
        """Test deployment result."""
        result = DeploymentResult(
            success=True,
            target=DeploymentTarget.VERCEL,
            url="https://app.vercel.app",
        )
        
        assert result.success
        assert result.url == "https://app.vercel.app"

    def test_deployment_result_to_dict(self):
        """Test deployment result serialization."""
        result = DeploymentResult(
            success=True,
            target=DeploymentTarget.VERCEL,
        )
        
        result_dict = result.to_dict()
        assert result_dict["success"]
        assert result_dict["target"] == "vercel"

    def test_deployment_config(self):
        """Test deployment configuration."""
        config = DeploymentConfig(
            target=DeploymentTarget.VERCEL,
            project_path="./my-app",
            environment="production",
        )
        
        assert config.target == DeploymentTarget.VERCEL
        assert config.environment == "production"

    def test_deployment_service_stats(self):
        """Test deployment service statistics."""
        service = DeploymentService()
        stats = service.get_stats()
        
        assert "total_deployments" in stats
        assert "providers" in stats


# ─────────────────────────────────────────────
# GitHub Sync Tests
# ─────────────────────────────────────────────

class TestGitHubSync:
    """Test GitHub sync functionality."""

    def test_sync_direction_enum(self):
        """Test sync direction enum."""
        assert SyncDirection.PULL.value == "pull"
        assert SyncDirection.PUSH.value == "push"
        assert SyncDirection.BIDIRECTIONAL.value == "bidirectional"

    def test_change_operation_enum(self):
        """Test change operation enum."""
        assert ChangeOperation.CREATE.value == "create"
        assert ChangeOperation.UPDATE.value == "update"
        assert ChangeOperation.DELETE.value == "delete"

    def test_change_creation(self):
        """Test change creation."""
        change = Change(
            path="src/main.py",
            content="print('hello')",
            operation=ChangeOperation.CREATE,
        )
        
        assert change.path == "src/main.py"
        assert change.operation == ChangeOperation.CREATE

    def test_change_to_dict(self):
        """Test change serialization."""
        change = Change(
            path="src/main.py",
            content="print('hello')",
            operation=ChangeOperation.CREATE,
        )
        
        change_dict = change.to_dict()
        assert change_dict["path"] == "src/main.py"
        assert change_dict["operation"] == "create"

    def test_sync_config(self):
        """Test sync configuration."""
        config = SyncConfig(
            repo_url="https://github.com/user/repo",
            branch="main",
            direction=SyncDirection.BIDIRECTIONAL,
        )
        
        assert config.repo_url == "https://github.com/user/repo"
        assert config.branch == "main"

    def test_sync_stats(self):
        """Test sync statistics."""
        sync = GitHubSync(token="test_token")
        stats = sync.get_stats()
        
        assert "total_syncs" in stats
        assert "connected_repos" in stats


# ─────────────────────────────────────────────
# Preview Server Tests
# ─────────────────────────────────────────────

class TestPreviewServer:
    """Test preview server functionality."""

    def test_preview_config(self):
        """Test preview configuration."""
        config = PreviewConfig(
            port=3000,
            host="localhost",
            hot_reload=True,
        )
        
        assert config.port == 3000
        assert config.hot_reload

    def test_preview_session(self):
        """Test preview session."""
        session = PreviewSession(
            project_path="./my-app",
            url="http://localhost:3000",
            port=3000,
        )
        
        assert session.project_path == "./my-app"
        assert session.url == "http://localhost:3000"

    def test_preview_session_to_dict(self):
        """Test preview session serialization."""
        session = PreviewSession(
            project_path="./my-app",
            url="http://localhost:3000",
            port=3000,
        )
        
        session_dict = session.to_dict()
        assert session_dict["project_path"] == "./my-app"
        assert session_dict["url"] == "http://localhost:3000"

    def test_preview_server_singleton(self):
        """Test preview server singleton."""
        server1 = PreviewServer()
        server2 = PreviewServer()
        assert server1 is server2

    def test_preview_server_stats(self):
        """Test preview server statistics."""
        server = PreviewServer()
        stats = server.get_stats()
        
        assert "total_previews" in stats
        assert "active_sessions" in stats


# ─────────────────────────────────────────────
# Database Generator Tests
# ─────────────────────────────────────────────

class TestDatabaseGenerator:
    """Test database generator functionality."""

    def test_column_creation(self):
        """Test column creation."""
        column = Column(
            name="id",
            type="UUID",
            primary_key=True,
        )
        
        assert column.name == "id"
        assert column.primary_key

    def test_column_to_dict(self):
        """Test column serialization."""
        column = Column(
            name="email",
            type="VARCHAR(255)",
            nullable=False,
            unique=True,
        )
        
        col_dict = column.to_dict()
        assert col_dict["name"] == "email"
        assert col_dict["nullable"] == False
        assert col_dict["unique"] == True

    def test_table_creation(self):
        """Test table creation."""
        table = Table(
            name="users",
            columns=[
                Column("id", "UUID", primary_key=True),
                Column("email", "VARCHAR(255)"),
            ],
        )
        
        assert table.name == "users"
        assert len(table.columns) == 2

    def test_table_to_dict(self):
        """Test table serialization."""
        table = Table(
            name="users",
            columns=[Column("id", "UUID", primary_key=True)],
        )
        
        table_dict = table.to_dict()
        assert table_dict["name"] == "users"
        assert len(table_dict["columns"]) == 1

    def test_database_schema(self):
        """Test database schema."""
        schema = DatabaseSchema(
            tables=[
                Table("users", [Column("id", "UUID", primary_key=True)]),
            ],
        )
        
        assert len(schema.tables) == 1

    def test_database_schema_to_dict(self):
        """Test database schema serialization."""
        schema = DatabaseSchema(
            tables=[Table("users", [Column("id", "UUID", primary_key=True)])],
        )
        
        schema_dict = schema.to_dict()
        assert len(schema_dict["tables"]) == 1

    def test_generator_stats(self):
        """Test generator statistics."""
        generator = DatabaseSchemaGenerator()
        stats = generator.get_stats()
        
        assert "total_generations" in stats


# ─────────────────────────────────────────────
# SwiftStack Integration Tests
# ─────────────────────────────────────────────

class TestSwiftStackIntegration:
    """Test SwiftStack integration functionality."""

    def test_swiftstack_config(self):
        """Test SwiftStack configuration."""
        config = SwiftStackConfig(
            enable_all=True,
            github_token="ghp_test",
            preview_port=3000,
        )
        
        assert config.enable_all
        assert config.github_token == "ghp_test"
        assert config.preview_port == 3000

    def test_swiftstack_config_enable_all(self):
        """Test enable_all flag."""
        # Default config has all features enabled
        config = SwiftStackConfig()
        assert config.component_library_enabled
        assert config.deployment_enabled
        
        # enable_all=True keeps all features enabled
        config = SwiftStackConfig(enable_all=True)
        assert config.component_library_enabled
        assert config.deployment_enabled

    def test_swiftstack_integration_creation(self):
        """Test SwiftStack integration creation."""
        config = SwiftStackConfig(enable_all=True)
        integration = SwiftStackIntegration(config)
        
        assert integration.component_library is not None
        assert integration.deployment_service is not None
        assert integration.fullstack_generator is not None

    def test_swiftstack_integration_stats(self):
        """Test SwiftStack integration statistics."""
        config = SwiftStackConfig(enable_all=True)
        integration = SwiftStackIntegration(config)
        
        stats = integration.get_stats()
        
        assert "enabled_features" in stats
        assert len(stats["enabled_features"]) > 0

    def test_swiftstack_integration_get_component(self):
        """Test getting component from integration."""
        config = SwiftStackConfig(enable_all=True)
        integration = SwiftStackIntegration(config)
        
        component = integration.get_component("form", "login")
        assert component is not None
        assert component.type == ComponentType.FORM


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
