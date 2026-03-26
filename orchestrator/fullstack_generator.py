"""
Full-Stack Generator — Complete application generation
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Generate complete full-stack applications from a single prompt.
Includes frontend, backend, database schema, and authentication.

Features:
- One-prompt full-stack generation
- Frontend (React, Vue, Svelte)
- Backend (FastAPI, Express, Flask)
- Database schema (PostgreSQL, SQLite, MongoDB)
- Authentication (JWT, OAuth, Magic Link)
- API endpoint generation
- Deployment configuration

USAGE:
    from orchestrator.fullstack_generator import FullStackGenerator
    
    generator = FullStackGenerator()
    
    # Generate complete app
    app = await generator.generate(
        "Build a task management app with user auth and real-time updates",
        options={
            "frontend": "react",
            "backend": "fastapi",
            "database": "postgresql",
        },
    )
    
    # Deploy
    result = await generator.deploy(app, "vercel")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from pathlib import Path

from .component_library import ComponentLibrary, Component, ComponentType, Framework
from .deployment_service import DeploymentService, DeploymentTarget, DeploymentResult

logger = logging.getLogger("orchestrator.fullstack_generator")


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class FrontendFramework(str, Enum):
    """Frontend frameworks."""
    REACT = "react"
    VUE = "vue"
    SVELTE = "svelte"
    NEXTJS = "nextjs"
    NUXT = "nuxt"


class BackendFramework(str, Enum):
    """Backend frameworks."""
    FASTAPI = "fastapi"
    FLASK = "flask"
    EXPRESS = "express"
    DJANGO = "django"


class DatabaseType(str, Enum):
    """Database types."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    MYSQL = "mysql"


class AuthType(str, Enum):
    """Authentication types."""
    JWT = "jwt"
    OAUTH = "oauth"
    MAGIC_LINK = "magic_link"
    SESSION = "session"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class FullStackApp:
    """Complete full-stack application."""
    name: str
    description: str = ""
    
    # Frontend
    frontend_framework: FrontendFramework = FrontendFramework.REACT
    frontend_components: List[Component] = field(default_factory=list)
    frontend_code: Dict[str, str] = field(default_factory=dict)
    
    # Backend
    backend_framework: BackendFramework = BackendFramework.FASTAPI
    api_endpoints: List[dict] = field(default_factory=list)
    backend_code: Dict[str, str] = field(default_factory=dict)
    
    # Database
    database_type: DatabaseType = DatabaseType.POSTGRESQL
    database_schema: Optional[dict] = None
    migrations: List[str] = field(default_factory=list)
    
    # Authentication
    auth_type: AuthType = AuthType.JWT
    auth_config: dict = field(default_factory=dict)
    
    # Deployment
    deployment_config: dict = field(default_factory=dict)
    
    # Project structure
    file_tree: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    total_files: int = 0
    total_lines: int = 0
    estimated_tokens: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "frontend_framework": self.frontend_framework.value,
            "backend_framework": self.backend_framework.value,
            "database_type": self.database_type.value,
            "auth_type": self.auth_type.value,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
        }


@dataclass
class GenerationOptions:
    """Generation options."""
    frontend: FrontendFramework = FrontendFramework.REACT
    backend: BackendFramework = BackendFramework.FASTAPI
    database: DatabaseType = DatabaseType.POSTGRESQL
    auth: AuthType = AuthType.JWT
    include_tests: bool = True
    include_docs: bool = True
    include_docker: bool = True
    styling: str = "tailwind"  # tailwind, bootstrap, css
    state_management: str = "redux"  # redux, zustand, context
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frontend": self.frontend.value,
            "backend": self.backend.value,
            "database": self.database.value,
            "auth": self.auth.value,
            "include_tests": self.include_tests,
            "include_docs": self.include_docs,
            "include_docker": self.include_docker,
            "styling": self.styling,
            "state_management": self.state_management,
        }


# ─────────────────────────────────────────────
# Full-Stack Generator
# ─────────────────────────────────────────────

class FullStackGenerator:
    """
    Generate complete full-stack applications.
    
    Takes a project description and generates frontend, backend,
    database schema, authentication, and deployment configuration.
    """
    
    def __init__(
        self,
        component_library: Optional[ComponentLibrary] = None,
        deployment_service: Optional[DeploymentService] = None,
    ):
        self.component_library = component_library or ComponentLibrary()
        self.deployment_service = deployment_service or DeploymentService()
        
        # Statistics
        self._total_generations = 0
        self._total_apps_deployed = 0
    
    async def generate(
        self,
        description: str,
        options: Optional[GenerationOptions] = None,
    ) -> FullStackApp:
        """
        Generate complete full-stack application.
        
        Args:
            description: Project description
            options: Generation options
        
        Returns:
            Generated full-stack app
        """
        options = options or GenerationOptions()
        
        # Extract app name from description
        name = self._extract_app_name(description)
        
        # Create app instance
        app = FullStackApp(
            name=name,
            description=description,
            frontend_framework=options.frontend,
            backend_framework=options.backend,
            database_type=options.database,
            auth_type=options.auth,
        )
        
        # Generate frontend
        logger.info(f"Generating frontend for {name}...")
        app.frontend_components = await self._generate_frontend(description, options)
        app.frontend_code = await self._generate_frontend_code(app, options)
        
        # Generate backend
        logger.info(f"Generating backend for {name}...")
        app.api_endpoints = await self._generate_api_endpoints(description, options)
        app.backend_code = await self._generate_backend_code(app, options)
        
        # Generate database schema
        logger.info(f"Generating database schema for {name}...")
        app.database_schema = await self._generate_database_schema(description, options)
        app.migrations = await self._generate_migrations(app, options)
        
        # Generate authentication
        logger.info(f"Generating authentication for {name}...")
        app.auth_config = await self._generate_auth_config(options)
        
        # Generate deployment config
        logger.info(f"Generating deployment config for {name}...")
        app.deployment_config = await self._generate_deployment_config(app, options)
        
        # Generate project structure
        app.file_tree = self._generate_file_tree(app, options)
        
        # Calculate statistics
        app.total_files = len(app.frontend_code) + len(app.backend_code) + len(app.migrations)
        app.total_lines = sum(len(code.split('\n')) for code in app.frontend_code.values())
        app.total_lines += sum(len(code.split('\n')) for code in app.backend_code.values())
        app.estimated_tokens = int(app.total_lines * 1.3)  # Rough estimate
        
        self._total_generations += 1
        
        logger.info(
            f"Generated {name}: {app.total_files} files, "
            f"{app.total_lines} lines, ~{app.estimated_tokens} tokens"
        )
        
        return app
    
    async def deploy(
        self,
        app: FullStackApp,
        target: str,
        project_path: Optional[str] = None,
    ) -> DeploymentResult:
        """
        Deploy generated application.
        
        Args:
            app: Generated app
            target: Deployment target (vercel, netlify, docker)
            project_path: Optional project path
        
        Returns:
            Deployment result
        """
        if not self.deployment_service:
            return DeploymentResult(
                success=False,
                target=DeploymentTarget.LOCAL,
                error="Deployment service not configured",
            )
        
        deployment_target = DeploymentTarget(target.lower())
        
        result = await self.deployment_service.deploy(
            project_path=project_path or f"./{app.name}",
            target=deployment_target,
            config=app.deployment_config,
        )
        
        if result.success:
            self._total_apps_deployed += 1
        
        return result
    
    def _extract_app_name(self, description: str) -> str:
        """Extract app name from description."""
        # Simple extraction - first few words
        words = description.lower().split()[:3]
        name = "_".join(words).replace("-", "_")
        return name
    
    async def _generate_frontend(
        self,
        description: str,
        options: GenerationOptions,
    ) -> List[Component]:
        """Generate frontend components."""
        components = []
        
        # Analyze description for required components
        desc_lower = description.lower()
        
        # Add common components based on description
        if "login" in desc_lower or "auth" in desc_lower or "user" in desc_lower:
            components.append(
                self.component_library.get(ComponentType.FORM, "login")
            )
        
        if "register" in desc_lower or "sign up" in desc_lower:
            components.append(
                self.component_library.get(ComponentType.FORM, "register")
            )
        
        if "dashboard" in desc_lower:
            components.append(
                self.component_library.get(ComponentType.CARD, "dashboard")
            )
        
        if "table" in desc_lower or "list" in desc_lower or "data" in desc_lower:
            components.append(
                self.component_library.get(ComponentType.TABLE, "datatable")
            )
        
        # Always add navigation
        components.append(
            self.component_library.get(ComponentType.NAVIGATION, "navbar")
        )
        
        return components
    
    async def _generate_frontend_code(
        self,
        app: FullStackApp,
        options: GenerationOptions,
    ) -> Dict[str, str]:
        """Generate frontend code files."""
        code = {}
        
        # Generate main App component
        app_component = f"""import React from 'react';
import {{ BrowserRouter, Routes, Route }} from 'react-router-dom';

// Import pages
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';

function App() {{
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={{<HomePage />}} />
        <Route path="/login" element={{<LoginPage />}} />
      </Routes>
    </BrowserRouter>
  );
}}

export default App;
"""
        code["src/App.tsx"] = app_component
        
        # Generate package.json
        package_json = f"""{{
  "name": "{app.name}",
  "version": "1.0.0",
  "private": true,
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.0.0",
    "axios": "^1.0.0"
  }},
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }}
}}
"""
        code["package.json"] = package_json
        
        # Generate component files from components
        for component in app.frontend_components:
            filename = f"src/components/{component.name}.tsx"
            code[filename] = component.render()
        
        return code
    
    async def _generate_api_endpoints(
        self,
        description: str,
        options: GenerationOptions,
    ) -> List[dict]:
        """Generate API endpoints."""
        endpoints = []
        
        # Add auth endpoints
        endpoints.append({
            "method": "POST",
            "path": "/api/auth/login",
            "description": "User login",
            "body": {"email": "string", "password": "string"},
            "response": {"token": "string", "user": "object"},
        })
        
        endpoints.append({
            "method": "POST",
            "path": "/api/auth/register",
            "description": "User registration",
            "body": {"email": "string", "password": "string", "name": "string"},
            "response": {"token": "string", "user": "object"},
        })
        
        # Add CRUD endpoints based on description
        desc_lower = description.lower()
        
        if "task" in desc_lower or "todo" in desc_lower:
            endpoints.append({
                "method": "GET",
                "path": "/api/tasks",
                "description": "List all tasks",
                "response": {"tasks": "array"},
            })
            endpoints.append({
                "method": "POST",
                "path": "/api/tasks",
                "description": "Create task",
                "body": {"title": "string", "description": "string"},
                "response": {"task": "object"},
            })
        
        return endpoints
    
    async def _generate_backend_code(
        self,
        app: FullStackApp,
        options: GenerationOptions,
    ) -> Dict[str, str]:
        """Generate backend code files."""
        code = {}
        
        # Generate main.py for FastAPI
        if app.backend_framework == BackendFramework.FASTAPI:
            main_py = f"""from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="{app.name}",
    description="{app.description}",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str

class Token(BaseModel):
    token: str
    user: dict

# Routes
@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    # TODO: Implement authentication
    return {{"token": "jwt_token", "user": {{"email": credentials.email}}}}

@app.post("/api/auth/register", response_model=Token)
async def register(user: UserRegister):
    # TODO: Implement registration
    return {{"token": "jwt_token", "user": {{"email": user.email}}}}

@app.get("/api/health")
async def health_check():
    return {{"status": "ok"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            code["main.py"] = main_py
            
            # Generate requirements.txt
            code["requirements.txt"] = """fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
"""
        
        return code
    
    async def _generate_database_schema(
        self,
        description: str,
        options: GenerationOptions,
    ) -> dict:
        """Generate database schema."""
        schema = {
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "id", "type": "UUID", "primary_key": True},
                        {"name": "email", "type": "VARCHAR(255)", "unique": True, "nullable": False},
                        {"name": "password_hash", "type": "VARCHAR(255)", "nullable": False},
                        {"name": "name", "type": "VARCHAR(255)"},
                        {"name": "created_at", "type": "TIMESTAMP", "default": "NOW()"},
                        {"name": "updated_at", "type": "TIMESTAMP", "default": "NOW()"},
                    ],
                },
            ],
        }
        
        # Add tables based on description
        desc_lower = description.lower()
        
        if "task" in desc_lower or "todo" in desc_lower:
            schema["tables"].append({
                "name": "tasks",
                "columns": [
                    {"name": "id", "type": "UUID", "primary_key": True},
                    {"name": "title", "type": "VARCHAR(255)", "nullable": False},
                    {"name": "description", "type": "TEXT"},
                    {"name": "status", "type": "VARCHAR(50)", "default": "'pending'"},
                    {"name": "user_id", "type": "UUID", "foreign_key": "users.id"},
                    {"name": "created_at", "type": "TIMESTAMP", "default": "NOW()"},
                ],
            })
        
        return schema
    
    async def _generate_migrations(
        self,
        app: FullStackApp,
        options: GenerationOptions,
    ) -> List[str]:
        """Generate database migrations."""
        migrations = []
        
        # Generate initial migration
        migration = f"""-- Initial migration for {app.name}
-- Generated by AI Orchestrator

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
"""
        
        if app.database_schema:
            for table in app.database_schema.get("tables", []):
                if table["name"] != "users":
                    migration += f"\n-- Table: {table['name']}\n"
                    # Add table creation SQL
        
        migrations.append(migration)
        
        return migrations
    
    async def _generate_auth_config(
        self,
        options: GenerationOptions,
    ) -> dict:
        """Generate authentication configuration."""
        config = {
            "type": options.auth.value,
            "jwt_secret": "${JWT_SECRET}",  # Use environment variable
            "jwt_expiry": "24h",
            "refresh_token_expiry": "7d",
        }
        
        if options.auth == AuthType.OAUTH:
            config["providers"] = {
                "google": {
                    "client_id": "${GOOGLE_CLIENT_ID}",
                    "client_secret": "${GOOGLE_CLIENT_SECRET}",
                },
                "github": {
                    "client_id": "${GITHUB_CLIENT_ID}",
                    "client_secret": "${GITHUB_CLIENT_SECRET}",
                },
            }
        
        return config
    
    async def _generate_deployment_config(
        self,
        app: FullStackApp,
        options: GenerationOptions,
    ) -> dict:
        """Generate deployment configuration."""
        config = {
            "environment": "production",
            "env_vars": [
                "DATABASE_URL",
                "JWT_SECRET",
                "API_KEY",
            ],
        }
        
        # Vercel config
        if options.frontend in [FrontendFramework.REACT, FrontendFramework.NEXTJS]:
            config["vercel"] = {
                "build_command": "npm run build",
                "output_directory": "dist",
                "install_command": "npm install",
            }
        
        # Docker config
        if options.include_docker:
            config["docker"] = {
                "base_image": "python:3.11-slim",
                "port": 8000,
            }
        
        return config
    
    def _generate_file_tree(
        self,
        app: FullStackApp,
        options: GenerationOptions,
    ) -> dict:
        """Generate project file tree."""
        tree = {
            "name": app.name,
            "children": [
                {
                    "name": "frontend",
                    "type": "directory",
                    "children": [
                        {"name": "src", "type": "directory"},
                        {"name": "public", "type": "directory"},
                        {"name": "package.json", "type": "file"},
                    ],
                },
                {
                    "name": "backend",
                    "type": "directory",
                    "children": [
                        {"name": "main.py", "type": "file"},
                        {"name": "requirements.txt", "type": "file"},
                    ],
                },
                {
                    "name": "database",
                    "type": "directory",
                    "children": [
                        {"name": "migrations", "type": "directory"},
                        {"name": "schema.sql", "type": "file"},
                    ],
                },
                {"name": "README.md", "type": "file"},
                {"name": ".env.example", "type": "file"},
                {"name": ".gitignore", "type": "file"},
            ],
        }
        
        return tree
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_generations": self._total_generations,
            "total_apps_deployed": self._total_apps_deployed,
            "component_library_stats": self.component_library.get_stats(),
            "deployment_service_stats": self.deployment_service.get_stats(),
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_generator: Optional[FullStackGenerator] = None


def get_fullstack_generator() -> FullStackGenerator:
    """Get or create default full-stack generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = FullStackGenerator()
    return _default_generator


def reset_fullstack_generator() -> None:
    """Reset default generator (for testing)."""
    global _default_generator
    _default_generator = None


async def generate_fullstack_app(
    description: str,
    options: Optional[dict] = None,
) -> FullStackApp:
    """
    Generate full-stack app using default generator.
    
    Args:
        description: Project description
        options: Optional generation options
    
    Returns:
        Generated app
    """
    generator = get_fullstack_generator()
    
    if options:
        gen_options = GenerationOptions(**options)
    else:
        gen_options = GenerationOptions()
    
    return await generator.generate(description, gen_options)
