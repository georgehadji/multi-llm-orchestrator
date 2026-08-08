"""
Docker Generator — Builder + Template Method Pattern
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Docker configuration generation using Builder Pattern for Dockerfile construction
and Template Method for different language runtimes.

Paradigm: OOP with Functional utilities
Patterns: Builder, Template Method, Factory Method, Strategy

Usage:
    from orchestrator.generators.docker_generator import DockerfileBuilder

    dockerfile = (DockerfileBuilder()
        .for_node("18-alpine")
        .add_workdir("/app")
        .copy_package_json()
        .run("npm ci")
        .copy_all()
        .run("npm run build")
        .expose(3000)
        .cmd("npm start")
        .build())
"""

from __future__ import annotations

import json
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


class LanguageRuntime(str, Enum):
    """Language runtime enumeration."""

    NODE = "node"
    PYTHON = "python"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    RUBY = "ruby"
    PHP = "php"
    DOTNET = "dotnet"


class BuildStage(str, Enum):
    """Build stage enumeration."""

    BASE = "base"
    BUILD = "build"
    PRODUCTION = "production"


@dataclass(frozen=True)
class DockerConfig:
    """
    Immutable Docker configuration.

    Attributes:
        runtime: Language runtime
        base_image: Base Docker image
        workdir: Working directory
        ports: Exposed ports
        env_vars: Environment variables
        volumes: Volume mounts
        healthcheck: Health check configuration
        multi_stage: Use multi-stage build
    """

    runtime: LanguageRuntime
    base_image: str = ""
    workdir: str = "/app"
    ports: List[int] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    healthcheck: Optional[Dict[str, Any]] = None
    multi_stage: bool = True


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE METHOD PATTERN — RUNTIME TEMPLATES
# ═══════════════════════════════════════════════════════════════════


class RuntimeTemplate(ABC):
    """
    Template Method for runtime-specific Dockerfile generation.

    Subclasses implement specific language runtime templates.
    """

    @abstractmethod
    def get_base_image(self, version: str = "latest") -> str:
        """Get base image for runtime."""
        pass

    @abstractmethod
    def get_install_command(self) -> str:
        """Get dependency installation command."""
        pass

    @abstractmethod
    def get_build_command(self) -> str:
        """Get build command."""
        pass

    @abstractmethod
    def get_start_command(self) -> str:
        """Get start command."""
        pass

    @abstractmethod
    def get_healthcheck(self) -> Optional[Dict[str, Any]]:
        """Get health check configuration."""
        pass


class NodeRuntimeTemplate(RuntimeTemplate):
    """Node.js runtime template."""

    def get_base_image(self, version: str = "latest") -> str:
        """Get Node.js base image."""
        if version == "latest":
            return "node:20-alpine"
        return f"node:{version}-alpine"

    def get_install_command(self) -> str:
        """Get npm install command."""
        return "npm ci --only=production"

    def get_build_command(self) -> str:
        """Get npm build command."""
        return "npm run build"

    def get_start_command(self) -> str:
        """Get npm start command."""
        return "npm start"

    def get_healthcheck(self) -> Optional[Dict[str, Any]]:
        """Get Node.js health check."""
        return {
            "cmd": "wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1",
            "interval": "30s",
            "timeout": "3s",
            "retries": 3,
        }


class PythonRuntimeTemplate(RuntimeTemplate):
    """Python runtime template."""

    def get_base_image(self, version: str = "latest") -> str:
        """Get Python base image."""
        if version == "latest":
            return "python:3.12-slim"
        return f"python:{version}-slim"

    def get_install_command(self) -> str:
        """Get pip install command."""
        return "pip install --no-cache-dir -r requirements.txt"

    def get_build_command(self) -> str:
        """Get Python build command."""
        return "python setup.py build"

    def get_start_command(self) -> str:
        """Get Python start command."""
        return "python -m gunicorn app:app"

    def get_healthcheck(self) -> Optional[Dict[str, Any]]:
        """Get Python health check."""
        return {
            "cmd": "curl -f http://localhost:8000/health || exit 1",
            "interval": "30s",
            "timeout": "3s",
            "retries": 3,
        }


class GoRuntimeTemplate(RuntimeTemplate):
    """Go runtime template."""

    def get_base_image(self, version: str = "latest") -> str:
        """Get Go base image."""
        if version == "latest":
            return "golang:1.21-alpine"
        return f"golang:{version}-alpine"

    def get_install_command(self) -> str:
        """Get Go get command."""
        return "go mod download"

    def get_build_command(self) -> str:
        """Get Go build command."""
        return "go build -o /app/main"

    def get_start_command(self) -> str:
        """Get Go start command."""
        return "/app/main"

    def get_healthcheck(self) -> Optional[Dict[str, Any]]:
        """Get Go health check."""
        return {
            "cmd": "wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1",
            "interval": "30s",
            "timeout": "3s",
            "retries": 3,
        }


class RustRuntimeTemplate(RuntimeTemplate):
    """Rust runtime template."""

    def get_base_image(self, version: str = "latest") -> str:
        """Get Rust base image."""
        if version == "latest":
            return "rust:1.75-slim"
        return f"rust:{version}-slim"

    def get_install_command(self) -> str:
        """Get Cargo build command."""
        return "cargo build --release"

    def get_build_command(self) -> str:
        """Get Rust build command."""
        return "cargo build --release"

    def get_start_command(self) -> str:
        """Get Rust start command."""
        return "/app/target/release/app"

    def get_healthcheck(self) -> Optional[Dict[str, Any]]:
        """Get Rust health check."""
        return {
            "cmd": "curl -f http://localhost:8080/health || exit 1",
            "interval": "30s",
            "timeout": "3s",
            "retries": 3,
        }


# ═══════════════════════════════════════════════════════════════════
# BUILDER PATTERN — DOCKERFILE BUILDER
# ═══════════════════════════════════════════════════════════════════


class DockerfileBuilder:
    """
    Builder Pattern for Dockerfile generation.

    Fluent interface for constructing Dockerfiles.

    Usage:
        dockerfile = (DockerfileBuilder()
            .for_node("18-alpine")
            .add_workdir("/app")
            .copy_package_json()
            .run("npm ci")
            .build())
    """

    def __init__(self):
        """Initialize Dockerfile builder."""
        self._lines: List[str] = []
        self._runtime: Optional[LanguageRuntime] = None
        self._runtime_template: Optional[RuntimeTemplate] = None
        self._stages: List[str] = []
        self._current_stage: str = ""

    def _add_line(self, line: str) -> "DockerfileBuilder":
        """Add line to Dockerfile."""
        self._lines.append(line)
        return self

    def _add_comment(self, comment: str) -> "DockerfileBuilder":
        """Add comment line."""
        self._lines.append(f"# {comment}")
        return self

    def _add_blank(self) -> "DockerfileBuilder":
        """Add blank line."""
        self._lines.append("")
        return self

    # ═══════════════════════════════════════════════════════════════
    # RUNTIME SELECTION
    # ═══════════════════════════════════════════════════════════════

    def for_node(self, version: str = "latest") -> "DockerfileBuilder":
        """Configure for Node.js."""
        self._runtime = LanguageRuntime.NODE
        self._runtime_template = NodeRuntimeTemplate()
        self._add_line(f"FROM {self._runtime_template.get_base_image(version)}")
        return self

    def for_python(self, version: str = "latest") -> "DockerfileBuilder":
        """Configure for Python."""
        self._runtime = LanguageRuntime.PYTHON
        self._runtime_template = PythonRuntimeTemplate()
        self._add_line(f"FROM {self._runtime_template.get_base_image(version)}")
        return self

    def for_go(self, version: str = "latest") -> "DockerfileBuilder":
        """Configure for Go."""
        self._runtime = LanguageRuntime.GO
        self._runtime_template = GoRuntimeTemplate()
        self._add_line(f"FROM {self._runtime_template.get_base_image(version)}")
        return self

    def for_rust(self, version: str = "latest") -> "DockerfileBuilder":
        """Configure for Rust."""
        self._runtime = LanguageRuntime.RUST
        self._runtime_template = RustRuntimeTemplate()
        self._add_line(f"FROM {self._runtime_template.get_base_image(version)}")
        return self

    # ═══════════════════════════════════════════════════════════════
    # MULTI-STAGE BUILD
    # ═══════════════════════════════════════════════════════════════

    def add_stage(self, stage_name: str, base_image: str = "") -> "DockerfileBuilder":
        """Add multi-stage build stage."""
        self._current_stage = stage_name
        if base_image:
            self._add_line(f"FROM {base_image} AS {stage_name}")
        else:
            self._add_line(f"FROM {self._runtime_template.get_base_image()} AS {stage_name}")
        self._stages.append(stage_name)
        return self

    def add_build_stage(self) -> "DockerfileBuilder":
        """Add build stage."""
        return self.add_stage("build")

    def add_production_stage(self) -> "DockerfileBuilder":
        """Add production stage."""
        if self._runtime == LanguageRuntime.NODE:
            return self.add_stage("production", "node:20-alpine")
        elif self._runtime == LanguageRuntime.PYTHON:
            return self.add_stage("production", "python:3.12-slim")
        else:
            return self.add_stage("production")

    # ═══════════════════════════════════════════════════════════════
    # BASIC INSTRUCTIONS
    # ═══════════════════════════════════════════════════════════════

    def add_workdir(self, path: str) -> "DockerfileBuilder":
        """Set working directory."""
        self._add_line(f"WORKDIR {path}")
        return self

    def add_env(self, key: str, value: str) -> "DockerfileBuilder":
        """Set environment variable."""
        self._add_line(f"ENV {key}={value}")
        return self

    def add_env_vars(self, env_vars: Dict[str, str]) -> "DockerfileBuilder":
        """Set multiple environment variables."""
        for key, value in env_vars.items():
            self.add_env(key, value)
        return self

    def copy(self, src: str, dest: str = ".") -> "DockerfileBuilder":
        """Copy files."""
        self._add_line(f"COPY {src} {dest}")
        return self

    def copy_all(self) -> "DockerfileBuilder":
        """Copy all files."""
        return self.copy(".", ".")

    def add_volume(self, path: str) -> "DockerfileBuilder":
        """Add volume mount."""
        self._add_line(f"VOLUME {path}")
        return self

    def expose(self, port: int) -> "DockerfileBuilder":
        """Expose port."""
        self._add_line(f"EXPOSE {port}")
        return self

    def expose_ports(self, ports: List[int]) -> "DockerfileBuilder":
        """Expose multiple ports."""
        for port in ports:
            self.expose(port)
        return self

    def run(self, command: str) -> "DockerfileBuilder":
        """Run command."""
        self._add_line(f"RUN {command}")
        return self

    def run_multi(self, commands: List[str]) -> "DockerfileBuilder":
        """Run multiple commands."""
        for cmd in commands:
            self.run(cmd)
        return self

    def add_user(self, user: str) -> "DockerfileBuilder":
        """Add user."""
        self._add_line(f"USER {user}")
        return self

    def cmd(self, command: str) -> "DockerfileBuilder":
        """Set CMD (exec form — a shell command string is split into argv,
        never emitted as a single unresolvable JSON array element)."""
        self._add_line(f"CMD {json.dumps(shlex.split(command))}")
        return self

    def entrypoint(self, command: str) -> "DockerfileBuilder":
        """Set ENTRYPOINT (exec form, see ``cmd``)."""
        self._add_line(f"ENTRYPOINT {json.dumps(shlex.split(command))}")
        return self

    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE-SPECIFIC HELPERS
    # ═══════════════════════════════════════════════════════════════

    def copy_package_json(self) -> "DockerfileBuilder":
        """Copy package.json (Node.js)."""
        return self.copy("package.json package-lock.json*", ".")

    def copy_requirements(self) -> "DockerfileBuilder":
        """Copy requirements.txt (Python)."""
        return self.copy("requirements.txt", ".")

    def copy_go_mod(self) -> "DockerfileBuilder":
        """Copy go.mod (Go)."""
        return self.copy("go.mod go.sum*", ".")

    def copy_cargo(self) -> "DockerfileBuilder":
        """Copy Cargo.toml (Rust)."""
        return self.copy("Cargo.toml Cargo.lock*", ".")

    # ═══════════════════════════════════════════════════════════════
    # HEALTH CHECK
    # ═══════════════════════════════════════════════════════════════

    def add_healthcheck(self) -> "DockerfileBuilder":
        """Add health check."""
        if self._runtime_template:
            healthcheck = self._runtime_template.get_healthcheck()
            if healthcheck:
                cmd = healthcheck.get("cmd", "")
                interval = healthcheck.get("interval", "30s")
                timeout = healthcheck.get("timeout", "3s")
                retries = healthcheck.get("retries", 3)

                self._add_line(
                    f"HEALTHCHECK --interval={interval} --timeout={timeout} --retries={retries}"
                )
                self._add_line(f"    CMD {cmd}")

        return self

    # ═══════════════════════════════════════════════════════════════
    # PRESET CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════

    def add_nonroot_user(self, user: str = "appuser", uid: int = 10001) -> "DockerfileBuilder":
        """Create and switch to a non-root user (container hardening best practice).

        Running containers as root means a container escape grants host root.
        This creates an unprivileged user and switches to it before CMD.
        """
        self._add_line(
            f"RUN (addgroup --system {user} 2>/dev/null || true) && "
            f"(adduser --system --uid {uid} --ingroup {user} {user} 2>/dev/null || "
            f"adduser -S -u {uid} {user} 2>/dev/null || true)"
        )
        return self.add_user(user)

    def for_node_app(self, version: str = "20-slim") -> "DockerfileBuilder":
        """Configure for Node.js app with best practices (pinned image, non-root)."""
        return (
            self.for_node(version)
            .add_env("NODE_ENV", "production")
            .add_workdir("/app")
            .copy_package_json()
            .run("npm ci --only=production")
            .copy_all()
            .run("npm run build")
            .expose(3000)
            .add_healthcheck()
            .add_nonroot_user()
            .cmd("npm start")
        )

    def for_python_app(self, version: str = "3.12-slim") -> "DockerfileBuilder":
        """Configure for Python app with best practices (pinned image, non-root)."""
        return (
            self.for_python(version)
            .add_env("PYTHONUNBUFFERED", "1")
            .add_workdir("/app")
            .copy_requirements()
            .run("pip install --no-cache-dir -r requirements.txt")
            .copy_all()
            .expose(8000)
            .add_healthcheck()
            .add_nonroot_user()
            .cmd("python -m gunicorn app:app")
        )

    def for_go_app(self, version: str = "1.23-alpine") -> "DockerfileBuilder":
        """Configure for Go app with best practices (pinned image, non-root)."""
        return (
            self.for_go(version)
            .add_workdir("/app")
            .copy_go_mod()
            .run("go mod download")
            .copy_all()
            .run("go build -o /app/main")
            .expose(8080)
            .add_healthcheck()
            .add_nonroot_user()
            .cmd("/app/main")
        )

    def for_rust_app(self, version: str = "1.82-slim") -> "DockerfileBuilder":
        """Configure for Rust app with best practices (pinned image, non-root)."""
        return (
            self.for_rust(version)
            .add_workdir("/app")
            .copy_cargo()
            .run("cargo build --release")
            .copy_all()
            .run("cargo build --release")
            .expose(8080)
            .add_healthcheck()
            .add_nonroot_user()
            .cmd("/app/target/release/app")
        )

    # ═══════════════════════════════════════════════════════════════
    # BUILD
    # ═══════════════════════════════════════════════════════════════

    def build(self) -> str:
        """
        Build Dockerfile.

        Returns:
            Dockerfile content
        """
        self._add_comment("Generated by AI Orchestrator")
        self._add_blank()

        return "\n".join(self._lines)

    def build_to_file(self, filepath: str = "Dockerfile") -> str:
        """
        Build and write to file.

        Args:
            filepath: Output file path

        Returns:
            File path
        """
        content = self.build()

        with open(filepath, "w") as f:
            f.write(content)

        return filepath


# ═══════════════════════════════════════════════════════════════════
# DOCKER COMPOSE BUILDER
# ═══════════════════════════════════════════════════════════════════


class DockerComposeBuilder:
    """
    Builder for Docker Compose files.
    """

    def __init__(self):
        """Initialize Compose builder."""
        self._version: str = "3.8"
        self._services: Dict[str, Dict[str, Any]] = {}

    def add_service(
        self,
        name: str,
        build: str = ".",
        ports: List[int] = None,
        environment: Dict[str, str] = None,
        volumes: List[str] = None,
        depends_on: List[str] = None,
    ) -> "DockerComposeBuilder":
        """Add service."""
        service = {
            "build": build,
        }

        if ports:
            service["ports"] = [f"{p}:{p}" for p in ports]

        if environment:
            service["environment"] = environment

        if volumes:
            service["volumes"] = volumes

        if depends_on:
            service["depends_on"] = depends_on

        self._services[name] = service
        return self

    def add_database(
        self,
        db_type: str = "postgres",
        name: str = "db",
    ) -> "DockerComposeBuilder":
        """Add database service.

        Credentials are read from environment variables (compose ${VAR} interpolation),
        never hardcoded. Define them in a local .env file that is git-ignored.
        """
        if db_type == "postgres":
            service = {
                "image": "postgres:16-alpine",
                "environment": {
                    "POSTGRES_USER": "${POSTGRES_USER:?set POSTGRES_USER in .env}",
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD in .env}",
                    "POSTGRES_DB": "${POSTGRES_DB:-app}",
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
            }
        elif db_type == "mysql":
            service = {
                "image": "mysql:8",
                "environment": {
                    "MYSQL_ROOT_PASSWORD": "${MYSQL_ROOT_PASSWORD:?set MYSQL_ROOT_PASSWORD in .env}",
                    "MYSQL_DATABASE": "${MYSQL_DATABASE:-app}",
                },
                "volumes": ["mysql_data:/var/lib/mysql"],
            }
        elif db_type == "mongo":
            service = {
                "image": "mongo:7",
                "volumes": ["mongo_data:/data/db"],
            }
        else:
            service = {"image": db_type}

        self._services[name] = service
        return self

    def add_redis(self, name: str = "redis") -> "DockerComposeBuilder":
        """Add Redis service."""
        self._services[name] = {
            "image": "redis:7-alpine",
            "volumes": ["redis_data:/data"],
        }
        return self

    def build(self) -> str:
        """Build Docker Compose file."""
        compose = f"""version: "{self._version}"
# Generated by AI Orchestrator

services:
"""

        for name, service in self._services.items():
            compose += f"  {name}:\n"
            for key, value in service.items():
                if isinstance(value, dict):
                    compose += f"    {key}:\n"
                    for k, v in value.items():
                        compose += f"      {k}: {v}\n"
                elif isinstance(value, list):
                    compose += f"    {key}:\n"
                    for item in value:
                        compose += f"      - {item}\n"
                else:
                    compose += f"    {key}: {value}\n"

        return compose

    def build_to_file(self, filepath: str = "docker-compose.yml") -> str:
        """Build and write to file."""
        content = self.build()

        with open(filepath, "w") as f:
            f.write(content)

        return filepath


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def create_dockerfile(
    runtime: str = "node",
    version: str = "latest",
) -> str:
    """
    Create Dockerfile for runtime.

    Args:
        runtime: Language runtime
        version: Runtime version

    Returns:
        Dockerfile content
    """
    builder = DockerfileBuilder()

    if runtime == "node":
        return builder.for_node_app(version).build()
    elif runtime == "python":
        return builder.for_python_app(version).build()
    elif runtime == "go":
        return builder.for_go_app(version).build()
    elif runtime == "rust":
        return builder.for_rust_app(version).build()
    else:
        return builder.build()


def create_docker_compose(
    services: List[str] = None,
    database: str = None,
    redis: bool = False,
) -> str:
    """
    Create Docker Compose file.

    Args:
        services: List of services
        database: Database type
        redis: Include Redis

    Returns:
        Docker Compose content
    """
    builder = DockerComposeBuilder()

    if services:
        for service in services:
            builder.add_service(service, build=f"./{service}")

    if database:
        builder.add_database(database)

    if redis:
        builder.add_redis()

    return builder.build()
