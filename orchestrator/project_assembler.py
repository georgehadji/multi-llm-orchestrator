"""
Project Assembler (Improvement 10)
==================================
Transforms fragmented task files into a complete, production-ready Python project.

Features:
- AST-based dependency detection from CODE_GEN task outputs
- Generates main.py: CLI entry point with --list, --step, --dry-run
- Generates config.py: Project configuration with Pydantic Settings
- Generates pyproject.toml: Modern Python packaging (PEP 518/621)
- Generates Makefile: install, run, test, lint targets
- Generates Dockerfile: Multi-stage build with non-root user
- Generates .github/workflows/ci.yml: CI/CD pipeline
- Generates .env.example: Configuration template
- Generates src/ structure with domain/application/infrastructure layers
- Generates tests/ with pytest configuration
- Generates docs/ with architecture overview

Author: Georgios-Chrysovalantis Chatzivantsidis
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import ProjectState, TaskType

logger = logging.getLogger("orchestrator.project_assembler")


@dataclass
class ModuleInfo:
    """Information about a Python module extracted from task output."""
    name: str
    content: str
    task_id: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)  # classes, functions defined


class DependencyAnalyzer:
    """Analyze Python code to extract imports and defined symbols."""
    
    @staticmethod
    def extract_imports(source: str) -> list[str]:
        """Extract top-level import statements from Python source."""
        imports = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback: regex-based extraction for malformed code
            return DependencyAnalyzer._extract_imports_regex(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
                # Also capture specific imports from module
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return sorted(set(imports))
    
    @staticmethod
    def _extract_imports_regex(source: str) -> list[str]:
        """Fallback regex-based import extraction."""
        imports = []
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)'
        from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'
        
        for line in source.split('\n'):
            line = line.strip()
            if match := re.match(import_pattern, line):
                imports.extend(name.strip() for name in match.group(1).split(','))
            elif match := re.match(from_pattern, line):
                imports.append(match.group(1))
        
        return sorted(set(imports))
    
    @staticmethod
    def extract_exports(source: str) -> list[str]:
        """Extract defined classes and functions from Python source."""
        exports = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return DependencyAnalyzer._extract_exports_regex(source)
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                exports.append(f"class:{node.name}")
            elif isinstance(node, ast.FunctionDef):
                exports.append(f"func:{node.name}")
            elif isinstance(node, ast.AsyncFunctionDef):
                exports.append(f"async_func:{node.name}")
        
        return exports
    
    @staticmethod
    def _extract_exports_regex(source: str) -> list[str]:
        """Fallback regex-based export extraction."""
        exports = []
        for match in re.finditer(r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)', source, re.MULTILINE):
            exports.append(f"class:{match.group(1)}")
        for match in re.finditer(r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)', source, re.MULTILINE):
            exports.append(f"func:{match.group(1)}")
        return exports


class ProjectAssembler:
    """
    Assemble a production-ready Python project from fragmented task outputs.
    
    Analyzes CODE_GEN task outputs, detects dependencies between modules,
    and generates a complete project structure following modern best practices.
    """
    
    def __init__(self, output_dir: Path, state: ProjectState):
        self.output_dir = Path(output_dir)
        self.state = state
        self.analyzer = DependencyAnalyzer()
        self.modules: list[ModuleInfo] = []
        self.project_name = self._sanitize_project_name()
    
    def assemble(self) -> list[str]:
        """
        Main entry point. Analyzes tasks and generates project files.
        
        Returns:
            List of created file paths (relative to output_dir)
        """
        # Extract modules from CODE_GEN tasks
        self.modules = self._extract_modules()
        
        if not self.modules:
            logger.info("No CODE_GEN modules found, skipping project assembly")
            return []
        
        # Analyze dependencies
        self._analyze_dependencies()
        
        # Generate project structure
        created_files = []
        created_files.extend(self._generate_directory_structure())
        created_files.extend(self._generate_module_files())
        created_files.extend(self._generate_config_layer())
        created_files.extend(self._generate_main_file())
        created_files.extend(self._generate_pyproject_toml())
        created_files.extend(self._generate_makefile())
        created_files.extend(self._generate_dockerfile())
        created_files.extend(self._generate_github_actions())
        created_files.extend(self._generate_env_example())
        created_files.extend(self._generate_test_structure())
        created_files.extend(self._generate_docs_structure())
        created_files.extend(self._generate_precommit_config())
        created_files.extend(self._generate_exception_hierarchy())
        created_files.extend(self._generate_logging_config())
        
        return created_files
    
    def _extract_modules(self) -> list[ModuleInfo]:
        """Extract Python modules from CODE_GEN task outputs."""
        modules = []
        order = self.state.execution_order or list(self.state.results.keys())
        
        for task_id in order:
            result = self.state.results.get(task_id)
            task = self.state.tasks.get(task_id)
            
            if not result or not task:
                continue
            if task.type != TaskType.CODE_GEN:
                continue
            if not result.output:
                continue
            
            code = self._extract_code(result.output)
            if not code or len(code.strip()) < 50:
                continue
            
            module_name = self._sanitize_module_name(task_id)
            module = ModuleInfo(
                name=module_name,
                content=code,
                task_id=task_id
            )
            modules.append(module)
        
        return modules
    
    def _extract_code(self, output: str) -> str:
        """Extract Python code from task output."""
        if match := re.search(r'```python\n(.*?)```', output, re.DOTALL):
            return match.group(1)
        if match := re.search(r'```\n(.*?)```', output, re.DOTALL):
            return match.group(1)
        return output
    
    def _sanitize_module_name(self, task_id: str) -> str:
        """Convert task_id to valid Python module name."""
        name = re.sub(r'^task_', '', task_id)
        name = re.sub(r'^_0+', '_', name)
        name = re.sub(r'^_', '', name)
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        if name and name[0].isdigit():
            name = 'task_' + name
        if not name or not name[0].isalpha():
            name = 'module_' + name
        
        return name.lower()
    
    def _sanitize_project_name(self) -> str:
        """Generate a valid Python package name from project description."""
        desc = self.state.project_description[:30]
        name = re.sub(r'[^a-zA-Z0-9_]', '_', desc)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_').lower()
        if not name or name[0].isdigit():
            name = 'generated_project'
        return name
    
    def _analyze_dependencies(self) -> None:
        """Analyze imports and exports for all modules."""
        for module in self.modules:
            module.imports = self.analyzer.extract_imports(module.content)
            module.exports = self.analyzer.extract_exports(module.content)
            logger.debug(
                "Module %s: %d imports, %d exports",
                module.name, len(module.imports), len(module.exports)
            )
    
    def _generate_directory_structure(self) -> list[str]:
        """Create clean directory structure following best practices."""
        dirs = [
            self.output_dir / "src" / self.project_name / "domain",
            self.output_dir / "src" / self.project_name / "application",
            self.output_dir / "src" / self.project_name / "infrastructure",
            self.output_dir / "tests" / "unit",
            self.output_dir / "tests" / "integration",
            self.output_dir / "tests" / "fixtures",
            self.output_dir / "docs" / "adr",
            self.output_dir / "scripts",
            self.output_dir / ".github" / "workflows",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created directory structure with domain/application/infrastructure layers")
        return []
    
    def _generate_module_files(self) -> list[str]:
        """Write extracted modules to src/{project_name}/ directory."""
        created = []
        base_dir = self.output_dir / "src" / self.project_name
        
        # Create __init__.py files
        for init_path in [
            self.output_dir / "src" / "__init__.py",
            base_dir / "__init__.py",
            base_dir / "domain" / "__init__.py",
            base_dir / "application" / "__init__.py",
            base_dir / "infrastructure" / "__init__.py",
        ]:
            init_path.write_text('"""Auto-generated module."""\n', encoding="utf-8")
            created.append(str(init_path.relative_to(self.output_dir)))
        
        # Distribute modules across layers based on content analysis
        for module in self.modules:
            # Simple heuristic: place in application layer by default
            target_dir = base_dir / "application"
            
            # Check if it looks like infrastructure (DB, HTTP, external services)
            if any(x in ' '.join(module.imports) for x in ['requests', 'http', 'boto', 'sql', 'redis']):
                target_dir = base_dir / "infrastructure"
            # Check if it looks like domain (pure business logic)
            elif not any(x in ' '.join(module.imports) for x in ['os', 'sys', 'requests', 'http']) and module.exports:
                target_dir = base_dir / "domain"
            
            file_path = target_dir / f"{module.name}.py"
            file_path.write_text(module.content, encoding="utf-8")
            created.append(str(file_path.relative_to(self.output_dir)))
        
        logger.info("Generated %d module files across layers", len(self.modules))
        return created
    
    def _generate_config_layer(self) -> list[str]:
        """Generate configuration layer with Pydantic Settings."""
        created = []
        base_dir = self.output_dir / "src" / self.project_name
        
        # __init__.py with version
        init_content = f'''"""
{self.project_name.replace('_', ' ').title()}
{'=' * len(self.project_name)}
{self.state.project_description}
"""

__version__ = "0.1.0"
__author__ = "Generated by Multi-LLM Orchestrator"

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
'''
        init_file = base_dir / "__init__.py"
        init_file.write_text(init_content, encoding="utf-8")
        created.append(f"src/{self.project_name}/__init__.py")
        
        # config.py with Pydantic Settings
        config_content = f'''"""
Configuration Layer
===================
Typed settings using Pydantic Settings.
All configuration is loaded from environment variables.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback for when pydantic is not installed
    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs):
            pass
    def Field(*args, **kwargs):  # type: ignore
        return None


# Project paths
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Usage:
        settings = get_settings()
        debug = settings.debug
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "{self.project_name}"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json, text")
    
    # Performance
    max_workers: int = Field(default=4, description="Maximum worker threads")
    timeout_seconds: int = Field(default=30, description="Default timeout for operations")
    
    # Feature flags
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    
    # Security (NEVER hardcode secrets, always use env vars)
    secret_key: Optional[str] = Field(default=None, description="Secret key for encryption")
    api_key: Optional[str] = Field(default=None, description="API key for external services")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to avoid reloading settings on every call.
    """
    if not HAS_PYDANTIC:
        logging.warning("Pydantic not installed, using default settings")
        return Settings()
    return Settings()


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure structured logging for the application."""
    if settings is None:
        settings = get_settings()
    
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    if settings.log_format == "json":
        # JSON format for production
        format_string = '{{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}}'
    else:
        # Human-readable format for development
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "app.log"),
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
'''
        
        config_file = base_dir / "config.py"
        config_file.write_text(config_content, encoding="utf-8")
        created.append(f"src/{self.project_name}/config.py")
        
        logger.info("Generated configuration layer with Pydantic Settings")
        return created
    
    def _generate_main_file(self) -> list[str]:
        """Generate main.py CLI entry point with proper error handling."""
        main_content = f'''#!/usr/bin/env python3
"""
Main Entry Point
================
Production-ready CLI entry point with proper error handling,
structured logging, and observability.

Usage:
    python main.py              # Run all pipeline steps
    python main.py --list       # Show all available steps
    python main.py --step 1     # Run specific step only
    python main.py --dry-run    # Preview execution plan
    python main.py --health     # Health check
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from {self.project_name}.config import configure_logging, get_settings
from {self.project_name}.exceptions import ApplicationError, ConfigurationError

logger = logging.getLogger(__name__)


def get_correlation_id() -> str:
    """Generate or retrieve correlation ID for request tracing."""
    # In a real app, this would come from request headers or context
    return str(uuid.uuid4())[:8]


def get_available_steps() -> list[tuple[int, str, str, str]]:
    """Return list of (step_num, task_id, module_name, layer) tuples."""
    steps = []
    base_dir = Path(__file__).parent / "src" / "{self.project_name}"
    
    layer_order = ["domain", "application", "infrastructure"]
    step_num = 0
    
    for layer in layer_order:
        layer_dir = base_dir / layer
        if not layer_dir.exists():
            continue
        for py_file in sorted(layer_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            step_num += 1
            module_name = py_file.stem
            task_id = f"task_{{step_num:03d}}"
            steps.append((step_num, task_id, module_name, layer))
    
    return steps


def health_check() -> dict[str, any]:
    """Perform health check and return status."""
    import datetime
    
    status = {{
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "checks": {{}}
    }}
    
    # Check configuration
    try:
        settings = get_settings()
        status["checks"]["configuration"] = "pass"
    except Exception as e:
        status["checks"]["configuration"] = f"fail: {{e}}"
        status["status"] = "unhealthy"
    
    # Check disk space
    try:
        import shutil
        stat = shutil.disk_stat(".")
        free_gb = stat.free / (1024**3)
        status["checks"]["disk_space"] = f"{{free_gb:.1f}}GB free"
        if free_gb < 1:
            status["checks"]["disk_space"] += " (low)"
            status["status"] = "degraded"
    except Exception as e:
        status["checks"]["disk_space"] = f"error: {{e}}"
    
    return status


def run_step(step_num: int, dry_run: bool = False, correlation_id: Optional[str] = None) -> bool:
    """Run a specific pipeline step with proper error handling."""
    cid = correlation_id or get_correlation_id()
    steps = get_available_steps()
    
    if step_num < 1 or step_num > len(steps):
        logger.error(f"[{{cid}}] Step {{step_num}} not found (1-{{len(steps)}} available)")
        return False
    
    _, task_id, module_name, layer = steps[step_num - 1]
    module_path = f"{self.project_name}.{{layer}}.{{module_name}}"
    
    if dry_run:
        logger.info(f"[{{cid}}] [DRY-RUN] Would execute step {{step_num}}: {{module_name}} ({{layer}})")
        return True
    
    logger.info(f"[{{cid}}] Starting step {{step_num}}: {{module_name}} ({{layer}})")
    start_time = time.time()
    
    try:
        module = importlib.import_module(module_path)
        logger.debug(f"[{{cid}}] Module {{module_name}} loaded successfully")
        
        executed = False
        result = None
        
        # Try to find and execute main functions
        if hasattr(module, 'main'):
            func = module.main
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func())
            else:
                result = func()
            logger.info(f"[{{cid}}] main() completed in {{time.time() - start_time:.2f}}s, result: {{result}}")
            executed = True
        elif hasattr(module, 'run'):
            func = module.run
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func())
            else:
                result = func()
            logger.info(f"[{{cid}}] run() completed in {{time.time() - start_time:.2f}}s, result: {{result}}")
            executed = True
        
        if not executed:
            logger.warning(f"[{{cid}}] No main() or run() found in {{module_name}}")
            available = [x for x in dir(module) if not x.startswith('_')]
            logger.debug(f"[{{cid}}] Available: {{available}}")
        
        return True
        
    except ApplicationError as e:
        logger.error(f"[{{cid}}] Application error in {{module_name}}: {{e}}")
        return False
    except Exception as e:
        logger.exception(f"[{{cid}}] Unexpected error in {{module_name}}: {{e}}")
        return False


def run_all(dry_run: bool = False) -> bool:
    """Run all pipeline steps with observability."""
    cid = get_correlation_id()
    settings = get_settings()
    
    logger.info(f"[{{cid}}] Starting pipeline: {{settings.app_name}}")
    logger.info(f"[{{cid}}] Environment: {{settings.environment}}")
    
    steps = get_available_steps()
    logger.info(f"[{{cid}}] Found {{len(steps)}} steps to execute")
    
    if dry_run:
        logger.info(f"[{{cid}}] DRY-RUN MODE: Preview only")
    
    success_count = 0
    failed_steps = []
    
    for step_num, task_id, module_name, layer in steps:
        if run_step(step_num, dry_run=dry_run, correlation_id=cid):
            success_count += 1
        else:
            failed_steps.append(module_name)
            if not dry_run:
                logger.error(f"[{{cid}}] Pipeline stopped at step {{step_num}}")
                break
    
    # Summary
    total = len(steps)
    logger.info(f"[{{cid}}] Pipeline complete: {{success_count}}/{{total}} steps succeeded")
    
    if failed_steps:
        logger.warning(f"[{{cid}}] Failed steps: {{failed_steps}}")
    
    return success_count == total


def list_steps() -> None:
    """Display all available pipeline steps."""
    steps = get_available_steps()
    settings = get_settings()
    
    print(f"\\n{{'='*60}}")
    print(f"Available Steps: {{settings.app_name}}")
    print(f"{{'='*60}}\\n")
    
    for step_num, task_id, module_name, layer in steps:
        print(f"  {{step_num}}. {{module_name:20s}} ({{layer:15s}}) - {{task_id}}")
    
    print(f"\\nTotal: {{len(steps)}} step(s)")
    print(f"Run with: python main.py [--step N] [--dry-run]")


def main() -> int:
    """Main entry point with proper CLI handling."""
    settings = get_settings()
    configure_logging(settings)
    
    parser = argparse.ArgumentParser(
        description=settings.app_name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run all steps
  python main.py --list       # Show available steps  
  python main.py --step 1     # Run only step 1
  python main.py --dry-run    # Preview without executing
  python main.py --health     # Health check
        """
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all pipeline steps"
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        metavar="N",
        help="Run only step N"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview execution without running"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run health check"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {{settings.app_version}}"
    )
    
    args = parser.parse_args()
    
    try:
        if args.health:
            import json
            status = health_check()
            print(json.dumps(status, indent=2))
            return 0 if status["status"] == "healthy" else 1
        
        if args.list:
            list_steps()
            return 0
        
        if args.step:
            success = run_step(args.step, dry_run=args.dry_run)
        else:
            success = run_all(dry_run=args.dry_run)
        
        return 0 if success else 1
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {{e}}")
        return 2
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {{e}}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
        
        main_file = self.output_dir / "main.py"
        main_file.write_text(main_content, encoding="utf-8")
        logger.info("Generated main.py with proper error handling and observability")
        return ["main.py"]
    
    def _generate_pyproject_toml(self) -> list[str]:
        """Generate comprehensive pyproject.toml with all tools configured."""
        dependencies = self._extract_dependencies()
        
        toml_content = f"""[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[project]
name = "{self.project_name}"
dynamic = ["version"]
description = "{self.state.project_description[:100]}"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "MIT"}}
authors = [
    {{name = "Generated by Multi-LLM Orchestrator"}},
]
keywords = ["automation", "pipeline"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = {dependencies!r}

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-xdist>=3.3.0",
    "black>=23.7.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
    "types-requests>=2.31.0",
]
security = [
    "bandit[toml]>=1.7.0",
    "safety>=2.3.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.2.0",
    "mkdocstrings[python]>=0.22.0",
]

[project.scripts]
{self.project_name} = "main:main"

[project.urls]
Homepage = "https://github.com/example/{self.project_name}"
Repository = "https://github.com/example/{self.project_name}"
Documentation = "https://example.github.io/{self.project_name}"

[tool.hatch.version]
path = "src/{self.project_name}/__init__.py"

# Black - Code formatting
[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312", "py313"]
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

# Ruff - Fast Python linter
[tool.ruff]
target-version = "py310"
line-length = 100
select = [
    "E",   # pycodestyle errors
    "F",   # Pyflakes
    "I",   # isort
    "W",   # pycodestyle warnings
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # Line too long (handled by black)
    "B008",  # Do not perform function calls in argument defaults
]
fixable = ["ALL"]
unfixable = []

[tool.ruff.mccabe]
max-complexity = 10

[tool.ruff.pydocstyle]
convention = "google"

# MyPy - Static type checking
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
ignore_missing_imports = true
show_error_codes = true
show_column_numbers = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false

# Pytest - Testing framework
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=src/{self.project_name}",
    "--cov-report=term-missing",
    "--cov-report=html:coverage_html",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

# Coverage - Code coverage reporting
[tool.coverage.run]
source = ["src/{self.project_name}"]
omit = [
    "*/tests/*",
    "*/test_*",
    "__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
fail_under = 80

[tool.coverage.html]
directory = "coverage_html"

# Bandit - Security linting
[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert_used warnings in non-test code

# Semantic versioning bumpversion
[tool.bumpversion]
current_version = "0.1.0"
commit = true
tag = true

[[tool.bumpversion.files]]
filename = "src/{self.project_name}/__init__.py"
"""
        
        toml_file = self.output_dir / "pyproject.toml"
        toml_file.write_text(toml_content, encoding="utf-8")
        logger.info("Generated comprehensive pyproject.toml with all tool configurations")
        return ["pyproject.toml"]
    
    def _extract_dependencies(self) -> list[str]:
        """Extract external package dependencies from module imports."""
        stdlib_modules = {
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib',
            'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'fnmatch',
            'functools', 'glob', 'hashlib', 'html', 'http', 'importlib', 'inspect',
            'io', 'itertools', 'json', 'logging', 'math', 'mimetypes', 'multiprocessing',
            'operator', 'os', 'pathlib', 'pickle', 'platform', 'pprint', 'random',
            're', 'shutil', 'signal', 'socket', 'sqlite3', 'statistics', 'string',
            'subprocess', 'sys', 'tempfile', 'textwrap', 'threading', 'time', 'traceback',
            'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'xml', 'zipfile',
            'builtins', '__future__', 'typing_extensions', 'zoneinfo', 'graphlib',
        }
        
        external = set()
        for module in self.modules:
            for imp in module.imports:
                pkg = imp.split('.')[0]
                if pkg not in stdlib_modules and not pkg.startswith('_'):
                    external.add(pkg)
        
        # Map common package names to PyPI names
        pypi_mapping = {
            'PIL': 'pillow>=10.0.0',
            'sklearn': 'scikit-learn>=1.3.0',
            'yaml': 'pyyaml>=6.0.1',
            'cv2': 'opencv-python>=4.8.0',
            'dateutil': 'python-dateutil>=2.8.2',
            'dotenv': 'python-dotenv>=1.0.0',
            'requests': 'requests>=2.31.0',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
        }
        
        result = []
        for pkg in sorted(external):
            pypi_name = pypi_mapping.get(pkg, f"{pkg}>=1.0.0")
            result.append(pypi_name)
        
        # Always include pydantic-settings for config layer
        if 'pydantic' not in external:
            result.append('pydantic-settings>=2.0.0')
            result.append('pydantic>=2.0.0')
        
        return sorted(result)
    
    def _generate_makefile(self) -> list[str]:
        """Generate comprehensive Makefile with all development tasks."""
        makefile_content = f'''# Auto-generated by Project Assembler
.PHONY: help install install-dev run test test-unit test-integration lint format type-check security-check clean docker-build docker-run

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  install           Install package with dependencies"
	@echo "  install-dev       Install with development dependencies"
	@echo "  run               Run the main pipeline"
	@echo "  run-dry           Run in dry-run mode"
	@echo "  list              List all pipeline steps"
	@echo "  test              Run all tests with coverage"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  lint              Run linters (ruff)"
	@echo "  format            Format code (black)"
	@echo "  format-check      Check code formatting"
	@echo "  type-check        Run type checker (mypy)"
	@echo "  security-check    Run security checks (bandit, safety)"
	@echo "  precommit         Install and run pre-commit hooks"
	@echo "  clean             Remove build artifacts"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-run        Run Docker container"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,security]"
	pre-commit install

run:
	python main.py

run-dry:
	python main.py --dry-run

list:
	python main.py --list

test:
	pytest -xvs

test-unit:
	pytest tests/unit -xvs -m unit

test-integration:
	pytest tests/integration -xvs -m integration

test-cov:
	pytest --cov=src/{self.project_name} --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/ main.py

lint-fix:
	ruff check --fix src/ tests/ main.py

format:
	black src/ tests/ main.py

format-check:
	black --check src/ tests/ main.py

type-check:
	mypy src/ main.py

security-check:
	bandit -r src/ -f json -o bandit-report.json || true
	bandit -r src/
	@echo "Note: Run 'safety check' manually for dependency vulnerabilities"

precommit:
	pre-commit install
	pre-commit run --all-files

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	rm -rf coverage_html/ bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docker-build:
	docker build -t {self.project_name}:latest .

docker-run:
	docker run --rm -it --env-file .env {self.project_name}:latest

docker-test:
	docker build --target test -t {self.project_name}:test .

ci: format-check lint type-check test security-check
	@echo "All CI checks passed!"
'''
        
        makefile_file = self.output_dir / "Makefile"
        makefile_file.write_text(makefile_content, encoding="utf-8")
        logger.info("Generated comprehensive Makefile")
        return ["Makefile"]
    
    def _generate_dockerfile(self) -> list[str]:
        """Generate multi-stage Dockerfile with best practices."""
        dockerfile_content = f'''# Multi-stage Dockerfile for production deployment
# Generated by Project Assembler

# ═══════════════════════════════════════════════════════════════
# Stage 1: Builder
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim as builder

# Security: Run as non-root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (better caching)
COPY pyproject.toml .
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir build && \\
    pip install --no-cache-dir -e .

# ═══════════════════════════════════════════════════════════════
# Stage 2: Production
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim as production

# Security: Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup main.py .
COPY --chown=appuser:appgroup README.md .
COPY --chown=appuser:appgroup pyproject.toml .

# Create necessary directories
RUN mkdir -p logs data && chown -R appuser:appgroup logs data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import {self.project_name}; print('OK')" || exit 1

# Default command
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]

# ═══════════════════════════════════════════════════════════════
# Stage 3: Development
# ═══════════════════════════════════════════════════════════════
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev]"

USER appuser

# ═══════════════════════════════════════════════════════════════
# Stage 4: Testing
# ═══════════════════════════════════════════════════════════════
FROM production as test

USER root

# Install test dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy tests
COPY tests/ ./tests/

USER appuser

# Run tests by default
CMD ["pytest", "-v"]
'''
        
        dockerfile = self.output_dir / "Dockerfile"
        dockerfile.write_text(dockerfile_content, encoding="utf-8")
        
        # Generate .dockerignore
        dockerignore_content = '''# Git
.git
.gitignore
.gitattributes

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/

# Documentation builds
docs/_build/
site/

# Logs
logs/
*.log

# Local environment
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile*
docker-compose*
.docker/

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml
'''
        
        dockerignore = self.output_dir / ".dockerignore"
        dockerignore.write_text(dockerignore_content, encoding="utf-8")
        
        logger.info("Generated Dockerfile (multi-stage) and .dockerignore")
        return ["Dockerfile", ".dockerignore"]
    
    def _generate_github_actions(self) -> list[str]:
        """Generate GitHub Actions CI/CD workflow."""
        workflow_content = f'''# CI/CD Pipeline
# Generated by Project Assembler

name: CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [created]

jobs:
  # ═══════════════════════════════════════════════════════════
  # Lint and Type Check
  # ═══════════════════════════════════════════════════════════
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run linter
        run: ruff check src/ tests/
        
      - name: Check formatting
        run: black --check src/ tests/
        
      - name: Type check
        run: mypy src/

  # ═══════════════════════════════════════════════════════════
  # Security Scan
  # ═══════════════════════════════════════════════════════════
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: |
          pip install -e ".[security]"
          
      - name: Run Bandit security scan
        run: bandit -r src/ -f json -o bandit-report.json || true
        
      - name: Upload security report
        uses: actions/upload-artifact@v4
        with:
          name: security-report
          path: bandit-report.json

  # ═══════════════════════════════════════════════════════════
  # Test Suite
  # ═══════════════════════════════════════════════════════════
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
      fail-fast: false
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run tests with coverage
        run: |
          pytest --cov=src/{self.project_name} --cov-report=xml --cov-report=term
          
      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.11'
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  # ═══════════════════════════════════════════════════════════
  # Build and Push Docker Image
  # ═══════════════════════════════════════════════════════════
  docker:
    needs: [lint, test]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ═══════════════════════════════════════════════════════════
  # Release
  # ═══════════════════════════════════════════════════════════
  release:
    needs: [lint, test]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Install build dependencies
        run: pip install build twine
        
      - name: Build package
        run: python -m build
        
      - name: Publish to PyPI
        if: github.event_name == 'release'
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
'''
        
        workflow_dir = self.output_dir / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_file = workflow_dir / "ci.yml"
        workflow_file.write_text(workflow_content, encoding="utf-8")
        
        logger.info("Generated GitHub Actions CI/CD workflow")
        return [".github/workflows/ci.yml"]
    
    def _generate_env_example(self) -> list[str]:
        """Generate .env.example with all configuration options."""
        env_content = f'''# Environment Configuration
# Copy this file to .env and fill in your actual values
# NEVER commit .env to version control!

# ═══════════════════════════════════════════════════════════════
# Application Settings
# ═══════════════════════════════════════════════════════════════

# Application name (default: {self.project_name})
APP_NAME={self.project_name}

# Application version
APP_VERSION=0.1.0

# Debug mode (true/false)
# WARNING: Never enable in production!
DEBUG=false

# Environment: development, staging, production
ENVIRONMENT=development

# ═══════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log format: json (production) or text (development)
LOG_FORMAT=text

# ═══════════════════════════════════════════════════════════════
# Performance
# ═══════════════════════════════════════════════════════════════

# Maximum worker threads
MAX_WORKERS=4

# Default timeout for operations (seconds)
TIMEOUT_SECONDS=30

# ═══════════════════════════════════════════════════════════════
# Feature Flags
# ═══════════════════════════════════════════════════════════════

# Enable metrics collection
ENABLE_METRICS=true

# Enable distributed tracing
ENABLE_TRACING=false

# ═══════════════════════════════════════════════════════════════
# Security - NEVER hardcode secrets!
# Use environment variables or secret manager
# ═══════════════════════════════════════════════════════════════

# Secret key for encryption (generate with: openssl rand -hex 32)
# SECRET_KEY=your-secret-key-here

# API key for external services
# API_KEY=your-api-key-here

# Database URL (if applicable)
# DATABASE_URL=postgresql://user:pass@localhost/dbname

# Redis URL (if applicable)  
# REDIS_URL=redis://localhost:6379/0
'''
        
        env_file = self.output_dir / ".env.example"
        env_file.write_text(env_content, encoding="utf-8")
        
        logger.info("Generated .env.example")
        return [".env.example"]
    
    def _generate_test_structure(self) -> list[str]:
        """Generate test structure with fixtures and conftest."""
        created = []
        
        # conftest.py
        conftest_content = f'''"""
Pytest Configuration and Fixtures
==================================
Shared fixtures for all tests.
"""
from __future__ import annotations

import os
import pytest
from pathlib import Path

# Ensure tests don't interfere with real environment
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_dir(project_root: Path) -> Path:
    """Return src directory."""
    return project_root / "src"


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return test data directory."""
    data_dir = project_root / "tests" / "fixtures"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment after each test."""
    yield
    # Cleanup code here
'''
        
        conftest_file = self.output_dir / "tests" / "conftest.py"
        conftest_file.write_text(conftest_content, encoding="utf-8")
        created.append("tests/conftest.py")
        
        # Unit test example
        unit_test_content = f'''"""
Unit Tests
==========
Tests for individual components in isolation.
"""
from __future__ import annotations

import pytest

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestExample:
    """Example unit test class."""
    
    def test_example_passes(self):
        """A simple test that always passes."""
        assert True
    
    def test_import_project(self):
        """Test that project module can be imported."""
        import {self.project_name}
        assert {self.project_name}.__version__ is not None
'''
        
        unit_test_file = self.output_dir / "tests" / "unit" / "test_example.py"
        unit_test_file.write_text(unit_test_content, encoding="utf-8")
        created.append("tests/unit/test_example.py")
        
        # Integration test example
        integration_test_content = f'''"""
Integration Tests
=================
Tests for component interactions.
"""
from __future__ import annotations

import pytest

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.slow
class TestIntegration:
    """Example integration test class."""
    
    def test_full_pipeline(self):
        """Test the full pipeline end-to-end."""
        # TODO: Implement integration test
        pass
'''
        
        integration_test_file = self.output_dir / "tests" / "integration" / "test_integration.py"
        integration_test_file.write_text(integration_test_content, encoding="utf-8")
        created.append("tests/integration/test_integration.py")
        
        # __init__.py files
        for init_path in [
            self.output_dir / "tests" / "__init__.py",
            self.output_dir / "tests" / "unit" / "__init__.py",
            self.output_dir / "tests" / "integration" / "__init__.py",
            self.output_dir / "tests" / "fixtures" / "__init__.py",
        ]:
            init_path.write_text('"""Test module."""\n', encoding="utf-8")
            created.append(str(init_path.relative_to(self.output_dir)))
        
        logger.info("Generated test structure with pytest fixtures")
        return created
    
    def _generate_docs_structure(self) -> list[str]:
        """Generate documentation structure."""
        created = []
        
        # Architecture Decision Record (ADR) template
        adr_content = '''# ADR 001: Project Structure

## Status
Accepted

## Context
We need a clean, maintainable project structure that follows Python best practices.

## Decision
We adopt a layered architecture with domain/application/infrastructure separation:

- `domain/`: Pure business logic, no external dependencies
- `application/`: Use cases and orchestration
- `infrastructure/`: External services, I/O, databases

## Consequences
- Clear separation of concerns
- Testability improves
- Dependencies are explicit
'''
        
        adr_file = self.output_dir / "docs" / "adr" / "001-project-structure.md"
        adr_file.write_text(adr_content, encoding="utf-8")
        created.append("docs/adr/001-project-structure.md")
        
        logger.info("Generated documentation structure")
        return created
    
    def _generate_precommit_config(self) -> list[str]:
        """Generate pre-commit configuration."""
        precommit_content = f'''# Pre-commit hooks configuration
# Install: pre-commit install
# Run manually: pre-commit run --all-files

repos:
  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key
      - id: check-case-conflict

  # Black - Code formatting
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  # Ruff - Fast Python linter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  # MyPy - Type checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  # Bandit - Security linting
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]
'''
        
        precommit_file = self.output_dir / ".pre-commit-config.yaml"
        precommit_file.write_text(precommit_content, encoding="utf-8")
        
        logger.info("Generated .pre-commit-config.yaml")
        return [".pre-commit-config.yaml"]
    
    def _generate_exception_hierarchy(self) -> list[str]:
        """Generate exception hierarchy for proper error handling."""
        exceptions_content = f'''"""
Exception Hierarchy
===================
Explicit exception hierarchy for proper error handling.

Usage:
    try:
        do_something()
    except DomainError as e:
        # Handle domain-specific errors
        pass
    except InfrastructureError as e:
        # Handle infrastructure errors (retriable)
        pass
"""
from __future__ import annotations


class ApplicationError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, *, code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {{}}
    
    def __str__(self) -> str:
        if self.details:
            return f"{{self.code}}: {{self.message}} ({{self.details}})"
        return f"{{self.code}}: {{self.message}}"


# ═══════════════════════════════════════════════════════════════
# Domain Errors
# ═══════════════════════════════════════════════════════════════

class DomainError(ApplicationError):
    """Errors in the business domain logic."""
    pass


class ValidationError(DomainError):
    """Input validation failed."""
    
    def __init__(self, message: str, *, field: str | None = None):
        super().__init__(message, code="VALIDATION_ERROR", details={{"field": field}})


class BusinessRuleError(DomainError):
    """Business rule was violated."""
    
    def __init__(self, message: str, *, rule: str | None = None):
        super().__init__(message, code="BUSINESS_RULE_VIOLATION", details={{"rule": rule}})


# ═══════════════════════════════════════════════════════════════
# Application Errors
# ═══════════════════════════════════════════════════════════════

class ApplicationLayerError(ApplicationError):
    """Errors in the application/orchestration layer."""
    pass


class ConfigurationError(ApplicationLayerError):
    """Configuration is invalid or missing."""
    
    def __init__(self, message: str, *, config_key: str | None = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details={{"key": config_key}})


class TimeoutError(ApplicationLayerError):
    """Operation timed out."""
    
    def __init__(self, message: str, *, timeout_seconds: float | None = None):
        super().__init__(message, code="TIMEOUT", details={{"timeout": timeout_seconds}})


# ═══════════════════════════════════════════════════════════════
# Infrastructure Errors
# ═══════════════════════════════════════════════════════════════

class InfrastructureError(ApplicationError):
    """Errors from external services or infrastructure.
    
    These errors are typically retriable.
    """
    pass


class ExternalServiceError(InfrastructureError):
    """External service returned an error."""
    
    def __init__(
        self,
        message: str,
        *,
        service: str | None = None,
        status_code: int | None = None,
        retriable: bool = True
    ):
        super().__init__(
            message,
            code="EXTERNAL_SERVICE_ERROR",
            details={{
                "service": service,
                "status_code": status_code,
                "retriable": retriable
            }}
        )
        self.retriable = retriable


class DatabaseError(InfrastructureError):
    """Database operation failed."""
    
    def __init__(self, message: str, *, operation: str | None = None):
        super().__init__(message, code="DATABASE_ERROR", details={{"operation": operation}})


class NetworkError(InfrastructureError):
    """Network operation failed."""
    
    def __init__(self, message: str, *, url: str | None = None):
        super().__init__(message, code="NETWORK_ERROR", details={{"url": url}})
'''
        
        base_dir = self.output_dir / "src" / self.project_name
        exceptions_file = base_dir / "exceptions.py"
        exceptions_file.write_text(exceptions_content, encoding="utf-8")
        
        logger.info("Generated exception hierarchy")
        return [f"src/{self.project_name}/exceptions.py"]
    
    def _generate_logging_config(self) -> list[str]:
        """Generate structured logging utilities."""
        logging_content = f'''"""
Structured Logging Utilities
============================
Structured logging with correlation IDs and context.

Usage:
    from {self.project_name}.logging import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started", extra={{"item_id": 123}})
"""
from __future__ import annotations

import logging
import sys
import json
from typing import Any
from contextvars import ContextVar
from datetime import datetime

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {{
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get() or getattr(record, 'correlation_id', ''),
        }}
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in getattr(record, 'extra', {{}}).items():
            log_obj[key] = value
        
        return json.dumps(log_obj, default=str)


class ContextFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the application name prefix."""
    return logging.getLogger(f"{self.project_name}.{{name}}")


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id.set(cid)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id.get()


def configure_structured_logging(level: str = "INFO", format: str = "json") -> None:
    """Configure structured logging for the application."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s"
        )
    
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(ContextFilter())
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
'''
        
        base_dir = self.output_dir / "src" / self.project_name
        logging_file = base_dir / "logging.py"
        logging_file.write_text(logging_content, encoding="utf-8")
        
        logger.info("Generated structured logging utilities")
        return [f"src/{self.project_name}/logging.py"]
