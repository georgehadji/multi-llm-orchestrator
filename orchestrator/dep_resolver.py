"""
DependencyResolver — scans Python source files and resolves third-party dependencies.

Steps:
1. Recursively find all .py files in output_dir
2. Parse each file with ast.parse() to extract import names
3. Skip stdlib modules (sys.stdlib_module_names) and relative imports
4. Map import names to PyPI package names using _IMPORT_TO_PYPI
5. Write requirements.txt
6. Update pyproject.toml [project.dependencies] if it exists
7. Return ResolveReport
"""
from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping from import name → PyPI package specifier (for cases where they differ)
_IMPORT_TO_PYPI: dict[str, str] = {
    "fastapi": "fastapi>=0.110.0",
    "uvicorn": "uvicorn[standard]>=0.27.0",
    "pydantic": "pydantic>=2.0.0",
    "sqlalchemy": "sqlalchemy>=2.0.0",
    "alembic": "alembic>=1.13.0",
    "httpx": "httpx>=0.27.0",
    "requests": "requests>=2.31.0",
    "aiohttp": "aiohttp>=3.9.0",
    "click": "click>=8.1.0",
    "typer": "typer>=0.12.0",
    "rich": "rich>=13.0.0",
    "pytest": "pytest>=8.0.0",
    "flask": "flask>=3.0.0",
    "django": "django>=5.0.0",
    "starlette": "starlette>=0.37.0",
    "celery": "celery>=5.3.0",
    "redis": "redis>=5.0.0",
    "boto3": "boto3>=1.34.0",
    "numpy": "numpy>=1.26.0",
    "pandas": "pandas>=2.2.0",
    "PIL": "Pillow>=10.0.0",
    "cv2": "opencv-python>=4.9.0",
    "sklearn": "scikit-learn>=1.4.0",
    "torch": "torch>=2.2.0",
    "transformers": "transformers>=4.40.0",
    "dotenv": "python-dotenv>=1.0.0",
    "yaml": "pyyaml>=6.0.1",
    "toml": "toml>=0.10.2",
    "jwt": "PyJWT>=2.8.0",
    "cryptography": "cryptography>=42.0.0",
    "psycopg2": "psycopg2-binary>=2.9.9",
    "pymongo": "pymongo>=4.6.0",
    "motor": "motor>=3.3.0",
    "beanie": "beanie>=1.25.0",
    "loguru": "loguru>=0.7.0",
    "structlog": "structlog>=24.0.0",
    "tenacity": "tenacity>=8.2.0",
    "anyio": "anyio>=4.3.0",
    "trio": "trio>=0.25.0",
}

# stdlib modules set — available in Python 3.10+
_STDLIB: frozenset[str] = frozenset(
    getattr(sys, "stdlib_module_names", frozenset())
)


@dataclass
class ResolveReport:
    """Report from DependencyResolver.resolve()."""

    packages: list[str] = field(default_factory=list)
    requirements_path: str = ""
    pyproject_updated: bool = False
    unresolved: list[str] = field(default_factory=list)


class DependencyResolver:
    """
    Scans Python source files for imports and resolves them to PyPI packages.

    Usage:
        resolver = DependencyResolver()
        report = resolver.resolve(output_dir)
    """

    def resolve(self, output_dir: Path) -> ResolveReport:
        """
        Scan all .py files in output_dir and write requirements.txt.

        Returns ResolveReport with detected packages (plain import names, deduplicated).
        """
        output_dir = Path(output_dir)
        report = ResolveReport()

        # Collect all third-party import names (plain, top-level)
        raw_imports: set[str] = set()
        for py_file in sorted(output_dir.rglob("*.py")):
            self._scan_file(py_file, raw_imports)

        # report.packages holds the plain (deduplicated) import/package names
        report.packages = sorted(raw_imports)

        # Build versioned specifiers for requirements.txt
        specifiers: list[str] = []
        unresolved: list[str] = []
        for imp in report.packages:
            if imp in _IMPORT_TO_PYPI:
                specifiers.append(_IMPORT_TO_PYPI[imp])
            else:
                # Use the import name directly as PyPI name (best guess)
                specifiers.append(imp)
                unresolved.append(imp)
                logger.debug(
                    "No PyPI mapping for '%s'; using import name as package name", imp
                )

        report.unresolved = sorted(unresolved)

        # Write requirements.txt with versioned specifiers
        req_path = output_dir / "requirements.txt"
        req_path.write_text(
            "\n".join(specifiers) + ("\n" if specifiers else ""),
            encoding="utf-8",
        )
        report.requirements_path = str(req_path)
        logger.debug("Wrote requirements.txt with %d packages", len(specifiers))

        # Update pyproject.toml if it exists
        pyproject_path = output_dir / "pyproject.toml"
        if pyproject_path.exists():
            report.pyproject_updated = self._update_pyproject(pyproject_path, specifiers)

        return report

    def _scan_file(self, py_file: Path, raw_imports: set[str]) -> None:
        """Parse a .py file and add third-party import names to raw_imports."""
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (OSError, SyntaxError) as exc:
            logger.warning("Could not parse %s: %s", py_file, exc)
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if self._is_third_party(top_level):
                        raw_imports.add(top_level)
            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (level > 0)
                if (node.level or 0) > 0:
                    continue
                if node.module:
                    top_level = node.module.split(".")[0]
                    if self._is_third_party(top_level):
                        raw_imports.add(top_level)

    def _is_third_party(self, module_name: str) -> bool:
        """Return True if module_name is not in stdlib and not a private name."""
        if not module_name or module_name.startswith("_"):
            return False
        if module_name in _STDLIB:
            return False
        # Skip common internal package names
        if module_name in {
            "src", "tests", "orchestrator", "config", "utils",
            "models", "app",
        }:
            return False
        return True

    def _update_pyproject(self, pyproject_path: Path, specifiers: list[str]) -> bool:
        """
        Update [project.dependencies] in pyproject.toml with detected packages.

        Simple regex-based replacement — only updates the dependencies list.
        Returns True if the file was modified.
        """
        try:
            content = pyproject_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read pyproject.toml: %s", exc)
            return False

        if not specifiers:
            return False

        # Build new dependencies list
        deps_lines = "\n".join(f'    "{pkg}",' for pkg in specifiers)
        new_deps_block = f"dependencies = [\n{deps_lines}\n]"

        # Replace existing dependencies = [...] block
        import re  # noqa: PLC0415 — local import to avoid top-level dependency

        pattern = r"dependencies\s*=\s*\[[^\]]*\]"
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_deps_block, content, flags=re.DOTALL)
            if new_content != content:
                pyproject_path.write_text(new_content, encoding="utf-8")
                logger.debug("Updated pyproject.toml dependencies")
                return True
        return False
