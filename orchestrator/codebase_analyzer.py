"""Codebase static analysis and structure extraction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class CodebaseMap:
    """Static analysis result of a codebase"""
    root_path: str
    total_files: int
    total_lines_of_code: int
    files_by_language: Dict[str, int] = field(default_factory=dict)
    key_files: list[str] = field(default_factory=list)
    module_structure: Dict[str, list[str]] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    has_tests: bool = False
    has_docs: bool = False
    estimated_complexity: str = "unknown"  # low, medium, high
    primary_language: Optional[str] = None
    secondary_languages: list[str] = field(default_factory=list)
    has_python_tests: bool = False
    has_js_tests: bool = False
    project_type: str = "generic"  # fastapi, django, nextjs, react, etc


class CodebaseAnalyzer:
    """Analyze codebase structure without LLM"""

    SKIP_DIRS = {
        ".git", ".venv", "venv", "env", "__pycache__",
        "node_modules", ".pytest_cache", ".mypy_cache",
        "dist", "build", ".egg-info", ".next", "out"
    }

    def scan(self, root_path: str) -> CodebaseMap:
        """
        Scan a directory and extract codebase structure.

        Args:
            root_path: Path to directory to scan

        Returns:
            CodebaseMap with analysis results
        """
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Directory does not exist: {root_path}")

        files_by_language: Dict[str, int] = {}
        total_files = 0
        total_lines = 0
        module_structure: Dict[str, list[str]] = {}
        key_files: list[str] = []

        # Scan all files
        for file_path in root.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip hidden and cache directories
            if any(skip in file_path.parts for skip in self.SKIP_DIRS):
                continue

            total_files += 1

            # Count by extension
            suffix = file_path.suffix.lower()
            files_by_language[suffix] = files_by_language.get(suffix, 0) + 1

            # Count lines for text files
            try:
                if self._is_text_file(file_path):
                    lines = len(file_path.read_text(encoding="utf-8").splitlines())
                    total_lines += lines
            except (UnicodeDecodeError, OSError):
                pass

            # Track key files
            if file_path.name in ["README.md", "requirements.txt", "package.json",
                                  "main.py", "app.py", "index.js", "Dockerfile"]:
                key_files.append(str(file_path.relative_to(root)))

        # Detect tests and docs
        has_tests = (root / "tests").exists() or (root / "test").exists()
        has_docs = (root / "docs").exists() or (root / "README.md").exists()

        # Detect language and project type
        primary_lang, secondary_langs = self._detect_primary_language(files_by_language)
        project_type = self._detect_project_type(root, key_files, [])
        has_python_tests = has_tests and primary_lang == "python"
        has_js_tests = has_tests and primary_lang in ["javascript", "typescript"]

        return CodebaseMap(
            root_path=str(root),
            total_files=total_files,
            total_lines_of_code=total_lines,
            files_by_language=files_by_language,
            key_files=key_files,
            module_structure=module_structure,
            has_tests=has_tests,
            has_docs=has_docs,
            primary_language=primary_lang,
            secondary_languages=secondary_langs,
            project_type=project_type,
            has_python_tests=has_python_tests,
            has_js_tests=has_js_tests,
        )

    def _is_text_file(self, path: Path) -> bool:
        """Check if file is text-based"""
        text_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
            ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt",
            ".yml", ".yaml", ".json", ".xml", ".html", ".css", ".md",
            ".txt", ".sql", ".sh", ".bash", ".env", ".properties"
        }
        return path.suffix.lower() in text_extensions

    def _detect_primary_language(self, files_by_language: Dict[str, int]) -> tuple[Optional[str], list[str]]:
        """Detect primary and secondary programming languages"""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
        }

        language_counts = {}
        for ext, count in files_by_language.items():
            if ext in language_map:
                lang = language_map[ext]
                language_counts[lang] = language_counts.get(lang, 0) + count

        if not language_counts:
            return None, []

        sorted_langs = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_langs[0][0]
        secondary = [lang for lang, _ in sorted_langs[1:]]

        return primary, secondary

    def _detect_project_type(self, root: Path, key_files: list[str], dependencies: list[str]) -> str:
        """Detect project type from config files and dependencies"""

        # Check Python projects
        if (root / "requirements.txt").exists():
            requirements = (root / "requirements.txt").read_text()
            if "fastapi" in requirements:
                return "fastapi"
            if "django" in requirements:
                return "django"
            if "flask" in requirements:
                return "flask"
            return "python"

        # Check JavaScript/Node projects
        package_json_path = root / "package.json"
        if package_json_path.exists():
            try:
                import json
                pkg = json.loads(package_json_path.read_text())
                deps = pkg.get("dependencies", {})
                if "next" in deps:
                    return "nextjs"
                if "react" in deps:
                    return "react"
                if "vue" in deps:
                    return "vue"
                return "nodejs"
            except Exception:
                return "nodejs"

        # Check Go projects
        if (root / "go.mod").exists():
            return "golang"

        # Check Rust projects
        if (root / "Cargo.toml").exists():
            return "rust"

        return "generic"
