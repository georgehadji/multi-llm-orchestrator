"""Python library scaffold template."""
from __future__ import annotations

FILES: dict[str, str] = {
    "src/__init__.py": '"""My library."""\n__version__ = "0.1.0"\n',
    "src/core.py": '"""Core library functionality."""\n\n\ndef hello() -> str:\n    return "Hello from library!"\n',
    "tests/__init__.py": "",
    "tests/test_core.py": (
        'from src.core import hello\n\n\n'
        'def test_hello():\n'
        '    assert hello() == "Hello from library!"\n'
    ),
    "pyproject.toml": (
        '[project]\n'
        'name = "my-library"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = []\n\n'
        '[build-system]\n'
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.backends.legacy:build"\n'
    ),
    ".gitignore": "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
    "README.md": "# My Library\n\n## Installation\n\n```bash\npip install -e .\n```\n",
    ".env.example": "# Environment variables\n",
}
