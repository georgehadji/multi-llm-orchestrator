"""CLI application scaffold template.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""
from __future__ import annotations

CLI_ENTRY = '''\
"""Command-line interface entry point."""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="My CLI App")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    print("Hello from CLI!")
    if args.verbose:
        print("Verbose mode enabled")


if __name__ == "__main__":
    main()
'''

FILES: dict[str, str] = {
    "cli.py": CLI_ENTRY,
    "src/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_cli.py": (
        'import subprocess\nimport sys\n\n\n'
        'def test_cli_runs():\n'
        '    result = subprocess.run(\n'
        '        [sys.executable, "cli.py"],\n'
        '        capture_output=True, text=True\n'
        '    )\n'
        '    assert result.returncode == 0\n'
    ),
    "pyproject.toml": (
        '[project]\n'
        'name = "my-cli"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = []\n\n'
        '[project.scripts]\n'
        'my-cli = "cli:main"\n\n'
        '[build-system]\n'
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.backends.legacy:build"\n'
    ),
    ".gitignore": "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
    "README.md": "# My CLI App\n\n## Running\n\n```bash\npython cli.py\n```\n",
    ".env.example": "# Environment variables\n",
}
