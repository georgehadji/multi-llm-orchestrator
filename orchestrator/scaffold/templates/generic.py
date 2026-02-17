"""Generic/fallback scaffold template — used for unknown app types."""
from __future__ import annotations

FILES: dict[str, str] = {
    "main.py": '"""Main entry point."""\n\n\ndef main() -> None:\n    print("Hello, World!")\n\n\nif __name__ == "__main__":\n    main()\n',
    "src/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_main.py": (
        'from main import main\n\n\n'
        'def test_main_runs(capsys):\n'
        '    main()\n'
        '    captured = capsys.readouterr()\n'
        '    assert "Hello" in captured.out\n'
    ),
    "pyproject.toml": (
        '[project]\n'
        'name = "my-app"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = []\n\n'
        '[build-system]\n'
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.backends.legacy:build"\n'
    ),
    ".gitignore": "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
    "README.md": "# My App\n\n## Running\n\n```bash\npython main.py\n```\n",
    ".env.example": "# Environment variables\n",
}
