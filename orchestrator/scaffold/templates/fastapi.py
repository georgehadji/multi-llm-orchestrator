"""FastAPI application scaffold template."""
from __future__ import annotations

FASTAPI_MAIN = '''\
from fastapi import FastAPI

app = FastAPI(title="My FastAPI App")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {"message": "Hello, World!"}
'''

FASTAPI_REQUIREMENTS = '''\
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
'''

FILES: dict[str, str] = {
    "src/__init__.py": "",
    "src/main.py": FASTAPI_MAIN,
    "tests/__init__.py": "",
    "tests/test_main.py": (
        'import pytest\n'
        'from fastapi.testclient import TestClient\n'
        'from src.main import app\n\n\n'
        'client = TestClient(app)\n\n\n'
        'def test_health():\n'
        '    response = client.get("/health")\n'
        '    assert response.status_code == 200\n'
    ),
    "pyproject.toml": (
        '[project]\n'
        'name = "my-app"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = [\n'
        '    "fastapi>=0.110.0",\n'
        '    "uvicorn[standard]>=0.27.0",\n'
        ']\n\n'
        '[build-system]\n'
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.backends.legacy:build"\n'
    ),
    ".gitignore": "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
    "README.md": "# My FastAPI App\n\n## Running\n\n```bash\nuvicorn src.main:app --reload\n```\n",
    ".env.example": "# Environment variables\nPORT=8000\n",
}
