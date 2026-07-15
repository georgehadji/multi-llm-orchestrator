"""FastAPI application scaffold template.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

FASTAPI_MAIN = """\
\"\"\"FastAPI application entry point — production-ready scaffold.\"\"\"
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Structured logging ──────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Configuration (from env, never hardcoded) ────────────────────────
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # noqa: ARG001
    \"\"\"Startup and shutdown lifecycle.\"\"\"
    logger.info("Application starting — environment=%s", ENVIRONMENT)
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="My FastAPI App",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    \"\"\"Catch-all handler — never leak stack traces to clients.\"\"\"
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    \"\"\"Structured HTTP error responses.\"\"\"
    logger.warning(
        "HTTP %d on %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


# ── Health / root endpoints ───────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    \"\"\"Kubernetes-style health check.\"\"\"
    return {"status": "healthy", "environment": ENVIRONMENT}


@app.get("/")
async def root() -> dict:
    return {"message": "Hello, World!", "version": app.version}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
"""

FASTAPI_REQUIREMENTS = """\
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
"""

FILES: dict[str, str] = {
    "src/__init__.py": "",
    "src/main.py": FASTAPI_MAIN,
    "tests/__init__.py": "",
    "tests/test_main.py": (
        "import pytest\n"
        "from fastapi.testclient import TestClient\n"
        "from src.main import app\n\n\n"
        "client = TestClient(app)\n\n\n"
        "def test_health():\n"
        '    response = client.get("/health")\n'
        "    assert response.status_code == 200\n"
        '    assert response.json()["status"] == "healthy"\n\n\n'
        "def test_root():\n"
        '    response = client.get("/")\n'
        "    assert response.status_code == 200\n"
        '    assert "message" in response.json()\n\n\n'
        "def test_global_exception_handler():\n"
        '    response = client.get("/nonexistent")\n'
        "    assert response.status_code == 404\n"
    ),
    "pyproject.toml": (
        "[project]\n"
        'name = "my-app"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        "dependencies = [\n"
        '    "fastapi>=0.110.0",\n'
        '    "uvicorn[standard]>=0.27.0",\n'
        "]\n\n"
        "[project.optional-dependencies]\n"
        "dev = [\n"
        '    "pytest>=8.0",\n'
        '    "pytest-cov>=5.0",\n'
        '    "ruff>=0.4",\n'
        '    "mypy>=1.8",\n'
        "]\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n'
    ),
    ".gitignore": "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\ndist/\nbuild/\n*.egg-info/\n",
    "README.md": r"""# 🚀 FastAPI Backend Application

**Project Type:** Backend (Python + FastAPI)
**Platform:** Windows, macOS, Linux

---

## ⚡ Quick Start (Windows)

### Prerequisites
- [Python 3.11+](https://python.org/downloads/)
- pip (included with Python)

### 1. Create Virtual Environment
```powershell
# Open PowerShell in project folder
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -e .
# OR
pip install fastapi uvicorn
```

### 3. Start Development Server
```powershell
uvicorn src.main:app --reload
```

### 4. Test the API
Open browser: [http://localhost:8000](http://localhost:8000)

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Run Tests

```powershell
# Make sure venv is activated
venv\Scripts\activate

# Run tests
pytest

# Run with coverage
pytest --cov=src
```

---

## 🛠️ Available Commands

| Command | Description |
|---------|-------------|
| `uvicorn src.main:app --reload` | Start dev server with auto-reload |
| `uvicorn src.main:app --host 0.0.0.0 --port 8000` | Start on all interfaces |
| `pytest` | Run unit tests |
| `pytest --cov=src` | Run tests with coverage |

---

## 📁 Project Structure

```
├── src/
│   ├── __init__.py
│   └── main.py         # FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_main.py    # Unit tests
├── venv/               # Virtual environment
├── pyproject.toml      # Dependencies
└── README.md           # This file
```

---

**Generated by Multi-LLM Orchestrator** — *This IS a Python project. Requires Python 3.11+*
""",
    ".env.example": (
        "# ── Server ──────────────────────────────────\n"
        "PORT=8000\n"
        "HOST=0.0.0.0\n"
        "# ── Environment ────────────────────────────\n"
        "ENVIRONMENT=development\n"
        "LOG_LEVEL=INFO\n"
        "# ── CORS ───────────────────────────────────\n"
        "ALLOWED_ORIGINS=*\n"
        "# ── Secrets (replace with real values) ─────\n"
        "# DATABASE_URL=postgresql://user:pass@localhost/db\n"
        "# SECRET_KEY=change-me-in-production\n"
    ),
}
