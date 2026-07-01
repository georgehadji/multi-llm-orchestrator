"""
DevServer - Auto-detect project type and start local server.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 10, Phase B6 (Base44-inspired).
"""

from __future__ import annotations
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProjectType:
    name: str
    indicators: list[str]  # Files that indicate this project type
    start_command: str  # Command to start dev server
    install_command: str = ""
    default_port: int = 8000


PROJECT_TYPES = [
    ProjectType(
        "FastAPI",
        ["main.py", "app.py", "api.py", "requirements.txt"],
        "uvicorn main:app --reload --port {port}",
        "pip install -r requirements.txt",
        8000,
    ),
    ProjectType(
        "Flask",
        ["app.py", "wsgi.py", "requirements.txt"],
        "flask run --port {port}",
        "pip install -r requirements.txt",
        5000,
    ),
    ProjectType(
        "Next.js",
        ["next.config.js", "next.config.mjs", "package.json"],
        "npm run dev -- -p {port}",
        "npm install",
        3000,
    ),
    ProjectType(
        "React", ["src/App.jsx", "src/App.tsx", "package.json"], "npm start", "npm install", 3000
    ),
    ProjectType(
        "Django",
        ["manage.py", "requirements.txt"],
        "python manage.py runserver 0.0.0.0:{port}",
        "pip install -r requirements.txt",
        8000,
    ),
    ProjectType(
        "Streamlit",
        ["streamlit_app.py", "app.py", "requirements.txt"],
        "streamlit run app.py --server.port {port}",
        "pip install -r requirements.txt",
        8501,
    ),
    ProjectType(
        "Gradio",
        ["app.py", "requirements.txt"],
        "python app.py",
        "pip install -r requirements.txt",
        7860,
    ),
    ProjectType(
        "Node.js", ["server.js", "index.js", "package.json"], "node server.js", "npm install", 3000
    ),
    ProjectType(
        "Express", ["app.js", "server.js", "package.json"], "node app.js", "npm install", 3000
    ),
    ProjectType(
        "CLI",
        ["cli.py", "main.py", "setup.py", "pyproject.toml"],
        "python cli.py",
        "pip install -e .",
        0,
    ),
    ProjectType("Static HTML", ["index.html"], "python -m http.server {port}", "", 8080),
]


class DevServer:
    """Detects project type and starts the appropriate dev server."""

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.detected = None

    def detect(self):
        """Auto-detect project type. Returns first match by indicator count."""
        if not self.project_dir.exists():
            return None
        existing = set()
        for f in self.project_dir.rglob("*"):
            if f.is_file():
                existing.add(f.name)
                existing.add(str(f.relative_to(self.project_dir)))

        best = None
        best_score = 0
        for pt in PROJECT_TYPES:
            score = sum(1 for ind in pt.indicators if ind in existing)
            if score > best_score:
                best_score = score
                best = pt

        self.detected = best
        return best

    def install(self, project_type=None):
        pt = project_type or self.detected or self.detect()
        if not pt or not pt.install_command:
            return False, "No install command"
        try:
            result = subprocess.run(
                pt.install_command,
                shell=True,  # nosec B602 — user-defined project commands
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.returncode == 0, result.stdout[-500:]
        except Exception as e:
            return False, str(e)

    def start(self, project_type=None, port=None, background=True):
        pt = project_type or self.detected or self.detect()
        if not pt:
            return None, "Could not detect project type"
        if pt.default_port == 0:
            return None, "CLI project - no server needed"

        resolved_port = port or pt.default_port
        try:
            resolved_port = int(resolved_port)
        except (TypeError, ValueError):
            return None, f"Invalid port value: {resolved_port!r}"
        if not (1 <= resolved_port <= 65535):
            return None, f"Port out of range: {resolved_port}"
        cmd = pt.start_command.format(port=resolved_port)
        if background:
            proc = subprocess.Popen(
                cmd,
                shell=True,  # nosec B602 — user-defined project commands
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return proc, f"Started {pt.name} on port {port or pt.default_port} (PID {proc.pid})"
        else:
            result = subprocess.run(
                cmd,
                shell=True,  # nosec B602 — user-defined project commands
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result, result.stdout[-500:]
