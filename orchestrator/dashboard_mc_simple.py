"""
Mission Control Dashboard v6.0 - Simplified Version
===================================================
"""
from __future__ import annotations

import asyncio
import json
import uuid
import webbrowser
from dataclasses import dataclass
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class MissionState:
    """Simple mission control state."""
    version: str = "6.0.0"
    status: str = "idle"
    active_project: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "active_project": self.active_project,
        }


class MissionControlServer:
    """Simple mission control server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.state = MissionState()
        self.app = None
        self.active_connections: list[Any] = []

    async def _setup_fastapi(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse

        self.app = FastAPI(title="Mission Control v6.0")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def get_dashboard():
            return HTMLResponse(content=self._get_html())

        @self.app.get("/api/state")
        async def get_state():
            return JSONResponse(content=self.state.to_dict())

        @self.app.post("/api/project/start")
        async def start_project(data: dict):
            self.state.status = "running"
            self.state.active_project = {
                "id": str(uuid.uuid4())[:8],
                "prompt": data.get("prompt", ""),
                "type": data.get("project_type", "fullstack"),
                "budget": data.get("budget", 5.0),
            }
            return {"status": "started"}

        @self.app.post("/api/project/stop")
        async def stop_project():
            self.state.status = "idle"
            self.state.active_project = None
            return {"status": "stopped"}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            await websocket.send_json({"type": "init", "data": self.state.to_dict()})

            try:
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    if msg.get("action") == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    async def run(self):
        from uvicorn import Config, Server

        await self._setup_fastapi()

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)

        logger.info(f"🚀 Mission Control running on http://{self.host}:{self.port}")
        await server.serve()

    def _get_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control v6.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-900 text-white p-6">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-3xl font-bold mb-6">🚀 Mission Control v6.0</h1>

        <div id="starter" class="bg-slate-800 p-6 rounded-lg">
            <h2 class="text-xl mb-4">Start New Project</h2>
            <select id="type" class="w-full p-2 mb-4 bg-slate-700 rounded">
                <option value="fullstack">Full-Stack</option>
                <option value="frontend">Front-End</option>
                <option value="backend">Back-End</option>
            </select>
            <textarea id="prompt" class="w-full p-2 mb-4 bg-slate-700 rounded" rows="3" placeholder="Enter prompt..."></textarea>
            <button onclick="start()" class="bg-blue-500 px-6 py-2 rounded font-bold">Start</button>
        </div>

        <div id="project" class="hidden bg-slate-800 p-6 rounded-lg mt-4">
            <h2 class="text-xl mb-4">Running Project</h2>
            <p id="project-info"></p>
            <button onclick="stop()" class="bg-red-500 px-6 py-2 rounded font-bold mt-4">Stop</button>
        </div>
    </div>

    <script>
        let ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.data.active_project) {
                document.getElementById('starter').classList.add('hidden');
                document.getElementById('project').classList.remove('hidden');
                document.getElementById('project-info').textContent = msg.data.active_project.prompt;
            } else {
                document.getElementById('starter').classList.remove('hidden');
                document.getElementById('project').classList.add('hidden');
            }
        };

        async function start() {
            await fetch('/api/project/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    prompt: document.getElementById('prompt').value,
                    project_type: document.getElementById('type').value,
                    budget: 5.0
                })
            });
        }

        async function stop() {
            await fetch('/api/project/stop', {method: 'POST'});
        }
    </script>
</body>
</html>
'''


def run_mission_control(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):

    url = f"http://{host}:{port}"
    print(f"🚀 Mission Control v6.0: {url}")

    if open_browser:
        webbrowser.open(url)

    server = MissionControlServer(host=host, port=port)
    asyncio.run(server.run())


if __name__ == "__main__":
    run_mission_control()
