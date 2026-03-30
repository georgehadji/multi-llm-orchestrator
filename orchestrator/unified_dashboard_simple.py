"""
Unified Dashboard v5.2 - Simple Version (Vanilla JS)
=====================================================
Χρησιμοποιεί vanilla JavaScript αντί για React - πιο αξιόπιστο.
"""
from __future__ import annotations

import asyncio
import json
import webbrowser
from dataclasses import asdict, dataclass, field
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class GamificationState:
    """Gamification state."""
    xp: int = 0
    level: int = 1
    streak: int = 0
    achievements_unlocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "xp": self.xp,
            "level": self.level,
            "streak": self.streak,
            "achievements": self.achievements_unlocked,
            "next_level_xp": self.level * 500,
            "progress": (self.xp % 500) / 500 * 100,
        }


@dataclass
class ApiConnectionStatus:
    """API connection status."""
    provider: str
    status: str = "disconnected"
    last_error: str = ""
    response_time_ms: float = 0.0
    models_available: list[str] = field(default_factory=list)


@dataclass
class UnifiedDashboardState:
    """Complete dashboard state."""
    status: str = "idle"
    version: str = "5.2.0"
    project_name: str = ""
    project_description: str = ""
    success_criteria: str = ""
    budget_usd: float = 0.0
    spent_usd: float = 0.0
    total_tasks: int = 0
    completed_tasks: int = 0
    current_task_index: int = 0
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: int = 0
    gamification: GamificationState = field(default_factory=GamificationState)
    api_connections: list[ApiConnectionStatus] = field(default_factory=list)
    total_api_calls: int = 0
    total_tokens: int = 0
    current_task: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "project": {
                "name": self.project_name,
                "description": self.project_description[:200] + "..." if len(self.project_description) > 200 else self.project_description,
                "budget_usd": round(self.budget_usd, 2),
                "spent_usd": round(self.spent_usd, 2),
                "budget_percent": round(self.spent_usd / self.budget_usd * 100, 1) if self.budget_usd > 0 else 0,
            },
            "progress": {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "percent": round(self.completed_tasks / self.total_tasks * 100, 1) if self.total_tasks > 0 else 0,
            },
            "time": {
                "elapsed": self._format_time(self.elapsed_seconds),
                "eta": self._format_time(self.eta_seconds),
            },
            "gamification": self.gamification.to_dict(),
            "api_connections": [asdict(c) for c in self.api_connections],
            "stats": {
                "api_calls": self.total_api_calls,
                "tokens": self.total_tokens,
            },
            "current_task": self.current_task,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"


class ApiConnectionManager:
    """Manages auto-connection to all configured APIs."""

    def __init__(self, state: UnifiedDashboardState):
        self.state = state

    async def connect_all(self):
        """Connect to all configured APIs."""
        logger.info("🔌 Starting auto-API connection...")

        providers = [
            ("deepseek", "DEEPSEEK_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("google", "GOOGLE_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("minimax", "MINIMAX_API_KEY"),
        ]

        import os

        for provider, env_var in providers:
            api_key = os.getenv(env_var)
            status = ApiConnectionStatus(provider=provider)

            if not api_key:
                status.status = "no_key"
            else:
                status.status = "connected"
                status.models_available = self._get_models(provider)

            self.state.api_connections.append(status)

        connected = sum(1 for c in self.state.api_connections if c.status == "connected")
        logger.info(f"🔌 API Connection complete: {connected}/{len(providers)} connected")

    def _get_models(self, provider: str) -> list[str]:
        models = {
            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "anthropic": ["claude-3-5-sonnet", "claude-3-haiku"],
            "minimax": ["minimax-3"],
        }
        return models.get(provider, [])


class UnifiedDashboardServer:
    """Ultimate dashboard with vanilla JS frontend."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.state = UnifiedDashboardState()
        self.api_manager = ApiConnectionManager(self.state)
        self.app = None
        self.active_connections: list[Any] = []

    async def _setup_fastapi(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse

        self.app = FastAPI(title="Mission Control ULTIMATE v5.2")

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

        @self.app.get("/api/connections")
        async def get_connections():
            return JSONResponse(content={
                "connections": [asdict(c) for c in self.state.api_connections],
            })

        @self.app.post("/api/connect")
        async def reconnect_apis():
            await self.api_manager.connect_all()
            return {"status": "ok"}

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

    async def _broadcast(self, message: dict):
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in self.active_connections:
                self.active_connections.remove(ws)

    async def run(self):
        from uvicorn import Config, Server

        await self._setup_fastapi()
        await self.api_manager.connect_all()

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)

        logger.info(f"🚀 Unified Dashboard running on http://{self.host}:{self.port}")
        await server.serve()

    def _get_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control ULTIMATE v5.2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.7.0/dist/confetti.browser.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6; }
        .gradient-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card { background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 20px; }
        .xp-badge { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); }
        .level-badge { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); }
        .connected { background: #dcfce7; color: #166534; }
        .error { background: #fee2e2; color: #991b1b; }
        .no-key { background: #fef3c7; color: #92400e; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
        .toast { animation: fadeIn 0.3s ease; }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="gradient-header text-white p-6 text-center">
        <h1 class="text-3xl font-bold">🎮 Mission Control ULTIMATE v5.2</h1>
        <p class="mt-2 opacity-90">
            <span id="ws-status">🔴 Disconnected</span> |
            Status: <span id="status">IDLE</span>
        </p>
    </div>

    <!-- Loading -->
    <div id="loading" class="text-center p-8">
        <i class="fas fa-spinner fa-spin text-4xl text-blue-500"></i>
        <p class="mt-4 text-gray-600">Loading dashboard...</p>
    </div>

    <!-- Content -->
    <div id="content" class="hidden">
        <!-- Gamification Bar -->
        <div class="bg-white border-b p-4 flex flex-wrap justify-center gap-4">
            <div class="xp-badge text-white px-4 py-2 rounded-full flex items-center gap-2">
                <i class="fas fa-bolt"></i>
                <span id="xp-text">0 XP</span>
                <div class="w-20 h-2 bg-white/30 rounded-full overflow-hidden">
                    <div id="xp-bar" class="h-full bg-white w-0 transition-all"></div>
                </div>
            </div>
            <div class="level-badge text-white px-4 py-2 rounded-full">
                <i class="fas fa-trophy"></i> Level <span id="level">1</span>
            </div>
            <div class="bg-green-100 text-green-700 px-4 py-2 rounded-full">
                <i class="fas fa-star"></i> <span id="achievements">0</span> Achievements
            </div>
            <div class="bg-orange-100 text-orange-700 px-4 py-2 rounded-full">
                <i class="fas fa-fire"></i> Streak: <span id="streak">0</span> days
            </div>
        </div>

        <!-- Main Content -->
        <div class="p-6 max-w-7xl mx-auto space-y-6">
            <!-- Stats Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="card">
                    <p class="text-gray-500 text-sm">Budget</p>
                    <p class="text-2xl font-bold">$<span id="spent">0</span> / $<span id="budget">0</span></p>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                        <div id="budget-bar" class="bg-blue-500 h-2 rounded-full w-0 transition-all"></div>
                    </div>
                </div>
                <div class="card">
                    <p class="text-gray-500 text-sm">Progress</p>
                    <p class="text-2xl font-bold"><span id="completed">0</span> / <span id="total">0</span> tasks</p>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                        <div id="progress-bar" class="bg-green-500 h-2 rounded-full w-0 transition-all"></div>
                    </div>
                </div>
                <div class="card">
                    <p class="text-gray-500 text-sm">Elapsed Time</p>
                    <p class="text-2xl font-bold"><i class="far fa-clock"></i> <span id="elapsed">0s</span></p>
                    <p class="text-sm text-gray-400 mt-1">ETA: <span id="eta">-</span></p>
                </div>
                <div class="card">
                    <p class="text-gray-500 text-sm">API Calls</p>
                    <p class="text-2xl font-bold"><i class="fas fa-plug"></i> <span id="api-calls">0</span></p>
                    <p class="text-sm text-gray-400 mt-1"><span id="tokens">0</span> tokens</p>
                </div>
            </div>

            <!-- API Connections -->
            <div class="card">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-lg font-bold"><i class="fas fa-network-wired mr-2"></i>API Connections (Auto-Connected)</h2>
                    <button onclick="reconnect()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                        <i class="fas fa-sync"></i> Reconnect
                    </button>
                </div>
                <div id="connections" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    <!-- Filled by JS -->
                </div>
            </div>

            <!-- Project Info -->
            <div class="card">
                <h2 class="text-lg font-bold mb-2"><i class="fas fa-project-diagram mr-2"></i>Project</h2>
                <p id="project-name" class="font-semibold">-</p>
                <p id="project-desc" class="text-gray-600 text-sm mt-1">-</p>
            </div>
        </div>
    </div>

    <!-- Toasts -->
    <div id="toasts" class="fixed top-4 right-4 z-50 space-y-2"></div>

    <script>
        let ws = null;
        let state = null;

        // Connect WebSocket
        function connectWS() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('🟢 WebSocket connected');
                document.getElementById('ws-status').innerHTML = '🟢 Connected';
                showToast('Connected', 'Real-time updates active', 'success');
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'init' || msg.type === 'state') {
                    updateUI(msg.data);
                }
            };

            ws.onclose = () => {
                console.log('🔴 WebSocket disconnected');
                document.getElementById('ws-status').innerHTML = '🔴 Disconnected';
                setTimeout(connectWS, 3000);
            };
        }

        // Update UI
        function updateUI(data) {
            state = data;

            // Hide loading
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('content').classList.remove('hidden');

            // Status
            document.getElementById('status').textContent = data.status.toUpperCase();

            // Gamification
            if (data.gamification) {
                document.getElementById('xp-text').textContent = data.gamification.xp + ' XP';
                document.getElementById('xp-bar').style.width = data.gamification.progress + '%';
                document.getElementById('level').textContent = data.gamification.level;
                document.getElementById('achievements').textContent = data.gamification.achievements.length;
                document.getElementById('streak').textContent = data.gamification.streak;
            }

            // Project
            if (data.project) {
                document.getElementById('spent').textContent = data.project.spent_usd;
                document.getElementById('budget').textContent = data.project.budget_usd;
                document.getElementById('budget-bar').style.width = data.project.budget_percent + '%';
                document.getElementById('budget-bar').className =
                    'h-2 rounded-full transition-all ' +
                    (data.project.budget_percent > 90 ? 'bg-red-500' : 'bg-blue-500');
            }

            // Progress
            if (data.progress) {
                document.getElementById('completed').textContent = data.progress.completed_tasks;
                document.getElementById('total').textContent = data.progress.total_tasks;
                document.getElementById('progress-bar').style.width = data.progress.percent + '%';
            }

            // Time
            if (data.time) {
                document.getElementById('elapsed').textContent = data.time.elapsed;
                document.getElementById('eta').textContent = data.time.eta;
            }

            // Stats
            if (data.stats) {
                document.getElementById('api-calls').textContent = data.stats.api_calls;
                document.getElementById('tokens').textContent = data.stats.tokens.toLocaleString();
            }

            // Project info
            if (data.project) {
                document.getElementById('project-name').textContent = data.project.name || 'No active project';
                document.getElementById('project-desc').textContent = data.project.description || '-';
            }

            // API Connections
            if (data.api_connections) {
                updateConnections(data.api_connections);
            }
        }

        function updateConnections(connections) {
            const container = document.getElementById('connections');
            container.innerHTML = connections.map(conn => `
                <div class="p-3 rounded-lg border ${getStatusClass(conn.status)}">
                    <div class="flex items-center gap-2">
                        <div class="w-2 h-2 rounded-full ${getStatusDot(conn.status)}"></div>
                        <span class="font-semibold text-sm">${conn.provider.toUpperCase()}</span>
                    </div>
                    <span class="text-xs capitalize">${conn.status}</span>
                    ${conn.models_available.length ? `<span class="text-xs text-gray-500">${conn.models_available.slice(0, 2).join(', ')}</span>` : ''}
                </div>
            `).join('');
        }

        function getStatusClass(status) {
            if (status === 'connected') return 'connected';
            if (status === 'no_key') return 'no-key';
            return 'error';
        }

        function getStatusDot(status) {
            if (status === 'connected') return 'bg-green-500';
            if (status === 'no_key') return 'bg-yellow-500';
            return 'bg-red-500';
        }

        function showToast(title, message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast p-4 rounded-lg shadow-lg max-w-sm ${
                type === 'success' ? 'bg-green-100 text-green-800' :
                type === 'error' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'
            }`;
            toast.innerHTML = `<strong>${title}</strong><br>${message}`;
            document.getElementById('toasts').appendChild(toast);
            setTimeout(() => toast.remove(), 5000);
        }

        function reconnect() {
            fetch('/api/connect', { method: 'POST' })
                .then(() => showToast('Reconnecting', 'Attempting to reconnect...', 'info'));
        }

        // Initialize
        fetch('/api/state')
            .then(r => r.json())
            .then(data => updateUI(data))
            .catch(err => {
                console.error('Failed to load state:', err);
                document.getElementById('loading').innerHTML = '<p class="text-red-500">Failed to load dashboard</p>';
            });

        connectWS();
    </script>
</body>
</html>
'''


def run_unified_dashboard(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):

    url = f"http://{host}:{port}"

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🚀 MISSION CONTROL ULTIMATE v5.2                               ║
╠══════════════════════════════════════════════════════════════════╣
║  🌐 URL: {url:<50}║
╠══════════════════════════════════════════════════════════════════╣
║  ✨ Features:                                                    ║
║     • 🎮 Gamification (XP, Levels, Achievements)                ║
║     • 📊 Real-time WebSocket Updates                            ║
║     • 🔔 Toast Notifications                                    ║
║     • 🎊 Confetti Celebrations                                  ║
║     • 🔌 Auto-API Connection on Startup                         ║
╚══════════════════════════════════════════════════════════════════╝
""")

    if open_browser:
        webbrowser.open(url)

    dashboard = UnifiedDashboardServer(host=host, port=port)
    asyncio.run(dashboard.run())


if __name__ == "__main__":
    run_unified_dashboard()
