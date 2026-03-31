"""
Mission Control Dashboard View — Plugin Implementation
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Mission Control dashboard as a plugin view for the unified dashboard core.
Consolidates features from dashboard_mission_control.py and dashboard_live.py
"""

from __future__ import annotations

from typing import Any

from .dashboard_core_core import DashboardView, ViewContext


class MissionControlView(DashboardView):
    """
    Mission Control dashboard view with gamification and real-time updates.
    """

    name = "mission-control"
    display_name = "🎮 Mission Control"
    version = "6.0.0"

    def get_assets(self) -> dict[str, list]:
        return {
            "css": [
                "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
            "js": [
                "https://cdn.jsdelivr.net/npm/chart.js",
                "https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js",
            ],
        }

    async def render(self, context: ViewContext) -> str:
        """Render Mission Control dashboard."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control v6.0</title>
    {self._render_head()}
</head>
<body class="bg-gray-900 text-white">
    {self._render_header(context)}
    {self._render_main(context)}
    {self._render_scripts()}
</body>
</html>
"""

    def _render_head(self) -> str:
        """Render head section with styles."""
        return """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root { --primary: #3b82f6; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; }
        .glass { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); }
        .glow { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .task-card { transition: all 0.3s ease; }
        .task-card:hover { transform: translateY(-2px); box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
        .metric-card { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); }
        .xp-bar { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
    </style>
        """

    def _render_header(self, context: ViewContext) -> str:
        """Render header with gamification elements."""
        budget_pct = (
            (context.budget.get("spent", 0) / context.budget.get("max", 1)) * 100
            if context.budget.get("max")
            else 0
        )

        return f"""
    <header class="glass fixed top-0 w-full z-50 border-b border-gray-700">
        <div class="container mx-auto px-6 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <i class="fas fa-rocket text-3xl text-blue-400"></i>
                <div>
                    <h1 class="text-2xl font-bold">Mission Control <span class="text-blue-400">v6.0</span></h1>
                    <p class="text-sm text-gray-400">Project: {context.project_id or 'No active project'}</p>
                </div>
            </div>

            <!-- XP & Level -->
            <div class="flex items-center space-x-6">
                <div class="text-center">
                    <div class="text-2xl font-bold text-yellow-400">Level <span id="level">1</span></div>
                    <div class="text-xs text-gray-400">Architect</div>
                </div>
                <div class="w-32">
                    <div class="flex justify-between text-xs mb-1">
                        <span>XP</span>
                        <span id="xp-text">0/100</span>
                    </div>
                    <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div id="xp-bar" class="xp-bar h-full rounded-full transition-all" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Budget -->
            <div class="flex items-center space-x-4">
                <div class="text-right">
                    <div class="text-sm text-gray-400">Budget</div>
                    <div class="text-xl font-bold ${'text-red-400' if budget_pct > 80 else 'text-green-400'}">
                        ${context.budget.get('spent', 0):.2f} / ${context.budget.get('max', 0):.2f}
                    </div>
                </div>
                <div class="w-16 h-16 relative">
                    <svg class="w-full h-full transform -rotate-90">
                        <circle cx="32" cy="32" r="28" stroke="#374151" stroke-width="4" fill="none"/>
                        <circle cx="32" cy="32" r="28" stroke="{'#ef4444' if budget_pct > 80 else '#10b981'}"
                                stroke-width="4" fill="none" stroke-dasharray="{175.9 * budget_pct / 100} 175.9"/>
                    </svg>
                    <span class="absolute inset-0 flex items-center justify-center text-sm font-bold">
                        {budget_pct:.0f}%
                    </span>
                </div>
            </div>
        </div>
    </header>
        """

    def _render_main(self, context: ViewContext) -> str:
        """Render main content area."""
        return f"""
    <main class="container mx-auto px-6 pt-24 pb-6">
        <!-- Metrics Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            {self._render_metric_card("Tasks", len(context.active_tasks), "fa-tasks", "blue")}
            {self._render_metric_card("Models", len(context.model_status), "fa-robot", "purple")}
            {self._render_metric_card("Score", f"{context.metrics.get('quality_score', 0):.0%}", "fa-star", "yellow")}
            {self._render_metric_card("Uptime", "∞", "fa-clock", "green")}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Active Tasks -->
            <div class="lg:col-span-2 space-y-4">
                <h2 class="text-xl font-bold flex items-center">
                    <i class="fas fa-list-check mr-2 text-blue-400"></i>
                    Active Tasks
                </h2>
                <div id="tasks-container" class="space-y-3">
                    {self._render_tasks(context)}
                </div>
            </div>

            <!-- Sidebar -->
            <div class="space-y-6">
                <!-- Model Status -->
                <div class="glass rounded-xl p-4">
                    <h3 class="font-bold mb-3 flex items-center">
                        <i class="fas fa-server mr-2 text-purple-400"></i>
                        Model Status
                    </h3>
                    <div id="models-container" class="space-y-2">
                        {self._render_models(context)}
                    </div>
                </div>

                <!-- Achievements -->
                <div class="glass rounded-xl p-4">
                    <h3 class="font-bold mb-3 flex items-center">
                        <i class="fas fa-trophy mr-2 text-yellow-400"></i>
                        Achievements
                    </h3>
                    <div id="achievements" class="grid grid-cols-4 gap-2">
                        {self._render_achievements()}
                    </div>
                </div>

                <!-- Events Log -->
                <div class="glass rounded-xl p-4">
                    <h3 class="font-bold mb-3 flex items-center">
                        <i class="fas fa-terminal mr-2 text-green-400"></i>
                        Event Log
                    </h3>
                    <div id="event-log" class="h-48 overflow-y-auto text-xs font-mono space-y-1">
                        {self._render_events(context)}
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Toast Container -->
    <div id="toast-container" class="fixed bottom-4 right-4 z-50 space-y-2"></div>
        """

    def _render_metric_card(self, label: str, value: Any, icon: str, color: str) -> str:
        """Render a metric card."""
        colors = {
            "blue": "from-blue-500 to-blue-600",
            "purple": "from-purple-500 to-purple-600",
            "yellow": "from-yellow-500 to-yellow-600",
            "green": "from-green-500 to-green-600",
        }
        return f"""
        <div class="metric-card glass rounded-xl p-4 glow">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-gray-400 text-sm">{label}</p>
                    <p class="text-3xl font-bold">{value}</p>
                </div>
                <div class="w-12 h-12 rounded-lg bg-gradient-to-br {colors.get(color)} flex items-center justify-center">
                    <i class="fas {icon} text-xl"></i>
                </div>
            </div>
        </div>
        """

    def _render_tasks(self, context: ViewContext) -> str:
        """Render task list."""
        if not context.active_tasks:
            return '<p class="text-gray-500 text-center py-8">No active tasks</p>'

        tasks_html = ""
        for task in context.active_tasks:
            status_colors = {
                "pending": "text-gray-400",
                "running": "text-blue-400 pulse",
                "completed": "text-green-400",
                "failed": "text-red-400",
            }
            status = task.get("status", "pending")
            color = status_colors.get(status, "text-gray-400")

            tasks_html += f"""
            <div class="task-card glass rounded-lg p-4 flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <i class="fas fa-circle {color}"></i>
                    <div>
                        <p class="font-medium">{task.get('id', 'Unknown')}</p>
                        <p class="text-sm text-gray-400">{task.get('type', 'task')}</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-right">
                        <p class="text-sm text-gray-400">Score</p>
                        <p class="font-bold">{task.get('score', 0):.2f}</p>
                    </div>
                    <div class="text-right">
                        <p class="text-sm text-gray-400">Model</p>
                        <p class="text-xs">{task.get('model', 'N/A')}</p>
                    </div>
                </div>
            </div>
            """
        return tasks_html

    def _render_models(self, context: ViewContext) -> str:
        """Render model status."""
        if not context.model_status:
            return '<p class="text-gray-500 text-sm">No model data</p>'

        models_html = ""
        for name, status in context.model_status.items():
            is_healthy = status.get("healthy", False)
            color = "text-green-400" if is_healthy else "text-red-400"
            icon = "fa-check-circle" if is_healthy else "fa-times-circle"

            models_html += f"""
            <div class="flex items-center justify-between text-sm">
                <span class="text-gray-300">{name}</span>
                <span class="{color}"><i class="fas {icon}"></i></span>
            </div>
            """
        return models_html

    def _render_achievements(self) -> str:
        """Render achievement badges."""
        achievements = [
            ("Task Master", "fa-bullseye", "text-gray-600"),
            ("Speed Demon", "fa-bolt", "text-gray-600"),
            ("Perfectionist", "fa-gem", "text-gray-600"),
            ("Budget Master", "fa-coins", "text-gray-600"),
        ]

        html = ""
        for name, icon, color in achievements:
            html += f"""
            <div class="achievement locked opacity-50" title="{name}">
                <div class="w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center {color}">
                    <i class="fas {icon}"></i>
                </div>
            </div>
            """
        return html

    def _render_events(self, context: ViewContext) -> str:
        """Render event log."""
        if not context.events:
            return '<p class="text-gray-500">Waiting for events...</p>'

        html = ""
        for event in context.events[-20:]:  # Last 20 events
            timestamp = event.get("timestamp", "")
            type_ = event.get("type", "info")
            html += f"""
            <div class="text-gray-400">
                <span class="text-gray-600">[{timestamp}]</span>
                <span class="text-blue-400">{type_}</span>
            </div>
            """
        return html

    def _render_scripts(self) -> str:
        """Render JavaScript for interactivity."""
        return """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
    <script>
        // WebSocket Connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);

        ws.onopen = () => {
            console.log('Connected to Mission Control');
            ws.send('subscribe');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleEvent(data);
        };

        ws.onclose = () => {
            console.log('Disconnected from Mission Control');
        };

        // Event Handlers
        function handleEvent(data) {
            switch(data.type) {
                case 'TASK_STARTED':
                    showToast('🚀 Task Started', data.data.task_id);
                    break;
                case 'TASK_COMPLETED':
                    showToast('✅ Task Complete', data.data.task_id, 'success');
                    addXP(10);
                    break;
                case 'TASK_FAILED':
                    showToast('❌ Task Failed', data.data.task_id, 'error');
                    break;
                case 'PROJECT_COMPLETED':
                    triggerConfetti();
                    showToast('🎉 Project Complete!', '', 'success');
                    unlockAchievement('Task Master');
                    break;
                case 'BUDGET_WARNING':
                    showToast('⚠️ Budget Warning', '80% budget used', 'warning');
                    break;
            }
        }

        // Toast Notifications
        function showToast(title, message, type = 'info') {
            const container = document.getElementById('toast-container');
            const colors = {
                info: 'border-blue-500 bg-blue-900',
                success: 'border-green-500 bg-green-900',
                warning: 'border-yellow-500 bg-yellow-900',
                error: 'border-red-500 bg-red-900',
            };

            const toast = document.createElement('div');
            toast.className = `p-4 rounded-lg border-l-4 ${colors[type]} text-white shadow-lg transform transition-all`;
            toast.innerHTML = `
                <div class="font-bold">${title}</div>
                ${message ? `<div class="text-sm">${message}</div>` : ''}
            `;

            container.appendChild(toast);
            setTimeout(() => toast.remove(), 5000);
        }

        // XP System
        let xp = 0;
        let level = 1;

        function addXP(amount) {
            xp += amount;
            const needed = level * 100;

            if (xp >= needed) {
                xp -= needed;
                level++;
                showToast('🆙 Level Up!', `You are now level ${level}`, 'success');
                document.getElementById('level').textContent = level;
            }

            document.getElementById('xp-text').textContent = `${xp}/${needed}`;
            document.getElementById('xp-bar').style.width = `${(xp / needed) * 100}%`;
        }

        // Achievements
        function unlockAchievement(name) {
            const achievements = document.querySelectorAll('.achievement');
            achievements.forEach(a => {
                if (a.title === name) {
                    a.classList.remove('locked', 'opacity-50');
                    a.querySelector('div').classList.add('bg-yellow-600', 'text-white');
                    showToast('🏆 Achievement Unlocked!', name, 'success');
                }
            });
        }

        // Confetti Celebration
        function triggerConfetti() {
            confetti({
                particleCount: 150,
                spread: 70,
                origin: { y: 0.6 },
                colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
            });
        }
    </script>
        """

    def handle_event(self, event: Any) -> dict[str, Any] | None:
        """Transform events for Mission Control view."""
        # Add gamification data to events
        if hasattr(event, "event_type"):
            return {
                "gamified": True,
                "xp_reward": self._get_xp_reward(event.event_type.value),
            }
        return None

    def _get_xp_reward(self, event_type: str) -> int:
        """Calculate XP reward for event type."""
        rewards = {
            "TASK_COMPLETED": 10,
            "PROJECT_COMPLETED": 100,
            "QUALITY_GATE_PASSED": 25,
            "BUDGET_MASTER": 50,
        }
        return rewards.get(event_type, 5)


# Convenience function for registration
def create_view() -> MissionControlView:
    """Create and return a Mission Control view instance."""
    return MissionControlView()
