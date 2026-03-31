"""
Live Gamified Dashboard v4.0
============================
Addictive, real-time dashboard with:
- WebSocket live streaming (no polling)
- Toast notifications for all events
- Gamification (achievements, streaks, levels)
- Confetti celebration on project completion
- Sound notifications
- Live task progress
- Test execution monitoring
- Visual effects and animations

Usage:
    from orchestrator.dashboard_live import LiveDashboard
    dashboard = LiveDashboard()
    dashboard.start()
"""

from __future__ import annotations

import asyncio
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class LiveTask:
    """Live task with real-time updates."""

    task_id: str
    task_type: str
    prompt: str
    status: str = "pending"  # pending, running, completed, failed
    iteration: int = 0
    max_iterations: int = 3
    score: float = 0.0
    model_used: str = ""
    start_time: float = 0.0
    elapsed_seconds: float = 0.0
    progress_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestExecution:
    """Test execution status."""

    test_file: str
    status: str = "pending"  # pending, running, passed, failed
    progress: float = 0.0
    output: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Achievement:
    """Gamification achievement."""

    id: str
    title: str
    description: str
    icon: str
    unlocked_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "unlocked": self.unlocked_at is not None,
            "unlocked_at": self.unlocked_at,
        }


@dataclass
class DashboardState:
    """Complete dashboard state for live updates."""

    # Project
    project_id: str = ""
    project_description: str = ""
    project_status: str = "idle"  # idle, running, completed, failed
    project_progress: float = 0.0

    # Tasks
    tasks: list[dict] = field(default_factory=list)
    active_task: dict | None = None
    completed_tasks: int = 0
    total_tasks: int = 0

    # Tests
    tests: list[dict] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    test_coverage: float = 0.0

    # Budget & Metrics
    budget_used: float = 0.0
    budget_total: float = 0.0
    total_calls: int = 0
    total_cost: float = 0.0

    # Gamification
    level: int = 1
    xp: int = 0
    xp_to_next_level: int = 100
    streak: int = 0
    achievements: list[dict] = field(default_factory=list)

    # Events
    recent_events: list[dict] = field(default_factory=list)

    # Timestamp
    last_update: float = field(default_factory=time.time)


class LiveDashboardServer:
    """
    Live dashboard with WebSocket support and gamification.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.state = DashboardState()
        self.connections: set[Any] = set()
        self.event_queue: deque = deque(maxlen=100)
        self.achievements_db = self._init_achievements()
        self._setup_app()

    def _init_achievements(self) -> dict[str, Achievement]:
        """Initialize achievement database."""
        return {
            "first_task": Achievement(
                id="first_task",
                title="Task Master",
                description="Complete your first task",
                icon="🎯",
            ),
            "speed_demon": Achievement(
                id="speed_demon",
                title="Speed Demon",
                description="Complete a task in under 30 seconds",
                icon="⚡",
            ),
            "perfect_score": Achievement(
                id="perfect_score",
                title="Perfectionist",
                description="Achieve 100% score on a task",
                icon="💯",
            ),
            "budget_master": Achievement(
                id="budget_master",
                title="Budget Master",
                description="Complete project under 50% of budget",
                icon="💰",
            ),
            "test_champion": Achievement(
                id="test_champion",
                title="Test Champion",
                description="Achieve 100% test coverage",
                icon="🧪",
            ),
            "streak_5": Achievement(
                id="streak_5",
                title="On Fire",
                description="Complete 5 tasks in a row without failure",
                icon="🔥",
            ),
            "architect": Achievement(
                id="architect",
                title="Architect",
                description="Complete a project with microservices architecture",
                icon="🏗️",
            ),
        }

    def _setup_app(self):
        """Setup FastAPI with WebSocket support."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import HTMLResponse

            self.app = FastAPI(title="Mission Control Live")

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )

            @self.app.get("/")
            async def dashboard():
                return HTMLResponse(content=self._get_html())

            @self.app.get("/api/state")
            async def get_state():
                """Get current state."""
                return self.state.to_dict()

            @self.app.post("/api/project/start")
            async def project_start(data: dict[str, Any]):
                """Start a new project."""
                self.state.project_id = data.get("project_id", str(uuid.uuid4())[:8])
                self.state.project_description = data.get("description", "")
                self.state.project_status = "running"
                self.state.total_tasks = data.get("total_tasks", 0)
                self.state.budget_total = data.get("budget", 0)
                self.state.start_time = time.time()

                await self._broadcast(
                    {
                        "type": "project_started",
                        "data": self.state.to_dict(),
                    }
                )

                return {"status": "ok"}

            @self.app.post("/api/task/update")
            async def task_update(data: dict[str, Any]):
                """Update task status."""
                task_id = data.get("task_id")

                # Find or create task
                task = None
                for t in self.state.tasks:
                    if t["task_id"] == task_id:
                        task = t
                        break

                if task is None:
                    task = LiveTask(
                        task_id=task_id,
                        task_type=data.get("task_type", "unknown"),
                        prompt=data.get("prompt", ""),
                        start_time=time.time(),
                    )
                    self.state.tasks.append(task.to_dict())
                else:
                    # Update existing
                    task["status"] = data.get("status", task.get("status"))
                    task["iteration"] = data.get("iteration", task.get("iteration", 0))
                    task["score"] = data.get("score", task.get("score", 0))
                    task["model_used"] = data.get("model_used", task.get("model_used", ""))
                    task["progress_percent"] = data.get("progress", 0)

                    if task["status"] == "completed":
                        task["elapsed_seconds"] = time.time() - task.get("start_time", time.time())
                        self.state.completed_tasks += 1
                        self._add_xp(25)  # XP for completing task

                        # Check achievements
                        if task["score"] >= 1.0:
                            await self._unlock_achievement("perfect_score")
                        if task["elapsed_seconds"] < 30:
                            await self._unlock_achievement("speed_demon")

                    # Update active task
                    if task["status"] == "running":
                        self.state.active_task = task

                # Update project progress
                if self.state.total_tasks > 0:
                    self.state.project_progress = (
                        self.state.completed_tasks / self.state.total_tasks
                    ) * 100

                # Broadcast update
                await self._broadcast(
                    {
                        "type": "task_update",
                        "task": task,
                        "progress": self.state.project_progress,
                    }
                )

                return {"status": "ok"}

            @self.app.post("/api/project/complete")
            async def project_complete(data: dict[str, Any]):
                """Mark project as complete."""
                self.state.project_status = "completed"
                self.state.project_progress = 100.0

                # Big XP bonus
                self._add_xp(100)

                # Check achievements
                if self.state.budget_used < self.state.budget_total * 0.5:
                    await self._unlock_achievement("budget_master")

                await self._broadcast(
                    {
                        "type": "project_completed",
                        "data": self.state.to_dict(),
                        "celebration": True,
                    }
                )

                return {"status": "ok"}

            @self.app.post("/api/test/update")
            async def test_update(data: dict[str, Any]):
                """Update test execution."""
                test_file = data.get("test_file")

                test = None
                for t in self.state.tests:
                    if t["test_file"] == test_file:
                        test = t
                        break

                if test is None:
                    test = {"test_file": test_file, "status": "running", "progress": 0}
                    self.state.tests.append(test)

                test["status"] = data.get("status", "running")
                test["progress"] = data.get("progress", 0)
                test["output"] = data.get("output", "")

                if test["status"] == "passed":
                    self.state.tests_passed += 1
                    self._add_xp(10)
                elif test["status"] == "failed":
                    self.state.tests_failed += 1

                await self._broadcast(
                    {
                        "type": "test_update",
                        "test": test,
                    }
                )

                return {"status": "ok"}

            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket for real-time updates."""
                await websocket.accept()
                self.connections.add(websocket)

                # Send initial state
                await websocket.send_json(
                    {
                        "type": "init",
                        "data": self.state.to_dict(),
                    }
                )

                try:
                    while True:
                        # Keep connection alive and handle pings
                        data = await websocket.receive_text()
                        if data == "ping":
                            await websocket.send_text("pong")
                except WebSocketDisconnect:
                    self.connections.discard(websocket)

        except ImportError as e:
            logger.error(f"Failed to setup app: {e}")
            raise

    async def _broadcast(self, message: dict[str, Any]):
        """Broadcast message to all connected clients."""
        dead_connections = set()

        for conn in self.connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead_connections.add(conn)

        # Clean up dead connections
        self.connections -= dead_connections

    def _add_xp(self, amount: int):
        """Add XP and check for level up."""
        self.state.xp += amount

        # Level up formula
        while self.state.xp >= self.state.xp_to_next_level:
            self.state.xp -= self.state.xp_to_next_level
            self.state.level += 1
            self.state.xp_to_next_level = int(self.state.xp_to_next_level * 1.5)

            # Broadcast level up
            asyncio.create_task(
                self._broadcast(
                    {
                        "type": "level_up",
                        "level": self.state.level,
                    }
                )
            )

    async def _unlock_achievement(self, achievement_id: str):
        """Unlock an achievement."""
        achievement = self.achievements_db.get(achievement_id)
        if achievement and achievement.unlocked_at is None:
            achievement.unlocked_at = time.time()
            self.state.achievements.append(achievement.to_dict())

            # Bonus XP
            self._add_xp(50)

            await self._broadcast(
                {
                    "type": "achievement_unlocked",
                    "achievement": achievement.to_dict(),
                }
            )

    def _get_html(self) -> str:
        """Generate HTML with gamified UI."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control Live | v4.0</title>

    <!-- Ant Design -->
    <link rel="stylesheet" href="https://unpkg.com/antd@5.12.0/dist/reset.css">
    <link rel="stylesheet" href="https://unpkg.com/antd@5.12.0/dist/antd.min.css">

    <!-- React -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

    <!-- Ant Design -->
    <script src="https://unpkg.com/antd@5.12.0/dist/antd.min.js"></script>
    <script src="https://unpkg.com/@ant-design/icons@5.2.6/dist/index.umd.min.js"></script>

    <!-- Canvas Confetti -->
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>

    <style>
        :root {
            --primary: #722ed1;
            --secondary: #eb2f96;
            --success: #52c41a;
            --warning: #faad14;
            --error: #ff4d4f;
            --gold: #ffd700;
        }

        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            margin: 0;
            min-height: 100vh;
        }

        .glow-text {
            text-shadow: 0 0 10px var(--primary), 0 0 20px var(--primary);
        }

        .level-badge {
            background: linear-gradient(135deg, var(--gold), #ff6b6b);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 20px;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .achievement-popup {
            animation: slideIn 0.5s ease-out;
            background: linear-gradient(135deg, #ffd700, #ff6b35);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
        }

        .task-running {
            border-left: 4px solid var(--primary);
            animation: pulse-border 2s infinite;
        }

        @keyframes pulse-border {
            0%, 100% { border-left-color: var(--primary); }
            50% { border-left-color: var(--secondary); }
        }

        .xp-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        }

        .xp-fill {
            background: linear-gradient(90deg, var(--gold), #ff6b6b);
            height: 100%;
            transition: width 0.5s ease;
        }

        .streak-flame {
            animation: flicker 0.5s infinite alternate;
        }

        @keyframes flicker {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0.8; transform: scale(1.1); }
        }

        .project-complete-banner {
            background: linear-gradient(135deg, #52c41a, #95de64);
            animation: celebrate 1s ease-out;
        }

        @keyframes celebrate {
            0% { transform: scale(0); opacity: 0; }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }

        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #52c41a;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef, useCallback } = React;
        const {
            Layout, Card, Row, Col, Statistic, Progress, Tag, Timeline,
            Badge, Descriptions, List, Avatar, Typography, Space, Divider,
            Alert, Button, Table, notification, Progress: AntProgress,
            FloatButton, Tooltip, Modal
        } = antd;
        const {
            DashboardOutlined, TrophyOutlined, FireOutlined, ThunderboltOutlined,
            CheckCircleOutlined, CloseCircleOutlined, SyncOutlined, ExperimentOutlined,
            CodeOutlined, CheckSquareOutlined, PlayCircleOutlined, CrownOutlined,
            StarOutlined, GiftOutlined, NotificationOutlined
        } = icons;

        const { Header, Content, Footer } = Layout;
        const { Title, Text } = Typography;

        // Sound effects (using AudioContext)
        const playSound = (type) => {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();

                osc.connect(gain);
                gain.connect(ctx.destination);

                if (type === 'achievement') {
                    osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
                    osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.1); // E5
                    osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.2); // G5
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                    osc.start(ctx.currentTime);
                    osc.stop(ctx.currentTime + 0.5);
                } else if (type === 'complete') {
                    osc.frequency.setValueAtTime(523.25, ctx.currentTime);
                    osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.1);
                    osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.2);
                    osc.frequency.setValueAtTime(1046.50, ctx.currentTime + 0.3);
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 1);
                    osc.start(ctx.currentTime);
                    osc.stop(ctx.currentTime + 1);
                }
            } catch (e) {
                console.log("Audio not supported");
            }
        };

        // Confetti celebration
        const celebrate = () => {
            const duration = 3000;
            const end = Date.now() + duration;

            const frame = () => {
                confetti({
                    particleCount: 5,
                    angle: 60,
                    spread: 55,
                    origin: { x: 0 },
                    colors: ['#ffd700', '#ff6b6b', '#4ecdc4', '#45b7d1']
                });
                confetti({
                    particleCount: 5,
                    angle: 120,
                    spread: 55,
                    origin: { x: 1 },
                    colors: ['#ffd700', '#ff6b6b', '#4ecdc4', '#45b7d1']
                });

                if (Date.now() < end) {
                    requestAnimationFrame(frame);
                }
            };
            frame();
        };

        // Toast notification
        const showToast = (type, title, description, icon) => {
            notification.open({
                message: title,
                description: description,
                icon: <span style={{ fontSize: 24 }}>{icon}</span>,
                placement: 'topRight',
                duration: 5,
                style: {
                    background: type === 'achievement'
                        ? 'linear-gradient(135deg, #ffd700, #ff6b35)'
                        : type === 'complete'
                        ? 'linear-gradient(135deg, #52c41a, #95de64)'
                        : '#fff',
                    borderRadius: 12,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                },
            });
        };

        // Gamification Panel
        function GamificationPanel({ level, xp, xpToNext, streak, achievements }) {
            const progress = (xp / xpToNext) * 100;

            return (
                <Card style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}>
                    <Row align="middle" gutter={16}>
                        <Col>
                            <div className="level-badge">{level}</div>
                        </Col>
                        <Col flex="auto">
                            <div style={{ marginBottom: 8 }}>
                                <Text style={{ color: '#ffd700', fontWeight: 'bold' }}>
                                    Level {level}
                                </Text>
                                <Text style={{ color: '#888', marginLeft: 16 }}>
                                    {xp} / {xpToNext} XP
                                </Text>
                            </div>
                            <div className="xp-bar">
                                <div className="xp-fill" style={{ width: `${progress}%` }} />
                            </div>
                        </Col>
                        <Col>
                            <Tooltip title="Current streak">
                                <Space className="streak-flame">
                                    <FireOutlined style={{ color: '#ff6b35', fontSize: 24 }} />
                                    <Text style={{ color: '#ff6b35', fontSize: 18, fontWeight: 'bold' }}>
                                        {streak}
                                    </Text>
                                </Space>
                            </Tooltip>
                        </Col>
                        <Col>
                            <Tooltip title="Achievements unlocked">
                                <Space>
                                    <TrophyOutlined style={{ color: '#ffd700', fontSize: 24 }} />
                                    <Text style={{ color: '#ffd700', fontSize: 18, fontWeight: 'bold' }}>
                                        {achievements.length}
                                    </Text>
                                </Space>
                            </Tooltip>
                        </Col>
                    </Row>

                    {achievements.length > 0 && (
                        <>
                            <Divider style={{ borderColor: 'rgba(255,255,255,0.1)' }} />
                            <Space wrap>
                                {achievements.slice(-3).map((a, i) => (
                                    <Tooltip key={i} title={a.description}>
                                        <Tag color="gold" style={{ fontSize: 16, padding: '4px 12px' }}>
                                            {a.icon} {a.title}
                                        </Tag>
                                    </Tooltip>
                                ))}
                            </Space>
                        </>
                    )}
                </Card>
            );
        }

        // Live Task Card
        function LiveTaskCard({ task }) {
            if (!task) {
                return (
                    <Card style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}>
                        <Text style={{ color: '#888' }}>No active task...</Text>
                    </Card>
                );
            }

            const isRunning = task.status === 'running';

            return (
                <Card
                    className={isRunning ? 'task-running' : ''}
                    style={{
                        background: 'rgba(255,255,255,0.05)',
                        border: 'none',
                        borderLeft: isRunning ? '4px solid #722ed1' : 'none'
                    }}
                    title={
                        <Space>
                            {isRunning && <span className="live-indicator" />}
                            <Text style={{ color: '#fff' }}>Live Task</Text>
                            {isRunning && (
                                <Tag color="processing" icon={<SyncOutlined spin />}>RUNNING</Tag>
                            )}
                        </Space>
                    }
                >
                    <Title level={5} style={{ color: '#fff' }}>{task.task_id}</Title>
                    <Tag color="blue">{task.task_type}</Tag>

                    <div style={{ marginTop: 16 }}>
                        <Row justify="space-between">
                            <Text style={{ color: '#888' }}>Iteration {task.iteration} / {task.max_iterations}</Text>
                            <Text style={{ color: '#52c41a', fontWeight: 'bold' }}>
                                Score: {Math.round((task.score || 0) * 100)}%
                            </Text>
                        </Row>
                        <Progress
                            percent={Math.round((task.score || 0) * 100)}
                            status={task.score >= 0.8 ? 'success' : 'active'}
                            strokeColor={{ from: '#722ed1', to: '#eb2f96' }}
                        />
                    </div>

                    <Text style={{ color: '#888', fontSize: 12 }}>{task.model_used}</Text>
                </Card>
            );
        }

        // Test Monitor
        function TestMonitor({ tests }) {
            return (
                <Card
                    style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}
                    title={<><CheckSquareOutlined style={{ color: '#52c41a' }} /> Test Monitor</>}
                >
                    <List
                        dataSource={tests.slice(-5)}
                        renderItem={test => (
                            <List.Item>
                                <Space>
                                    {test.status === 'running' && <SyncOutlined spin style={{ color: '#722ed1' }} />}
                                    {test.status === 'passed' && <CheckCircleOutlined style={{ color: '#52c41a' }} />}
                                    {test.status === 'failed' && <CloseCircleOutlined style={{ color: '#ff4d4f' }} />}
                                    <Text style={{ color: '#fff' }}>{test.test_file}</Text>
                                    {test.status === 'running' && (
                                        <Progress percent={test.progress} size="small" style={{ width: 100 }} />
                                    )}
                                </Space>
                            </List.Item>
                        )}
                    />
                </Card>
            );
        }

        // Project Complete Modal
        function ProjectCompleteModal({ visible, onClose, stats }) {
            useEffect(() => {
                if (visible) {
                    celebrate();
                    playSound('complete');
                }
            }, [visible]);

            return (
                <Modal
                    visible={visible}
                    onCancel={onClose}
                    footer={[
                        <Button key="ok" type="primary" onClick={onClose}>
                            Awesome!
                        </Button>
                    ]}
                    width={600}
                    style={{ top: 50 }}
                >
                    <div style={{ textAlign: 'center', padding: '40px 0' }}>
                        <CrownOutlined style={{ fontSize: 80, color: '#ffd700' }} />
                        <Title level={2} style={{ marginTop: 24 }}>Project Complete!</Title>
                        <Text style={{ fontSize: 18 }}>You crushed it! 🎉</Text>

                        <Row gutter={24} style={{ marginTop: 40 }}>
                            <Col span={8}>
                                <Statistic
                                    title="Tasks"
                                    value={stats?.completed_tasks || 0}
                                    suffix={`/ ${stats?.total_tasks || 0}`}
                                />
                            </Col>
                            <Col span={8}>
                                <Statistic
                                    title="Tests Passed"
                                    value={stats?.tests_passed || 0}
                                    valueStyle={{ color: '#52c41a' }}
                                />
                            </Col>
                            <Col span={8}>
                                <Statistic
                                    title="XP Earned"
                                    value={100}
                                    prefix={<StarOutlined />}
                                    valueStyle={{ color: '#ffd700' }}
                                />
                            </Col>
                        </Row>
                    </div>
                </Modal>
            );
        }

        // Main App
        function App() {
            const [state, setState] = useState({
                project_status: 'idle',
                project_progress: 0,
                tasks: [],
                active_task: null,
                completed_tasks: 0,
                total_tasks: 0,
                tests: [],
                tests_passed: 0,
                level: 1,
                xp: 0,
                xp_to_next_level: 100,
                streak: 0,
                achievements: [],
            });
            const [connected, setConnected] = useState(false);
            const [showCompleteModal, setShowCompleteModal] = useState(false);
            const ws = useRef(null);

            useEffect(() => {
                // Connect WebSocket
                const connect = () => {
                    ws.current = new WebSocket(`ws://${window.location.host}/ws`);

                    ws.current.onopen = () => {
                        setConnected(true);
                        showToast('info', 'Connected', 'Live updates enabled', '🟢');
                    };

                    ws.current.onmessage = (event) => {
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'init') {
                            setState(msg.data);
                        } else if (msg.type === 'task_update') {
                            setState(prev => ({
                                ...prev,
                                tasks: prev.tasks.map(t =>
                                    t.task_id === msg.task.task_id ? msg.task : t
                                ),
                                active_task: msg.task.status === 'running' ? msg.task : prev.active_task,
                                project_progress: msg.progress,
                            }));

                            if (msg.task.status === 'completed') {
                                showToast('success', 'Task Complete!', `${msg.task.task_id} finished`, '✅');
                            }
                        } else if (msg.type === 'project_completed') {
                            setState(msg.data);
                            setShowCompleteModal(true);
                            showToast('complete', 'PROJECT COMPLETE!', 'All tasks finished!', '🎉');
                        } else if (msg.type === 'achievement_unlocked') {
                            playSound('achievement');
                            showToast('achievement', 'Achievement Unlocked!',
                                `${msg.achievement.title}: ${msg.achievement.description}`,
                                msg.achievement.icon);
                        } else if (msg.type === 'level_up') {
                            showToast('success', 'Level Up!', `You reached level ${msg.level}!`, '⬆️');
                        } else if (msg.type === 'test_update') {
                            setState(prev => ({
                                ...prev,
                                tests: prev.tests.map(t =>
                                    t.test_file === msg.test.test_file ? msg.test : t
                                ),
                            }));
                        }
                    };

                    ws.current.onclose = () => {
                        setConnected(false);
                        setTimeout(connect, 3000); // Reconnect after 3s
                    };
                };

                connect();

                // Keepalive ping
                const ping = setInterval(() => {
                    if (ws.current?.readyState === WebSocket.OPEN) {
                        ws.current.send('ping');
                    }
                }, 30000);

                return () => {
                    clearInterval(ping);
                    ws.current?.close();
                };
            }, []);

            return (
                <Layout style={{ minHeight: '100vh', background: 'transparent' }}>
                    <Header style={{ background: 'rgba(0,0,0,0.3)', backdropFilter: 'blur(10px)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Space>
                                <DashboardOutlined style={{ color: '#722ed1', fontSize: 28 }} />
                                <Title level={3} style={{ color: '#fff', margin: 0 }} className="glow-text">
                                    Mission Control <Text style={{ color: '#722ed1' }}>LIVE</Text>
                                </Title>
                                {connected && <span className="live-indicator" />}
                            </Space>

                            <Space>
                                <Text style={{ color: '#888' }}>v4.0</Text>
                                <Button
                                    type="primary"
                                    icon={<NotificationOutlined />}
                                    onClick={() => showToast('info', 'Test', 'Notification test', '🔔')}
                                >
                                    Test
                                </Button>
                            </Space>
                        </div>
                    </Header>

                    <Content style={{ padding: '24px 50px' }}>
                        {/* Gamification Panel */}
                        <Row style={{ marginBottom: 24 }}>
                            <Col span={24}>
                                <GamificationPanel
                                    level={state.level}
                                    xp={state.xp}
                                    xpToNext={state.xp_to_next_level}
                                    streak={state.streak}
                                    achievements={state.achievements}
                                />
                            </Col>
                        </Row>

                        {/* Project Progress */}
                        <Row style={{ marginBottom: 24 }}>
                            <Col span={24}>
                                <Card style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}>
                                    <Row justify="space-between" align="middle">
                                        <Col>
                                            <Title level={4} style={{ color: '#fff', margin: 0 }}>
                                                Project Progress
                                            </Title>
                                            <Text style={{ color: '#888' }}>
                                                {state.completed_tasks} / {state.total_tasks} tasks completed
                                            </Text>
                                        </Col>
                                        <Col>
                                            <Title level={2} style={{ color: '#722ed1', margin: 0 }}>
                                                {Math.round(state.project_progress)}%
                                            </Title>
                                        </Col>
                                    </Row>
                                    <Progress
                                        percent={Math.round(state.project_progress)}
                                        status={state.project_status === 'completed' ? 'success' : 'active'}
                                        strokeColor={{ from: '#722ed1', to: '#eb2f96' }}
                                        strokeWidth={12}
                                    />
                                </Card>
                            </Col>
                        </Row>

                        {/* Main Content */}
                        <Row gutter={24}>
                            <Col span={16}>
                                <LiveTaskCard task={state.active_task} />
                                <div style={{ marginTop: 24 }}>
                                    <TestMonitor tests={state.tests} />
                                </div>
                            </Col>
                            <Col span={8}>
                                <Card style={{ background: 'rgba(255,255,255,0.05)', border: 'none' }}>
                                    <Title level={5} style={{ color: '#fff' }}>Recent Events</Title>
                                    <Timeline
                                        items={state.recent_events?.slice(-5).map((e, i) => ({
                                            color: e.type === 'success' ? 'green' : e.type === 'error' ? 'red' : 'blue',
                                            children: <Text style={{ color: '#888' }}>{e.message}</Text>,
                                        }))}
                                    />
                                </Card>
                            </Col>
                        </Row>
                    </Content>

                    <Footer style={{ textAlign: 'center', background: 'transparent', color: '#888' }}>
                        Multi-LLM Orchestrator v4.0 | Keep Building! 🚀
                    </Footer>

                    {/* Project Complete Celebration */}
                    <ProjectCompleteModal
                        visible={showCompleteModal}
                        onClose={() => setShowCompleteModal(false)}
                        stats={state}
                    />

                    {/* Floating Action Button */}
                    <FloatButton
                        icon={<ThunderboltOutlined />}
                        type="primary"
                        style={{ right: 24, bottom: 24 }}
                        tooltip="Quick Action"
                    />
                </Layout>
            );
        }

        // Mount app
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>"""

    async def run(self):
        """Start the server."""
        from uvicorn import Config, Server

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)
        await server.serve()


def run_live_dashboard(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):
    """Run the live gamified dashboard."""
    import asyncio

    url = f"http://{host}:{port}"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     ◈ MISSION CONTROL LIVE v4.0 ◈                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Dashboard URL: {url:<44} ║
║                                                                  ║
║  🎮 Features:                                                    ║
║     • Real-time WebSocket updates (no polling!)                  ║
║     • Gamification (XP, levels, achievements)                    ║
║     • Toast notifications for all events                         ║
║     • Confetti celebration on project completion                 ║
║     • Sound effects                                              ║
║     • Live task progress with visual effects                     ║
║     • Test execution monitoring                                  ║
║                                                                  ║
║  🎯 Achievements:                                                ║
║     🎯 Task Master • ⚡ Speed Demon • 💯 Perfectionist           ║
║     💰 Budget Master • 🧪 Test Champion • 🔥 On Fire             ║
║     🏗️ Architect                                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if open_browser:
        webbrowser.open(url)

    dashboard = LiveDashboardServer(host=host, port=port)
    asyncio.run(dashboard.run())


# Integration helper
class DashboardLiveIntegration:
    """Integrate live dashboard with orchestrator."""

    def __init__(self, server: LiveDashboardServer):
        self.server = server

    async def on_project_start(
        self, project_id: str, description: str, total_tasks: int, budget: float
    ):
        """Notify dashboard of project start."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://{self.server.host}:{self.server.port}/api/project/start",
                json={
                    "project_id": project_id,
                    "description": description,
                    "total_tasks": total_tasks,
                    "budget": budget,
                },
            )

    async def on_task_update(
        self, task_id: str, task_type: str, status: str, iteration: int, score: float, model: str
    ):
        """Update task status."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://{self.server.host}:{self.server.port}/api/task/update",
                json={
                    "task_id": task_id,
                    "task_type": task_type,
                    "status": status,
                    "iteration": iteration,
                    "score": score,
                    "model_used": model,
                },
            )

    async def on_project_complete(self):
        """Notify project completion."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://{self.server.host}:{self.server.port}/api/project/complete", json={}
            )

    async def on_test_update(self, test_file: str, status: str, progress: float, output: str = ""):
        """Update test execution."""
        import httpx

        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://{self.server.host}:{self.server.port}/api/test/update",
                json={
                    "test_file": test_file,
                    "status": status,
                    "progress": progress,
                    "output": output,
                },
            )


if __name__ == "__main__":
    run_live_dashboard()
