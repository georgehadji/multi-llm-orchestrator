"""
Ant Design Dashboard v3.0
==========================
Modern dashboard using Ant Design System.

Features:
- Ant Design component library
- Clean, professional UI
- Real-time data visualization
- Architecture decisions panel
- Model status with detailed info
- Task progress tracking

Usage:
    from orchestrator.dashboard_antd import AntDesignDashboard
    dashboard = AntDesignDashboard()
    dashboard.start()
"""

from __future__ import annotations

import asyncio
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardState:
    """Complete dashboard state."""

    project: dict[str, Any]
    architecture: dict[str, Any]
    active_task: dict[str, Any]
    models: list[dict[str, Any]]
    metrics: dict[str, Any]


class AntDesignDashboardServer:
    """
    Modern dashboard using Ant Design System.

    Clean, professional UI with:
    - Real-time updates
    - Architecture visualization
    - Model health monitoring
    - Task progress tracking
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self._setup_app()

    def _setup_app(self):
        """Setup FastAPI app with endpoints."""
        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import HTMLResponse, JSONResponse

            self.app = FastAPI(title="Mission Control - Ant Design")

            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )

            @self.app.get("/")
            async def dashboard():
                """Serve dashboard HTML with Ant Design."""
                return HTMLResponse(content=self._get_html())

            @self.app.get("/api/status")
            async def get_status():
                """Get complete status."""
                return JSONResponse(content=self._get_mock_data())

            @self.app.get("/api/models")
            async def get_models():
                """Get model status."""
                return JSONResponse(content=self._get_mock_data()["models"])

            @self.app.get("/api/project")
            async def get_project():
                """Get project info."""
                return JSONResponse(content=self._get_mock_data()["project"])

            @self.app.get("/api/architecture")
            async def get_architecture():
                """Get architecture decisions."""
                return JSONResponse(content=self._get_mock_data()["architecture"])

            @self.app.get("/api/active-task")
            async def get_active_task():
                """Get active task."""
                return JSONResponse(content=self._get_mock_data()["active_task"])

        except ImportError:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            raise

    def _get_mock_data(self) -> dict[str, Any]:
        """Generate mock data for demonstration."""
        return {
            "project": {
                "project_id": "ecommerce-api-001",
                "description": "Build a scalable e-commerce API with microservices architecture",
                "success_criteria": "Handle 10k requests/sec, 99.9% uptime, JWT authentication",
                "status": "running",
                "total_tasks": 12,
                "completed_tasks": 8,
                "failed_tasks": 0,
                "progress_percent": 66.7,
                "budget_used": 2.45,
                "budget_total": 5.0,
                "elapsed_seconds": 485,
            },
            "architecture": {
                "style": "microservices",
                "paradigm": "object_oriented",
                "api_style": "rest",
                "database_type": "relational",
                "primary_language": "python",
                "frameworks": ["FastAPI", "Pydantic", "SQLAlchemy"],
                "libraries": ["uvicorn", "httpx", "redis-py", "celery"],
                "databases": ["PostgreSQL", "Redis"],
                "tools": ["Docker", "pytest", "black", "ruff"],
                "constraints": [
                    "All services must be stateless",
                    "JWT tokens expire after 24 hours",
                    "Database connections pooled (max 20)",
                    "API rate limit: 100 req/min per client",
                ],
                "patterns": [
                    "CQRS",
                    "Event Sourcing",
                    "Circuit Breaker",
                    "API Gateway",
                    "Repository Pattern",
                ],
                "rationale": "Microservices chosen for independent scaling and deployment",
            },
            "active_task": {
                "task_id": "task_009_payment_service",
                "task_type": "code_generation",
                "prompt": "Implement the Payment Service with Stripe integration. Support multiple payment methods (credit card, PayPal, Apple Pay). Include webhook handling for payment confirmations.",
                "status": "running",
                "iteration": 2,
                "max_iterations": 3,
                "score": 0.87,
                "model_used": "gpt-4o",
                "elapsed_seconds": 45.2,
            },
            "models": [
                {
                    "name": "gpt-4o",
                    "provider": "openai",
                    "available": True,
                    "health_status": "healthy",
                    "reason": "",
                    "success_rate": 0.98,
                    "avg_latency": 125,
                    "call_count": 1523,
                    "cost_input": 2.5,
                    "cost_output": 10.0,
                },
                {
                    "name": "deepseek-chat",
                    "provider": "deepseek",
                    "available": True,
                    "health_status": "healthy",
                    "reason": "",
                    "success_rate": 0.96,
                    "avg_latency": 180,
                    "call_count": 892,
                    "cost_input": 0.14,
                    "cost_output": 0.28,
                },
                {
                    "name": "claude-3-5-sonnet",
                    "provider": "anthropic",
                    "available": False,
                    "health_status": "unhealthy",
                    "reason": "API key not configured",
                    "success_rate": 0.0,
                    "avg_latency": 0,
                    "call_count": 0,
                    "cost_input": 3.0,
                    "cost_output": 15.0,
                },
                {
                    "name": "gemini-2.5-pro",
                    "provider": "google",
                    "available": True,
                    "health_status": "healthy",
                    "reason": "",
                    "success_rate": 0.94,
                    "avg_latency": 210,
                    "call_count": 456,
                    "cost_input": 1.25,
                    "cost_output": 10.0,
                },
            ],
            "metrics": {
                "total_calls": 2871,
                "total_cost": 4.23,
                "avg_latency_ms": 158,
                "active_projects": 1,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _get_html(self) -> str:
        """Generate Ant Design HTML."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mission Control | Multi-LLM Orchestrator</title>

    <!-- Ant Design CSS -->
    <link rel="stylesheet" href="https://unpkg.com/antd@5.12.0/dist/reset.css">
    <link rel="stylesheet" href="https://unpkg.com/antd@5.12.0/dist/antd.min.css">

    <!-- React & ReactDOM -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

    <!-- Babel -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

    <!-- Ant Design -->
    <script src="https://unpkg.com/antd@5.12.0/dist/antd.min.js"></script>

    <!-- Ant Design Icons -->
    <script src="https://unpkg.com/@ant-design/icons@5.2.6/dist/index.umd.min.js"></script>

    <style>
        body {
            background: #f0f2f5;
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }
        .site-layout-content {
            min-height: 280px;
            padding: 24px;
            background: #fff;
        }
        .logo {
            float: left;
            width: 120px;
            height: 31px;
            margin: 16px 24px 16px 0;
            background: rgba(255, 255, 255, 0.3);
            color: #fff;
            font-size: 18px;
            font-weight: 600;
            line-height: 31px;
            text-align: center;
        }
        .ant-layout-header {
            background: #001529;
            padding: 0 50px;
        }
        .ant-card {
            margin-bottom: 24px;
        }
        .metric-card .ant-statistic-title {
            color: #8c8c8c;
            font-size: 14px;
        }
        .metric-card .ant-statistic-content {
            color: #1890ff;
            font-size: 24px;
            font-weight: 600;
        }
        .model-tag-available {
            background: #f6ffed;
            border-color: #b7eb8f;
            color: #52c41a;
        }
        .model-tag-unavailable {
            background: #fff2f0;
            border-color: #ffccc7;
            color: #ff4d4f;
        }
        .progress-ring {
            position: relative;
            display: inline-block;
        }
        .tech-badge {
            margin: 4px;
        }
        .constraint-item, .pattern-item {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .constraint-item:last-child, .pattern-item:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect } = React;
        const {
            Layout, Menu, Card, Row, Col, Statistic, Progress, Tag, Timeline,
            Badge, Descriptions, List, Avatar, Typography, Space, Divider,
            Alert, Skeleton, Tooltip, Button, Table
        } = antd;
        const {
            DashboardOutlined, ApiOutlined, SettingOutlined, FileTextOutlined,
            CheckCircleOutlined, CloseCircleOutlined, SyncOutlined, ClusterOutlined,
            CodeOutlined, DatabaseOutlined, ToolOutlined, ExperimentOutlined,
            ReloadOutlined, ClockCircleOutlined, DollarOutlined
        } = icons;

        const { Header, Content, Footer } = Layout;
        const { Title, Text, Paragraph } = Typography;

        // Mock data - will be replaced with API calls
        const initialData = {
            project: {
                project_id: "-",
                description: "Loading...",
                status: "idle",
                total_tasks: 0,
                completed_tasks: 0,
                progress_percent: 0,
                budget_used: 0,
                budget_total: 0,
                elapsed_seconds: 0,
            },
            architecture: {},
            active_task: { status: "idle" },
            models: [],
            metrics: {},
        };

        function formatDuration(seconds) {
            if (!seconds) return "0s";
            if (seconds < 60) return Math.round(seconds) + "s";
            if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.round(seconds % 60) + "s";
            return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
        }

        function formatCurrency(value) {
            return "$" + (value || 0).toFixed(2);
        }

        // Components
        function ProjectCard({ project }) {
            const statusColors = {
                idle: "default",
                running: "processing",
                completed: "success",
                failed: "error",
            };

            return (
                <Card
                    title={<><DashboardOutlined /> Project Overview</>}
                    extra={<Badge status={statusColors[project.status] || "default"} text={project.status.toUpperCase()} />}
                >
                    <Title level={4}>{project.description}</Title>
                    <Text type="secondary">ID: {project.project_id}</Text>

                    <Divider />

                    <Row gutter={16}>
                        <Col span={6}>
                            <Statistic
                                title="Progress"
                                value={project.progress_percent || 0}
                                suffix="%"
                                precision={1}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic
                                title="Tasks"
                                value={`${project.completed_tasks || 0}/${project.total_tasks || 0}`}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic
                                title="Budget Used"
                                value={formatCurrency(project.budget_used)}
                                prefix={<DollarOutlined />}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic
                                title="Elapsed"
                                value={formatDuration(project.elapsed_seconds)}
                                prefix={<ClockCircleOutlined />}
                            />
                        </Col>
                    </Row>

                    <Divider />

                    <Progress
                        percent={Math.round(project.progress_percent || 0)}
                        status={project.status === "failed" ? "exception" : "active"}
                        strokeColor={{ "0%": "#108ee9", "100%": "#87d068" }}
                    />
                </Card>
            );
        }

        function ActiveTaskCard({ task }) {
            if (!task || task.status === "idle") {
                return (
                    <Card title={<><SyncOutlined spin /> Active Task</>}>
                        <Alert message="No active task" type="info" showIcon />
                    </Card>
                );
            }

            const statusColors = {
                running: "blue",
                completed: "green",
                failed: "red",
            };

            return (
                <Card
                    title={<><CodeOutlined /> Active Task</>}
                    extra={<Tag color={statusColors[task.status]}>{task.status.toUpperCase()}</Tag>}
                >
                    <Descriptions bordered column={1} size="small">
                        <Descriptions.Item label="Task ID">{task.task_id}</Descriptions.Item>
                        <Descriptions.Item label="Type">
                            <Tag color="blue">{task.task_type}</Tag>
                        </Descriptions.Item>
                        <Descriptions.Item label="Model">{task.model_used}</Descriptions.Item>
                        <Descriptions.Item label="Iteration">
                            {task.iteration} / {task.max_iterations}
                        </Descriptions.Item>
                        <Descriptions.Item label="Score">
                            <Progress
                                type="circle"
                                percent={Math.round((task.score || 0) * 100)}
                                width={50}
                                status={task.score >= 0.8 ? "success" : "normal"}
                            />
                        </Descriptions.Item>
                    </Descriptions>

                    <Divider />

                    <Title level={5}>Prompt</Title>
                    <Paragraph ellipsis={{ rows: 3, expandable: true }}>
                        {task.prompt}
                    </Paragraph>
                </Card>
            );
        }

        function ArchitectureCard({ arch }) {
            if (!arch || !arch.style) {
                return (
                    <Card title={<><ClusterOutlined /> Architecture</>}>
                        <Skeleton active />
                    </Card>
                );
            }

            return (
                <Card title={<><ClusterOutlined /> Architecture Decisions</>}>
                    <Descriptions bordered column={2} size="small">
                        <Descriptions.Item label="Style">{arch.style}</Descriptions.Item>
                        <Descriptions.Item label="Paradigm">{arch.paradigm}</Descriptions.Item>
                        <Descriptions.Item label="API Style">{arch.api_style}</Descriptions.Item>
                        <Descriptions.Item label="Database">{arch.database_type}</Descriptions.Item>
                        <Descriptions.Item label="Language" span={2}>{arch.primary_language}</Descriptions.Item>
                    </Descriptions>

                    <Divider />

                    <Title level={5}>Technology Stack</Title>
                    <Space wrap>
                        {arch.frameworks?.map(f => <Tag key={f} color="blue" className="tech-badge">{f}</Tag>)}
                        {arch.libraries?.map(l => <Tag key={l} color="green" className="tech-badge">{l}</Tag>)}
                        {arch.databases?.map(d => <Tag key={d} color="purple" className="tech-badge">{d}</Tag>)}
                        {arch.tools?.map(t => <Tag key={t} color="orange" className="tech-badge">{t}</Tag>)}
                    </Space>

                    <Divider />

                    <Row gutter={16}>
                        <Col span={12}>
                            <Title level={5}>Constraints</Title>
                            <List
                                size="small"
                                dataSource={arch.constraints || []}
                                renderItem={item => (
                                    <List.Item className="constraint-item">
                                        <CheckCircleOutlined style={{ color: "#52c41a", marginRight: 8 }} />
                                        {item}
                                    </List.Item>
                                )}
                            />
                        </Col>
                        <Col span={12}>
                            <Title level={5}>Patterns</Title>
                            <List
                                size="small"
                                dataSource={arch.patterns || []}
                                renderItem={item => (
                                    <List.Item className="pattern-item">
                                        <ExperimentOutlined style={{ color: "#1890ff", marginRight: 8 }} />
                                        {item}
                                    </List.Item>
                                )}
                            />
                        </Col>
                    </Row>
                </Card>
            );
        }

        function ModelsCard({ models }) {
            const columns = [
                {
                    title: "Model",
                    dataIndex: "name",
                    key: "name",
                    render: (text, record) => (
                        <Space>
                            <Avatar
                                style={{
                                    backgroundColor: record.available ? "#52c41a" : "#ff4d4f",
                                    fontSize: 12
                                }}
                                size="small"
                            >
                                {text[0].toUpperCase()}
                            </Avatar>
                            <Text strong>{text}</Text>
                        </Space>
                    ),
                },
                {
                    title: "Provider",
                    dataIndex: "provider",
                    key: "provider",
                },
                {
                    title: "Status",
                    dataIndex: "available",
                    key: "status",
                    render: (available, record) => (
                        <Tooltip title={record.reason || ""}>
                            <Tag className={available ? "model-tag-available" : "model-tag-unavailable"}>
                                {available ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                                {" "}
                                {available ? "Available" : "Unavailable"}
                            </Tag>
                        </Tooltip>
                    ),
                },
                {
                    title: "Success Rate",
                    dataIndex: "success_rate",
                    key: "success_rate",
                    render: (rate) => `${(rate * 100).toFixed(0)}%`,
                },
                {
                    title: "Latency",
                    dataIndex: "avg_latency",
                    key: "avg_latency",
                    render: (lat) => `${lat}ms`,
                },
                {
                    title: "Calls",
                    dataIndex: "call_count",
                    key: "call_count",
                },
                {
                    title: "Cost",
                    key: "cost",
                    render: (_, record) => `$${record.cost_input}/$${record.cost_output}`,
                },
            ];

            return (
                <Card
                    title={<><ApiOutlined /> Model Status</>}
                    extra={
                        <Button
                            type="primary"
                            icon={<ReloadOutlined />}
                            onClick={() => window.location.reload()}
                        >
                            Refresh
                        </Button>
                    }
                >
                    <Table
                        dataSource={models}
                        columns={columns}
                        rowKey="name"
                        pagination={false}
                        size="small"
                    />
                </Card>
            );
        }

        function MetricsCard({ metrics }) {
            return (
                <Row gutter={16}>
                    <Col span={6}>
                        <Card className="metric-card">
                            <Statistic
                                title="Total Calls"
                                value={metrics.total_calls || 0}
                                prefix={<ApiOutlined />}
                            />
                        </Card>
                    </Col>
                    <Col span={6}>
                        <Card className="metric-card">
                            <Statistic
                                title="Total Cost"
                                value={formatCurrency(metrics.total_cost)}
                                prefix={<DollarOutlined />}
                            />
                        </Card>
                    </Col>
                    <Col span={6}>
                        <Card className="metric-card">
                            <Statistic
                                title="Avg Latency"
                                value={metrics.avg_latency_ms || 0}
                                suffix="ms"
                                prefix={<ClockCircleOutlined />}
                            />
                        </Card>
                    </Col>
                    <Col span={6}>
                        <Card className="metric-card">
                            <Statistic
                                title="Active Projects"
                                value={metrics.active_projects || 0}
                                prefix={<DashboardOutlined />}
                            />
                        </Card>
                    </Col>
                </Row>
            );
        }

        function App() {
            const [data, setData] = useState(initialData);
            const [loading, setLoading] = useState(true);

            useEffect(() => {
                // Initial data load
                fetchData();

                // Auto-refresh every 3 seconds
                const interval = setInterval(fetchData, 3000);
                return () => clearInterval(interval);
            }, []);

            async function fetchData() {
                try {
                    const response = await fetch("/api/status");
                    const newData = await response.json();
                    setData(newData);
                    setLoading(false);
                } catch (err) {
                    console.error("Failed to fetch:", err);
                    // Use mock data for demo
                    if (loading) {
                        setData({
                            project: initialData.project,
                            architecture: {},
                            active_task: { status: "idle" },
                            models: [],
                            metrics: {},
                        });
                    }
                }
            }

            return (
                <Layout className="layout">
                    <Header>
                        <div className="logo">◈ Mission Control</div>
                        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={["dashboard"]}>
                            <Menu.Item key="dashboard" icon={<DashboardOutlined />}>
                                Dashboard
                            </Menu.Item>
                            <Menu.Item key="models" icon={<ApiOutlined />}>
                                Models
                            </Menu.Item>
                            <Menu.Item key="settings" icon={<SettingOutlined />}>
                                Settings
                            </Menu.Item>
                        </Menu>
                    </Header>

                    <Content style={{ padding: "0 50px" }}>
                        <div className="site-layout-content" style={{ marginTop: 24 }}>
                            {loading ? (
                                <Skeleton active />
                            ) : (
                                <>
                                    <MetricsCard metrics={data.metrics} />

                                    <Row gutter={24} style={{ marginTop: 24 }}>
                                        <Col span={16}>
                                            <ProjectCard project={data.project} />
                                            <ActiveTaskCard task={data.active_task} />
                                        </Col>
                                        <Col span={8}>
                                            <ArchitectureCard arch={data.architecture} />
                                        </Col>
                                    </Row>

                                    <Row style={{ marginTop: 24 }}>
                                        <Col span={24}>
                                            <ModelsCard models={data.models} />
                                        </Col>
                                    </Row>
                                </>
                            )}
                        </div>
                    </Content>

                    <Footer style={{ textAlign: "center" }}>
                        Multi-LLM Orchestrator v5.1 | Built with Ant Design
                        <br />
                        <Text type="secondary">
                            Last updated: {new Date().toLocaleTimeString()}
                        </Text>
                    </Footer>
                </Layout>
            );
        }

        // Mount the app
        const root = ReactDOM.createRoot(document.getElementById("root"));
        root.render(<App />);
    </script>
</body>
</html>"""

    async def run(self):
        """Start the dashboard server."""
        from uvicorn import Config, Server

        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = Server(config)
        await server.serve()


def run_ant_design_dashboard(host: str = "127.0.0.1", port: int = 8888, open_browser: bool = True):
    """Run the Ant Design dashboard."""

    url = f"http://{host}:{port}"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     ◈ MISSION CONTROL - Ant Design v3.0 ◈                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌐 Dashboard URL: {url:<44} ║
║                                                                  ║
║  🎨 UI Framework: Ant Design 5.x                                 ║
║                                                                  ║
║  📊 Features:                                                    ║
║     • Modern, professional UI                                    ║
║     • Real-time data visualization                               ║
║     • Architecture decisions panel                               ║
║     • Model health monitoring                                    ║
║     • Task progress tracking                                     ║
║                                                                  ║
║  🔄 Auto-refresh: Every 3 seconds                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if open_browser:
        webbrowser.open(url)

    dashboard = AntDesignDashboardServer(host=host, port=port)
    asyncio.run(dashboard.run())


if __name__ == "__main__":
    run_ant_design_dashboard()
