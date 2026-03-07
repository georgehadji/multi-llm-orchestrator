"""
Orchestrator Dashboard Views Plugin
===================================
Save this as orchestrator_plugins/dashboards/__init__.py

Additional dashboard views for the unified dashboard core.
"""

from __future__ import annotations

from typing import Any, Optional, Dict

# Import from core dashboard system
try:
    from orchestrator.dashboard_core_core import DashboardView, ViewContext
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    # Define minimal base class
    class ViewContext:
        pass
    
    class DashboardView:
        name = "base"
        display_name = "Base"
        version = "1.0.0"
        
        async def render(self, context: ViewContext) -> str:
            return "<html><body>Dashboard not available</body></html>"
        
        def get_assets(self) -> Dict[str, list]:
            return {"css": [], "js": []}
        
        def handle_event(self, event: Any) -> Optional[Dict[str, Any]]:
            return None


class AntDesignView(DashboardView):
    """
    Ant Design Pro dashboard view.
    Modern, professional enterprise UI.
    """
    
    name = "ant-design"
    display_name = "📊 Ant Design Pro"
    version = "3.0.0"
    
    def get_assets(self) -> Dict[str, list]:
        return {
            "css": [
                "https://cdn.jsdelivr.net/npm/antd@4.24.0/dist/reset.css",
            ],
            "js": [
                "https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js",
                "https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.production.min.js",
                "https://cdn.jsdelivr.net/npm/antd@4.24.0/dist/antd.min.js",
            ],
        }
    
    async def render(self, context: ViewContext) -> str:
        """Render Ant Design dashboard."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orchestrator Dashboard - Ant Design</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/antd@4.24.0/dist/reset.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/antd@4.24.0/dist/antd.min.css">
    <style>
        body {{ margin: 0; background: #f0f2f5; }}
        .ant-layout {{ min-height: 100vh; }}
        .ant-statistic-title {{ font-size: 14px; }}
        .ant-statistic-content {{ font-size: 24px; font-weight: 600; }}
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script src="https://cdn.jsdelivr.net/npm/react@18/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/antd@4.24.0/dist/antd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@ant-design/icons@4.7.0/dist/index.umd.min.js"></script>
    
    <script>
        const {{ Layout, Menu, Card, Statistic, Row, Col, Table, Tag, Progress, Typography }} = antd;
        const {{ Header, Content, Footer, Sider }} = Layout;
        const {{ Title, Text }} = Typography;
        const {{ PlayCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, 
                 DollarOutlined, ClockCircleOutlined, RocketOutlined }} = icons;
        
        const App = () => {{
            const [collapsed, setCollapsed] = React.useState(false);
            const [data, setData] = React.useState({{
                tasks: [],
                metrics: {{}},
                models: {{}},
            }});
            
            // WebSocket connection
            React.useEffect(() => {{
                const ws = new WebSocket(`ws://${{window.location.host}}/ws`);
                ws.onmessage = (event) => {{
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'TASK_STARTED' || msg.type === 'TASK_COMPLETED') {{
                        fetchContext();
                    }}
                }};
                ws.send('subscribe');
                
                // Initial data fetch
                fetchContext();
                
                return () => ws.close();
            }}, []);
            
            const fetchContext = () => {{
                fetch('/api/context')
                    .then(r => r.json())
                    .then(setData);
            }};
            
            const taskColumns = [
                {{ title: 'Task', dataIndex: 'id', key: 'id' }},
                {{ title: 'Type', dataIndex: 'type', key: 'type', render: t => <Tag>{{t}}</Tag> }},
                {{ title: 'Status', dataIndex: 'status', key: 'status', 
                   render: s => {{
                       const colors = {{ pending: 'default', running: 'processing', completed: 'success', failed: 'error' }};
                       return <Tag color={{colors[s] || 'default'}}>{{s}}</Tag>;
                   }}
                }},
                {{ title: 'Score', dataIndex: 'score', key: 'score', render: s => s?.toFixed(2) || '-' }},
            ];
            
            return (
                <Layout>
                    <Sider trigger={{null}} collapsible collapsed={{collapsed}} theme="light">
                        <div style={{{{ height: 32, margin: 16, background: '#1890ff', borderRadius: 4 }}}} />
                        <Menu theme="light" mode="inline" defaultSelectedKeys={{['dashboard']}}>
                            <Menu.Item key="dashboard" icon=<RocketOutlined />>Dashboard</Menu.Item>
                            <Menu.Item key="tasks" icon=<PlayCircleOutlined />>Tasks</Menu.Item>
                            <Menu.Item key="models" icon=<CheckCircleOutlined />>Models</Menu.Item>
                        </Menu>
                    </Sider>
                    <Layout>
                        <Header style={{{{ padding: 0, background: '#fff', paddingLeft: 24 }}}}>
                            <Title level={{4}} style={{{{ margin: 0, lineHeight: '64px' }}}}>
                                Orchestrator Dashboard
                            </Title>
                        </Header>
                        <Content style={{{{ margin: '24px 16px', padding: 24, background: '#fff', minHeight: 280 }}}}>
                            <Row gutter={{[16, 16]}}>
                                <Col span={{6}}>
                                    <Card>
                                        <Statistic 
                                            title="Tasks" 
                                            value={{data.active_tasks?.length || 0}}
                                            prefix=<PlayCircleOutlined /> 
                                        />
                                    </Card>
                                </Col>
                                <Col span={{6}}>
                                    <Card>
                                        <Statistic 
                                            title="Budget Used" 
                                            value={{data.budget?.spent || 0}}
                                            prefix=<DollarOutlined />
                                            suffix={{`/ $${{data.budget?.max || 0}}`}}
                                            precision={{2}}
                                        />
                                    </Card>
                                </Col>
                                <Col span={{6}}>
                                    <Card>
                                        <Statistic 
                                            title="Quality Score" 
                                            value={{(data.metrics?.quality_score || 0) * 100}}
                                            suffix="%"
                                        />
                                    </Card>
                                </Col>
                                <Col span={{6}}>
                                    <Card>
                                        <Statistic 
                                            title="Models Active" 
                                            value={{Object.keys(data.model_status || {{}}).length}}
                                            prefix=<CheckCircleOutlined />
                                        />
                                    </Card>
                                </Col>
                            </Row>
                            
                            <Card title="Active Tasks" style={{{{ marginTop: 24 }}}}>
                                <Table 
                                    columns={{taskColumns}} 
                                    dataSource={{data.active_tasks}}
                                    rowKey="id"
                                />
                            </Card>
                        </Content>
                    </Layout>
                </Layout>
            );
        }};
        
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
"""


class MinimalView(DashboardView):
    """
    Minimal dashboard view for low-bandwidth environments.
    Lightweight, fast-loading, no external dependencies.
    """
    
    name = "minimal"
    display_name = "📄 Minimal"
    version = "1.0.0"
    
    def get_assets(self) -> Dict[str, list]:
        return {"css": [], "js": []}  # No external deps
    
    async def render(self, context: ViewContext) -> str:
        """Render minimal HTML dashboard."""
        tasks = getattr(context, 'active_tasks', [])
        budget = getattr(context, 'budget', {})
        
        tasks_html = ""
        for task in tasks:
            status = task.get('status', 'pending')
            color = {'completed': 'green', 'failed': 'red', 'running': 'blue'}.get(status, 'gray')
            tasks_html += f"""
            <tr>
                <td>{task.get('id', '-')}</td>
                <td>{task.get('type', '-')}</td>
                <td style="color: {color}">{status}</td>
                <td>{task.get('score', '-'):.2f}</td>
            </tr>
            """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Orchestrator - Minimal View</title>
    <style>
        body {{ font-family: monospace; max-width: 1200px; margin: 40px auto; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
        .metric {{ display: inline-block; margin-right: 40px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ color: #666; }}
    </style>
</head>
<body>
    <h1>🚀 Orchestrator Dashboard (Minimal)</h1>
    
    <div style="margin: 20px 0;">
        <div class="metric">
            <div class="metric-label">Project</div>
            <div class="metric-value">{getattr(context, 'project_id', 'N/A')}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Budget</div>
            <div class="metric-value">${budget.get('spent', 0):.2f} / ${budget.get('max', 0):.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Tasks</div>
            <div class="metric-value">{len(tasks)}</div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Task ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>Score</th>
            </tr>
        </thead>
        <tbody>
            {tasks_html or '<tr><td colspan="4">No tasks</td></tr>'}
        </tbody>
    </table>
    
    <footer style="margin-top: 40px; color: #666; font-size: 12px;">
        Auto-refresh: 5s | View: minimal v1.0
    </footer>
    
    <script>
        setTimeout(() => location.reload(), 5000);
    </script>
</body>
</html>
"""


# Auto-register views if core is available
if HAS_CORE:
    try:
        from orchestrator.dashboard_core_core import get_dashboard_core
        
        async def register_views():
            core = await get_dashboard_core()
            core.register_view(AntDesignView())
            core.register_view(MinimalView())
        
        # Schedule registration
        import asyncio
        try:
            asyncio.create_task(register_views())
        except RuntimeError:
            pass  # No event loop
        
    except Exception as e:
        print(f"Failed to auto-register dashboard views: {e}")


__all__ = [
    "AntDesignView",
    "MinimalView",
]
