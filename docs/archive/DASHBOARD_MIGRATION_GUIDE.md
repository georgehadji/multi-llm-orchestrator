# Dashboard Migration Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Migrate from legacy dashboards to unified dashboard_core.** This guide covers migration from all 9 dashboard implementations to the single unified dashboard.

---

## Overview

The AI Orchestrator previously had **9 separate dashboard implementations**, causing confusion and maintenance overhead. The unified `dashboard_core/` module consolidates all functionality into a single, maintainable dashboard.

### Legacy Dashboards

| Dashboard | Status | Replacement |
|-----------|--------|-------------|
| `dashboard.py` | ❌ Deprecated | `dashboard_core/core.py` |
| `dashboard_enhanced.py` | ❌ Deprecated | `dashboard_core/core.py` |
| `dashboard_optimized.py` | ❌ Deprecated | `dashboard_core/core.py` |
| `dashboard_antd.py` | ❌ Deprecated | `dashboard_core/core.py` + Ant Design theme |
| `dashboard_live.py` | ❌ Deprecated | `dashboard_core/core.py` + WebSocket |
| `dashboard_mc_simple.py` | ❌ Deprecated | `dashboard_core/mission_control.py` |
| `dashboard_mission_control.py` | ❌ Deprecated | `dashboard_core/mission_control.py` |
| `dashboard_mission_control_fix.py` | ❌ Deprecated | `dashboard_core/mission_control.py` |
| `dashboard_real.py` | ❌ Deprecated | `dashboard_core/core.py` |

### Unified Dashboard

```
orchestrator/dashboard_core/
├── __init__.py           # Public API
├── core.py               # Main dashboard core
└── mission_control.py    # Mission Control view
```

---

## Quick Migration

### Before (Legacy)

```python
# Multiple imports, different APIs
from orchestrator.dashboard import run_dashboard
from orchestrator.dashboard_live import run_live_dashboard
from orchestrator.dashboard_mission_control import run_mission_control

# Different APIs for each
run_dashboard(port=8888)
run_live_dashboard(port=8889)
run_mission_control(port=8900)
```

### After (Unified)

```python
# Single import
from orchestrator.dashboard_core import DashboardCore, get_dashboard_core

# Initialize once
dashboard = get_dashboard_core()

# Configure views
dashboard.register_view("mission_control")
dashboard.register_view("real_time")
dashboard.register_view("analytics")

# Run with unified API
await dashboard.run(port=8888)
```

---

## Migration Steps

### Step 1: Update Imports

#### Old Imports
```python
from orchestrator.dashboard import Dashboard
from orchestrator.dashboard_live import LiveDashboard
from orchestrator.dashboard_antd import AntDDashboard
```

#### New Imports
```python
from orchestrator.dashboard_core import (
    DashboardCore,
    get_dashboard_core,
    DashboardView,
    ViewContext,
    ViewRegistry,
)

# Mission Control (if needed)
from orchestrator.dashboard_core import MissionControlView
```

---

### Step 2: Update Initialization

#### Old: Basic Dashboard
```python
from orchestrator.dashboard import Dashboard

dashboard = Dashboard(
    port=8888,
    title="AI Orchestrator Dashboard",
)
dashboard.run()
```

#### New: Unified Dashboard
```python
from orchestrator.dashboard_core import get_dashboard_core

dashboard = get_dashboard_core(
    port=8888,
    title="AI Orchestrator Dashboard",
)
await dashboard.run()
```

---

### Step 3: Update Views

#### Old: Multiple Dashboard Files
```python
# dashboard.py
def render_kpi_panel():
    return html.Div(...)

# dashboard_live.py
def render_live_metrics():
    return html.Div(...)

# dashboard_mission_control.py
def render_mission_control():
    return html.Div(...)
```

#### New: Unified Views
```python
from orchestrator.dashboard_core import DashboardView, ViewContext

class KPIView(DashboardView):
    """KPI Panel View."""
    
    def render(self, ctx: ViewContext) -> str:
        return f"""
        <div class="kpi-panel">
            <h2>Key Metrics</h2>
            <!-- KPI content -->
        </div>
        """

class LiveMetricsView(DashboardView):
    """Live Metrics View."""
    
    def render(self, ctx: ViewContext) -> str:
        return f"""
        <div class="live-metrics">
            <h2>Live Metrics</h2>
            <!-- Live content -->
        </div>
        """

# Register views
dashboard.register_view(KPIView())
dashboard.register_view(LiveMetricsView())
```

---

### Step 4: Update WebSocket (if using live dashboard)

#### Old: dashboard_live.py
```python
from orchestrator.dashboard_live import LiveDashboard

dashboard = LiveDashboard(
    port=8888,
    websocket_path="/ws",
)

async def send_update(data):
    await dashboard.broadcast(data)
```

#### New: Unified with WebSocket
```python
from orchestrator.dashboard_core import DashboardCore

dashboard = get_dashboard_core(
    port=8888,
    enable_websocket=True,
    websocket_path="/ws",
)

async def send_update(data):
    await dashboard.broadcast("update", data)
```

---

### Step 5: Update Mission Control

#### Old: dashboard_mission_control.py
```python
from orchestrator.dashboard_mission_control import run_mission_control

run_mission_control(port=8900)
```

#### New: Unified Mission Control
```python
from orchestrator.dashboard_core import MissionControlView, get_dashboard_core

dashboard = get_dashboard_core(port=8888)
dashboard.register_view(MissionControlView())
await dashboard.run()
```

---

## Feature Mapping

### Legacy Features → Unified Features

| Legacy Feature | Old Location | New Location |
|----------------|--------------|--------------|
| KPI Panel | `dashboard.py` | `core.py` → `KPIView` |
| Live Metrics | `dashboard_live.py` | `core.py` → `LiveMetricsView` |
| Mission Control | `dashboard_mission_control.py` | `mission_control.py` |
| Ant Design UI | `dashboard_antd.py` | Theme configuration |
| Real-time Updates | `dashboard_live.py` | WebSocket in `core.py` |
| Enhanced Layout | `dashboard_enhanced.py` | `core.py` → `EnhancedView` |
| Optimized Rendering | `dashboard_optimized.py` | Built-in optimization |

---

## API Reference

### DashboardCore

```python
class DashboardCore:
    """Unified dashboard core."""
    
    def __init__(
        self,
        port: int = 8888,
        host: str = "localhost",
        title: str = "AI Orchestrator",
        enable_websocket: bool = True,
        websocket_path: str = "/ws",
    )
    
    def register_view(self, view: DashboardView) -> None:
        """Register a dashboard view."""
    
    def unregister_view(self, view_name: str) -> None:
        """Unregister a view."""
    
    async def run(self) -> None:
        """Start dashboard server."""
    
    async def stop(self) -> None:
        """Stop dashboard server."""
    
    async def broadcast(self, event: str, data: dict) -> None:
        """Broadcast event to all connected clients."""
    
    def get_metrics(self) -> dict:
        """Get dashboard metrics."""
```

### DashboardView

```python
class DashboardView(ABC):
    """Base class for dashboard views."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """View name."""
    
    @abstractmethod
    def render(self, ctx: ViewContext) -> str:
        """Render view HTML."""
    
    async def on_connect(self, client_id: str) -> None:
        """Called when client connects."""
    
    async def on_disconnect(self, client_id: str) -> None:
        """Called when client disconnects."""
    
    async def on_event(self, event: str, data: dict) -> None:
        """Handle client event."""
```

### ViewContext

```python
@dataclass
class ViewContext:
    """Context for view rendering."""
    
    request: Request
    session: dict
    user: Optional[dict]
    metrics: dict
    theme: str
```

### ViewRegistry

```python
class ViewRegistry:
    """Registry for dashboard views."""
    
    def register(self, view: DashboardView) -> None:
        """Register a view."""
    
    def get(self, name: str) -> Optional[DashboardView]:
        """Get view by name."""
    
    def list_views(self) -> List[str]:
        """List all registered views."""
    
    def unregister(self, name: str) -> None:
        """Unregister a view."""
```

---

## Built-in Views

### KPIView

```python
from orchestrator.dashboard_core import KPIView

dashboard.register_view(KPIView(
    metrics=[
        "cost_per_hour",
        "tasks_completed",
        "quality_score",
        "model_health",
    ],
    refresh_interval=5,  # seconds
))
```

### LiveMetricsView

```python
from orchestrator.dashboard_core import LiveMetricsView

dashboard.register_view(LiveMetricsView(
    websocket_enabled=True,
    metrics=[
        "latency_p95",
        "token_usage",
        "active_tasks",
        "error_rate",
    ],
))
```

### MissionControlView

```python
from orchestrator.dashboard_core import MissionControlView

dashboard.register_view(MissionControlView(
    show_alerts=True,
    show_events=True,
    alert_severities=["critical", "high", "medium"],
))
```

### AnalyticsView

```python
from orchestrator.dashboard_core import AnalyticsView

dashboard.register_view(AnalyticsView(
    charts=[
        "cost_trend",
        "model_distribution",
        "task_success_rate",
        "latency_over_time",
    ],
    time_range="24h",
))
```

---

## Configuration

### Environment Variables

```bash
# Dashboard
export DASHBOARD_PORT=8888
export DASHBOARD_HOST=localhost
export DASHBOARD_TITLE="AI Orchestrator"

# WebSocket
export DASHBOARD_WS_ENABLED=true
export DASHBOARD_WS_PATH=/ws

# Authentication
export DASHBOARD_AUTH_ENABLED=false
export DASHBOARD_AUTH_TOKEN=your-token

# Refresh
export DASHBOARD_REFRESH_INTERVAL=5
```

### Python Configuration

```python
from orchestrator.dashboard_core import configure_dashboard

configure_dashboard(
    port=8888,
    host="0.0.0.0",
    title="AI Orchestrator Dashboard",
    enable_websocket=True,
    websocket_path="/ws",
    auth_enabled=True,
    auth_token="your-secure-token",
    refresh_interval=5,
    theme="dark",  # or "light", "auto"
)
```

---

## Examples

### Example 1: Basic Dashboard

```python
from orchestrator.dashboard_core import get_dashboard_core

dashboard = get_dashboard_core(port=8888)
await dashboard.run()
```

### Example 2: Dashboard with Custom Views

```python
from orchestrator.dashboard_core import (
    DashboardCore,
    DashboardView,
    ViewContext,
    KPIView,
    LiveMetricsView,
)

class CustomView(DashboardView):
    @property
    def name(self) -> str:
        return "custom"
    
    def render(self, ctx: ViewContext) -> str:
        return """
        <div class="custom-view">
            <h2>Custom View</h2>
            <p>Custom content here</p>
        </div>
        """

dashboard = get_dashboard_core(port=8888)
dashboard.register_view(KPIView())
dashboard.register_view(LiveMetricsView())
dashboard.register_view(CustomView())
await dashboard.run()
```

### Example 3: Mission Control Dashboard

```python
from orchestrator.dashboard_core import (
    get_dashboard_core,
    MissionControlView,
    KPIView,
)

dashboard = get_dashboard_core(
    port=8888,
    title="Mission Control",
)

dashboard.register_view(MissionControlView(
    show_alerts=True,
    show_events=True,
))
dashboard.register_view(KPIView())

await dashboard.run()
```

### Example 4: Real-time Dashboard with WebSocket

```python
from orchestrator.dashboard_core import get_dashboard_core
import asyncio

dashboard = get_dashboard_core(
    port=8888,
    enable_websocket=True,
    websocket_path="/ws",
)

async def broadcast_updates():
    """Broadcast updates every 5 seconds."""
    while True:
        await dashboard.broadcast("update", {
            "timestamp": asyncio.get_event_loop().time(),
            "metrics": dashboard.get_metrics(),
        })
        await asyncio.sleep(5)

# Run dashboard and broadcast in parallel
await asyncio.gather(
    dashboard.run(),
    broadcast_updates(),
)
```

---

## Troubleshooting

### Dashboard Won't Start

```python
# Check port availability
import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False

print(f"Port 8888 available: {is_port_available(8888)}")
```

### WebSocket Not Connecting

```bash
# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Host: localhost:8888" \
  http://localhost:8888/ws
```

### Views Not Rendering

```python
# List registered views
dashboard = get_dashboard_core()
print(f"Registered views: {dashboard.view_registry.list_views()}")

# Check view registration
from orchestrator.dashboard_core import KPIView
view = KPIView()
print(f"View name: {view.name}")
```

---

## Migration Checklist

- [ ] Update imports from legacy modules to `dashboard_core`
- [ ] Replace `Dashboard()` with `get_dashboard_core()`
- [ ] Migrate custom views to `DashboardView` base class
- [ ] Update WebSocket code to use `broadcast()` method
- [ ] Migrate Mission Control to `MissionControlView`
- [ ] Update configuration to use new environment variables
- [ ] Test all views render correctly
- [ ] Verify WebSocket connections work
- [ ] Remove legacy dashboard files
- [ ] Update documentation references

---

## Legacy File Removal

After migration, remove legacy files:

```bash
# Backup first
mkdir dashboard_backup
cp orchestrator/dashboard*.py dashboard_backup/

# Remove legacy files
rm orchestrator/dashboard.py
rm orchestrator/dashboard_enhanced.py
rm orchestrator/dashboard_optimized.py
rm orchestrator/dashboard_antd.py
rm orchestrator/dashboard_live.py
rm orchestrator/dashboard_mc_simple.py
rm orchestrator/dashboard_mission_control.py
rm orchestrator/dashboard_mission_control_fix.py
rm orchestrator/dashboard_real.py
```

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide with dashboard examples
- [CAPABILITIES.md](./CAPABILITIES.md) — Dashboard capabilities overview
- [COMMAND_CENTER_IMPLEMENTATION.md](./COMMAND_CENTER_IMPLEMENTATION.md) — Mission Control details

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
