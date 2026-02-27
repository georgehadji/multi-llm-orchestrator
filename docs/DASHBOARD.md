# 🚀 Mission Control Dashboard

Real-time web-based monitoring and control interface for the Multi-LLM Orchestrator.

## Features

| Tab | Description |
|-----|-------------|
| 📊 **Overview** | System metrics, budget status, active models |
| 🤖 **Models** | Complete model inventory with costs & availability |
| ▶️ **Execute** | Launch new projects directly from the browser |
| 📝 **Prompts** | View and edit prompt templates |
| 📋 **Logs** | Real-time log streaming via WebSocket |

## Quick Start

### Install Dependencies

```bash
# With dashboard support
pip install -e ".[dashboard]"

# Or manually
pip install fastapi uvicorn
```

### Launch Dashboard

```bash
# Via CLI
dashboard

# With options
dashboard --host 127.0.0.1 --port 8888 --no-browser

# Via Python
python -m orchestrator.dashboard
python -c "from orchestrator import run_dashboard; run_dashboard()"
```

### Access Dashboard

Open browser to: **http://localhost:8080**

## Screenshots

The dashboard features a dark, modern interface with:

- **Cyberpunk color scheme**: Cyan/magenta accents on dark background
- **Live WebSocket updates**: Real-time data without page refresh
- **Responsive design**: Works on desktop, tablet, mobile
- **Model cost display**: Input/output pricing at a glance
- **Budget progress bars**: Visual budget tracking
- **Connection status**: Always visible connection indicator

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│   Browser UI    │ ◄────────────────► │  DashboardServer │
│  (Embedded HTML)│      /api/*        │   (FastAPI)      │
└─────────────────┘ ─────────────────► └──────────────────┘
                                                │
                                                ▼
                                       ┌──────────────────┐
                                       │   Orchestrator   │
                                       │   Engine Data    │
                                       └──────────────────┘
```

## Configuration

The dashboard auto-discovers model information from `COST_TABLE` and routing configuration from `ROUTING_TABLE`. No manual configuration needed.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard HTML |
| `/api/models` | GET | All models with cost/availability |
| `/api/routing` | GET | Routing configuration |
| `/ws` | WS | WebSocket for live updates |

## WebSocket Messages

**Client → Server:**
- `{type: "get_models"}` - Request model status
- `{type: "execute", description: "...", budget: 5.0}` - Launch project

**Server → Client:**
- `{type: "model_status", models: {...}}` - Model availability update
- `{type: "metrics", activeProjects: 3, totalCost: 1.50}` - System metrics
- `{type: "log", message: "...", timestamp: "..."}` - Log entry
- `{type: "routing", routing: {...}}` - Routing configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_HOST` | `0.0.0.0` | Bind address |
| `DASHBOARD_PORT` | `8080` | Listen port |
| `DASHBOARD_NO_BROWSER` | `false` | Disable auto-open |

## Version History

- **v1.2.0**: Initial dashboard release
  - Real-time model status
  - Budget tracking
  - Project execution UI
  - WebSocket live updates
