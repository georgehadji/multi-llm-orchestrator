# dashboard-start

Launch and navigate the AI Orchestrator monitoring dashboard. Use when the user wants to start the dashboard, check model rankings, or view profiling data.

## Start the dashboard

```bash
# Option 1: Python script (if present)
python start_dashboard.py

# Option 2: Direct uvicorn (always works)
python -m uvicorn orchestrator.dashboard_core.core:create_app \
  --factory --host 0.0.0.0 --port 8888 --reload

# Option 3: CLI entry point
python -m orchestrator dashboard

# Option 4: Installed script
dashboard
```

**Dashboard URL:** http://localhost:8888

## Install dashboard dependencies

```bash
pip install -e ".[dashboard]"
# Installs: fastapi, uvicorn[standard], websockets, httpx
```

## Key API routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web UI |
| `/api/models` | GET | Model rankings, cost stats, success rates |
| `/api/projects` | GET | All project runs and status |
| `/api/telemetry` | GET | Live EMA latency + trust scores |
| `/api/nexusscope/sessions` | GET | NexusScope profiling sessions list |
| `/api/nexusscope/sessions?name=<n>&last_n=20` | GET | Filter sessions by name |
| `/api/nexusscope/report` | GET | Last profiling report (text) |
| `/api/nexusscope/report?fmt=html` | GET | Flame graph HTML |
| `/api/nexusscope/report?fmt=json` | GET | Raw JSON for processing |
| `/api/nexusscope/report?fmt=speedscope` | GET | Speedscope format |

## With NexusScope profiling enabled

```bash
# Start orchestrator with profiling
ORCHESTRATOR_PROFILING=1 python -m orchestrator --project "..." --budget 1.0

# Then view results
curl http://localhost:8888/api/nexusscope/sessions
curl "http://localhost:8888/api/nexusscope/report?fmt=html" > profile.html

# Or via CLI
python -m orchestrator nexusscope sessions
python -m orchestrator nexusscope report --format html --output profile.html
```

## Install profiling dependency

```bash
pip install -e ".[profiling]"
# Installs: pyinstrument>=4.6
```

## WebSocket live feed

The dashboard streams live telemetry over WebSocket at `ws://localhost:8888/ws`.
Connects automatically when the web UI is open.
