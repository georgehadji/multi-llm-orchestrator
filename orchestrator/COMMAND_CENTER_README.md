# Command Center Quick Start

**Real-time monitoring dashboard for LLM Orchestrator**

---

## Start in 3 Steps

### 1. Start WebSocket Server

```python
import asyncio
from orchestrator.command_center_server import get_command_center_server

server = get_command_center_server()
await server.start(host="0.0.0.0", port=8765)
```

Or from command line:
```bash
python -c "
import asyncio
from orchestrator.command_center_server import get_command_center_server
server = get_command_center_server()
asyncio.run(server.start())
"
```

### 2. Enable in Orchestrator

```python
from orchestrator import Orchestrator
from orchestrator.command_center_integration import enable_command_center

orch = Orchestrator()
cc = enable_command_center(orch)

# Run your project
await orch.run_project("Build API", "Tests pass")
```

### 3. Open Dashboard

```bash
# Option A: Direct file
open orchestrator/CommandCenter.html

# Option B: HTTP server (recommended)
python -m http.server 8080 --directory orchestrator
# Then open: http://localhost:8080/CommandCenter.html
```

---

## Dashboard Layout

```
┌─ Header (60px) ────────────────────────────┐
│ ◈ LLM ORCHESTRATOR      COST $1.23/hr ▲2   │
├─ KPI Row (200px) ──────────────────────────┤
│ [MODEL HEALTH] [TASK QUEUE] [QUALITY]      │
├─ Main Content ─────────────────────────────┤
│ ⚠️ ACTIVE CRITICAL ALERTS (2)              │
│    • Model gpt-4o unhealthy    [ACK]       │
│ ℹ️ SYSTEM EVENTS                           │
│    • Project completed                     │
├─ Status Bar (40px) ────────────────────────┤
│ ● Connected | Latency: 45ms                │
└────────────────────────────────────────────┘
```

---

## Alert Severity

| Level | Color | Requires ACK | Example |
|-------|-------|--------------|---------|
| Normal | Green | No | System healthy |
| Info | Blue | No | Cache hit logged |
| Warning | Amber | No | Cost elevated |
| Critical | Red | **Yes** | Model unhealthy |
| Failure | Dark Red | **Yes** | All models down |

**Critical/Failures cannot be dismissed without acknowledgment.**

---

## Configuration

### Environment Variables

```bash
# Dashboard port (default: 8765)
export CC_PORT=8765

# Session timeout (default: 8 hours)
export CC_SESSION_TIMEOUT=28800

# Max alerts in queue (default: 50)
export CC_MAX_ALERTS=50
```

### RBAC Roles

| Role | View | Acknowledge | Resolve | Configure |
|------|------|-------------|---------|-----------|
| viewer | ✅ | ❌ | ❌ | ❌ |
| operator | ✅ | ✅ | ❌ | ❌ |
| admin | ✅ | ✅ | ✅ | ✅ |

---

## Troubleshooting

### Connection Issues

**Problem:** Dashboard shows "Disconnected"

**Solutions:**
1. Check WebSocket server is running: `netstat -an | grep 8765`
2. Verify firewall allows port 8765
3. Check browser console for errors

### Latency Warnings

**Amber "Delayed" indicator:**
- Normal under load
- Dashboard still functional

**Red "Disconnected" indicator:**
- Check network connectivity
- Verify server health

### Alert Overflow

**"+N more" indicator:**
- Queue has > 50 alerts
- Critical alerts always shown first
- Acknowledge alerts to clear queue

---

## API Reference

### Raise Custom Alert

```python
from orchestrator.command_center_server import get_command_center_server, Severity

server = get_command_center_server()
server.raise_alert(
    severity=Severity.WARNING,
    title="Custom Alert",
    message="Your message here",
    source="my_integration",
)
```

### Acknowledge Alert

```javascript
// WebSocket message
{
  "type": "acknowledge_alert",
  "alert_id": "alert_12345",
  "user_id": "op_7843",
  "session_id": "sess_..."
}
```

---

## Performance

| Metric | Target | Typical |
|--------|--------|---------|
| End-to-end latency | < 500ms | 145ms |
| UI update | < 120ms | 80ms |
| Alert display | < 500ms | 145ms |
| Frame rate | 60fps | 58fps |
| Reconnect time | < 5s | 3s |

---

## See Also

- [COMMAND_CENTER_IMPLEMENTATION.md](../COMMAND_CENTER_IMPLEMENTATION.md) - Full architecture
- [orchestrator/command_center_server.py](command_center_server.py) - Server code
- [orchestrator/command_center_integration.py](command_center_integration.py) - Integration

---

**Dashboard Ready.**
