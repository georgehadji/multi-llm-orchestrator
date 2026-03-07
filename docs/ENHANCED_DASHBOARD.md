# Enhanced Dashboard v2.0

Το Enhanced Dashboard v2.0 παρέχει πλήρη ορατότητα στον orchestrator με real-time ενημερώσεις για:

## 🎯 Βασικά Χαρακτηριστικά

### 1. Architecture Decisions
- **Architectural Style**: Microservices, Monolith, Serverless, Event-Driven, Hexagonal, CQRS
- **Programming Paradigm**: Object-Oriented, Functional, Reactive, Declarative
- **Technology Stack**:
  - Primary Language (Python, TypeScript, etc.)
  - Frameworks (FastAPI, React, etc.)
  - Libraries (Pydantic, SQLAlchemy, etc.)
  - Databases (PostgreSQL, MongoDB, Redis, etc.)
  - Tools (Docker, pytest, black, etc.)
- **Constraints**: Architecture-specific rules
- **Patterns**: Recommended design patterns

### 2. Project Progress
- **Current Task**: Ποιο task τρέχει τώρα
- **Progress**: X/Y tasks completed (Z%)
- **Statistics**:
  - Completed tasks count
  - Failed tasks count
  - Budget used ($)
  - Elapsed time
- **Active Task Details**:
  - Task ID και Type
  - Current prompt (truncated)
  - Iteration counter (e.g., "2/3")
  - Current score
  - Model being used

### 3. Project Information
- **Project ID**: Unique identifier
- **Description**: Project description
- **Success Criteria**: Quality gates
- **Status Badge**: Idle/Running/Completed/Failed

### 4. Model Status
Για κάθε model δείχνει:
- **Availability**: ✅ Online / ❌ Offline / ⚠️ Degraded
- **Provider**: OpenAI, Google, DeepSeek, etc.
- **Metrics**:
  - Success rate (%)
  - Average latency (ms)
  - Total calls
- **Reason if Unavailable**:
  - "API key not configured"
  - "Circuit breaker tripped (3 consecutive failures)"
  - "Adaptive router: degraded"
  - "Model not found (404)"

## 🚀 Χρήση

### 1. Απλή Χρήση

```python
from orchestrator import Orchestrator, Budget
from orchestrator.dashboard_enhanced import (
    DashboardIntegration,
    EnhancedDataProvider,
    run_enhanced_dashboard
)

# Start dashboard (σε ξεχωριστό terminal)
run_enhanced_dashboard(host="127.0.0.1", port=8888)

# Στον κώδικά σου
data_provider = EnhancedDataProvider()
dashboard_integration = DashboardIntegration(data_provider)

orch = Orchestrator(budget=Budget(max_usd=10.0))
orth.set_dashboard_integration(dashboard_integration)

state = await orch.run_project(
    project_description="Build a REST API...",
    success_criteria="Handle 1000 req/s..."
)
```

### 2. Προβολή στο Browser

Ανοίξτε το dashboard στο http://127.0.0.1:8888

### 3. Auto-refresh

Το dashboard ενημερώνεται αυτόματα κάθε 3 δευτερόλεπτα.

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ◈ MISSION CONTROL v2.0                    [● Live]             │
├──────────────┬──────────────────────────────┬───────────────────┤
│              │                              │                   │
│  📋 PROJECT  │  🎯 SUCCESS CRITERIA         │  🤖 MODELS        │
│  ─────────── │  ─────────────────────       │  ─────────────    │
│  Name        │  Quality gates...            │  gpt-4o [●]       │
│  Progress    │                              │  98% | 120ms      │
│  ████████░░  │  ⚡ ACTIVE TASK              │                   │
│  8/10 tasks  │  ─────────────────           │  deepseek-r1 [○]  │
│              │  task_003: code_generation   │  ⚠ Key not set    │
│  ┌───┬───┐   │                              │                   │
│  │ 8 │ 0 │   │  Iteration 2/3               │  kimi-k2.5 [●]    │
│  │ ✓ │ ✗ │   │  Score: 0.85                 │  95% | 200ms      │
│  └───┴───┘   │                              │                   │
│              │  Prompt: Implement...        │                   │
│              │                              │                   │
├──────────────┼──────────────────────────────┴───────────────────┤
│  🏗️ ARCH     │  🔒 CONSTRAINTS    │  📐 PATTERNS               │
│  ─────────── │  ───────────────   │  ───────────               │
│  Style:      │  ✓ All typed       │  ◈ CQRS                    │
│  Microservices│  ✓ Max 10 lines   │  ◈ Event Sourcing          │
│              │  ✓ 80% coverage   │  ◈ Circuit Breaker         │
│  Paradigm:   │                   │                            │
│  Object-     │                   │                            │
│  Oriented    │                   │                            │
│              │                   │                            │
│  Stack:      │                   │                            │
│  [Python]    │                   │                            │
│  [FastAPI]   │                   │                            │
│  [PostgreSQL]│                   │                            │
└──────────────┴───────────────────┴────────────────────────────┘
```

## 🔌 Integration με Engine

### DashboardIntegration Class

```python
class DashboardIntegration:
    """Συνδέει τον orchestrator με το dashboard."""
    
    def on_project_start(self, project_id, state, architecture_rules):
        """Καλείται όταν ξεκινάει ένα project."""
        pass
    
    def on_task_start(self, task_id, task, model):
        """Καλείται όταν ξεκινάει ένα task."""
        pass
    
    def on_task_progress(self, iteration, score):
        """Καλείται σε κάθε iteration."""
        pass
    
    def on_task_complete(self, task_id, status):
        """Καλείται όταν ολοκληρώνεται ένα task."""
        pass
    
    def on_model_failure(self, model):
        """Καλείται όταν αποτυγχάνει ένα model."""
        pass
    
    def on_model_success(self, model):
        """Καλείται όταν επιτυγχάνει ένα model."""
        pass
```

### Engine Hooks

Ο Orchestrator καλεί αυτόματα τα hooks:

```python
# Στον Orchestrator
orch.set_dashboard_integration(integration)

# Αυτόματα καλείται:
# - _notify_dashboard_project_start() στο run_project()
# - _notify_dashboard_task_start() πριν από κάθε task
# - _notify_dashboard_task_progress() σε κάθε iteration
# - _notify_dashboard_task_complete() μετά από κάθε task
# - _notify_dashboard_model_success/failure() σε API calls
```

## 🎨 UI Components

### Status Badges

| Badge | Meaning |
|-------|---------|
| 🟢 Online | Model is available |
| 🔴 Offline | Model is unavailable |
| 🟠 Degraded | Model is throttled |
| 🔵 Idle | No project running |
| 🟡 Running | Project in progress |
| 🟣 Completed | Project finished |

### Progress Indicators

- **Progress Bar**: Visual task completion
- **Iteration Counter**: Current/max iterations
- **Score Badge**: Current task score (0.0-1.0)
- **Stats Grid**: Quick metrics overview

## 📡 API Endpoints

Το dashboard εκθέτει τα εξής endpoints:

```
GET /api/status          # Full status (project + architecture + task + models)
GET /api/project         # Project info
GET /api/architecture    # Architecture decisions
GET /api/active-task     # Currently running task
GET /api/models          # Model status with reasons
GET /api/metrics         # System metrics
GET /api/routing         # Routing table
```

## 🛠️ Customization

### Dark Theme Variables

```css
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #1a1a24;
    --accent-cyan: #00d4ff;
    --accent-pink: #ff4db8;
    --accent-green: #00ff88;
    --accent-yellow: #ffcc00;
    --accent-red: #ff5577;
}
```

### Refresh Rate

```python
# Στο EnhancedDataProvider
self._cache_ttl = 3  # seconds
```

## 🔧 Troubleshooting

### Dashboard δεν ενημερώνεται
1. Ελέγξτε αν το dashboard τρέχει: `curl http://127.0.0.1:8888/api/status`
2. Ελέγξτε τα logs για errors
3. Βεβαιωθείτε ότι έχετε καλέσει `set_dashboard_integration()`

### Models εμφανίζονται ως unavailable
1. Ελέγξτε τα API keys: `orchestrator check-keys`
2. Ελέγξτε το circuit breaker: 3+ failures = disabled
3. Ελέγξτε τον adaptive router state

### Architecture decisions δεν εμφανίζονται
1. Βεβαιωθείτε ότι έχετε architecture_rules.py
2. Ελέγξτε ότι η `_generate_architecture_rules()` επιστρέφει αποτέλεσμα

## 📈 Performance

- **Refresh Rate**: 3 seconds (configurable)
- **Memory**: ~50MB για το dashboard
- **CPU**: <1% όταν idle
- **Latency**: <100ms για API responses

## 📝 Changelog

### v2.0 (Current)
- ✅ Architecture decisions visibility
- ✅ Real-time task progress
- ✅ Model status with reasons
- ✅ Project details panel
- ✅ Success criteria display
- ✅ Constraints & patterns lists
- ✅ Auto-refresh (3s)
- ✅ Responsive layout

### v1.0 (Legacy)
- Basic metrics
- Model list
- Static data

## 🔮 Roadmap

- [ ] WebSocket support για true real-time updates
- [ ] Historical data charts
- [ ] Cost analysis graphs
- [ ] Task dependency graph visualization
- [ ] Export reports
- [ ] Multi-project view
