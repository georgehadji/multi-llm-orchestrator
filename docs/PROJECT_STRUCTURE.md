# Project Structure

Multi-LLM Orchestrator - Codebase Organization

```
multi-llm-orchestrator/
├── orchestrator/               # Main package
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # CLI entry point
│   ├── cli.py                 # Command-line interface
│   │
│   ├── # Core Engine
│   ├── engine.py              # Main orchestration engine
│   ├── models.py              # Data models (Task, ProjectState, etc.)
│   ├── state.py               # State persistence
│   ├── cache.py               # Disk cache
│   │
│   ├── # API & Routing
│   ├── api_clients.py         # Unified API client
│   ├── adaptive_router.py     # Smart routing
│   ├── routing.py             # Routing tables
│   │
│   ├── # Task Management
│   ├── planner.py             # Constraint-based planning
│   ├── policy.py              # Policy definitions
│   ├── policy_engine.py       # Policy enforcement
│   ├── validators.py          # Output validators
│   │
│   ├── # Quality & Analysis
│   ├── project_analyzer.py    # Post-project analysis
│   ├── improvement_suggester.py  # Improvement suggestions
│   ├── codebase_analyzer.py   # Codebase understanding
│   │
│   ├── # Output & Organization
│   ├── output_writer.py       # Write task outputs
│   ├── output_organizer.py    # Organize output, run tests
│   ├── progress_writer.py     # Progressive output
│   │
│   ├── # Dashboards (Multiple implementations)
│   ├── dashboard.py           # Original dashboard
│   ├── dashboard_real.py      # Real-time dashboard
│   ├── dashboard_optimized.py # Performance-optimized
│   ├── dashboard_enhanced.py  # Enhanced v2.0
│   ├── dashboard_antd.py      # Ant Design v3.0
│   ├── dashboard_live.py      # ⭐ NEW: Gamified LIVE v4.0 (WebSocket)
│   │
│   ├── # Architecture & Rules
│   ├── architecture_rules.py      # Rules engine
│   ├── architecture_advisor.py    # Architecture advisor
│   ├── architecture_selector.py   # Auto-architecture selection
│   │
│   ├── # Management Systems
│   ├── knowledge_base.py      # Knowledge management
│   ├── project_manager.py     # Project management
│   ├── product_manager.py     # Product management
│   ├── quality_control.py     # Quality management
│   │
│   ├── # Telemetry & Observability
│   ├── telemetry.py           # Telemetry collection
│   ├── telemetry_store.py     # Persistent telemetry
│   ├── metrics.py             # Metrics export
│   ├── monitoring.py          # KPI monitoring
│   │
│   ├── # Utilities
│   ├── logging.py             # Structured logging
│   ├── exceptions.py          # Exception hierarchy
│   ├── streaming.py           # Streaming events
│   └── ...                    # Other utilities
│
├── scripts/                    # Utility scripts ⭐ NEW
│   ├── __init__.py
│   ├── run_dashboard.py       # Start dashboard
│   ├── run_tests.py           # Run test suite
│   ├── organize_output.py     # Organize project output
│   ├── check_models.py        # Check model availability
│   ├── cleanup_cache.py       # Clean cache files
│   └── create_project.py      # Create new project
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE_RULES.md
│   ├── ENHANCED_DASHBOARD.md
│   ├── OUTPUT_ORGANIZER.md
│   ├── PROJECT_STRUCTURE.md   # This file
│   └── ...
│
├── tests/                      # Test suite
│   ├── test_*.py
│   └── ...
│
├── projects/                   # Project files
│   └── *.yaml
│
├── outputs/                    # Default output directory
│   └── <project_id>/
│       ├── tasks/             # Task files
│       ├── tests/             # Test files
│       ├── src/               # Source code
│       └── summary.json
│
├── results/                    # Legacy output directory
│
├── examples/                   # Example scripts
│   ├── example_enhanced_dashboard.py
│   ├── example_architecture_rules.py
│   └── ...
│
├── README.md                   # Main readme
├── pyproject.toml             # Package configuration
├── setup.py                   # Setup script
├── Makefile                   # Development tasks
├── .env                       # Environment variables
├── .env.example               # Environment template
├── .gitignore
└── LICENSE
```

## 📂 Directory Descriptions

### `orchestrator/` - Main Package

Core orchestrator package containing all functionality.

#### Core Engine
- **engine.py**: Main orchestration loop (generate → critique → revise → evaluate)
- **models.py**: Data models (Task, ProjectState, Budget, etc.)
- **state.py**: Persistent state management with SQLite
- **cache.py**: Response caching with TTL

#### API & Routing
- **api_clients.py**: Unified client for all LLM providers
- **adaptive_router.py**: Smart routing with health tracking
- Handles 6 providers: OpenAI, DeepSeek, Google, Kimi, MiniMax

#### Task Management
- **planner.py**: Constraint-based task planning
- **policy.py**: Policy definitions (cost, quality, rate limits)
- **policy_engine.py**: Policy enforcement
- **validators.py**: Deterministic output validation

#### Quality & Analysis
- **project_analyzer.py**: Post-project analysis
- **improvement_suggester.py**: AI-powered improvements
- **codebase_analyzer.py**: Codebase understanding

#### Output & Organization
- **output_writer.py**: Write task outputs to files
- **output_organizer.py**: Organize output, auto-generate tests, run tests
- **progress_writer.py**: Progressive output during execution

#### Dashboards (Multiple Implementations)

| Dashboard | Status | Features |
|-----------|--------|----------|
| `dashboard.py` | Legacy | Original implementation |
| `dashboard_real.py` | Active | Real-time data |
| `dashboard_optimized.py` | Active | Performance optimized |
| `dashboard_enhanced.py` | Active | Enhanced v2.0, architecture visibility |
| `dashboard_live.py` | **NEW** | Gamified LIVE v4.0, WebSocket real-time |
| `dashboard_antd.py` | Active | Ant Design v3.0, modern UI |

#### Architecture & Rules
- **architecture_rules.py**: Rules engine with constraints
- **architecture_advisor.py**: Architecture recommendations
- **architecture_selector.py**: Auto-select optimal architecture

#### Management Systems (v5.1)
- **knowledge_base.py**: Knowledge management
- **project_manager.py**: Project management
- **product_manager.py**: Product management
- **quality_control.py**: Quality management

#### Telemetry & Observability
- **telemetry.py**: Real-time telemetry
- **telemetry_store.py**: Persistent telemetry storage
- **metrics.py**: Metrics export (Prometheus, etc.)
- **monitoring.py**: KPI monitoring with alerts

### `scripts/` - Utility Scripts

Quick utility scripts for common tasks:

```bash
# Start dashboard
python scripts/run_dashboard.py

# Run tests
python scripts/run_tests.py

# Organize project output
python scripts/organize_output.py ./output/project_123

# Check model availability
python scripts/check_models.py

# Clean cache
python scripts/cleanup_cache.py

# Create project
python scripts/create_project.py -p "Build API" -c "Tests pass" -b 5.0
```

### `docs/` - Documentation

- Architecture documentation
- Feature guides
- API documentation
- Usage examples

### `tests/` - Test Suite

Unit and integration tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python scripts/run_tests.py
```

### `projects/` - Project Files

YAML project specifications:

```yaml
# projects/my_api.yaml
project: "Build a REST API"
criteria: "All endpoints tested"
budget: 5.0
time: 3600
```

### `outputs/` - Output Directory

Default location for project outputs:

```
outputs/
└── project_123/
    ├── tasks/           # Task files
    │   ├── task_001.py
    │   └── task_002.md
    ├── tests/           # Test files
    │   ├── test_main.py
    │   └── test_utils.py
    ├── src/             # Source code
    │   ├── main.py
    │   └── utils.py
    ├── organization_report.json
    └── summary.json
```

## 🔄 Workflow

```
1. CLI Entry (cli.py)
   ↓
2. Orchestrator Engine (engine.py)
   - Architecture selection
   - Task decomposition
   - Task execution loop
   ↓
3. Output Writing (output_writer.py)
   - Write task files
   - Extract code
   ↓
4. Organization (output_organizer.py)
   - Move tasks to tasks/
   - Generate tests
   - Run tests
   - Move tests to tests/
   ↓
5. Dashboard (dashboard_antd.py)
   - Real-time monitoring
   - Architecture visibility
```

## 🎨 Dashboard Evolution

| Version | Technology | Status | Features |
|---------|------------|--------|----------|
| v1.0 | Vanilla CSS | Legacy | Basic metrics |
| v2.0 | Custom CSS | Active | Real-time data |
| v3.0 | **Ant Design** | **NEW** | Modern, professional UI |

### Ant Design Dashboard Features
- 🎨 Modern, clean UI
- 📊 Real-time data visualization
- 🏗️ Architecture decisions panel
- 🤖 Model health table
- ⚡ Task progress tracking
- 🔄 Auto-refresh (3s)
- 📱 Responsive design

## 📦 Package Exports

```python
# Core
from orchestrator import Orchestrator, Budget, Model

# Dashboard
from orchestrator import run_enhanced_dashboard
from orchestrator.dashboard_antd import run_ant_design_dashboard

# Organization
from orchestrator import OutputOrganizer, organize_project_output

# Architecture
from orchestrator import ArchitectureRulesEngine

# Management
from orchestrator import get_knowledge_base, get_project_manager
```

## 🚀 Quick Start

```bash
# 1. Install
pip install -e .

# 2. Run dashboard
python scripts/run_dashboard.py

# 3. Create project
python scripts/create_project.py \
  -p "Build a REST API" \
  -c "All endpoints tested" \
  -b 5.0

# 4. Check output
ls -la outputs/<project_id>/
```
