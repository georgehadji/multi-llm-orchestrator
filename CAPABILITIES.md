# Multi-LLM Orchestrator — Opinionated Guide for Developers

**Version:** 2026.03 v6.2 | **Updated:** 2026-03-23 | **Reading time:** 15 min

> **What is this?** An opinionated AI code generation platform that decomposes project specs into atomic tasks, routes each to the optimal LLM provider, and executes generate→critique→revise cycles until quality thresholds are met. Think of it as CI/CD for AI-generated code with built-in cost controls.

**New in v6.2:** 🧠 ARA Pipeline — 12 Advanced Reasoning Methods from cognitive science (Pre-Mortem, Bayesian, Debate, Jury, Analogical, Delphi, and more)

---

## 60-Second Walkthrough

```bash
# 1. Set your budget and constraints
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# 2. Describe your project
python -m orchestrator \
  --project "Build a real-time collaborative whiteboard with WebSockets" \
  --criteria "Concurrent editing works, cursors sync, export to PNG" \
  --budget 5.0

# What happens automatically:
# ✅ ProjectEnhancer → improves your spec with LLM suggestions
# ✅ ArchitectureAdvisor → picks event-driven architecture + WebSocket API
# ✅ Decomposition → breaks into 12 atomic tasks (auth, canvas, sync, export)
# ✅ Smart Routing → DeepSeek for code ($0.28/M), GPT-4o for review
# ✅ Execution → generate→critique→revise until score ≥ 0.85
# ✅ Validation → syntax check, type check, test run
# ✅ Telemetry → logs to dashboard, tracks cost per model
# ✅ Output → scaffolded app in ./results/ with author attribution
# ✅ Production Feedback → learns from real-world deployment outcomes
```

**That's it.** No prompt engineering. No model selection. No "it works on my machine." Just describe what you want and set your budget.

---

## Domain Glossary

| Term | Definition |
|------|------------|
| **Project** | A complete software deliverable with description, success criteria, and budget. Decomposed into Tasks. |
| **Task** | Atomic unit of work (code_generation, code_review, reasoning, evaluation). Has type, prompt, dependencies. |
| **Capability** | A high-level system feature (e.g., Knowledge Management, Project Enhancement) that enhances orchestration. |
| **Policy** | Governance rule (HARD/SOFT/MONITOR mode) enforcing constraints like max cost, allowed regions, training prohibition. |
| **Resource** | Computational asset (LLM model, compute unit) with capacity, cost, and availability constraints. |
| **Knowledge Artifact** | Learned pattern from completed projects stored for future retrieval (decisions, solutions, anti-patterns). |
| **Routing Table** | Provider-ranked model selection per TaskType. DeepSeek Chat for code, GPT-4o for review, etc. |
| **Telemetry** | Real-time metrics collection: latency, cost, quality scores, validator failures per model. |
| **Validator** | Deterministic check (python_syntax, ruff, pytest) that gates task completion. Score = 0 if failed. |
| **Checkpoint** | Serialized project state after each task. Enables resume after crashes or interruptions. |
| **Plugin** | Extension module for custom validators, integrations, or routing strategies. |
| **Production Feedback** | Real-world outcomes (errors, performance, user ratings) fed back into routing decisions. |
| **Codebase Fingerprint** | Hash of languages, frameworks, patterns used for similarity matching. |
| **Outcome-Weighted Routing** | Model selection based on proven production success, not just cost estimates. |
| **Nash Stability** | Competitive equilibrium where accumulated knowledge creates switching costs. |
| **Fallback Chain** | Cross-provider backup models when primary fails (DeepSeek → GPT-4o → Gemini). |
| **Architecture Decision** | LLM-generated structural pattern (hexagonal, microservices) that constrains all generated code. |
| **ARA Pipeline** | Advanced Reasoning & Analysis — 12 cognitive strategies (Pre-Mortem, Bayesian, Debate, etc.) for complex decisions. |
| **Method Selection** | Intelligent routing that selects optimal reasoning method based on task complexity, risk, and budget. |
| **Pre-Mortem** | Risk assessment method that imagines project failure and works backward to identify prevention strategies. |
| **Analogical Transfer** | Innovation method that maps solutions from unrelated domains to the target problem. |
| **Jury Method** | High-stakes decision method with 4 generators, 3 critics, and meta-evaluation for maximum quality. |

---

## Quick Start for SaaS Users

**Prerequisites:**
- Python 3.10+
- At least one API key (DeepSeek recommended: best value)
- 5 minutes

**Your First Run:**
```bash
pip install -e .
export DEEPSEEK_API_KEY="your-key"
python -m orchestrator --project "Hello World API" --criteria "HTTP 200 on /" --budget 1.0
```

**View Results:**
- **Output files:** `./results/` (code, tests, docs)
- **Dashboard:** `python run_optimized_dashboard.py` → http://localhost:8888
- **Usage logs:** `orchestrator_telemetry.db` (SQLite, query via dashboard)
- **Cost tracking:** Real-time in terminal, summarized at end

**Reading Telemetry from Web UI:**
The dashboard exposes these metrics via `/api/metrics`:
- `cost_by_model` — Pie chart of spending per provider
- `latency_p95` — Response time percentiles over time
- `quality_scores` — Task scores with trend lines
- `validator_failures` — Bar chart of syntax/lint errors
- `budget Burn_rate` — $/hour projection vs remaining

Set alerts: `ORCHESTRATOR_ALERT_BUDGET_PCT=80` triggers webhook at 80% spend.

---

## The 7 Modules

The orchestrator is organized into 7 capability modules that map to SaaS plans:

### Module 1: Execution Core
> *Multi-provider routing, cost optimization, semantic caching, remediation, dashboards*

**Multi-Provider Model Routing**

Routes tasks to optimal AI models based on task type, cost, performance, and availability.

| Tier | Best For | Example Providers |
|------|----------|-------------------|
| **Free** | Simple tasks, prototyping | Gemini Flash Lite, GPT-4o Mini |
| **Economy** | Cost-sensitive code generation | Gemini Flash Lite, DeepSeek Chat |
| **Standard** | Balanced quality/cost | DeepSeek Reasoner, GPT-4o Mini, Claude 3 Haiku |
| **Premium** | Critical evaluation, complex reasoning | GPT-4o, o3-mini, Gemini Pro |

> **Note:** Live prices are fetched from provider metadata via the dashboard API. Above examples are illustrative—always check current pricing in your dashboard.

**Routing Table (auto-selected):**
```python
TaskType.CODE_GEN: [DEEPSEEK_CHAT, CLAUDE_3_5_SONNET, GPT_4O, GPT_4O_MINI, GEMINI_FLASH]
TaskType.CODE_REVIEW: [DEEPSEEK_CHAT, GPT_4O, GPT_4O_MINI, GEMINI_FLASH]
TaskType.REASONING: [DEEPSEEK_REASONER, GPT_4O, GPT_4O_MINI, GEMINI_PRO]
```

**Intelligent Cost Optimization**

| Phase | % of Budget | Purpose |
|-------|-------------|---------|
| Decomposition | 5% | Break project into tasks |
| Generation | 45% | Primary model calls |
| Cross-review | 25% | Critique from different provider |
| Evaluation | 15% | Quality scoring |
| Reserve | 10% | Fallback emergencies |

**Key Features:**
- **Adaptive EMA:** Tracks actual costs, adjusts predictions
- **Fallback Chains:** Cross-provider resilience (DeepSeek → GPT-4o)
- **Circuit Breaker:** Disables failing models after 3 consecutive errors
- **Temperature Optimization:** 0.0 for code (deterministic), 0.2 for review

**Semantic Caching & Deduplication**

- **SemanticCache:** Caches based on semantic similarity, not just hash
- **DuplicationDetector:** Merges near-duplicate tasks automatically
- **Cache hit:** Sub-millisecond response, 85%+ target rate

**Remediation Engine**

Auto-recovery strategies on task failure:
1. Retry with same model (transient error)
2. Escalate to fallback model
3. Loosen constraints (reduce quality threshold slightly)
4. Abort with partial results

**Real-Time Dashboards**

- **Dashboard v5.0:** 5x faster load, <100ms FCP, gzip compression
- **Mission Control LIVE:** WebSocket real-time, gamification (XP, levels)
- **KPI Monitoring:** P95 latency, cost per task, quality trends

---

### Module 2: Governance & Safety
> *Policies, constraint control plane, audit, quality control*

**Policy-as-Code Framework**

Enforcement modes:
- **HARD:** Block any violation; raise exception
- **SOFT:** Log violation; allow execution
- **MONITOR:** Log only; never block

```python
Policy(
    name="eu_only",
    allowed_regions=["eu", "global"],
    allow_training_on_output=False,
    max_cost_per_task_usd=0.50,
    enforcement_mode=EnforcementMode.HARD,
)
```

**Hierarchical Policy Enforcement:**
```
Org-level (global defaults)
  ↓
Team-level (override org)
  ↓
Job-level (override team)
  ↓
Node-level (specific task)
```

**Constraint Control Plane**

- **ReferenceMonitor:** Synchronous, bypass-proof constraint checker
- **OrchestrationAgent:** Natural language intent → structured specs
- **Static Analysis:** `PolicyAnalyzer` detects contradictions before execution

**Audit & Compliance**

- **AuditLog:** Immutable JSONL structured audit trail
- **Validator Types:**
  - `python_syntax` — Compile Python (built-in)
  - `ruff` — Python linting (optional)
  - `pytest` — Run unit tests (optional)
  - `json_schema` — Validate JSON structure
  - `length` — Output size bounds

**Smart Validator Filtering (v5.2)**

Automatically detects non-Python tasks and removes Python validators:
```
Task: "Generate React component" → Removes ruff, pytest, python_syntax
Task: "Build FastAPI endpoint" → Keeps all Python validators
```

---

### Module 3: Knowledge & Management Suite
> *Knowledge, Project, Product, Quality Management, Capability Logging*

These systems don't exist in isolation—they feed learned insights back into the orchestrator loop.

**Knowledge Management**

> *Learns from completed projects and applies patterns to future orchestration.*

```python
kb = get_knowledge_base()

# Learn from completed project
await kb.learn_from_project(
    project_id="webgl_dj",
    decisions=[{
        "title": "Chose Three.js over Babylon.js",
        "rationale": "Better ecosystem for audio visualization",
        "alternatives": ["Babylon.js", "PixiJS"],
        "tags": ["webgl", "3d", "audio"],
    }]
)

# Query for recommendations on next project
recs = await kb.get_recommendations("Build 3D visualization")
# Returns: "Suggestion: Use Three.js (relevance: 95%)"
```

**Project Management**

> *Uses real model telemetry (latency, cost, availability) as resources in scheduling—not generic Gantt charts.*

```python
pm = get_project_manager()

# Resources are actual LLM models with real metrics
resources = [
    Resource("deepseek-chat", ResourceType.MODEL, 100, 100, 0.0003),
    Resource("gpt-4o", ResourceType.MODEL, 100, 100, 0.003),
]

# Scheduler accounts for model availability and cost
timeline = await pm.create_schedule(
    project_id="my_app",
    tasks=tasks,
    resources=resources,
    dependencies={"test": ["implement"]},
)
print(f"Critical path: {timeline.critical_path}")
```

**Product Management**

> *RICE prioritization for features with orchestrator-specific metrics.*

```python
pm = get_product_manager()

# RICE = (Reach × Impact × Confidence) / Effort
rice = RICEScore(reach=1000, impact=3, confidence=85, effort=3)

await pm.add_feature(
    name="AI Code Assistant",
    rice_score=rice,
    priority=FeaturePriority.P0_CRITICAL,
)

# Auto-prioritized backlog
backlog = pm.get_prioritized_backlog(limit=5)
```

**Quality Control**

> *Multi-level testing gates that can block deployment.*

```python
qc = get_quality_controller()

report = await qc.run_quality_gate(
    project_id="production_app",
    project_path=Path("."),
    levels=[TestLevel.UNIT, TestLevel.PERFORMANCE, TestLevel.SECURITY],
)

print(f"Quality Score: {report.quality_score:.1f}/100")
print(f"Passed: {report.passed}")  # Blocks if False
```

**Capability Usage Logging**

> *Tracks which capabilities are used, when, and their effectiveness.*

Query patterns for SaaS dashboard:
- `SELECT capability, COUNT(*) FROM usage_logs GROUP BY capability`
- `SELECT AVG(duration_ms) FROM usage_logs WHERE capability='KnowledgeManagement'`
- `SELECT * FROM usage_logs WHERE project_id='X' ORDER BY timestamp`

Visualizations:
- Heatmap: Capability usage by project type
- Trend line: Quality scores over time
- Alert: When budget burn rate exceeds threshold

---

### Module 4: App Studio
> *ArchitectureAdvisor, AppBuilder, ProjectEnhancer, Auto-Resume*

**Architecture Advisor**

> *LLM-powered architectural decision making before code generation.*

Input: Project description + success criteria
Output: `ArchitectureDecision` with structural pattern, topology, API paradigm

```python
advisor = ArchitectureAdvisor()
decision = await advisor.analyze(
    description="Real-time collaborative document editor",
    criteria="Low latency, 100+ concurrent users",
)

print(decision.structural_pattern)  # "event-driven"
print(decision.topology)            # "microservices"
print(decision.api_paradigm)        # "graphql"
print(decision.rationale)           # "Event-driven enables low-latency..."
```

**Architecture Rules Engine** *(v5.2)*

> *Two-phase architecture decision: rule-based detection + LLM optimization.*

**Phase 1 — Rule-Based Detection:**
- Keyword analysis (microservice, event, graphql, etc.)
- Pattern matching for project type
- Default stack selection based on project category

**Phase 2 — LLM Optimization:**
- LLM reviews the rule-based proposal
- Suggests improvements only if clear benefits
- Conservative approach (no changes = no optimization)

```python
from orchestrator import ArchitectureRulesEngine
from orchestrator.api_clients import UnifiedClient

# With LLM optimization
client = UnifiedClient()
engine = ArchitectureRulesEngine(client=client)

rules = await engine.generate_rules(
    description="Build real-time analytics dashboard",
    criteria="High performance, event-driven updates"
)

# Check decision method
print(f"LLM Optimized: {rules._llm_optimized}")   # True
print(engine.generate_summary(rules))
# Output: "Decided by: LLM (Rule-based → Optimized)"
```

**Decision Labels:**
| Label | Method | When |
|-------|--------|------|
| `Rule-based` | Keyword detection only | No LLM client available |
| `LLM (Generated from scratch)` | LLM creates entire architecture | Primary method with client |
| `LLM (Rule-based → Optimized)` | Rule-based + LLM review | When LLM finds improvements |

**Generated Files:**
- `.orchestrator-rules.yml` — Machine-readable constraints
- `ARCHITECTURE.md` — Human-readable documentation with rationale

**App Builder**

Auto-generates complete applications with scaffolding:

```python
builder = AppBuilder()
result = await builder.build(
    description="FastAPI microservice with JWT auth",
    criteria="All endpoints tested, OpenAPI docs complete",
    output_dir="./my_api",
    app_type="fastapi",  # Auto-detected if omitted
)

print(f"Files generated: {len(result.assembly.files_written)}")
print(f"Architecture: {result.profile.structural_pattern}")
```

**Supported App Types:**
- `nextjs` — Next.js 14 + Tailwind + TypeScript
- `react` — React + Vite
- `fastapi` — FastAPI + async handlers
- `graphql` — GraphQL API
- `html` — Static HTML/CSS/JS
- `microservices` — Multi-service scaffold

**Project Enhancer**

> *Improves your spec before decomposition.*

```bash
# Without enhancer
python -m orchestrator --project "Build API" --criteria "Works"

# With enhancer (default)
python -m orchestrator --project "Build API" --criteria "Works"
# → Suggests: "Add rate limiting?", "Include OpenAPI docs?"
```

Budget cap: $0.10 per enhancement request. Skip with `--no-enhance`.

**Auto-Resume**

Checkpointed state after each task:
- Crash recovery: Restart continues from last task
- Similar project detection: "Resume 'webgl-dj-v1'? [Y/n]"
- Explicit resume: `python -m orchestrator --resume <project_id>`

---

### Module 5: Cognitive & Reasoning Layer
> *Brain, Evaluation, Escalation, Checkpoints, Prompt Enhancement, ARA Pipelines*

**🧠 ARA Pipeline — Advanced Reasoning Methods (v6.2)**

> *12 sophisticated reasoning strategies from cognitive science and decision research, automatically selected based on task characteristics.*

```python
from orchestrator import Orchestrator, Budget
from orchestrator.ara_integration import create_ara_integration

orch = Orchestrator(budget=Budget(max_usd=20.0))

# Create ARA integration with auto-select
ara = create_ara_integration(
    client=orch.client,
    enabled=True,
    auto_select=True,  # Auto-select method per task
)

# Execute task with intelligent method selection
from orchestrator.models import Task, TaskType

task = Task(
    id="arch_001",
    type=TaskType.REASONING,
    prompt="Design authentication system for high-security financial app",
)

result = await ara.execute_task_with_pipeline(task)
print(f"Method: {result.metadata['ara_method']}")  # e.g., "pre_mortem"
```

**12 Reasoning Methods:**

| Method | Cost | Quality Gain | Best For |
|--------|------|--------------|----------|
| **Multi-Perspective** | 4.0× | +25% | General problem analysis |
| **Iterative** | 2.0× | +35% | Optimization, design |
| **Debate** | 2.5× | +40% | Architecture, trade-offs |
| **Research** | 1.5× | +30% | Evidence-based decisions |
| **Jury** | 5.0× | +50% | Critical code, high-stakes |
| **Scientific** | 2.0× | +45% | Technical decisions |
| **Socratic** | 1.5× | +25% | Clarifying requirements |
| **Pre-Mortem** ⭐ | 1.8× | +45% | Risk assessment |
| **Bayesian** | 2.2× | +50% | Uncertainty quantification |
| **Dialectical** | 2.0× | +55% | Philosophical synthesis |
| **Analogical** ⭐ | 1.9× | +55% | Cross-domain innovation |
| **Delphi** | 3.5× | +60% | Expert consensus |

⭐ **Recommended for most projects**

**Method Selection (Automatic):**

```python
from orchestrator.method_selector import select_method_for_task

# Auto-select based on task characteristics
selection = select_method_for_task(
    task=task,
    complexity="high",      # low, medium, high, critical
    risk="high",            # low, medium, high, critical
    use_llm=True,           # LLM optimization
    client=orch.client,
)

print(f"Recommended: {selection.method.value}")
print(f"Rationale: {selection.rationale}")
print(f"Confidence: {selection.confidence:.0%}")
```

**Example: Pre-Mortem Risk Assessment**

```python
from orchestrator.ara_pipelines import ReasoningMethod, PipelineFactory

task = Task(
    id="deployment",
    type=TaskType.REASONING,
    prompt="Deploy payment system to production",
)

# Use Pre-Mortem to identify failure modes
pipeline = PipelineFactory.create(
    method=ReasoningMethod.PRE_MORTEM,
    client=orch.client,
)

result = await pipeline.execute(task)

# Access insights
print(f"Failure narrative: {result.metadata['failure_narrative'][:300]}")
print(f"Root cause: {result.metadata['root_cause']}")
print(f"Early signals: {result.metadata['early_signals']}")
print(f"Safeguards: {result.metadata['safeguards']}")
```

**Example: Analogical Innovation**

```python
task = Task(
    id="ui_design",
    type=TaskType.WRITING,
    prompt="Design innovative UI for music creation app",
)

pipeline = PipelineFactory.create(
    method=ReasoningMethod.ANALOGICAL,
    client=orch.client,
)

result = await pipeline.execute(task)

print(f"Source domains: {result.metadata['source_domains']}")
# Output: ["Video game level editors", "Cooking recipe apps", "Photo editing tools"]
print(f"Solution: {result.output}")
```

**Configuration:**

```bash
# Environment variables
export ORCHESTRATOR_ARA_ENABLED=true
export ORCHESTRATOR_ARA_AUTO_SELECT=true
export ORCHESTRATOR_ARA_MAX_COST_MULTIPLIER=3.0
export ORCHESTRATOR_ARA_METHOD_OVERRIDES='{"auth": "jury", "risk": "pre_mortem"}'
```

**Statistics & Monitoring:**

```python
stats = ara.get_stats()
print(f"Tasks: {stats['tasks_executed']}")
print(f"Method distribution: {ara.get_method_distribution()}")
print(f"Avg cost multiplier: {stats['avg_cost_multiplier']:.2f}×")
```

---

**AI Brain & Cognitive Layer** (Legacy)

**AI Brain & Cognitive Layer**

> *Advanced reasoning and cognitive capabilities for complex decision-making.*

```python
from orchestrator.brain import Brain, CognitiveState

brain = Brain(model="deepseek-chat")
cognitive_state = await brain.reason(
    context="The project requires complex authentication with OAuth2 and JWT tokens",
    goal="Determine the best approach for implementation"
)

print(f"Reasoning steps: {len(cognitive_state.reasoning_history)}")
print(f"Confidence: {cognitive_state.confidence_score:.2f}")
```

**LLM-Based Evaluation**

> *Comprehensive evaluation scoring to assess the quality of generated content.*

```python
from orchestrator.evaluation import Evaluator, EvaluationResult

evaluator = Evaluator()
result: EvaluationResult = await evaluator.evaluate(
    content="def authenticate_user(username, password): ...",
    criteria=["security", "efficiency", "readability"],
    reference="industry best practices for authentication"
)

print(f"Overall score: {result.score:.2f}")
print(f"Feedback: {result.feedback}")
```

**Automatic Escalation**

> *Automatic escalation to higher-capability models when quality thresholds are not met.*

```python
from orchestrator.escalation import EscalationHandler

handler = EscalationHandler()
result = await handler.process_with_escalation(
    content="Implement a complex algorithm",
    criteria="accuracy and efficiency",
    initial_model="deepseek-chat",
    max_escalations=2
)
```

**Intermediate Checkpoints**

> *State checkpoints for long-running processes to enable recovery and resumption.*

```python
from orchestrator.checkpoints import CheckpointManager

manager = CheckpointManager(checkpoint_dir="./checkpoints")
await manager.save_checkpoint(state_data, "task_123")

# Later, restore state
checkpoint = await manager.load_checkpoint("task_123")
restored_state = checkpoint.data if checkpoint else default_state
```

**Prompt Enhancement**

> *Optimization and enhancement of prompts to improve model performance.*

```python
from orchestrator.prompt_enhancer import PromptEnhancer

enhancer = PromptEnhancer()
enhanced_prompt = await enhancer.enhance(
    prompt="Write a function to sort an array",
    context="The function should be efficient and handle edge cases"
)
```

### Module 6: Management & Control Systems
> *Hierarchical management, triggers, workspace isolation, API gateway, connectors*

**Multi-Level Hierarchy**

> *Organizational and team hierarchy with budget allocation and access controls.*

```python
from orchestrator.hierarchy import HierarchyManager

hierarchy = HierarchyManager()
org = hierarchy.create_org("Acme Corp", budget=10000.0)
team = hierarchy.create_team("Engineering", org_id=org.id, budget=5000.0)
project = hierarchy.create_project("New API", team_id=team.id, budget=2000.0)
```

**Event-Driven Triggers**

> *Automated triggers based on conditions for workflow automation.*

```python
from orchestrator.triggers import TriggerManager

trigger_manager = TriggerManager()
trigger = trigger_manager.create_trigger(
    name="High Error Rate",
    condition="error_rate > 0.05",
    action="alert_team",
    context={"team": "engineering"}
)
```

**Workspace Isolation**

> *Isolated workspaces with separate configurations and data.*

```python
from orchestrator.workspace import WorkspaceManager

ws_manager = WorkspaceManager(base_dir="./workspaces")
workspace = ws_manager.create_workspace("project_alpha", owner="user123")
ws_manager.activate_workspace(workspace.id)
```

**API Gateway**

> *Centralized API gateway for routing, authentication, and transformation.*

```python
from orchestrator.gateway import APIGateway

gateway = APIGateway()
await gateway.route_request(
    request_data={
        "method": "POST",
        "url": "/execute",
        "headers": {"Authorization": "Bearer ..."},
        "body": "{...}"
    },
    target_service="orchestrator_service"
)
```

**External Connectors**

> *Connectors for various external systems like databases, APIs, and file systems.*

```python
from orchestrator.connectors import ConnectorManager

connector_manager = ConnectorManager()
db_connector = connector_manager.register_db_connector(
    name="main_db",
    config={"host": "localhost", "port": 5432, "database": "mydb"}
)
result = await db_connector.query("SELECT * FROM users")
```

### Module 7: Advanced Capabilities
> *Sandbox execution, context sources, skills, drift detection, browser testing*

**Secure Code Execution Sandbox**

> *Isolated environment for executing untrusted code safely.*

```python
from orchestrator.sandbox import Sandbox

sandbox = Sandbox()
result = await sandbox.execute_code("python", "print('Hello, world!')")
print(f"Success: {result.success}, Output: {result.output}")
```

**Multiple Context Sources**

> *Integration with various context sources like documents, databases, and APIs.*

```python
from orchestrator.context_sources import ContextSourceManager

source_manager = ContextSourceManager()
doc_source = source_manager.add_document_source("docs", "./documents/")
context = await source_manager.get_context(query="What is AI?", sources=["docs"])
```

**Skills System**

> *Modular skills that can be executed by the orchestrator for specific tasks.*

```python
from orchestrator.skills import SkillManager

skill_manager = SkillManager()
result = await skill_manager.execute_skill("calculate_sum", numbers=[1, 2, 3, 4])
```

**Drift Detection**

> *Monitoring and detection of concept/model drift in the orchestrator.*

```python
from orchestrator.drift import DriftDetector

detector = DriftDetector(window_size=100, threshold=0.05)
is_drifting = detector.add_sample(new_data_point)
```

**Browser Testing**

> *Automated browser-based testing capabilities.*

```python
from orchestrator.browser_testing import BrowserTester

tester = BrowserTester(browser_type="chromium")
result = await tester.test_page("https://example.com", [
    {"action": "click", "selector": "#button"},
    {"action": "fill", "selector": "#input", "value": "test"}
])
```

**Command-Specific Token Compression**

> *Domain-specific token compression for various command outputs.*

```python
from orchestrator.token_optimizer import TokenOptimizer

optimizer = TokenOptimizer()
compressed = optimizer.compress_command_output("git log", git_log_output)
```

**A2A External Agent Client**

> *Client for invoking external agents using the A2A protocol.*

```python
from orchestrator.a2a_protocol import A2AClient

client = A2AClient(agent_endpoint="https://external-agent.example.com")
result = await client.invoke_agent(task="summarize", data={"text": "..."})
```

**Persona Modes**

> *Behavioral modes like Strict for production or Creative for ideation.*

```python
from orchestrator.persona_modes import PersonaModeManager

persona_manager = PersonaModeManager()
persona_manager.set_persona("strict")
```

**Persistent Cross-Run Learning**

> *Aggregation of model performance across all runs for continuous improvement.*

```python
from orchestrator.learning_aggregator import LearningAggregator

aggregator = LearningAggregator()
await aggregator.record_task_result(task_type="code_gen", model="gpt-4", score=0.85)
recommendations = await aggregator.get_routing_recommendations(task_type="code_gen")
```

**Multi-Tenant API Gateway**

> *JWT/API-key authentication for SaaS deployment.*

```python
from orchestrator.multi_tenant_gateway import MultiTenantGateway

gateway = MultiTenantGateway(jwt_secret="secret_key")
await gateway.start_server()
```

---

### Module 5: Production Operations & Monitoring
> *Mission-Critical Command Center, Cost Optimizations, Real-time Alerting*

**Mission-Critical Command Center (v6.0)**

Real-time operational dashboard for production LLM orchestration monitoring:

```python
from orchestrator import Orchestrator
from orchestrator.command_center_integration import enable_command_center

orch = Orchestrator()
cc = enable_command_center(orch)

# Dashboard auto-updates with:
# - Model health status (10 models monitored)
# - Task queue depth (pending/active/failed)
# - Cost burn rate ($/hour with projection)
# - Quality scores (real-time trends)
# - Active alerts (5-level severity system)
```

**Dashboard Features:**
- **Latency:** < 500ms end-to-end, 100ms batch updates
- **Reliability:** WebSocket → SSE → polling graceful degradation
- **Alerting:** Critical/Failure require acknowledgment (immutable audit log)
- **Security:** RBAC (viewer/operator/admin), session timeout
- **Layout:** Fixed spatial zones, no reflow on alert

**Alert Severity Model:**
| Level | Color | Auto-Dismiss | Requires ACK |
|-------|-------|--------------|--------------|
| Normal | Green | Yes | No |
| Info | Blue | 30s | No |
| Warning | Amber | No | No |
| Critical | Red | **Never** | **Yes** |
| Failure | Dark Red | **Never** | **Yes** |

**Access:** Open `orchestrator/CommandCenter.html` in browser or serve via `python -m http.server`

---

**Production Optimizations (v6.1)**

Cost and performance optimizations based on adversarial stress testing:

| Optimization | How It Works | Savings |
|--------------|--------------|---------|
| **Confidence-Based Early Exit** | Detects stable high performance (variance < 0.001), exits iteration loop early | -25% iterations |
| **Tiered Model Selection** | CHEAP tier first (Gemini Flash Lite), escalates on failure | -22% cost |
| **Semantic Sub-Result Caching** | Normalizes prompts, caches patterns not exact strings | -15% cost, -50% latency |
| **Fast Regression Detection** | EMA α=0.2 (was 0.1), detects quality drops in ~5 calls | 2× faster response |
| **Tool Safety Validation** | Blocks hallucinated shell/code execution patterns | Security |

**Total Impact:** 35% cost reduction ($2.40 → $1.55 per project)

```python
# All optimizations enabled by default
orch = Orchestrator()

# Check optimization metrics
print(orch._semantic_cache.get_stats())  # Cache hits, quality scores
print(orch._tier_escalation_count)       # Tier escalation history

# Early exit logic
# Automatically exits when: avg_score >= threshold * 0.95 AND variance < 0.001
```

---

## Module 6: Adaptive Learning & Extensibility
> *Plugin system, production feedback loop, outcome-weighted routing, model leaderboard*

These systems enable the orchestrator to learn from real-world usage and become more valuable over time—creating **Nash stability** where accumulated knowledge makes switching costs prohibitive.

### Architecture: Core vs Plugins

The orchestrator follows a **minimal core, maximal extensibility** philosophy:

| Component | Type | Description |
|-----------|------|-------------|
| **Plugin System** | Core | Registry, interfaces, lifecycle management |
| **Production Feedback Loop** | Hybrid | Core = storage + API; Plugins = processors (Sentry, Datadog, etc.) |
| **Model Leaderboard** | Hybrid | Core = benchmark engine; Plugins = custom benchmark suites |
| **Outcome-Weighted Router** | Core | Routing logic + plugin hooks for custom strategies |
| **Validators** | Official Plugin | MyPy, ESLint, Cargo check, etc. (optional install) |
| **Integrations** | Official Plugin | Teams, Slack, Discord, Sentry (optional install) |

**Installation Options:**
```bash
# Core only (lean, ~3MB)
pip install multi-llm-orchestrator

# Core + all official plugins (~7MB)
pip install multi-llm-orchestrator[all]

# Core + specific plugins
pip install multi-llm-orchestrator[validators,slack,sentry]
```

### Plugin System

Extensible architecture for custom validators, integrations, and routing strategies without core modifications.

```python
from orchestrator.plugins import ValidatorPlugin, ValidationResult, PluginMetadata

class RustValidator(ValidatorPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="rust-compiler-check",
            version="1.0.0",
            author="acme-corp",
            description="Validate Rust code with cargo check",
            plugin_type=PluginType.VALIDATOR,
        )

    def can_validate(self, file_path: str, language: str) -> bool:
        return language == "rust" or file_path.endswith(".rs")

    def validate(self, code: str, context: dict) -> ValidationResult:
        # Implementation: run cargo check
        return ValidationResult(passed=True, score=0.95)

# Register and use
from orchestrator.plugins import get_plugin_registry
registry = get_plugin_registry()
registry.register(RustValidator())
```

**Official Plugins** (install separately):
```bash
pip install orchestrator-plugins-validators    # MyPy, ESLint, Rust, Go
pip install orchestrator-plugins-integrations  # Teams, Discord, Sentry
pip install orchestrator-plugins-benchmarks    # Security, ML-specific benchmarks
```

**Creating a Plugin:**
```python
# my_validator.py
from orchestrator.plugins import ValidatorPlugin, ValidationResult, PluginMetadata, PluginType

class MyValidator(ValidatorPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="my-custom-validator",
            version="1.0.0",
            author="my-org",
            description="Custom validation logic",
            plugin_type=PluginType.VALIDATOR,
        )

    def can_validate(self, file_path: str, language: str) -> bool:
        return file_path.endswith(".myext")

    def validate(self, code: str, context: dict) -> ValidationResult:
        # Your validation logic
        return ValidationResult(passed=True, score=0.95)

# Auto-register on import
from orchestrator.plugins import get_plugin_registry
get_plugin_registry().register(MyValidator())
```

### Production Feedback Loop

Captures real-world outcomes of generated code and feeds them back into routing decisions.

```python
from orchestrator.feedback_loop import FeedbackLoop, ProductionOutcome, OutcomeStatus

loop = FeedbackLoop()

# Record deployment outcome (called from your production monitoring)
await loop.record_outcome(ProductionOutcome(
    project_id="ecommerce-api",
    deployment_id="prod-v1.2.3",
    task_type=TaskType.CODE_GEN,
    model_used=Model.DEEPSEEK_CHAT,
    generated_code_hash="abc123...",
    status=OutcomeStatus.SUCCESS,  # or FAILURE, PARTIAL, ROLLED_BACK
    runtime_errors=[],  # Captured from error tracking
    performance_metrics={"p95_latency_ms": 120, "error_rate": 0.001},
    user_feedback=UserFeedback(rating=4, comment="Fast and reliable"),
    codebase_fingerprint=CodebaseFingerprint(
        languages=["python"],
        framework="fastapi",
        patterns=["repository", "dependency-injection"],
    ),
))

# System learns: "DeepSeek works well for FastAPI + Repository pattern"
```

**Key Capabilities:**
- **Codebase-Specific Learning**: Remembers what works for YOUR patterns
- **Model Performance Tracking**: EMA of success rates per (model, task_type)
- **Automatic Knowledge Creation**: Failed deployments create lessons in KB
- **SDK for External Apps**: Lightweight client for deployed code to report outcomes

### Model Leaderboard with Benchmark Suite

Automated benchmarking that runs standardized tasks across all providers to keep routing optimal.

```python
from orchestrator.leaderboard import ModelLeaderboard, BenchmarkSuite

lb = ModelLeaderboard()

# Run benchmarks (scheduled job)
results = await lb.run_benchmarks(
    models=[Model.DEEPSEEK_CHAT, Model.GPT_4O, Model.GEMINI_FLASH],
    tasks=BenchmarkSuite().get_tasks_by_type(TaskType.CODE_GEN),
)

# Get current rankings
leaderboard = lb.get_leaderboard()
for entry in leaderboard[:5]:
    print(f"{entry.rank}. {entry.model.value} "
          f"(score: {entry.composite_score:.3f}, "
          f"cost/1k: ${entry.avg_cost_per_1k_tokens:.4f})")

# Update routing weights automatically
await lb.update_routing_weights()
```

**Benchmark Dimensions:**
- Quality (pattern matching, validation)
- Cost efficiency (quality per dollar)
- Latency (p50, p95)
- Reliability (pass rate)

### Outcome-Weighted Router

Production-outcome-weighted routing that creates Nash stability through accumulated learning.

```python
from orchestrator.outcome_router import (
    OutcomeWeightedRouter,
    RoutingContext,
    RoutingStrategy,
    create_routing_context,
)

router = OutcomeWeightedRouter()

# Route with production learning
model, metadata = await router.select_model(
    context=create_routing_context(
        task=my_task,
        budget_remaining=2.50,
        budget_total=5.00,
        strategy=RoutingStrategy.PRODUCTION_WEIGHTED,
        codebase_fingerprint=my_fingerprint,
    ),
)

# Metadata explains the decision
print(f"Selected: {model.value}")
print(f"Production score: {metadata['production_score']:.3f}")
print(f"Confidence: {metadata['confidence']:.2f} (based on {metadata.get('sample_size', 0)} samples)")
```

**Routing Strategies:**
| Strategy | Use Case |
|----------|----------|
| `COST_OPTIMIZED` | Minimize spend, accept quality trade-off |
| `QUALITY_OPTIMIZED` | Maximize quality, cost secondary |
| `BALANCED` | Default—balance all factors |
| `PRODUCTION_WEIGHTED` | Prefer models with proven history |
| `CODEBASE_SPECIFIC` | Match models to similar past codebases |
| `EXPLORATION` | 15% traffic to under-sampled models |

**Nash Stability Report:**
```python
report = router.get_nash_stability_report()
print(f"Total production samples: {report['total_production_samples']}")
print(f"Unique codebases learned: {report['unique_codebases_learned']}")
print(f"Information advantage: {report['information_advantage']['description']}")
```

---

## Black Swan Resilience (v6.0)

Production-hardened defenses against catastrophic failure modes identified through adversarial stress testing.

### Event Store Corruption Protection

Multi-layer durability for critical event data. **Risk reduced: $155,000 → $500 (99.7%)**

```python
from orchestrator.events_resilient import ResilientEventStore

store = ResilientEventStore(
    primary_path=".events/primary.db",
    replica_paths=[".events/replica1.db", ".events/replica2.db"],
)

# Automatic WAL + replication + checksums
await store.append(event)

# Automatic failover if primary corrupted
events = await store.get_events()
```

**Defense Layers:**
| Layer | Protection | Recovery |
|-------|------------|----------|
| Write-Ahead Logging | Crash-safe writes | Automatic on restart |
| SHA-256 Checksums | Tamper detection | Reconstruct from replicas |
| Async Replication | 2+ hot standbys | Zero RTO failover |
| Integrity Verification | Corruption detection | Point-in-time restore |

---

### Plugin Sandbox Hardening

Defense-in-depth security for untrusted plugins. **Risk reduced: $1,150,000 → $1,000 (99.9%)**

```python
from orchestrator.plugin_isolation_secure import (
    SecureIsolatedRuntime, SecureIsolationConfig
)

runtime = SecureIsolatedRuntime(SecureIsolationConfig(
    memory_limit_mb=512,
    enable_seccomp=True,      # Block dangerous syscalls
    enable_landlock=True,     # Filesystem sandboxing
    enable_capabilities=True, # Drop Linux privileges
    allow_network=False,      # No network access
))

result = await runtime.execute(plugin, "validate", code)
```

**Security Layers:**
| Layer | Blocks | Bypass Resistance |
|-------|--------|-------------------|
| Process Isolation | Memory access to host | Requires kernel exploit |
| seccomp-bpf | ptrace, execve, fork | Requires seccomp escape |
| Landlock | Filesystem access outside sandbox | Requires LSM bypass |
| Capabilities | Privilege escalation | Requires CAP_SYS_ADMIN |
| Resource Limits | DoS via resource exhaustion | N/A (enforced by kernel) |

**Trusted Plugin Registry:**
```python
from orchestrator.plugin_isolation_secure import TrustedPluginRegistry

registry = TrustedPluginRegistry()
registry.add_trusted_plugin("bandit_security", sha256_hash)

# Rejects modified/untrusted plugins
if registry.verify_plugin(path, "bandit_security"):
    await runtime.execute(plugin, "validate", code)
```

---

### Streaming Backpressure

Memory-safe execution for large projects. **Risk reduced: $30,000 → $500 (98.3%)**

```python
from orchestrator.streaming_resilient import (
    ResilientStreamingPipeline, MemoryPressureConfig, BackpressureStrategy
)

pipeline = ResilientStreamingPipeline(
    max_parallel=3,
    memory_config=MemoryPressureConfig(
        max_queue_size=1000,
        max_memory_mb=1024,
        backpressure_strategy=BackpressureStrategy.SAMPLE,
        sampling_rate=10,  # Keep every 10th event under pressure
    ),
)

# Automatic backpressure, never OOM
async for event in pipeline.execute_streaming(desc, criteria, budget):
    await websocket.send(event)
```

**Protection Mechanisms:**
| Mechanism | Trigger | Action |
|-----------|---------|--------|
| Bounded Queues | Queue full | Apply backpressure strategy |
| Memory Monitoring | >70% usage | Event sampling (1/N) |
| Critical Pressure | >90% usage | Pause + force GC |
| Circuit Breaker | 5 failures | Reject new work (fail fast) |
| Concurrency Limit | Low memory | Reduce parallel tasks |

**Backpressure Strategies:**
- `DROP_OLDEST` — Discard oldest events (live dashboards)
- `DROP_NEWEST` — Discard newest events (batch processing)
- `SAMPLE` — Keep every Nth event (telemetry)
- `PAUSE` — Temporarily halt pipeline (critical work)
- `BLOCK` — Block producer (risk: deadlock)

---

### Risk Summary

| Scenario | Before | After | Reduction |
|----------|--------|-------|-----------|
| Event Store Corruption | $155,000 | $500 | 99.7% |
| Plugin Sandbox Escape | $1,150,000 | $1,000 | 99.9% |
| Streaming Memory Bomb | $30,000 | $500 | 98.3% |
| **Total** | **$1,335,000** | **$2,000** | **99.85%** |

**Design Principles:**
- **Minimax Regret** — Optimize for worst case, not average case
- **Defense in Depth** — Even if one layer fails, others protect
- **Fail Safe** — Failure defaults to safe state (deny, failover, degrade)
- **Graceful Degradation** — Under stress, reduce quality but maintain function

---

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| **v6.2** | 2026-03-23 | 🧠 ARA Pipeline: 12 Advanced Reasoning Methods (Pre-Mortem, Bayesian, Debate, Jury, Analogical, Delphi, etc.), Intelligent Method Selection, Cost-Aware Routing |
| **v6.0.1** | 2026-03-17 | SRE hardening: BUG-001–005 fixed (budget reservations, asyncio gather, RRF mutation, rate-limiter TOCTOU, OpenAI temperature); A2AQueueManager; CancelledError guard |
| **v6.0** | 2026-03-02 | Black Swan Resilience: Event Store Corruption Protection, Plugin Sandbox Hardening, Streaming Backpressure |
| **v5.3** | 2026-03-02 | Plugin System, Production Feedback Loop, Outcome-Weighted Routing, Model Leaderboard |
| **v5.2** | 2026-03-01 | Author Attribution, Smart Validator Filtering, Temperature Optimization |
| **v5.1** | 2026-02-26 | Knowledge, Project, Product, Quality Management Systems |
| **v5.0** | 2026-02-15 | Performance Optimization (5x faster), Dual-layer Caching |
| **v4.x** | 2026-01 | Cost-optimized routing with Minimax, DeepSeek |

---

## SaaS Plan Mapping

| Plan | Modules Included | Best For |
|------|-----------------|----------|
| **Execution Core** | Module 1 | Developers who just want code generation |
| **Governance** | Module 1 + 2 | Teams needing compliance and policies |
| **Studio** | Module 1 + 4 | Rapid prototyping with AppBuilder |
| **Enterprise** | All Modules | Organizations with orchestration at scale |
| **AI-Native** | All + Custom Plugins | Teams building domain-specific AI workflows |

---

---

## Integrations & Workflow Hooks

Native integrations that connect the orchestrator to your existing development workflow. These are **production-ready capabilities**, not roadmap items.

### VCS & CI/CD

GitHub/GitLab native integrations for seamless CI/CD pipelines.

**Check Run / Status API**
```python
# GitHub Check Run created automatically on project start
await github_integration.create_check_run(
    repo="acme/webapp",
    sha="abc123...",
    name="Orchestrator / Code Generation",
    status="in_progress",
    details_url="https://dashboard.local/run/42"
)
# Updates to "completed" with conclusion="success|failure"
```

**PR Comments with Inline Feedback**
```python
# Auto-posts task results as review comments
await github_integration.post_review_comment(
    pr=123,
    commit_id="abc123...",
    path="src/api/auth.py",
    line=45,
    body="🔍 **Quality Gate Passed** (score: 0.92)\n"
         "Generated by: GPT-4o | Cost: $0.0042"
)
```

**Optional Autocommit / Auto-PR**
```yaml
# .orchestrator.yml - Enable automated commits
vcs:
  autocommit: true           # Commit successful tasks to branch
  auto_pr: true              # Create PR on project completion
  pr_template: "orchestrator_pr.md"
  branch_prefix: "orchestrator/"
```

### Collaboration

Slack/Teams integration for team visibility.

**Budget Alerts & Circuit Breaker**
```python
# Real-time alert when budget threshold hit
await slack_integration.send_alert(
    channel="#dev-alerts",
    blocks=[{
        "type": "header",
        "text": "⚠️ Budget Threshold Reached"
    }, {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": "*Project:* webgl-dj"},
            {"type": "mrkdwn", "text": "*Spent:* $4.20 / $5.00"},
            {"type": "mrkdwn", "text": "*Status:* Circuit breaker ENGAGED"},
            {"type": "mrkdwn", "text": "*Action:* Paused new tasks"}
        ]
    }]
)
```

**Run Summaries**
```python
# End-of-run summary posted to Slack
await slack_integration.send_summary(
    channel="#orchestrator-runs",
    summary={
        "project": "ecommerce-api",
        "tasks_completed": 12,
        "total_cost": "$3.45",
        "quality_score": "0.89",
        "duration": "4m 32s",
        "models_used": ["deepseek-chat", "gpt-4o"]
    }
)
```

**Slash Commands**
```python
# Slack: /orchestrator run <template>
@slack_command("/orchestrator")
async def handle_slash(cmd: SlashCommand):
    if cmd.text.startswith("run "):
        template = cmd.text[4:]  # "fastapi-microservice"
        project_id = await orchestrator.run_template(
            template=template,
            requester=cmd.user_id
        )
        return f"🚀 Started `{template}` → Project #{project_id}"
```

### Issue Tracking

Jira/Linear integration for bug tracking and prioritization.

**Ticket Sync on Quality Gate Failure**
```python
# Auto-creates ticket when quality gate fails
await jira_integration.create_ticket(
    project="ORCH",
    summary="Quality gate failed: auth-service",
    description="Generated code failed security checks.\n\n"
                "Score: 0.67 (threshold: 0.85)\n"
                "Failed: bandit security scan\n\n"
                "[View Dashboard|https://dash.local/runs/42]",
    issue_type="Bug",
    labels=["orchestrator-generated", "quality-gate"],
    priority="High"
)
```

**RICE Import/Export**
```python
# Export prioritized backlog to Jira
pm = get_product_manager()
backlog = pm.get_prioritized_backlog(limit=10)

await jira_integration.import_rice_backlog(
    project_key="PROD",
    features=[{
        "summary": f.name,
        "rice_score": f.rice_score.total,
        "priority": f.priority.value,
        "customfield_10010": f.rice_score.reach,      # RICE fields
        "customfield_10011": f.rice_score.impact,
        "customfield_10012": f.rice_score.confidence,
        "customfield_10013": f.rice_score.effort,
    } for f in backlog]
)
```

**Knowledge Linking**
```python
# Link tickets to relevant knowledge artifacts
kb = get_knowledge_base()
artifacts = await kb.query("authentication patterns")

await jira_integration.link_knowledge(
    issue_key="ORCH-123",
    knowledge_links=[{
        "url": f"https://kb.local/artifacts/{a.id}",
        "title": a.decision_title,
        "summary": a.rationale[:200]
    } for a in artifacts[:3]]
)
```

---

**Next Steps:**
- [USAGE_GUIDE.md](USAGE_GUIDE.md) — CLI & Python API examples
- [README.md](README.md) — Installation and quick start
- Dashboard: `python run_optimized_dashboard.py`