# Four Pillars Enhancement Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-24  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The AI Orchestrator has 7 agents, 20 ARA methods, a blackboard workspace, and 59 models. It knows how to build software but doesn't know how to **manage** the process. This plan adds four pillars:

| Pillar | Gap | Target |
|--------|-----|--------|
| **Project Management** | No milestones, no deadlines, no sprint tracking | Sprint planning, task prioritization, progress tracking |
| **Product Management** | No requirements, no user stories, no priorities | Requirements mapping, user story generation, priority matrix |
| **Knowledge Management** | Decisions logged but not searchable | Searchable knowledge base, cross-project learning, docs generation |
| **Quality Control** | Syntax + hallucination only | Full CI, test coverage, regression detection, quality dashboard |

**Total effort:** 12 days | **8 new agents/roles** | **5 new modules**

---

## Pillar 1: Project Management (3 days)

### Current State
- `AgentOrchestrator` dispatches tasks but has no concept of time, milestones, or progress
- No way to track "we're 40% done" or "3 of 5 milestones hit"
- `GoalDecomposer` creates task DAGs but doesn't assign estimates or priorities

### Target State

```
Sprint Planner → Milestone Tracker → Progress Reporter
      │                 │                  │
      ▼                 ▼                  ▼
Task estimates +    Milestone state    Completion %
priority matrix     (pending/active/done)    per milestone
```

### What to Build

#### 1.1 Sprint Planner (`orchestrator/project/sprint_planner.py`)

```python
@dataclass
class Sprint:
    id: str
    goal: str
    tasks: list[str]  # task IDs
    start_date: str
    deadline: str
    status: str = "pending"  # pending/active/completed

class SprintPlanner:
    def create_sprint(self, goal: str, tasks: list[str]) -> Sprint:
        # Wraps a group of tasks into a sprint with a deadline
    
    def get_progress(self, sprint_id: str) -> float:
        # Returns 0.0-1.0 based on completed tasks / total tasks
```

#### 1.2 Progress Reporter (`orchestrator/project/progress_reporter.py`)

```python
class ProgressReporter:
    def report(self, project) -> str:
        # Returns a formatted progress table:
        # Milestone 1: 3/5 tasks done [===================>          ] 60%
        # Milestone 2: 0/3 tasks done [                              ] 0%
    
    def estimate_completion(self, project) -> str:
        # Based on historical velocity, when will this finish?
```

#### 1.3 Integration

- `AgentOrchestrator.execute_goal()` creates a `Sprint` automatically
- `ProgressReporter.report()` is called by `UserAgent` when user asks "how are we doing?"
- Sprint state is persisted in `ProjectWorkspace`

### Models & Costs
- Sprint planning: `OWL_ALPHA` (free, 1M ctx) or `CLAUDE_SONNET_4_6` ($3/M)
- All task tracking: deterministic — no LLM cost

---

## Pillar 2: Product Management (3 days)

### Current State
- `Decomposer.decompose()` takes a spec string, returns tasks
- No concept of requirements, user stories, acceptance criteria, or priority

### Target State

```
User wants "a todo app"
        │
        ▼
ProductManagerAgent
  ├─ Generate user stories ("As a user, I can add tasks...")
  ├─ Map stories to modules (auth → auth.py, ui → template.html)
  ├─ Prioritize (P0 = account creation, P3 = dark mode)
  └─ Store in requirements backlog
        │
        ▼
ArchitectAgent (reads backlog, designs architecture)
        │
        ▼
DeveloperAgent (implements stories in priority order)
```

### What to Build

#### 2.1 ProductManagerAgent (`orchestrator/agents/product_manager.py`)

```python
class ProductManagerAgent(AgentBase):
    """Generates user stories, maps to modules, prioritizes."""
    
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.PRODUCT_MANAGER)
    
    @property
    def system_prompt(self) -> str:
        return "You are a product manager. Convert high-level goals into user stories. Prioritize ruthlessly."
    
    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        # 1. Generate user stories from the goal
        # 2. Map stories to existing codebase modules
        # 3. Assign priorities (P0-P3)
        # 4. Store in requirements backlog
```

#### 2.2 Requirements Backlog (`orchestrator/product/backlog.py`)

```python
@dataclass
class UserStory:
    id: str
    title: str
    description: str  # "As a <role>, I want <feature> so that <value>"
    priority: str     # P0 (critical), P1 (high), P2 (medium), P3 (low)
    module: str       # Which module implements it
    acceptance_criteria: list[str]
    status: str = "backlog"  # backlog/in_progress/done

class ProductBacklog:
    def add_story(self, story: UserStory) -> None: ...
    
    def stories_by_priority(self) -> list[UserStory]:
        # Returns P0 first, then P1, etc.
    
    def completion_status(self) -> dict:
        # Returns {P0: 3/5 done, P1: 1/3 done, ...}
```

#### 2.3 Integration

- User says "Build a todo app" → ProductManager generates user stories → back to UserAgent asking "Here are the stories, is this right?"
- ArchitectAgent reads the backlog to design around P0/P1 items first
- DeveloperAgent implements stories from the backlog in priority order

### Models & Costs
- ProductManager: `DEEPSEEK_REASONER` ($0.70/M budget) or `GPT_5` ($1.25/M premium)

---

## Pillar 3: Knowledge Management (3 days)

### Current State
- `ArchitectureDecision` logs exist in the workspace but aren't searchable
- No auto-generated documentation
- Cross-project knowledge doesn't transfer between orchestrator runs

### Target State

```
ArchitectAgent makes a decision ("Use FastAPI over Django")
        │
        ▼
DecisionLogger records: {what, why, alternatives, tradeoffs}
        │
        ▼
KnowledgeBase stores: tagged, searchable, cross-project
        │
        ▼
DocsGenerator outputs: ARCHITECTURE.md, DECISIONS.md, CONTRIBUTING.md
```

### What to Build

#### 3.1 KnowledgeBase (`orchestrator/knowledge/knowledge_base.py`)

```python
@dataclass
class KnowledgeEntry:
    id: str
    category: str  # "architecture", "pattern", "lesson_learned", "best_practice"
    content: str
    tags: list[str]
    date: str
    project: str
    source_agent: str

class KnowledgeBase:
    def add(self, entry: KnowledgeEntry) -> None:
        # Store in a local SQLite DB or JSON file
    
    def search(self, query: str, tags: list[str] = None) -> list[KnowledgeEntry]:
        # Simple keyword + tag search across all entries
    
    def find_similar(self, entry: KnowledgeEntry) -> list[KnowledgeEntry]:
        # Find similar entries across projects for transfer learning
    
    def export_docs(self, output_dir: Path) -> None:
        # Generate ARCHITECTURE.md, DECISIONS.md, PATTERNS.md
```

#### 3.2 Auto-Documentation Generator (`orchestrator/knowledge/docs_generator.py`)

```python
class DocsGenerator:
    def generate_architecture_doc(self, kb: KnowledgeBase) -> str:
        # Collects all architecture decisions, generates a markdown doc
    
    def generate_decisions_log(self, kb: KnowledgeBase) -> str:
        # Chronological log of every architecture decision with rationale
    
    def generate_contributing_guide(self, kb: KnowledgeBase) -> str:
        # Based on the codebase patterns, generates a CONTRIBUTING.md
```

#### 3.3 Integration

- Every `ArchitectAgent` decision is logged as a `KnowledgeEntry`
- On project completion, `DocsGenerator.export_docs()` writes to `<project>/docs/`
- `UserAgent` can query the knowledge base: "Why did we choose Postgres?"
- Cross-project: new projects query old knowledge bases for similar patterns

### Models & Costs
- Knowledge querying: `LING_2_6_FLASH` ($0.01/M, cheapest search)
- Documentation generation: `CODESTRAL_2508` ($0.30/M) or `GPT_5_3_CODEX` ($1.75/M)

---

## Pillar 4: Quality Control (3 days)

### Current State
- Syntax validation (Python `ast.parse`)
- PersuasionDefense (hallucination detection)
- Secret scanning (regex)
- No test coverage tracking, no regression detection

### Target State

```
DeveloperAgent generates code
        │
        ▼
QCAgent runs full CI pipeline:
  ├─ Lint (ruff, eslint, go vet)
  ├─ Type check (mypy, tsc)
  ├─ Test (generated tests)
  ├─ Coverage check (must be > 80%)
  ├─ Build (compile, bundle)
  └─ Quality Report (score 0-10)
        │
        ▼
If score < 7 → feed failures back to DeveloperAgent → retry
```

### What to Build

#### 4.1 QCAgent (`orchestrator/agents/qc.py`)

```python
class QCAgent(AgentBase):
    """Runs all quality checks and generates a report."""
    
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.QA)
    
    @property
    def system_prompt(self) -> str:
        return "You are a QA engineer. Run quality gates and report findings concisely."
    
    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        # 1. Run lint
        # 2. Run type check
        # 3. Run tests
        # 4. Check coverage
        # 5. Generate quality report with score
```

#### 4.2 Quality Report (`orchestrator/quality/report.py`)

```python
@dataclass
class QualityReport:
    overall_score: float  # 0.0-10.0
    lint: int  # errors found
    type_check: int  # errors found
    tests_passed: int
    tests_failed: int
    coverage_pct: float
    security_issues: int
    recommendation: str  # "PASS", "REVISE", "BLOCK"
```

#### 4.3 Regression Detection (`orchestrator/quality/regression.py`)

```python
class RegressionDetector:
    def compare(self, before: QualityReport, after: QualityReport) -> bool:
        # Returns True if quality regressed (more errors, fewer tests passed, etc.)
        # Prevents merging code that makes things worse
```

#### 4.4 Integration

- DeveloperAgent finishes → QCAgent runs CI → QualityReport generated
- If score < 7: feedback goes back to DeveloperAgent for revision
- If score >= 7: code passes, ReviewerAgent does the final review
- Regression detection prevents backsliding

### Models & Costs
- QC analysis: `CODESTRAL_2508` ($0.30/M budget) or `CLAUDE_SONNET_4_6` ($3.00/M premium)

---

## Implementation Order & Effort Summary

| # | Module | Pillar | Days | Files |
|---|--------|--------|------|-------|
| 1 | Sprint Planner + Progress Reporter | Project Mgmt | 1.5 | `project/sprint_planner.py`, `project/progress_reporter.py` |
| 2 | ProductManagerAgent + Backlog | Product Mgmt | 2 | `agents/product_manager.py`, `product/backlog.py` |
| 3 | KnowledgeBase + DocsGenerator | Knowledge Mgmt | 2 | `knowledge/knowledge_base.py`, `knowledge/docs_generator.py` |
| 4 | QCAgent + QualityReport + Regression | Quality Control | 2.5 | `agents/qc.py`, `quality/report.py`, `quality/regression.py` |
| 5 | Wire into AgentOrchestrator (all pillars) | Integration | 1.5 | `agents/coordinator.py` |
| 6 | AgentRole updates + docs | Plumbing | 1.5 | `agents/base.py`, test files |
| 7 | Tests (32 new tests) | Quality | 1 | 4 test files |
| **Total** | | | **12 days** | |

---

## New Agent Roles Added

| Role | Agent Class | Best Budget Model | Best Premium Model |
|------|------------|-------------------|-------------------|
| `PRODUCT_MANAGER` | `ProductManagerAgent` | `DEEPSEEK_REASONER` ($0.70/M) | `GPT_5` ($1.25/M) |
| `QA` | `QCAgent` | `CODESTRAL_2508` ($0.30/M) | `CLAUDE_SONNET_4_6` ($3.00/M) |

**Total agent roles: 7 → 9**

## Files to Create

| File | Purpose | Pillar |
|------|---------|--------|
| `orchestrator/project/sprint_planner.py` | Sprint creation + progress tracking | PM |
| `orchestrator/project/progress_reporter.py` | Completion % + estimates | PM |
| `orchestrator/agents/product_manager.py` | User story generation + prioritization | Prod |
| `orchestrator/product/backlog.py` | Requirements backlog with priorities | Prod |
| `orchestrator/knowledge/knowledge_base.py` | Searchable cross-project knowledge | KM |
| `orchestrator/knowledge/docs_generator.py` | Auto-generated documentation | KM |
| `orchestrator/agents/qc.py` | Full CI pipeline agent | QC |
| `orchestrator/quality/report.py` | Quality report with score | QC |
| `orchestrator/quality/regression.py` | Before/after quality comparison | QC |
| `orchestrator/agents/base.py` | Add 2 new AgentRole values | All |
| `orchestrator/agents/coordinator.py` | Wire new agents into dispatch | All |

**Total: 11 files (10 new, 1 modified)**

---

## Verification Gates

- [ ] `python -m orchestrator agentic --goal "Build a todo app"` — ProductManager generates stories, UserAgent presents them
- [ ] Sprint progress: UserAgent reports "Milestone 1: 60% complete, 2 tasks remaining"
- [ ] Knowledge base: "Why did we choose Postgres?" returns the decision with rationale
- [ ] Quality gate: generated code with syntax error → QCAgent returns score < 7 → DeveloperAgent retries
- [ ] All existing 176+ tests pass
- [ ] 32 new tests pass

---

## Appendix: Target Cost per Pillar

| Pillar | Budget Cost (per sprint) | Premium Cost (per sprint) |
|--------|-------------------------|--------------------------|
| Project Management | $0.00 (deterministic) | $0.00 |
| Product Management | $0.70 (DEEPSEEK_REASONER) | $1.25 (GPT_5) |
| Knowledge Management | $0.01 (LING_2_6_FLASH) | $1.75 (GPT_5_3_CODEX) |
| Quality Control | $0.30 (CODESTRAL_2508) | $3.00 (CLAUDE_SONNET_4_6) |
| **Total per sprint** | **~$1.01** | **~$6.00** |

---

**Last updated:** 2026-05-24
