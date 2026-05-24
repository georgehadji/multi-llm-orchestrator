# Codebase-Aware AI Orchestrator — Architecture Research

> **Date:** 2026-05-23  
> **Purpose:** Research how to extend the AI Orchestrator from a spec-to-output generator into a codebase-aware analysis and modification engine.

---

## The Gap

| Current AI Orchestrator | Target State |
|------------------------|--------------|
| Takes a text specification | Takes a codebase path + objective |
| Generates new output files | Reads, analyzes, modifies existing code |
| Zero awareness of existing code | Full codebase context (AST graph, deps, patterns) |
| Cannot "finish" incomplete projects | Can identify gaps, implement missing pieces, fix bugs |
| Single "generate → critique → revise" path | Adaptive: read → analyze → plan → modify → verify |

---

## State of the Art (Industry Research)

### 1. Aider's Repo-Map (reference implementation)

**URL:** aider.chat/docs/repomap.html  
**Architecture:** Tree-sitter AST parser → symbol graph → PageRank ranking → token-budgeted context

```
Source files → Tree-sitter parse → Symbol definitions + references
                                         ↓
                                    Graph building (nodes = symbols, edges = references)
                                         ↓
                                    PageRank ranking by relevance
                                         ↓
                                    Token-budgeted context pruning
                                         ↓
                                    LLM prompt inclusion
```

**Key innovations:**
- **Graph-based ranking**: Uses PageRank to identify the most important symbols in the codebase for a given task
- **Token-budgeted context**: Never dumps entire files — selects only the most relevant portions within a configurable token limit (default 1024 tokens)
- **Incremental updates**: Only re-ranks files when relevant ones change

**Applicable patterns:** Symbol-level graph, PageRank relevance scoring, token-budgeted context inclusion

### 2. Qodo Open-Aware (deep code research agent)

**URL:** github.com/qodo-ai/open-aware  
**Architecture:** MCP server with 3 specialized agents

```
open-aware (MCP server)
  ├── get_context    → Semantic code retrieval (find relevant code for a query)
  ├── deep_research  → Architecture analysis, pattern detection, dependency tracing
  └── issues         → Breaking change detection, bug analysis, security audit
```

**Key innovations:**
- **Pre-indexed OSS**: Daily updated indexes of popular open-source libraries
- **Multi-repo analysis**: Cross-repository relationship analysis
- **Agent-based decomposition**: Separate agents for context retrieval vs deep analysis

**Applicable patterns:** Agent-based decomposition, pre-computed indexes for dependency analysis

### 3. Codebase Graph / CodePrism (graph-based analysis engine)

**URL:** rustic-ai.github.io/codeprism/  
**Architecture:** Universal AST → graph representation

```
Files → Universal AST parser → Graph database (Neo4j)
                                  ↓
                            Query layer (GQL)
                                  ↓
                            Analysis: blast radius, dead code, circular deps
```

**Key innovations:**
- **Graph database storage**: Persistent graph enables complex queries (GQL) across the entire codebase
- **Blast radius analysis**: Identify all code affected by a change
- **Cross-language**: Universal AST handles multiple languages

**Applicable patterns:** Graph database for persistent codebase understanding, blast radius queries

---

## Proposed Architecture: Codebase-Aware Orchestrator

```
                        ┌─────────────────────────┐
                        │     Orchestrator CLI     │
                        │  --repo ./my-project     │
                        │  --objective "Add auth"  │
                        └──────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  CodebaseReader  │  │   TaskPipeline   │  │  CodebaseWriter  │
    │  (READ phase)    │  │   (MODIFY phase) │  │  (WRITE phase)   │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  AST Index       │  │   ARA Reasoning  │  │  Diff Engine    │
    │  Dependency Graph│  │   Methods (20)   │  │  Git Integration│
    │  Symbol Cache    │  │   Plan + Execute │  │  Review Gate    │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Phase 1: CodebaseReader Module

**New file:** `orchestrator/codebase_reader.py`

### 1.1 File System Walker

Walks a directory tree, respects `.gitignore`, collects relevant source files.

```python
class FileWalker:
    def walk(self, root: Path, include: list[str], exclude: list[str]) -> list[Path]:
        # Uses pathspec (gitignore patterns)
        # Returns all source files
```

### 1.2 AST Indexer

Uses `ast` (Python stdlib) or `tree-sitter` (multi-language) to parse each file and extract:

```python
@dataclass
class Symbol:
    name: str
    type: SymbolType  # FUNCTION, CLASS, VARIABLE, MODULE
    file: Path
    line: int
    docstring: str
    dependencies: list[str]  # imports, function calls

class ASTIndexer:
    def index_file(self, path: Path) -> list[Symbol]:
        # Parse AST, extract symbols and their references
```

### 1.3 Dependency Graph Builder

Builds a directed graph of which modules/symbols depend on which:

```python
class DependencyGraph:
    nodes: dict[str, SymbolNode]   # module → node
    edges: dict[str, list[str]]    # dependent → dependencies
    
    def find_unused_code(self) -> list[str]:
        # Nodes with zero dependents
    
    def find_circular_deps(self) -> list[list[str]]:
        # Cycles in the graph
    
    def find_blast_radius(self, symbol: str) -> set[str]:
        # All transitive dependents
```

### 1.4 Project Profile

High-level project understanding:

```python
@dataclass
class ProjectProfile:
    language: str
    framework: str
    package_manager: str
    entry_points: list[str]
    test_framework: str
    code_metrics: CodeMetrics  # LOC, complexity, coverage gaps

class ProjectProfiler:
    def profile(self, root: Path) -> ProjectProfile:
        # Detect package.json, pyproject.toml, Cargo.toml
        # Identify framework (Django, FastAPI, React, etc.)
        # Find test files and coverage gaps
```

---

## Phase 2: CodebaseContext Module

**New file:** `orchestrator/codebase_context.py`

Combines the reader output into a structured context that feeds into the existing `ProjectContext` and task prompts.

### 2.1 Context Builder

```python
class CodebaseContext:
    def __init__(self, root: Path):
        self.profile = ProjectProfiler().profile(root)
        self.graph = DependencyGraph()
        self.index = ASTIndexer()
    
    def to_task_context(self, objective: str) -> str:
        """Build a structured context prompt for the LLM."""
        # 1. Project overview (language, framework, entry points)
        # 2. Most relevant files (via PageRank or keyword match)
        # 3. Key symbols (classes, functions) near the objective
        # 4. Test coverage gaps
```

### 2.2 Objective Parser

```python
class ObjectiveParser:
    """Parse user objective into structured tasks."""
    def parse(self, objective: str, context: CodebaseContext) -> list[Task]:
        # "Add authentication" → 
        #   1. Install auth library → 2. Create auth middleware →
        #   3. Add login route → 4. Add tests
```

---

## Phase 3: Codebase Integration with Existing Pipeline

### 3.1 New Entry Points

```python
# orchestrator/cli.py — new subcommand
parser.add_argument("--repo", type=str, help="Path to existing codebase")
parser.add_argument("--objective", type=str, help="What to do")

# orchestrator/engine.py — new method
async def modify_project(self, repo_path: Path, objective: str) -> ProjectState:
    context = CodebaseContext(repo_path)
    tasks = self._plan_modifications(objective, context)
    state = await self._execute_all(tasks, context)
    return state
```

### 3.2 Modified Decomposition

Instead of decomposing a specification into tasks from scratch, the decomposer analyzes the existing codebase and plans modifications:

```python
class CodebaseDecomposer:
    async def decompose(self, objective: str, context: CodebaseContext) -> dict[str, Task]:
        # 1. Load context into prompt
        # 2. Ask LLM: "Given this objective and codebase, list the changes needed"
        # 3. Parse JSON response into tasks with modify/delete semantics
```

### 3.3 New Task Types

```python
class TaskType(str, Enum):
    CODE_GEN = "code_generation"      # existing — create new file
    CODE_REVIEW = "code_review"       # existing — review output
    MODIFY_FILE = "modify_file"       # new — modify existing file
    DELETE_FILE = "delete_file"       # new — remove file
    INSTALL_DEP = "install_dependency" # new — add package
```

### 3.4 ARA Methods for Codebase Analysis

The existing ARA methods can be repurposed for codebase analysis:

| ARA Method | Codebase Application |
|-----------|---------------------|
| **Multi-Perspective** | Analyze code from 4 angles: security, performance, maintainability, correctness |
| **PersuasionDefense** | Verify code changes don't introduce hallucinated API calls |
| **Pre-Mortem** | Anticipate failure modes of proposed changes before writing code |
| **Research** | Search for best practices / library alternatives for identified issues |
| **Debate** | Compare two refactoring approaches |
| **Delphi** | Multi-model consensus on architectural decisions |

---

## Phase 4: CodebaseWriter Module

**New file:** `orchestrator/codebase_writer.py`

Handles the write-back phase — applies generated code to the existing codebase.

### 4.1 File Operations

```python
class CodebaseWriter:
    def modify_file(self, path: Path, new_content: str, strategy: str):
        """Apply changes to existing file."""
        # strategy = "replace" | "insert_at_line" | "append" | "patch_function"
    
    def create_file(self, path: Path, content: str):
        """Create new file, creating parent directories."""
    
    def delete_file(self, path: Path):
        """Delete file with safety checks."""
```

### 4.2 Diff Generation

```python
class DiffEngine:
    def generate_diff(self, original: str, modified: str) -> str:
        """Generate unified diff for review."""
    
    def apply_patch(self, diff: str) -> bool:
        """Apply previously reviewed patch."""
```

### 4.3 Safety Gates

```python
class ModificationGate:
    async def verify_modification(self, task: Task, result: TaskResult) -> bool:
        # Before writing: 
        # 1. Validate syntax of modified file
        # 2. Check imports resolve correctly
        # 3. Run existing tests that cover modified area
        # 4. Verify no security issues introduced
```

---

## Phase 5: Git Integration

**New file:** `orchestrator/git_integration.py`

### 5.1 Branch Management

```python
class GitIntegration:
    async def create_branch(self, name: str) -> bool:
        """Create feature branch for changes."""
    
    async def commit_changes(self, message: str) -> str:
        """Stage and commit all changes."""
    
    async def create_pr(self, title: str, body: str) -> str:
        """Create GitHub/GitLab PR."""
```

---

## Implementation Plan Summary

| Phase | Module | Effort | Key Dependencies |
|-------|--------|--------|-----------------|
| 1 | `CodebaseReader` (walker + AST indexer + dep graph) | 5 days | `ast` / `tree-sitter`, `pathspec` |
| 2 | `CodebaseContext` (context builder + objective parser) | 3 days | Phase 1 |
| 3 | Pipeline integration (modified `Decomposer`, new TaskTypes) | 4 days | Phase 2 |
| 4 | `CodebaseWriter` (file operations + diff + safety gates) | 3 days | Phase 3 |
| 5 | `GitIntegration` (branch, commit, PR) | 2 days | Phase 4 |
| **Total** | | **17 days** | |

---

## Technology Choices

| Component | Option A (Stdlib, simple) | Option B (External, powerful) | Recommended |
|-----------|--------------------------|-------------------------------|-------------|
| AST parsing | Python `ast` module | `tree-sitter` (multi-language) | **Both** — stdlib for Python, tree-sitter for JS/TS/Go |
| Git ignore | Manual `.gitignore` parser | `pathspec` library | `pathspec` |
| Graph storage | In-memory `networkx` | Neo4j graph database | **networkx** (no infra needed) |
| Diff generation | Python `difflib` | `unidiff` library | **difflib** (stdlib, sufficient) |
| Code formatting | None | `black`, `prettier` | **subprocess** call to formatter |

---

## Key Design Principles

1. **Fail-open:** If codebase analysis fails, fall back to specification-only generation
2. **Read-only analysis:** Phase 1 and 2 never modify the codebase — only Phase 4 and 5 do
3. **Safety gate before write:** Every modification must pass syntax + test + security validation
4. **Git as undo:** All modifications go through git branches — rollback is `git checkout .`
5. **ARA reuse:** All 20 ARA methods remain unchanged — only their invocation context changes (codebase analysis instead of spec generation)

---

**Last updated:** 2026-05-23
