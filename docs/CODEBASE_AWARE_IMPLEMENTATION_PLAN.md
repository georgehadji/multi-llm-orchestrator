# Codebase-Aware AI Orchestrator — Implementation Plan

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The AI Orchestrator currently generates projects from text specifications. This plan extends it to **read, analyze, and modify existing codebases** — turning it from a spec-to-output generator into a codebase-aware development tool.

### Capability Delta

| Before | After |
|--------|-------|
| `python -m orchestrator --project "Build API"` | `python -m orchestrator --repo ./my-app --objective "Add auth"` |
| Generates files to `outputs/` | Modifies existing files in-place |
| Zero awareness of existing code | Full AST index + dependency graph |
| Cannot analyze code quality | Static analysis runner (radon, bandit, mypy) |
| Single generate path | Read → Analyze → Plan → Modify → Verify |

---

## Architecture Overview

### Target Flow

```
User: --repo ./my-app --objective "Add JWT authentication"
         │
         ▼
┌─────────────────────────┐
│   1. CodebaseReader     │  Walk files, parse ASTs, build symbol graph
│   (READ phase)          │  Detect framework, entry points, test gaps
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   2. CodebaseContext     │  Build compressed LLM context from graph
│   (UNDERSTAND phase)    │  Rank symbols by relevance to objective
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   3. ObjectiveParser    │  "Add auth" → [install lib, create middleware,
│   (PLAN phase)          │   add routes, write tests, update config]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   4. TaskPipeline       │  Existing pipeline, new task types:
│   (MODIFY phase)        │  MODIFY_FILE, INSTALL_DEP, DELETE_FILE
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│   5. CodebaseWriter     │  Apply changes, generate diffs, run safety gates
│   (WRITE phase)         │  Git branch, commit, PR
└───────────┬─────────────┘
            ▼
       Modified codebase
```

---

## Phased Implementation Plan

---

### Phase 1: CodebaseReader Module

**New file:** `orchestrator/codebase_reader.py`  
**Effort:** 5 days  
**Risk:** Low (pure computation, no external effects)

#### 1.1 FileSystemWalker

Walks a directory tree, respects `.gitignore`, collects source files by extension.

```python
@dataclass
class FileNode:
    path: Path
    language: str  # "python", "javascript", "typescript"
    size: int
    lines: int

class FileSystemWalker:
    """
    Walks a directory tree collecting source files.
    Respects .gitignore patterns using the 'pathspec' library.
    """
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._ignored = self._load_gitignore()
    
    def walk(self, extensions: set[str] | None = None) -> list[FileNode]:
        """Walk directory, return source files matching extensions."""
        ...
    
    def _load_gitignore(self) -> pathspec.PathSpec:
        """Load and parse .gitignore from root."""
        ...
```

**Dependencies added:** `pathspec`  
**Verification:** Walk a known repo, verify .gitignore patterns are respected, verify file count matches `find . -name "*.py" | wc -l`.

#### 1.2 ASTIndexer

Parses each source file, extracts symbol definitions (classes, functions, methods) and their dependencies.

```python
@dataclass  
class Symbol:
    name: str
    type: str  # "class", "function", "method", "variable"
    file: Path
    line_start: int
    line_end: int
    docstring: str
    dependencies: list[str]  # imported symbols, called functions

class ASTIndexer:
    """
    Extracts symbols and dependencies from source files.
    Uses Python's 'ast' module for Python files.
    Uses 'tree-sitter' for JS/TS/Go files.
    """
    def index_file(self, path: Path) -> list[Symbol]:
        """Parse a single file and return its symbols."""
        ...
    
    def index_directory(self, files: list[FileNode]) -> dict[Path, list[Symbol]]:
        """Index multiple files."""
        ...

    def _index_python(self, path: Path) -> list[Symbol]:
        """Parse Python AST — extract classes, functions, imports."""
        ...
```

**Verification:** Index a Python file, verify it extracts all classes, functions, imports correctly.

#### 1.3 DependencyGraph

Builds a directed graph of module-level dependencies using `networkx`.

```python
import networkx as nx

class DependencyGraph:
    """
    Directed graph of module dependencies.
    Nodes = file paths, Edges = imports.
    """
    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
    
    def add_module(self, path: Path, dependencies: list[str]) -> None:
        """Add a module node and its dependency edges."""
        ...
    
    def find_unused_modules(self) -> list[Path]:
        """Modules with zero dependents (excluding entry points)."""
        ...
    
    def find_circular_dependencies(self) -> list[list[Path]]:
        """Detect circular import chains."""
        ...
    
    def find_blast_radius(self, module: Path) -> set[Path]:
        """All transitive dependents of a module."""
        ...
    
    def rank_by_centrality(self) -> list[tuple[Path, float]]:
        """PageRank on the dependency graph. Higher = more central."""
        ...
```

**Dependencies added:** `networkx`  
**Verification:** Build graph for the orchestrator itself, verify it detects the engine.py circular import.

#### 1.4 ProjectProfiler

Detects project type, framework, package manager, test framework, entry points.

```python
@dataclass
class ProjectProfile:
    languages: list[str]
    framework: str | None  # "fastapi", "django", "react", "nextjs"
    package_manager: str | None  # "pip", "npm", "cargo"
    test_framework: str | None  # "pytest", "jest", "vitest"
    entry_points: list[Path]
    has_docker: bool
    has_ci: bool
    loc: int  # total lines of code
    file_count: int
    test_count: int  # number of test files

class ProjectProfiler:
    """
    Detect project type and structure from config files.
    """
    def profile(self, root: Path) -> ProjectProfile:
        # Check for: pyproject.toml, package.json, Cargo.toml
        # Check for: Dockerfile, .github/workflows
        # Count test files (*_test.py, test_*.py, *.spec.ts)
        ...
```

**Verification:** Run on the orchestrator's own repo — verify it detects Python, pytest, 350+ Python files.

#### 1.5 Module Assembly

```python
class CodebaseReader:
    """
    Top-level reader that orchestrates the walker, indexer, graph builder,
    and profiler into a single pass.
    """
    def __init__(self, root: Path):
        self.root = root
        self.files: list[FileNode] = []
        self.symbols: dict[Path, list[Symbol]] = {}
        self.graph: DependencyGraph = DependencyGraph()
        self.profile: ProjectProfile | None = None
    
    async def read(self) -> "CodebaseReader":
        """Walk, index, graph, profile — all in one call."""
        ...
```

---

### Phase 2: CodebaseContext Module

**New file:** `orchestrator/codebase_context.py`  
**Effort:** 3 days  
**Dependency:** Phase 1

#### 2.1 RelevanceRanker

Given a reader output and an objective, selects the most relevant files/symbols.

```python
class RelevanceRanker:
    """
    Ranks files and symbols by relevance to a given objective.
    Uses combination of:
    - Keyword matching (objective keywords vs symbol names)
    - Graph centrality (PageRank from DependencyGraph)
    - File proximity (files that changed together in git history)
    """
    def rank_files(self, objective: str, graph: DependencyGraph, 
                   symbols: dict[Path, list[Symbol]]) -> list[tuple[Path, float]]:
        """Return ranked list of (file, relevance_score)."""
        ...
    
    def rank_symbols(self, objective: str, symbols: dict[Path, list[Symbol]],
                     top_n: int = 50) -> list[tuple[Symbol, float]]:
        """Return ranked list of top symbols within token budget."""
        ...
```

#### 2.2 CodebaseContext

Builds the final LLM context string from the reader + ranker output.

```python
class CodebaseContext:
    """
    Structured context about the codebase, token-budgeted for LLM inclusion.
    """
    def __init__(self, reader: CodebaseReader):
        self.reader = reader
        self.ranker = RelevanceRanker()
    
    def to_llm_prompt(self, objective: str, max_tokens: int = 4096) -> str:
        """
        Build a structured context string for the LLM.
        
        Contents (in order, within token budget):
        1. Project overview (language, framework, test gaps)
        2. Entry points and key files (ranked by relevance)
        3. Key symbols near the objective (ranked)
        4. Relevant test files
        5. Dependency graph structure (compressed)
        """
        ...
```

#### 2.3 QualityAnalyzer

Runs existing static analysis tools on the codebase and reports findings.

```python
@dataclass
class AnalysisFinding:
    severity: str  # "error", "warning", "info"
    category: str  # "bug", "security", "performance", "style"
    file: Path
    line: int
    message: str
    suggestion: str | None

class QualityAnalyzer:
    """
    Run static analysis tools and collect findings.
    """
    async def analyze(self, root: Path) -> list[AnalysisFinding]:
        """
        Run:
        - bandit (security)
        - radon (complexity)
        - mypy (type errors, if pyproject.toml present)
        - ruff (lint, if config present)
        """
        ...
    
    async def find_coverage_gaps(self, root: Path) -> list[str]:
        """Find Python modules without corresponding test files."""
        ...
```

---

### Phase 3: Pipeline Integration

**Files modified:** `orchestrator/engine.py`, `orchestrator/models.py`, `orchestrator/cli.py`, `orchestrator/codebase_decomposer.py`  
**Effort:** 4 days  
**Dependency:** Phase 1, 2

#### 3.1 New Task Types

```python
# orchestrator/models.py
class TaskType(str, Enum):
    CODE_GEN = "code_generation"      # existing
    CODE_REVIEW = "code_review"       # existing
    REASONING = "complex_reasoning"   # existing
    WRITING = "creative_writing"      # existing
    DATA_EXTRACT = "data_extraction"  # existing
    SUMMARIZE = "summarization"       # existing
    EVALUATE = "evaluation"           # existing
    MODIFY_FILE = "modify_file"       # NEW — modify existing file
    DELETE_FILE = "delete_file"       # NEW — remove file
    INSTALL_DEP = "install_dependency"  # NEW — add/update package
```

#### 3.2 New Task Fields

```python
# orchestrator/models.py — Task dataclass additions
@dataclass
class Task:
    # ... existing fields ...
    target_file: Path | None = None       # For MODIFY_FILE / DELETE_FILE
    modification_strategy: str = "replace"  # "replace" | "insert" | "patch"
    dependencies_to_install: list[str] = field(default_factory=list)  # For INSTALL_DEP
```

#### 3.3 CodebaseDecomposer

New decomposer variant that plans modifications to an existing codebase.

```python
# orchestrator/codebase_decomposer.py
class CodebaseDecomposer:
    """
    Given an objective and codebase context, produces a task plan.
    
    Example:
        Objective: "Add JWT authentication"
        → Task 1: INSTALL_DEP "python-jose[cryptography]"
        → Task 2: MODIFY_FILE src/auth.py (insert auth middleware)
        → Task 3: MODIFY_FILE src/routes/users.py (add login endpoint)
        → Task 4: CODE_GEN tests/test_auth.py (new test file)
    """
    def __init__(self, client: UnifiedClient, selector: ModelSelector):
        ...
    
    async def decompose(self, objective: str, context: CodebaseContext) -> dict[str, Task]:
        """
        1. Call LLM with codebase context + objective
        2. LLM returns JSON list of modification tasks
        3. Each task has: id, type, target_file, prompt, dependencies
        """
        ...
```

#### 3.4 New CLI Entry Points

```python
# orchestrator/cli.py — new subcommand
def _codebase_subparsers(subparsers):
    p = subparsers.add_parser("modify", help="Modify an existing codebase")
    p.add_argument("--repo", required=True, help="Path to codebase")
    p.add_argument("--objective", required=True, help="What to do")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--output-dir", help="Where to write results")
    p.add_argument("--dry-run", action="store_true", help="Plan only, no changes")
    p.add_argument("--create-pr", action="store_true", help="Create PR with changes")
```

#### 3.5 Modified Engine

```python
# orchestrator/engine.py — new method
async def modify_codebase(
    self,
    repo_path: Path,
    objective: str,
    create_pr: bool = False,
    dry_run: bool = False,
) -> ModificationResult:
    """
    Read → Analyze → Plan → Modify → Verify
    """
    # Phase 1-2: Read + understand
    reader = CodebaseReader(repo_path)
    await reader.read()
    context = CodebaseContext(reader)
    
    # Quality analysis
    analyzer = QualityAnalyzer()
    findings = await analyzer.analyze(repo_path)
    coverage_gaps = await analyzer.find_coverage_gaps(repo_path)
    
    # Phase 3: Plan modifications
    tasks = await self._codebase_decomposer.decompose(objective, context)
    
    # Phase 4: Execute modifications
    writer = CodebaseWriter(repo_path)
    for task in tasks:
        result = await self._execute_task(task, context=context.to_llm_prompt(objective))
        if result.status == TaskStatus.COMPLETED:
            await writer.apply(task, result)
    
    # Phase 5: Git
    if create_pr:
        git = GitIntegration(repo_path)
        await git.commit_changes(f"feat: {objective}")
        pr = await git.create_pr(
            title=objective,
            body=self._build_pr_body(tasks, context.profile),
        )
    
    return ModificationResult(tasks=tasks, pr=pr, findings=findings)
```

#### 3.6 ARA Methods for Codebase Analysis

The existing 20 ARA methods are repurposed for codebase-level analysis.

| ARA Method | Codebase Usage | Integration Point |
|-----------|----------------|-------------------|
| **QualityAnalyzer** | Multi-perspective code review from 4 angles | Analyze phase |
| **PersuasionDefense** | Verify proposed changes against hallucinated API calls | Pre-apply gate |
| **Pre-Mortem** | "Assume this change broke the build — why?" | Planning phase |
| **Research** | Search for best practices / alternative libraries | Planning phase |
| **Debate** | Compare two implementation approaches | Planning phase |
| **CoVE** | Verify factual claims in codebase documentation | Analyze phase |

**Integration:**

```python
# In modify_codebase(), after planning:
if hasattr(self, '_ara') and self._ara:
    # Pre-mortem analysis of the planned changes
    pm_result = await self._ara.execute_task_with_pipeline(
        task=Task(
            id="premortem",
            type=TaskType.REASONING,
            prompt=f"Review this change plan for failure modes:\n{json.dumps(tasks)}",
        ),
        method=ReasoningMethod.PRE_MORTEM,
    )
    self.project_context.record_phase_learning(
        "pre_mortem", pm_result.score, pm_result.output
    )
```

---

### Phase 4: CodebaseWriter Module

**New file:** `orchestrator/codebase_writer.py`  
**Effort:** 3 days  
**Dependency:** Phase 3

#### 4.1 FileOperations

```python
class FileOperations:
    """Safe file modification operations."""
    
    def read_file(self, path: Path) -> str:
        """Read existing file, preserving line endings."""
        ...
    
    def create_file(self, path: Path, content: str) -> None:
        """Create new file with parent directories."""
        ...
    
    def modify_file(self, path: Path, content: str, strategy: str = "replace") -> None:
        """
        Apply changes to an existing file.
        
        Strategies:
        - "replace": Replace entire file content
        - "insert_before": Insert at a specific line
        - "insert_after": Insert after a specific line
        - "patch_function": Replace a specific function body
        """
        ...
    
    def delete_file(self, path: Path) -> None:
        """Delete file (moves to trash, not permanent)."""
        ...
```

#### 4.2 DiffEngine

```python
class DiffEngine:
    """Generate and apply diffs."""
    
    def generate_diff(self, original: str, modified: str, path: str) -> str:
        """Generate unified diff string."""
        ...
    
    def save_diff(self, diff: str, output_dir: Path) -> Path:
        """Save diff to file for human review."""
        ...
```

#### 4.3 ModificationGate (Safety)

```python
class ModificationGate:
    """
    Safety checks before applying any modification.
    All checks must pass before write.
    """
    
    async def verify(self, task: Task, result: TaskResult, repo_root: Path) -> VerificationResult:
        """
        Run:
        1. Syntax validation of the modified file
        2. Import resolution (imports in modified file must exist)
        3. Existing tests in modified area still pass
        4. No security issues (bandit on changed code)
        5. No hardcoded secrets in diff
        """
        ...
```

---

### Phase 5: Git Integration Module

**New file:** `orchestrator/codebase_git.py`  
**Effort:** 2 days  
**Dependency:** Phase 4

#### 5.1 GitOperations

```python
class GitIntegration:
    """
    Git operations for the modify flow.
    Uses 'gitpython' library.
    """
    def __init__(self, root: Path):
        self.repo = git.Repo(root)
    
    def create_branch(self, name: str) -> bool:
        """Create feature branch from current HEAD."""
        ...
    
    def stage_changes(self, paths: list[Path]) -> None:
        """Stage specific files for commit."""
        ...
    
    def commit(self, message: str) -> str:
        """Commit staged changes, return commit hash."""
        ...
    
    async def create_pr(self, title: str, body: str) -> str:
        """Create GitHub PR via gh CLI."""
        ...
```

---

### Phase 6: Testing

**New files:** `tests/test_codebase_reader.py`, `tests/test_codebase_context.py`, `tests/test_codebase_writer.py`  
**Effort:** 3 days

#### 6.1 Test: CodebaseReader

```python
class TestFileSystemWalker:
    def test_walks_python_files(self):
        """Walk a small test repo, count Python files."""
    def test_respects_gitignore(self):
        """Ignored files should not appear in output."""
    def test_empty_directory(self):
        """Empty dir returns empty list."""

class TestASTIndexer:
    def test_extracts_functions(self):
        """Python function definitions are found."""
    def test_extracts_classes(self):
        """Python class definitions are found."""
    def test_extracts_imports(self):
        """Import statements are captured as dependencies."""
    def test_empty_file(self):
        """Empty file returns empty list."""

class TestDependencyGraph:
    def test_finds_circular_imports(self):
        """Circular A→B→A is detected."""
    def test_finds_unused_modules(self):
        """Module with zero dependents is flagged."""
    def test_blast_radius(self):
        """Changing A returns A's transitive dependents."""
```

#### 6.2 Test: CodebaseWriter

```python
class TestFileOperations:
    def test_create_file(self, tmp_path):
        """File is created with correct content."""
    def test_modify_file_replace(self, tmp_path):
        """Full file replacement works."""
    def test_modify_file_noop(self, tmp_path):
        """No-op modification doesn't change file."""
    def test_delete_file(self, tmp_path):
        """File is removed."""

class TestModificationGate:
    def test_syntax_valid(self, tmp_path):
        """Valid Python code passes syntax check."""
    def test_syntax_invalid(self, tmp_path):
        """Invalid Python code fails syntax check."""
```

#### 6.3 Integration Test

```python
class TestModifyFlow:
    @pytest.mark.asyncio
    async def test_modify_small_repo(self, tmp_path):
        """End-to-end: read, analyze, plan, write on a small test repo."""
```

---

## Implementation Order & Effort

| Phase | Module | Days | Risk | Dependencies |
|-------|--------|------|------|--------------|
| 1 | `CodebaseReader` (walker + indexer + graph + profiler) | 5 | Low | None |
| 2 | `CodebaseContext` (ranker + context builder + analyzer) | 3 | Low | Phase 1 |
| 3 | Pipeline integration (decomposer + CLI + engine) | 4 | Medium | Phase 2 |
| 4 | `CodebaseWriter` (file ops + diff + safety gate) | 3 | Medium | Phase 3 |
| 5 | `GitIntegration` (branch + commit + PR) | 2 | Low | Phase 4 |
| 6 | Testing (unit + integration) | 3 | None | All above |
| **Total** | | **20 days** | | |

---

## Files to Create / Modify

| File | Action | Phase |
|------|--------|-------|
| `orchestrator/codebase_reader.py` | **Create** | 1 |
| `orchestrator/codebase_context.py` | **Create** | 2 |
| `orchestrator/codebase_decomposer.py` | **Create** | 3 |
| `orchestrator/codebase_writer.py` | **Create** | 4 |
| `orchestrator/codebase_git.py` | **Create** | 5 |
| `orchestrator/models.py` | Modify (add `TaskType`, Task fields) | 3 |
| `orchestrator/engine.py` | Modify (add `modify_codebase()`) | 3 |
| `orchestrator/cli.py` | Modify (add `modify` subcommand) | 3 |
| `orchestrator/project_context.py` | Modify (add codebase-specific fields) | 3 |
| `orchestrator/engine_deps.py` | Modify (add optional imports) | 3 |
| `tests/test_codebase_reader.py` | **Create** | 6 |
| `tests/test_codebase_context.py` | **Create** | 6 |
| `tests/test_codebase_writer.py` | **Create** | 6 |
| `pyproject.toml` | Modify (add `networkx`, `pathspec` deps) | 1 |
| `.env.example` | Modify (add codebase env vars) | 6 |

---

## Dependencies to Add

| Library | Purpose | Phase |
|---------|---------|-------|
| `pathspec>=0.12.0` | `.gitignore` pattern matching | 1 |
| `networkx>=3.0` | Dependency graph + PageRank | 1 |
| `tree-sitter>=0.21` | Multi-language AST parsing (optional) | 1 |
| `gitpython>=3.1` | Git operations | 5 |

---

## Rollback Safety

| Phase | Rollback Mechanism |
|-------|-------------------|
| 1-2 | Pure computation, no side effects — revert commit |
| 3 | `ModificationResult` captures all changes before writing |
| 4 | `CodebaseWriter` saves backup of modified files |
| 5 | `GitIntegration` creates branch — `git checkout .` to revert |
| All | `--dry-run` flag prevents any writes |

---

## Verification Gates

- [ ] `python -m pytest tests/test_codebase_reader.py` — all pass
- [ ] `python -m orchestrator modify --repo ./orchestrator --objective "Add logging" --dry-run` — produces task plan without changes
- [ ] `python -m orchestrator modify --repo ./tests/fixtures/small_repo --objective "Add tests"` — produces modified files
- [ ] Syntax validation catches broken code before write
- [ ] `git diff` on modified repo shows only intended changes
- [ ] All existing tests still pass (no regression)

---

## Appendix A: CLI Examples

```bash
# Analyze a codebase
python -m orchestrator modify --repo ./my-app --objective "Add authentication" --dry-run

# Modify with budget limit
python -m orchestrator modify --repo ./my-app --objective "Add JWT auth" --budget 10.0

# Modify without safety gates (faster, riskier)
python -m orchestrator modify --repo ./my-app --objective "Fix all lint errors" --bypass-gates

# Create PR with changes
python -m orchestrator modify --repo ./my-app --objective "Migrate to FastAPI" --create-pr

# Full analysis report
python -m orchestrator analyze --repo ./my-app --output report.json
```

## Appendix B: Safety Gate Pipeline

```
Task Result → ModificationGate
                │
                ├── Syntax Check → FAIL → BLOCKED (report error)
                │
                ├── Import Resolution → FAIL → BLOCKED (report missing import)
                │
                ├── Security Scan → FAIL → BLOCKED (report vulnerability)
                │
                ├── Secret Detection → FAIL → BLOCKED (redact and report)
                │
                └── All PASS → WRITE to disk
                                │
                                ├── Backup original file
                                ├── Write modified content
                                └── Run targeted tests → FAIL → ROLLBACK
```

---

**Last updated:** 2026-05-23
