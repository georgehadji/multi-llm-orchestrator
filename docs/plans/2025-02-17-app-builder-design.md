# App Builder Design
**Date:** 2025-02-17
**Status:** Approved
**Goal:** Extend the orchestrator to produce verified, production-ready apps from a plain-text description.

---

## Problem Statement

The current orchestrator produces individual task output files (`task_001_code_generation.py`, etc.) but does not assemble them into a runnable project. Building a "ready app" requires:

1. Detecting what kind of app to build
2. Scaffolding the correct folder structure
3. Routing LLM outputs to the right file paths
4. Resolving dependencies
5. Verifying the app actually runs (local + Docker)

---

## Architecture Overview

```
ProjectDescription + app_type (optional)
         │
    [AppDetector]            orchestrator/app_detector.py
         │  app_type, tech_stack, entry_point hint
    [ScaffoldEngine]         orchestrator/scaffold/
         │  empty folder tree + boilerplate files
    [SmartDecomposer]        orchestrator/decomposer.py (extended)
         │  Task[] with target_path field per task
    [Orchestrator]           orchestrator/engine.py (unchanged core)
         │  TaskResult[] with output = file content
    [AppAssembler]           orchestrator/app_assembler.py
         │  writes files to output_dir/, resolves cross-file imports
    [DependencyResolver]     orchestrator/dep_resolver.py
         │  pyproject.toml / requirements.txt
    [AppVerifier]            orchestrator/app_verifier.py
         │  local: pytest + health check subprocess
         │  docker: Dockerfile generation + docker build + docker run
         ▼
    ✅  output_dir/ — verified running app
```

The core `Orchestrator` engine is **not modified** — the new pipeline wraps it.

---

## Component Designs

### 1. AppDetector (`orchestrator/app_detector.py`)

One LLM call with a structured JSON response:

```python
@dataclass
class AppProfile:
    app_type: Literal["fastapi", "flask", "cli", "library", "script",
                      "react-fastapi", "nextjs", "generic"]
    tech_stack: list[str]          # ["python", "fastapi", "sqlalchemy"]
    entry_point: str               # "src/main.py" | "src/app.py" | "cli.py"
    test_command: str              # "pytest" | "python -m pytest"
    run_command: str               # "uvicorn src.main:app" | "python cli.py"
    requires_docker: bool          # True for full-stack apps
    detected_from: str             # "auto" | "yaml_override"
```

Detection logic:
- If YAML has `app_type:` field → `detected_from = "yaml_override"`, skip LLM call
- Otherwise → single LLM call to `AppDetector.detect(description) -> AppProfile`
- Fallback: `app_type = "script"` (safest default)

---

### 2. ScaffoldEngine (`orchestrator/scaffold/`)

Per-app-type folder templates stored as Python dicts (no external files needed):

```
orchestrator/scaffold/
├── __init__.py          # ScaffoldEngine class
├── templates/
│   ├── fastapi.py       # {"src/__init__.py": "", "src/main.py": FASTAPI_MAIN, ...}
│   ├── cli.py           # {"cli.py": CLI_ENTRY, "src/__init__.py": "", ...}
│   ├── library.py
│   ├── react_fastapi.py
│   └── generic.py       # fallback: flat structure
```

`ScaffoldEngine.scaffold(profile, output_dir)` → creates folders + boilerplate files, returns `dict[str, str]` (path → initial content).

Boilerplate includes: `pyproject.toml` skeleton, `.env.example`, `README.md` stub, `Dockerfile` stub, `.gitignore`.

---

### 3. SmartDecomposer (extension of existing decomposer)

Adds `target_path` to each Task's prompt context:

```python
@dataclass
class Task:
    ...
    target_path: str = ""    # NEW: "src/routes/auth.py", "tests/test_auth.py"
    module_name: str = ""    # NEW: "src.routes.auth"
```

The decomposition system prompt is extended with:
```
Each task must output the complete content of a single file.
The file path is specified in the task description as TARGET_PATH.
Use absolute imports based on MODULE_NAME.
```

The `SmartDecomposer` receives the `ScaffoldEngine` output and assigns `target_path` to each task during decomposition.

---

### 4. AppAssembler (`orchestrator/app_assembler.py`)

After all tasks complete:

```python
class AppAssembler:
    def assemble(
        self,
        results: dict[str, TaskResult],
        tasks: dict[str, Task],
        scaffold: dict[str, str],
        output_dir: Path,
    ) -> AssemblyReport:
        ...
```

Steps:
1. For each `TaskResult`, write `result.output` to `task.target_path` inside `output_dir`
2. Scaffold files not overwritten by tasks remain as boilerplate
3. Run `ImportFixer`: scan all `.py` files, detect cross-file imports, add missing `__init__.py` files
4. Return `AssemblyReport(files_written, files_kept_from_scaffold, import_issues)`

---

### 5. DependencyResolver (`orchestrator/dep_resolver.py`)

```python
class DependencyResolver:
    def resolve(self, output_dir: Path, profile: AppProfile) -> ResolveReport:
        ...
```

Steps:
1. Walk all `.py` files, collect `import X` / `from X import` statements
2. Filter to third-party packages (not stdlib, not local modules)
3. Map known package names → PyPI names (e.g. `cv2` → `opencv-python`)
4. Write `pyproject.toml` `[project.dependencies]` + `requirements.txt`
5. Return `ResolveReport(packages_found, pyproject_path, requirements_path)`

No `pip install` at this stage — just file generation.

---

### 6. AppVerifier (`orchestrator/app_verifier.py`)

Two-phase verification:

#### Phase 1 — Local
```python
@dataclass
class LocalVerifyResult:
    install_ok: bool      # pip install -e . exit code
    tests_pass: bool      # pytest exit code
    app_starts: bool      # run_command starts without crash in 3s
    health_ok: bool       # GET /health returns 2xx (web apps only)
    errors: list[str]
```

Steps:
1. `pip install -e .` in output_dir (or `uv sync` if available)
2. `pytest` → capture pass/fail
3. `subprocess.Popen(run_command)` → wait 3s → check process alive
4. For web apps: `GET http://localhost:PORT/health` → expect 2xx

#### Phase 2 — Docker (optional, triggered if `profile.requires_docker` or user flag)
```python
@dataclass
class DockerVerifyResult:
    dockerfile_generated: bool
    build_ok: bool
    run_ok: bool
    health_ok: bool
    errors: list[str]
```

Steps:
1. Generate `Dockerfile` from template (multi-stage, non-root user)
2. `docker build -t app-{project_id} .`
3. `docker run -d -p PORT:PORT app-{project_id}`
4. `GET http://localhost:PORT/health` → expect 2xx
5. `docker stop` + cleanup

---

### 7. AppBuildResult (top-level return type)

```python
@dataclass
class AppBuildResult:
    project_state: ProjectState      # existing orchestrator result
    profile: AppProfile
    assembly: AssemblyReport
    dependencies: ResolveReport
    local_verify: LocalVerifyResult
    docker_verify: Optional[DockerVerifyResult]
    output_dir: Path
    success: bool                    # all phases passed
```

---

## Entry Points

### Python API

```python
from orchestrator import AppBuilder, Budget

builder = AppBuilder(budget=Budget(max_usd=15.0))
result = asyncio.run(builder.build(
    description="Build a FastAPI REST API with JWT auth and SQLite",
    success_criteria="All endpoints return correct status codes",
    output_dir="./my_app",
    docker=True,
))

print(result.success)                          # True
print(result.local_verify.tests_pass)          # True
print(result.docker_verify.health_ok)          # True
```

### CLI

```bash
python -m orchestrator build \
  --description "Build a FastAPI REST API with JWT auth and SQLite" \
  --criteria "All endpoints tested" \
  --output-dir ./my_app \
  --docker \
  --budget 15.0
```

### YAML project file (new fields)

```yaml
project_description: "Build a FastAPI REST API with JWT auth"
success_criteria: "All endpoints return correct status codes"
budget:
  max_usd: 15.0
app_type: fastapi        # optional override
docker: true             # optional
output_dir: ./my_app     # optional
```

---

## Data Flow

```
build(description, criteria, output_dir, docker)
  │
  ├─ AppDetector.detect(description)              → AppProfile
  ├─ ScaffoldEngine.scaffold(profile, output_dir) → scaffold dict
  ├─ SmartDecomposer.decompose(description, scaffold) → Task[] with target_path
  ├─ Orchestrator.run_job(tasks)                  → ProjectState
  ├─ AppAssembler.assemble(state, tasks, scaffold, output_dir) → AssemblyReport
  ├─ DependencyResolver.resolve(output_dir, profile) → ResolveReport
  ├─ AppVerifier.verify_local(output_dir, profile)   → LocalVerifyResult
  └─ AppVerifier.verify_docker(output_dir, profile)  → DockerVerifyResult (optional)
```

---

## Error Handling

| Failure Point | Behaviour |
|---------------|-----------|
| AppDetector fails | Fallback to `app_type = "script"` |
| ScaffoldEngine unknown type | Use `generic` template |
| Task output is empty/invalid | AssemblyReport records as `files_skipped`, scaffold boilerplate kept |
| ImportFixer can't resolve import | Logged as warning, not blocking |
| DependencyResolver unknown package | Listed in `ResolveReport.unknown_packages` for user review |
| Local install fails | `local_verify.install_ok = False`, verification continues with best effort |
| pytest fails | Recorded in result; does NOT block Docker phase |
| Docker not available | `docker_verify = None`, logged as info |

---

## New Files

| File | Role |
|------|------|
| `orchestrator/app_builder.py` | Top-level `AppBuilder` class, pipeline orchestration |
| `orchestrator/app_detector.py` | `AppDetector`, `AppProfile` |
| `orchestrator/app_assembler.py` | `AppAssembler`, `AssemblyReport` |
| `orchestrator/dep_resolver.py` | `DependencyResolver`, `ResolveReport` |
| `orchestrator/app_verifier.py` | `AppVerifier`, `LocalVerifyResult`, `DockerVerifyResult` |
| `orchestrator/scaffold/__init__.py` | `ScaffoldEngine` |
| `orchestrator/scaffold/templates/fastapi.py` | FastAPI template |
| `orchestrator/scaffold/templates/cli.py` | CLI template |
| `orchestrator/scaffold/templates/library.py` | Library template |
| `orchestrator/scaffold/templates/react_fastapi.py` | Full-stack template |
| `orchestrator/scaffold/templates/generic.py` | Fallback template |

## Modified Files

| File | Change |
|------|--------|
| `orchestrator/models.py` | Add `target_path: str`, `module_name: str` to `Task` |
| `orchestrator/__init__.py` | Export `AppBuilder`, `AppBuildResult` |
| `orchestrator/cli.py` | Add `build` subcommand |
| `orchestrator/project_file.py` | Add `app_type`, `docker`, `output_dir` fields |
| `tests/` | 6 new test files (~120 tests) |

---

## Testing Strategy

| Test File | Covers |
|-----------|--------|
| `test_app_detector.py` | Auto-detection, YAML override, fallback |
| `test_scaffold_engine.py` | All templates, file structure validation |
| `test_smart_decomposer.py` | target_path assignment, module_name generation |
| `test_app_assembler.py` | File writing, ImportFixer, scaffold preservation |
| `test_dep_resolver.py` | Import scanning, pyproject.toml generation |
| `test_app_verifier.py` | Local verify (mocked subprocess), Docker verify (mocked) |

All subprocess and Docker calls are **mocked** in tests — no actual process spawning.

---

## Constraints & Non-Goals

- **No frontend build tools** (webpack, npm) in Phase 1 — React/Next.js scaffolded but build step is user's responsibility
- **No cloud deploy** — local + Docker only
- **No multi-repo** — single output directory
- **Backwards compatible** — existing `Orchestrator` API unchanged; `AppBuilder` is additive
