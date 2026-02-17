# App Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the orchestrator so that `AppBuilder.build(description)` produces a verified, runnable app — from folder scaffold through local pytest + optional Docker health check.

**Architecture:** Additive pipeline wrapping the unchanged `Orchestrator` core: `AppDetector → ScaffoldEngine → SmartDecomposer → Orchestrator → AppAssembler → DependencyResolver → AppVerifier`. Each component is a standalone class with no circular imports.

**Tech Stack:** Python 3.10+, stdlib only (subprocess, ast, pathlib, dataclasses). No new runtime dependencies. pytest + unittest.mock for tests.

---

## Task 1: Extend `Task` model with `target_path` and `module_name`

**Files:**
- Modify: `orchestrator/models.py` (Task dataclass, ~line 174)
- Modify: `tests/test_constraint_planner.py` (verify nothing breaks)

**Step 1: Write failing test**

```python
# tests/test_app_builder_models.py
from orchestrator.models import Task, TaskType

def test_task_has_target_path_default_empty():
    t = Task(id="t1", type=TaskType.CODE_GEN, prompt="write code")
    assert t.target_path == ""

def test_task_has_module_name_default_empty():
    t = Task(id="t1", type=TaskType.CODE_GEN, prompt="write code")
    assert t.module_name == ""

def test_task_target_path_can_be_set():
    t = Task(id="t1", type=TaskType.CODE_GEN, prompt="p",
             target_path="src/routes/auth.py", module_name="src.routes.auth")
    assert t.target_path == "src/routes/auth.py"
    assert t.module_name == "src.routes.auth"
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_app_builder_models.py -v
```
Expected: FAIL — `Task.__init__() got an unexpected keyword argument 'target_path'`

**Step 3: Add fields to Task**

In `orchestrator/models.py`, after `hard_validators` field in the `Task` dataclass:

```python
    # App Builder fields (Improvement 8)
    target_path: str = ""     # e.g. "src/routes/auth.py"
    module_name: str = ""     # e.g. "src.routes.auth"
```

**Step 4: Run tests**

```
pytest tests/test_app_builder_models.py tests/test_constraint_planner.py tests/test_engine_e2e.py -v
```
Expected: ALL PASS (new fields have defaults — fully backwards compatible)

**Step 5: Commit**

```bash
git add orchestrator/models.py tests/test_app_builder_models.py
git commit -m "feat: add target_path and module_name fields to Task (App Builder prep)"
```

---

## Task 2: `AppProfile` dataclass + `AppDetector`

**Files:**
- Create: `orchestrator/app_detector.py`
- Create: `tests/test_app_detector.py`

**Step 1: Write failing tests**

```python
# tests/test_app_detector.py
import pytest
from orchestrator.app_detector import AppDetector, AppProfile

def test_app_profile_defaults():
    p = AppProfile(app_type="fastapi", tech_stack=["python","fastapi"],
                   entry_point="src/main.py", test_command="pytest",
                   run_command="uvicorn src.main:app --host 0.0.0.0 --port 8000",
                   requires_docker=False, detected_from="auto")
    assert p.app_type == "fastapi"
    assert p.detected_from == "auto"

def test_detect_from_yaml_override_skips_llm():
    detector = AppDetector(api_client=None)  # no client needed
    profile = detector.detect_from_yaml({"app_type": "cli", "project_description": "a tool"})
    assert profile.app_type == "cli"
    assert profile.detected_from == "yaml_override"

def test_detect_from_yaml_no_app_type_returns_none():
    detector = AppDetector(api_client=None)
    profile = detector.detect_from_yaml({"project_description": "a tool"})
    assert profile is None  # must fall through to LLM detection

def test_parse_llm_response_valid_json():
    detector = AppDetector(api_client=None)
    raw = '{"app_type":"fastapi","tech_stack":["python","fastapi"],"entry_point":"src/main.py","test_command":"pytest","run_command":"uvicorn src.main:app","requires_docker":false}'
    profile = detector._parse_llm_response(raw)
    assert profile.app_type == "fastapi"
    assert "fastapi" in profile.tech_stack
    assert profile.requires_docker is False

def test_parse_llm_response_invalid_json_returns_fallback():
    detector = AppDetector(api_client=None)
    profile = detector._parse_llm_response("not json at all")
    assert profile.app_type == "script"
    assert profile.detected_from == "fallback"

def test_parse_llm_response_missing_fields_uses_defaults():
    detector = AppDetector(api_client=None)
    # Partial JSON — only app_type
    profile = detector._parse_llm_response('{"app_type":"flask"}')
    assert profile.app_type == "flask"
    assert profile.entry_point == "src/main.py"  # default for web apps
    assert profile.test_command == "pytest"

def test_run_command_default_for_known_types():
    detector = AppDetector(api_client=None)
    for raw, expected_fragment in [
        ('{"app_type":"fastapi"}', "uvicorn"),
        ('{"app_type":"cli"}',    "python"),
        ('{"app_type":"script"}', "python"),
    ]:
        p = detector._parse_llm_response(raw)
        assert expected_fragment in p.run_command

def test_requires_docker_true_for_full_stack():
    detector = AppDetector(api_client=None)
    p = detector._parse_llm_response('{"app_type":"react-fastapi"}')
    assert p.requires_docker is True
```

**Step 2: Run to verify failures**

```
pytest tests/test_app_detector.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator.app_detector'`

**Step 3: Implement `orchestrator/app_detector.py`**

```python
"""
AppDetector — infers app type, tech stack, and run commands from a project description.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger("orchestrator.app_detector")

AppType = Literal["fastapi", "flask", "cli", "library", "script",
                  "react-fastapi", "nextjs", "generic"]

_DEFAULTS: dict[str, dict] = {
    "fastapi":      {"entry_point": "src/main.py",  "run_command": "uvicorn src.main:app --host 0.0.0.0 --port 8000", "requires_docker": False},
    "flask":        {"entry_point": "src/main.py",  "run_command": "python src/main.py", "requires_docker": False},
    "cli":          {"entry_point": "cli.py",        "run_command": "python cli.py --help", "requires_docker": False},
    "library":      {"entry_point": "src/__init__.py","run_command": "python -c 'import src'", "requires_docker": False},
    "script":       {"entry_point": "main.py",       "run_command": "python main.py", "requires_docker": False},
    "react-fastapi":{"entry_point": "backend/main.py","run_command": "uvicorn backend.main:app --host 0.0.0.0 --port 8000", "requires_docker": True},
    "nextjs":       {"entry_point": "src/app/page.tsx","run_command": "npm run dev", "requires_docker": True},
    "generic":      {"entry_point": "main.py",       "run_command": "python main.py", "requires_docker": False},
}


@dataclass
class AppProfile:
    app_type: str
    tech_stack: list[str]
    entry_point: str
    test_command: str
    run_command: str
    requires_docker: bool
    detected_from: str  # "auto" | "yaml_override" | "fallback"


_FALLBACK = AppProfile(
    app_type="script",
    tech_stack=["python"],
    entry_point="main.py",
    test_command="pytest",
    run_command="python main.py",
    requires_docker=False,
    detected_from="fallback",
)

_DETECT_PROMPT = """\
Analyze the following project description and return a JSON object with these fields:
- app_type: one of fastapi|flask|cli|library|script|react-fastapi|nextjs|generic
- tech_stack: list of key technologies (e.g. ["python","fastapi","sqlalchemy"])
- entry_point: relative path to main entry file (e.g. "src/main.py")
- test_command: command to run tests (usually "pytest")
- run_command: command to start the app
- requires_docker: true only for full-stack or multi-service apps

Return ONLY the JSON object, no markdown fences.

Project description:
{description}
"""


class AppDetector:
    def __init__(self, api_client) -> None:
        self._client = api_client

    def detect_from_yaml(self, yaml_dict: dict) -> Optional[AppProfile]:
        """Return AppProfile from YAML override, or None if no app_type key."""
        app_type = yaml_dict.get("app_type")
        if not app_type:
            return None
        defaults = _DEFAULTS.get(str(app_type), _DEFAULTS["generic"])
        return AppProfile(
            app_type=str(app_type),
            tech_stack=yaml_dict.get("tech_stack", ["python"]),
            entry_point=yaml_dict.get("entry_point", defaults["entry_point"]),
            test_command=yaml_dict.get("test_command", "pytest"),
            run_command=yaml_dict.get("run_command", defaults["run_command"]),
            requires_docker=yaml_dict.get("requires_docker", defaults["requires_docker"]),
            detected_from="yaml_override",
        )

    def _parse_llm_response(self, raw: str) -> AppProfile:
        """Parse LLM JSON response into AppProfile; returns fallback on any error."""
        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
                text = text.rstrip("`").strip()
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("AppDetector: could not parse LLM response, using fallback")
            return _FALLBACK

        app_type = str(data.get("app_type", "script"))
        defaults = _DEFAULTS.get(app_type, _DEFAULTS["generic"])
        return AppProfile(
            app_type=app_type,
            tech_stack=data.get("tech_stack", ["python"]),
            entry_point=data.get("entry_point", defaults["entry_point"]),
            test_command=data.get("test_command", "pytest"),
            run_command=data.get("run_command", defaults["run_command"]),
            requires_docker=data.get("requires_docker", defaults["requires_docker"]),
            detected_from="auto",
        )

    async def detect(self, description: str) -> AppProfile:
        """Call LLM to detect app type. Falls back to 'script' on failure."""
        if self._client is None:
            return _FALLBACK
        try:
            prompt = _DETECT_PROMPT.format(description=description)
            raw = await self._client.generate(prompt, max_tokens=256)
            return self._parse_llm_response(raw)
        except Exception as exc:
            logger.warning("AppDetector.detect failed: %s — using fallback", exc)
            return _FALLBACK
```

**Step 4: Run tests**

```
pytest tests/test_app_detector.py -v
```
Expected: ALL 8 PASS

**Step 5: Commit**

```bash
git add orchestrator/app_detector.py tests/test_app_detector.py
git commit -m "feat: AppDetector — infer app_type from description or YAML override"
```

---

## Task 3: `ScaffoldEngine` + templates

**Files:**
- Create: `orchestrator/scaffold/__init__.py`
- Create: `orchestrator/scaffold/templates/__init__.py`
- Create: `orchestrator/scaffold/templates/fastapi.py`
- Create: `orchestrator/scaffold/templates/cli.py`
- Create: `orchestrator/scaffold/templates/library.py`
- Create: `orchestrator/scaffold/templates/generic.py`
- Create: `tests/test_scaffold_engine.py`

**Step 1: Write failing tests**

```python
# tests/test_scaffold_engine.py
import pytest
from pathlib import Path
from orchestrator.scaffold import ScaffoldEngine
from orchestrator.app_detector import AppProfile

def _profile(app_type: str) -> AppProfile:
    return AppProfile(app_type=app_type, tech_stack=["python"],
                      entry_point="src/main.py", test_command="pytest",
                      run_command="python main.py", requires_docker=False,
                      detected_from="auto")

def test_scaffold_fastapi_creates_src_dir(tmp_path):
    engine = ScaffoldEngine()
    scaffold = engine.scaffold(_profile("fastapi"), tmp_path)
    assert (tmp_path / "src" / "__init__.py").exists()
    assert (tmp_path / "src" / "main.py").exists()

def test_scaffold_fastapi_has_pyproject_toml(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("fastapi"), tmp_path)
    assert (tmp_path / "pyproject.toml").exists()

def test_scaffold_cli_creates_cli_py(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("cli"), tmp_path)
    assert (tmp_path / "cli.py").exists()

def test_scaffold_library_has_src_and_tests(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("library"), tmp_path)
    assert (tmp_path / "src" / "__init__.py").exists()
    assert (tmp_path / "tests" / "__init__.py").exists()

def test_scaffold_unknown_type_uses_generic(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("unknown_xyz"), tmp_path)
    assert (tmp_path / "main.py").exists()

def test_scaffold_returns_dict_of_path_to_content(tmp_path):
    engine = ScaffoldEngine()
    result = engine.scaffold(_profile("fastapi"), tmp_path)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result)

def test_scaffold_gitignore_always_created(tmp_path):
    engine = ScaffoldEngine()
    for app_type in ["fastapi", "cli", "library", "generic"]:
        d = tmp_path / app_type
        d.mkdir()
        engine.scaffold(_profile(app_type), d)
        assert (d / ".gitignore").exists(), f"Missing .gitignore for {app_type}"

def test_scaffold_env_example_always_created(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("fastapi"), tmp_path)
    assert (tmp_path / ".env.example").exists()

def test_scaffold_does_not_overwrite_existing_files(tmp_path):
    (tmp_path / "src").mkdir()
    existing = tmp_path / "src" / "main.py"
    existing.write_text("# existing content")
    engine = ScaffoldEngine()
    engine.scaffold(_profile("fastapi"), tmp_path)
    assert existing.read_text() == "# existing content"

def test_scaffold_readme_stub_created(tmp_path):
    engine = ScaffoldEngine()
    engine.scaffold(_profile("fastapi"), tmp_path)
    assert (tmp_path / "README.md").exists()
```

**Step 2: Run to verify failures**

```
pytest tests/test_scaffold_engine.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator.scaffold'`

**Step 3: Implement scaffold package**

Create `orchestrator/scaffold/templates/__init__.py` (empty).

Create `orchestrator/scaffold/templates/fastapi.py`:
```python
FASTAPI_MAIN = '''\
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
'''

FASTAPI_PYPROJECT = '''\
[project]
name = "app"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"
'''

FILES: dict[str, str] = {
    "src/__init__.py": "",
    "src/main.py": FASTAPI_MAIN,
    "tests/__init__.py": "",
    "tests/test_health.py": "from fastapi.testclient import TestClient\nfrom src.main import app\n\ndef test_health():\n    r = TestClient(app).get('/health')\n    assert r.status_code == 200\n",
    "pyproject.toml": FASTAPI_PYPROJECT,
    ".env.example": "# Add environment variables here\n",
    ".gitignore": "__pycache__/\n*.pyc\n.env\n.venv/\ndist/\n",
    "README.md": "# App\n\nGenerated by multi-llm-orchestrator.\n\n## Run\n\n```bash\nuvicorn src.main:app --reload\n```\n",
}
```

Create `orchestrator/scaffold/templates/cli.py`:
```python
CLI_MAIN = '''\
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Hello from CLI!")

if __name__ == "__main__":
    main()
'''

FILES: dict[str, str] = {
    "cli.py": CLI_MAIN,
    "src/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_cli.py": "from cli import main\n\ndef test_main_runs():\n    main()\n",
    "pyproject.toml": '[project]\nname = "app"\nversion = "0.1.0"\n',
    ".env.example": "",
    ".gitignore": "__pycache__/\n*.pyc\n.env\n.venv/\n",
    "README.md": "# CLI App\n\n```bash\npython cli.py\n```\n",
}
```

Create `orchestrator/scaffold/templates/library.py`:
```python
FILES: dict[str, str] = {
    "src/__init__.py": '"""Library package."""\n',
    "src/core.py": "",
    "tests/__init__.py": "",
    "tests/test_core.py": "from src.core import *\n",
    "pyproject.toml": '[project]\nname = "app"\nversion = "0.1.0"\n',
    ".env.example": "",
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\ndist/\n",
    "README.md": "# Library\n",
}
```

Create `orchestrator/scaffold/templates/generic.py`:
```python
FILES: dict[str, str] = {
    "main.py": "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
    "tests/__init__.py": "",
    "tests/test_main.py": "from main import main\n\ndef test_main():\n    main()\n",
    "pyproject.toml": '[project]\nname = "app"\nversion = "0.1.0"\n',
    ".env.example": "",
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\n",
    "README.md": "# App\n",
}
```

Create `orchestrator/scaffold/__init__.py`:
```python
"""ScaffoldEngine — creates folder structure + boilerplate for generated apps."""
from __future__ import annotations
import logging
from pathlib import Path
from orchestrator.app_detector import AppProfile

logger = logging.getLogger("orchestrator.scaffold")

_TEMPLATE_MAP: dict[str, str] = {
    "fastapi": "fastapi", "flask": "fastapi",
    "cli": "cli", "script": "cli",
    "library": "library",
    "react-fastapi": "generic", "nextjs": "generic", "generic": "generic",
}


class ScaffoldEngine:
    def scaffold(self, profile: AppProfile, output_dir: Path) -> dict[str, str]:
        """
        Create directory structure and boilerplate files.
        Returns dict[relative_path_str → content] of ALL scaffold files.
        Existing files are NOT overwritten.
        """
        template_name = _TEMPLATE_MAP.get(profile.app_type, "generic")
        module = _load_template(template_name)
        files: dict[str, str] = module.FILES

        result: dict[str, str] = {}
        for rel_path, content in files.items():
            dest = output_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_text(content, encoding="utf-8")
                logger.debug("Scaffold: created %s", rel_path)
            else:
                logger.debug("Scaffold: skipped existing %s", rel_path)
            result[rel_path] = content
        return result


def _load_template(name: str):
    import importlib
    return importlib.import_module(f"orchestrator.scaffold.templates.{name}")
```

**Step 4: Run tests**

```
pytest tests/test_scaffold_engine.py -v
```
Expected: ALL 10 PASS

**Step 5: Commit**

```bash
git add orchestrator/scaffold/ tests/test_scaffold_engine.py
git commit -m "feat: ScaffoldEngine — folder templates for fastapi/cli/library/generic"
```

---

## Task 4: `AppAssembler` + `ImportFixer`

**Files:**
- Create: `orchestrator/app_assembler.py`
- Create: `tests/test_app_assembler.py`

**Step 1: Write failing tests**

```python
# tests/test_app_assembler.py
import pytest
from pathlib import Path
from orchestrator.app_assembler import AppAssembler, AssemblyReport
from orchestrator.models import Task, TaskType, TaskResult, Model

def _make_task(task_id, path):
    t = Task(id=task_id, type=TaskType.CODE_GEN, prompt="write code")
    t.target_path = path
    return t

def _make_result(task_id, content):
    return TaskResult(task_id=task_id, output=content, score=0.9, model_used=Model.KIMI_K2_5)

def test_assemble_writes_files_to_target_paths(tmp_path):
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/main.py")}
    results = {"t1": _make_result("t1", "# main")}
    report = assembler.assemble(results, tasks, {}, tmp_path)
    assert (tmp_path / "src" / "main.py").read_text() == "# main"
    assert "src/main.py" in report.files_written

def test_assemble_skips_task_with_empty_target_path(tmp_path):
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "")}  # no target_path
    results = {"t1": _make_result("t1", "# code")}
    report = assembler.assemble(results, tasks, {}, tmp_path)
    assert report.files_written == []
    assert len(report.files_skipped) == 1

def test_assemble_skips_task_with_empty_output(tmp_path):
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/main.py")}
    results = {"t1": _make_result("t1", "")}
    report = assembler.assemble(results, tasks, {}, tmp_path)
    assert report.files_written == []

def test_assemble_keeps_scaffold_files_not_overwritten(tmp_path):
    scaffold = {"pyproject.toml": "[project]"}
    (tmp_path / "pyproject.toml").write_text("[project]")
    assembler = AppAssembler()
    report = assembler.assemble({}, {}, scaffold, tmp_path)
    assert "pyproject.toml" in report.files_from_scaffold

def test_assemble_creates_parent_directories(tmp_path):
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/routes/auth.py")}
    results = {"t1": _make_result("t1", "# auth")}
    assembler.assemble(results, tasks, {}, tmp_path)
    assert (tmp_path / "src" / "routes" / "auth.py").exists()

def test_assemble_adds_missing_init_files(tmp_path):
    assembler = AppAssembler()
    tasks = {"t1": _make_task("t1", "src/routes/auth.py")}
    results = {"t1": _make_result("t1", "# auth")}
    assembler.assemble(results, tasks, {}, tmp_path)
    # src/__init__.py and src/routes/__init__.py should be created
    assert (tmp_path / "src" / "__init__.py").exists()
    assert (tmp_path / "src" / "routes" / "__init__.py").exists()

def test_assemble_returns_assembly_report(tmp_path):
    assembler = AppAssembler()
    report = assembler.assemble({}, {}, {}, tmp_path)
    assert isinstance(report, AssemblyReport)
    assert isinstance(report.files_written, list)
    assert isinstance(report.files_skipped, list)
    assert isinstance(report.files_from_scaffold, list)
    assert isinstance(report.import_issues, list)

def test_assemble_multiple_tasks_all_written(tmp_path):
    assembler = AppAssembler()
    tasks = {
        "t1": _make_task("t1", "src/main.py"),
        "t2": _make_task("t2", "src/models.py"),
    }
    results = {
        "t1": _make_result("t1", "# main"),
        "t2": _make_result("t2", "# models"),
    }
    report = assembler.assemble(results, tasks, {}, tmp_path)
    assert len(report.files_written) == 2
```

**Step 2: Run to verify failures**

```
pytest tests/test_app_assembler.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `orchestrator/app_assembler.py`**

```python
"""
AppAssembler — writes TaskResult outputs to target_path locations,
preserves scaffold boilerplate, and ensures __init__.py coverage.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from orchestrator.models import Task, TaskResult

logger = logging.getLogger("orchestrator.app_assembler")


@dataclass
class AssemblyReport:
    files_written:       list[str] = field(default_factory=list)
    files_skipped:       list[str] = field(default_factory=list)
    files_from_scaffold: list[str] = field(default_factory=list)
    import_issues:       list[str] = field(default_factory=list)


class AppAssembler:
    def assemble(
        self,
        results:    dict[str, TaskResult],
        tasks:      dict[str, Task],
        scaffold:   dict[str, str],
        output_dir: Path,
    ) -> AssemblyReport:
        report = AssemblyReport()
        written_paths: set[str] = set()

        # 1. Write task outputs to their target paths
        for task_id, result in results.items():
            task = tasks.get(task_id)
            if task is None:
                continue
            rel = task.target_path.strip()
            if not rel:
                logger.debug("Skipping task %s — no target_path", task_id)
                report.files_skipped.append(task_id)
                continue
            if not result.output.strip():
                logger.debug("Skipping task %s — empty output", task_id)
                report.files_skipped.append(task_id)
                continue

            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(result.output, encoding="utf-8")
            report.files_written.append(rel)
            written_paths.add(rel)
            logger.debug("Assembled %s → %s", task_id, rel)

        # 2. Record scaffold files not overwritten by tasks
        for rel in scaffold:
            if rel not in written_paths and (output_dir / rel).exists():
                report.files_from_scaffold.append(rel)

        # 3. Ensure __init__.py exists in every Python package directory
        _ensure_init_files(output_dir, report)

        return report


def _ensure_init_files(output_dir: Path, report: AssemblyReport) -> None:
    """Add missing __init__.py in every directory that contains .py files."""
    for py_file in output_dir.rglob("*.py"):
        pkg_dir = py_file.parent
        init = pkg_dir / "__init__.py"
        if not init.exists() and pkg_dir != output_dir:
            init.write_text("", encoding="utf-8")
            logger.debug("ImportFixer: created %s", init.relative_to(output_dir))
```

**Step 4: Run tests**

```
pytest tests/test_app_assembler.py -v
```
Expected: ALL 8 PASS

**Step 5: Commit**

```bash
git add orchestrator/app_assembler.py tests/test_app_assembler.py
git commit -m "feat: AppAssembler — writes task outputs to target_path, ImportFixer"
```

---

## Task 5: `DependencyResolver`

**Files:**
- Create: `orchestrator/dep_resolver.py`
- Create: `tests/test_dep_resolver.py`

**Step 1: Write failing tests**

```python
# tests/test_dep_resolver.py
import pytest
from pathlib import Path
from orchestrator.dep_resolver import DependencyResolver, ResolveReport

def test_resolve_empty_dir_returns_empty_packages(tmp_path):
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert report.packages_found == []

def test_resolve_detects_third_party_imports(tmp_path):
    (tmp_path / "main.py").write_text("import fastapi\nimport sqlalchemy\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="fastapi")
    assert "fastapi" in report.packages_found

def test_resolve_ignores_stdlib_imports(tmp_path):
    (tmp_path / "main.py").write_text("import os\nimport sys\nimport json\nimport pathlib\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert report.packages_found == []

def test_resolve_ignores_local_imports(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "main.py").write_text("from src import core\nimport src.models\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert "src" not in report.packages_found

def test_resolve_writes_requirements_txt(tmp_path):
    (tmp_path / "main.py").write_text("import fastapi\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="fastapi")
    assert report.requirements_path.exists()
    content = report.requirements_path.read_text()
    assert "fastapi" in content

def test_resolve_writes_pyproject_dependencies(tmp_path):
    (tmp_path / "main.py").write_text("import fastapi\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='app'\ndependencies=[]\n")
    dr = DependencyResolver()
    dr.resolve(tmp_path, app_type="fastapi")
    content = (tmp_path / "pyproject.toml").read_text()
    assert "fastapi" in content

def test_resolve_maps_cv2_to_opencv_python(tmp_path):
    (tmp_path / "main.py").write_text("import cv2\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert "opencv-python" in report.packages_found

def test_resolve_unknown_packages_listed(tmp_path):
    (tmp_path / "main.py").write_text("import totally_unknown_pkg_xyz\n")
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert "totally_unknown_pkg_xyz" in report.unknown_packages

def test_resolve_report_has_expected_fields(tmp_path):
    dr = DependencyResolver()
    report = dr.resolve(tmp_path, app_type="generic")
    assert isinstance(report, ResolveReport)
    assert isinstance(report.packages_found, list)
    assert isinstance(report.unknown_packages, list)
```

**Step 2: Run to verify failures**

```
pytest tests/test_dep_resolver.py -v
```

**Step 3: Implement `orchestrator/dep_resolver.py`**

```python
"""
DependencyResolver — scans generated Python files for third-party imports,
maps them to PyPI package names, writes requirements.txt + pyproject.toml.
"""
from __future__ import annotations
import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("orchestrator.dep_resolver")

# import name → PyPI package name
_IMPORT_TO_PYPI: dict[str, str] = {
    "cv2":          "opencv-python",
    "PIL":          "Pillow",
    "sklearn":      "scikit-learn",
    "bs4":          "beautifulsoup4",
    "yaml":         "pyyaml",
    "dotenv":       "python-dotenv",
    "jose":         "python-jose",
    "passlib":      "passlib",
    "aiofiles":     "aiofiles",
    "pydantic":     "pydantic",
    "fastapi":      "fastapi",
    "uvicorn":      "uvicorn",
    "sqlalchemy":   "SQLAlchemy",
    "alembic":      "alembic",
    "httpx":        "httpx",
    "click":        "click",
    "rich":         "rich",
    "typer":        "typer",
    "pytest":       "pytest",
}

_STDLIB = set(sys.stdlib_module_names)  # Python 3.10+


@dataclass
class ResolveReport:
    packages_found:     list[str] = field(default_factory=list)
    unknown_packages:   list[str] = field(default_factory=list)
    requirements_path:  Path = field(default_factory=Path)
    pyproject_updated:  bool = False


class DependencyResolver:
    def resolve(self, output_dir: Path, app_type: str) -> ResolveReport:
        # Find all local package names (directories with __init__.py)
        local_pkgs = {
            d.name for d in output_dir.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
        }

        # Collect raw import names from all .py files
        raw_imports: set[str] = set()
        for py_file in output_dir.rglob("*.py"):
            raw_imports |= _extract_imports(py_file)

        # Classify: stdlib, local, third-party
        third_party: list[str] = []
        unknown: list[str] = []
        for name in sorted(raw_imports):
            root = name.split(".")[0]
            if root in _STDLIB or root in local_pkgs or root == "":
                continue
            pypi_name = _IMPORT_TO_PYPI.get(root, None)
            if pypi_name:
                if pypi_name not in third_party:
                    third_party.append(pypi_name)
            else:
                unknown.append(root)

        # Write requirements.txt
        req_path = output_dir / "requirements.txt"
        req_path.write_text("\n".join(third_party) + "\n", encoding="utf-8")

        # Update pyproject.toml dependencies if it exists
        pyproject_updated = False
        pyproject = output_dir / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text(encoding="utf-8")
            deps_line = "dependencies = [" + ", ".join(f'"{p}"' for p in third_party) + "]"
            if "dependencies = []" in text:
                text = text.replace("dependencies = []", deps_line)
            elif "dependencies=" in text:
                text = text.replace("dependencies=[]", deps_line)
            else:
                text += f"\n{deps_line}\n"
            pyproject.write_text(text, encoding="utf-8")
            pyproject_updated = True

        if unknown:
            logger.warning("DependencyResolver: unknown packages (add to _IMPORT_TO_PYPI): %s", unknown)

        return ResolveReport(
            packages_found=third_party,
            unknown_packages=unknown,
            requirements_path=req_path,
            pyproject_updated=pyproject_updated,
        )


def _extract_imports(py_file: Path) -> set[str]:
    """Parse a .py file and return all top-level import root names."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute import only
                names.add(node.module.split(".")[0])
    return names
```

**Step 4: Run tests**

```
pytest tests/test_dep_resolver.py -v
```
Expected: ALL 9 PASS

**Step 5: Commit**

```bash
git add orchestrator/dep_resolver.py tests/test_dep_resolver.py
git commit -m "feat: DependencyResolver — scan imports, write requirements.txt + pyproject.toml"
```

---

## Task 6: `AppVerifier` (local + Docker, all mocked in tests)

**Files:**
- Create: `orchestrator/app_verifier.py`
- Create: `tests/test_app_verifier.py`

**Step 1: Write failing tests**

```python
# tests/test_app_verifier.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from orchestrator.app_verifier import AppVerifier, LocalVerifyResult, DockerVerifyResult
from orchestrator.app_detector import AppProfile

def _profile(app_type="generic", run_command="python main.py", requires_docker=False):
    return AppProfile(app_type=app_type, tech_stack=["python"],
                      entry_point="main.py", test_command="pytest",
                      run_command=run_command, requires_docker=requires_docker,
                      detected_from="auto")

# ── Local verification ────────────────────────────────────────────

def test_local_verify_install_ok_when_subprocess_returns_0(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        verifier = AppVerifier()
        result = verifier.verify_local(tmp_path, _profile())
        assert result.install_ok is True

def test_local_verify_install_fails_when_subprocess_returns_1(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr=b"error")
        verifier = AppVerifier()
        result = verifier.verify_local(tmp_path, _profile())
        assert result.install_ok is False

def test_local_verify_tests_pass_when_pytest_returns_0(tmp_path):
    (tmp_path / "tests").mkdir()
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        verifier = AppVerifier()
        result = verifier.verify_local(tmp_path, _profile())
        assert result.tests_pass is True

def test_local_verify_tests_fail_when_pytest_returns_1(tmp_path):
    (tmp_path / "tests").mkdir()
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        # First call (install) ok, second (pytest) fails
        mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=1, stderr=b"")]
        verifier = AppVerifier()
        result = verifier.verify_local(tmp_path, _profile())
        assert result.tests_pass is False

def test_local_verify_app_starts_check(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("orchestrator.app_verifier.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = None  # still alive after 3s
            mock_popen.return_value = proc
            verifier = AppVerifier()
            result = verifier.verify_local(tmp_path, _profile())
            assert result.app_starts is True
            proc.terminate.assert_called_once()

def test_local_verify_app_crashed_immediately(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("orchestrator.app_verifier.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = 1  # crashed
            mock_popen.return_value = proc
            verifier = AppVerifier()
            result = verifier.verify_local(tmp_path, _profile())
            assert result.app_starts is False

def test_local_verify_returns_local_verify_result(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("orchestrator.app_verifier.subprocess.Popen", return_value=MagicMock(poll=lambda: None)):
            verifier = AppVerifier()
            result = verifier.verify_local(tmp_path, _profile())
            assert isinstance(result, LocalVerifyResult)

# ── Docker verification ───────────────────────────────────────────

def test_docker_verify_returns_none_when_docker_not_available(tmp_path):
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)  # docker not found
        verifier = AppVerifier()
        result = verifier.verify_docker(tmp_path, _profile(), project_id="p1")
        assert result is None

def test_docker_verify_generates_dockerfile(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\n")
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        verifier = AppVerifier()
        verifier.verify_docker(tmp_path, _profile(app_type="fastapi",
                                                   run_command="uvicorn src.main:app --host 0.0.0.0 --port 8000"),
                                project_id="test_proj")
        assert (tmp_path / "Dockerfile").exists()

def test_docker_verify_result_fields(tmp_path):
    (tmp_path / "requirements.txt").write_text("")
    with patch("orchestrator.app_verifier.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=b"abc123")
        verifier = AppVerifier()
        result = verifier.verify_docker(tmp_path, _profile(), project_id="p1")
        if result is not None:
            assert isinstance(result, DockerVerifyResult)
            assert hasattr(result, "build_ok")
            assert hasattr(result, "run_ok")
```

**Step 2: Run to verify failures**

```
pytest tests/test_app_verifier.py -v
```

**Step 3: Implement `orchestrator/app_verifier.py`**

```python
"""
AppVerifier — runs local and optional Docker verification on a generated app.
All real subprocess calls happen here; tests mock at 'orchestrator.app_verifier.subprocess'.
"""
from __future__ import annotations
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from orchestrator.app_detector import AppProfile

logger = logging.getLogger("orchestrator.app_verifier")

_DOCKERFILE_TEMPLATE = """\
FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
CMD [{run_cmd_args}]
"""


@dataclass
class LocalVerifyResult:
    install_ok:  bool = False
    tests_pass:  bool = False
    app_starts:  bool = False
    health_ok:   bool = False
    errors:      list[str] = field(default_factory=list)


@dataclass
class DockerVerifyResult:
    dockerfile_generated: bool = False
    build_ok:             bool = False
    run_ok:               bool = False
    health_ok:            bool = False
    errors:               list[str] = field(default_factory=list)


class AppVerifier:
    def verify_local(self, output_dir: Path, profile: AppProfile,
                     startup_wait: float = 3.0) -> LocalVerifyResult:
        result = LocalVerifyResult()

        # 1. Install dependencies
        install = subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=output_dir, capture_output=True,
        )
        result.install_ok = (install.returncode == 0)
        if not result.install_ok:
            result.errors.append(f"pip install failed: {install.stderr.decode(errors='replace')[:200]}")

        # 2. Run pytest
        tests_dir = output_dir / "tests"
        pytest_cmd = profile.test_command.split()
        pt = subprocess.run(pytest_cmd, cwd=output_dir, capture_output=True)
        result.tests_pass = (pt.returncode == 0)
        if not result.tests_pass:
            result.errors.append(f"pytest failed (exit {pt.returncode})")

        # 3. Start app and check it stays alive
        proc = None
        try:
            cmd = profile.run_command.split()
            proc = subprocess.Popen(cmd, cwd=output_dir,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(startup_wait)
            alive = proc.poll() is None
            result.app_starts = alive
            if not alive:
                result.errors.append(f"App exited immediately (run_command: {profile.run_command})")
        except Exception as exc:
            result.errors.append(f"Could not start app: {exc}")
        finally:
            if proc is not None:
                proc.terminate()

        return result

    def verify_docker(self, output_dir: Path, profile: AppProfile,
                      project_id: str) -> Optional[DockerVerifyResult]:
        # Check docker is available
        check = subprocess.run(["docker", "info"], capture_output=True)
        if check.returncode != 0:
            logger.info("Docker not available — skipping Docker verification")
            return None

        result = DockerVerifyResult()
        image_tag = f"orchestrator-app-{project_id[:8].lower()}"

        # 1. Generate Dockerfile
        req = output_dir / "requirements.txt"
        run_parts = profile.run_command.split()
        run_cmd_args = ", ".join(f'"{p}"' for p in run_parts)
        dockerfile = _DOCKERFILE_TEMPLATE.format(run_cmd_args=run_cmd_args)
        (output_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        result.dockerfile_generated = True

        # 2. Build
        build = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=output_dir, capture_output=True,
        )
        result.build_ok = (build.returncode == 0)
        if not result.build_ok:
            result.errors.append(f"docker build failed: {build.stderr.decode(errors='replace')[:200]}")
            return result

        # 3. Run
        run = subprocess.run(
            ["docker", "run", "-d", "--rm", image_tag],
            capture_output=True,
        )
        result.run_ok = (run.returncode == 0)
        container_id = run.stdout.decode().strip()

        if result.run_ok and container_id:
            # 4. Cleanup
            subprocess.run(["docker", "stop", container_id], capture_output=True)

        return result
```

**Step 4: Run tests**

```
pytest tests/test_app_verifier.py -v
```
Expected: ALL 11 PASS

**Step 5: Commit**

```bash
git add orchestrator/app_verifier.py tests/test_app_verifier.py
git commit -m "feat: AppVerifier — local pytest + startup check + Docker build/run (all mocked in tests)"
```

---

## Task 7: `AppBuilder` — top-level pipeline

**Files:**
- Create: `orchestrator/app_builder.py`
- Modify: `orchestrator/__init__.py` (add exports)
- Create: `tests/test_app_builder.py`

**Step 1: Write failing tests**

```python
# tests/test_app_builder.py
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.app_builder import AppBuilder, AppBuildResult
from orchestrator.app_detector import AppProfile
from orchestrator.models import Budget, ProjectState, ProjectStatus

def run(coro):
    return asyncio.run(coro)

def _mock_profile():
    return AppProfile(app_type="script", tech_stack=["python"],
                      entry_point="main.py", test_command="pytest",
                      run_command="python main.py", requires_docker=False,
                      detected_from="fallback")

def _mock_state():
    return ProjectState(project_description="test", success_criteria="pass",
                        budget=Budget(max_usd=5.0), status=ProjectStatus.SUCCESS)

def test_app_builder_constructs_with_budget():
    builder = AppBuilder(budget=Budget(max_usd=10.0))
    assert builder is not None

def test_build_returns_app_build_result(tmp_path):
    async def _run():
        builder = AppBuilder(budget=Budget(max_usd=5.0))
        with patch.object(builder._detector, "detect", new=AsyncMock(return_value=_mock_profile())):
            with patch.object(builder._orchestrator, "run_job", new=AsyncMock(return_value=_mock_state())):
                with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
                    with patch("orchestrator.app_verifier.subprocess.Popen", return_value=MagicMock(poll=lambda: None)):
                        return await builder.build("make a script", "it runs", output_dir=tmp_path)
    result = run(_run())
    assert isinstance(result, AppBuildResult)

def test_build_result_has_all_fields(tmp_path):
    async def _run():
        builder = AppBuilder(budget=Budget(max_usd=5.0))
        with patch.object(builder._detector, "detect", new=AsyncMock(return_value=_mock_profile())):
            with patch.object(builder._orchestrator, "run_job", new=AsyncMock(return_value=_mock_state())):
                with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
                    with patch("orchestrator.app_verifier.subprocess.Popen", return_value=MagicMock(poll=lambda: None)):
                        return await builder.build("make a script", "it runs", output_dir=tmp_path)
    result = run(_run())
    assert hasattr(result, "profile")
    assert hasattr(result, "assembly")
    assert hasattr(result, "dependencies")
    assert hasattr(result, "local_verify")
    assert hasattr(result, "docker_verify")
    assert hasattr(result, "output_dir")
    assert hasattr(result, "success")

def test_build_success_false_when_tests_fail(tmp_path):
    async def _run():
        builder = AppBuilder(budget=Budget(max_usd=5.0))
        with patch.object(builder._detector, "detect", new=AsyncMock(return_value=_mock_profile())):
            with patch.object(builder._orchestrator, "run_job", new=AsyncMock(return_value=_mock_state())):
                with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=1, stderr=b"")):
                    with patch("orchestrator.app_verifier.subprocess.Popen", return_value=MagicMock(poll=lambda: 1)):
                        return await builder.build("make a script", "it runs", output_dir=tmp_path)
    result = run(_run())
    assert result.success is False

def test_build_yaml_override_used_when_app_type_provided(tmp_path):
    async def _run():
        builder = AppBuilder(budget=Budget(max_usd=5.0))
        with patch.object(builder._orchestrator, "run_job", new=AsyncMock(return_value=_mock_state())):
            with patch("orchestrator.app_verifier.subprocess.run", return_value=MagicMock(returncode=0)):
                with patch("orchestrator.app_verifier.subprocess.Popen", return_value=MagicMock(poll=lambda: None)):
                    return await builder.build("make a script", "it runs",
                                               output_dir=tmp_path, app_type="cli")
    result = run(_run())
    assert result.profile.app_type == "cli"
    assert result.profile.detected_from == "yaml_override"
```

**Step 2: Run to verify failures**

```
pytest tests/test_app_builder.py -v
```

**Step 3: Implement `orchestrator/app_builder.py`**

```python
"""
AppBuilder — top-level pipeline: description → verified running app.

Pipeline:
  AppDetector → ScaffoldEngine → Orchestrator → AppAssembler
  → DependencyResolver → AppVerifier (local + optional Docker)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import asyncio

from orchestrator.models import Budget, ProjectState
from orchestrator.policy import JobSpec
from orchestrator.engine import Orchestrator
from orchestrator.app_detector import AppDetector, AppProfile
from orchestrator.scaffold import ScaffoldEngine
from orchestrator.app_assembler import AppAssembler, AssemblyReport
from orchestrator.dep_resolver import DependencyResolver, ResolveReport
from orchestrator.app_verifier import AppVerifier, LocalVerifyResult, DockerVerifyResult

logger = logging.getLogger("orchestrator.app_builder")


@dataclass
class AppBuildResult:
    project_state:  ProjectState
    profile:        AppProfile
    assembly:       AssemblyReport
    dependencies:   ResolveReport
    local_verify:   LocalVerifyResult
    docker_verify:  Optional[DockerVerifyResult]
    output_dir:     Path
    success:        bool


class AppBuilder:
    def __init__(self, budget: Optional[Budget] = None) -> None:
        self._budget = budget or Budget()
        self._orchestrator = Orchestrator(budget=self._budget)
        self._detector = AppDetector(api_client=self._orchestrator._client)
        self._scaffold = ScaffoldEngine()
        self._assembler = AppAssembler()
        self._dep_resolver = DependencyResolver()
        self._verifier = AppVerifier()

    async def build(
        self,
        description:      str,
        success_criteria: str,
        output_dir:       Path | str = "./app_output",
        app_type:         Optional[str] = None,
        docker:           bool = False,
    ) -> AppBuildResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Detect app type
        if app_type:
            profile = self._detector.detect_from_yaml(
                {"app_type": app_type, "project_description": description}
            ) or await self._detector.detect(description)
        else:
            profile = await self._detector.detect(description)
        logger.info("App type detected: %s (from: %s)", profile.app_type, profile.detected_from)

        # 2. Scaffold
        scaffold = self._scaffold.scaffold(profile, output_dir)
        logger.info("Scaffold created: %d files", len(scaffold))

        # 3. Run orchestrator
        spec = JobSpec(
            project_description=f"[{profile.app_type.upper()} APP] {description}\n\n"
                                 f"Write complete file contents. Each task output is one file.\n"
                                 f"Use absolute imports. Entry point: {profile.entry_point}.",
            success_criteria=success_criteria,
            budget=self._budget,
        )
        project_state = await self._orchestrator.run_job(spec)
        logger.info("Orchestrator finished: %s", project_state.status.value)

        # 4. Assemble
        assembly = self._assembler.assemble(
            project_state.results, project_state.tasks, scaffold, output_dir
        )
        logger.info("Assembly: %d files written, %d skipped",
                    len(assembly.files_written), len(assembly.files_skipped))

        # 5. Resolve dependencies
        deps = self._dep_resolver.resolve(output_dir, profile.app_type)
        logger.info("Dependencies: %s", deps.packages_found)

        # 6. Local verification
        local = self._verifier.verify_local(output_dir, profile)
        logger.info("Local verify: install=%s tests=%s starts=%s",
                    local.install_ok, local.tests_pass, local.app_starts)

        # 7. Docker (optional)
        docker_result: Optional[DockerVerifyResult] = None
        if docker or profile.requires_docker:
            docker_result = self._verifier.verify_docker(
                output_dir, profile,
                project_id=getattr(project_state, "project_id", "app"),
            )
            if docker_result:
                logger.info("Docker verify: build=%s run=%s", docker_result.build_ok, docker_result.run_ok)

        success = (
            local.tests_pass and local.app_starts
            and (docker_result is None or (docker_result.build_ok and docker_result.run_ok))
        )

        return AppBuildResult(
            project_state=project_state,
            profile=profile,
            assembly=assembly,
            dependencies=deps,
            local_verify=local,
            docker_verify=docker_result,
            output_dir=output_dir,
            success=success,
        )
```

**Step 4: Add exports to `orchestrator/__init__.py`**

Add to the imports block:
```python
from .app_builder  import AppBuilder, AppBuildResult
from .app_detector import AppDetector, AppProfile
```

Add to `__all__`:
```python
"AppBuilder", "AppBuildResult", "AppDetector", "AppProfile",
```

**Step 5: Run tests**

```
pytest tests/test_app_builder.py tests/test_app_builder_models.py -v
```
Expected: ALL PASS

**Step 6: Run full suite**

```
pytest tests/ --ignore=tests/stress_test.py -q
```
Expected: ALL existing 319 + new tests PASS

**Step 7: Commit**

```bash
git add orchestrator/app_builder.py orchestrator/__init__.py tests/test_app_builder.py
git commit -m "feat: AppBuilder — top-level pipeline, description → verified running app"
```

---

## Task 8: CLI `build` subcommand

**Files:**
- Modify: `orchestrator/cli.py` (add `build` subcommand)
- Modify: `orchestrator/project_file.py` (add `app_type`, `docker`, `output_dir` fields)
- Create: `tests/test_app_builder_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_app_builder_cli.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from orchestrator.cli import build_command   # new function to add
from orchestrator.app_builder import AppBuildResult
from orchestrator.app_verifier import LocalVerifyResult
from orchestrator.app_assembler import AssemblyReport
from orchestrator.dep_resolver import ResolveReport
from orchestrator.app_detector import AppProfile
from orchestrator.models import Budget, ProjectState, ProjectStatus

def _mock_result(tmp_path):
    return AppBuildResult(
        project_state=ProjectState(project_description="t", success_criteria="p",
                                   budget=Budget(max_usd=1.0), status=ProjectStatus.SUCCESS),
        profile=AppProfile("script",["python"],"main.py","pytest","python main.py",False,"auto"),
        assembly=AssemblyReport(files_written=["main.py"]),
        dependencies=ResolveReport(),
        local_verify=LocalVerifyResult(install_ok=True, tests_pass=True, app_starts=True),
        docker_verify=None,
        output_dir=tmp_path,
        success=True,
    )

def test_build_command_is_callable():
    assert callable(build_command)

def test_project_file_accepts_app_type_field():
    from orchestrator.project_file import load_project_file
    import json, tempfile, os
    data = {
        "project_description": "test",
        "success_criteria": "pass",
        "app_type": "fastapi",
        "docker": True,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        spec = load_project_file(path)
        assert getattr(spec, "app_type", None) == "fastapi"
        assert getattr(spec, "docker", False) is True
    finally:
        os.unlink(path)
```

**Step 2: Add `app_type` / `docker` / `output_dir` to `JobSpec`**

In `orchestrator/policy.py`, add to `JobSpec` dataclass (after `budget`):

```python
    # App Builder fields
    app_type:   Optional[str]  = None   # e.g. "fastapi", "cli"
    docker:     bool           = False
    output_dir: Optional[str]  = None
```

**Step 3: Update `orchestrator/project_file.py`** to read and pass `app_type`, `docker`, `output_dir` from the YAML/JSON dict into `JobSpec`.

**Step 4: Add `build_command` to `orchestrator/cli.py`**

```python
def build_command(args):
    """Handle `python -m orchestrator build` subcommand."""
    import asyncio
    from orchestrator.app_builder import AppBuilder
    from orchestrator.models import Budget

    builder = AppBuilder(budget=Budget(
        max_usd=args.budget,
        max_time_seconds=args.time,
    ))
    result = asyncio.run(builder.build(
        description=args.description or args.project,
        success_criteria=args.criteria,
        output_dir=args.output_dir or "./app_output",
        app_type=getattr(args, "app_type", None),
        docker=getattr(args, "docker", False),
    ))
    print(f"\n{'✅' if result.success else '❌'} App build {'succeeded' if result.success else 'FAILED'}")
    print(f"  Output: {result.output_dir}")
    print(f"  App type: {result.profile.app_type}")
    print(f"  Tests: {'PASS' if result.local_verify.tests_pass else 'FAIL'}")
    print(f"  Starts: {'YES' if result.local_verify.app_starts else 'NO'}")
    if result.docker_verify:
        print(f"  Docker: {'OK' if result.docker_verify.build_ok else 'FAIL'}")
```

Register `build` as a subparser in the main argparse setup.

**Step 5: Run tests**

```
pytest tests/test_app_builder_cli.py -v
pytest tests/ --ignore=tests/stress_test.py -q
```
Expected: ALL PASS

**Step 6: Final commit**

```bash
git add orchestrator/cli.py orchestrator/project_file.py orchestrator/policy.py tests/test_app_builder_cli.py
git commit -m "feat: CLI build subcommand + YAML app_type/docker/output_dir fields"
```

---

## Final Verification

```bash
# Full test suite — must be 319+ passing, 0 failing
pytest tests/ --ignore=tests/stress_test.py -q

# Smoke test: imports work
python -c "from orchestrator import AppBuilder, AppBuildResult, AppDetector, AppProfile; print('OK')"

# CLI help visible
python -m orchestrator build --help
```

---

## Summary of Changes

| Component | Files | Tests |
|-----------|-------|-------|
| Task model extension | `models.py` | 3 |
| AppDetector | `app_detector.py` | 8 |
| ScaffoldEngine | `scaffold/` (6 files) | 10 |
| AppAssembler | `app_assembler.py` | 8 |
| DependencyResolver | `dep_resolver.py` | 9 |
| AppVerifier | `app_verifier.py` | 11 |
| AppBuilder | `app_builder.py`, `__init__.py` | 5 |
| CLI | `cli.py`, `project_file.py`, `policy.py` | 2 |
| **Total** | **~15 files** | **~56 new tests** |
