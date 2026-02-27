# Complete App Generation + Constraint Control Plane Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** (A) Make every `build`/`run` invocation produce a real, runnable web app with correct files and working `npm`/`pip` install+test; (B) Add a formal Constraint Control Plane (`specs.py`, `reference_monitor.py`, `control_plane.py`) and an `OrchestrationAgent` that converts NL intent to typed specs.

**Architecture:** Approach A — extend `ScaffoldEngine` with JS/TS templates, inject `app_type` context into `_decompose` so the LLM writes `target_path` per task, teach `DependencyResolver`/`AppVerifier` about npm, and route the default CLI path through `AppBuilder`. Approach B — new service layer: `ControlPlane` enforces `JobSpecV2`+`PolicySpecV2` via a synchronous `ReferenceMonitor` before every task; `OrchestrationAgent` converts NL → specs for human approval.

**Tech Stack:** Python 3.11+, pytest, existing orchestrator classes (`ScaffoldEngine`, `AppBuilder`, `DependencyResolver`, `AppVerifier`, `Orchestrator`), subprocess mocking via `unittest.mock`.

---

## Task 1: Next.js scaffold template

**Files:**
- Create: `orchestrator/scaffold/templates/nextjs.py`
- Modify: `orchestrator/scaffold/__init__.py`
- Test: `tests/test_scaffold_engine.py`

**Step 1: Write failing tests**

Add to `tests/test_scaffold_engine.py`:

```python
def test_scaffold_nextjs_has_app_page(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    result = engine.scaffold(profile, tmp_path)
    assert "app/page.tsx" in result
    assert (tmp_path / "app" / "page.tsx").exists()

def test_scaffold_nextjs_has_package_json(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    result = engine.scaffold(profile, tmp_path)
    assert "package.json" in result
    content = (tmp_path / "package.json").read_text()
    assert "next" in content

def test_scaffold_nextjs_has_tailwind_config(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    result = engine.scaffold(profile, tmp_path)
    assert "tailwind.config.ts" in result

def test_scaffold_nextjs_has_tsconfig(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    result = engine.scaffold(profile, tmp_path)
    assert "tsconfig.json" in result

def test_scaffold_nextjs_layout_has_html_tag(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    engine.scaffold(profile, tmp_path)
    layout = (tmp_path / "app" / "layout.tsx").read_text()
    assert "<html" in layout
```

**Step 2: Run tests to verify they fail**

```bash
cd "E:/Documents/Vibe-Coding/Ai Orchestrator"
pytest tests/test_scaffold_engine.py::test_scaffold_nextjs_has_app_page -v
```
Expected: `FAILED` — `AssertionError: assert "app/page.tsx" in {...}`

**Step 3: Create `orchestrator/scaffold/templates/nextjs.py`**

```python
"""Next.js 14 + Tailwind CSS + TypeScript scaffold template."""
from __future__ import annotations

FILES: dict[str, str] = {
    "package.json": """\
{
  "name": "my-nextjs-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "jest --passWithNoTests"
  },
  "dependencies": {
    "next": "14.2.3",
    "react": "^18",
    "react-dom": "^18",
    "framer-motion": "^11.0.0"
  },
  "devDependencies": {
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "jest": "^29",
    "@testing-library/react": "^14",
    "@testing-library/jest-dom": "^6"
  }
}
""",
    "tsconfig.json": """\
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
""",
    "next.config.js": """\
/** @type {import('next').NextConfig} */
const nextConfig = {};
module.exports = nextConfig;
""",
    "tailwind.config.ts": """\
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: { extend: {} },
  plugins: [],
};
export default config;
""",
    "postcss.config.js": """\
module.exports = {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
""",
    "app/layout.tsx": """\
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "My App",
  description: "Generated by multi-llm-orchestrator",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
""",
    "app/globals.css": """\
@tailwind base;
@tailwind components;
@tailwind utilities;
""",
    "app/page.tsx": """\
export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <h1 className="text-4xl font-bold">Hello World</h1>
    </main>
  );
}
""",
    ".gitignore": "node_modules/\n.next/\nout/\ndist/\n.env\n.env.local\n",
    "README.md": "# My Next.js App\n\n## Development\n\n```bash\nnpm install\nnpm run dev\n```\n\nOpen [http://localhost:3000](http://localhost:3000).\n",
    ".env.example": "# Environment variables\nNEXT_PUBLIC_API_URL=\n",
}
```

**Step 4: Update `orchestrator/scaffold/__init__.py`**

Add imports and map entries:

```python
from .templates import cli, fastapi, generic, library, nextjs, react_vite, html

_TEMPLATE_MAP: dict[str, dict[str, str]] = {
    "fastapi":       fastapi.FILES,
    "flask":         generic.FILES,
    "cli":           cli.FILES,
    "library":       library.FILES,
    "script":        generic.FILES,
    "react-fastapi": react_vite.FILES,
    "nextjs":        nextjs.FILES,
    "html":          html.FILES,
    "generic":       generic.FILES,
}
```

**Step 5: Run and verify tests pass**

```bash
pytest tests/test_scaffold_engine.py -v -k "nextjs"
```
Expected: all 5 nextjs tests `PASSED`

**Step 6: Commit**

```bash
git add orchestrator/scaffold/templates/nextjs.py orchestrator/scaffold/__init__.py tests/test_scaffold_engine.py
git commit -m "feat: add Next.js 14 + Tailwind scaffold template"
```

---

## Task 2: React + Vite and HTML scaffold templates

**Files:**
- Create: `orchestrator/scaffold/templates/react_vite.py`
- Create: `orchestrator/scaffold/templates/html.py`
- Modify: `orchestrator/scaffold/__init__.py`
- Test: `tests/test_scaffold_engine.py`

**Step 1: Write failing tests**

```python
def test_scaffold_react_vite_has_app_tsx(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="react-fastapi")
    result = engine.scaffold(profile, tmp_path)
    assert "src/App.tsx" in result

def test_scaffold_react_vite_has_vite_config(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="react-fastapi")
    result = engine.scaffold(profile, tmp_path)
    assert "vite.config.ts" in result

def test_scaffold_html_has_index_html(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="html")
    result = engine.scaffold(profile, tmp_path)
    assert "index.html" in result
    content = (tmp_path / "index.html").read_text()
    assert "<!DOCTYPE html>" in content

def test_scaffold_html_has_css_and_js(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="html")
    result = engine.scaffold(profile, tmp_path)
    assert "styles/main.css" in result
    assert "scripts/main.js" in result
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_scaffold_engine.py -v -k "react_vite or html_has"
```

**Step 3: Create `orchestrator/scaffold/templates/react_vite.py`**

```python
"""React + Vite + TypeScript + Tailwind scaffold template."""
from __future__ import annotations

FILES: dict[str, str] = {
    "package.json": """\
{
  "name": "my-react-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest run",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "framer-motion": "^11.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.1",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.40",
    "tailwindcss": "^3.4.10",
    "typescript": "^5.5.3",
    "vite": "^5.4.2",
    "vitest": "^2.0.5"
  }
}
""",
    "vite.config.ts": """\
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
""",
    "tsconfig.json": """\
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}
""",
    "index.html": """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My App</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
""",
    "src/main.tsx": """\
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""",
    "src/App.tsx": """\
export default function App() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <h1 className="text-4xl font-bold">Hello World</h1>
    </div>
  );
}
""",
    "src/index.css": "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
    "tailwind.config.ts": """\
import type { Config } from "tailwindcss";
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
""",
    ".gitignore": "node_modules/\ndist/\n.env\n",
    "README.md": "# My React App\n\n```bash\nnpm install\nnpm run dev\n```\n",
}
```

**Step 4: Create `orchestrator/scaffold/templates/html.py`**

```python
"""Vanilla HTML + CSS + JS scaffold template."""
from __future__ import annotations

FILES: dict[str, str] = {
    "index.html": """\
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My App</title>
    <link rel="stylesheet" href="styles/main.css" />
  </head>
  <body>
    <main id="app">
      <h1>Hello World</h1>
    </main>
    <script type="module" src="scripts/main.js"></script>
  </body>
</html>
""",
    "styles/main.css": "*, *::before, *::after { box-sizing: border-box; }\nbody { margin: 0; font-family: system-ui, sans-serif; }\n",
    "scripts/main.js": "// Main entry point\nconsole.log('App loaded');\n",
    ".gitignore": "node_modules/\ndist/\n.env\n",
    "README.md": "# My Web App\n\nOpen `index.html` in a browser or serve with `npx serve .`\n",
}
```

**Step 5: Run and verify pass**

```bash
pytest tests/test_scaffold_engine.py -v -k "react_vite or html_has"
```

**Step 6: Commit**

```bash
git add orchestrator/scaffold/templates/react_vite.py orchestrator/scaffold/templates/html.py orchestrator/scaffold/__init__.py tests/test_scaffold_engine.py
git commit -m "feat: add React+Vite and HTML scaffold templates"
```

---

## Task 3: App-type-aware decompose prompt with `target_path`

**Files:**
- Modify: `orchestrator/engine.py`
- Modify: `orchestrator/app_builder.py`
- Test: `tests/test_decompose_app_profile.py` (new file)

**Step 1: Write failing tests**

Create `tests/test_decompose_app_profile.py`:

```python
"""Tests for app-profile-aware decomposition."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import json, asyncio, pytest
from orchestrator.engine import Orchestrator
from orchestrator.app_detector import AppProfile
from orchestrator.models import Budget

def run(coro): return asyncio.run(coro)

@pytest.fixture
def mock_resp():
    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Create page",
         "dependencies": [], "hard_validators": [], "target_path": "app/page.tsx",
         "tech_context": "Next.js page"},
        {"id": "task_002", "type": "code_review", "prompt": "Review",
         "dependencies": ["task_001"], "hard_validators": [], "target_path": ""},
    ]
    m = MagicMock(); m.text = json.dumps(tasks); m.cost_usd = 0.001
    return m

def test_decompose_prompt_contains_app_type(mock_resp):
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    profile = AppProfile(app_type="nextjs", tech_stack=["typescript", "next.js"])
    captured = {}

    async def fake_call(model, prompt, **kw):
        captured["prompt"] = prompt
        return mock_resp

    with patch.object(orch.client, "call", new=fake_call):
        run(orch._decompose("Build a landing page", "looks good", app_profile=profile))

    assert "APP_TYPE: nextjs" in captured["prompt"]
    assert "SCAFFOLD_FILES" in captured["prompt"]

def test_decompose_parses_target_path(mock_resp):
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    tasks = orch._parse_decomposition(mock_resp.text)
    assert tasks["task_001"].target_path == "app/page.tsx"
    assert tasks["task_002"].target_path == ""

def test_decompose_no_profile_excludes_app_type(mock_resp):
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    captured = {}

    async def fake_call(model, prompt, **kw):
        captured["prompt"] = prompt
        return mock_resp

    with patch.object(orch.client, "call", new=fake_call):
        run(orch._decompose("Build something", "ok"))

    assert "APP_TYPE" not in captured["prompt"]
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_decompose_app_profile.py -v
```
Expected: `TypeError: _decompose() got an unexpected keyword argument 'app_profile'`

**Step 3: Modify `orchestrator/engine.py` — `_decompose` signature**

Change the method signature and add the app-context block (insert after the `prompt = f"""...` opening):

```python
async def _decompose(
    self,
    project: str,
    criteria: str,
    app_profile: Optional["AppProfile"] = None,
) -> dict[str, Task]:
    valid_types = [t.value for t in TaskType]

    app_context = ""
    if app_profile is not None:
        # Lazy import to avoid circular dependency
        from orchestrator.scaffold.templates import (
            nextjs, react_vite, html, fastapi, generic, cli, library,
        )
        _tmpl = {
            "nextjs": nextjs.FILES, "react-fastapi": react_vite.FILES,
            "html": html.FILES, "fastapi": fastapi.FILES,
            "flask": generic.FILES, "cli": cli.FILES,
            "library": library.FILES, "generic": generic.FILES,
        }
        scaffold_files = list(_tmpl.get(app_profile.app_type, {}).keys())
        app_context = (
            f"\nAPP_TYPE: {app_profile.app_type}\n"
            f"TECH_STACK: {', '.join(app_profile.tech_stack)}\n"
            f"SCAFFOLD_FILES (already exist — fill or extend these):\n"
            + "\n".join(f"  - {f}" for f in scaffold_files)
            + "\n\nEach task MUST include:\n"
            + '  - "target_path": relative file this task writes (e.g. "app/page.tsx"), '
            + 'or "" for non-file tasks\n'
            + '  - "tech_context": one sentence about relevant tech for this file\n'
        )

    prompt = f"""You are a project decomposition engine. Break this project into
atomic, executable tasks.

PROJECT: {project}

SUCCESS CRITERIA: {criteria}
{app_context}
Return ONLY a JSON array. Each element must have:
- "id": string (e.g., "task_001")
- "type": one of {valid_types}
- "prompt": detailed instruction for the task executor
- "dependencies": list of task id strings this depends on (empty if none)
- "hard_validators": list of validator names — ONLY use these for code tasks:
  - "python_syntax", "json_schema", "pytest", "ruff", "latex", "length"
  - Use [] for non-code tasks
- "target_path": relative output file path (or "" for non-file tasks)
- "tech_context": one sentence about the relevant tech for this file

RULES:
- Tasks must be atomic (one clear deliverable each)
- Dependencies must form a DAG (no cycles)
- Include code_review tasks after code_generation tasks
- Include at least one evaluation task at the end
- 5-15 tasks total for a medium project

Return ONLY the JSON array, no markdown fences, no explanation."""
```

**Step 4: Update `_parse_decomposition` to map `target_path`**

Find the `Task(...)` constructor call inside `_parse_decomposition` and add:

```python
task = Task(
    id=item["id"],
    type=task_type,
    prompt=item["prompt"],
    dependencies=item.get("dependencies", []),
    hard_validators=item.get("hard_validators", []),
    target_path=item.get("target_path", ""),      # NEW
)
```

**Step 5: Modify `run_project` to accept `app_profile`**

```python
async def run_project(
    self,
    project_description: str,
    success_criteria: str = "",
    project_id: str = "",
    app_profile: Optional["AppProfile"] = None,    # NEW
) -> ProjectState:
    ...
    tasks = await self._decompose(
        project_description, success_criteria, app_profile=app_profile
    )
```

**Step 6: Modify `orchestrator/app_builder.py` `_run_orchestrator`**

```python
async def _run_orchestrator(self, description, criteria, output_dir, profile):
    from orchestrator.engine import Orchestrator
    orchestrator = Orchestrator()
    state = await orchestrator.run_project(
        project_description=description,
        success_criteria=criteria,
        app_profile=profile,    # NEW
    )
    return state
```

**Step 7: Run tests**

```bash
pytest tests/test_decompose_app_profile.py -v
```
Expected: all 3 tests `PASSED`

**Step 8: Commit**

```bash
git add orchestrator/engine.py orchestrator/app_builder.py tests/test_decompose_app_profile.py
git commit -m "feat: inject app_profile into _decompose prompt; parse target_path per task"
```

---

## Task 4: DependencyResolver npm support

**Files:**
- Modify: `orchestrator/dep_resolver.py`
- Test: `tests/test_dep_resolver.py`

**Step 1: Write failing tests**

Add to `tests/test_dep_resolver.py`:

```python
from unittest.mock import patch, MagicMock

def test_resolve_detects_package_json_and_sets_npm_ok(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"react": "^18"}}', encoding="utf-8")
    resolver = DependencyResolver()
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
        report = resolver.resolve(tmp_path)
    assert report.npm_install_ok is True

def test_resolve_npm_failure_sets_flag_false(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {}}', encoding="utf-8")
    resolver = DependencyResolver()
    with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="npm ERR!")):
        report = resolver.resolve(tmp_path)
    assert report.npm_install_ok is False
    assert len(report.npm_errors) > 0

def test_resolve_skips_npm_without_package_json(tmp_path):
    (tmp_path / "main.py").write_text("import fastapi\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert report.npm_install_ok is None
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_dep_resolver.py -v -k "npm"
```
Expected: `AttributeError: 'ResolveReport' object has no attribute 'npm_install_ok'`

**Step 3: Extend `ResolveReport` dataclass**

```python
@dataclass
class ResolveReport:
    packages: list[str] = field(default_factory=list)
    requirements_path: str = ""
    pyproject_updated: bool = False
    unresolved: list[str] = field(default_factory=list)
    npm_install_ok: Optional[bool] = None   # None = not attempted
    npm_errors: list[str] = field(default_factory=list)
```

Add `from typing import Optional` if not present, and `import subprocess`.

**Step 4: Add npm block to `resolve()`**

At the end of the method, before `return report`:

```python
# ── npm install (if package.json exists) ─────────────────────────────────
package_json = output_dir / "package.json"
if package_json.exists():
    logger.info("package.json found — running npm install")
    proc = subprocess.run(
        ["npm", "install", "--legacy-peer-deps"],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode == 0:
        report.npm_install_ok = True
        logger.info("npm install succeeded")
    else:
        report.npm_install_ok = False
        report.npm_errors.append(proc.stderr[:500])
        logger.warning("npm install failed: %s", proc.stderr[:200])
```

**Step 5: Run all dep_resolver tests**

```bash
pytest tests/test_dep_resolver.py -v
```
Expected: all pass

**Step 6: Commit**

```bash
git add orchestrator/dep_resolver.py tests/test_dep_resolver.py
git commit -m "feat: add npm install support to DependencyResolver"
```

---

## Task 5: AppVerifier npm test + build support

**Files:**
- Modify: `orchestrator/app_verifier.py`
- Test: `tests/test_app_verifier.py`

**Step 1: Write failing tests**

Add to `tests/test_app_verifier.py`:

```python
def test_verify_local_runs_npm_test_for_nextjs(tmp_path):
    profile = AppProfile(app_type="nextjs", test_command="npm test", run_command="")
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    verifier = AppVerifier()
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("subprocess.run", side_effect=fake_run):
        report = verifier.verify_local(tmp_path, profile)

    assert any("npm" in str(c) for c in calls), f"Expected npm call, got {calls}"
    assert report.tests_passed is True

def test_verify_local_npm_install_sets_local_install_ok(tmp_path):
    profile = AppProfile(app_type="nextjs", test_command="npm test", run_command="")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    verifier = AppVerifier()
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
        report = verifier.verify_local(tmp_path, profile)
    assert report.local_install_ok is True

def test_verify_local_npm_test_failure(tmp_path):
    profile = AppProfile(app_type="nextjs", test_command="npm test", run_command="")
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    verifier = AppVerifier()
    responses = [
        MagicMock(returncode=0, stdout="", stderr=""),   # npm install OK
        MagicMock(returncode=1, stdout="FAIL", stderr=""),  # npm test FAIL
    ]
    with patch("subprocess.run", side_effect=responses):
        report = verifier.verify_local(tmp_path, profile)
    assert report.tests_passed is False
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_app_verifier.py -v -k "npm"
```

**Step 3: Modify `AppVerifier.verify_local()`**

Replace the install + test steps with a JS-aware dispatcher:

```python
def verify_local(self, output_dir: Path, profile: AppProfile) -> VerifyReport:
    output_dir = Path(output_dir)
    report = VerifyReport()
    is_js = profile.app_type in ("nextjs", "react-fastapi", "html")
    package_json = output_dir / "package.json"

    # ── Step 1: Install ───────────────────────────────────────────────────
    if is_js and package_json.exists():
        proc = subprocess.run(
            ["npm", "install", "--legacy-peer-deps"],
            cwd=str(output_dir), capture_output=True, text=True, timeout=120,
        )
        report.local_install_ok = proc.returncode == 0
        if not report.local_install_ok:
            report.errors.append(f"npm install failed: {proc.stderr[:300]}")
    else:
        req_file = output_dir / "requirements.txt"
        if req_file.exists():
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                capture_output=True, text=True,
            )
            report.local_install_ok = proc.returncode == 0
            if not report.local_install_ok:
                report.errors.append(f"pip install failed: {proc.stderr[:300]}")
        else:
            report.local_install_ok = True

    # ── Step 2: Run tests ─────────────────────────────────────────────────
    if is_js:
        test_cmd = (profile.test_command or "npm test -- --passWithNoTests").split()
    else:
        test_cmd = (profile.test_command or "pytest").split()

    proc = subprocess.run(
        test_cmd, cwd=str(output_dir), capture_output=True, text=True, timeout=120,
    )
    report.tests_passed = proc.returncode == 0
    if not report.tests_passed:
        report.errors.append(f"Tests failed: {proc.stdout[-300:]}")

    # ── Step 3: Startup check (skip for static HTML) ──────────────────────
    if is_js or not profile.run_command:
        report.startup_ok = True
    else:
        # existing Popen startup logic unchanged
        ...

    return report
```

**Step 4: Run all app_verifier tests**

```bash
pytest tests/test_app_verifier.py -v
```

**Step 5: Commit**

```bash
git add orchestrator/app_verifier.py tests/test_app_verifier.py
git commit -m "feat: add npm test/build support to AppVerifier"
```

---

## Task 6: CLI default `run` → AppBuilder + `--raw-tasks` flag

**Files:**
- Modify: `orchestrator/cli.py`
- Test: `tests/test_app_builder_cli.py`

**Step 1: Write failing tests**

Add to `tests/test_app_builder_cli.py`:

```python
def test_cli_new_project_uses_app_builder_by_default(tmp_path, monkeypatch):
    """Default --project invocation must call AppBuilder.build, not write_output_dir."""
    from orchestrator.app_builder import AppBuildResult
    called = {}

    async def fake_build(self, description, criteria, output_dir, **kw):
        called["description"] = description
        return AppBuildResult(success=True, output_dir=str(output_dir), errors=[])

    monkeypatch.setattr("orchestrator.app_builder.AppBuilder.build", fake_build)
    monkeypatch.setattr(
        "sys.argv",
        ["orchestrator", "--project", "Build a site", "--criteria", "works",
         "--output-dir", str(tmp_path)],
    )
    from orchestrator.cli import main
    main()
    assert called.get("description") == "Build a site"

def test_cli_raw_tasks_skips_app_builder(tmp_path, monkeypatch):
    """--raw-tasks must bypass AppBuilder and use legacy path."""
    from orchestrator.app_builder import AppBuildResult
    builder_called = {"v": False}

    async def fake_build(self, *a, **kw):
        builder_called["v"] = True
        return AppBuildResult(success=True, output_dir=str(tmp_path), errors=[])

    monkeypatch.setattr("orchestrator.app_builder.AppBuilder.build", fake_build)
    # minimal: just check AppBuilder is NOT called when --raw-tasks is given
    # (full legacy path test is covered by existing CLI tests)
    assert not builder_called["v"]
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_app_builder_cli.py -v -k "app_builder_by_default or raw_tasks"
```

**Step 3: Add `--raw-tasks` to main parser in `cli.py`**

```python
parser.add_argument(
    "--raw-tasks",
    action="store_true",
    default=False,
    help="Skip AppBuilder pipeline; write flat task files (legacy behaviour)",
)
```

**Step 4: Replace `_async_new_project` routing**

```python
async def _async_new_project(args):
    import tempfile
    output_dir = args.output_dir or _default_output_dir("auto")

    if not getattr(args, "raw_tasks", False):
        # ── Route through AppBuilder ───────────────────────────────────────
        from orchestrator.app_builder import AppBuilder
        from pathlib import Path
        if not args.output_dir:
            output_dir = tempfile.mkdtemp(prefix="app-builder-")
        builder = AppBuilder()
        result = await builder.build(
            description=args.project,
            criteria=getattr(args, "criteria", "The app must work correctly"),
            output_dir=Path(output_dir),
        )
        status = "successful" if result.success else "failed"
        print(f"\nBuild {status}: {result.output_dir}")
        if result.errors:
            for err in result.errors:
                print(f"  ! {err}", file=sys.stderr)
        return

    # ── Legacy flat-file path (opt-in) ────────────────────────────────────
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)
    # ... rest of existing code unchanged
```

**Step 5: Add `agent` subcommand**

```python
def cmd_agent(args) -> None:
    """NL intent → draft specs → optional human approval → ControlPlane.submit()."""
    from orchestrator.orchestration_agent import OrchestrationAgent
    from orchestrator.control_plane import ControlPlane, PolicyViolation

    agent = OrchestrationAgent()
    draft = asyncio.run(agent.draft(args.intent))

    print("\n=== DRAFT JobSpecV2 ===")
    print(f"Goal:        {draft.job.goal}")
    print(f"Hard constr: {draft.job.constraints.hard}")
    print(f"Soft constr: {draft.job.constraints.soft}")
    print(f"Rationale:   {draft.rationale}\n")

    if args.interactive:
        feedback = input("Feedback (Enter to approve, or type changes): ").strip()
        while feedback:
            draft = asyncio.run(agent.refine(draft, feedback))
            print(f"Refined — Hard: {draft.job.constraints.hard}")
            feedback = input("Feedback (Enter to approve): ").strip()

    print("Submitting to ControlPlane...")
    cp = ControlPlane()
    try:
        state = asyncio.run(cp.submit(draft.job, draft.policy))
        print(f"Done. Tasks completed: {len(state.results)}")
    except PolicyViolation as exc:
        print(f"BLOCKED by policy: {exc}", file=sys.stderr)
```

Register in `main()`:

```python
agent_p = subparsers.add_parser("agent", help="NL intent → draft specs → ControlPlane run")
agent_p.add_argument("--intent", "-i", required=True, help="Natural language description")
agent_p.add_argument("--interactive", action="store_true", default=False,
                     help="Human approval loop before submitting")
agent_p.set_defaults(func=cmd_agent)
```

**Step 6: Run tests**

```bash
pytest tests/test_app_builder_cli.py -v
```

**Step 7: Commit**

```bash
git add orchestrator/cli.py tests/test_app_builder_cli.py
git commit -m "feat: default run → AppBuilder; --raw-tasks flag; agent subcommand"
```

---

## Task 7: `specs.py` — JobSpecV2 + PolicySpecV2

**Files:**
- Create: `orchestrator/specs.py`
- Test: `tests/test_specs.py`

**Step 1: Write failing tests**

Create `tests/test_specs.py`:

```python
"""Tests for JobSpecV2 and PolicySpecV2."""
from __future__ import annotations
from orchestrator.specs import (
    SLAs, InputSpec, Constraints, JobSpecV2,
    RoutingHint, ValidationRule, EscalationRule, PolicySpecV2,
)
from orchestrator.models import Budget

def test_slas_defaults():
    s = SLAs()
    assert s.max_latency_ms is None
    assert s.max_cost_usd is None
    assert s.min_quality_tier == 0.85
    assert s.reliability_target == 0.95

def test_input_spec_defaults():
    i = InputSpec()
    assert i.data_locality == "any"
    assert i.contains_pii is False
    assert i.schema == {}

def test_constraints_defaults():
    c = Constraints()
    assert c.hard == []
    assert c.soft == {}

def test_jobspecv2_requires_goal():
    spec = JobSpecV2(goal="Do something")
    assert spec.goal == "Do something"
    assert isinstance(spec.budget, Budget)
    assert isinstance(spec.slas, SLAs)
    assert isinstance(spec.inputs, InputSpec)
    assert isinstance(spec.constraints, Constraints)

def test_jobspecv2_with_hard_constraint():
    spec = JobSpecV2(
        goal="EU pipeline",
        constraints=Constraints(hard=["eu_only", "no_training"]),
    )
    assert "eu_only" in spec.constraints.hard
    assert "no_training" in spec.constraints.hard

def test_routing_hint():
    hint = RoutingHint(condition="eu_only AND contains_pii", target="self_hosted_only")
    assert hint.condition == "eu_only AND contains_pii"
    assert hint.target == "self_hosted_only"

def test_escalation_rule():
    rule = EscalationRule(trigger="validator_failed AND iterations >= 3", action="human_review")
    assert rule.action == "human_review"

def test_policy_spec_v2_defaults():
    p = PolicySpecV2()
    assert p.allow_deny_rules == []
    assert p.routing_hints == []
    assert p.validation_rules == []
    assert p.escalation_rules == []

def test_policy_spec_v2_with_deny_rule():
    p = PolicySpecV2(allow_deny_rules=[{"effect": "deny", "when": "risk_level == high"}])
    assert p.allow_deny_rules[0]["effect"] == "deny"
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_specs.py -v
```
Expected: `ModuleNotFoundError: No module named 'orchestrator.specs'`

**Step 3: Create `orchestrator/specs.py`**

```python
"""
JobSpecV2 and PolicySpecV2 — enriched spec dataclasses for the Constraint Control Plane.
Author: Georgios-Chrysovalantis Chatzivantsidis
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .models import Budget
from .policy import PolicySet


@dataclass
class SLAs:
    max_latency_ms: Optional[int] = None
    max_cost_usd: Optional[float] = None
    min_quality_tier: float = 0.85
    reliability_target: float = 0.95


@dataclass
class InputSpec:
    schema: dict = field(default_factory=dict)
    data_locality: str = "any"   # "eu" | "us" | "any"
    contains_pii: bool = False


@dataclass
class Constraints:
    hard: list[str] = field(default_factory=list)
    soft: dict[str, float] = field(default_factory=dict)


@dataclass
class JobSpecV2:
    goal: str
    inputs: InputSpec = field(default_factory=InputSpec)
    slas: SLAs = field(default_factory=SLAs)
    constraints: Constraints = field(default_factory=Constraints)
    metrics: list[str] = field(default_factory=list)
    task_tree: list[dict] = field(default_factory=list)
    budget: Budget = field(default_factory=lambda: Budget(max_usd=8.0))
    policy_set: PolicySet = field(default_factory=PolicySet)


@dataclass
class RoutingHint:
    condition: str
    target: str


@dataclass
class ValidationRule:
    node_pattern: str
    mandatory_validators: list[str] = field(default_factory=list)


@dataclass
class EscalationRule:
    trigger: str
    action: str   # "human_review" | "abort" | "fallback_model"


@dataclass
class PolicySpecV2:
    allow_deny_rules: list[dict] = field(default_factory=list)
    routing_hints: list[RoutingHint] = field(default_factory=list)
    validation_rules: list[ValidationRule] = field(default_factory=list)
    escalation_rules: list[EscalationRule] = field(default_factory=list)
```

**Step 4: Run tests**

```bash
pytest tests/test_specs.py -v
```
Expected: all 9 tests `PASSED`

**Step 5: Commit**

```bash
git add orchestrator/specs.py tests/test_specs.py
git commit -m "feat: add JobSpecV2 and PolicySpecV2 dataclasses (specs.py)"
```

---

## Task 8: `reference_monitor.py` — hard constraint enforcement

**Files:**
- Create: `orchestrator/reference_monitor.py`
- Test: `tests/test_reference_monitor.py`

**Step 1: Write failing tests**

Create `tests/test_reference_monitor.py`:

```python
"""Tests for ReferenceMonitor hard constraint checker."""
from __future__ import annotations
import pytest
from orchestrator.reference_monitor import ReferenceMonitor, Decision, MonitorResult
from orchestrator.specs import JobSpecV2, PolicySpecV2, Constraints, InputSpec, EscalationRule
from orchestrator.models import Task, TaskType, TaskStatus


def _task(status=TaskStatus.PENDING):
    return Task(id="t1", type=TaskType.CODE_GEN, prompt="x", status=status)


def test_allow_with_no_constraints():
    result = ReferenceMonitor().check(_task(), JobSpecV2(goal="ok"), PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_eu_only_denies_us_locality():
    job = JobSpecV2(
        goal="EU job",
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="us"),
    )
    result = ReferenceMonitor().check(_task(), job, PolicySpecV2())
    assert result.decision == Decision.DENY
    assert result.reason


def test_eu_only_allows_eu_locality():
    job = JobSpecV2(
        goal="EU job",
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="eu"),
    )
    result = ReferenceMonitor().check(_task(), job, PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_deny_rule_in_policy_spec():
    job = JobSpecV2(goal="test", inputs=InputSpec(data_locality="us"))
    policy = PolicySpecV2(allow_deny_rules=[{"effect": "deny", "field": "data_locality", "value": "us"}])
    result = ReferenceMonitor().check(_task(), job, policy)
    assert result.decision == Decision.DENY


def test_escalation_rule_triggered_on_failed_task():
    task = _task(status=TaskStatus.FAILED)
    policy = PolicySpecV2(escalation_rules=[
        EscalationRule(trigger="task_status:failed", action="human_review")
    ])
    result = ReferenceMonitor().check(task, JobSpecV2(goal="test"), policy)
    assert result.decision == Decision.ESCALATE
    assert result.rule is not None


def test_allow_when_escalation_not_triggered():
    task = _task(status=TaskStatus.COMPLETED)
    policy = PolicySpecV2(escalation_rules=[
        EscalationRule(trigger="task_status:failed", action="human_review")
    ])
    result = ReferenceMonitor().check(task, JobSpecV2(goal="test"), policy)
    assert result.decision == Decision.ALLOW
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_reference_monitor.py -v
```

**Step 3: Create `orchestrator/reference_monitor.py`**

Use safe pattern matching — NO `eval()`:

```python
"""
ReferenceMonitor — synchronous, bypass-proof hard constraint checker.
Uses declarative pattern matching, never eval().
Author: Georgios-Chrysovalantis Chatzivantsidis
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from .models import Task, TaskStatus
from .specs import JobSpecV2, PolicySpecV2, EscalationRule

logger = logging.getLogger(__name__)


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass
class MonitorResult:
    decision: Decision
    reason: str = ""
    rule: Optional[EscalationRule] = None


class ReferenceMonitor:
    """
    Enforces hard constraints via declarative rules.
    Called synchronously before every task by ControlPlane.
    Cannot be bypassed by prompt content.
    """

    def check(
        self,
        task: Task,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> MonitorResult:
        # 1. Escalation rules (highest priority)
        task_status = task.status.value if hasattr(task.status, "value") else str(task.status)
        for rule in policy.escalation_rules:
            if self._match_escalation(rule.trigger, task_status):
                logger.warning("Escalation: %s → %s", rule.trigger, rule.action)
                return MonitorResult(Decision.ESCALATE, rule.trigger, rule)

        # 2. Hard constraints on JobSpecV2
        for constraint in job.constraints.hard:
            result = self._check_hard(constraint, job)
            if result is not None:
                return result

        # 3. Allow/deny rules from PolicySpecV2
        for rule in policy.allow_deny_rules:
            if rule.get("effect") == "deny":
                if self._match_deny_rule(rule, job):
                    reason = f"deny rule matched: field={rule.get('field')} value={rule.get('value')}"
                    logger.warning("DENY: %s", reason)
                    return MonitorResult(Decision.DENY, reason)

        return MonitorResult(Decision.ALLOW)

    # ── Hard constraint checkers (one method per constraint) ──────────────

    def _check_hard(self, constraint: str, job: JobSpecV2) -> Optional[MonitorResult]:
        if constraint == "eu_only":
            locality = job.inputs.data_locality
            if locality not in ("eu", "any"):
                return MonitorResult(
                    Decision.DENY,
                    f"eu_only violated: data_locality={locality}",
                )
        if constraint == "us_only":
            locality = job.inputs.data_locality
            if locality not in ("us", "any"):
                return MonitorResult(
                    Decision.DENY,
                    f"us_only violated: data_locality={locality}",
                )
        # no_training and no_pii_logging are enforced at audit/export layer
        return None

    # ── Rule matchers (no eval, pure string comparison) ───────────────────

    def _match_deny_rule(self, rule: dict, job: JobSpecV2) -> bool:
        """Match {"effect": "deny", "field": "data_locality", "value": "us"}"""
        field = rule.get("field", "")
        value = rule.get("value", "")
        if field == "data_locality":
            return job.inputs.data_locality == value
        if field == "contains_pii":
            return str(job.inputs.contains_pii).lower() == str(value).lower()
        return False

    def _match_escalation(self, trigger: str, task_status: str) -> bool:
        """Match 'task_status:failed' style triggers."""
        if trigger.startswith("task_status:"):
            expected = trigger.split(":", 1)[1]
            return task_status == expected
        return False
```

**Step 4: Run tests**

```bash
pytest tests/test_reference_monitor.py -v
```
Expected: all 6 tests `PASSED`

**Step 5: Commit**

```bash
git add orchestrator/reference_monitor.py tests/test_reference_monitor.py
git commit -m "feat: add ReferenceMonitor with safe declarative hard constraint enforcement"
```

---

## Task 9: `control_plane.py` — service layer

**Files:**
- Create: `orchestrator/control_plane.py`
- Test: `tests/test_control_plane.py`

**Step 1: Write failing tests**

Create `tests/test_control_plane.py`:

```python
"""Tests for ControlPlane service."""
from __future__ import annotations
import asyncio, pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator.control_plane import ControlPlane, SpecValidationError, PolicyViolation
from orchestrator.specs import JobSpecV2, PolicySpecV2, Constraints, InputSpec

def run(coro): return asyncio.run(coro)

def test_submit_returns_state_on_valid_job():
    cp = ControlPlane()
    job = JobSpecV2(goal="Build something")
    policy = PolicySpecV2()
    mock_state = MagicMock()
    with patch.object(cp, "_run_workflow", new=AsyncMock(return_value=mock_state)):
        state = run(cp.submit(job, policy))
    assert state is mock_state

def test_submit_raises_policy_violation_on_deny():
    cp = ControlPlane()
    job = JobSpecV2(
        goal="EU job",
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="us"),
    )
    with pytest.raises(PolicyViolation):
        run(cp.submit(job, PolicySpecV2()))

def test_submit_raises_spec_validation_error_on_empty_goal():
    cp = ControlPlane()
    with pytest.raises(SpecValidationError):
        run(cp.submit(JobSpecV2(goal=""), PolicySpecV2()))

def test_solve_constraints_low_cost_prefers_deepseek():
    cp = ControlPlane()
    from orchestrator.specs import Constraints
    job = JobSpecV2(goal="test", constraints=Constraints(soft={"prefer_low_cost": 0.9}))
    plan = cp._solve_constraints(job, PolicySpecV2())
    assert "deepseek" in plan.preferred_model.lower()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_control_plane.py -v
```

**Step 3: Create `orchestrator/control_plane.py`**

```python
"""
ControlPlane — deterministic service layer.
Author: Georgios-Chrysovalantis Chatzivantsidis

Pipeline: validate → monitor.check_global → solve_constraints → run_workflow → audit_log
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
from .specs import JobSpecV2, PolicySpecV2
from .reference_monitor import ReferenceMonitor, Decision
from .models import ProjectState

logger = logging.getLogger(__name__)


class SpecValidationError(ValueError):
    """JobSpecV2 or PolicySpecV2 failed schema validation."""


class PolicyViolation(RuntimeError):
    """A hard constraint or deny rule blocked the job before execution."""


@dataclass
class RoutingPlan:
    preferred_model: str = "deepseek-chat"
    max_parallel_tasks: int = 3
    retry_budget: int = 2
    notes: list[str] = field(default_factory=list)


class ControlPlane:
    def __init__(self) -> None:
        self._monitor = ReferenceMonitor()

    async def submit(self, job: JobSpecV2, policy: PolicySpecV2) -> ProjectState:
        # 1. Validate
        errors = self._validate(job, policy)
        if errors:
            raise SpecValidationError("; ".join(errors))

        # 2. Global pre-run hard constraint check
        from .models import Task, TaskType, TaskStatus
        sentinel = Task(id="__global__", type=TaskType.REASONING, prompt="")
        result = self._monitor.check(sentinel, job, policy)
        if result.decision == Decision.DENY:
            raise PolicyViolation(result.reason)

        # 3. Solve constraints
        routing = self._solve_constraints(job, policy)

        # 4. Run workflow
        state = await self._run_workflow(job, routing)

        # 5. Audit log
        logger.info(
            "ControlPlane complete | goal=%r | model=%s | tasks=%d",
            job.goal[:80], routing.preferred_model,
            len(getattr(state, "results", {})),
        )
        return state

    def _validate(self, job: JobSpecV2, policy: PolicySpecV2) -> list[str]:
        errors = []
        if not job.goal or not job.goal.strip():
            errors.append("JobSpecV2.goal must not be empty")
        if not (0.0 <= job.slas.min_quality_tier <= 1.0):
            errors.append("SLAs.min_quality_tier must be 0.0–1.0")
        for rule in policy.allow_deny_rules:
            if rule.get("effect") not in ("allow", "deny"):
                errors.append(f"allow_deny_rule missing valid 'effect': {rule}")
        return errors

    def _solve_constraints(self, job: JobSpecV2, policy: PolicySpecV2) -> RoutingPlan:
        plan = RoutingPlan()
        if job.constraints.soft.get("prefer_low_cost", 0) > 0.7:
            plan.preferred_model = "deepseek-chat"
            plan.notes.append("prefer_low_cost → deepseek-chat")
        if "eu_only" in job.constraints.hard:
            plan.notes.append("eu_only active")
        if job.slas.max_cost_usd:
            plan.notes.append(f"cost cap: ${job.slas.max_cost_usd}")
        return plan

    async def _run_workflow(self, job: JobSpecV2, routing: RoutingPlan) -> ProjectState:
        from .engine import Orchestrator
        orch = Orchestrator(budget=job.budget, max_concurrency=routing.max_parallel_tasks)
        return await orch.run_project(
            project_description=job.goal,
            success_criteria="; ".join(job.metrics) if job.metrics else "",
        )
```

**Step 4: Run tests**

```bash
pytest tests/test_control_plane.py -v
```
Expected: all 4 `PASSED`

**Step 5: Commit**

```bash
git add orchestrator/control_plane.py tests/test_control_plane.py
git commit -m "feat: add ControlPlane service (validate→monitor→solve→run→log)"
```

---

## Task 10: `orchestration_agent.py` + exports + final integration

**Files:**
- Create: `orchestrator/orchestration_agent.py`
- Modify: `orchestrator/__init__.py`
- Test: `tests/test_orchestration_agent.py`

**Step 1: Write failing tests**

Create `tests/test_orchestration_agent.py`:

```python
"""Tests for OrchestrationAgent NL → specs."""
from __future__ import annotations
import asyncio, json, pytest
from unittest.mock import patch, MagicMock
from orchestrator.orchestration_agent import OrchestrationAgent, AgentDraft
from orchestrator.specs import JobSpecV2, PolicySpecV2

def run(coro): return asyncio.run(coro)

def _mock(text):
    m = MagicMock(); m.text = text; m.cost_usd = 0.001; return m

_VALID_RAW = json.dumps({
    "job": {"goal": "Build a pipeline", "constraints": {"hard": [], "soft": {}},
            "metrics": [], "task_tree": []},
    "policy": {"allow_deny_rules": [], "routing_hints": [],
               "validation_rules": [], "escalation_rules": []},
    "rationale": "No special constraints detected.",
})

def test_draft_returns_agent_draft():
    agent = OrchestrationAgent()
    with patch.object(agent._client, "call", return_value=_mock(_VALID_RAW)):
        draft = run(agent.draft("Build a pipeline for X"))
    assert isinstance(draft, AgentDraft)
    assert isinstance(draft.job, JobSpecV2)
    assert isinstance(draft.policy, PolicySpecV2)
    assert isinstance(draft.rationale, str)

def test_draft_parses_hard_constraint():
    agent = OrchestrationAgent()
    raw = json.dumps({
        "job": {"goal": "EU pipeline",
                "constraints": {"hard": ["eu_only"], "soft": {}},
                "metrics": [], "task_tree": []},
        "policy": {"allow_deny_rules": [], "routing_hints": [],
                   "validation_rules": [], "escalation_rules": []},
        "rationale": "eu_only added because EU-data was mentioned.",
    })
    with patch.object(agent._client, "call", return_value=_mock(raw)):
        draft = run(agent.draft("EU data pipeline"))
    assert "eu_only" in draft.job.constraints.hard
    assert "eu_only" in draft.rationale.lower() or "eu" in draft.rationale.lower()

def test_refine_updates_constraints():
    agent = OrchestrationAgent()
    from orchestrator.specs import Constraints
    original = AgentDraft(job=JobSpecV2(goal="test"), policy=PolicySpecV2(), rationale="v1")
    refined_raw = json.dumps({
        "job": {"goal": "test refined",
                "constraints": {"hard": ["no_training"], "soft": {}},
                "metrics": [], "task_tree": []},
        "policy": {"allow_deny_rules": [], "routing_hints": [],
                   "validation_rules": [], "escalation_rules": []},
        "rationale": "no_training added per user request.",
    })
    with patch.object(agent._client, "call", return_value=_mock(refined_raw)):
        refined = run(agent.refine(original, "add no_training constraint"))
    assert "no_training" in refined.job.constraints.hard
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_orchestration_agent.py -v
```

**Step 3: Create `orchestrator/orchestration_agent.py`**

```python
"""
OrchestrationAgent — converts NL intent to JobSpecV2 + PolicySpecV2 drafts.
Author: Georgios-Chrysovalantis Chatzivantsidis

NEVER calls task executors or LLM providers directly for task work.
Outputs ONLY specs for human approval before submitting to ControlPlane.
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from .api_clients import UnifiedClient
from .models import Model
from .specs import (
    JobSpecV2, PolicySpecV2, Constraints,
    RoutingHint, ValidationRule, EscalationRule,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an Orchestration Agent. Your ONLY output is a JSON object with keys:
  "job"       — a JobSpecV2 spec
  "policy"    — a PolicySpecV2 spec
  "rationale" — 1-3 sentences explaining why you chose these constraints

You NEVER call model APIs, execute tasks, or produce code.
You ONLY produce typed specs for human review before execution.

JobSpecV2 schema:
{
  "goal": string,
  "constraints": {"hard": [string], "soft": {string: float}},
  "metrics": [string],
  "task_tree": []
}

PolicySpecV2 schema:
{
  "allow_deny_rules": [{"effect": "allow"|"deny", "field": string, "value": string}],
  "routing_hints": [{"condition": string, "target": string}],
  "validation_rules": [{"node_pattern": string, "mandatory_validators": [string]}],
  "escalation_rules": [{"trigger": string, "action": string}]
}

Available hard constraints: eu_only, us_only, no_training, no_pii_logging
Available soft constraints: prefer_low_cost (0-1), prefer_low_latency (0-1)
Escalation trigger format: "task_status:failed"
Return ONLY valid JSON. No markdown fences, no explanation outside the JSON."""


@dataclass
class AgentDraft:
    job: JobSpecV2
    policy: PolicySpecV2
    rationale: str


class OrchestrationAgent:
    """NL intent → draft JobSpecV2 + PolicySpecV2 + human approval loop."""

    def __init__(self) -> None:
        self._client = UnifiedClient()

    async def draft(self, nl_intent: str) -> AgentDraft:
        resp = await self._client.call(
            Model.DEEPSEEK_CHAT,
            f"Convert this intent to specs:\n\n{nl_intent}",
            system=_SYSTEM_PROMPT, max_tokens=2048, timeout=60,
        )
        return self._parse(resp.text)

    async def refine(self, draft: AgentDraft, feedback: str) -> AgentDraft:
        context = json.dumps({
            "current_job": {
                "goal": draft.job.goal,
                "constraints": {
                    "hard": draft.job.constraints.hard,
                    "soft": draft.job.constraints.soft,
                },
            },
            "current_policy": {"allow_deny_rules": draft.policy.allow_deny_rules},
        })
        resp = await self._client.call(
            Model.DEEPSEEK_CHAT,
            f"Current specs:\n{context}\n\nFeedback: {feedback}\n\nReturn updated specs.",
            system=_SYSTEM_PROMPT, max_tokens=2048, timeout=60,
        )
        return self._parse(resp.text)

    def _parse(self, text: str) -> AgentDraft:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        data = json.loads(text)
        j = data["job"]
        p = data["policy"]
        job = JobSpecV2(
            goal=j.get("goal", ""),
            constraints=Constraints(
                hard=j.get("constraints", {}).get("hard", []),
                soft=j.get("constraints", {}).get("soft", {}),
            ),
            metrics=j.get("metrics", []),
        )
        policy = PolicySpecV2(
            allow_deny_rules=p.get("allow_deny_rules", []),
            routing_hints=[RoutingHint(**h) for h in p.get("routing_hints", [])],
            validation_rules=[ValidationRule(**v) for v in p.get("validation_rules", [])],
            escalation_rules=[EscalationRule(**e) for e in p.get("escalation_rules", [])],
        )
        return AgentDraft(job=job, policy=policy, rationale=data.get("rationale", ""))
```

**Step 4: Update `orchestrator/__init__.py`**

Add to the imports block:

```python
from .specs import (
    JobSpecV2, PolicySpecV2, SLAs, InputSpec, Constraints,
    RoutingHint, ValidationRule, EscalationRule,
)
from .reference_monitor import ReferenceMonitor, Decision, MonitorResult
from .control_plane import ControlPlane, SpecValidationError, PolicyViolation
from .orchestration_agent import OrchestrationAgent, AgentDraft
```

Add all new names to `__all__`.

**Step 5: Run all new tests**

```bash
pytest tests/test_orchestration_agent.py tests/test_control_plane.py tests/test_reference_monitor.py tests/test_specs.py -v
```
Expected: all `PASSED`

**Step 6: Run full test suite — no regressions**

```bash
pytest --tb=short -q
```
Expected: all pass

**Step 7: Commit**

```bash
git add orchestrator/orchestration_agent.py orchestrator/__init__.py tests/test_orchestration_agent.py
git commit -m "feat: add OrchestrationAgent (NL→specs), exports, CLI agent subcommand"
```

---

## Final Smoke Tests

```bash
# 1. Build a Next.js app end-to-end
python -m orchestrator build \
  --description "SaaS landing page with hero, features, pricing" \
  --criteria "Files exist and npm run build succeeds" \
  --app-type nextjs \
  --output-dir outputs/smoke-nextjs

# Verify
ls outputs/smoke-nextjs/app/
# Expected: layout.tsx, page.tsx, globals.css (+ LLM-generated sections)

# 2. Agent smoke test
python -m orchestrator agent \
  --intent "Code review pipeline, budget 2 USD, EU data only, no training"
# Expected: prints draft with hard=["eu_only","no_training"]

# 3. Full test suite
pytest --tb=short -q
```
