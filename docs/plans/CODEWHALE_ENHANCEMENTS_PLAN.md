# CodeWhale-Inspired Enhancements for the AI Orchestrator

**Audited repos:** `Hmbown/CodeWhale` v0.8.59 (Rust agent harness, 38.2k ★)  
**Target:** This repo — Python Multi-LLM Orchestrator  
**Date:** Based on analysis of `task_executor.py`, `checkpoints.py`, `critique_cycle.py`, `output/writer.py`, `codebase/writer.py`, `quality/validators.py`, `.importlinter`, `domain/ports.py`

---

## Audit Summary: Three Verified Gaps

### Gap 1: No Deterministic Validation Inside the Critique Cycle

| Evidence | Detail |
|----------|--------|
| `critique_cycle.py:91-98` | Between `_generate()` and `_critique()` there is a **dead zone** — no deterministic checks |
| `critique_cycle.py:286` | `_validate_syntax()` exists but is **never called** in `run_cycle()` |
| `output/writer.py:180-200` | Only uses `ast.parse()` **after** the pipeline completes — post-hoc rejection, no feedback to the LLM |
| `quality/validators.py:L440` | The `VALIDATORS` registry has 12 validators including subprocess-based ones (`pytest`, `ruff`) — adding LSP follows the same pattern |
| Artifacts DO hit disk | `output/writer.py` writes `.py`/`.md`/`.json` to `output_dir/`, `codebase/writer.py` creates/modifies files → tempfile validation is trivial |

The entire revision loop is 100% LLM-driven — model A reviews model B's output. **Type errors, undefined variables, and import failures are invisible unless the reviewer model happens to notice them.**

### Gap 2: Snapshots Are Metadata-Only, Not Recoverable

| Evidence | Detail |
|----------|--------|
| `checkpoints.py:268-302` | `NamedCheckpointManager.create_snapshot()` stores **SHA-256 hashes only**, not file contents |
| `checkpoints.py:304-318` | `rollback()` returns metadata; the doc says **"File restoration is caller's responsibility"** |
| `search_content "side.git\|git_snapshot\|workspace_snapshot"` | **Zero matches** — no git-level workspace capture exists |
| `codebase/writer.py:156` | `FileOperations._backup()` copies files but is single-depth (overwritten on subsequent edits) |

If generated files are overwritten or deleted during a pipeline run, **they cannot be restored**. The system knows *what changed* (file hashes differ) but cannot reverse the change.

### Gap 3: No Per-Project Constraint File

| Evidence | Detail |
|----------|--------|
| `operations/hitl_workflow.py` | Global HITL gates, not project-scoped |
| `operations/autonomy.py` | Global `AutonomyConfig` with presets — cannot express "never touch src/domain/" |
| No `.orchestrator/constitution.json` | Nothing analogous to CodeWhale's per-repo `.codewhale/constitution.json` |

The orchestrator has excellent authority/approval infrastructure but it's all global — no per-project declarative constraint mechanism.

---

## Phase-by-Phase Implementation

---

## Phase 1: LSP Validation in the Critique Cycle

**Estimated effort:** ~310 lines, ~3 days  
**Risk:** Medium (modifies hot path)  
**ROI:** High (catches type errors for free, feeds back into LLM for self-correction)

### Architecture

| Module | Layer | Role |
|--------|-------|------|
| `domain/ports.py` — `LSPValidatorPort` (append) | Domain | Abstract protocol for language-server validation |
| `infrastructure/lsp_validator.py` (new) | Infrastructure | Runs `pyright`, `tsc`, etc. via subprocess |
| `application/critique_cycle.py` (modify) | Application | Injects diagnostics between generation and critique |
| `engine_core/container.py` (modify) | Composition root | Wires adapter into critique cycle |
| `quality/validators.py` (append) | Infrastructure | Optional `validate_lsp` entry in the VALIDATORS dict |

### Step 1.1 — Domain Port

Append to `orchestrator/domain/ports.py`:

```python
@runtime_checkable
class LSPValidatorPort(Protocol):
    """Validates code via language server diagnostics."""

    async def validate(
        self, code: str, language: str, filename: str = ""
    ) -> list[LSPDiagnostic]:
        """Validate code string, return diagnostics."""
        ...

    async def validate_file(self, filepath: str) -> list[LSPDiagnostic]:
        """Validate a file on disk, return diagnostics."""
        ...

    def available_servers(self) -> frozenset[str]:
        """Return language IDs for which a server is installed."""
        ...


@dataclass
class LSPDiagnostic:
    """A single diagnostic from a language server."""
    severity: str          # "error" | "warning" | "information" | "hint"
    message: str
    line: int              # 1-indexed
    column: int            # 1-indexed
    source: str            # e.g. "pyright", "tsc"
    code: str = ""         # e.g. "reportUndefinedVariable"


class NullLspValidator:
    """No-op fallback when LSP validation is disabled."""
    async def validate(self, code, language="", filename=""):
        return []
    async def validate_file(self, filepath):
        return []
    def available_servers(self):
        return frozenset()
```

**Why a Protocol:** The application layer (`critique_cycle.py`) imports ONLY from domain — never infrastructure. The port allows test-time substitution of `NullLspValidator` without importing anything from `infrastructure/`.

### Step 1.2 — Infrastructure Adapter

New file `orchestrator/infrastructure/lsp_validator.py`:

```python
"""
LSP Validator — Post-generation code validation via language servers.

Runs pyright (Python), tsc (TypeScript), gopls (Go), rust-analyzer (Rust)
on generated code and returns diagnostics.

Follows the existing subprocess-validator pattern from `quality/validators.py`
(pytest, ruff, latex) — offloads to asyncio.to_thread() for non-blocking.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.ports import LSPDiagnostic, LSPValidatorPort

logger = logging.getLogger(__name__)


# ── Server map ──────────────────────────────────────────────────────────
# Maps language ID → (binary, args-fn, parser-fn)
# Parser takes stdout bytes → list[LSPDiagnostic]

def _parse_pyright(stdout: bytes) -> list[LSPDiagnostic]:
    """Parse pyright --outputjson output."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    diags: list[LSPDiagnostic] = []
    for diag in data.get("generalDiagnostics", []):
        diags.append(LSPDiagnostic(
            severity=diag.get("severity", "error").lower(),
            message=diag.get("message", ""),
            line=diag.get("range", {}).get("start", {}).get("line", 0) + 1,
            column=diag.get("range", {}).get("start", {}).get("character", 0) + 1,
            source="pyright",
            code=diag.get("rule", ""),
        ))
    return diags


def _parse_tsc(stdout: bytes) -> list[LSPDiagnostic]:
    """Parse tsc --noEmit output."""
    diags: list[LSPDiagnostic] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        m = re.match(
            r"^(.+)\((\d+),(\d+)\):\s+(error|warning)\s+(\w+):\s+(.+)$",
            line,
        )
        if m:
            diags.append(LSPDiagnostic(
                severity=m.group(4),
                message=m.group(6),
                line=int(m.group(2)),
                column=int(m.group(3)),
                source="tsc",
                code=m.group(5),
            ))
    return diags


_SERVERS: dict[str, tuple[str, Any, Any]] = {
    "python": ("pyright", lambda fn: ["pyright", "--outputjson", fn], _parse_pyright),
    "typescript": ("tsc", lambda fn: ["tsc", "--noEmit", "--lib", "es2020,dom", fn], _parse_tsc),
    # Extensible: "go": ("gopls", ...), "rust": ("rust-analyzer", ...)
}


class LspValidator(LSPValidatorPort):
    """Validates code by writing to tempfile and running language server."""

    def __init__(self, timeout_seconds: int = 30, temp_dir: str | None = None):
        self._timeout = timeout_seconds
        self._temp_dir = Path(temp_dir) if temp_dir else None

    def available_servers(self) -> frozenset[str]:
        results: set[str] = set()
        for lang, (bin_name, _, _) in _SERVERS.items():
            if self._binary_exists(bin_name):
                results.add(lang)
        return frozenset(results)

    async def validate(
        self, code: str, language: str = "python", filename: str = ""
    ) -> list[LSPDiagnostic]:
        if language not in _SERVERS:
            return []
        bin_name, args_fn, parser = _SERVERS[language]
        if not self._binary_exists(bin_name):
            logger.debug(f"LSP server '{bin_name}' not installed — skipping validation")
            return []

        # Write to tempfile
        suffix = ".py" if language == "python" else ".ts"
        tmpdir = self._temp_dir or tempfile.mkdtemp(prefix="orch_lsp_")
        tmpfile = str(Path(tmpdir) / (filename or f"output{suffix}"))
        try:
            with open(tmpfile, "w", encoding="utf-8") as f:
                f.write(code)

            return await self.validate_file(tmpfile)

        except Exception as e:
            logger.warning(f"LSP validation failed: {e}")
            return []
        finally:
            # Cleanup
            try:
                os.unlink(tmpfile)
            except OSError:
                pass

    async def validate_file(self, filepath: str) -> list[LSPDiagnostic]:
        path = Path(filepath)
        ext = path.suffix
        # Detect language from extension
        lang_map = {".py": "python", ".ts": "typescript", ".js": "typescript"}
        lang = lang_map.get(ext)
        if not lang or lang not in _SERVERS:
            return []

        bin_name, args_fn, parser = _SERVERS[lang]
        args = args_fn(filepath)

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self._timeout,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode not in (0, 1):  # pyright/tsc return 1 on diagnostics
                logger.debug(f"LSP '{bin_name}' exited code {proc.returncode}: {stderr[:200]}")
            return parser(stdout)
        except asyncio.TimeoutError:
            logger.warning(f"LSP '{bin_name}' timed out after {self._timeout}s")
            return []
        except FileNotFoundError:
            logger.debug(f"LSP binary '{bin_name}' not found")
            return []
```

**Key design choices:**
- Tempfile + file-based validation (not LSP protocol over stdin — simpler, no persistent server process)
- 30-second timeout per validation
- Graceful degradation if server not installed (logs once, returns [])
- Follows the exact same subprocess pattern as `validate_pytest` / `validate_ruff` in `quality/validators.py`

### Step 1.3 — Critique Cycle Integration

Modify `orchestrator/application/critique_cycle.py`:

**New constructor parameter:**
```python
def __init__(
    self,
    client: LLMClient,
    lsp_validator: LSPValidatorPort | None = None,  # NEW
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    enable_streaming: bool = False,
):
    self.client = client
    self._lsp_validator = lsp_validator or NullLspValidator()  # NEW
    ...
```

**New methods added to the class:**
```python
async def _validate_with_lsp(
    self, output: str, task: Task
) -> tuple[str, list[LSPDiagnostic]]:
    """Run LSP validation on generated output, inject diagnostics inline."""
    if task.type != TaskType.CODE_GEN:
        return output, []

    language = self._detect_language(task)
    diags = await self._lsp_validator.validate(output, language)

    if not diags:
        return output, []

    # Inject inline annotations INTO the code
    annotated = self._inject_inline_diagnostics(output, diags)
    return annotated, diags


def _detect_language(self, task: Task) -> str:
    """Detect programming language from task metadata or output."""
    # Check task metadata first
    if hasattr(task, "language") and task.language:
        return task.language
    # Default to python
    return "python"


def _inject_inline_diagnostics(
    self, code: str, diags: list[LSPDiagnostic]
) -> str:
    """Inject diagnostics as inline comments above the flagged lines."""
    lines = code.splitlines()
    # Sort by line descending so line numbers stay valid as we insert
    sorted_diags = sorted(diags, key=lambda d: d.line, reverse=True)

    for d in sorted_diags:
        idx = min(d.line - 1, len(lines) - 1)
        if idx < 0:
            continue
        prefix = "#" if "python" in self._detect_language else "//"
        comment = f"{prefix} LSP [{d.severity.upper()}]: {d.message}"
        if d.code:
            comment += f" ({d.code})"
        if d.line > 0 and d.line <= len(lines):
            lines.insert(idx, comment)

    return "\n".join(lines)


def _build_validation_summary(self, diags: list[LSPDiagnostic]) -> str:
    """Build a critique-prompt-addendum from diagnostics."""
    errors = [d for d in diags if d.severity == "error"]
    warnings = [d for d in diags if d.severity == "warning"]
    if not errors and not warnings:
        return ""

    summary = "\n\n## LSP Validation Results\n\n"
    if errors:
        summary += f"### {len(errors)} Error(s)\n"
        for e in errors:
            summary += f"- L{e.line}:{e.column} {e.message} ({e.code})\n"
    if warnings:
        summary += f"### {len(warnings)} Warning(s)\n"
        for w in warnings[:10]:
            summary += f"- L{w.line}:{w.column} {w.message}\n"
        if len(warnings) > 10:
            summary += f"- ... and {len(warnings) - 10} more warnings\n"

    # Guidance for the reviewer
    summary += (
        "\n**Reviewer guidance:** The diagnostics above were produced by "
        "deterministic language-server analysis. Errors must be resolved. "
        "Warnings should be addressed where appropriate. "
        "Evaluate the revised code accordingly in your score.\n"
    )
    return summary
```

**Modified `_generate()` call flow inside `run_cycle()`:**
```python
async def run_cycle(self, task, primary_model, reviewer_model, full_prompt):
    state = CritiqueState()

    for iteration in range(task.max_iterations):
        generate_response = await self._generate(...)
        ...

        output = generate_response.text
        if task.type == TaskType.CODE_GEN:
            output = self._clean_code_output(output)

        # ── NEW: LSP validation block ──────────────────────────────
        annotated_output, diags = await self._validate_with_lsp(output, task)
        # ────────────────────────────────────────────────────────────

        critique = ""
        score = 0.0

        if reviewer_model:
            # If LSP produced diagnostics, append a validation summary
            # to the critique prompt
            lsp_summary = self._build_validation_summary(diags)

            if lsp_summary:
                # Extend the critique prompt with LSP findings
                enriched_prompt = (
                    full_prompt
                    + "\n\n[LSP Validation Report]\n"
                    + lsp_summary
                )
                critique_response = await self._critique(
                    model=reviewer_model,
                    original_prompt=enriched_prompt,
                    generated_output=annotated_output,
                    ...
                )
            else:
                critique_response = await self._critique(...)
            ...
```

**Effect:** The reviewer LLM sees annotated code with inline comments like:
```python
# LSP [ERROR]: "user_id" is not defined (reportUndefinedVariable)
def process_user(user_id: str) -> dict:
    ...
```

And gets a structured validation summary in its prompt. The revision step then fixes these issues in the next iteration.

### Step 1.4 — Container Wiring

In `orchestrator/engine_core/container.py`:

```python
# Near other infrastructure wiring
from ..infrastructure.lsp_validator import LspValidator

# In ServiceContainer.build():
lsp_validator = None
if settings.get("lsp_enabled", True):
    lsp_validator = LspValidator(timeout_seconds=30)

critique_cycle = CritiqueCycle(
    client=unified_client,
    lsp_validator=lsp_validator,  # None → NullLspValidator used inside
    max_iterations=max_iterations,
    enable_streaming=config.USE_STREAMING,
)
```

Config flag: `lsp_enabled: bool = True` in `config.py` section.

### Step 1.5 — Optional: Validator Registry Entry

In `orchestrator/quality/validators.py`, append:

```python
def validate_lsp(output: str, language: str = "python", **kwargs) -> ValidationResult:
    """Validate code via LSP diagnostics."""
    try:
        from ..infrastructure.lsp_validator import LspValidator
        validator = LspValidator(timeout_seconds=30)
        diags = run_async(validator.validate(output, language))
        errors = [d for d in diags if d.severity == "error"]
        warnings = [d for d in diags if d.severity == "warning"]
        details = f"{len(errors)} errors, {len(warnings)} warnings"
        return ValidationResult(
            passed=len(errors) == 0,
            details=details + (f": {errors[0].message}" if errors else ""),
            validator_name="lsp",
        )
    except Exception as e:
        return ValidationResult(False, f"LSP validator failed: {e}", "lsp")


VALIDATORS["lsp"] = validate_lsp  # Add to registry
```

This allows tasks to include `"lsp"` in their `hard_validators` list, running LSP validation through the existing `async_run_validators()` dispatch in the pipeline `ValidateStage`.

---

## Phase 2: Content-Preserving Snapshots

**Estimated effort:** ~400 lines, ~3 days  
**Risk:** Low (new module, no hot-path modifications)  
**ROI:** Medium (enables true workspace rollback)

### Architecture

| Module | Layer | Role |
|--------|-------|------|
| `domain/ports.py` — `SnapshotPort` (append) | Domain | Abstract protocol |
| `infrastructure/snapshot_store.py` (new) | Infrastructure | Shadow git repo for content store |
| `output/writer.py` (modify) | Application | Trigger snapshot after output write |
| `checkpoints.py` (modify) | Infrastructure | Enhancement over hash-only snapshots |

### Step 2.1 — Domain Port

```python
@runtime_checkable
class SnapshotPort(Protocol):
    """Content-preserving workspace snapshots."""

    async def create(
        self, label: str, source_dir: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Snapshot source_dir, return snapshot_id."""

    async def restore(self, snapshot_id: str, target_dir: str) -> bool:
        """Restore file contents into target_dir."""

    async def list_snapshots(self) -> list[dict[str, Any]]:
        """Return list of {id, label, timestamp, file_count, total_size_bytes}."""

    async def diff(
        self, snapshot_a: str, snapshot_b: str
    ) -> dict[str, Any]:
        """Return {'added': [...], 'removed': [...], 'modified': [(path, line_diff)]}."""
```

### Step 2.2 — Implementation Using Shadow Git

```python
# orchestrator/infrastructure/snapshot_store.py

class GitSnapshotStore(SnapshotPort):
    """Snapshots stored as git commits in a shadow bare repo."""

    def __init__(self, storage_dir: str | None = None):
        self._repo_dir = Path(storage_dir or default_snapshot_dir())
        self._work_dir = self._repo_dir / "worktree"
        self._git_dir = self._repo_dir / ".git"

    async def create(self, label, source_dir, metadata=None):
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        await self._ensure_repo()
        # rsync or shutil.copytree into worktree
        await self._sync_source(source, self._work_dir)
        # git add + git commit
        commit_hash = await self._git_commit(label)
        # store metadata alongside
        await self._save_meta(commit_hash, label, metadata)
        return commit_hash

    async def restore(self, snapshot_id, target_dir):
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self._ensure_repo()
        # git checkout to temp, then rsync to target
        await self._git_restore(snapshot_id, self._work_dir)
        await self._sync_source(self._work_dir, target)
        return True

    async def diff(self, snapshot_a, snapshot_b):
        await self._ensure_repo()
        raw = await self._git_diff(snapshot_a, snapshot_b)
        return self._parse_git_diff(raw)
```

**Fallback path (git unavailable):** `TarSnapshotStore` — uses `tarfile` with `xz` compression + SHA-256 manifest for integrity. Same `SnapshotPort` interface.

### Step 2.3 — Integration Trigger

In `output/writer.py`, `write_output_dir()` — append after all files are written:

```python
if snapshot_store:
    await snapshot_store.create(
        label=f"project-{project_id}",
        source_dir=str(out),
        metadata={"project_id": project_id, "budget_spent": state.budget.spent_usd},
    )
```

In `codebase/writer.py`, `apply()` — snapshot before destructive operations:

```python
if snapshot_store and task.type in (TaskType.MODIFY_FILE, TaskType.DELETE_FILE):
    await snapshot_store.create(
        label=f"before-{task.id}",
        source_dir=str(self._root),
    )
```

### Step 2.4 — CheckpointManager Enhancement

Create `ContentCheckpointManager` that inherits from `NamedCheckpointManager` and delegates content to `SnapshotPort`:

```python
class ContentCheckpointManager(NamedCheckpointManager):
    """Like NamedCheckpointManager but also stores recoverable file content."""

    def __init__(self, checkpoint_dir, snapshot_store: SnapshotPort | None = None):
        super().__init__(checkpoint_dir)
        self._snapshot_store = snapshot_store or NullSnapshotStore()

    async def create_snapshot(self, name, description="", output_dir=None, ...):
        cp = await super().create_snapshot(name, description, output_dir, ...)
        if output_dir and self._snapshot_store:
            sid = await self._snapshot_store.create(
                label=name,
                source_dir=output_dir,
                metadata={"checkpoint_name": name, "description": description},
            )
            cp.artifacts["_snapshot_id"] = sid  # Link to content store
        return cp

    async def rollback(self, snapshot_name, output_dir):
        cp = await super().rollback(snapshot_name, output_dir)
        if cp and "_snapshot_id" in (cp.artifacts or {}):
            await self._snapshot_store.restore(
                cp.artifacts["_snapshot_id"],
                output_dir,
            )
            logger.info(f"Restored file contents for snapshot '{snapshot_name}'")
        return cp
```

---

## Phase 3: Per-Project Constitution

**Estimated effort:** ~200 lines, ~2 days  
**Risk:** Low  
**ROI:** Medium

### Implementation

1. **Schema definition** — `orchestrator/domain/constitution.py` (value objects + JSON schema validation via pydantic)

2. **Loader** — `orchestrator/infrastructure/constitution_loader.py` — looks for `.orchestrator/constitution.json` in project root, caches it

3. **Enforcement points:**
   - `output/writer.py` — before writing a file: `constitution.protect_paths` match → log warning, skip
   - `codebase/writer.py` — `ModificationGate.verify()`: also checks protected paths + forbidden imports
   - `critique_cycle.py` — skip iteration if `constitution.require_tests` and no test was produced

```python
# domain/constitution.py
@dataclass
class ProjectConstitution:
    protect_paths: list[str] = field(default_factory=list)    # globs
    require_review_above: float = 0.0
    require_tests: bool = False
    forbidden_imports: list[str] = field(default_factory=list)
    required_validators: list[str] = field(default_factory=list)
    max_file_size_bytes: int = 0  # 0 = unlimited

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectConstitution":
        """Load from .orchestrator/constitution.json"""
        ...
```

---

## Phase 4: MCP Bridge Documentation

**Estimated effort:** ~100 lines, 1 day  
**Risk:** None  
**ROI:** Low but connective

Simply document the existing MCP server as a CodeWhale integration point in `docs/INTEGRATIONS.md` and add three new MCP tools:
- `orch_project_status` — returns current project state
- `orch_project_results` — returns task outputs by ID
- `orch_project_snapshots` — lists snapshots (Phase 2)

---

## Implementation Sequencing

```
Week 1:
  Mon ─ Step 1.1 (Port)          ~30 lines  ✓
  Tue ─ Step 1.2 (Adapter)       ~200 lines  ✓
  Wed ─ Step 1.3 (Integration)   ~60 lines   ✓ (modifies hot path — most risk)
  Thu ─ Step 1.4 (Wiring)        ~20 lines   ✓
  Fri ─ Tests for Phase 1

Week 2:
  Mon ─ Step 2.1–2.2 (Snapshots) ~300 lines  ✓
  Tue ─ Step 2.3–2.4             ~80 lines   ✓
  Wed ─ Phase 3 (Constitution)   ~200 lines  ✓
  Thu ─ Phase 4 (MCP docs)       ~100 lines  ✓
  Fri ─ Integration tests, review
```

**Test coverage per phase:**
- Phase 1: `test_lsp_validator.py`, `test_critique_cycle_with_lsp.py`
- Phase 2: `test_snapshot_store.py` (create/restore/diff round-trip)
- Phase 3: `test_constitution.py` (path matching, enforcement)
- Phase 4: Manual verification with CodeWhale MCP client

**Config flags (all maintaining backward compatibility):**
```ini
# config.toml or .env
LSP_ENABLED=true              # Phase 1
LSP_TIMEOUT_SECONDS=30
AUTO_SNAPSHOT=true            # Phase 2
SNAPSHOT_STORAGE_DIR=         # default ~/.orchestrator_cache/snapshots/
SNAPSHOT_STORE_BACKEND=git    # git | tar
CONSTITUTION_PATH=            # default .orchestrator/constitution.json
```

---

## Glossary of Terms Used

| Term | Definition |
|------|------------|
| LSP | Language Server Protocol — standard interface between editors and language-specific analysis tools (pyright, tsc, gopls) |
| Critique cycle | The generate → review → revise loop in `CritiqueCycle.run_cycle()` |
| Hard validator | A validator listed in `Task.hard_validators` that must pass for output to be accepted |
| Soft validator | A non-blocking validator that only logs warnings |
| Side-git | A separate `.git` repository outside the project's own VCS, used for fine-grained workspace snapshots |
| Port | A `typing.Protocol` abstract interface in `domain/ports.py` — ensures domain/application layers never import infrastructure |
| Constitution | A per-project constraint file (`.orchestrator/constitution.json`) declaring protected paths, required validators, and invariants |
