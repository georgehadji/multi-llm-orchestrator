---
name: orchestrator-build-and-env
description: Recreate the Multi-LLM Orchestrator dev environment from a clean checkout (Windows PowerShell or Linux bash), install with the right extras, and avoid the known environment traps. Load this BEFORE running `pip install -e ".[dev]"` for the first time, before setting up pre-commit, when a fresh clone/CI fails but your local machine "works", or when you see symptoms like: "ModuleNotFoundError: No module named 'instructor'", "black --check fails only in CI", "lint-imports Rust panics on Windows", "import orchestrator takes forever", "pre-commit stashed my changes", "SyntaxError only on Linux/CI" (BOM), "0-byte .git/index.lock", or "AuthenticationError: OpenRouter API key not found" during a plain smoke test. For flag/env-var *reference* use orchestrator-config-and-flags; for live-bug triage use orchestrator-debugging-playbook; for gates/process use orchestrator-change-control.
---

# Orchestrator Build and Env

Ground-truth setup runbook for the Multi-LLM Orchestrator. Verified against
`pyproject.toml`, `requirements.txt`, `requirements-dev.txt`,
`tests/requirements-dev.txt`, `.pre-commit-config.yaml`, `.importlinter`,
`.github/workflows/ci.yml`, and `orchestrator/infrastructure/llm_client.py`
on 2026-07-08 (branch `feat/response-healing`).

**Working tree note:** as of 2026-07-08, `pyproject.toml`, `requirements.txt`,
and `orchestrator/infrastructure/llm_client.py` are all showing as modified
(uncommitted) in `git status`. The facts below describe the *current working
tree*, not necessarily the last commit — re-run the verification commands in
§8 if you're on a different commit.

## 1. Python version policy

`pyproject.toml`:
```toml
requires-python = ">=3.10"
```
Classifiers declare 3.10–3.13 support. CI (`ci.yml`) pins **3.12** for every
job (`actions/setup-python@v5`, `python-version: "3.12"`). `mypy`'s
`python_version = "3.12"`. **Use 3.12 locally** unless you have a specific
reason to test another supported version — it's what CI actually runs and
what `mypy`/`black` target.

## 2. Create the venv

### Windows (PowerShell — repo dev machine)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / CI (bash)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install — `pip install -e ".[dev]"` and what each extra adds

Base `dependencies` (always installed, from `pyproject.toml:29-44`):
`openai`, `google-genai`, `aiosqlite`, `pydantic`, `pydantic-settings`,
`typing-extensions`, `python-dotenv`, `playwright`, `newspaper3k`, `json5`,
`httpx`, `aiohttp`, `tenacity`, **`instructor`** (structured outputs used by
`UnifiedClient` in `orchestrator/infrastructure/llm_client.py`).

| Extra | Command | What it adds | Why you'd need it |
|---|---|---|---|
| `dev` | `pip install -e ".[dev]"` | pytest, pytest-cov, pytest-asyncio, pytest-xdist, black, ruff, mypy, pre-commit, `import-linter>=2.1,<2.2`, **`grimp>=3.3,<3.4`** (pinned — see §7 trap), networkx/numpy/scipy (for `codebase_reader.py`'s `DependencyGraph` / `networkx.pagerank`), type stubs | Standard local dev loop: test, lint, format, type-check, architecture gate |
| `security` | `pip install -e ".[security]"` | `bandit[toml]`, `safety` | Security scanning outside pre-commit's bandit hook |
| `tracing` | `pip install -e ".[tracing]"` | `opentelemetry-api/sdk/exporter-otlp-proto-grpc` | Only if you're wiring distributed tracing — not required for normal dev |
| `dashboard` | `pip install -e ".[dashboard]"` | `fastapi`, `uvicorn[standard]`, `websockets`, `httpx` | Running the web dashboard (`start_dashboard.bat` / `python start_dashboard.py`) |
| `docs` | `pip install -e ".[docs]"` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building docs site |
| `image` | `pip install -e ".[image]"` | `Pillow` | Image-generation output paths |

Everyday dev: `pip install -e ".[dev]"` is sufficient. Add `security`,
`dashboard`, etc. only when your task actually touches those surfaces —
CLAUDE.md's dependency-declaration rule applies (no undeclared deps; see §7).

```powershell
pip install -e ".[dev,security,tracing]"                 # from CLAUDE.md example
pip install -e ".[dev,security,tracing,dashboard,docs]"  # everything
```

## 4. requirements.txt vs pyproject.toml — verify before trusting either

There are **four** dependency manifests in this repo. They are not kept in
lockstep and have drifted before. Know which one governs what:

| File | Role | Installed by |
|---|---|---|
| `pyproject.toml` | Canonical. `dependencies` + `[project.optional-dependencies]` | `pip install -e ".[dev]"` — **this is what CI uses** |
| `requirements.txt` | Pinned snapshot, "Generated: 2026-03-26" per its header comment | Nothing in CI installs it directly; legacy/manual reference |
| `requirements-dev.txt` (root) | `-r requirements.txt` + pinned lint/test/security tool versions | Not referenced by CI or pre-commit |
| `tests/requirements-dev.txt` | A **third, independent** dev-deps list (own pytest/black/mypy pins, includes `pytest-timeout`) | Not referenced by CI or pre-commit either |

**Verified divergences as of 2026-07-08** (re-check with the commands in §8 —
these are exactly the kind of thing that silently rots):

- **`anthropic` pin**: `requirements.txt` pins `anthropic==0.42.0`, but
  `pyproject.toml`'s `dependencies` list has **no `anthropic` entry at all**.
  The orchestrator talks to Anthropic models through OpenRouter, not the
  `anthropic` SDK directly, so this is likely a stale/unused pin in
  `requirements.txt` rather than a missing pyproject dependency — but verify
  with `grep -rn "^import anthropic\|^from anthropic" orchestrator/` before
  assuming either file is "right".
- **Bogus `asyncio==3.4.3`**: `requirements.txt` line 21 pins a PyPI package
  called `asyncio` at `3.4.3`. `asyncio` is part of the Python **standard
  library** since 3.4 — the PyPI package of the same name is an old Python
  2 backport and pinning it is either a no-op (pip usually just installs the
  useless backport alongside stdlib asyncio) or actively confusing. Do not
  copy this line into a fresh manifest. `pyproject.toml` correctly has no
  `asyncio` dependency at all.
- **`instructor` added 2026-07-07**: both `pyproject.toml`
  (`instructor>=1.0,<2.0`, base `dependencies`) and `requirements.txt`
  (`instructor==1.14.4`) now declare it, and `orchestrator/infrastructure/llm_client.py`
  imports it (lazily — see §6). This addition is what closed the incident
  described in §7 ("undeclared runtime dependency").
- **Tool version pins differ across all three dev-deps files** — see the
  black divergence in §7; the same pattern (three different pins for one
  tool) recurs for pytest, mypy, ruff across `requirements-dev.txt` and
  `tests/requirements-dev.txt`. Don't assume they agree; grep before relying
  on any one of them for "the" version.

**Rule of thumb:** if `pyproject.toml` and `requirements.txt` disagree, trust
`pyproject.toml` — it's what `pip install -e ".[dev]"` and CI actually use.
Treat `requirements.txt` / `requirements-dev.txt` / `tests/requirements-dev.txt`
as legacy references that need reconciling, not sources of truth.

## 5. `.env` keys — required vs optional

Full flag/env-var catalogue lives in **orchestrator-config-and-flags**
(§5 "API keys and `.env` loading" in that skill) — this section is the
one-screen version you need to get a smoke test running.

| Key | Status | Where it's read | Notes |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Required** for any real LLM call | `orchestrator/infrastructure/llm_client.py:180` (`UnifiedClient.__init__`) | **Validated eagerly at construction** — raises `AuthenticationError("OpenRouter API key not found...")` immediately if unset, not on first call. This eager check is a recent change (feat/response-healing branch). Everything is reached through OpenRouter; this is *the* key. |
| `XAI_API_KEY` (alias `GROK_API_KEY`) | Optional | `llm_client.py:416`, `knowledge/xai_search.py`, `rate_limiter.py` | Enables a direct xAI client; without it, xAI models route through OpenRouter instead. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | **Not in the LLM call path** | Only `operations/diagnostics.py` (health checks), `generators/secrets_manager.py`, `slash_integrations.py`, example files | Peripheral. Don't tell a new contributor these are required to run a project — they aren't. |

`orchestrator/cli.py:19-21` runs `load_dotenv(override=True)` at import time.
`override=True` means **values in `.env` beat already-exported shell env
vars** for any CLI invocation — if an env var you exported "won't take
effect", check `.env` first.

For tests/CI that must not hit the real network, the pattern is:
```python
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
```
A dummy key satisfies the eager `UnifiedClient` constructor check but will
still make a real network call and retry/hang if you actually invoke it — see
the integration-fixture trap in §7 (last row) for the real fix.

## 6. Smoke verification sequence

Run these in order after installing. Expected characteristics verified on
this repo/branch as of 2026-07-08:

```powershell
# 1. Import timing — dominated by openai's import graph, ~12-13s warm.
#    This is normal; don't chase it unless it regresses noticeably.
Measure-Command { python -c "import orchestrator" }

# 2. If it regresses, find the new heavy import:
python -X importtime -c "import orchestrator" 2> importtime_out.txt
# sort by self time (column 2, microseconds) — Get-Content works fine for viewing:
Get-Content importtime_out.txt | Sort-Object { [int]($_ -split '\|')[1] } -Descending | Select-Object -First 20

# 3. CLI wiring sanity check
python -m orchestrator --help

# 4. Small pytest run — do NOT add --timeout locally (see §7)
pytest tests/unit -m unit -q --no-cov
```

**Why `import orchestrator` is ~12-13s and not instant:** `openai`'s import
graph is the dominant cost. `instructor` used to add **~30s** to this on cold
Windows imports when it was imported at module scope; it was made a
function-local ("lazy") import inside `UnifiedClient._instructor_mode()` and
the two `_build_client` methods on 2026-07-07 specifically to keep
`import orchestrator` (and CLI startup) fast for code paths that never build
an LLM client. The module still has a `TYPE_CHECKING`-only top-level
`from instructor.client import AsyncInstructor` (`llm_client.py:15`) — that's
free at runtime, it only exists for type checkers, so don't "fix" it into a
real import.

**Rule for future top-level imports:** before adding any new *unconditional*
top-level import to a module reachable from `orchestrator/__init__.py`,
profile with `python -X importtime -c "import orchestrator"` first and
compare against the ~12-13s baseline. If it's heavy, import it lazily inside
the function(s) that actually need it, following the `_instructor_mode()`
pattern above.

## 7. Known traps

| Trap | Story | Detection | Fix |
|---|---|---|---|
| **BOM characters** | 32 `.py` files had UTF-8 BOM markers, causing cryptic import errors on Linux (but not Windows) — fixed in commit `d913d136` along with 4 bare `except:` clauses that were masking `SystemExit`/`KeyboardInterrupt` | `python .claude/skills/orchestrator-diagnostics-and-tooling/scripts/check_bom.py` (if present); or `git grep -lP '^\xef\xbb\xbf'` won't work in PowerShell — use a Python one-liner: `python -c "import pathlib,sys;[print(p) for p in pathlib.Path('orchestrator').rglob('*.py') if open(p,'rb').read(3)==b'\xef\xbb\xbf']"` | Strip BOM (save as UTF-8 without BOM) before committing |
| **grimp Rust panic on Windows** | `import-linter`'s graph backend `grimp` is pinned `>=3.3,<3.4` in `pyproject.toml` (`dev` extra) because 3.4+ Rust-panics on Windows with this codebase. Even with the pin, `lint-imports` run locally on Windows can still surface a raw Rust `PanicException`/traceback instead of a clean pass/fail report on some machines — this is a known-flaky local experience, not a contract failure | `pip show grimp` (must report `3.3.x`); if `lint-imports` crashes with a panic trace rather than printing contract results, don't treat that as "the contracts are broken" | **CI is authoritative** — the `architecture` job in `ci.yml` runs on `ubuntu-latest`, where grimp does not panic. Trust CI's `lint-imports` result over a crashing local Windows run. If you need a local signal, re-run in WSL/Linux. |
| **black CI vs local version drift** | CI pins `black==26.1.0` explicitly (`ci.yml:22`, with a comment: *"Pin black so CI and local pre-commit format identically (a floating black version reformats files that pass locally and breaks the gate)"*), but `requirements-dev.txt` (root) pins `black==24.10.0` and `pyproject.toml`'s `dev` extra only constrains `black>=23.7,<25.0` — so a plain `pip install -e ".[dev]"` gives you **24.x**, not the 26.1.0 CI actually checks against | `rg -n "black==|black>=" .github/workflows/ci.yml requirements-dev.txt pyproject.toml` | If `black --check` passes locally but you're unsure it'll pass in CI (or vice versa), install the exact CI pin in a scratch venv: `pip install "black==26.1.0"` and re-run `black --check orchestrator/ tests/`. Format with whichever version you're about to be gated by. |
| **Stale `.git/index.lock` after a crashed session** | An interrupted Claude Code / git session can leave a 0-byte `.git/index.lock` behind, which makes every subsequent `git` command fail with `fatal: Unable to create '...index.lock': File exists` | `Test-Path .git/index.lock` (PowerShell) or check its size is 0 bytes | Before deleting, confirm no live git process actually holds it: `tasklist | findstr git` (Windows) — if the only matches are `git-fsmonitor--daemon` background processes (which are expected to persist and are *not* the process that created the lock), it is safe to delete: `Remove-Item .git/index.lock`. If an actual `git.exe` process for a foreground command shows up, wait for it or investigate before deleting. |
| **Undeclared runtime dependency ships in a module** | 2026-07-07: `import instructor` landed in `orchestrator/infrastructure/llm_client.py` without a corresponding `pyproject.toml` entry at the time. Local dev machines already had `instructor` installed from earlier work, so nothing caught it — until CI's clean `pip install -e ".[dev]"` hit `ModuleNotFoundError: No module named 'instructor'` during `conftest.py` collection | Any `ModuleNotFoundError` in CI that doesn't reproduce locally is a signal your local venv has accumulated packages that aren't actually declared | **Always verify a change is complete by installing in a genuinely clean venv** (`python -m venv .venv-clean && .venv-clean\Scripts\pip install -e ".[dev]"`), not by trusting an accumulated local environment. This incident is also why `instructor` is now correctly declared in both `pyproject.toml` and `requirements.txt` (§4). |
| **Heavy top-level imports in hot paths** | Same `instructor` addition initially imported it at module scope, adding ~30s to cold `import orchestrator` on Windows | `python -X importtime -c "import orchestrator"` before/after any new top-level import in a module reachable from `orchestrator/__init__.py` | Move the import inside the function(s) that need it (see §6's `_instructor_mode()` pattern) if it's expensive and not always needed |
| **`.claude/hooks` fire on every edit** | `post_edit_ruff.py` runs `ruff check --fix --quiet` on any `.py` file you Write/Edit under `orchestrator/` or `tests/` (advisory, never blocks); `pre_edit_core.py` prints an architecture-guard warning before edits to `engine.py`/`models.py` (informational, never blocks); `post_bash_pytest.py` summarizes pytest output after any Bash command containing `pytest` | `.claude/hooks/*.py` (3 files) | These are advisory/non-blocking by design — don't be surprised if a file you just wrote gets auto-fixed by ruff between your edit and your next read. |
| **`pytest-timeout` thread-based kill can nuke an entire run** | `pytest-timeout` is declared in `tests/requirements-dev.txt` (not in `pyproject.toml`'s `dev` extra, not in root `requirements-dev.txt`) but **no `--timeout` flag is configured anywhere** — not in `pyproject.toml`'s `addopts`, not in `ci.yml`'s pytest invocation. If you manually add `--timeout=N` locally and a subprocess under test hangs, the default `thread` timeout method can leave the whole pytest process in a bad state rather than cleanly failing one test | `rg -n "timeout" pyproject.toml .github/workflows/ci.yml` (should show no `--timeout` usage) | Don't pass `--timeout` for smoke/CLI-driving test runs locally; CI doesn't use it either, so parity is preserved by leaving it off. |
| **Integration fixtures need network neutered, not just keyed** | A dummy `OPENROUTER_API_KEY` satisfies `UnifiedClient.__init__`'s eager check (§5) but a real call will still dispatch over the network and retry for tens of seconds against fake credentials. The working fix in `tests/integration/conftest.py` patches `orch._c.client._dispatch` — **not** `.call()` itself, because patching `.call()` would also swallow the circuit breaker's own fail-fast path that other tests exercise directly | A "fast" integration test that takes 10s+ despite a dummy key is not neutered, just keyed | Patch at the `_dispatch` layer, one level below `.call()`. Full incident narrative and rationale: `orchestrator-debugging-playbook`. |

## 8. When NOT to use this skill

- **What each env var/flag actually *does* at runtime** (not just where it's
  read) → `orchestrator-config-and-flags`.
- **Diagnosing a live failure** (hangs, silent failures, wrong scores) →
  `orchestrator-debugging-playbook`.
- **Why a gate exists / whether you're allowed to weaken it** →
  `orchestrator-change-control`.
- **Historical incident narratives beyond the one-line traps above** →
  `orchestrator-failure-archaeology`.
- **The layering rules themselves (where new code goes)** →
  `orchestrator-architecture-contract`.
- **Running a real project, the dashboard, or the API gateway** →
  `orchestrator-run-and-operate`.
- **Proving a change works (markers, coverage gate, contract tests)** →
  `orchestrator-validation-and-qa`.

## 9. Provenance and maintenance

Re-verify anything below that may have drifted — this file was authored
2026-07-08 against branch `feat/response-healing`.

| Fact | Re-verification command |
|---|---|
| Python version policy | `Select-String -Path pyproject.toml -Pattern "requires-python"`; `Select-String -Path .github/workflows/ci.yml -Pattern "python-version"` |
| `dev` extra contents | `Select-String -Path pyproject.toml -Pattern "^dev = \[" -Context 0,20` |
| grimp pin | `Select-String -Path pyproject.toml -Pattern grimp`; `pip show grimp` (want `3.3.x`) |
| black version drift | `Select-String -Path .github/workflows/ci.yml,requirements-dev.txt,pyproject.toml -Pattern "black=="` |
| requirements.txt `anthropic`/`asyncio` lines | `Select-String -Path requirements.txt -Pattern "anthropic|asyncio"` |
| `instructor` declared in both manifests | `Select-String -Path pyproject.toml,requirements.txt -Pattern instructor` |
| instructor lazy-import pattern still lazy | `Select-String -Path orchestrator/infrastructure/llm_client.py -Pattern "import instructor"` (should be indented / inside a function, plus one `TYPE_CHECKING`-guarded top-level import) |
| Eager `OPENROUTER_API_KEY` check | `Select-String -Path orchestrator/infrastructure/llm_client.py -Pattern "OpenRouter API key not found"` |
| `.env` `override=True` | `Select-String -Path orchestrator/cli.py -Pattern "load_dotenv"` |
| Import-linter contracts (5) | `Get-Content .importlinter | Select-String "\[importlinter:contract:"` |
| `.claude/hooks` inventory | `Get-ChildItem .claude/hooks` |
| pytest-timeout not wired via `--timeout` | `Select-String -Path pyproject.toml,.github/workflows/ci.yml -Pattern "timeout"` |
| BOM incident commit | `git show d913d136 --stat` |
| Coverage floor (local ratchet vs CI floor) | `Select-String -Path pyproject.toml -Pattern "fail_under"`; `Select-String -Path .github/workflows/ci.yml -Pattern "cov-fail-under"` |
