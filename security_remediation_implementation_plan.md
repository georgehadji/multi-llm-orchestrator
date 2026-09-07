# Security Remediation and Hardening Implementation Plan

> **Revision 2 (2026-09-07).** Rewritten after verifying every finding against the
> post-sync tree (master at `b7759097`, 505 commits newer than the audited state)
> and auditing what the repository already provides. All six findings still
> reproduce. Three of them are wiring tasks, not new subsystems. See §2.

## 0. Delivery status (2026-09-07)

| Slice | Finding | State | Commit |
|---|---|---|---|
| T0 | `secure_execution.py` untested | **done** | `46c716a6` |
| T1 | SEC-001a exposure | **done** | `1aff8a12` |
| T2 | SEC-005 path traversal | **done** | `46c716a6` |
| T3 | SEC-003 shell execution | **done** | `3b820994` |
| T4 | SEC-006 permission enforcement | **done** | `3ae363d6` |
| T5 | SEC-002 pickle cache | **done** — module deleted, not re-coded | `5580d037` |
| T6 | SEC-004 SSRF | **done** | `53b83b20` |
| T7–T10 | IDE auth/ownership, key lifecycle, legacy entry point, CI gates | **open** | — |

121 security tests pass. All six import-linter contracts KEPT. Two breaking
changes shipped deliberately: `supervisor:*` is no longer implied by `read` or
`execute` (re-register keys), and `ShellTool` is disabled unless
`ORCHESTRATOR_SHELL_TOOL_ENABLED` is set.

Known unrelated breakage in the tree, pre-existing and untouched by this work:
`tests/unit/` has ~38 failures concentrated in `test_hunt_*`,
`test_orchestrator_construction.py`, `test_engine_run_project.py` and
`test_import_integrity.py` — remote's own in-flight remediation, tracked by
`docs/plans/2026-09-0*`.

## 1. Purpose and outcome

Turn the findings in `security_best_practices_report.md` (SEC-001 … SEC-006) into
a staged, test-driven programme ordered by **risk retired per hour of work**, not
by architectural tidiness.

Target end state:

- The IDE service is loopback-bound by default, authenticated when exposed, and
  ownership-aware for REST and WebSocket operations.
- Cache data is decoded with a constrained, versioned format rather than by
  executing Python object graphs.
- Shell execution is disabled by default and, when enabled, runs argv through an
  allowlist — never a host shell string.
- Outbound URL fetching is policy-controlled against SSRF, redirects, DNS
  rebinding, oversized responses, and unsafe schemes.
- User-controlled filesystem paths are rooted and canonicalized.
- API-key permissions are enforced, not merely recorded.
- Each of the above is pinned by a regression test that fails without the fix.

## 2. What already exists — reuse before building

The previous revision specified new ports, adapters, facades and services for
command execution and filesystem containment. **Most of that already exists** in
`orchestrator/safety/secure_execution.py` and must be reused rather than
reinvented:

| Previously planned | Already in the tree | Status |
|---|---|---|
| `WorkspacePathPort` + path root service (G1) | `SecurePath`, `sanitize_path(base, user_input)` | Verified: blocks `../`, `..\`, absolute paths, null bytes; allows clean names |
| `CommandRequest` / `CommandPolicyPort` (E1) | `SafeCommand`, `SecureSubprocess`, `secure_subprocess_run` | Present; never uses `shell=True` |
| Security error taxonomy (A1) | `SecurityError`, `PathTraversalError`, `CommandInjectionError`, `InputValidationError` | Present |
| Shell-metacharacter denial (E1) | `validate_no_shell_injection(text)` | Present |
| Command risk classification | `orchestrator/safety/command_guard.py` — `classify_command`, `requires_explicit_approval` | Present |

**Consequence:** SEC-003 and SEC-005 collapse from "design and build a subsystem"
to "call the guard that is already here at the site that skips it."

**Caveat that becomes work item T0:** `secure_execution.py` has **no test file**.
It is untested security infrastructure that several fixes below will depend on.
Cover it before leaning on it.

### Corrections to revision 1

- `orchestrator/ports.py` is a 263-byte shim; the real port surface is
  `orchestrator/domain/ports.py`. Target the latter.
- CLAUDE.md rule 4 forbids new root-level `orchestrator/*.py` modules, and
  `scripts/check_root_module_freeze.py` enforces it in CI. New code goes in
  existing subpackages (`domain/`, `application/`, `infrastructure/`, `safety/`).
- Revision 1 proposed a 12-slice sequence that fixed the one **Critical** finding
  sixth, behind four workstreams of architecture. That is backwards; see §3.

### Exposure reassessment

- **SEC-003 is latent, not live.** `ShellTool` is referenced only by
  `orchestrator/tools/__init__.py` and `tests/test_agentic_system.py`. No
  production caller invokes it. Still fix it — an unreachable footgun becomes
  reachable the first time someone registers the tool — but it does not warrant
  a container-runtime programme before the IDE is closed.
- **SEC-001 is live.** `orchestrator/ide_backend/launch.py` is a real launcher
  and is documented in `docs/INTEGRATION.md` and `ide_frontend/README.md`.

## 3. Delivery sequence, ordered by risk retired

Each slice is RED test → GREEN implementation → verification → one focused commit.

### T0 — Cover the guards the rest of the plan depends on
Add `tests/unit/security/test_secure_execution.py` for `SecurePath`,
`SafeCommand`, `SecureSubprocess`, `sanitize_path`, `validate_no_shell_injection`:
traversal (`..`, `..\`, absolute, null byte, symlink escape, Windows case), shell
metacharacters, and the "clean input still works" path. No fix depends on
untested code after this slice.

### T1 — SEC-001a: close the exposure (highest risk-per-line in the plan)
Three edits retire most of the Critical finding:

1. `orchestrator/ide_backend/launch.py:24` — `--host` default `0.0.0.0` → `127.0.0.1`.
2. `orchestrator/ide_backend/server.py:124` — parameter default likewise.
3. `orchestrator/ide_backend/server.py:47-48` — replace `allow_origins=["*"]` +
   `allow_credentials=True` with an explicit origin list, defaulting to the
   loopback dev origins.

Then refuse to start when a non-loopback host is requested without
authentication, gated on an explicit `ORCHESTRATOR_IDE_ALLOW_REMOTE=true`.
Test: startup raises on `host=0.0.0.0` with auth disabled; loopback still starts;
wildcard origin with credentials is rejected.

### T2 — SEC-005: root the legacy file handler
`ide_orchestrator_server.py` builds `Path.cwd() / "ide_outputs" / session_id /
file_path` at **three** sites — `:2245`, `:2642`, `:2919`. The audit named only
`:2919`. Route all three through `SecurePath`/`sanitize_path`. Test `..`,
absolute paths, mixed separators, symlink escape, null bytes.

### T3 — SEC-003: argv allowlist, disabled by default
Rewrite `ShellTool.execute` (`orchestrator/tools/shell_tool.py:35`) to take an
executable from an allowlist plus an argv list, run through `SecureSubprocess`
(`create_subprocess_exec`), and contain `cwd` with `SecurePath`. Default the
capability to disabled; require explicit opt-in to enable. Keep
`classify_command`/`requires_explicit_approval` for the approval path. Never call
`create_subprocess_shell` with model- or user-derived text.

### T4 — SEC-006: enforce the permissions already stored
`_verify_api_key` (`orchestrator/api_server.py:1469`) returns `bool`; make it
return an immutable principal (or a uniform failure), and have `_require_auth`
evaluate a typed permission per route. Keep `hmac.compare_digest`. Default-deny
unknown or malformed permissions. Never treat a request-supplied permission
string as authority.

Route → permission mapping:

| Operation | Permission |
|---|---|
| Health | public |
| Models / stats / status | `read` |
| Execute project or tasks | `execute` |
| Cancel project | `project:cancel` |
| Supervisor reads | `supervisor:read` |
| Supervisor directive | `supervisor:execute` |
| Register / revoke keys | `key:register` or `admin` |
| IDE session create/read/write/delete | corresponding `session:*` |
| WebSocket connect | `session:read`, then per-event permission |

Test: a read-only key cannot execute, cancel, register, or reach supervisor.

### T5 — SEC-002: retire pickle from the cache — **done, by deletion**

The gate this slice carried — *"do this only after enumerating the concrete value
types actually cached"* — was run, and it invalidated the design below. The
enumeration found **no cached types at all**:

- **No importers.** Nothing in `orchestrator/`, `tests/`, or the docs imports
  `orchestrator.infrastructure.caching` or its root shim `orchestrator.caching`.
  Every `@cached` decorator in the repo resolves to `performance.py:360`, an
  unrelated decorator.
- **No working writes.** Every write path raised
  `AttributeError: type object 'datetime.datetime' has no attribute 'timezone'` —
  the module wrote `datetime.now(datetime.timezone.utc)()`, which is wrong twice
  over (`timezone` is a module attribute, not a class attribute, and the result
  is then called). `InMemoryCache.set` raised outright; `DiskCache.set` swallowed
  it into a log line and stored nothing.

So there was no envelope to version, no payload to migrate, and no legacy entry
to quarantine. Both files were deleted, together with the shim's `.importlinter`
exemption. This removes the `pickle.loads` sink instead of guarding it.

Kept as a regression guard:
`tests/unit/security/test_no_pickle_deserialization.py` — an AST scan banning
`{pickle,cPickle,dill,marshal}.{load,loads,Unpickler}` across the package, plus
two tests asserting the deleted modules stay deleted. AST rather than regex so
the benchmark prompt in `analysis/leaderboard.py:366`, which contains the literal
text `pickle.load(f)` inside a string, is not flagged forever. This also
discharges the pickle half of T10.

Not confused with `orchestrator/infrastructure/cache.py` — the live response
cache, a different module, which stores text and never used pickle.

<details>
<summary>Superseded design (kept for the record)</summary>

Introduce a versioned envelope (record type, schema version, timestamps,
JSON-compatible payload) behind a codec seam, with JSON as the default.
Namespace keys with a codec/version prefix so old entries cannot decode by
accident. On encountering a legacy entry: fail closed, delete or quarantine it,
emit a security event. No automatic in-process pickle migration. Ship a
cache-flush command and invalidate during rollout.

</details>

### T6 — SEC-004: SSRF policy on dynamic fetches
`orchestrator/api_builder.py:209` fetches an arbitrary `spec_url`. Add a
validation chain applied before the request and again after every redirect:
strict parse → `https` only by default → reject credentials/fragments/odd
ports → resolve DNS and reject loopback, link-local, multicast, private,
reserved and unspecified addresses → connect to the validated address →
cap redirects, response bytes, decompression ratio and total time.

Prefer local-file import as the default for untrusted workflows. Treat
`servers[].url` (`:574-579`) as untrusted data, not as authorization to call it.
Existing SSRF-adjacent logic in `deployment_feedback.py` and
`generators/website_generator.py` should converge on the same helper rather than
growing a third copy.

Regression tests: localhost, IPv4/IPv6 loopback, private ranges, `169.254.169.254`,
decimal/hex IP forms, redirect-to-blocked, DNS rebinding, userinfo URLs,
oversized responses, decompression bombs, unsupported schemes.

### T7 — SEC-001b: IDE authentication and ownership
With exposure already closed by T1, build the durable fix: a shared
authentication path used by both the REST routes
(`orchestrator/ide_backend/api/routes.py`) and the WebSocket handshake
(`server.py:66-77`), so the two cannot drift. Bind each session to an owner
principal and verify ownership on every read, write, delete, task update, file
access and socket event — never trust a caller-supplied session ID because it
exists. Use UUID4 session IDs. Add message-size, rate, concurrency and idle
limits. Return stable error codes, never raw exception text.

### T8 — Key lifecycle and persistence
Persist keys as a keyed digest (HMAC with a server-side pepper) plus key ID,
principal, permissions, and creation/expiry/revocation timestamps — never the raw
token. Add rotation, revocation, bounded key count per principal, one-time raw
key display, and audit events. Keep an in-memory adapter for tests; legacy
in-memory keys become explicit development-only opt-in.

### T9 — Legacy entry point decision
Decide whether `ide_orchestrator_server.py` is still deployed. If not, remove it
from launch paths and document the migration. If it stays, route it through the
same auth, ownership, path and command policy as the FastAPI server. Do not
maintain a second security implementation.

### T10 — CI and operational gates
Add `pip-audit` (or `safety`) with a documented exception/expiry process. Add
lint rules or tests banning ~~`pickle.load(s)`~~ (**done** — T5's AST guard),
`create_subprocess_shell`, wildcard CORS with credentials, and unguarded
outbound fetches, so fixed findings cannot silently regress. Add secret scanning. Document a production profile: loopback or
private binding behind an authenticated proxy, TLS termination, explicit CORS,
secret-manager injection, isolated Redis and SQLite permissions, audit retention
and alerting on repeated denials. Add a startup posture command that reports
effective security settings without printing secrets.

## 4. Architectural rules

From `AGENTS.md` and CLAUDE.md:

| Concern | Location | Rule |
|---|---|---|
| Principal, permission, security errors | `orchestrator/domain/` | Pure data/enums/exceptions; no I/O, no framework imports |
| Authorization decisions, ownership use-cases | `orchestrator/application/` | Depend on ports/protocols, never concrete infrastructure |
| Key persistence, cache codecs, outbound policy, sandboxes | `orchestrator/infrastructure/` | Concrete adapters only |
| Reusable guards | `orchestrator/safety/` | Where `secure_execution.py` already lives |
| Port contracts | `orchestrator/domain/ports.py` | Protocol-based inversion |
| aiohttp/FastAPI/WebSocket integration | existing API/IDE edge modules | Thin adapters; delegate decisions inward |
| Composition | `orchestrator/engine_core/container.py` | Construct adapters once and inject |

No security policy in `engine.py`. No serialization, filesystem, socket,
subprocess or framework behavior in `models.py` or domain models. No new
root-level `orchestrator/*.py`. Public service boundaries stay async; blocking
DNS/filesystem/process work goes through async APIs or bounded offloading.

Cross-cutting principle — **fail closed**: missing configuration, unknown
permissions, invalid sessions, unavailable policy services and malformed
credentials all deny.

## 5. Security telemetry

Emit structured events for authentication failure, authorization deny, SSRF
block, traversal block, command policy deny, sandbox failure and cache decode
failure. Log principal/key identifiers, reason codes and request IDs — never raw
API keys, bearer tokens, credential-bearing prompts, or command output by
default. Reuse the existing secret-masking filter and add masking regression
tests.

## 6. Verification gates

After each coherent slice, in this order:

1. `black --check orchestrator/ tests/`
2. `ruff check orchestrator/ tests/`
3. `lint-imports`
4. the configured strict `mypy` command
5. targeted security tests, then the non-slow/non-API suite
6. `pytest tests/contracts/ -v --tb=short --no-cov`
7. `bandit -r orchestrator/ --severity-level high --confidence-level medium`
8. dependency and secret scans

Note: `lint-imports` requires the pinned toolchain (`import-linter>=2.1,<2.2`,
`grimp>=3.3,<3.4` per `pyproject.toml`). Newer import-linter crashes on this
codebase with a `rich` nested-live-display error.

Before merging, threat-model review covering: remote IDE exposure, stolen API
keys, prompt injection into tools, malicious cache contents, SSRF through every
dynamic URL, symlink/path manipulation, denial of service, and session isolation.

## 7. Rollout and rollback

Feature flags exist for compatibility, never to weaken a secure default. Order:

- close IDE exposure (T1) — no flag, this is the default change;
- ship auth/authorization primitives and observe denials before enforcing;
- deploy key persistence, then revoke and rotate legacy in-memory keys;
- ~~invalidate pickle caches *before* enabling the new codec~~ — moot: the cache
  never stored anything, so there is nothing to invalidate (see T5);
- ship command execution disabled by default;
- enable SSRF and path policies in blocking mode once coverage is complete;
- remove legacy handlers and compatibility flags after one release cycle.

Rollback must never restore remote-by-default binding or pickle loading. If a
policy causes operational problems, roll back the consumer or switch it to a safe
deny/disabled mode while retaining authentication, ownership and validation.

## 8. Definition of done

- Every SEC-001 … SEC-006 regression test passes, and each fails without its fix.
- No production path pickle-deserializes cache data.
- No remotely exposed service starts without authentication and explicit origins.
- Principal ownership and permissions are enforced on every API and IDE action.
- Command execution is disabled or argv-allowlisted with resource limits.
- Every dynamic outbound fetch goes through the SSRF policy.
- Every user/model-controlled path goes through canonical root checks.
- `secure_execution.py` is covered by tests (T0).
- CI runs security, dependency and secret checks with documented exceptions.
- Architecture contracts, mypy, formatting, lint and the required suites pass.
- Deployment documentation and rotation procedures are updated.
