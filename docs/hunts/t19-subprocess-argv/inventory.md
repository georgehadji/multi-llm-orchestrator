# T19 — subprocess/exec argument construction

Wave 19 of the depth pass. Shape hunted: a command string **built by
interpolation** and then handed to a shell, so that a value from outside the
file becomes shell syntax rather than data.

## Phase 0–1 — census

AST enumeration of every process-spawning call under `orchestrator/`:

| API | call sites |
|---|---|
| `subprocess.*` (argv form) | 71 |
| `asyncio.create_subprocess_exec` | 25 |
| `compile`/`exec`/`eval` (raw) | 7 |
| `asyncio.create_subprocess_shell` | 4 |
| `os.system` | 1 |
| **total** | **108 across 43 files** |

Only the last three rows can shell-interpret, so the sweep narrowed to those
**8 sites**. The argv-form calls cannot inject by construction — that is the
whole point of argv — and were not read line by line. That is a deliberate
scope limit, recorded in `coverage.md`.

## Phase 3–4 — triage

| Site | Verdict |
|---|---|
| `nexus_search/server_manager.py:177` (`start`) | **VERIFIED DEFECT — FIXED** |
| `nexus_search/server_manager.py:314` (`stop`) | **VERIFIED DEFECT — FIXED** |
| `scripts/utils/push_to_github.py:113` | **VERIFIED DEFECT (low) — FIXED** |
| `commands/nash.py:117` | FALSE — both branches literal, `os.system("cls" if nt else "clear")` |
| `dev_server.py:124,152,161` | FALSE — hardcoded `ProjectType` table; the only interpolated value is a port already validated as an int in 1..65535 |
| `safety/sandbox_executor.py:76` | FALSE — runs the caller's own `test_command`; that is the API's purpose |
| `tools/shell_tool.py:35` | FALSE — running a shell is the tool's entire purpose |
| `scripts/git/*.py` (4 sites) | FALSE — every interpolated value comes from a hardcoded literal list |
| bandit `# nosec ID — prose` comments repo-wide | **FALSE — hypothesis falsified**, see below |

### The defect

`NexusServerManager.start()` and `.stop()` built:

```python
cmd = f"{self._docker_compose_cmd} -f {self.compose_file} up -d"
process = await asyncio.create_subprocess_shell(cmd, ...)
```

`_docker_compose_cmd` is hardcoded (`"docker compose"` / `"docker-compose"`),
but `compose_file` is the constructor's caller-supplied parameter
(`self.compose_file = Path(compose_file)`). Two independent failures follow:

* **By accident** — a path containing a *space* splits into two arguments.
  No malice needed; `~/my projects/compose.yml` is enough.
* **On purpose** — a path containing `;` runs whatever follows as its own
  command.

Both were demonstrated, not argued. Running the test suite against the
**unmodified production code** (RED pass) made docker itself report the
argument it had received:

```
Failed to start server: unknown docker command: "compose projects/docker-compose.yml"
```

and the injected `touch` created its marker file `/tmp/t19_pwned`. The
`stop()` path left a stray file named `down` in the working directory —
the tail of its own shredded command line.

Severity: this is arbitrary command execution, but reachable only by whoever
supplies `compose_file`, which today is either the default or an operator
argument. It is a latent hole rather than a remotely-triggerable one — the
same disposition as T8's fail-open finding, not above it.

### The script defect

`scripts/utils/push_to_github.py:113` built `f"git push origin {branch}"`
with `shell=True`, where `branch` comes from `git branch --show-current`.
Verified with `git check-ref-format --branch` that git **accepts** `;`,
`$()`, backticks, `&&` and `|` in branch names — only the space is rejected.
So a checked-out branch named `main;id` executes `id`. Low severity
(maintainer-local, requires running a dev script on a hostile branch), but
real, and the fix is two lines.

The other four `scripts/` shell sites interpolate only hardcoded literals and
were left alone; `scripts/` is outside the black/ruff/CI enforced scope and
reformatting it would be churn.

### Falsified hypothesis — bandit nosec scoping

Bandit emits `WARNING Test in comment: <word> is not a test name or id` for
every prose word in a `# nosec B602 — reason` comment. Hypothesis: the prose
defeats the targeting and silently degrades it to a **blanket** nosec that
would hide unrelated future findings.

Probed directly:

| comment | result |
|---|---|
| `# nosec B999` (no valid id at all) | suppressed → blanket fallback |
| `# nosec B324` (valid id, does not match the finding) | **reported** |
| `# nosec B324 — prose here` | **reported** |

Targeting survives the prose; the fallback happens only when bandit finds no
valid id whatsoever. The warnings are cosmetic. **No defect — no change made.**

## Phase 5 — fix

Both `server_manager.py` sites now build argv and call
`create_subprocess_exec`:

```python
argv = [*self._docker_compose_cmd.split(), "-f", str(self.compose_file), "up", "-d"]
process = await asyncio.create_subprocess_exec(*argv, ...)
```

`_docker_compose_cmd.split()` is correct precisely because that value *is*
hardcoded: it is the one token in the line that may be split, and splitting
`"docker compose"` into two argv entries is what execing it requires.

Three files, 5 + 5 changed lines. No new abstraction, no wrapper.

## Phase 7 — proof

`tests/unit/test_hunt_t19_subprocess.py`, 5 tests. RED verified by stashing
the fixed sources: 4 of 5 failed for the predicted reason (empty argv — the
exec call was never reached because the code went to the shell instead), and
the 5th is a pure shell-semantics assertion that correctly passes either way.

The committed test's injected payload is `echo`, deliberately inert: if this
fix is ever reverted, the RED path reaches a real shell again, and it must
not be able to touch the filesystem to prove the point.

## Phase 8 — gate

`scripts/check_shell_injection.py` fails any `create_subprocess_shell`,
`shell=True`, or `os.system` under `orchestrator/` whose command is not a
literal (constants, literal concatenation, `IfExp` of literals, and
`{}`-free f-strings all count as literal). Three reviewed sites are
allowlisted with their reason in the source.

Verified to actually catch the bug, not merely to pass: re-run with the fix
stashed, it reports both `server_manager.py` lines and exits 1.
