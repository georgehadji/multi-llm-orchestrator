# T19 coverage and residual risk

## Counters

| | |
|---|---|
| `hunt_iterations` | 1 |
| `fix_revisions` | 1 (test payload made inert after the RED run showed it touching the filesystem) |
| `budget_spent` | 3 files changed, 10 source lines, 1 test file (5 tests), 1 gate script |

## What was actually examined

**Fully examined (every site read):** the 8 shell-interpreting call sites
under `orchestrator/` — the only places in shipped code where an interpolated
value can become shell syntax. Plus the 5 `shell=True` sites under `scripts/`.

**Examined mechanically, not read:** the remaining 100 process-spawning call
sites (71 `subprocess.*` argv-form, 25 `create_subprocess_exec`, 4 others).
They were enumerated by AST and classified, not reviewed line by line. This
is defensible for *this* shape — argv-form calls cannot shell-inject by
construction — and **not** defensible for any other. An argv call site can
still pass an attacker-controlled *executable path*, pass `cwd` or `env` from
untrusted input, or leak output; none of that was hunted here.

**Not examined at all:** the 7 raw `compile`/`exec`/`eval` sites. They are a
different shape (code injection, not argument injection) and belong to their
own wave. Counting them as covered here would be false.

## Residual risk

1. **`env=`/`cwd=` provenance is unaudited.** The fix stops the *command*
   from being shell-interpreted; it says nothing about where a subprocess
   runs or what environment it inherits. UNKNOWN whether any site passes a
   caller-controlled `cwd`.
2. **The gate scans `orchestrator/` only.** `scripts/` is deliberately out of
   scope — it is outside CI's black/ruff enforcement too — so a new
   `shell=True` in a dev script will not be caught. One such defect was found
   and fixed by hand this wave; another could be introduced without the gate
   noticing.
3. **The gate's notion of "literal" is syntactic.** A module-level
   `CMD = "docker compose"` referenced by name reads as non-literal and would
   fail the gate (conservative, safe). Conversely a literal that is
   *formatted later* by something the AST walk does not follow could slip
   through. No such case exists today; that is an assertion about the current
   tree, not a proof about all future ones.
4. **`dev_server.py` was cleared by reading, not by test.** Its safety rests
   on the `ProjectType` command table staying hardcoded and the port staying
   int-validated. Nothing enforces either; the gate allowlists the file
   wholesale. If someone makes those commands user-configurable, the
   allowlist entry becomes a lie. **This is the weakest link in the wave.**
5. **No runtime assertion.** The fix is structural (argv), so a regression
   is caught at test/gate time, not at runtime. There is no defence in depth.

## Claims NOT made

- Not claimed: that `orchestrator/` is free of command injection. Only that
  the 8 shell-interpreting sites were each examined and dispositioned.
- Not claimed: that the argv-form call sites are correct. They were not read.
- Not claimed: that the fixed `docker compose` invocation works end to end
  against a real docker daemon. No docker daemon was exercised; the tests
  stub the exec syscall and assert on argv.
