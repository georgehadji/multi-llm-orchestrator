# T18 — Silent-failure sweep at scale — Inventory

Second wave of `docs/hunts/BACKEND_DEPTH_PASS_PLAN.md` (T17–T24). Like T17, a
detector-first wave whose Phase-1 surface is an exhaustively enumerated pattern
instance set rather than a sampled file list.

## Phase 0 — Scope

**In scope:** every broad exception handler in `orchestrator/` — `except:`,
`except Exception:`, `except BaseException:`, and tuples containing them.

**Threat model:** a handler that swallows a failure and lets the caller believe
nothing went wrong. The severe variant is *fail-open* — a validator, scanner or
health probe that reports **success because it failed**. This shape produced
confirmed defects in T6, T8, T9, T13 and T16.

**Plan estimate vs. measured.** The plan projected "923 broad-except sites, 77
with an immediate pass/continue", derived from `grep` plus a one-line lookahead.
The AST census measures **916** handlers, of which **82** are a bare
`pass`/`continue`. Both plan figures were close but neither was exact; the AST
numbers supersede them.

## Phase 1 — Surface map (exact)

Every handler was classified by what it actually does with the failure:

| Disposition | Count | Meaning |
|---|---:|---|
| `logged` | 627 | logs or prints — failure is surfaced |
| `silent_return` | 98 | returns without logging |
| `SILENT_pass` | 75 | body is exactly `pass` |
| `reraise` | 67 | re-raises — failure propagates |
| `silent_other` | 42 | acts, but does not log |
| `SILENT_continue` | 7 | body is exactly `continue` |
| **total** | **916** | **222 silent** |

The 222 silent handlers were ranked by adjacency, using the T6/T16 severity
scheme (config-loading graceful degradation is deliberately *not* elevated, per
T16's explicit disposition):

| Tier | Count | Category |
|---|---:|---|
| 1 | 7 | money — budget, cost, charge, spend, token accounting |
| 2 | 46 | validation gate — validate, verify, scan, check, secret, auth, policy |
| 3 | 24 | persistence/state — save, load, commit, checkpoint |
| 4 | 3 | config-load (not elevated, per T16) |
| 5 | 142 | everything else |

Tier 1 was audited exhaustively; tier 2 was audited by mechanical fail-open
detection plus reading of the highest-risk sites.

## Phase 2–4 — Candidates and verdicts

### The headline is a negative result

Two mechanical sweeps looked for the **fail-open** shape across all 916
handlers — a handler that neither logs nor re-raises, returning either:

1. a bare `True`, or
2. a result object built with `passed=True` / `success=True` / `ok=True` /
   `valid=True` / `healthy=True` / `available=True`.

**Both sweeps returned zero.** No handler anywhere in `orchestrator/` reports
success because it failed. The high-severity variant of this pattern is absent,
which is consistent with T6/T8/T9/T13/T16 having fixed the instances that
existed. This is the wave's most important finding and it is a negative one.

### C1/C2 — `website_validator.py`'s rate-limit and auth-flow scans — **VERIFIED DEFECT, FIXED**

`_check_rate_limiting` (scans ≤50 files) and `_check_auth_flow` (≤20) each
`continue` past any file they cannot read, with no log and no record.

This is the same file T8's C5 fixed — but a materially **less severe** variant,
and the difference matters:

| | T8 C5 (secret scanner) | T18 C1/C2 |
|---|---|---|
| Unreadable file leaves | `leaks` empty | `found = False` |
| Reported result | **passed** ("no secrets found") | **failed** |
| Direction | fail-**open** — a security hole | fail-**closed** — safe |
| Real harm | a scan claiming clean without looking | a misreport: says "no rate limiting found" when the truth is "could not read the files" |

**Trigger:** patch `Path.read_text` to raise for the scanned suffixes, run each
check against a fixture tree; pre-fix, `details` reads *"No rate limiting found.
Contact forms and registration endpoints must include IP-based rate limiting."*
with no indication anything was skipped. FIRED.
**Innocence:** none for the misreport — the result is indistinguishable from a
genuine negative; but the *severity* defence holds, so this is not a security
finding. Severity **LOW**: it sends a developer to add protection that may
already exist, and it silently reduces the coverage of a check gating the
documented `orchestrator website --min-quality` / `--require-all-checks` flags.

**Fix:** log the read failure and append *"N file(s) could not be read and were
skipped, so this scan is incomplete"* to `details` — the pattern T8 C5 already
established in this same file, applied to its two remaining siblings.

### Candidates investigated and CLEARED

Three plausible-looking tier-1/tier-2 candidates were killed by the innocence
attempt. Recording them prevents re-raising:

- **`cost.py:504` `_static_estimate` — FALSE (innocent).** `except (KeyError,
  Exception): return 0.0` in a *cost estimator* looked like it would make
  unknown-cost models appear free and therefore always "cheapest". Both halves
  of that hypothesis are wrong: `estimate_cost` cannot raise (it uses
  `COST_TABLE.get(model, {...})` with a default), and `cheapest_model`
  explicitly **filters out** zero-cost candidates (`[m for m in candidates if
  self.predict(...) > 0]`) rather than preferring them. The docstring documents
  0.0 as a sentinel callers must treat as "no prediction". The
  `except (KeyError, Exception)` tuple is redundant (`KeyError ⊂ Exception`) but
  behaviourally inert.
- **`safety/code_executor.py:168` `_is_sandbox_available` — FALSE (innocent).**
  A silent `except Exception: return False` in a *sandbox availability probe*,
  whose caller falls through to `_execute_local` (commented "insecure"), looked
  like a sandbox bypass. The guard above it handles exactly this: with
  `require_sandbox` and no sandbox, it either returns a blocking error result
  (`fail_if_sandbox_unavailable=True`) or logs `"Executing code without sandbox
  - security risk!"` and records a security warning. Fail-closed by default with
  an explicit, logged opt-out. Swallowing the probe error is correct — "cannot
  reach Docker" genuinely means "not available".
- **`infrastructure/state.py:401` `save_checkpoint` — FALSE (innocent).** The
  flagged `except Exception: pass` wraps `await db.rollback()` *inside* an outer
  handler that re-raises. A rollback that itself fails must not mask the
  original exception. Textbook-correct.

### Recorded, not elevated

- **`control_plane.py:288` `_write_audit`** — `except Exception: pass` with the
  comment *"audit failures must never break the main flow"*. Documented,
  deliberate, and the "audit record" is a `logger.info` line rather than a
  durable store. Per T13's lesson (check for a pre-existing comment documenting
  a gap as a known tradeoff before calling it a discovery), recorded only.
- **`infrastructure/streaming_resilient.py:145` `get_usage_percent`** — returns
  a fabricated `50.0` ("assume moderate") when the memory probe fails, feeding
  backpressure decisions. Same shape as T6's fabricated-neutral-score finding,
  which was left `[REQUIRES HUMAN REVIEW]`; treated consistently here.
- **`cost_optimization/batch_client.py:397`** — polling loop swallows
  `batches.retrieve` errors; bounded by an explicit `TimeoutError` above, so it
  cannot hang, but a persistent API error is indistinguishable from "not yet
  done" until timeout. Minor.

### Detector precision — an honest caveat

The `SILENT_pass`/`silent_return` classifier has a substantial false-positive
rate for *severity*: it flags correct rollback guards, best-effort cleanup, and
documented-deliberate swallows. That is why the shipped gate does **not** police
"silence" in general — 222 handlers would be a meaningless CI signal — but only
the fail-open shape, which is unambiguous and currently at zero.

## Deliverable — `scripts/check_silent_failure.py`

Fails when a broad handler that neither logs nor re-raises returns an
affirmative value. Reports the count of handlers examined so the denominator is
visible. Self-tested three ways: it catches `return True`, catches
`return R(passed=True)`, and correctly does **not** flag a handler that logs
before returning True.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3. `fix_revisions`: 0 — both fixes correct on first pass,
RED→GREEN on first attempt. Three candidate hypotheses were falsified during
Phase 3 before any fix was written (cost estimator, sandbox probe, checkpoint
rollback) — the innocence attempt did most of this wave's work.
