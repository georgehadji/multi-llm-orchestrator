# P1 — coverage & residual risk

**Surface audited:** 8 files, 13,786 LOC (measured pre-fix) — the DEEP tier's first wave
per `docs/hunts/PRECISION_AUDIT_V4_WAVE_PLAN.md` §5. Not skimmed: every file was read in
full or by targeted section pass (large template-generation bodies in
`ide_orchestrator_server.py` and `first_generator.py` were sampled at their highest-risk
methods — process/port lifecycle, output parsing — rather than read line-by-line end to end).

**Defect classes actually hunted, per V4's taxonomy:**
- Concurrency (unsynchronized shared state, TOCTOU) — hit twice (P1-7, P1-3), a third
  candidate (P1-6) is class-level shared state, adjacent to this category.
- Logic (silent failure, wrong comparator, undefined name) — hit four times (P1-1, P1-2, P1-4, P1-8).
- Dependencies/Config (dead flag) — hit once (P1-2, doubles as Logic).
- Injection/Taint — checked (subprocess argv construction in `ide_orchestrator_server.py`,
  already argv-list not shell-string; `check_shell_injection.py` confirms clean) — none found.
- Memory/Resource — checked (`tempfile.TemporaryDirectory` context-manager usage in
  `first_generator.py`; async cleanup paths in `engine.py::_cleanup_resources`) — none found,
  both correctly guarded.
- Edge cases (off-by-one, encoding) — not specifically swept beyond what surfaced incidentally.

**Clean-claim scope:** the 8 P1 files were audited for the classes above with 8 VERIFIED/
well-mechanized defects found and fixed, and 6 additional candidates cleared by direct
evidence (recorded in `inventory.md`'s Cleared section). No claim is made about classes not
listed, or about the ~85% of these files' combined logic not read in this pass (the
multi-thousand-line template-generation bodies in `ide_orchestrator_server.py` and
`first_generator.py` in particular).

**Runtime-dependent set:**
- P1-7 (api_server.py budget race) is HYPOTHESIS-grade: the full mechanism is traced
  statically (constructor injection, confirmed-live read at `engine.py:860,897`, unguarded
  mutation, fire-and-forget background execution), but forcing the actual interleaving of two
  real concurrent HTTP requests was not attempted in this wave. What would raise confidence:
  a live two-request integration test against a running `APIServer` in long-running-orchestrator
  mode, timed to land the second request's dispatch mid-execution of the first.
- P1-3's "false success" consequence (the TOCTOU loser's process dying between spawn and the
  `await asyncio.sleep()` check) is HYPOTHESIS-grade for the same reason — the *fix* (liveness
  check via `process.poll()`) is unconditionally correct and tested, but the specific race
  timing that makes it matter was not forced live.

**Highest-value next step:** P2 (next wave per the wave manifest, priority 7-8,
`slack_integration.py`/`models.py`/`nash/infrastructure_v2.py` among others) — or, if a
narrower target is preferred, the ~2,700 lines of `ide_orchestrator_server.py`'s own
route/handler bodies this wave did not read (P1-3 covered only `SessionManager`; the
FastAPI route handlers themselves are still unaudited).

## Uncertainty acknowledgment

**Most likely false positive:** none of the 8 — each has either VERIFIED-EXEC evidence
(P1-1's log check, P1-4, P1-5, P1-6, P1-8 all directly reproduced) or a fully-traced static
mechanism corroborated by an independent source (P1-4/P1-5 additionally confirmed by mypy's
own isolated-diff output, unprompted). If one is weakest, it is P1-2: the *fix* (visibility)
is unambiguous, but "should this actually be wired to `AutonomyConfig.from_agent_profile`"
remains a live product question this wave deliberately did not resolve.

**Most likely missed real defect:** the un-read majority of `ide_orchestrator_server.py`'s
~1,100-line `generate_premium_website` template body and the FastAPI route handlers
downstream of `SessionManager` — this wave read the session/process-lifecycle surface, not
the request-handling surface built on top of it.

**Tail coverage:** Toggle C (tail sweep) ran on all 8 DEEP-tier files; no atypical-class
candidate from that pass survived the Innocence Check independently of the 8 listed above
(the TOKEN_BUCKET and `Item`/`{{item_id}}` checks were tail-adjacent probes that cleared).

**Cannot be determined statically:** the real-world frequency of P1-7's race (depends on
actual concurrent request volume against a long-running-orchestrator deployment, which this
audit has no production telemetry for) and of P1-3's collision (depends on how often two
same-stack IDE sessions actually run concurrently in practice).

**Input that would most raise confidence:** production request logs or telemetry from an
`APIServer` instance actually run in long-running-orchestrator mode, to confirm whether
concurrent overlapping requests are a realistic load pattern or a theoretical one.
