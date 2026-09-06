# T20 coverage and residual risk

## Counters

| | |
|---|---|
| `hunt_iterations` | 1 |
| `fix_revisions` | 2 (detector rewritten twice before its answer was trusted; lazy import replaced with a top-level one) |
| `budget_spent` | 1 source file changed (7 lines), 1 test file (3 tests), 1 gate script |

## What was actually examined

**Fully examined:** every `add_argument` declaration (109), every
`BaseSettings` field in `crosscutting/config.py` (74 across both classes).
Each of the 20 unread fields was individually checked for an indirect
consumer — `os.getenv` on the matching env var, dynamic `getattr`, or a
string key — before being called dead. That check reclassified seven of them
as innocent (C6, C7).

**Not examined:** config read from anywhere other than
`crosscutting/config.py`. Other modules define their own dataclasses and read
their own env vars; `OrchestratorSettings.env_str` exists precisely because
`os.getenv` is scattered. A dead setting living outside this one file is
invisible to both the census and the gate.

**Not examined:** whether a flag that *is* read is read *correctly* — at the
right time, on the right code path, or by all the consumers that should honour
it. The census answers "does anything read this", not "does the right thing
read it". `bilevel_level15_enabled` (C3) is the visible edge of that gap: its
feature is wired by DI while its flag is not read at all, and nothing here
proves the two agree.

## Residual risk

1. **The 19 remaining unread fields are frozen, not fixed.** The gate stops
   the count growing; it does not shrink it. Twelve `OrchestratorSettings`
   fields still accept an `ORCH_*` value and discard it silently.
2. **C5 is the item most worth a human decision.** `dashboard_host` says
   `127.0.0.1`; `dashboard_core/core.py` binds `0.0.0.0`. Whether to change
   the bind or the setting is a product call with a security dimension, so
   nothing was changed unilaterally — but the config file currently
   misdescribes what the dashboard does, and a reader could reasonably
   believe the dashboard is loopback-only when it is not.
3. **The gate's "read" test is a substring match.** A field named `port` or
   `host` would be considered read by almost any file. It is calibrated for
   the descriptive names this config uses and would be weak against short
   generic ones.
4. **`extra="ignore"` is unchanged.** Even with every field wired, a
   *misspelled* `ORCH_*` variable is still accepted in silence. Switching to
   `extra="forbid"` would catch typos but is a behaviour change beyond this
   wave.
5. **The rerank path is proven wired, not proven good.** The tests show the
   flag reaches `find_similar`; no run against a live reranker was made, so
   whether stage-2 reranking actually improves recall is UNKNOWN here.

## Claims NOT made

- Not claimed: that every control surface in the codebase is wired. Only that
  the argparse surface and `crosscutting/config.py` were enumerated.
- Not claimed: that the 12 unread `OrchestratorSettings` fields are harmless.
  They were escalated, not dispositioned.
- Not claimed: that `engine_core/tabu_search.py` should be deleted. It is
  unreferenced today; whether it is abandoned or staged work is the
  maintainer's call.
