# T22 coverage and residual risk

## Counters

| | |
|---|---|
| `hunt_iterations` | 1 |
| `fix_revisions` | 2 (the empty-evidence probe's 24 "defects" discarded once the auditor's gate was found; G7's fix narrowed from "declare SCRIPTS" to the ambiguous case only, after the static-site cases showed declaring it would lose real verdicts) |
| `budget_spent` | 2 source files changed (~14 lines), 1 test file (5 tests), 1 gate script |

## What was actually examined

**Fully examined, mechanically:** all 82 WF-100 check implementations, on two
dimensions — behaviour under empty evidence (all 82 run, 0 errors), and
declared `requires` versus the `ev.*` attributes each body reads. That is a
complete sweep of `checks.py` for *this shape*, and the gate now holds it.

**Read closely:** the 4 mismatch candidates (A9, D9, E8, G7) in full, the 4
division sites and their guards, `auditor.py`'s dispatch, `SiteEvidence` and
its `available` property, and `_phase_jury_verify_and_meta_eval` /
`_phase_jury_weighted_ranking` in `ara_pipelines.py`.

**Not read:** the overwhelming majority of both regions. `ara_pipelines.py` is
4,288 lines of which perhaps 150 were read. `checks.py`'s 82 check *bodies*
were analysed by AST for evidence reads but only 4 were read as logic — the
other 78 could each be wrong about their own subject matter and this wave
would not know. `detectors.py`, `evidence.py`, `render.py`, `report.py`,
`standard.py`, `auditor.py` (beyond dispatch), `ara_execution_strategy.py`,
`ara_integration.py`, `brain.py` and `brainstorming.py` were not audited.

**Roughly 10,000 of the 10,571 lines in scope remain unread.** T22 hunted two
shapes across them exhaustively; it did not audit the regions.

## Residual risk

1. **C2's wasted LLM call is still being paid for.** The fix removed the dead
   statement and the false comment, not the spend. Every run of this pipeline
   phase still makes a `max_tokens=1500` call whose output nothing consumes.
   This is the item most worth a decision — wire the weighting, or delete the
   call.
2. **The gate maps attributes to evidence kinds by name.** `ev.pages` ⇒ MARKUP
   and so on. A check that reaches evidence some other way — through a helper
   the allowlist does not know, or off an object passed in — is invisible to
   it. `_content_pages` and `_images` are handled explicitly; a new helper
   would need adding.
3. **"Guarded" is a human judgement, not a machine one.** The allowlist records
   *why* each of the four reads is safe; nothing re-checks that claim. If A9's
   `if ev.http is not None` were removed, the gate would still pass.
4. **The 24 empty-evidence PASSes were dismissed on the strength of the
   auditor's gate.** That is correct for `audit()`, the only caller found. Any
   other caller invoking implementations directly would get those vacuous
   passes — the probe showed exactly what that looks like.
5. **The two fabricated neutral defaults (C7) remain.** A 5.0 average from no
   scores and a 0.5 final score from no claims are indistinguishable
   downstream from genuinely middling results.

## Claims NOT made

- Not claimed: that `checks.py` is correct. 78 of 82 checks were never read as
  logic — only their evidence declarations were verified.
- Not claimed: that `ara_pipelines.py` is sound. ~4,100 of its lines are
  unread, including all pipeline state transitions apart from the one phase
  pair examined for C2.
- Not claimed: that G7 is now right about form protection in general. It is
  now right about *not asserting* what it could not see.
- Not claimed: that the meta-evaluation weighting should exist. Only that the
  code claimed it did, and paid for data to do it with.
