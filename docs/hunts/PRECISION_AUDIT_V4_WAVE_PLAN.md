# Precision Defect Auditor V4 — backend wave plan

Measured 2026-09-06 against `claude/llm-orchestrator-website-factory-luk61w` @ `5d4017f`.
Wave assignments are generated: `python scripts/plan_v4_waves.py` → `docs/hunts/v4_waves.tsv`.

---

## 1. Why this is not a repeat of T0–T22

V4 and the V7 protocol that produced T0–T22 sweep **orthogonal axes**, and the
distinction decides whether this plan is worth running at all.

| | V7 (T0–T22, shipped) | V4 (this plan) |
|---|---|---|
| Unit of work | a **defect shape** | an **audited file** |
| Question asked | "where else does this shape occur?" | "what could be wrong in *this* file?" |
| Exhaustive over | the shape, across all 890 files | the taxonomy, within the audited file |
| Output | a CI gate that freezes the shape | a Fix Package per finding |
| Blind to | any shape nobody thought to sweep | any file below the budget line |

T0–T22 never claimed unit coverage and said so explicitly — T22's own coverage
statement records **"~10,000 of the 10,571 lines remain unread"**, having verified
only the evidence declarations of 78 of 82 checks. The 77 defects found were
found *by shape*. V4 is the complementary instrument: it reads a file and asks
what is wrong with it, which is how you find the defect whose shape nobody
anticipated.

**INFERENCE, not established:** that the unit axis still yields at the rate the
shape axis did. The shape sweeps have already harvested the seven recurring
shapes; V4's yield per wave is genuinely UNKNOWN until P1 runs. Treat P1 as a
calibration wave and re-decide after it.

---

## 2. Measured surface

890 Python files, 242,921 LOC under `orchestrator/`.

| Region | Files | LOC | Sink files | Churn (master, last 400) |
|---|---|---|---|---|
| root kernel `orchestrator/*.py` | 257 | 63,817 | 47 | 806 |
| `generators/` | 35 | 21,469 | 8 | 92 |
| `infrastructure/` | 53 | 12,978 | 19 | 84 |
| `operations/` | 33 | 10,829 | 5 | 83 |
| `application/` | 45 | 10,181 | 4 | 129 |
| `engine_core/` | 47 | 9,235 | 3 | 100 |
| `design/` | 33 | 8,799 | — | 70 |
| `safety/` | 22 | 7,949 | 12 | 28 |

Priority distribution (exposure + mutation + complexity + churn, each 0–3):

```
prio 12: 1    prio 8: 10   prio 5: 47   prio 2: 139
prio 10: 3    prio 7: 32   prio 4: 111  prio 1: 416
prio  9: 3    prio 6: 35   prio 3: 93
```

**416 of 890 files (47%) score 1.** 269 tail files are under 50 LOC — mostly
backward-compat shims. Auditing those at the same depth as the top 84 spends
half the budget on the least likely half of the codebase.

---

## 3. The constraint that decides everything

`SCAN_BUDGET = 40 files or 15k LOC, whichever first`. Packed greedily against
both caps:

| Coverage | Files | LOC | Waves | Binding cap |
|---|---|---|---|---|
| **everything** | 890 | 242,921 | **27** | mixed (15 file, 12 LOC) |
| prio ≥3 | 335 | 164,536 | 13 | mostly LOC |
| prio ≥4 | 242 | 134,285 | **10** | mostly LOC |
| prio ≥5 | 131 | 94,350 | 7 | LOC only |
| prio ≥6 | 84 | 63,533 | 5 | LOC only |

Two consequences worth stating plainly:

1. **Full coverage costs 27 waves and no configuration change avoids that.**
   `SCAN_BUDGET` is a *reading* budget. Coarsening the audit unit from file to
   module reduces elicitation cost (K candidates per unit) but not reading cost,
   so it does not reduce wave count. The only lever on wave count is reading
   fewer files.
2. **In the high-risk tiers, LOC binds, not files.** Priority ≥5 averages 720
   LOC/file against a backend mean of 273. Waves there are 9–25 files, not 40.

---

## 4. Three options

**A — Full literal sweep. 27 waves, all 890 files, unit = file.**
Buys the only unqualified coverage claim available. Costs 27 heavy waves;
at K=5–8 per unit that is 4,450–7,120 candidates each needing an Innocence
Check, and roughly half of them spent on priority-1 shims. The dedup burden
against `INVENTORY.md` (873 lines and growing) compounds every wave.

**B — Risk-tiered. 10 waves, priority ≥4 (242 files, 55% of backend LOC).**
Concentrates on the measured exposure×mutation×complexity×churn mass. The
un-audited 648 files are *declared*, which V4's Phase 3 requires anyway — an
honest scoped clean-claim, not a silent gap. Misses, by construction, any
defect in the tail.

**C — Two-speed within B. 5 DEEP waves + 6 STD waves.** ← **recommended**
Same 242 files as B, but the instrument is matched to the risk:

| Tier | Waves | Files | LOC | Config |
|---|---|---|---|---|
| DEEP (prio ≥6) | P1–P5 | 84 | 63,533 | `K=8`, `TOGGLE_B=ON`, `TOGGLE_C=ON` |
| STD (prio 4–5) | P6–P11 | 158 | 70,752 | `K=5`, `TOGGLE_B=ON`, `TOGGLE_C=OFF` |
| TAIL (prio ≤3) | — | 648 | 108,636 | not audited; declared in every Phase 3 |

C beats A on cost-per-defect and beats B on precision where it matters. It
costs one wave more than B — 11 against 10 — because packing the two tiers
separately leaves a part-full wave at the boundary (P5, 16 files), plus the
extra token spend of Toggle C on 84 files.

**Why the tail is not naked.** It carries no *unit* audit, but it is not
unexamined: T18 swept all 916 broad exception handlers, T19 every subprocess
site, T20 every declared config field, T17 all 59 duplicate pairs — across
100% of files, tail included — and eleven CI gates run on every commit. The
correct claim is "no unit audit below priority 4", not "unaudited".

---

## 5. Wave manifest

Generated into `docs/hunts/v4_waves.tsv`; regenerate with
`python scripts/plan_v4_waves.py`, verify with `--check`.

| Wave | Tier | Files | LOC | Priority band |
|---|---|---|---|---|
| P1 | DEEP | 8 | 13,786 | 12–8 |
| P2 | DEEP | 16 | 14,799 | 8–7 |
| P3 | DEEP | 25 | 14,664 | 7 |
| P4 | DEEP | 19 | 14,994 | 6 |
| P5 | DEEP | 16 | 5,290 | 6 |
| P6 | STD | 9 | 14,235 | 5 |
| P7 | STD | 30 | 14,997 | 5 |
| P8 | STD | 26 | 14,983 | 5–4 |
| P9 | STD | 30 | 14,890 | 4 |
| P10 | STD | 40 | 9,655 | 4 |
| P11 | STD | 23 | 1,992 | 4 |

DEEP is dominated by the root kernel (24 files) and `infrastructure/` (12) —
the two regions with the highest churn and sink density in the repo.

---

## 6. Configuration — three calls that are yours, not mine

```
MODE:                   AGENTIC-REPO      # tools + working dir present
EXECUTION:              AUTO              # runner exists; VERIFIED-EXEC reachable
APPLY_FIXES:            ???               # ← decision 1
SCAN_BUDGET:            40 files / 15k LOC (V4 default, unchanged)
DIVERSITY_ELICITATION:  ON
ELICITATION_K:          8 (DEEP) / 5 (STD)
TOGGLE_B_INNOCENCE:     ON                # ← decision 2 (recommended ON)
TOGGLE_C_TAIL_SWEEP:    ON for DEEP only  # ← decision 3
CLEARED_LIST_CAP:       5 per unit
```

**Decision 1 — `APPLY_FIXES`.** V4 defaults to `OFF`: propose diffs, never edit.
That is a real change from how T0–T22 ran (fix → RED/GREEN → gate → commit).
`OFF` across 11 waves accumulates an unmerged backlog of proposals someone must
later apply and re-verify by hand. **Recommend `ON`**, because this repo already
has the safety net V4's `OFF` default substitutes for: eleven CI gates, an
isolated-diff mypy check, RED→GREEN discipline, and a branch nobody merges
without review. If you want proposals only, say so and P1 will produce diffs
without touching the tree.

**Decision 2 — `TOGGLE_B_INNOCENCE`.** V4 says turn it on "when false positives
are expensive (CI gating)". They have been expensive here, twice in one session:
a grep-based "app shadowing" finding that AST disproved (the second `FastAPI(`
was inside a string template), and 24 apparent WF-100 "empty evidence" passes
that came from a probe bypassing the auditor's own gate. Both were caught, but
only because something adversarial was run against them. Cost is ~2× tokens.
**Recommend ON for both tiers.**

**Decision 3 — `TOGGLE_C_TAIL_SWEEP`.** ≥2× tokens for atypical-class coverage.
**Recommend DEEP only** — the 84 files where a missed defect is most expensive.

---

## 7. Per-wave procedure

1. **Ingest the ledger first.** Read `docs/hunts/INVENTORY.md` for prior
   dispositions. V4 rule 1C forbids re-raising a candidate cleared elsewhere for
   the same reason; that only works if the ledger is loaded before elicitation.
2. Run V4 Phases 0→3 over exactly the files that wave lists in `v4_waves.tsv`.
   No drifting into neighbouring files — a swap above/below the cut line is
   allowed by V4 §0.4.5 but must be logged.
3. Every fix: RED→GREEN by `git stash` — confirm the new test fails **for the
   predicted reason** before the fix restores it. A test that passes both ways
   proves nothing.
4. Before commit, all eleven gates plus `pytest -m "not slow and not requires_api
   and not stress and not e2e"` via `python -m pytest` (the `pytest` on PATH in
   this container is a `uv` tool with its own interpreter that lacks the package).
5. Append the wave's inventory to `INVENTORY.md`; write
   `docs/hunts/pN-<name>/{inventory,coverage}.md`. `docs/` is gitignored wholesale
   (`.gitignore:136-137`) — stage with `git add -f` or the files silently vanish
   from the commit.
6. One commit per wave. Push; CI must be green before the next wave starts.

**Dedup cost, and the fix for it.** `INVENTORY.md` is 873 lines today and grows
every wave; by P11 re-reading it per wave is a large fixed cost. If it exceeds
~1,500 lines, emit a machine-readable `docs/hunts/CLEARED.tsv`
(`id, path, class, disposition, reason`) and have waves consult that instead of
the prose. Do this when it hurts, not pre-emptively.

---

## 8. Where this plan is weak

- **The priority score is a regex proxy, labelled INFERENCE.** `mutation` counts
  hits on `cost|budget|price|.write(`, which matches comments and identifiers as
  readily as behaviour. A genuinely dangerous file can score 1 and land in the
  un-audited tail. The score decides *reading order only* — it must never be
  cited as evidence about a finding, per V4's quarantine rule.
- **Severity does not correlate with priority.** A `CRITICAL` defect in a
  priority-1 shim is entirely possible; option C will miss it. This is the
  accepted cost of C over A, and it belongs in every Phase 3 coverage statement
  rather than in a footnote.
- **Churn is measured from `origin/master` only** and keys on current paths, so
  a renamed file reads as zero-churn. Files renamed in T21 (`standalone_server.py`)
  are scored as if new. UNKNOWN magnitude; likely small.
- **Yield is unproven.** Section 1 states the orthogonality argument; it predicts
  V4 *can* find what shape-sweeps cannot, not that it *will* here. P1 is the
  calibration wave — if 8 files at priority 8–12 yield no VERIFIED finding, that
  is real evidence about the remaining ten waves and the plan should be re-cut,
  not completed on momentum.
- **11 waves is a floor, not an estimate of effort.** Each wave is a full
  Phase 0–3 pass with RED→GREEN verification and a green CI run. No wall-clock
  or token projection is offered because none can be measured before P1.
