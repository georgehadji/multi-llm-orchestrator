# T22 — reasoning & generation depth

Wave 22 of the depth pass. Scope: `reasoning/` (5,450 lines, of which
`ara_pipelines.py` is 4,288) and `generators/wf100/` (5,121 lines, of which
`checks.py` is 2,251) — the largest line-count debt in the programme, and both
on money paths.

Method, per the plan: too large to read uniformly, so prioritise scoring and
budget arithmetic, stage transitions, and **any check that can report a verdict
having skipped**.

## Phase 1 — the probe that mattered

`wf100` gates every check on declared evidence (`auditor.py`):

```
if check.requires - evidence.available:  ->  OUTSTANDING
```

Running all 82 check implementations directly against `SiteEvidence()` — no
pages, no styles, no scripts — produced **24 PASSes**, with self-incriminating
detail strings: *"exactly one `<h1>` on each of 0 pages"*, *"0 unique titles"*,
*"no machine-detectable WCAG violations across 0 pages"*.

That looked like 24 defects. It was not. The probe called implementations
directly and **bypassed the gate** that would have marked them OUTSTANDING.
Hypothesis falsified.

But it sharpened into a precise one: the gate is only as good as each check's
declaration. **A check that reads evidence it does not declare slips the gate.**
Comparing declared `requires` against the `ev.*` attributes each function body
actually touches gave 4 candidates, and reading them settled 3.

## Phase 3–4 — triage

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | G7 (form abuse protection) computed `combined = ev.markup + ev.scripts` while declaring `requires = {MARKUP}`. Captcha, honeypot and rate limiting are normally wired in JavaScript, so with the scripts uncollected the check searched an empty string for half its evidence and still returned a definite verdict. Demonstrated: identical markup, PASS *"form abuse protection in place: captcha"* with scripts collected, FAIL *"public forms with no captcha, honeypot or rate limiting — they will be found by bots"* without. |
| C2 | **VERIFIED DEFECT — FIXED** | `ara_pipelines.py:1424` read `state.metadata.get("meta_evaluation", {})` and **discarded the result**, under the comment *"# Weight by meta-evaluation quality"*, inside `_phase_jury_weighted_ranking`. `meta_evaluation` is referenced exactly twice in the file: written at 1397, fetched-and-dropped at 1424. The ranking is not weighted by it. |
| — | **Cost note on C2** | The data is produced by a real `self.client.call(model=verifier, max_tokens=1500, temperature=0.2)` at 1388–1394, in a reachable phase (`_phase_jury_weighted_ranking` is called at 1260). Its sibling call's output (`verifications`) *is* used, at 1431, 1464 and 1493. So this is a **paid LLM call on every run whose result is thrown away**. |
| C3 | FALSE (innocent) | A9 reads `ev.http` under `if ev.http is not None`, falling back to `404.html`/`_redirects`. |
| C4 | FALSE (innocent) | D9 reads `ev.http` under the same guard, deciding from page markup otherwise. |
| C5 | FALSE (innocent) | E8 reads `ev.record` under `if record else`, falls back to structured data, and returns OUTSTANDING (`_out`) when no locality can be established either way — the pattern C1 was missing. |
| C6 | **FALSE — hypothesis falsified** | All four `/ len(...)` sites in both regions are guarded: `if not verified: return` (3485), `if state.scores:` (511), `if scores else 5.0` (1428), and `if len(words) < 50: continue` (checks.py:1467). No division-by-zero, no vacuous mean. |
| C7 | Recorded, not elevated | Two fabricated neutral defaults: `avg_score = ... if scores else 5.0` (1428) and `state.final_score = 0.5` when there are no claims (3489). Same shape as T6's fabricated-neutral-score and T18's C6 `get_usage_percent` returning 50%; treated consistently rather than re-litigated. |

## Phase 5 — fixes

**C1** — the fix is narrow on purpose. Adding SCRIPTS to G7's `requires` would
have been wrong: a site with no JavaScript at all has no SCRIPTS evidence, so
the auditor would skip G7 entirely and lose a verdict it can make perfectly
well from markup. Instead G7 returns OUTSTANDING only in the genuinely
ambiguous case — nothing found in markup, *and* the markup loads scripts,
*and* none were collected:

```python
if not protections and re.search(r"<script\b", ev.markup, re.I) and not ev.scripts.strip():
    return _out("G7", "the pages load scripts the auditor did not collect, ...")
```

Five cases verified: protection visible → PASS; scripts uncollected →
OUTSTANDING; scripts collected but genuinely unprotected → FAIL; static site
with a honeypot in HTML → PASS; static site genuinely unprotected → FAIL.
The last two are why `requires` was not widened.

**C2** — the no-op fetch and the comment that misdescribed it are replaced by
a note stating plainly that the weighting does not exist and that the call
producing the data is unused. Behaviour is unchanged: the statement had no
effect. Whether to wire the weighting or drop the call is escalated, because
the weighting formula is nowhere specified and inventing one would be
fabricating intent.

## Phase 8 — gate

`scripts/check_wf100_evidence.py` fails any check reading evidence it does not
declare. Four guarded reads are allowlisted with their reasons (A9, D9, E8, and
G7 now that its read reports OUTSTANDING instead of asserting).

Verified to catch the rule rather than bless the exemption: with G7 removed
from the allowlist *and* its fix stashed, the gate reports
`G7: reads SCRIPTS but declares MARKUP` and exits 1.
