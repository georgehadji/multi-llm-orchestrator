# T2 — Credentials & Trust Boundary — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`orchestrator/generators/secrets_generator.py` + `orchestrator/secrets_generator.py`,
`orchestrator/codebase_writer.py` + `orchestrator/codebase/writer.py`,
`orchestrator/tenancy.py` + `orchestrator/integrations/tenancy.py`. A dedicated deep
pass additionally covered (without independently fixing every finding — see
below): API key loading in `orchestrator/infrastructure/llm_client.py`, logging
exposure across `llm_client.py`/`feedback_loop.py`/`xai_search.py`/
`config_sync.py`/`a2a_protocol.py`, `generators/secrets_manager.py`'s masking
machinery, generated-website credential handling, and the auth surfaces in
`api_server.py`/`gateway.py`/`commands/server.py`.

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>                    PASS
ruff check <changed files>                                                 PASS
lint-imports                                                                PASS (5/5 contracts KEPT, 824 files/1389 deps)
python scripts/check_root_module_freeze.py                                 PASS (256/256, no additions)
python scripts/check_test_markers.py                                       PASS
mypy orchestrator/domain/ orchestrator/application/ .../container.py       PASS (58 files, 0 issues)
bandit -lll -r <changed files>                                              PASS (0 issues in changed code; 3 pre-existing
                                                                             Low findings at codebase/writer.py:121,124
                                                                             are install_dependency()'s subprocess.run,
                                                                             unrelated to this tier's changes)
python -m pytest tests/unit/test_hunt_t2_credentials.py -m unit            PASS (13/13)
python -m pytest tests/ -q -m "unit or integration"                        see docs/hunts/INVENTORY.md for this run's totals
```

## Existing tests updated to match the corrected behavior
`tests/test_e2e_full_suite.py::TestSafetyGates::test_secret_detection` and
`tests/test_phase6_10_comprehensive.py::TestCodebaseWriter::test_secret_detection`
asserted `len(vr.warnings) > 0` — i.e. they encoded C3's bug (secret lands in
the field `apply()` never reads) as the expected, correct behavior. Both
updated to assert on `.errors` instead, matching the fix; both pass.
Found via the full-suite run below, not missed silently.

## RED→GREEN verification
All five fixes retroactively verified via `git stash push --keep-index` on
the five fixed source files (test file kept), full T2 suite re-run, fixes
restored via `git stash pop`. 12 of 13 tests failed against the pre-fix tree
for the exact predicted reason (empty module, wrong class identity, secret
landing in `.warnings` not `.errors`, missing `api_key` key, `ModuleNotFoundError`
on `integrations/tenancy.py`'s own broken import). 1 test
(`test_c4_tenant_to_dict_includes_api_key[orchestrator.tenancy]`'s sibling for
`orchestrator.integrations.tenancy`) additionally exposed C5 *during* the RED
run — the parametrized case failed with `ModuleNotFoundError`, not the
assertion it was written to check, which is exactly what led to discovering
and fixing C5. All 13 pass on the fixed tree.

## Verdict
- **VERIFIED DEFECTs fixed:** 5 — C1 (self-importing, silently-empty
  `secrets_generator` shim), C2 (`codebase_writer.py` root/subpackage
  divergence, missing safety net), C3 (secret-detection gate populated
  `warnings`, which `apply()` never reads — a hardcoded secret never
  blocked a write), C4 (`Tenant.to_dict()` never serialized `api_key`,
  losing every tenant's key on restart), C5 (`integrations/tenancy.py`
  could not be imported at all — a single-dot relative import wrong at
  its actual package depth).
- **CLEARED (innocent):** 0 candidates opened and withdrawn. Five
  additional root-vs-subpackage pairs were checked for the same divergence
  pattern as C1/C2 and found clean (see inventory.md's Phase 0 delta) —
  recorded as checked-and-clean, not raised as candidates in the first
  place, so not counted as "cleared."
- **Residual, not independently fixed, from the dedicated credential-flow
  pass — each is real, each is `[UNK]` or `[REQUIRES HUMAN REVIEW]`, none
  silently dropped:**
  - `orchestrator/operations/diagnostics.py`'s environment health-check
    treats `OPENAI_API_KEY`/`GOOGLE_API_KEY`/`ANTHROPIC_API_KEY` as
    interchangeable with `OPENROUTER_API_KEY`/`DEEPSEEK_API_KEY` for
    "is the orchestrator configured" purposes, but the live
    `UnifiedClient` only ever reads the latter two and raises
    `AuthenticationError` if both are unset — a config that diagnostics
    calls healthy can still crash on first real call. `[REQUIRES HUMAN
    REVIEW]`: fixing this means deciding whether diagnostics should match
    the client's real requirement or whether the client should widen to
    accept the other three (a product decision, not a 1-line bug fix).
  - `generators/secrets_manager.py`'s `SecretsFilter`/`mask_string`/
    `setup_secure_logging` — a fully-implemented log-scrubbing mechanism
    for exactly the `sk-`/`AIza`/`ghp_`/Slack/Bearer patterns this tier
    cares about — is never installed on any real logger (`log_config.py`
    only installs `CorrelationIdFilter`). `[REQUIRES HUMAN REVIEW]`:
    wiring a global logging filter is a cross-cutting change with its own
    perf/behavior blast radius, not scoped to one file.
  - Several `logger.error("...%s", e)` / `logger.error(f"...{e}")` sites
    (`llm_client.py:441`, `feedback_loop.py:648`, `xai_search.py:235-239`,
    `config_sync.py:89-109`, `a2a_protocol.py:169-171`) log a raw SDK/HTTP
    exception that was raised from a call built with a real
    `Authorization: Bearer <key>` header. Whether any of these SDKs'
    exception `str()` actually embeds that header is `[UNK]` — not
    triggered with a real failure in this pass. If the (unwired)
    `SecretsFilter` above were installed, this class of risk would be
    substantially mitigated regardless of the individual answer.
  - **The three purpose-built "catch a hardcoded secret before write"
    scanners besides the one fixed as C3**: `orchestrator/codebase_writer.py`
    (root — this tier's C2 fix makes it a shim of the now-fixed C3 code,
    so it inherits the fix, not a separate gap); `orchestrator/safety/
    security_review.py`'s `SecurityReviewer.quick_scan` (rule `SEC-001`,
    embeds up to 80 raw characters of a matched `sk-`/`AIza`-style key
    into `SecurityFinding.description`) and `orchestrator/safety/
    security_validator.py`'s `check_hardcoded_secrets()` (embeds up to 100
    raw characters) both have **zero callers anywhere outside their own
    module** — dead code, `[REQUIRES HUMAN REVIEW]` on whether either
    should be wired in (and if so, redacted the same way C3 now is) rather
    than fixed blind, since neither is reachable today.
  - `orchestrator/gateway.py`'s `verify_api_key`/`register_api_key` use a
    plain dict lookup, not `hmac.compare_digest` (unlike `api_server.py`
    and `tenancy.py`'s equivalent methods) — a timing-attack-relevant gap,
    `[REQUIRES HUMAN REVIEW]` rather than fixed here because `gateway.py`'s
    own `_forward_request` is an explicit simulation stub with no real
    HTTP listener found anywhere in the live pipeline (reachability
    unconfirmed, unlike C3/C4/C5 which are all live).
  - `orchestrator/integrations/multi_tenant_gateway.py`'s tenant/JWT
    lookups use plain `==`-style comparison, not constant-time — same
    disposition as the gateway.py finding above.
- No live path was found where a real orchestrator-owned provider API key
  is written into generated website/app output — every `.env`/config
  template found uses an explicit placeholder (empty value + a comment
  telling the deployer to generate their own), consistent with the
  standing project rule. This is `[VF]` for every generator file actually
  read, not exhaustively for `website_generator.py`'s full 3699 lines.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above, no VERIFIED defect remains unfixed in the
five files this tier's candidates touched. This does not claim the broader
credential-logging surface is safe — three concrete, still-open questions
(SDK exception content, the unwired masking filter, the two dead-but-not-
yet-decided secret scanners) are recorded above as residual, not resolved.

## What this tier does NOT claim
- It does not claim `orchestrator/generators/website_generator.py` (3699
  lines) is free of a secret-writing path — only that the five specific
  provider-env-var names were absent from it.
- It does not claim `gateway.py`/`multi_tenant_gateway.py`'s non-constant-
  time comparisons are exploitable in practice — reachability from a real
  HTTP entry point was not confirmed either way.
- It does not claim the `security/` package (its actual content —
  website-generator design-rule files, not credential logic) was audited
  for defects; it was excluded from scope once its true contents were
  established in Phase 0.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used — all five fixes
correct on first pass, confirmed via the retroactive RED→GREEN stash test
above.
