# T2 — Credentials & Trust Boundary — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Budget: 14 candidates. Spent: 5.

## Phase 0 delta — census correction

The plan's T2 file list is stale against the current tree:
- `orchestrator/security/` contains **no credential/security logic at all** —
  `enhancer.py`, `indesign_plugin_rules.py`, `ios_hig_prompts.py`,
  `wordpress_plugin_rules.py` are website-generator content-rule files.
  `security/__init__.py` is a one-line empty placeholder.
- `orchestrator/api_clients.py` is a 21-line re-export shim; the real
  `UnifiedClient`/`APIResponse` live in `orchestrator/infrastructure/llm_client.py`
  (established in T1, reconfirmed here).
- Only **31** files match `api_key` today, not the plan's 43.
- **A systemic pattern already seen in T0/T1 recurred immediately on a casual
  scope grep**: `orchestrator/` has many root-vs-subpackage file pairs sharing
  a base name. Checked five beyond the two below for divergence:
  `tenancy.py`/`integrations/tenancy.py` (byte-identical), `gateway.py`/
  `integrations/gateway.py` (byte-identical), `provisioned_throughput.py`/
  `operations/provisioned_throughput.py` (byte-identical), `feedback_loop.py`/
  `operations/feedback_loop.py` and `quality_control.py`/`quality/
  quality_control.py` (only import-depth differences from a clean subpackage
  move, no logic divergence). None of these five needed a fix. The two below
  did.
- A dedicated Explore pass covered the credential-exposure/logging/
  generated-output/gateway-auth surface in depth after the two candidates
  below were already found; C3/C4/C5 came out of that pass. Its remaining,
  not-independently-fixed observations are recorded in coverage.md's
  residual-risk statement rather than expanded into more candidates here,
  given this tier's time budget.

## Candidates

### C1 — VERIFIED DEFECT — `generators/secrets_generator.py` was a self-import, resolving to an empty module
- **Property violated:** class 8 (contract) — an import statement must
  resolve to real content, not to itself.
- **Location:** `orchestrator/generators/secrets_generator.py` (pre-fix, 2 lines).
- **Finding:** the file's entire content was
  `from ..generators.secrets_generator import *`. Evaluated from within
  `orchestrator.generators` (this file's own package), `..` reaches
  `orchestrator`, and `.generators.secrets_generator` appended gives
  `orchestrator.generators.secrets_generator` — itself. Python registers a
  module in `sys.modules` before executing its body, so the `import *`
  bound to the same (at that point still-empty) module object and picked up
  nothing. Confirmed directly: `python -c "import
  orchestrator.generators.secrets_generator as m; print([n for n in dir(m)
  if not n.startswith('_')])"` → `[]`. The real 745-line implementation
  (`SecretsGenerator`, `EnvFileBuilder`, `DatabaseEnvBuilder`,
  `RedisEnvBuilder`, JWT/DB-password generation) lives at
  `orchestrator/secrets_generator.py` (root) — not a shim relationship the
  way its sibling `orchestrator/secrets_manager.py` (root, a *correct*
  7-line shim) points to `orchestrator/generators/secrets_manager.py`
  (canonical). The comment left in the file
  (`# FIXED: from .generators.secrets_generator import *`) reads as a
  copy-paste of the sibling shim's pattern with the path never corrected.
- **Reachability:** grepped every plausible import path
  (`from orchestrator.generators.secrets_generator import`, package
  `__init__.py` re-export): zero live importers found anywhere, including
  the root `secrets_generator.py` implementation itself (its own
  `SecretsGenerator`/`EnvFileBuilder` names are referenced only inside that
  same file's own docstring and internal functions). Dead on both sides —
  a landmine (silent-empty-module, not a crash) rather than an active leak.
- **Innocence attempt:** none available — an empty module masquerading as a
  working one is a defect regardless of current reachability, per V7's
  "silent wrong result ranked above crash" framing, and it sits directly on
  this tier's remit (JWT/DB/Redis secret generation for scaffolded
  projects).
- **Fix:** rewritten as a plain re-export shim of the actually-complete
  root module (`from ..secrets_generator import *`), mirroring
  `secrets_manager.py`'s already-correct pattern but in the direction the
  real implementation happens to live (root, not subpackage — noted as
  directionally inconsistent with the manager/generator naming symmetry,
  a naming-convention hazard worth a maintainer's attention, not something
  this tier relocated 745 lines to "fix").
- **Tests:** `test_c1_generators_secrets_generator_is_not_empty`,
  `test_c1_generators_secrets_generator_exposes_canonical_classes`,
  `test_c1_secrets_generator_still_masks_no_hardcoded_defaults`.

### C2 — VERIFIED DEFECT — `codebase_writer.py` (root) silently lacked the newer safety net its subpackage twin gained
- **Property violated:** threat 6 (trust boundary — "a website generator
  that writes to disk from LLM-authored content") — the two paths for
  applying LLM-generated file modifications must behave identically, or a
  caller importing the "wrong" one gets weaker safety guarantees with no
  signal that anything is missing.
- **Location:** `orchestrator/codebase_writer.py` (pre-fix, 317 lines) vs
  `orchestrator/codebase/writer.py` (473 lines).
- **Finding:** byte diff showed the subpackage version is a strict
  superset: it adds `SearchReplaceBlock` + Aider-style SEARCH/REPLACE patch
  parsing (`DiffEngine.parse_search_replace_blocks`/
  `apply_search_replace_blocks`), a pre-destructive-operation snapshot
  safety net in `ModificationGate` (CodeWhale Phase 2 — snapshots the
  project before `MODIFY_FILE`/`DELETE_FILE` tasks via an injected
  `SnapshotStore`, so a bad LLM-authored change can be rolled back), and
  `DELETE_FILE`/`INSTALL_DEP` task-type handling the root version doesn't
  have at all. All five class names (`VerificationResult`,
  `FileOperations`, `DiffEngine`, `ModificationGate`, `CodebaseWriter`) are
  present in both, confirming this is a straight divergence, not two
  unrelated modules.
- **Reachability:** unlike T1/T3's dormant duplicate pairs, this one is
  **actively imported from both paths by existing tests** —
  `tests/test_codebase_optimizations.py` imports `DiffEngine`,
  `SearchReplaceBlock` from `orchestrator.codebase.writer` (the fuller
  path — it has to, `SearchReplaceBlock` doesn't exist on the other side);
  `tests/test_phase6_10_comprehensive.py` and `tests/test_e2e_full_suite.py`
  import `ModificationGate`/`VerificationResult`/`CodebaseWriter`/
  `DiffEngine` from `orchestrator.codebase_writer` (root) specifically.
  Two independent test files already depend on two different, diverged
  copies of the same safety-critical class — a live example of the exact
  hazard T1's C1 warned about, not just a theoretical one.
- **Innocence attempt:** none available. No comment or docstring anywhere
  marks either copy deprecated; nothing signals to a caller of the root
  path that a newer, safer version exists elsewhere.
- **Fix:** `orchestrator/codebase_writer.py` (root) rewritten as a
  re-export shim of `orchestrator.codebase.writer` (the superset).
  Verified this doesn't regress the existing root-path importers: ran
  `tests/test_codebase_optimizations.py` and
  `tests/test_phase6_10_comprehensive.py`'s Modification/Verification/
  DiffEngine/CodebaseWriter-related tests post-fix — 19/19 pass.
- **Tests:** `test_c2_root_codebase_writer_modification_gate_is_canonical`,
  `test_c2_root_codebase_writer_exposes_search_replace_support`,
  `test_c2_root_codebase_writer_still_exposes_pre_existing_names`.

### C3 — VERIFIED DEFECT — the codebase-writing pipeline's own secret-detection gate never blocked anything
- **Property violated:** threat 6 (trust boundary) + threat 2 (credential
  exposure), directly on the standing project rule "never passwords in the
  code."
- **Location:** `orchestrator/codebase/writer.py::ModificationGate.
  _check_secrets()` (pre-fix) — the canonical path, actually wired into
  the live pipeline via `orchestrator/codebase/__init__.py`.
- **Finding:** `_check_secrets()` regex-matches `password=`/`api_key=`/
  `secret=`/`token=`-shaped assignments in LLM-generated content and, on a
  match, appended to `result.warnings`. `apply()` (the method that decides
  whether a modification actually reaches disk) only inspects `ver.errors`
  to block a write (`if ver.errors: ... return False`) — `ver.warnings` is
  populated but read by nothing anywhere in the file. So a hardcoded
  secret detected in LLM-authored output never stopped that output from
  being written. Compounding this, the discarded warning message itself
  embedded up to 20 raw characters of the matched secret
  (`f"Possible {message}: {match.group(1)[:20]}"`) — harmless only because
  nothing read it, not because it was safe to construct.
- **Reachability:** live. `codebase/writer.py` is the version
  `orchestrator/codebase/__init__.py` exports, and (per C2) is now also
  what the root `codebase_writer.py` shim resolves to.
- **Innocence attempt:** none — checked whether some other layer re-scans
  before a real write commits; found no such layer (see C4/C5's sibling
  scanners in coverage.md, which are separately either unwired or
  filename-restricted, not a safety net for this specific gap).
- **Fix:** `_check_secrets()` now appends to `result.errors` (so `apply()`
  actually blocks the write) and no longer echoes the matched value —
  the message states only that a secret pattern was detected.
- **Tests:** `test_c3_detected_secret_blocks_via_errors_not_warnings`,
  `test_c3_detected_secret_message_does_not_echo_the_value`,
  `test_c3_clean_content_has_no_errors_or_warnings`.

### C4 — VERIFIED DEFECT — tenant API keys never survived a restart
- **Property violated:** threat 4/2 (persistence of credentials) — a
  serialization round-trip must preserve the data it's responsible for.
- **Location:** `Tenant.to_dict()` in both `orchestrator/tenancy.py` and
  the (pre-C5-fix, non-importable) `orchestrator/integrations/tenancy.py`.
- **Finding:** `to_dict()` omitted the `api_key` field entirely.
  `TenantManager._save_tenants()` serializes tenants via `to_dict()` for
  the on-disk `tenants.json`; `_load_tenants()` reads it back with
  `api_key=tenant_data.get("api_key", "")` — always `""`, since it was
  never written. Effect: every tenant's real API key is silently lost on
  every process restart, and every restored tenant collides onto the same
  `self.api_keys[""]` entry (last-loaded tenant wins), breaking per-tenant
  isolation the moment the process restarts.
- **Reachability:** live — `_save_tenants()`/`_load_tenants()` are the
  only persistence path `TenantManager` has; any real deployment using
  file-backed multi-tenancy restarts eventually.
- **Innocence attempt:** none — no other field carries the key forward,
  and `get_tenant_by_api_key()`'s lookup is otherwise correct
  (`hmac.compare_digest`), so this is squarely a missing-field bug, not a
  deliberate redaction.
- **Fix:** added `"api_key": self.api_key` to `to_dict()` in both files
  (kept in sync rather than converted to a shim — they were already
  byte-identical with no divergence to reconcile beyond this one field).
- **Tests:** `test_c4_tenant_to_dict_includes_api_key` (parametrized
  across both modules), `test_c4_tenant_manager_survives_restart_with_same_api_key`
  (real trigger: two `TenantManager` instances against the same
  `tmp_path`, simulating an actual restart).

### C5 — VERIFIED DEFECT — `integrations/tenancy.py` could not be imported at all
- **Property violated:** class 8 (contract) — same shape as the T0
  docker_sandbox fix: a relative import correct at one package depth,
  wrong at another.
- **Location:** `orchestrator/integrations/tenancy.py:36` (pre-fix):
  `from .log_config import get_logger`.
- **Finding:** discovered while writing C4's parametrized test — this line
  is byte-identical to root `orchestrator/tenancy.py`'s same line, and
  correct *there* (`orchestrator/log_config.py` is one dot from
  `orchestrator/`). From within `orchestrator/integrations/`, one dot
  resolves to the nonexistent `orchestrator.integrations.log_config`.
  Confirmed directly: `python -c "import
  orchestrator.integrations.tenancy"` raised `ModuleNotFoundError`
  unconditionally. This is the concrete lesson of this tier's Phase 0
  finding that "byte-identical" duplicate pairs (`tenancy.py`/
  `integrations/tenancy.py`, dismissed as a non-issue earlier in this
  tier) can still diverge in *behavior* purely from where the file sits,
  when single-dot relative imports are involved — a blind `diff` is not
  sufficient evidence of "no bug" for this class of duplicate.
- **Reachability:** root `tenancy.py`'s own module docstring recommends
  `from orchestrator.integrations.tenancy import TenantManager, Plan` —
  the one documented usage example for this feature pointed at a path
  that could not be imported.
- **Innocence attempt:** none — this is a plain broken import, unconditional.
- **Fix:** one-line change to `..log_config`, matching the correct
  depth (mirrors the T0 docker_sandbox fix exactly).
- **Tests:** `test_c5_integrations_tenancy_module_actually_imports`.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
