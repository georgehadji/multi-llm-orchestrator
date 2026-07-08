# SKILL LIBRARY AUTHORING BRIEF — Multi-LLM Orchestrator
Read this entire brief before writing anything. You are one of 16 parallel agents each authoring ONE skill under `.claude/skills/` in the repo at `E:\Documents\Vibe-Coding\Ai Orchestrator`.

## Mission
A retiring distinguished fellow is encoding project mastery so junior/mid-level engineers and Sonnet-class AI models can debug, extend, validate, and advance this project without him. Wrong runbooks are worse than none.

## HARD RULES (non-negotiable)
1. **Write ONLY inside `.claude/skills/<your-skill-name>/`** (SKILL.md + optional scripts/, references/). The rest of the repo is READ-ONLY. No mutating git commands (log/show/diff/branch OK).
2. **GROUND TRUTH ONLY**: verify EVERY command, flag, path, class name, and claim against the repo (Read/Grep/Bash) before stating it. If you cannot verify, either omit or label "UNVERIFIED — check before use".
3. Format: `.claude/skills/<name>/SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: <name>
   description: <trigger-rich one-paragraph description: exactly WHEN a model should load this skill, with symptom keywords>
   ---
   ```
4. Audience: zero-context mid-level engineer or Sonnet-class model. Imperative runbook voice. Copy-pasteable commands (Windows PowerShell-compatible where shell-specific; repo dev is Windows 11, CI is ubuntu-latest — note differences). Define every jargon term once. Tables and checklists over prose.
5. Each skill MUST include: (a) a "When NOT to use this skill" section pointing to sibling skills by name; (b) date-stamps on volatile facts (today: 2026-07-07); (c) final section "Provenance and maintenance" with one-line re-verification commands for anything that may drift.
6. No oversell: unproven/candidate things labeled as such. Nothing may contradict CLAUDE.md or route around change control.
7. Do NOT cite private/user paths (C:\Users\...) as load-bearing sources. Embed the knowledge itself.
8. Target length: 200–600 lines per SKILL.md. Dense, scannable.

## PROJECT FACTS (verified 2026-07-06/07 — re-verify anything you use)
- Multi-LLM Orchestrator v6.0.0, Python >=3.10 (3.12 primary), hatchling build. Repo: github.com/georgehadji/multi-llm-orchestrator. Current branch: feat/response-healing.
- Install: `pip install -e ".[dev]"`; extras: dev, dashboard, security, tracing, docs, image. Entry points: `orchestrator`, `mllm` (both → orchestrator.cli:main), `dashboard` (cli_dashboard:main).
- Hexagonal architecture. Four Unbreakable Rules (CLAUDE.md): engine.py = Mediator only; models.py = pure data; TDD without exception; no new root-level orchestrator/*.py modules.
- CI (.github/workflows/ci.yml): lint (informational), architecture boundaries (BLOCKING: `lint-imports` 5 contracts + `python scripts/check_new_root_files.py --baseline origin/master`), mypy on domain/application/engine_core/container.py (blocking for core), pytest `-m "not slow and not requires_api and not stress and not e2e" --cov=orchestrator --cov-fail-under=6` (pyproject fail_under=7 ratchet), model audit (non-blocking), bandit HIGH (blocking).
- .importlinter 5 contracts: domain-purity; application-no-concrete-infra; application-services-no-engine; engine-core-no-loose-infra; root-modules-no-infra.
- Pre-commit (language:system): black --check (line-length 100), ruff check, bandit HIGH, lint-imports.
- Tests: tests/ (139 files); real suites in tests/unit, tests/contracts, tests/integration, tests/regression, tests/smoke + root-level; 25 legacy files in pyproject `--ignore` list; markers: unit/integration/slow/requires_api/e2e/stress/contract etc.; asyncio_mode=auto. tests/test_preexisting_problems.py = xfail(strict) ledger of catalogued bugs. 6 skips from engine↔container circular imports in test_phase6_10_comprehensive.py.
- Config: orchestrator/config/{costs,routing,fallbacks,limits,thresholds}.json; orchestrator/config.py (Timeout, TokenLimits, BudgetDefaults, QualityThresholds, OpenRouterOptimizations.from_env with USE_JSON_SCHEMA_RESPONSES, USE_MODEL_VARIANTS, USE_NATIVE_FALLBACKS, USE_PROVIDER_SORTING, USE_STREAMING, USE_EMBEDDING_CACHE, USE_RESPONSE_HEALING — all default false). ENABLE_PR_COMMENTS (default true), ENABLE_AUTO_COMMIT (default false). API keys: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, XAI_API_KEY.
- Known drift trap: config JSON keys MUST exactly equal Model enum values (orchestrator/models.py); builders don't resolve aliases — mismatched entries silently drop.
- Core pipeline: decompose → per task: generate → critique → revise → evaluate (iterate to max_iterations). Dual budget: per-run Budget (models.py/budget.py) + cross-run BudgetHierarchy (cost.py). Evaluation: 2-pass self-consistency, delta ≤ threshold → mean else conservative; _aggregate now median for 3+ runs (was bug: discarded runs).
- HITL: fail-closed DecisionChannel gate (orchestrator/hitl/), ORCH_HITL_AUTOAPPROVE=true legacy escape hatch.
- Existing .claude assets (do NOT duplicate; cross-reference): skills mindmap (loads docs/CODEBASE_MINDMAP.md), architecture-guard, orchestrator-run, dashboard-start, quality-check, test-smart, ponytail/-audit/-help/-review; hooks pre_edit_core.py, post_edit_ruff.py, post_bash_pytest.py.

## KEY INCIDENTS (for stories/rationale — cite commit hashes, verify with `git log`/`git show <hash> --stat`)
- 11deb573 HITL silent auto-approval → fail-closed gate (FIX-1), 11 tests.
- d913d136 BOM chars in 32 .py files → cryptic import errors on Linux only; + 4 bare `except:` fixed.
- 416b9e18 P4 hierarchy node-ID collision (len-based IDs reused after delete → monotonic counter); P5 BatchClient `if request.result:` truthiness → 300s hang on valid falsy results → gate on status==COMPLETED.
- e863f0c8 EvaluatorService._aggregate discarded consistency runs 3..N (returned scores[0]) → 0/1/2/3+ handling with median.
- 611c1403 9 security vulns fixed (admin key guard hmac.compare_digest, dev_server port injection, SSRF guards, WS bind 127.0.0.1, SQL identifier allowlists, CSP unsafe-eval removal).
- 431fc89c ARCH-AUDIT-V2 score 5/10 → multi-phase remediation A1–E2; engine.py 1867→demolition toward <300 lines (4ac9b4f1, b7263052, 9fb2b826 removed dead methods + fixed self._container→self._c cleanup bug and set_optimization_backend planner ref).
- Config/enum model-id drift fixed repeatedly (2026-06-23; also ab17b5f4 resync).
- Command-module extraction from cli.py left wrong func= names/missing imports; a relative-import bug crashed the whole CLI.
- Verbalized Sampling pipeline judged dead code + not paper-faithful → documented retirement (docs/VERBALIZED_SAMPLING_ANALYSIS.md).
- use_provider_sorting = dead flag (declared, never affects call path); semantic cache not wired into call path; response DiskCache 48h TTL is the real cost win — cache can defeat eval self-consistency.

## USER'S ANSWERS (fold in)
- Hardest live problems (campaign covers ALL 4 tracks): (1) engine/container circular imports + engine demolition; (2) response-healing hardening end-to-end; (3) routing/config drift permanent fix; (4) generated-output quality ceiling (evaluator/design gates/ponytail integration).
- Unwritten rules (MUST appear in change-control): never weaken a gate to pass (no coverage-floor lowering, no contract exemptions, no xfail-to-green, no allowlist expansion without approval); config JSON keys ≡ Model enum values + run drift tests after touching; no new dependencies without explicit approval (stdlib-first, ponytail doctrine).
- Costliest failure classes (playbook leads with, in order): silent auto-approve/silent failure; model-id/config drift; refactor fallout; environment/encoding traps (BOM, Windows/Linux, black CI-vs-local version drift, grimp==3.3 pin for Windows Rust panic).
- "Beyond SOTA" = all four: cost-quality frontier (VFM/free-tier-first routing), autonomy reliability (zero-silent-failure unattended runs), generated-output quality (Awwwards-level/secure-by-default), orchestration science (adversarial eval, VS/MAP-Elites, self-consistency).

## SKILL INVENTORY (for cross-references — use these exact names)
1. orchestrator-change-control
2. orchestrator-debugging-playbook
3. orchestrator-failure-archaeology
4. orchestrator-architecture-contract
5. llm-orchestration-reference
6. orchestrator-config-and-flags
7. orchestrator-build-and-env
8. orchestrator-run-and-operate
9. orchestrator-diagnostics-and-tooling
10. orchestrator-validation-and-qa
11. orchestrator-docs-and-writing
12. orchestrator-external-positioning
13. orchestrator-hardest-problems-campaign
14. orchestrator-proof-and-analysis-toolkit
15. orchestrator-research-frontier
16. orchestrator-research-methodology

Cross-reference siblings by name; one home per fact — if another skill owns a fact, reference it instead of duplicating (ownership follows the numbered list: e.g., flag catalog lives in orchestrator-config-and-flags; incident chronicle lives in orchestrator-failure-archaeology; gates live in orchestrator-change-control).
