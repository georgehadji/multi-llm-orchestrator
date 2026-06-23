"""
Guards against model-registry drift from the live OpenRouter catalogue.

Two layers:
* Offline structural guards (always run): no genuinely-dead id may appear as a
  live reference in the registry, the Model enum, or as a redirect *target*.
* A live audit (skipped without network): the full audit reports zero dead ids.

Background: the registry repeatedly drifted from OpenRouter (renamed/removed
ids → 404 crashes and "Unknown model" warnings). See
scripts/audit_openrouter_models.py.
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_openrouter_models.py"
SNAPSHOT_PATH = REPO_ROOT / "scripts" / "openrouter_models_snapshot.json"

# Ids verified dead via real OpenRouter calls (404/400) or absent providers.
# Each was remapped to a live replacement; none may resurface as a live
# reference. anthropic/claude-*-N-M (hyphen) are deliberately EXCLUDED — they
# resolve at runtime via server-side normalization.
KNOWN_DEAD_IDS = frozenset(
    {
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "openai/gpt-5.4-codex",
        "openai/o4",
        "meta-llama/llama-3.1-405b-instruct",
        "stepfun/step-3.5",
        "nvidia/nemotron-3-super",
        "deepseek/deepseek-reasoner",
        "aionlabs/aion-2.0",
        "qwen/qwen-3-coder-next",
        "qwen/qwen-3-coder",
        "qwen/qwen-3-max-thinking",
        "qwen/qwen-3.5-235b-a22b-thinking-2507",
        "qwen/qwen-3.5-397b-a17b",
        "qwen/qwen-3-697b-a17b",
        "black-forest-labs/flux.2-klein-4b",
        "black-forest-labs/flux.2-max",
        "black-forest-labs/flux.2-flex",
        "black-forest-labs/flux.2-pro",
        "recraft/recraft-v4.1-utility",
        "recraft/recraft-v4.1-pro",
        "recraft/recraft-v4.1",
        "recraft/recraft-v4-pro-vector",
        "recraft/recraft-v4-vector",
        "recraft/recraft-v4-pro",
        "recraft/recraft-v4",
        "recraft/recraft-v3",
        "sourceful/riverflow-v2-pro",
        "sourceful/riverflow-v2-fast",
        "sourceful/riverflow-v2-max-preview",
        "sourceful/riverflow-v2-standard-preview",
        "sourceful/riverflow-v2-fast-preview",
        "bytedance-seed/seedream-4.5",
    }
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("or_audit", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit_module()


def _has_network() -> bool:
    try:
        socket.create_connection(("openrouter.ai", 443), timeout=3).close()
        return True
    except OSError:
        return False


@pytest.mark.unit
def test_no_dead_id_referenced_as_live_in_registry_files():
    """A dead id may only appear as an UNAVAILABLE_MODELS key, never as a live
    reference (enum value, cost-table key, routing list, …)."""
    from orchestrator.domain.model_registry import ModelRegistry

    documented = set(ModelRegistry.UNAVAILABLE_MODELS)
    refs = audit.extract_referenced_ids(audit.DEFAULT_TARGET_FILES)
    offending = {
        mid: sorted(sources)
        for mid, sources in refs.items()
        if mid in KNOWN_DEAD_IDS and mid not in documented
    }
    assert offending == {}, f"dead ids referenced as live: {offending}"


@pytest.mark.unit
def test_model_enum_has_no_dead_values():
    from orchestrator.models import Model

    dead_values = {m.value for m in Model} & KNOWN_DEAD_IDS
    assert dead_values == set(), f"Model enum still maps to dead ids: {dead_values}"


@pytest.mark.unit
def test_unavailable_models_never_redirect_to_a_dead_id():
    from orchestrator.domain.model_registry import ModelRegistry

    bad = {
        dead: repl
        for dead, repl in ModelRegistry.UNAVAILABLE_MODELS.items()
        if repl in KNOWN_DEAD_IDS
    }
    assert bad == {}, f"UNAVAILABLE_MODELS redirects to dead ids: {bad}"


@pytest.mark.unit
def test_audit_against_committed_snapshot_is_clean():
    """The audit must pass against a locally-cached catalogue snapshot.

    The snapshot is a generated artifact (gitignored) — regenerate with
    ``python scripts/audit_openrouter_models.py --save-snapshot
    scripts/openrouter_models_snapshot.json``. Skipped when absent so CI relies
    on the deterministic structural guards above plus the live integration test.
    """
    if not SNAPSHOT_PATH.exists():
        pytest.skip("no cached catalogue snapshot; run with --save-snapshot")
    live_ids = audit.load_snapshot_ids(SNAPSHOT_PATH)
    known = audit.load_known_deprecated()
    refs = audit.extract_referenced_ids(audit.DEFAULT_TARGET_FILES)
    dead = audit.find_dead_ids(refs, live_ids, known)
    stale = audit.find_stale_replacements(known, live_ids)
    assert dead == {}, f"dead referenced ids vs snapshot: {dead}"
    assert stale == {}, f"UNAVAILABLE_MODELS replacements gone dead: {stale}"


@pytest.mark.integration
def test_audit_against_live_catalogue_is_clean():
    if not _has_network():
        pytest.skip("no network access to openrouter.ai")
    live_ids = audit.fetch_live_ids()
    known = audit.load_known_deprecated()
    refs = audit.extract_referenced_ids(audit.DEFAULT_TARGET_FILES)
    dead = audit.find_dead_ids(refs, live_ids, known)
    stale = audit.find_stale_replacements(known, live_ids)
    assert dead == {}, f"dead referenced ids vs live catalogue: {dead}"
    assert stale == {}, f"UNAVAILABLE_MODELS replacements gone dead: {stale}"


@pytest.mark.integration
def test_runtime_only_ids_resolve_via_endpoints():
    """Video-gen ids are absent from /models but must resolve at /endpoints.

    Keeps the RUNTIME_ONLY_IDS allowlist honest: a retired or mistyped id is
    reported instead of being silently trusted by find_dead_ids.
    """
    if not _has_network():
        pytest.skip("no network access to openrouter.ai")
    failures = audit.verify_runtime_only_ids()
    assert failures == {}, f"runtime-only ids that no longer resolve: {failures}"
