"""
T2 (credentials & trust boundary) proof-of-defect and no-regression tests.

Two VERIFIED DEFECTs from docs/hunts/t2-credentials/inventory.md:

C1 — orchestrator/generators/secrets_generator.py was self-referential:
     `from ..generators.secrets_generator import *` — evaluated from within
     orchestrator.generators — resolves to orchestrator.generators.
     secrets_generator, i.e. itself. Since the module was still being
     initialized (nothing defined yet), `import *` picked up nothing, so
     the module silently ended up completely empty (zero public names)
     despite "succeeding". The real 745-line implementation lives at
     orchestrator/secrets_generator.py (root).

C2 — orchestrator/codebase_writer.py (root) and orchestrator/codebase/
     writer.py (subpackage) were two independently-diverged copies of the
     same modification-gate logic. The subpackage version gained
     SearchReplaceBlock (Aider-style patch parsing) and a pre-destructive-
     operation snapshot safety net that the root version never got —
     existing tests import DiffEngine/VerificationResult/CodebaseWriter/
     ModificationGate from BOTH paths, so picking the "wrong" one silently
     gave you the weaker, snapshot-less rollback behavior.

C3 — ModificationGate._check_secrets() (orchestrator/codebase/writer.py, the
     canonical, actually-wired path) appended detected hardcoded-secret
     patterns to VerificationResult.warnings, but apply() only ever checks
     .errors to decide whether to block a write — a detected hardcoded
     password/api_key/secret/token never actually stopped the file from
     being written. The old message also embedded up to 20 raw characters
     of the matched secret into the (silently-discarded) warning text.

C4 — Tenant.to_dict() (orchestrator/tenancy.py and the byte-identical
     orchestrator/integrations/tenancy.py) never serialized the tenant's
     api_key, so TenantManager._save_tenants() persisted every tenant
     without its key; _load_tenants() then defaulted it back to "" on
     restart, silently losing every tenant's real key and colliding every
     restored tenant onto the same empty-string entry in self.api_keys.

C5 — Found while writing C4's test: orchestrator/integrations/tenancy.py
     did `from .log_config import get_logger` — correct if evaluated from
     orchestrator/ (where the byte-identical root tenancy.py actually
     lives), but this file sits in orchestrator/integrations/, where one
     dot resolves to the nonexistent orchestrator.integrations.log_config.
     "Byte-identical" text does not mean "identical behavior" once a
     single-dot relative import is involved — the same line is only
     correct at one of the two package depths. Root tenancy.py's own
     docstring even recommends importing from this broken path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_generators_secrets_generator_is_not_empty():
    import orchestrator.generators.secrets_generator as mod

    public_names = [n for n in dir(mod) if not n.startswith("_")]
    assert public_names, "orchestrator.generators.secrets_generator is empty"


@pytest.mark.unit
def test_c1_generators_secrets_generator_exposes_canonical_classes():
    from orchestrator.generators.secrets_generator import EnvFileBuilder, SecretsGenerator
    from orchestrator.secrets_generator import EnvFileBuilder as canonical_efb
    from orchestrator.secrets_generator import SecretsGenerator as canonical_sg

    assert SecretsGenerator is canonical_sg
    assert EnvFileBuilder is canonical_efb


@pytest.mark.unit
def test_c1_secrets_generator_still_masks_no_hardcoded_defaults():
    """No-regression sanity check on the real generator this shim now exposes:
    generated secrets must not be a fixed, guessable string."""
    from orchestrator.generators.secrets_generator import SecretsGenerator

    a = SecretsGenerator.create_generic_secret(32)
    b = SecretsGenerator.create_generic_secret(32)
    assert a != b
    assert len(a) > 0


# --- C2 -------------------------------------------------------------------


@pytest.mark.unit
def test_c2_root_codebase_writer_modification_gate_is_canonical():
    from orchestrator.codebase.writer import ModificationGate as canonical
    from orchestrator.codebase_writer import ModificationGate as via_root

    assert via_root is canonical


@pytest.mark.unit
def test_c2_root_codebase_writer_exposes_search_replace_support():
    """No-regression: SearchReplaceBlock (only ever defined in the subpackage
    version) must now be reachable via the root import path too."""
    from orchestrator.codebase_writer import SearchReplaceBlock

    block = SearchReplaceBlock(search="old", replace="new", target_file="f.py")
    assert block.search == "old"
    assert block.replace == "new"


@pytest.mark.unit
def test_c2_root_codebase_writer_still_exposes_pre_existing_names():
    """No-regression: the names existing tests already import from the root
    path must still resolve after the shim conversion."""
    import orchestrator.codebase_writer as mod

    for name in ("DiffEngine", "VerificationResult", "CodebaseWriter", "ModificationGate"):
        assert hasattr(mod, name), f"{name} missing from codebase_writer shim"


# --- C3 -------------------------------------------------------------------


@pytest.mark.unit
def test_c3_detected_secret_blocks_via_errors_not_warnings():
    """The core defect: a hardcoded secret must land in .errors (which apply()
    checks), not .warnings (which apply() never reads)."""
    from orchestrator.codebase.writer import ModificationGate, VerificationResult

    result = VerificationResult()
    gate = ModificationGate()

    gate._check_secrets('api_key = "sk-realsecretvalue1234567890"', result)

    assert result.errors, "hardcoded secret must be recorded as a blocking error"
    assert not result.warnings, "must not also land in warnings (apply() never reads them)"


@pytest.mark.unit
def test_c3_detected_secret_message_does_not_echo_the_value():
    """No further leakage: the error message must not contain the actual
    secret value, only that one was found."""
    from orchestrator.codebase.writer import ModificationGate, VerificationResult

    secret_value = "sk-realsecretvalue1234567890"
    result = VerificationResult()
    gate = ModificationGate()

    gate._check_secrets(f'api_key = "{secret_value}"', result)

    assert all(secret_value not in msg for msg in result.errors)


@pytest.mark.unit
def test_c3_clean_content_has_no_errors_or_warnings():
    """No-regression: content with no secret-shaped strings must pass clean."""
    from orchestrator.codebase.writer import ModificationGate, VerificationResult

    result = VerificationResult()
    gate = ModificationGate()

    gate._check_secrets("def handler(): return {'status': 'ok'}", result)

    assert not result.errors
    assert not result.warnings


# --- C4 -------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path", ["orchestrator.tenancy", "orchestrator.integrations.tenancy"]
)
@pytest.mark.unit
def test_c4_tenant_to_dict_includes_api_key(module_path):
    import importlib

    mod = importlib.import_module(module_path)
    tenant = mod.Tenant(
        id="t1", name="Acme", plan=mod.PLANS[mod.PlanTier.FREE], api_key="real-key-123"
    )

    d = tenant.to_dict()

    assert d.get("api_key") == "real-key-123"


@pytest.mark.unit
def test_c5_integrations_tenancy_module_actually_imports():
    """The core defect: orchestrator.integrations.tenancy must import
    cleanly, not raise ModuleNotFoundError on its own log_config import."""
    import orchestrator.integrations.tenancy  # noqa: F401


@pytest.mark.asyncio
async def test_c4_tenant_manager_survives_restart_with_same_api_key(tmp_path):
    """Real trigger: create a tenant, force a save, construct a FRESH
    TenantManager against the same storage path (simulating a restart), and
    confirm the api_key round-trips instead of coming back empty."""
    from orchestrator.tenancy import TenantManager

    mgr1 = TenantManager(storage_path=str(tmp_path))
    tenant = await mgr1.create_tenant(name="Acme", plan_name="free")
    original_key = tenant.api_key
    assert original_key

    mgr2 = TenantManager(storage_path=str(tmp_path))
    reloaded = mgr2.tenants[tenant.id]

    assert reloaded.api_key == original_key
    assert mgr2.api_keys.get(original_key) == tenant.id
