"""
Port Conformance Tests — every domain Protocol has a Null adapter + basic smoke test.

Validates:
1. Every @runtime_checkable Protocol in domain/ports.py has a corresponding Null adapter
2. Each Null adapter passes isinstance() against its Protocol
3. All public methods on the Protocol exist on the Null adapter
"""

import inspect
import pytest

from orchestrator.domain import ports as domain_ports

# ── Registry: every @runtime_checkable Protocol → its Null adapter ──────────

NULL_ADAPTERS = {
    "CachePort": "NullCache",
    "StatePort": "NullState",
    "EventPort": "NullEventBus",
    "ConfigPort": "NullConfig",
    "LLMClient": "NullLLMClient",
    "PlannerPort": "NullPlanner",
    "TelemetryPort": "NullTelemetry",
    "PolicyEnginePort": "NullPolicyEngine",
    "HookRegistryPort": "NullHookRegistry",
    "ValidatorPort": "NullValidator",
    "TaskQueuePort": "NullTaskQueue",
    "SkillStorePort": "NullSkillStore",
    "LSPValidatorPort": "NullLspValidator",
    "SnapshotPort": "NullSnapshotStore",
}


def _get_protocols():
    """Return {name: class} for every @runtime_checkable Protocol in ports.py."""
    protocols = {}
    for name, obj in inspect.getmembers(domain_ports, inspect.isclass):
        if hasattr(obj, "__instancecheck__") and hasattr(obj, "__protocol__"):
            protocols[name] = obj
    return protocols


def test_all_null_adapters_in_registry():
    """All known runtime-checkable protocols have a Null adapter entry."""
    for proto_name in _get_protocols():
        assert proto_name in NULL_ADAPTERS, (
            f"Protocol {proto_name} has no Null adapter in NULL_ADAPTERS registry. "
            f"Add it."
        )


def test_all_registry_entries_exist():
    """Every entry in NULL_ADAPTERS maps to an existing class."""
    for proto_name, null_name in NULL_ADAPTERS.items():
        assert hasattr(domain_ports, proto_name), (
            f"Protocol {proto_name} not found in orchestrator.domain.ports"
        )
        assert hasattr(domain_ports, null_name), (
            f"Null adapter {null_name} not found in orchestrator.domain.ports"
        )


def test_all_null_adapters_have_protocols():
    """Every Null* class in ports.py has a corresponding Protocol entry."""
    null_classes = {
        name
        for name, obj in inspect.getmembers(domain_ports, inspect.isclass)
        if name.startswith("Null")
    }
    null_to_proto = {v: k for k, v in NULL_ADAPTERS.items()}
    for null_name in null_classes:
        assert null_name in null_to_proto, (
            f"{null_name} exists in ports.py but has no Protocol in NULL_ADAPTERS. "
            f"Either add it to the registry or remove the dead code."
        )


# ── Parametrized conformance tests ──────────────────────────────────────────


@pytest.mark.parametrize(
    "proto_name,null_name",
    list(NULL_ADAPTERS.items()),
    ids=[f"{n}≜{k}" for k, n in NULL_ADAPTERS.items()],
)
def test_null_adapter_satisfies_protocol(proto_name, null_name):
    """Null adapter passes isinstance() check and has all protocol methods."""
    Protocol = getattr(domain_ports, proto_name)
    NullCls = getattr(domain_ports, null_name)
    instance = NullCls()

    # 1. isinstance check — structural typing satisfaction
    assert isinstance(instance, Protocol), (
        f"{null_name}() is not an instance of {proto_name}. "
        f"Check that the method signatures match the protocol."
    )

    # 2. All public protocol methods exist on the Null adapter
    proto_methods = {
        m
        for m in dir(Protocol)
        if not m.startswith("_") and callable(getattr(Protocol, m, None))
    }
    for method_name in proto_methods:
        assert hasattr(instance, method_name), (
            f"{null_name} is missing method '{method_name}' required by {proto_name}"
        )
