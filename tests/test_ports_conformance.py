"""Port Conformance Tests — every domain Protocol has a Null adapter."""

import inspect
import pytest
from orchestrator.domain import ports as domain_ports

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
    "FileReaderPort": "NullFileReader",
    "QualityScorer": None,
    "Reranker": None,
}


def _get_protocols():
    protocols = {}
    for name, obj in inspect.getmembers(domain_ports, inspect.isclass):
        if hasattr(obj, "__instancecheck__") and hasattr(obj, "__protocol__"):
            protocols[name] = obj
    return protocols


def test_all_protocols_in_registry():
    for proto_name in _get_protocols():
        assert proto_name in NULL_ADAPTERS, f"Protocol {proto_name} missing from registry"


def test_all_adapters_exist():
    for proto_name, null_name in NULL_ADAPTERS.items():
        assert hasattr(domain_ports, proto_name), f"Protocol {proto_name} not found"
        if null_name is not None:
            assert hasattr(domain_ports, null_name), f"Null adapter {null_name} not found"


def test_all_null_adapters_have_protocols():
    null_classes = {
        n for n, _ in inspect.getmembers(domain_ports, inspect.isclass) if n.startswith("Null")
    }
    null_to_proto = {v: k for k, v in NULL_ADAPTERS.items() if v}
    for null_name in null_classes:
        assert null_name in null_to_proto, f"{null_name} has no Protocol entry"


@pytest.mark.parametrize(
    "proto_name,null_name",
    [(k, v) for k, v in NULL_ADAPTERS.items() if v],
    ids=[f"{n}≜{k}" for k, n in NULL_ADAPTERS.items() if n],
)
def test_null_adapter_satisfies_protocol(proto_name, null_name):
    Protocol = getattr(domain_ports, proto_name)
    NullCls = getattr(domain_ports, null_name)
    instance = NullCls()
    assert isinstance(instance, Protocol), f"{null_name}() is not instance of {proto_name}"
    proto_methods = {
        m for m in dir(Protocol) if not m.startswith("_") and callable(getattr(Protocol, m, None))
    }
    for method_name in proto_methods:
        assert hasattr(instance, method_name), f"{null_name} missing '{method_name}'"
