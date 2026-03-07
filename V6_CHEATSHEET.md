# Multi-LLM Orchestrator v6.0 — Quick Reference
## Black Swan Resilience Cheatsheet

---

## 🛡️ Resilient Event Store

**Import:**
```python
from orchestrator.events_resilient import ResilientEventStore
```

**Setup:**
```python
store = ResilientEventStore(
    primary_path=".events/primary.db",
    replica_paths=[".events/replica1.db", ".events/replica2.db"],
)
```

**Usage:**
```python
# Write (auto-replicates)
await store.append(event)

# Read (auto-failover)
events = await store.get_events()

# Verify integrity
report = await store.verify_integrity()
# Returns: {"primary_healthy": True, "corrupted_events": [], ...}
```

**Health Check:**
```python
if not report["primary_healthy"]:
    logger.critical("Event store corruption detected!")
```

---

## 🔒 Secure Plugin Runtime

**Import:**
```python
from orchestrator.plugin_isolation_secure import (
    SecureIsolatedRuntime, SecureIsolationConfig
)
```

**Setup:**
```python
runtime = SecureIsolatedRuntime(SecureIsolationConfig(
    memory_limit_mb=512,
    cpu_limit_percent=50,
    timeout_seconds=30.0,
    allow_network=False,
    enable_seccomp=True,
    enable_landlock=True,
    enable_capabilities=True,
))
```

**Usage:**
```python
result = await runtime.execute(plugin, "validate", code)
# result.success, result.result, result.error
```

**Check Security:**
```python
for layer, available in runtime.features.items():
    print(f"{layer.name}: {'✅' if available else '❌'}")
```

**Trusted Plugins:**
```python
from orchestrator.plugin_isolation_secure import TrustedPluginRegistry

registry = TrustedPluginRegistry()
registry.add_trusted_plugin("my_plugin", expected_hash)

if registry.verify_plugin(path, "my_plugin"):
    await runtime.execute(plugin, "validate", code)
```

---

## 📊 Resilient Streaming

**Import:**
```python
from orchestrator.streaming_resilient import (
    ResilientStreamingPipeline,
    MemoryPressureConfig,
    BackpressureStrategy,
)
```

**Setup:**
```python
pipeline = ResilientStreamingPipeline(
    max_parallel=3,
    memory_config=MemoryPressureConfig(
        max_queue_size=1000,
        max_memory_mb=1024,
        warning_threshold_percent=70.0,
        critical_threshold_percent=90.0,
        backpressure_strategy=BackpressureStrategy.SAMPLE,
        sampling_rate=10,
    ),
)
```

**Usage:**
```python
async for event in pipeline.execute_streaming(desc, criteria, budget):
    await websocket.send(event)
```

**Health Check:**
```python
health = pipeline.get_health()
# {
#   "circuit_state": "CLOSED",
#   "circuit_failures": 0,
#   "memory_pressure": "NORMAL",
#   "memory_used_mb": 512,
#   "memory_available_mb": 1536,
# }
```

**Backpressure Strategies:**
- `DROP_OLDEST` — Discard oldest events
- `DROP_NEWEST` — Discard newest events  
- `SAMPLE` — Keep every Nth event (use N=sampling_rate)
- `PAUSE` — Halt pipeline temporarily
- `BLOCK` — Block producer (risk: deadlock)

---

## ⚙️ Configuration (`.orchestrator.yml`)

```yaml
resilience:
  event_store:
    type: resilient
    primary_path: .events/primary.db
    replica_paths:
      - .events/replica1.db
      - .events/replica2.db
  
  plugin_security:
    level: secure
    enable_seccomp: true
    enable_landlock: true
    enable_capabilities: true
  
  streaming:
    backpressure: true
    max_queue_size: 1000
    max_memory_mb: 1024
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 30.0
```

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_RESILIENT_STORE` | `false` | Use ResilientEventStore |
| `ORCHESTRATOR_SECURE_PLUGINS` | `false` | Use SecureIsolatedRuntime |
| `ORCHESTRATOR_BACKPRESSURE` | `false` | Use ResilientStreamingPipeline |
| `ORCHESTRATOR_SECCOMP` | `true` | Enable seccomp filtering |
| `ORCHESTRATOR_LANDLOCK` | `true` | Enable Landlock sandbox |

---

## 🧪 Testing

```bash
# Run all resilient tests
pytest tests/test_resilient_improvements.py -v

# Test specific component
pytest tests/test_resilient_improvements.py::TestResilientEventStore -v
pytest tests/test_resilient_improvements.py::TestSecurePluginIsolation -v
pytest tests/test_resilient_improvements.py::TestResilientStreaming -v
```

---

## 🚨 Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| `Circuit breaker open` | `pipeline.circuit.state = CircuitState.CLOSED` |
| `Memory pressure critical` | Reduce `max_queue_size`, increase `sampling_rate` |
| `Replication lag` | Check disk I/O, reduce replica count |
| `seccomp unavailable` | Install `python-seccomp` (Linux only) |
| `Landlock unavailable` | Upgrade to Linux 5.13+ |

---

## 📚 Documentation

- [CAPABILITIES.md](CAPABILITIES.md) — Full feature guide
- [MIGRATION_GUIDE_v6.md](MIGRATION_GUIDE_v6.md) — v5.x → v6.0 migration
- [ADVERSARIAL_STRESS_TEST_REPORT.md](ADVERSARIAL_STRESS_TEST_REPORT.md) — Security analysis

---

**v6.0 — Resilient by default, unstoppable by design.**
