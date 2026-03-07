# Migration Guide: v5.x → v6.0
## Multi-LLM Orchestrator — Black Swan Resilience Upgrade

**Target Version:** v6.0 | **Estimated Effort:** 30 minutes | **Risk Level:** Low

---

## Overview

v6.0 introduces **Black Swan Resilience** — production-hardened defenses against catastrophic failure modes:

1. **Event Store Corruption Protection** — WAL + replication + checksums
2. **Plugin Sandbox Hardening** — seccomp + Landlock + capabilities
3. **Streaming Backpressure** — Bounded queues + circuit breaker

These features are **opt-in** — your existing code continues to work unchanged.

---

## Breaking Changes

None. v6.0 is fully backward compatible. All new resilient features are opt-in via new classes.

| Component | v5.x | v6.0 (New) | Status |
|-----------|------|------------|--------|
| Event Store | `SQLiteEventStore` | `ResilientEventStore` | Optional upgrade |
| Plugin Runtime | `IsolatedPluginRuntime` | `SecureIsolatedRuntime` | Optional upgrade |
| Streaming | `StreamingPipeline` | `ResilientStreamingPipeline` | Optional upgrade |

---

## Quick Migration (Recommended)

### Step 1: Upgrade Event Store (5 minutes)

**Before (v5.x):**
```python
from orchestrator.events import SQLiteEventStore

store = SQLiteEventStore("events.db")
```

**After (v6.0):**
```python
from orchestrator.events_resilient import ResilientEventStore

store = ResilientEventStore(
    primary_path=".events/primary.db",
    replica_paths=[
        ".events/replica1.db",
        ".events/replica2.db",
    ],
)
```

**Benefits:**
- Automatic replication to 2+ replicas
- SHA-256 checksums detect corruption
- Automatic failover if primary fails
- Write-Ahead Logging for crash safety

---

### Step 2: Upgrade Plugin Security (10 minutes)

**Before (v5.x):**
```python
from orchestrator.plugin_isolation import IsolatedPluginRuntime, IsolationConfig

runtime = IsolatedPluginRuntime(IsolationConfig(
    memory_limit_mb=512,
    timeout_seconds=30.0,
))
```

**After (v6.0):**
```python
from orchestrator.plugin_isolation_secure import (
    SecureIsolatedRuntime, SecureIsolationConfig
)

runtime = SecureIsolatedRuntime(SecureIsolationConfig(
    memory_limit_mb=512,
    timeout_seconds=30.0,
    # New security layers:
    enable_seccomp=True,      # Block dangerous syscalls
    enable_landlock=True,     # Filesystem sandboxing
    enable_capabilities=True, # Drop Linux privileges
    allow_network=False,      # Network access policy
))
```

**Note:** Security layers auto-detect availability. If `seccomp` or `Landlock` aren't available (e.g., on Windows or older Linux), the runtime gracefully degrades to available layers.

**Benefits:**
- 5-layer defense (process + seccomp + Landlock + capabilities + resources)
- Blocks ptrace, execve, fork, socket calls
- Prevents sandbox escape even if one layer fails

---

### Step 3: Upgrade Streaming Pipeline (10 minutes)

**Before (v5.x):**
```python
from orchestrator.streaming import StreamingPipeline

pipeline = StreamingPipeline(max_parallel=3)

async for event in pipeline.execute_streaming(desc, criteria, budget):
    await websocket.send(event)
```

**After (v6.0):**
```python
from orchestrator.streaming_resilient import (
    ResilientStreamingPipeline,
    MemoryPressureConfig,
    BackpressureStrategy,
)

pipeline = ResilientStreamingPipeline(
    max_parallel=3,
    memory_config=MemoryPressureConfig(
        max_queue_size=1000,           # Bounded queue
        max_memory_mb=1024,            # Memory limit
        backpressure_strategy=BackpressureStrategy.SAMPLE,
        sampling_rate=10,              # Keep 1/10 under pressure
    ),
)

async for event in pipeline.execute_streaming(desc, criteria, budget):
    await websocket.send(event)
```

**Benefits:**
- Bounded queues prevent memory exhaustion
- Automatic event sampling under pressure
- Circuit breaker prevents cascade failures
- Memory monitoring with automatic GC

---

## Configuration File Migration

### `.orchestrator.yml` Changes

Add new resilience section:

```yaml
# v6.0 — New resilience section
resilience:
  # Event store
  event_store:
    type: resilient  # or "sqlite" for v5.x behavior
    primary_path: .events/primary.db
    replica_paths:
      - .events/replica1.db
      - .events/replica2.db
  
  # Plugin isolation
  plugin_security:
    level: secure  # or "standard" for v5.x behavior
    enable_seccomp: true
    enable_landlock: true
    enable_capabilities: true
  
  # Streaming
  streaming:
    backpressure: true
    max_queue_size: 1000
    max_memory_mb: 1024
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 30.0
```

---

## Feature Flags

Control resilient features via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_RESILIENT_STORE` | `false` | Enable ResilientEventStore |
| `ORCHESTRATOR_SECURE_PLUGINS` | `false` | Enable SecureIsolatedRuntime |
| `ORCHESTRATOR_BACKPRESSURE` | `false` | Enable ResilientStreamingPipeline |
| `ORCHESTRATOR_STORE_REPLICAS` | `[]` | JSON array of replica paths |
| `ORCHESTRATOR_SECCOMP` | `true` | Enable seccomp (Linux only) |
| `ORCHESTRATOR_LANDLOCK` | `true` | Enable Landlock (Linux 5.13+) |

---

## Testing the Migration

### 1. Verify Event Store Replication

```python
import asyncio
from orchestrator.events import TaskCompletedEvent
from orchestrator.events_resilient import ResilientEventStore

async def test_replication():
    store = ResilientEventStore(
        primary_path=".events/primary.db",
        replica_paths=[".events/replica1.db"],
    )
    
    # Write event
    event = TaskCompletedEvent(
        task_id="test-1",
        project_id="test-proj",
        result={"test": True},
    )
    await store.append(event)
    await asyncio.sleep(0.2)
    
    # Verify in replica
    replica_events = await store.replicas[0].get_events()
    assert len(replica_events) == 1
    print("✅ Replication working")
    
    await store.close()

asyncio.run(test_replication())
```

### 2. Verify Security Layers

```python
from orchestrator.plugin_isolation_secure import SecureIsolatedRuntime

runtime = SecureIsolatedRuntime(SecureIsolationConfig())

print("Available security layers:")
for layer, available in runtime.features.items():
    status = "✅" if available else "❌"
    print(f"  {status} {layer.name}")
```

### 3. Verify Backpressure

```python
from orchestrator.streaming_resilient import ResilientStreamingPipeline

pipeline = ResilientStreamingPipeline()

# Check health
health = pipeline.get_health()
print(f"Circuit state: {health['circuit_state']}")
print(f"Memory pressure: {health['memory_pressure']}")
```

---

## Rollback Plan

If issues occur, instant rollback to v5.x behavior:

### Option 1: Environment Variable
```bash
export ORCHESTRATOR_RESILIENT_STORE=false
export ORCHESTRATOR_SECURE_PLUGINS=false
export ORCHESTRATOR_BACKPRESSURE=false
```

### Option 2: Code Change
```python
# Switch back to v5.x classes
from orchestrator.events import SQLiteEventStore
from orchestrator.plugin_isolation import IsolatedPluginRuntime
from orchestrator.streaming import StreamingPipeline
```

### Option 3: Configuration
```yaml
resilience:
  event_store:
    type: sqlite  # Use v5.x store
  plugin_security:
    level: standard  # Use v5.x runtime
  streaming:
    backpressure: false  # Use v5.x pipeline
```

---

## Performance Considerations

| Feature | Overhead | Mitigation |
|---------|----------|------------|
| Event replication | +5-10ms per write | Async replication, non-blocking |
| Checksum calculation | +1-2ms per write | Hardware-accelerated SHA-256 |
| seccomp filtering | +0.5ms per syscall | One-time setup per process |
| Landlock sandbox | +1ms per file op | Cached ruleset |
| Memory monitoring | +0.1ms per check | 1-second polling interval |

**Total expected overhead: <5%** for typical workloads.

---

## Troubleshooting

### Issue: Event store replicas out of sync

**Symptom:** Integrity check shows missing events in replicas

**Solution:**
```python
# Force re-replication
report = await store.verify_integrity()
if report["missing_in_replicas"]:
    # Replay from primary to replicas
    events = await store.primary.get_events()
    for replica in store.replicas:
        for event in events:
            await replica.append(event)
```

### Issue: seccomp not available

**Symptom:** Warning in logs: "seccomp not available"

**Solution:**
- **Linux:** Install `python-seccomp` package
- **Windows/macOS:** Security layer gracefully skipped; process isolation still active

### Issue: Circuit breaker constantly open

**Symptom:** "Streaming circuit breaker is open" errors

**Solution:**
```python
# Check for underlying issues
health = pipeline.get_health()
print(f"Failures: {health['circuit_failures']}")

# Adjust thresholds
pipeline.circuit.config.failure_threshold = 10  # Less sensitive
pipeline.circuit.config.recovery_timeout = 60.0  # Longer recovery
```

### Issue: High memory usage despite backpressure

**Symptom:** Memory still grows under load

**Solution:**
```python
# Reduce queue size and sampling rate
MemoryPressureConfig(
    max_queue_size=100,      # Smaller buffer
    sampling_rate=5,         # More aggressive sampling
    gc_threshold_mb=200,     # Force GC earlier
)
```

---

## FAQ

**Q: Do I have to migrate?**  
A: No. v6.0 is fully backward compatible. Migrate when you're ready.

**Q: Can I use only some resilient features?**  
A: Yes. Mix and match — e.g., use ResilientEventStore but keep standard plugin isolation.

**Q: Are there any performance penalties?**  
A: <5% overhead for typical workloads. See Performance Considerations section.

**Q: Does this work on Windows?**  
A: Yes, but some Linux-specific security features (seccomp, Landlock) are skipped on Windows. Process isolation still works.

**Q: How do I monitor the resilient components?**  
A: Use the health endpoints:
```python
# Event store
report = await store.verify_integrity()

# Streaming
health = pipeline.get_health()
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/your-org/orchestrator/issues)
- **Documentation:** [CAPABILITIES.md](CAPABILITIES.md)
- **Stress Test Report:** [ADVERSARIAL_STRESS_TEST_REPORT.md](ADVERSARIAL_STRESS_TEST_REPORT.md)

---

**Migration complete!** Your orchestrator is now resilient against black swan events.
