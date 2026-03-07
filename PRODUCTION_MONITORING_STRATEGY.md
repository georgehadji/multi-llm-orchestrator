# PRODUCTION MONITORING STRATEGY

**System**: AI Orchestrator v6.0  
**Date**: 2026-03-07  
**Classification**: PRODUCTION-READY

---

## 1. CRITICAL PATH MONITORING

### 1.1 Task Execution Pipeline

**Path**: `Orchestrator.run_project()` → `_execute_all()` → `_execute_task()` → `UnifiedClient.chat_completion()`

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Latency (per task)** | < 30s | 15s | +15s | WARNING |
| **p99 Latency (per task)** | < 60s | 25s | +35s | CRITICAL |
| **Error Rate** | < 2% | 0.5% | +1.5% | CRITICAL |
| **Memory Budget** | < 512 MB | 256 MB | +256 MB | WARNING |
| **CPU Budget** | < 80% | 40% | +40% | WARNING |
| **Timeout Budget** | 300s total | 180s | +120s | CRITICAL |

**Rollback Trigger**:
```
IF (p95_latency > baseline + 15s) AND (duration > 5 minutes) → ROLLBACK
IF (error_rate > 2%) AND (duration > 2 minutes) → IMMEDIATE ROLLBACK
```

**Measurement Points**:
```python
# In orchestrator/engine.py
async def _execute_task(self, task: Task) -> TaskResult:
    start_time = time.perf_counter()
    try:
        result = await self._execute_task(task)
        # METRIC: task_execution_duration.observe(time.perf_counter() - start_time)
        # METRIC: task_execution_total.inc(status=result.status.value)
        return result
    except Exception as e:
        # METRIC: task_execution_errors.inc(type=type(e).__name__)
        raise
```

---

### 1.2 LLM API Call Path

**Path**: `UnifiedClient.chat_completion()` → Provider SDK → External API

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Latency (OpenAI)** | < 5s | 2s | +3s | WARNING |
| **p95 Latency (Anthropic)** | < 8s | 4s | +4s | WARNING |
| **p95 Latency (Google)** | < 6s | 3s | +3s | WARNING |
| **Error Rate (per provider)** | < 3% | 1% | +2% | CRITICAL |
| **Rate Limit Hits** | < 10/hour | 2/hour | +8/hour | WARNING |
| **Token Usage Rate** | < 1M tokens/min | 500K/min | +500K/min | WARNING |
| **Cost Rate** | < $10/hour | $5/hour | +$5/hour | CRITICAL |

**Rollback Trigger**:
```
IF (error_rate[provider] > 3%) AND (duration > 3 minutes) → DISABLE_PROVIDER
IF (cost_rate > $10/hour) AND (duration > 10 minutes) → THROTTLE
```

**Circuit Breaker Integration**:
```python
# In orchestrator/adaptive_router.py
class ModelState(Enum):
    HEALTHY = "healthy"      # error_rate < 1%
    DEGRADED = "degraded"    # 1% <= error_rate < 3%
    DISABLED = "disabled"    # error_rate >= 3%

# Rollback: Auto-transition to DISABLED on sustained errors
```

---

### 1.3 State Persistence Path

**Path**: `StateManager.save_checkpoint()` → SQLite WAL → Disk

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Write Latency** | < 100ms | 50ms | +50ms | WARNING |
| **p99 Write Latency** | < 500ms | 200ms | +300ms | CRITICAL |
| **Error Rate** | < 0.1% | 0.01% | +0.09% | CRITICAL |
| **DB Size Growth** | < 100MB/hour | 50MB/hour | +50MB/hour | WARNING |
| **WAL File Size** | < 1GB | 500MB | +500MB | WARNING |
| **Connection Pool Usage** | < 80% | 40% | +40% | WARNING |

**Rollback Trigger**:
```
IF (write_latency_p99 > 500ms) AND (duration > 5 minutes) → SWITCH_TO_IN_MEMORY
IF (error_rate > 0.1%) AND (duration > 1 minute) → EMERGENCY_BACKUP
```

---

### 1.4 Event Bus Path

**Path**: `UnifiedEventBus.publish()` → Handlers → SQLite Event Store

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Publish Latency** | < 50ms | 20ms | +30ms | WARNING |
| **Event Queue Depth** | < 1000 | 200 | +800 | WARNING |
| **Handler Error Rate** | < 1% | 0.2% | +0.8% | CRITICAL |
| **Event Loss Rate** | 0% | 0% | +0% | CRITICAL |
| **Memory Usage** | < 256 MB | 128 MB | +128 MB | WARNING |

**Rollback Trigger**:
```
IF (queue_depth > 1000) AND (duration > 3 minutes) → DISABLE_NON_CRITICAL_HANDLERS
IF (event_loss_rate > 0%) → IMMEDIATE_INVESTIGATION
```

---

### 1.5 A2A Communication Path

**Path**: `A2AManager.send_task()` → Message Queue → Handler → Response

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Round-Trip** | < 10s | 5s | +5s | WARNING |
| **Timeout Rate** | < 5% | 1% | +4% | WARNING |
| **Queue Depth (per agent)** | < 100 | 20 | +80 | WARNING |
| **Deadlock Incidents** | 0 | 0 | +0 | CRITICAL |
| **Orphaned Response Rate** | 0% | 0% | +0% | CRITICAL |

**Rollback Trigger**:
```
IF (timeout_rate > 5%) AND (duration > 5 minutes) → INCREASE_TIMEOUTS
IF (deadlock_incidents > 0) → RESTART_A2A_MANAGER
```

---

### 1.6 Memory Tier Operations

**Path**: `MemoryTierManager.store()` → JSONL Write → BM25 Index Update

| Metric | Threshold | Baseline | τ (Tau) | Alert Level |
|--------|-----------|----------|---------|-------------|
| **p95 Store Latency** | < 200ms | 100ms | +100ms | WARNING |
| **p95 Retrieve Latency** | < 500ms | 250ms | +250ms | WARNING |
| **Tier Migration Rate** | < 100/hour | 50/hour | +50/hour | WARNING |
| **BM25 Index Size** | < 5GB | 2GB | +3GB | WARNING |
| **Memory Usage (HOT tier)** | < 1GB | 500MB | +500MB | WARNING |

**Rollback Trigger**:
```
IF (retrieve_latency_p95 > 500ms) AND (duration > 10 minutes) → DISABLE_BM25
IF (memory_usage > 1GB) AND (duration > 5 minutes) → FORCE_COLD_MIGRATION
```

---

## 2. ROLLBACK TRIGGER DEFINITIONS

### 2.1 Rollback Threshold Matrix

| Metric Category | Baseline | τ (Tau) | Threshold | Duration | Action |
|-----------------|----------|---------|-----------|----------|--------|
| **Latency (p95)** | Task-specific | +100% | baseline + τ | 5 min | Scale up → Rollback |
| **Error Rate** | < 1% | +2% | 3% | 2 min | Immediate Rollback |
| **Memory** | 256 MB | +256 MB | 512 MB | 5 min | GC → Restart → Rollback |
| **CPU** | 40% | +40% | 80% | 5 min | Throttle → Rollback |
| **Cost** | $5/hour | +$5/hour | $10/hour | 10 min | Throttle → Rollback |
| **Queue Depth** | 200 | +800 | 1000 | 3 min | Disable handlers → Rollback |
| **Timeout Rate** | 1% | +4% | 5% | 5 min | Increase timeouts → Rollback |

### 2.2 Rollback Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROLLBACK DECISION TREE                              │
└─────────────────────────────────────────────────────────────────────────────┘

Metric Exceeds Threshold?
    │
    ├─ NO ──→ Continue monitoring
    │
    └─ YES
        │
        ├─ Duration < X minutes?
        │   │
        │   └─ YES ──→ Log warning, continue monitoring
        │
        └─ Duration >= X minutes?
            │
            ├─ Error Rate > 3%?
            │   │
            │   └─ YES ──→ IMMEDIATE ROLLBACK (data loss risk)
            │
            ├─ Memory > 512 MB?
            │   │
            │   ├─ YES ──→ Force GC
            │   │   │
            │   │   └─ Still > 512 MB after 2 min?
            │   │       │
            │   │       └─ YES ──→ ROLLBACK
            │   │
            │   └─ NO ──→ Continue
            │
            ├─ Latency > baseline + τ?
            │   │
            │   ├─ YES ──→ Scale up resources
            │   │   │
            │   │   └─ Still high after 5 min?
            │   │       │
            │   │       └─ YES ──→ ROLLBACK
            │   │
            │   └─ NO ──→ Continue
            │
            └─ Cost Rate > $10/hour?
                │
                └─ YES ──→ Throttle LLM calls
                    │
                    └─ Still high after 10 min?
                        │
                        └─ YES ──→ ROLLBACK
```

---

## 3. FAILURE SIMULATION SCENARIOS

### 3.1 High Load Simulation

**Scenario**: Sudden 10x traffic spike

**Simulation Script**:
```python
#!/usr/bin/env python
"""
High Load Simulation Test
Simulates 10x normal traffic to test monitoring and rollback triggers.
"""

import asyncio
import time
from orchestrator import Orchestrator, Budget, Task, TaskType

async def simulate_high_load():
    """Generate 10x normal load for 15 minutes."""
    
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    
    # Normal baseline: 10 tasks/minute
    # High load: 100 tasks/minute
    
    tasks_created = 0
    start_time = time.time()
    
    while time.time() - start_time < 900:  # 15 minutes
        # Create burst of 10 tasks every 6 seconds (100/minute)
        for _ in range(10):
            task = Task(
                id=f"load_test_{tasks_created}",
                prompt="Write a simple Python function",
                type=TaskType.CODE_GEN,
            )
            asyncio.create_task(orch._execute_task(task))
            tasks_created += 1
        
        await asyncio.sleep(6)
        
        # Monitor metrics
        if tasks_created % 100 == 0:
            print(f"Tasks created: {tasks_created}")
            print(f"Background tasks: {len(orch._background_tasks)}")
            print(f"Results count: {len(orch.results)}")
    
    # Cleanup
    await orch.close()
    
    return {
        "total_tasks": tasks_created,
        "duration_seconds": time.time() - start_time,
        "tasks_per_minute": tasks_created / (time.time() - start_time) * 60,
    }

if __name__ == "__main__":
    results = asyncio.run(simulate_high_load())
    print(f"\nSimulation Results: {results}")
```

**Expected Metrics Under Load**:

| Metric | Normal | High Load | Threshold | Action |
|--------|--------|-----------|-----------|--------|
| Tasks/minute | 10 | 100 | 50 | Scale up |
| p95 Latency | 15s | 45s | 30s | WARNING |
| Error Rate | 0.5% | 2.5% | 2% | CRITICAL |
| Memory | 256 MB | 450 MB | 512 MB | WARNING |
| CPU | 40% | 75% | 80% | WARNING |

**Rollback Decision**:
- If error rate > 2% for > 2 minutes → ROLLBACK
- If latency > 30s for > 5 minutes → ROLLBACK

---

### 3.2 Partial Dependency Failure Simulation

**Scenario**: One LLM provider (e.g., OpenAI) becomes unavailable

**Simulation Script**:
```python
#!/usr/bin/env python
"""
Partial Dependency Failure Simulation
Simulates OpenAI API failures to test circuit breaker and fallback.
"""

import asyncio
from unittest.mock import AsyncMock, patch
from orchestrator import Orchestrator, Budget, Task, TaskType
from orchestrator.models import Model

async def simulate_provider_failure():
    """Simulate OpenAI API failures for 10 minutes."""
    
    orch = Orchestrator(budget=Budget(max_usd=50.0))
    
    # Track failures
    failures = 0
    fallbacks = 0
    successes = 0
    
    for i in range(100):
        task = Task(
            id=f"failure_test_{i}",
            prompt="Write a Python function",
            type=TaskType.CODE_GEN,
        )
        
        # Simulate OpenAI failure 50% of the time
        if i % 2 == 0:
            with patch.object(orch.client, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = Exception("OpenAI API unavailable")
                
                try:
                    result = await orch._execute_task(task)
                    if result.status.value == "completed":
                        fallbacks += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        else:
            result = await orch._execute_task(task)
            successes += 1
        
        # Monitor circuit breaker state
        if i % 10 == 0:
            print(f"Successes: {successes}, Fallbacks: {fallbacks}, Failures: {failures}")
            print(f"OpenAI health: {orch.api_health[Model.GPT_4O]}")
    
    await orch.close()
    
    return {
        "successes": successes,
        "fallbacks": fallbacks,
        "failures": failures,
        "fallback_rate": fallbacks / (successes + fallbacks + failures),
    }

if __name__ == "__main__":
    results = asyncio.run(simulate_provider_failure())
    print(f"\nSimulation Results: {results}")
```

**Expected Metrics Under Partial Failure**:

| Metric | Normal | Partial Failure | Threshold | Action |
|--------|--------|-----------------|-----------|--------|
| OpenAI Error Rate | 0.5% | 50% | 3% | CIRCUIT BREAKER |
| Fallback Rate | 5% | 45% | 30% | WARNING |
| Overall Error Rate | 0.5% | 5% | 2% | CRITICAL |
| p95 Latency | 15s | 35s | 30s | WARNING |

**Circuit Breaker Behavior**:
```
OpenAI: HEALTHY → DEGRADED (after 3 consecutive failures) → DISABLED (after 10 failures)
Fallback: Anthropic/Google activated automatically
```

**Rollback Decision**:
- If overall error rate > 2% for > 2 minutes → DISABLE_OPENAI
- If fallback rate > 30% for > 5 minutes → INVESTIGATE

---

### 3.3 Malformed Input Burst Simulation

**Scenario**: Sudden burst of malformed/invalid inputs

**Simulation Script**:
```python
#!/usr/bin/env python
"""
Malformed Input Burst Simulation
Tests input validation and error handling under attack.
"""

import asyncio
from orchestrator import Orchestrator, Budget, Task, TaskType

MALFORMED_INPUTS = [
    "",  # Empty
    "x" * 100000,  # Very long
    "<script>alert('xss')</script>",  # XSS attempt
    "'; DROP TABLE tasks; --",  # SQL injection attempt
    "\x00\x01\x02\x03",  # Binary data
    "正常な日本語のテキスト" * 1000,  # Unicode overflow
]

async def simulate_malformed_inputs():
    """Send burst of malformed inputs."""
    
    orch = Orchestrator(budget=Budget(max_usd=20.0))
    
    validation_errors = 0
    processing_errors = 0
    successes = 0
    
    for i, malformed_input in enumerate(MALFORMED_INPUTS * 10):  # 60 total
        task = Task(
            id=f"malformed_{i}",
            prompt=malformed_input,
            type=TaskType.CODE_GEN,
        )
        
        try:
            result = await orch._execute_task(task)
            
            if result.status.value == "failed":
                validation_errors += 1
            else:
                successes += 1
                
        except Exception as e:
            processing_errors += 1
            print(f"Processing error: {type(e).__name__}: {e}")
        
        # Monitor every 10 inputs
        if i % 10 == 0:
            print(f"Successes: {successes}, Validation errors: {validation_errors}, Processing errors: {processing_errors}")
    
    await orch.close()
    
    return {
        "successes": successes,
        "validation_errors": validation_errors,
        "processing_errors": processing_errors,
        "error_rate": (validation_errors + processing_errors) / 60,
    }

if __name__ == "__main__":
    results = asyncio.run(simulate_malformed_inputs())
    print(f"\nSimulation Results: {results}")
```

**Expected Metrics Under Malformed Input**:

| Metric | Normal | Malformed Burst | Threshold | Action |
|--------|--------|-----------------|-----------|--------|
| Validation Error Rate | 1% | 80% | 50% | RATE LIMIT |
| Processing Error Rate | 0.5% | 5% | 2% | CRITICAL |
| p95 Latency | 15s | 20s | 30s | OK |
| Memory Spike | 256 MB | 350 MB | 512 MB | OK |

**Rollback Decision**:
- If processing error rate > 2% for > 2 minutes → ENABLE_INPUT_SANITIZATION
- If validation error rate > 50% for > 5 minutes → RATE_LIMIT_INPUTS

---

## 4. FAILURE CONTAINMENT PLAN

### 4.1 Containment Strategies by Failure Type

| Failure Type | Detection | Containment | Recovery Time |
|--------------|-----------|-------------|---------------|
| **High Latency** | p95 > threshold | Scale up, throttle | 5-10 minutes |
| **High Error Rate** | Error rate > 3% | Circuit breaker, fallback | 2-5 minutes |
| **Memory Leak** | Memory > 512 MB | Force GC, restart | 5-15 minutes |
| **CPU Exhaustion** | CPU > 80% | Throttle, scale up | 5-10 minutes |
| **Cost Overrun** | Cost > $10/hour | Throttle LLM calls | Immediate |
| **Queue Backlog** | Depth > 1000 | Disable handlers | 3-5 minutes |
| **DB Corruption** | Write errors > 0.1% | Switch to in-memory | 10-30 minutes |
| **Provider Outage** | Provider error > 3% | Fallback chain | Immediate |

### 4.2 Containment Runbook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FAILURE CONTAINMENT RUNBOOK                         │
└─────────────────────────────────────────────────────────────────────────────┘

FAILURE DETECTED
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: CLASSIFY FAILURE                                                    │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Check metric type:                                                          │
│   ├─ Latency spike     → Go to Section 2.1                                  │
│   ├─ Error rate spike  → Go to Section 2.2                                  │
│   ├─ Memory/CPU spike  → Go to Section 2.3                                  │
│   ├─ Cost overrun      → Go to Section 2.4                                  │
│   └─ Queue backlog     → Go to Section 2.5                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: IMMEDIATE CONTAINMENT (< 1 minute)                                  │
│ ─────────────────────────────────────────────────────────────────────────── │
│ 2.1 Latency Containment:                                                    │
│   ├─ Enable response caching                                                │
│   ├─ Reduce max_tokens for LLM calls                                        │
│   └─ Enable simpler prompt templates                                        │
│                                                                             │
│ 2.2 Error Rate Containment:                                                 │
│   ├─ Activate circuit breaker for failing providers                         │
│   ├─ Switch to fallback providers                                           │
│   └─ Enable request retry with exponential backoff                          │
│                                                                             │
│ 2.3 Resource Containment:                                                   │
│   ├─ Force garbage collection                                               │
│   ├─ Clear semantic cache                                                   │
│   ├─ Migrate HOT memories to WARM/COLD                                      │
│   └─ Kill long-running background tasks                                     │
│                                                                             │
│ 2.4 Cost Containment:                                                       │
│   ├─ Switch to cheaper models (GPT-4 → GPT-4o-mini)                         │
│   ├─ Reduce max_iterations per task                                         │
│   ├─ Disable cross-review                                                   │
│   └─ Enable token optimization (RTK)                                        │
│                                                                             │
│ 2.5 Queue Containment:                                                      │
│   ├─ Disable non-critical event handlers                                    │
│   ├─ Increase queue processing concurrency                                  │
│   └─ Drop low-priority messages                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: ASSESS CONTAINMENT SUCCESS (after 2-5 minutes)                      │
│ ─────────────────────────────────────────────────────────────────────────── │
│ IF metrics improving:                                                       │
│   ├─ Continue monitoring                                                    │
│   ├─ Document incident                                                      │
│   └─ Plan post-mortem                                                       │
│                                                                             │
│ IF metrics NOT improving:                                                   │
│   ├─ Escalate to on-call engineer                                           │
│   ├─ Prepare rollback                                                       │
│   └─ Notify stakeholders                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ROLLBACK DECISION (if containment fails)                            │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Rollback triggers:                                                          │
│   ├─ Error rate > 3% for > 2 minutes                                        │
│   ├─ Latency > 2x baseline for > 5 minutes                                  │
│   ├─ Memory > 512 MB for > 5 minutes                                        │
│   ├─ Cost > $10/hour for > 10 minutes                                       │
│   └─ Any data loss detected                                                 │
│                                                                             │
│ Rollback procedure:                                                         │
│   1. Stop accepting new projects                                            │
│   2. Wait for in-flight tasks to complete (max 5 min)                       │
│   3. Save all checkpoints                                                   │
│   4. Deploy previous version                                                │
│   5. Restore from backup if needed                                          │
│   6. Verify health checks                                                   │
│   7. Resume accepting projects                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. RECOVERY STEPS

### 5.1 Immediate Recovery (< 5 minutes)

```bash
# 1. Force garbage collection
curl -X POST http://localhost:8000/admin/gc

# 2. Clear caches
curl -X POST http://localhost:8000/admin/cache/clear

# 3. Reset circuit breakers
curl -X POST http://localhost:8000/admin/circuit-breakers/reset

# 4. Kill long-running tasks
curl -X POST http://localhost:8000/admin/tasks/kill?max_duration=300

# 5. Check health
curl http://localhost:8000/health
```

### 5.2 Short-term Recovery (5-30 minutes)

```bash
# 1. Restart orchestrator service
systemctl restart orchestrator

# 2. Verify database integrity
sqlite3 ~/.orchestrator_cache/state.db "PRAGMA integrity_check;"

# 3. Rebuild BM25 index if corrupted
python -c "from orchestrator.bm25_search import BM25Search; bm25 = BM25Search(); bm25.rebuild_index()"

# 4. Restore from backup if needed
python -m orchestrator.backup restore --latest

# 5. Verify all components
python -m orchestrator.health_check
```

### 5.3 Long-term Recovery (30+ minutes)

```bash
# 1. Full system restore from backup
python -m orchestrator.backup restore --backup-id=<backup_id>

# 2. Rebuild all indexes
python -m orchestrator.maintenance rebuild-all

# 3. Run full test suite
pytest tests/ -v

# 4. Gradual traffic ramp-up
python -m orchestrator.traffic_control set --rate-limit=10/minute
# Gradually increase...
python -m orchestrator.traffic_control set --rate-limit=100/minute
```

---

## 6. DEPLOYMENT CHECKLIST

### 6.1 Pre-Deployment Checklist

| Item | Status | Verified By | Date |
|------|--------|-------------|------|
| **Code Review** | ☐ | | |
| - All changes reviewed by 2+ engineers | ☐ | | |
| - Security review completed | ☐ | | |
| - Performance impact assessed | ☐ | | |
| **Testing** | ☐ | | |
| - Unit tests passing (>90% coverage) | ☐ | | |
| - Integration tests passing | ☐ | | |
| - Load tests completed | ☐ | | |
| - Failure simulation completed | ☐ | | |
| **Monitoring** | ☐ | | |
| - Metrics endpoints verified | ☐ | | |
| - Alert thresholds configured | ☐ | | |
| - Dashboard updated | ☐ | | |
| - Log aggregation verified | ☐ | | |
| **Rollback Preparation** | ☐ | | |
| - Previous version tagged | ☐ | | |
| - Backup completed | ☐ | | |
| - Rollback procedure tested | ☐ | | |
| **Documentation** | ☐ | | |
| - Release notes written | ☐ | | |
| - Runbook updated | ☐ | | |
| - API docs updated | ☐ | | |

### 6.2 Deployment Execution Checklist

| Step | Command | Expected Output | Status |
|------|---------|-----------------|--------|
| **1. Health Check (pre)** | `curl http://localhost:8000/health` | `{"status": "healthy"}` | ☐ |
| **2. Stop Traffic** | `curl -X POST http://localhost:8000/admin/traffic/stop` | `{"status": "stopped"}` | ☐ |
| **3. Wait for Drain** | `curl http://localhost:8000/admin/tasks/count` | `{"active": 0}` | ☐ |
| **4. Backup State** | `python -m orchestrator.backup create` | `Backup ID: <id>` | ☐ |
| **5. Deploy** | `kubectl apply -f deployment.yaml` | `deployment configured` | ☐ |
| **6. Wait for Ready** | `kubectl rollout status deployment/orchestrator` | `rollout complete` | ☐ |
| **7. Health Check (post)** | `curl http://localhost:8000/health` | `{"status": "healthy"}` | ☐ |
| **8. Verify Metrics** | `curl http://localhost:8000/metrics` | Prometheus metrics | ☐ |
| **9. Resume Traffic** | `curl -X POST http://localhost:8000/admin/traffic/resume` | `{"status": "resumed"}` | ☐ |
| **10. Monitor** | Check dashboard for 15 minutes | No alerts | ☐ |

### 6.3 Post-Deployment Checklist

| Item | Status | Verified By | Date |
|------|--------|-------------|------|
| **Monitoring** | ☐ | | |
| - Error rate < 1% for 30 minutes | ☐ | | |
| - Latency p95 < threshold for 30 minutes | ☐ | | |
| - Memory stable | ☐ | | |
| - CPU stable | ☐ | | |
| **Functionality** | ☐ | | |
| - Sample task executes successfully | ☐ | | |
| - All LLM providers responding | ☐ | | |
| - State persistence working | ☐ | | |
| - Event bus operational | ☐ | | |
| **Rollback Decision Point (30 min)** | ☐ | | |
| - IF all checks pass → Deployment successful | ☐ | | |
| - IF any check fails → Initiate rollback | ☐ | | |

---

## 7. MONITORING DASHBOARD SPECIFICATION

### 7.1 Required Panels

| Panel | Metrics | Refresh | Alert |
|-------|---------|---------|-------|
| **Task Execution** | Tasks/min, p95 latency, error rate | 10s | Error rate > 2% |
| **LLM Providers** | Per-provider latency, error rate, cost | 10s | Any provider > 3% errors |
| **Resources** | Memory, CPU, disk I/O | 30s | Memory > 512 MB |
| **Queues** | Event queue depth, A2A queue depth | 10s | Depth > 1000 |
| **Cost** | Hourly cost, projected daily cost | 1 min | > $10/hour |
| **Circuit Breakers** | Provider states (HEALTHY/DEGRADED/DISABLED) | 10s | Any DISABLED |
| **Database** | Write latency, connection pool usage | 30s | Latency > 500ms |

### 7.2 Alert Routing

| Alert Level | Channels | Response Time | Escalation |
|-------------|----------|---------------|------------|
| **INFO** | Slack #monitoring | 1 hour | None |
| **WARNING** | Slack #alerts, Email | 15 minutes | 30 min → On-call |
| **CRITICAL** | Slack, Email, PagerDuty | 5 minutes | 10 min → Team lead |
| **EMERGENCY** | All channels + Phone | Immediate | 5 min → VP Engineering |

---

## 8. APPENDIX: METRIC COLLECTION IMPLEMENTATION

### 8.1 Prometheus Metrics Setup

```python
# In orchestrator/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Task execution metrics
TASK_EXECUTION_DURATION = Histogram(
    'task_execution_duration_seconds',
    'Task execution duration',
    ['task_type', 'model'],
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

TASK_EXECUTION_TOTAL = Counter(
    'task_execution_total',
    'Total task executions',
    ['task_type', 'model', 'status'],
)

TASK_EXECUTION_ERRORS = Counter(
    'task_execution_errors_total',
    'Task execution errors',
    ['task_type', 'error_type'],
)

# LLM API metrics
LLM_API_LATENCY = Histogram(
    'llm_api_latency_seconds',
    'LLM API call latency',
    ['provider', 'model'],
    buckets=[0.5, 1, 2, 5, 10, 30],
)

LLM_API_ERRORS = Counter(
    'llm_api_errors_total',
    'LLM API errors',
    ['provider', 'error_type'],
)

LLM_TOKEN_USAGE = Counter(
    'llm_tokens_total',
    'Token usage',
    ['provider', 'type'],  # type: input, output
)

# Resource metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage',
    ['component'],
)

BACKGROUND_TASK_COUNT = Gauge(
    'background_tasks_count',
    'Number of background tasks',
)

A2A_QUEUE_DEPTH = Gauge(
    'a2a_queue_depth',
    'A2A message queue depth',
    ['agent_id'],
)

# Cost metrics
COST_USD = Counter(
    'cost_usd_total',
    'Total cost in USD',
    ['category'],  # category: generation, cross_review, evaluation
)
```

### 8.2 Health Check Endpoint

```python
# In orchestrator/health.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthStatus(BaseModel):
    status: str
    version: str
    components: dict
    uptime_seconds: float

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check."""
    
    components = {
        "api": {"status": "healthy", "latency_ms": 0},
        "database": {"status": "healthy", "latency_ms": 0},
        "cache": {"status": "healthy", "hit_rate": 0.0},
        "llm_providers": {},
    }
    
    # Check database
    try:
        start = time.perf_counter()
        await state_manager.health_check()
        components["database"]["latency_ms"] = (time.perf_counter() - start) * 1000
    except Exception as e:
        components["database"]["status"] = "unhealthy"
        components["database"]["error"] = str(e)
    
    # Check LLM providers
    for provider in ["openai", "anthropic", "google"]:
        try:
            # Quick ping test
            await api_clients.check_provider(provider)
            components["llm_providers"][provider] = {"status": "healthy"}
        except Exception as e:
            components["llm_providers"][provider] = {
                "status": "unhealthy",
                "error": str(e),
            }
    
    # Determine overall status
    unhealthy = [
        name for name, comp in components.items()
        if comp.get("status") == "unhealthy"
    ]
    
    status = "unhealthy" if unhealthy else "healthy"
    
    return HealthStatus(
        status=status,
        version=__version__,
        components=components,
        uptime_seconds=time.perf_counter() - START_TIME,
    )
```

---

**Document End**

*This monitoring strategy should be reviewed and updated quarterly or after any major incident.*
