# Capability Usage Logging

## Overview

The Capability Logging system tracks usage of all system capabilities for analytics, debugging, and optimization purposes.

## Quick Start

```python
from orchestrator import log_capability, CapabilityType

# Log a capability usage
log_capability(
    capability=CapabilityType.TASK_STARTED,
    task_type="code_generation",
    model="GPT_4O",
    duration_ms=1500,
    success=True,
    details={"tokens": 500}
)
```

## Capability Types

### Model Routing
- `ROUTING_DECISION` - Model selection made
- `FALLBACK_TRIGGERED` - Fallback chain activated
- `COST_OPTIMIZATION` - Cost-based optimization applied

### Task Execution
- `TASK_STARTED` - Task execution began
- `TASK_COMPLETED` - Task finished successfully
- `TASK_FAILED` - Task failed

### Quality Assurance
- `VALIDATION_PASSED` - Output validation succeeded
- `VALIDATION_FAILED` - Output validation failed
- `CRITIQUE_ROUND` - Critique iteration completed
- `REVISION_ROUND` - Revision iteration completed

### Cache & Performance
- `CACHE_HIT` - Data found in cache
- `CACHE_MISS` - Data not in cache
- `SEMANTIC_CACHE_USED` - Semantic cache utilized

## Automatic Logging with Decorator

```python
from orchestrator import log_capability_use, CapabilityType

@log_capability_use(CapabilityType.CODEBASE_ANALYSIS)
def analyze_project(path):
    # Your code here
    return result
```

## Log File Format

**Location:** `logs/capabilities/capabilities_YYYY-MM-DD.jsonl`

**Format:**
```json
{
  "timestamp": "2026-02-26T10:30:00Z",
  "capability": "TASK_STARTED",
  "task_type": "code_generation",
  "model": "GPT_4O",
  "project_id": "proj_123",
  "duration_ms": 1500,
  "success": true,
  "details": {"tokens": 500}
}
```

## Statistics

```python
from orchestrator import get_capability_logger

logger = get_capability_logger()
stats = logger.get_stats()

print(f"Total events: {stats['total_events']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"By capability: {stats['by_capability']}")
```

## Configuration

Default settings:
- **Flush interval:** 10 events
- **Log rotation:** Daily
- **Retention:** 30 days
- **Location:** `./logs/capabilities/`

## Integration with Telemetry

Capability logs complement the existing telemetry system:
- **Telemetry:** Real-time metrics and performance
- **Capability Logs:** Detailed usage tracking and audit trail
