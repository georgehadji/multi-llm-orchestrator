# Meta-Optimization V2 — Safe Strategy Deployment

> **Production-Ready Meta-Optimization** with A/B Testing, Human-in-the-Loop Approval, and Gradual Rollout.

## Overview

Meta-Optimization V2 adds **safety mechanisms** to strategy optimization:

| Feature | Purpose | Risk Reduction |
|---------|---------|----------------|
| **A/B Testing** | Live experiments with statistical validation | 🔴 High |
| **HITL Approval** | Human oversight for structural changes | 🔴 High |
| **Gradual Rollout** | Staged deployment with auto-rollback | 🔴 High |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Meta-Optimization V2 Pipeline                        │
│                                                                         │
│  Proposal → [Impact Assessment] → ──────────────────────────────────┐  │
│                                                                       ▼  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Impact Level: LOW + High Confidence                               │  │
│  │ → Auto-Approve                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Impact Level: MEDIUM/HIGH                                         │  │
│  │ → A/B Test (10% traffic) → Statistical Analysis → Rollout        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Impact Level: STRUCTURAL                                          │  │
│  │ → HITL Approval → Rollout                                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from orchestrator.meta_v2_integration import MetaOptimizationV2, MetaV2Config
from orchestrator.meta_orchestrator import ExecutionArchive

# Initialize
archive = ExecutionArchive()
config = MetaV2Config(
    ab_testing_enabled=True,
    hitl_enabled=True,
    rollout_enabled=True,
)

meta_v2 = MetaOptimizationV2(orchestrator, archive, config)

# After each project completion
await meta_v2.record_project_completion(trajectory)

# Periodic optimization
outcomes = await meta_v2.maybe_optimize()

for outcome in outcomes:
    print(f"Proposal: {outcome.proposal.description}")
    print(f"Decision: {outcome.decision.value}")
    print(f"Reason: {outcome.reason}")
```

## A/B Testing

### How It Works

1. **Create Experiment** — Proposal becomes treatment variant
2. **Traffic Routing** — 10% of projects routed to treatment
3. **Outcome Recording** — Track success, score, cost per execution
4. **Statistical Analysis** — T-test for significance
5. **Decision** — Adopt if significant improvement, reject otherwise

### Configuration

```python
from orchestrator.ab_testing import ABTestingEngine

ab_engine = ABTestingEngine(
    archive,
    storage_path=Path.home() / ".orchestrator_cache" / "ab_testing",
)

# Create experiment
experiment = await ab_engine.create_experiment(
    proposal,
    traffic_split=0.1,      # 10% to treatment
    min_samples=30,          # Minimum per variant
    significance_level=0.05, # P-value threshold
)

# Route execution
variant = await ab_engine.route_execution(project_id)

# Record outcome
await ab_engine.record_outcome(
    experiment_id=experiment.experiment_id,
    variant=variant,
    project_id=project_id,
    success=True,
    score=0.9,
    cost_usd=0.01,
    latency_ms=1000,
)

# Analyze results
result = await ab_engine.analyze_results(experiment.experiment_id)
if result.recommendation == Recommendation.ADOPT:
    print(f"Adopt! p={result.p_value:.4f}, d={result.effect_size:.3f}")
```

### Statistical Methods

| Method | Purpose |
|--------|---------|
| **Welch's t-test** | Compare means (unequal variances) |
| **Cohen's d** | Effect size (0.2=small, 0.5=medium, 0.8=large) |
| **Confidence Interval** | Range of likely true difference |

### Interpretation

| P-Value | Confidence | Action |
|---------|------------|--------|
| < 0.01 | >99% | Strong adopt/reject |
| 0.01-0.05 | 95-99% | Moderate adopt/reject |
| > 0.05 | <95% | Inconclusive |

---

## Human-in-the-Loop (HITL)

### How It Works

1. **Submit Proposal** — Auto-classify impact level
2. **Auto-Approval Check** — Low-risk + high confidence → auto-approve
3. **Pending Queue** — Medium/high impact → human review
4. **Review Decision** — Approve or reject with notes
5. **Audit Trail** — Immutable log of all decisions

### Impact Levels

| Level | Criteria | Routing |
|-------|----------|---------|
| **LOW** | Template tweaks, <5% impact | Auto-approve if confidence >90% |
| **MEDIUM** | Model routing, 5-20% budget | A/B test |
| **HIGH** | Disable models, >20% budget | HITL approval |
| **STRUCTURAL** | Core system changes | HITL approval + audit |

### Configuration

```python
from orchestrator.hitl_workflow import HITLWorkflow, ApprovalConfig

config = ApprovalConfig(
    auto_approve_low_risk=True,
    approval_timeout_hours=72.0,
    notification_channels=["log", "file"],
    auto_approve_thresholds={
        "max_cost_impact": 0.05,   # 5% auto-approved
        "min_confidence": 0.9,     # 90% confidence auto-approved
    },
)

hitl = HITLWorkflow(config)

# Submit for approval
request = await hitl.submit_for_approval(proposal)

# Check status
if await hitl.is_approved(request.request_id):
    # Proceed
    pass

# Manual approval
await hitl.approve(request.request_id, "admin", "Looks good!")

# Manual rejection
await hitl.reject(request.request_id, "admin", "Too risky")
```

### Audit Trail

All decisions are logged to `.orchestrator_cache/hitl/audit_log.jsonl`:

```json
{
  "entry_id": "audit_1234567890",
  "timestamp": 1711234567.890,
  "event_type": "request_approved",
  "request_id": "approval_xyz",
  "details": {"reviewer": "admin", "notes": "Looks good!"},
  "signature": "abc123"
}
```

---

## Gradual Rollout

### How It Works

1. **Start Rollout** — Begin at 5% traffic
2. **Record Outcomes** — Track successes/failures
3. **Check Progress** — Monitor thresholds
4. **Advance Stage** — 5% → 25% → 50% → 100%
5. **Auto-Rollback** — On failure threshold

### Default Stages

| Stage | Traffic | Min Successes | Max Failures | Timeout |
|-------|---------|---------------|--------------|---------|
| 1 | 5% | 10 | 3 | 24h |
| 2 | 25% | 25 | 5 | 48h |
| 3 | 50% | 50 | 10 | 72h |
| 4 | 100% | - | - | - |

### Configuration

```python
from orchestrator.gradual_rollout import GradualRolloutManager, RolloutConfig, RolloutStage

config = RolloutConfig(
    stages=[
        RolloutStage(stage_index=0, percentage=5, min_successes=10, max_failures=3, timeout_hours=24),
        RolloutStage(stage_index=1, percentage=25, min_successes=25, max_failures=5, timeout_hours=48),
        RolloutStage(stage_index=2, percentage=50, min_successes=50, max_failures=10, timeout_hours=72),
        RolloutStage(stage_index=3, percentage=100, min_successes=0, max_failures=0, timeout_hours=0),
    ],
    auto_rollback_enabled=True,
)

rollout_mgr = GradualRolloutManager(archive, config)

# Start rollout
rollout = await rollout_mgr.start_rollout(proposal)

# Record execution
await rollout_mgr.record_execution(
    rollout_id=rollout.rollout_id,
    success=True,
    score=0.9,
    cost_usd=0.01,
    latency_ms=1000,
    project_id="project_123",
)

# Check progress
decision = await rollout_mgr.check_stage_progress(rollout.rollout_id)

if decision.decision == "advance":
    await rollout_mgr.advance_stage(rollout.rollout_id)
elif decision.decision == "rollback":
    await rollout_mgr.trigger_rollback(rollout.rollout_id, decision.reason)
```

### Rollback Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Max Failures** | Failures ≥ threshold | Immediate rollback |
| **Timeout** | Time in stage ≥ timeout | Advance or rollback |
| **Manual** | Human intervention | Pause/rollback |

---

## Integration Patterns

### Pattern 1: Full Pipeline (Recommended)

```python
from orchestrator.meta_v2_integration import MetaOptimizationV2, MetaV2Config

config = MetaV2Config(
    ab_testing_enabled=True,
    hitl_enabled=True,
    rollout_enabled=True,
    min_executions_for_optimization=50,
)

meta_v2 = MetaOptimizationV2(orchestrator, archive, config)

# Automatic routing based on impact
outcomes = await meta_v2.maybe_optimize()

# Outcomes automatically routed to:
# - Auto-approve (low impact)
# - A/B test (medium impact)
# - HITL (high/structural impact)
```

### Pattern 2: A/B Testing Only

```python
from orchestrator.ab_testing import ABTestingEngine

ab_engine = ABTestingEngine(archive)

# Test specific proposal
experiment = await ab_engine.create_experiment(proposal)

# Manual analysis
result = await ab_engine.analyze_results(experiment.experiment_id)
```

### Pattern 3: HITL Only

```python
from orchestrator.hitl_workflow import HITLWorkflow, ApprovalConfig

hitl = HITLWorkflow(ApprovalConfig(auto_approve_low_risk=False))

# All proposals require approval
request = await hitl.submit_for_approval(proposal)

# Wait for human decision
while not await hitl.is_approved(request.request_id):
    await asyncio.sleep(60)
```

---

## Monitoring

### Status Dashboard

```python
status = meta_v2.get_status()

print(f"Archive: {status['archive']['total_executions']} executions")
print(f"A/B Tests: {status['ab_testing']['active_experiments']} active")
print(f"HITL: {status['hitl']['pending_count']} pending")
print(f"Rollouts: {status['rollout']['active_rollouts']} active")
```

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| A/B Test Duration | <7 days | >14 days |
| HITL Response Time | <24 hours | >72 hours |
| Rollout Success Rate | >95% | <80% |
| Auto-Approval Rate | 30-50% | <10% or >80% |

---

## Troubleshooting

### "Experiment never completes"

**Cause:** Insufficient traffic to treatment group.

**Solution:** Increase `traffic_split` or lower `min_samples`.

### "HITL requests piling up"

**Cause:** Approvers not reviewing requests.

**Solution:** Enable more notification channels, set up alerts.

### "Rollout stuck in stage"

**Cause:** Not enough executions to meet thresholds.

**Solution:** Lower `min_successes` or increase project volume.

### "Too many rollbacks"

**Cause:** Proposals not properly validated before rollout.

**Solution:** Require A/B test before rollout for medium+ impact.

---

## Best Practices

### A/B Testing
- Run for minimum 7 days (weekly patterns)
- Minimum 30 samples per variant
- Check for contamination (same project in both groups)

### HITL
- Auto-approve low-risk to reduce bottleneck
- Set clear approval SLAs (e.g., 24 hours)
- Document rejection reasons for learning

### Gradual Rollout
- Start conservative (5% traffic)
- Set realistic success thresholds
- Enable auto-rollback for safety

---

## API Reference

### ABTestingEngine

| Method | Description |
|--------|-------------|
| `create_experiment()` | Create new A/B test |
| `route_execution()` | Route project to variant |
| `record_outcome()` | Record execution result |
| `analyze_results()` | Statistical analysis |
| `get_active_experiments()` | List running experiments |

### HITLWorkflow

| Method | Description |
|--------|-------------|
| `submit_for_approval()` | Submit proposal |
| `get_pending_requests()` | List pending |
| `approve()` | Approve request |
| `reject()` | Reject request |
| `is_approved()` | Check status |

### GradualRolloutManager

| Method | Description |
|--------|-------------|
| `start_rollout()` | Start rollout |
| `record_execution()` | Record outcome |
| `check_stage_progress()` | Check thresholds |
| `advance_stage()` | Move to next stage |
| `trigger_rollback()` | Rollback rollout |

---

## Related Documentation

- [Meta-Optimization V1](./META_OPTIMIZATION.md) — Base system
- [Adaptive Templates](./ADAPTIVE_TEMPLATES.md) — Template system
- [Model Routing](./MODEL_ROUTING.md) — Routing strategies

## References

- **A/B Testing:** Kohavi et al., "Trustworthy Online Controlled Experiments"
- **Gradual Rollout:** Netflix, "Canarying Deployments"
- **HITL:** Google, "Human-in-the-Loop Machine Learning"
