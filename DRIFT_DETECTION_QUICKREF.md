# DRIFT DETECTION - QUICK REFERENCE

**System**: AI Orchestrator v6.0  
**Date**: 2026-03-07

---

## DRIFT SIGNALS AT A GLANCE

### Performance Drift

| Signal | Interval | WARNING | CRITICAL | Cooldown |
|--------|----------|---------|----------|----------|
| Latency (p95) | 1 hour | magnitude 0.5 | magnitude 0.7 | 4h / 1h |
| Throughput | 1 hour | magnitude 0.5 | magnitude 0.7 | 4h / 1h |
| Memory | 4 hours | slope > 10 MB/h | slope > 50 MB/h | 4h / 1h |
| CPU | 4 hours | slope > 5%/h | slope > 20%/h | 4h / 1h |

### Error Pattern Drift

| Signal | Interval | WARNING | CRITICAL | Cooldown |
|--------|----------|---------|----------|----------|
| Error Rate | 1 hour | > 1.5% | > 3% | 4h / 1h |
| Error Type Dist. | 4 hours | χ² p < 0.01 | New critical errors | 4h / 1h |
| New Errors | 1 hour | Any new | Security/OOM | 4h / 1h |

### Data Distribution Drift

| Signal | Interval | WARNING | CRITICAL | Cooldown |
|--------|----------|---------|----------|----------|
| Input Length | 1 hour | PSI > 0.2 | PSI > 0.25 | 4h / 1h |
| Task Type Dist. | 4 hours | χ² p < 0.01 | Major shift | 4h / 1h |
| Model Selection | 4 hours | χ² p < 0.01 | Provider change | 4h / 1h |
| Cache Hit Rate | 1 hour | > 10% change | > 25% change | 4h / 1h |

---

## ALERT SEVERITY TIERS

| Tier | Magnitude | Response Time | Notification |
|------|-----------|---------------|--------------|
| **T0 INFO** | 0.3 - 0.5 | 24 hours | Slack #monitoring |
| **T1 WARNING** | 0.5 - 0.7 | 4 hours | Slack #alerts + Email |
| **T2 CRITICAL** | 0.7 - 0.85 | 30 minutes | + PagerDuty |
| **T3 EMERGENCY** | > 0.85 | Immediate | + Phone call |

---

## COOLDOWN PERIODS

| Tier | Duration | Behavior |
|------|----------|----------|
| INFO | 1 hour | Same alert suppressed |
| WARNING | 4 hours | Investigation time |
| CRITICAL | 1 hour | Verify resolution |
| EMERGENCY | 30 minutes | Rapid iteration |

---

## MANUAL OVERRIDE CONDITIONS

| Override | Who | Max Duration |
|----------|-----|--------------|
| Suppress Alert | Team lead | 24 hours |
| Extend Cooldown | On-call engineer | 4 hours |
| Disable Detection | Engineering manager | 7 days |
| Adjust Thresholds | 2+ engineers | Permanent (reviewed) |
| Emergency Override | VP Engineering | Until resolved |

**Request Format**:
```yaml
override_request:
  signal_name: "latency_drift"
  override_type: "suppress_alert"
  reason: "Known issue - maintenance"
  requested_by: "engineer"
  approved_by: "team_lead"
  end_time: "2026-03-07T22:00:00Z"
  ticket: "INC-1234"
```

---

## DRIFT MAGNITUDE INTERPRETATION

```
0.0 ───── 0.3 ───── 0.5 ───── 0.7 ───── 0.85 ───── 1.0
│          │         │         │          │
│       Negligible  Minor    Moderate   Severe
│                     │         │          │
│                   WARNING  CRITICAL  EMERGENCY
│
└────────────────── INFO (log only)
```

---

## RESPONSE PLAYBOOK

### T0 INFO
- Log drift detection
- Add to daily report
- No immediate action

### T1 WARNING
- Notify on-call engineer
- Investigate within 4 hours
- Document in incident log

### T2 CRITICAL
- Page on-call (30 min response)
- Freeze deployments
- Root cause within 2 hours
- Mitigate or rollback

### T3 EMERGENCY
- Page entire team
- Emergency war room
- Consider immediate rollback
- Executive notification

---

## STATISTICAL THRESHOLDS

| Test | WARNING | CRITICAL |
|------|---------|----------|
| **KS Test (p-value)** | < 0.05 | < 0.01 |
| **Chi-square (p-value)** | < 0.05 | < 0.01 |
| **Z-score** | > 2.0 | > 3.0 |
| **PSI** | > 0.1 | > 0.25 |
| **Cohen's d** | > 0.5 | > 0.8 |

---

## BASELINE MANAGEMENT

| Strategy | Update Trigger | Use Case |
|----------|----------------|----------|
| Rolling | Continuous | Stable systems |
| Fixed | Manual only | Regulated |
| Adaptive | Legitimate drift | Evolving systems |
| Seasonal | Time-based | Time-dependent |

**Reset Baseline When**:
- [ ] Major version deployment
- [ ] Infrastructure migration
- [ ] Model change
- [ ] Confirmed false positive
- [ ] Business requirement change

---

## IMPLEMENTATION EXAMPLE

```python
from orchestrator.drift_detection import DriftDetector, DriftConfig

# Initialize
detector = DriftDetector(DriftConfig(
    baseline_window_days=7,
    detection_window_hours=1,
))

# Record metrics
detector.record_metric('task_latency', 15.2)
detector.record_metric('error_rate', 0.01)

# Analyze for drift
alert = await detector.analyze('task_latency')

if alert:
    print(f"Drift detected: {alert.tier} (magnitude: {alert.magnitude})")
    
    # Add override if needed
    if alert.signal_name == "known_issue":
        detector.add_override(
            signal_name="known_issue",
            duration_hours=4,
            reason="Investigation in progress"
        )
```

---

## DAILY DRIFT REPORT TEMPLATE

```yaml
date: "2026-03-07"
summary:
  total_alerts: 5
  by_tier: { info: 3, warning: 2, critical: 0, emergency: 0 }

signals:
  - name: "task_latency"
    status: "stable"
    current: 15.2s
    baseline: 14.8s
    magnitude: 0.15
    
  - name: "error_rate"
    status: "drifting"
    current: 1.2%
    baseline: 0.8%
    magnitude: 0.55
    tier: "warning"

actions:
  - "Investigated error_rate drift"
  - "Updated baseline for task_type_distribution"
```

---

*For full details, see `DRIFT_DETECTION_STRATEGY.md`*
