# DRIFT DETECTION STRATEGY

**System**: AI Orchestrator v6.0  
**Date**: 2026-03-07  
**Classification**: PRODUCTION-READY

---

## 1. OVERVIEW

Drift detection identifies gradual changes in system behavior that may indicate:
- Model degradation
- Changing input patterns
- Infrastructure issues
- Emerging failure modes

Unlike threshold-based alerts, drift detection uses **statistical comparison** against baselines to identify **significant deviations**.

---

## 2. DRIFT SIGNALS

### 2.1 Performance Drift

**Definition**: Gradual degradation in latency, throughput, or resource efficiency.

| Signal | Metric | Baseline Window | Detection Window | Statistical Test |
|--------|--------|-----------------|------------------|------------------|
| **Latency Drift** | p50, p95, p99 | 7 days (hourly buckets) | 1 hour | Kolmogorov-Smirnov |
| **Throughput Drift** | Tasks/minute | 7 days (hourly buckets) | 1 hour | CUSUM |
| **Memory Drift** | RSS MB (hourly avg) | 7 days | 4 hours | Linear regression slope |
| **CPU Drift** | CPU % (hourly avg) | 7 days | 4 hours | Linear regression slope |
| **Token Efficiency Drift** | Tokens/task | 7 days | 1 hour | Z-score |
| **Cost Efficiency Drift** | Cost/task | 7 days | 1 hour | Z-score |

**Detection Algorithm**:
```python
def detect_latency_drift(current_window, baseline_window):
    """
    Detect latency distribution drift using KS test.
    
    Returns:
        drift_detected: bool
        drift_magnitude: float (0-1, effect size)
        p_value: float
    """
    from scipy import stats
    
    # Kolmogorov-Smirnov test for distribution difference
    ks_statistic, p_value = stats.ks_2samp(baseline_window, current_window)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline_window)**2 + np.std(current_window)**2) / 2)
    cohens_d = abs(np.mean(current_window) - np.mean(baseline_window)) / pooled_std
    
    drift_detected = (p_value < 0.01) and (cohens_d > 0.5)
    drift_magnitude = min(cohens_d / 2.0, 1.0)  # Normalize to 0-1
    
    return drift_detected, drift_magnitude, p_value
```

**Drift Magnitude Interpretation**:

| Magnitude | Interpretation | Action |
|-----------|----------------|--------|
| 0.0 - 0.3 | Negligible | Log only |
| 0.3 - 0.5 | Minor | INFO alert |
| 0.5 - 0.7 | Moderate | WARNING alert |
| 0.7 - 1.0 | Severe | CRITICAL alert |

---

### 2.2 Error Pattern Drift

**Definition**: Changes in error types, frequencies, or distributions.

| Signal | Metric | Baseline Window | Detection Window | Statistical Test |
|--------|--------|-----------------|------------------|------------------|
| **Error Rate Drift** | Errors/1000 tasks | 7 days (hourly) | 1 hour | Proportion test |
| **Error Type Distribution** | Error type histogram | 7 days | 4 hours | Chi-square test |
| **New Error Emergence** | Unique error signatures | 30 days | 1 hour | Anomaly detection |
| **Provider Error Drift** | Per-provider error rate | 7 days | 1 hour | Z-score |
| **Validation Error Drift** | Validation vs processing errors | 7 days | 1 hour | Proportion test |
| **Timeout Pattern Drift** | Timeout rate by operation | 7 days | 1 hour | CUSUM |

**Detection Algorithm**:
```python
def detect_error_pattern_drift(current_errors, baseline_errors):
    """
    Detect error pattern drift using chi-square test.
    
    Returns:
        drift_detected: bool
        new_error_types: List[str]
        drift_magnitude: float
    """
    from scipy import stats
    from collections import Counter
    
    # Count error types
    baseline_counts = Counter(baseline_errors)
    current_counts = Counter(current_errors)
    
    # Get all error types
    all_types = set(baseline_counts.keys()) | set(current_counts.keys())
    
    # Find new error types
    new_error_types = list(set(current_counts.keys()) - set(baseline_counts.keys()))
    
    # Chi-square test for distribution change
    baseline_vector = [baseline_counts.get(t, 0) for t in all_types]
    current_vector = [current_counts.get(t, 0) for t in all_types]
    
    # Avoid division by zero
    if sum(baseline_vector) == 0 or sum(current_vector) == 0:
        return False, new_error_types, 0.0
    
    chi2, p_value = stats.chisquare(current_vector, baseline_vector)
    
    # Cramer's V for effect size
    n = sum(current_vector)
    k = len(all_types)
    cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else 0
    
    drift_detected = (p_value < 0.01) or (len(new_error_types) > 0)
    drift_magnitude = min(cramers_v, 1.0)
    
    return drift_detected, new_error_types, drift_magnitude
```

**New Error Classification**:

| Category | Examples | Severity |
|----------|----------|----------|
| **Expected New Errors** | New task types, new providers | INFO |
| **Concerning New Errors** | Timeout patterns, rate limits | WARNING |
| **Critical New Errors** | Security, data corruption, OOM | CRITICAL |

---

### 2.3 Data Distribution Drift

**Definition**: Changes in input characteristics, output distributions, or model behavior.

| Signal | Metric | Baseline Window | Detection Window | Statistical Test |
|--------|--------|-----------------|------------------|------------------|
| **Input Length Drift** | Prompt token count | 7 days | 1 hour | KS test |
| **Input Complexity Drift** | Code lines, nested depth | 7 days | 1 hour | KS test |
| **Task Type Distribution** | Task type histogram | 7 days | 4 hours | Chi-square |
| **Output Length Drift** | Response token count | 7 days | 1 hour | KS test |
| **Model Selection Drift** | Model usage distribution | 7 days | 4 hours | Chi-square |
| **Score Distribution Drift** | Quality score histogram | 7 days | 4 hours | KS test |
| **Revision Count Drift** | Iterations per task | 7 days | 1 hour | Z-score |
| **Cache Hit Rate Drift** | Hit/miss ratio | 7 days | 1 hour | Proportion test |

**Detection Algorithm**:
```python
def detect_data_distribution_drift(current_features, baseline_features):
    """
    Detect data distribution drift using population stability index (PSI).
    
    Returns:
        drift_detected: bool
        psi_score: float
        drifted_features: List[str]
    """
    def calculate_psi(expected, actual, buckets=10):
        """Calculate Population Stability Index."""
        from scipy.stats import percentileofscore
        
        psi = 0.0
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        
        for i in range(len(breakpoints) - 1):
            expected_pct = np.sum((expected >= breakpoints[i]) & 
                                  (expected < breakpoints[i+1])) / len(expected)
            actual_pct = np.sum((actual >= breakpoints[i]) & 
                                (actual < breakpoints[i+1])) / len(actual)
            
            # Avoid log(0)
            expected_pct = max(expected_pct, 0.0001)
            actual_pct = max(actual_pct, 0.0001)
            
            psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        
        return psi
    
    drifted_features = []
    max_psi = 0.0
    
    for feature_name in current_features.keys():
        if feature_name not in baseline_features:
            continue
        
        psi = calculate_psi(
            np.array(baseline_features[feature_name]),
            np.array(current_features[feature_name])
        )
        
        max_psi = max(max_psi, psi)
        
        if psi > 0.1:  # PSI threshold
            drifted_features.append({
                'feature': feature_name,
                'psi': psi,
                'severity': 'HIGH' if psi > 0.25 else 'MEDIUM'
            })
    
    drift_detected = max_psi > 0.1 or len(drifted_features) > 0
    
    return drift_detected, max_psi, drifted_features
```

**PSI Interpretation**:

| PSI Score | Drift Level | Action |
|-----------|-------------|--------|
| < 0.1 | Negligible | No action |
| 0.1 - 0.2 | Minor | Monitor |
| 0.2 - 0.25 | Moderate | WARNING |
| > 0.25 | Significant | CRITICAL |

---

## 3. MONITORING CONFIGURATION

### 3.1 Monitoring Intervals

| Signal Type | Collection Interval | Analysis Interval | Baseline Update |
|-------------|--------------------|-------------------|-----------------|
| **Latency Drift** | 1 minute | 1 hour | 7 days (rolling) |
| **Throughput Drift** | 1 minute | 1 hour | 7 days (rolling) |
| **Memory/CPU Drift** | 5 minutes | 4 hours | 7 days (rolling) |
| **Error Rate Drift** | 1 minute | 1 hour | 7 days (rolling) |
| **Error Pattern Drift** | 1 minute | 4 hours | 7 days (rolling) |
| **New Error Detection** | Real-time | 1 hour | 30 days (rolling) |
| **Input Distribution** | Per-task | 1 hour | 7 days (rolling) |
| **Output Distribution** | Per-task | 1 hour | 7 days (rolling) |
| **Model Selection** | Per-task | 4 hours | 7 days (rolling) |

### 3.2 Alert Severity Tiers

| Tier | Name | Drift Magnitude | Response Time | Notification |
|------|------|-----------------|---------------|--------------|
| **T0** | INFO | 0.3 - 0.5 | 24 hours | Slack #monitoring |
| **T1** | WARNING | 0.5 - 0.7 | 4 hours | Slack #alerts, Email |
| **T2** | CRITICAL | 0.7 - 0.85 | 30 minutes | Slack, Email, PagerDuty |
| **T3** | EMERGENCY | > 0.85 | Immediate | All channels + Phone |

**Severity Escalation Matrix**:

| Condition | Initial Tier | After 1 hour | After 4 hours |
|-----------|--------------|--------------|---------------|
| Drift magnitude increases | +0 tiers | +1 tier | +2 tiers |
| Multiple signals drift | +1 tier | +2 tiers | +3 tiers |
| Business hours (9-5) | Normal | Normal | Normal |
| Off-hours (5-9, weekends) | +1 tier | +2 tiers | +3 tiers |

### 3.3 Cooldown Periods

| Alert Tier | Cooldown Duration | Purpose |
|------------|-------------------|---------|
| **T0 (INFO)** | 1 hour | Prevent notification spam |
| **T1 (WARNING)** | 4 hours | Allow time for investigation |
| **T2 (CRITICAL)** | 1 hour | Ensure issue is resolved |
| **T3 (EMERGENCY)** | 30 minutes | Rapid iteration on critical issues |

**Cooldown Behavior**:
```
Alert triggered → Cooldown starts → Same alert suppressed
If alert persists after cooldown → Re-alert with escalation
If alert clears → Cooldown resets
```

### 3.4 Manual Override Conditions

| Override Type | Conditions | Approval Required | Duration |
|---------------|------------|-------------------|----------|
| **Suppress Alert** | Known issue, planned maintenance | Team lead | Max 24 hours |
| **Extend Cooldown** | Investigation in progress | On-call engineer | Max 4 hours |
| **Disable Detection** | False positive confirmed | Engineering manager | Max 7 days |
| **Adjust Thresholds** | Baseline shift confirmed | 2+ engineers | Permanent (with review) |
| **Emergency Override** | System-wide incident | VP Engineering | Until resolved |

**Override Request Template**:
```yaml
override_request:
  signal_name: "latency_drift"
  override_type: "suppress_alert"
  reason: "Known issue - database migration in progress"
  requested_by: "engineer_name"
  approved_by: "team_lead_name"
  start_time: "2026-03-07T10:00:00Z"
  end_time: "2026-03-07T22:00:00Z"
  ticket_reference: "INC-1234"
```

---

## 4. DRIFT RESPONSE PLAYBOOK

### 4.1 Performance Drift Response

**T0 (INFO)**:
```
1. Log drift detection
2. Add to daily monitoring report
3. No immediate action required
```

**T1 (WARNING)**:
```
1. Notify on-call engineer
2. Compare with recent deployments
3. Check resource utilization trends
4. Document in incident log
5. Schedule investigation within 4 hours
```

**T2 (CRITICAL)**:
```
1. Page on-call engineer (30 min response)
2. Freeze deployments
3. Run diagnostic scripts:
   - python -m diagnostics.performance_profile
   - python -m diagnostics.resource_analysis
4. Compare with baseline distributions
5. Identify root cause within 2 hours
6. Implement mitigation or rollback
```

**T3 (EMERGENCY)**:
```
1. Immediate page to entire team
2. Emergency war room
3. Consider immediate rollback
4. Executive notification
5. Customer communication if needed
```

### 4.2 Error Pattern Drift Response

**T0 (INFO)**:
```
1. Log new error patterns
2. Categorize error types
3. Review in next team meeting
```

**T1 (WARNING)**:
```
1. Notify on-call engineer
2. Analyze error stack traces
3. Check for common patterns
4. Search error database for similar issues
5. Create investigation ticket
```

**T2 (CRITICAL)**:
```
1. Page on-call engineer
2. Immediate error analysis:
   - Group by error type
   - Identify affected components
   - Check for security implications
3. Run error simulation tests
4. Implement fix or rollback within 2 hours
```

**T3 (EMERGENCY)**:
```
1. Immediate page to entire team
2. Security team notification (if applicable)
3. Emergency fix deployment
4. Post-incident review required
```

### 4.3 Data Distribution Drift Response

**T0 (INFO)**:
```
1. Log distribution changes
2. Update data documentation
3. No immediate action
```

**T1 (WARNING)**:
```
1. Notify ML/data team
2. Analyze input characteristic changes
3. Check for data pipeline issues
4. Validate model performance on new distribution
```

**T2 (CRITICAL)**:
```
1. Page ML/data team
2. Immediate analysis:
   - Compare input distributions
   - Check model performance metrics
   - Validate output quality
3. Consider model retraining
4. Update baselines if drift is legitimate
```

**T3 (EMERGENCY)**:
```
1. Immediate page to ML/data team
2. Emergency model evaluation
3. Consider fallback to previous model
4. Customer impact assessment
```

---

## 5. IMPLEMENTATION

### 5.1 Drift Detection Service

```python
# In orchestrator/drift_detection.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
from collections import deque
import asyncio

class AlertTier(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DriftAlert:
    """Represents a drift detection alert."""
    signal_name: str
    drift_type: str  # performance, error_pattern, data_distribution
    magnitude: float
    p_value: float
    tier: AlertTier
    detected_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    baseline_window_days: int = 7
    detection_window_hours: int = 1
    analysis_interval_minutes: int = 60
    cooldown_hours: Dict[AlertTier, int] = field(default_factory=lambda: {
        AlertTier.INFO: 1,
        AlertTier.WARNING: 4,
        AlertTier.CRITICAL: 1,
        AlertTier.EMERGENCY: 0.5,
    })
    magnitude_thresholds: Dict[AlertTier, float] = field(default_factory=lambda: {
        AlertTier.INFO: 0.3,
        AlertTier.WARNING: 0.5,
        AlertTier.CRITICAL: 0.7,
        AlertTier.EMERGENCY: 0.85,
    })


class DriftDetector:
    """
    Main drift detection service.
    
    Monitors multiple signals and generates alerts when drift is detected.
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        
        # Baseline data storage (rolling windows)
        self.baselines: Dict[str, deque] = {}
        
        # Current window data
        self.current_window: Dict[str, deque] = {}
        
        # Alert cooldown tracking
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Manual overrides
        self.overrides: Dict[str, datetime] = {}
        
        # Alert history
        self.alert_history: List[DriftAlert] = []
    
    def record_metric(self, signal_name: str, value: float, timestamp: datetime = None):
        """Record a metric value for drift analysis."""
        timestamp = timestamp or datetime.utcnow()
        
        if signal_name not in self.current_window:
            self.current_window[signal_name] = deque(maxlen=10000)
        
        self.current_window[signal_name].append({
            'value': value,
            'timestamp': timestamp,
        })
    
    async def analyze(self, signal_name: str) -> Optional[DriftAlert]:
        """
        Analyze a signal for drift.
        
        Returns DriftAlert if drift detected, None otherwise.
        """
        # Check for manual override
        if signal_name in self.overrides:
            if datetime.utcnow() < self.overrides[signal_name]:
                return None  # Override active
            else:
                del self.overrides[signal_name]  # Override expired
        
        # Check cooldown
        if signal_name in self.last_alert_time:
            cooldown = self.config.cooldown_hours[AlertTier.WARNING]  # Conservative
            if datetime.utcnow() - self.last_alert_time[signal_name] < timedelta(hours=cooldown):
                return None  # Still in cooldown
        
        # Get baseline and current data
        baseline_data = self._get_baseline_data(signal_name)
        current_data = self._get_current_data(signal_name)
        
        if len(baseline_data) < 100 or len(current_data) < 10:
            return None  # Insufficient data
        
        # Detect drift
        drift_detected, magnitude, p_value = self._detect_drift(
            baseline_data, current_data
        )
        
        if not drift_detected:
            return None
        
        # Determine tier
        tier = self._get_tier_for_magnitude(magnitude)
        
        # Create alert
        alert = DriftAlert(
            signal_name=signal_name,
            drift_type=self._classify_drift_type(signal_name),
            magnitude=magnitude,
            p_value=p_value,
            tier=tier,
            detected_at=datetime.utcnow(),
            details={
                'baseline_mean': np.mean(baseline_data),
                'current_mean': np.mean(current_data),
                'baseline_std': np.std(baseline_data),
                'current_std': np.std(current_data),
            }
        )
        
        # Record alert
        self.alert_history.append(alert)
        self.last_alert_time[signal_name] = datetime.utcnow()
        
        return alert
    
    def _get_baseline_data(self, signal_name: str) -> np.ndarray:
        """Get baseline data for signal."""
        if signal_name not in self.baselines:
            return np.array([])
        return np.array([x['value'] for x in self.baselines[signal_name]])
    
    def _get_current_data(self, signal_name: str) -> np.ndarray:
        """Get current window data for signal."""
        if signal_name not in self.current_window:
            return np.array([])
        
        cutoff = datetime.utcnow() - timedelta(hours=self.config.detection_window_hours)
        return np.array([
            x['value'] for x in self.current_window[signal_name]
            if x['timestamp'] > cutoff
        ])
    
    def _detect_drift(self, baseline: np.ndarray, current: np.ndarray) -> tuple:
        """Detect drift using statistical tests."""
        from scipy import stats
        
        # KS test for distribution difference
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline)**2 + np.std(current)**2) / 2)
        if pooled_std == 0:
            return False, 0.0, p_value
        
        cohens_d = abs(np.mean(current) - np.mean(baseline)) / pooled_std
        
        # Drift detected if significant and meaningful effect
        drift_detected = (p_value < 0.01) and (cohens_d > 0.3)
        magnitude = min(cohens_d / 2.0, 1.0)
        
        return drift_detected, magnitude, p_value
    
    def _get_tier_for_magnitude(self, magnitude: float) -> AlertTier:
        """Determine alert tier based on drift magnitude."""
        if magnitude >= self.config.magnitude_thresholds[AlertTier.EMERGENCY]:
            return AlertTier.EMERGENCY
        elif magnitude >= self.config.magnitude_thresholds[AlertTier.CRITICAL]:
            return AlertTier.CRITICAL
        elif magnitude >= self.config.magnitude_thresholds[AlertTier.WARNING]:
            return AlertTier.WARNING
        else:
            return AlertTier.INFO
    
    def _classify_drift_type(self, signal_name: str) -> str:
        """Classify drift type based on signal name."""
        if 'latency' in signal_name or 'throughput' in signal_name:
            return 'performance'
        elif 'error' in signal_name:
            return 'error_pattern'
        else:
            return 'data_distribution'
    
    def add_override(self, signal_name: str, duration_hours: int, reason: str):
        """Add manual override for a signal."""
        self.overrides[signal_name] = datetime.utcnow() + timedelta(hours=duration_hours)
        logger.info(f"Added override for {signal_name}: {reason} (expires in {duration_hours}h)")
    
    def update_baseline(self, signal_name: str):
        """Update baseline with current window data."""
        if signal_name not in self.current_window:
            return
        
        # Move current window to baseline
        if signal_name not in self.baselines:
            self.baselines[signal_name] = deque(maxlen=100000)
        
        for item in self.current_window[signal_name]:
            self.baselines[signal_name].append(item)
        
        # Clear current window
        self.current_window[signal_name].clear()
```

### 5.2 Integration with Orchestrator

```python
# In orchestrator/engine.py - Add drift tracking

class Orchestrator:
    def __init__(self, ...):
        # ... existing init ...
        
        # Drift detection
        from .drift_detection import DriftDetector, DriftConfig
        self.drift_detector = DriftDetector(DriftConfig(
            baseline_window_days=7,
            detection_window_hours=1,
        ))
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute task with drift tracking."""
        start_time = time.perf_counter()
        
        try:
            result = await self._execute_task(task)
            
            # Record metrics for drift detection
            latency = time.perf_counter() - start_time
            self.drift_detector.record_metric('task_latency', latency)
            self.drift_detector.record_metric('task_output_length', len(result.output))
            self.drift_detector.record_metric('task_score', result.score)
            
            return result
            
        except Exception as e:
            # Record error for drift detection
            self.drift_detector.record_metric('error_rate', 1)
            self.drift_detector.record_metric('error_type', type(e).__name__)
            raise
```

### 5.3 Alert Routing

```python
# In orchestrator/alerting.py

async def route_drift_alert(alert: DriftAlert):
    """Route drift alert to appropriate channels."""
    
    if alert.tier == AlertTier.INFO:
        await send_slack_message(
            channel="#monitoring",
            message=f"📊 Drift detected: {alert.signal_name} (magnitude: {alert.magnitude:.2f})"
        )
    
    elif alert.tier == AlertTier.WARNING:
        await send_slack_message(
            channel="#alerts",
            message=f"⚠️ WARNING: Drift detected in {alert.signal_name}"
        )
        await send_email(
            to=["oncall@company.com"],
            subject=f"Drift Alert: {alert.signal_name}",
            body=f"Drift magnitude: {alert.magnitude:.2f}\nDetails: {alert.details}"
        )
    
    elif alert.tier == AlertTier.CRITICAL:
        await send_slack_message(
            channel="#alerts",
            message=f"🚨 CRITICAL: Significant drift in {alert.signal_name}"
        )
        await send_pagerduty_incident(
            service="orchestrator",
            severity="critical",
            description=f"Drift magnitude: {alert.magnitude:.2f}"
        )
    
    elif alert.tier == AlertTier.EMERGENCY:
        await send_slack_message(
            channel="#emergency",
            message=f"🆘 EMERGENCY: Severe drift in {alert.signal_name}"
        )
        await send_pagerduty_incident(
            service="orchestrator",
            severity="emergency",
            description=f"Drift magnitude: {alert.magnitude:.2f}"
        )
        await phone_call_oncall()
```

---

## 6. BASELINE MANAGEMENT

### 6.1 Baseline Update Strategy

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Rolling Window** | Continuously update with recent data | Stable systems |
| **Fixed Baseline** | Manual baseline updates only | Regulated environments |
| **Adaptive Baseline** | Auto-update when drift is legitimate | Evolving systems |
| **Seasonal Baseline** | Separate baselines for different times | Time-dependent patterns |

### 6.2 Baseline Reset Conditions

Reset baseline when:
- [ ] Major version deployment
- [ ] Infrastructure migration
- [ ] Model change
- [ ] Confirmed false positive drift
- [ ] Business requirement change

---

## 7. REPORTING

### 7.1 Daily Drift Report

```yaml
drift_report:
  date: "2026-03-07"
  summary:
    total_alerts: 5
    by_tier:
      info: 3
      warning: 2
      critical: 0
      emergency: 0
  signals:
    - name: "task_latency"
      status: "stable"
      current_mean: 15.2s
      baseline_mean: 14.8s
      drift_magnitude: 0.15
    - name: "error_rate"
      status: "drifting"
      current_mean: 1.2%
      baseline_mean: 0.8%
      drift_magnitude: 0.55
      alert_tier: "warning"
  actions_taken:
    - "Investigated error_rate drift - new task type identified"
    - "Updated baseline for task_type_distribution"
```

### 7.2 Weekly Drift Summary

| Signal | Week 1 | Week 2 | Week 3 | Week 4 | Trend |
|--------|--------|--------|--------|--------|-------|
| Latency | Stable | Stable | Minor drift | Stable | ↗️ |
| Error Rate | Stable | Warning | Resolved | Stable | → |
| Memory | Stable | Stable | Stable | Stable | → |

---

## 8. APPENDIX: STATISTICAL TEST REFERENCE

| Test | Use Case | Threshold |
|------|----------|-----------|
| **Kolmogorov-Smirnov** | Distribution comparison | p < 0.01 |
| **Chi-square** | Categorical distribution | p < 0.01 |
| **CUSUM** | Sequential mean shift | k = 0.5, h = 4 |
| **Z-score** | Point anomaly | |z| > 3 |
| **PSI** | Population stability | > 0.1 |
| **Linear Regression** | Trend detection | slope p < 0.01 |

---

*This drift detection strategy should be reviewed quarterly and after any significant incident.*
