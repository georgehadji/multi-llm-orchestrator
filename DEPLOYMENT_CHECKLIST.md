# DEPLOYMENT CHECKLIST

**System**: AI Orchestrator v6.0  
**Version**: _______________  
**Date**: _______________  
**Deployed By**: _______________

---

## PRE-DEPLOYMENT (T-24 hours)

### Code Review
- [ ] All changes reviewed by 2+ engineers
- [ ] Security review completed
- [ ] Performance impact assessed
- [ ] API compatibility verified
- [ ] Database migrations reviewed

**Reviewer 1**: _______________ **Date**: _______________  
**Reviewer 2**: _______________ **Date**: _______________

### Testing
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Load tests completed
- [ ] Failure simulation completed
- [ ] Rollback procedure tested

**Test Report**: _______________

### Monitoring
- [ ] Metrics endpoints verified
- [ ] Alert thresholds configured
- [ ] Dashboard updated
- [ ] Log aggregation verified
- [ ] PagerDuty integration tested

**Monitoring Verified By**: _______________

### Rollback Preparation
- [ ] Previous version tagged: _______________
- [ ] Backup completed: _______________
- [ ] Rollback procedure tested
- [ ] Rollback time < 15 minutes verified

**Backup ID**: _______________

### Documentation
- [ ] Release notes written
- [ ] Runbook updated
- [ ] API docs updated
- [ ] Stakeholders notified

---

## DEPLOYMENT DAY (T-0)

### T-30 Minutes: Final Checks
- [ ] Team availability confirmed
- [ ] On-call engineer ready
- [ ] Communication channels open (Slack, etc.)
- [ ] No other deployments scheduled

**Go/No-Go Decision**: ☐ GO ☐ NO-GO  
**Decision By**: _______________

### T-10 Minutes: Pre-Deployment Health Check
| Check | Command | Expected | Actual | Status |
|-------|---------|----------|--------|--------|
| API Health | `curl http://localhost:8000/health` | `{"status": "healthy"}` | | ☐ |
| Task Count | `curl http://localhost:8000/admin/tasks/count` | `{"active": <10}` | | ☐ |
| Memory | `curl http://localhost:8000/metrics` | `< 400 MB` | | ☐ |
| Error Rate | Check dashboard | `< 1%` | | ☐ |

**Verified By**: _______________

### T-5 Minutes: Stop Traffic
```bash
curl -X POST http://localhost:8000/admin/traffic/stop
```
- [ ] Traffic stopped
- [ ] Response: `{"status": "stopped"}`

**Verified By**: _______________ **Time**: _______________

### T-0: Wait for Drain
```bash
# Check every 30 seconds
curl http://localhost:8000/admin/tasks/count
```
- [ ] Active tasks = 0
- [ ] Background tasks completed

**Drain Complete Time**: _______________

### T+5 Minutes: Create Backup
```bash
python -m orchestrator.backup create
```
- [ ] Backup created
- [ ] Backup ID: _______________
- [ ] Backup verified

**Verified By**: _______________

### T+10 Minutes: Deploy
```bash
# Kubernetes
kubectl apply -f deployment.yaml
kubectl rollout status deployment/orchestrator

# OR Docker
docker-compose up -d orchestrator

# OR Systemd
systemctl restart orchestrator
```
- [ ] Deployment command executed
- [ ] Rollout complete

**Deployed By**: _______________ **Time**: _______________

### T+15 Minutes: Post-Deployment Health Check
| Check | Command | Expected | Actual | Status |
|-------|---------|----------|--------|--------|
| API Health | `curl http://localhost:8000/health` | `{"status": "healthy"}` | | ☐ |
| Version | `curl http://localhost:8000/version` | `{"version": "___"}` | | ☐ |
| Metrics | `curl http://localhost:8000/metrics` | Prometheus format | | ☐ |
| Database | `sqlite3 ~/.orchestrator_cache/state.db "PRAGMA integrity_check;"` | `ok` | | ☐ |

**Verified By**: _______________

### T+20 Minutes: Resume Traffic
```bash
curl -X POST http://localhost:8000/admin/traffic/resume
```
- [ ] Traffic resumed
- [ ] Response: `{"status": "resumed"}`

**Verified By**: _______________ **Time**: _______________

### T+20 Minutes: Smoke Tests
- [ ] Create test project: `orchestrator run --test`
- [ ] Verify task execution
- [ ] Verify state persistence
- [ ] Verify event bus
- [ ] Verify all LLM providers

**Smoke Test Results**: _______________

---

## POST-DEPLOYMENT MONITORING

### T+30 Minutes
| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Error Rate | < 1% | | ☐ |
| p95 Latency | < 30s | | ☐ |
| Memory | < 512 MB | | ☐ |
| CPU | < 80% | | ☐ |
| Active Tasks | < 50 | | ☐ |

**Verified By**: _______________

### T+1 Hour
| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Error Rate | < 1% | | ☐ |
| p95 Latency | < 30s | | ☐ |
| Memory | < 512 MB | | ☐ |
| CPU | < 80% | | ☐ |
| Cost Rate | < $10/hour | | ☐ |

**Verified By**: _______________

### T+4 Hours
| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Error Rate | < 1% | | ☐ |
| p95 Latency | < 30s | | ☐ |
| Memory | < 512 MB | | ☐ |
| CPU | < 80% | | ☐ |
| Cost Rate | < $10/hour | | ☐ |

**Verified By**: _______________

### T+24 Hours
| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Error Rate (24h avg) | < 1% | | ☐ |
| p95 Latency (24h avg) | < 30s | | ☐ |
| Total Cost (24h) | < $240 | | ☐ |
| Incidents | 0 | | ☐ |

**Verified By**: _______________

---

## ROLLBACK DECISION POINTS

### Immediate Rollback (< 5 minutes)
Trigger if ANY of the following:
- [ ] Error rate > 3%
- [ ] Data loss detected
- [ ] Security vulnerability found
- [ ] Complete service outage

**Rollback Initiated By**: _______________ **Time**: _______________

### Urgent Rollback (< 30 minutes)
Trigger if ANY of the following persist for > 10 minutes:
- [ ] Error rate > 2%
- [ ] p95 latency > 60s
- [ ] Memory > 768 MB
- [ ] Cost rate > $20/hour

**Rollback Initiated By**: _______________ **Time**: _______________

### Scheduled Rollback (next business day)
Trigger if ANY of the following persist for > 1 hour:
- [ ] Error rate > 1.5%
- [ ] p95 latency > 45s
- [ ] User complaints increase
- [ ] Feature not working as expected

**Rollback Scheduled For**: _______________

---

## ROLLBACK PROCEDURE

### Step 1: Stop Traffic
```bash
curl -X POST http://localhost:8000/admin/traffic/stop
```
- [ ] Traffic stopped

### Step 2: Wait for Drain
```bash
curl http://localhost:8000/admin/tasks/count
```
- [ ] Active tasks = 0

### Step 3: Deploy Previous Version
```bash
# Kubernetes
kubectl rollout undo deployment/orchestrator

# OR Docker
docker-compose up -d orchestrator --force-recreate

# OR Systemd
# Restore backup files
python -m orchestrator.backup restore --backup-id=<backup_id>
systemctl restart orchestrator
```
- [ ] Previous version deployed

### Step 4: Verify Health
```bash
curl http://localhost:8000/health
```
- [ ] Health check passes

### Step 5: Resume Traffic
```bash
curl -X POST http://localhost:8000/admin/traffic/resume
```
- [ ] Traffic resumed

### Step 6: Verify Monitoring
- [ ] Error rate < 1%
- [ ] p95 latency < 30s
- [ ] Memory < 512 MB

**Rollback Complete Time**: _______________

---

## INCIDENT DOCUMENTATION

### Issues Encountered
| Time | Issue | Severity | Resolution |
|------|-------|----------|------------|
| | | | |
| | | | |

### Lessons Learned
| Category | Observation | Action Item | Owner |
|----------|-------------|-------------|-------|
| Process | | | |
| Technical | | | |
| Communication | | | |

### Follow-up Actions
- [ ] Post-mortem scheduled: _______________
- [ ] Runbook updates needed: _______________
- [ ] Monitoring improvements: _______________
- [ ] Test coverage gaps: _______________

---

## SIGN-OFF

**Deployment Successful**: ☐ YES ☐ NO (rolled back)

**Deployed By**: _______________ **Signature**: _______________

**Reviewed By**: _______________ **Signature**: _______________

**Date/Time**: _______________

---

*Keep this checklist for audit purposes. Upload completed checklist to project documentation.*
