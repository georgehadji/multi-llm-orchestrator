# Issue Tracking Integration

Deep integration with Jira and Linear for automatic ticket management, RICE scoring, and knowledge base synchronization.

## Overview

This module provides three main capabilities:

1. **Ticket Sync** - Auto-create tickets when quality gates fail
2. **RICE Scoring** - Import backlog, score with RICE, export priorities
3. **Knowledge Linking** - Convert resolved tickets to Knowledge Artifacts

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Orchestrator Core                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Quality      │  │ Product      │  │ Knowledge    │  │ Audit        │    │
│  │ Control      │  │ Management   │  │ Base         │  │ Logs         │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │                 │
          │                 │                 │                 │
┌─────────▼─────────────────▼─────────────────▼─────────────────▼────────────┐
│                    Issue Tracking Integration                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TicketSyncHooks                                                    │   │
│  │  ├── on_quality_gate_evaluated() → Create ticket on failure        │   │
│  │  └── on_run_completed() → Update linked issues                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BacklogSyncService                                                 │   │
│  │  ├── fetch_backlog_issues() → Import from tracker                  │   │
│  │  ├── calculate_rice() → Score with RICE formula                    │   │
│  │  └── update_issue_rice() → Export scores/priorities                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TicketKnowledgeSync                                                │   │
│  │  ├── sync_resolved_tickets() → Periodic sync                       │   │
│  │  ├── handle_webhook() → Real-time webhook handler                  │   │
│  │  └── ticket_to_draft() → Convert to Knowledge Artifact             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ├── IssueTrackerService (ABC)
          │     ├── JiraIssueTrackerService
          │     └── LinearIssueTrackerService
          │
          └── External APIs (Jira REST, Linear GraphQL)
```

## Configuration

### Environment Variables

```bash
# Required
export ISSUE_TRACKER_PROVIDER=jira        # or "linear"
export ISSUE_TRACKER_API_TOKEN="your-api-token"
export ISSUE_TRACKER_PROJECT_KEY="PROJ"   # Jira project key or Linear team ID

# Optional
export ISSUE_TRACKER_URL="https://your-domain.atlassian.net"  # Jira only
export ISSUE_TRACKER_REPEAT_THRESHOLD=3
export ISSUE_TRACKER_AUTO_CREATE=true
export KNOWLEDGE_AUTO_CREATE=false
export KNOWLEDGE_MIN_CONFIDENCE=0.7
export ORCHESTRATOR_HOST="dashboard.example.com"
```

### Jira Setup

1. Create API token: **Account Settings → Security → API Tokens**
2. Note your project key (e.g., `PROJ`)
3. Custom fields for RICE (optional):
   - Create custom fields: `customfield_rice_score`, etc.
   - Update field IDs in code if different

### Linear Setup

1. Create API key: **Settings → API → Personal API Keys**
2. Get Team ID from URL or GraphQL API
3. No custom fields needed (uses labels for RICE)

## Usage

### 1. Quality Gate Ticket Sync

Automatic ticket creation when quality gates fail:

```python
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketSyncHooks,
    QualityGateReport,
)

# Initialize
tracker = IssueTrackerService.from_env()
hooks = TicketSyncHooks(tracker)

# In your orchestrator quality gate handler:
async def on_quality_gate(report: QualityGateReport):
    if not report.passed:
        issue_id = await hooks.on_quality_gate_evaluated(report)
        print(f"Created ticket: {issue_id}")
```

**Ticket Format:**

```markdown
# Quality Gate Failed: api-gateway-v2

**Run ID:** `run-2024-001`
**Quality Score:** 0.42
**Test Coverage:** 65%
**Tests:** 5/42 failed

## Top Issues

1. **[SECURITY]** Hardcoded API key detected in config.py
   - Location: `src/config.py:45`
   - Rule: `bandit-B105`

2. **[TEST]** 3 tests failing in test_auth.py
   - Location: `tests/test_auth.py`
   - Test: `test_token_validation`

## Links

- [Dashboard](https://orchestrator.example.com/runs/run-2024-001)
- [Logs](https://orchestrator.example.com/runs/run-2024-001/logs)

<!-- signature:a3f7b2d9e8c5a1b0 -->
```

**Features:**
- Deduplication by issue signature (file + test + message hash)
- Repeat bug detection (threshold = 3 occurrences in 30 days)
- Links new runs to existing open tickets
- Labels: `ai-orchestrator`, `quality-gate-failure`, `<type>`, `<severity>`

### 2. RICE Scoring Sync

Import backlog, calculate RICE, update priorities:

```python
from orchestrator.issue_tracking import (
    IssueTrackerService,
    BacklogSyncService,
    RICEMappingConfig,
)

# Configure field mapping
config = RICEMappingConfig(
    reach_field="customfield_10001",
    impact_field="customfield_10002",
    confidence_field="customfield_10003",
    effort_field="customfield_10004",
)

# Initialize
tracker = IssueTrackerService.from_env()
service = BacklogSyncService(tracker, config)

# Dry run first
results = await service.sync_rice_scores(dry_run=True)
report = service.generate_report(results)
print(report)

# Apply changes
results = await service.sync_rice_scores(dry_run=False)
```

**RICE Formula:**

```
RICE Score = (Reach × Impact × Confidence) / Effort

Where:
- Reach: Users/events per period (e.g., 1000)
- Impact: 0-3 scale (0=minimal, 3=massive)
- Confidence: 0-100% (50% if using defaults)
- Effort: Person-weeks of work
```

**Priority Mapping:**

| RICE Score | Priority | Jira Priority |
|------------|----------|---------------|
| ≥ 50 | Critical | Highest |
| 20-49 | High | High |
| 5-19 | Medium | Medium |
| < 5 | Low | Low |

### 3. Knowledge Artifact Sync

Convert resolved tickets to Knowledge Base entries:

```python
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketKnowledgeSync,
)
from orchestrator import get_knowledge_base

# Initialize
tracker = IssueTrackerService.from_env()
kb = get_knowledge_base()

sync = TicketKnowledgeSync(
    tracker=tracker,
    knowledge_base=kb,
    auto_create=False,  # Set True to skip review
    min_confidence=0.7,
)

# Sync resolved tickets from last 7 days
from datetime import datetime, timedelta
since = datetime.utcnow() - timedelta(days=7)

drafts = await sync.sync_resolved_tickets(
    since=since,
    label_filter=["ai-orchestrator"],
)

for draft in drafts:
    print(f"{'✅' if draft.created else '⏳'} {draft.title}")
```

**Knowledge Artifact Format:**

```yaml
title: "[Resolved] Auth token validation fails with 500 error"
type: resolved_issue
tags: ["ai-orchestrator", "auth", "bug", "api"]
content:
  problem: |
    Auth token validation was returning 500 errors
    instead of 401 for expired tokens.
  solution: |
    Updated error handling to catch TokenExpired
    exception and return proper 401 response.
  rationale: |
    500 errors were triggering alerts and masking
    the actual client-side token refresh issue.
links:
  source_issue: https://jira.example.com/browse/PROJ-123
  orchestrator_runs:
    - https://orchestrator.example.com/runs/run-456
```

## CLI Commands

```bash
# RICE scoring (dry run)
orchestrator backlog sync-rice --dry-run

# RICE scoring (apply changes)
orchestrator backlog sync-rice --apply

# Sync knowledge from tickets (last 7 days)
orchestrator knowledge sync-from-tickets --since 7d

# Auto-create artifacts without review
orchestrator knowledge sync-from-tickets --since 7d --auto-create
```

## Webhook Integration

### FastAPI Example

```python
from fastapi import FastAPI, Request
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketSyncHooks,
    TicketKnowledgeSync,
    QualityGateReport,
)

app = FastAPI()
tracker = IssueTrackerService.from_env()

# Quality gate webhook
@app.post("/webhooks/quality-gate")
async def quality_gate_webhook(request: Request):
    data = await request.json()
    
    report = QualityGateReport(
        project_id=data["project_id"],
        run_id=data["run_id"],
        passed=data["passed"],
        quality_score=data["quality_score"],
        issues=[QualityIssue(**i) for i in data["issues"]],
        dashboard_url=data["dashboard_url"],
    )
    
    hooks = TicketSyncHooks(tracker)
    issue_id = await hooks.on_quality_gate_evaluated(report)
    
    return {"issue_id": issue_id}

# Issue tracker webhook
@app.post("/webhooks/issue-tracker")
async def issue_tracker_webhook(request: Request):
    data = await request.json()
    
    sync = TicketKnowledgeSync(
        tracker=tracker,
        knowledge_base=get_knowledge_base(),
        auto_create=os.environ.get("KNOWLEDGE_AUTO_CREATE") == "true",
    )
    
    draft = await sync.handle_webhook(
        event_type=data["event_type"],
        ticket_data=data["ticket"],
    )
    
    return {"draft_created": draft is not None}
```

### Jira Webhook Setup

1. **Jira Settings → System → WebHooks**
2. Create webhook:
   - URL: `https://your-domain.com/webhooks/issue-tracker`
   - Events: `Issue → updated`
   - JQL: `labels = "ai-orchestrator"`

### Linear Webhook Setup

1. **Linear Settings → API → Webhooks**
2. Create webhook:
   - URL: `https://your-domain.com/webhooks/issue-tracker`
   - Events: `Issue` (state changes)

## Data Models

### QualityIssue

```python
@dataclass
class QualityIssue:
    type: IssueType           # TEST, SECURITY, PERFORMANCE, STYLE, BUG
    severity: IssueSeverity   # LOW, MEDIUM, HIGH, CRITICAL
    message: str              # Human-readable description
    file_path: Optional[str]  # File location
    line_start: Optional[int]
    line_end: Optional[int]
    test_name: Optional[str]  # For test failures
    rule_id: Optional[str]    # For linting/security rules
    
    def signature(self) -> str:
        """Unique hash for deduplication"""
```

### BacklogItem

```python
@dataclass
class BacklogItem:
    id: str
    title: str
    description: str
    status: str
    
    # RICE inputs
    reach: int = 100
    impact: int = 2
    confidence: int = 80
    effort: float = 1.0
    
    # Metadata
    labels: list[str] = []
    component: Optional[str] = None
    watchers: int = 0
    story_points: Optional[int] = None
```

## Troubleshooting

### Common Issues

**Issue:** `KeyError: 'ISSUE_TRACKER_API_TOKEN'`
- **Solution:** Set the required environment variables

**Issue:** `ValueError: Jira requires ISSUE_TRACKER_URL`
- **Solution:** For Jira, set `ISSUE_TRACKER_URL` to your instance URL

**Issue:** Tickets not being created
- **Solution:** Check `ISSUE_TRACKER_AUTO_CREATE=true` and API token permissions

**Issue:** RICE fields not updating
- **Solution:** Verify custom field IDs match your Jira configuration

### Debug Logging

```python
import logging
logging.getLogger("orchestrator.issue_tracking").setLevel(logging.DEBUG)
```

## API Reference

See docstrings in `orchestrator/issue_tracking.py` for full API documentation.

Key classes:
- `IssueTrackerService` - Abstract base for tracker implementations
- `JiraIssueTrackerService` - Jira Cloud REST API implementation
- `LinearIssueTrackerService` - Linear GraphQL API implementation
- `TicketSyncHooks` - Orchestrator lifecycle hooks
- `BacklogSyncService` - RICE scoring service
- `TicketKnowledgeSync` - Knowledge artifact sync
- `IssueTrackingCLI` - CLI command handlers
