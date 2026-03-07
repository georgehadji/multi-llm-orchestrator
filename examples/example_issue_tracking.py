"""
Example: Issue Tracking Integration for Multi-LLM Orchestrator
===============================================================

This example demonstrates:
1. Auto-creating tickets on quality gate failures
2. RICE scoring sync with backlog
3. Knowledge artifact creation from resolved tickets

Setup:
1. Configure Jira or Linear credentials
2. Set environment variables (see below)
3. Run the example commands

Environment Variables:
    export ISSUE_TRACKER_PROVIDER=jira  # or "linear"
    export ISSUE_TRACKER_API_TOKEN="your-api-token"
    export ISSUE_TRACKER_PROJECT_KEY="PROJ"
    export ISSUE_TRACKER_URL="https://your-domain.atlassian.net"
    export KNOWLEDGE_AUTO_CREATE=false
"""

import os
import asyncio
from datetime import datetime, timedelta

# Import the orchestrator and issue tracking
from orchestrator import Orchestrator, Budget, ProjectState
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketSyncHooks,
    BacklogSyncService,
    RICEMappingConfig,
    TicketKnowledgeSync,
    IssueTrackingCLI,
    QualityGateReport,
    QualityIssue,
    IssueType,
    IssueSeverity,
)

# Set up environment for demo
os.environ.setdefault("ISSUE_TRACKER_PROVIDER", "jira")
os.environ.setdefault("ISSUE_TRACKER_PROJECT_KEY", "DEMO")
os.environ.setdefault("ISSUE_TRACKER_URL", "https://example.atlassian.net")


# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: Quality Gate Failure  Ticket Creation
# ═══════════════════════════════════════════════════════════════════════════════

async def example_quality_gate_ticket():
    """
    Demonstrate automatic ticket creation when quality gate fails.
    
    In production, this is called automatically by the orchestrator
    when quality_gate_passed == False.
    """
    print("=" * 60)
    print("Example 1: Quality Gate Failure → Ticket Creation")
    print("=" * 60)
    
    # Initialize tracker (requires valid credentials)
    try:
        tracker = IssueTrackerService.from_env()
        hooks = TicketSyncHooks(tracker)
    except (KeyError, ValueError) as e:
        print(f"Note: {e}")
        print("This example requires valid Jira/Linear credentials.")
        print("Skipping actual API calls.\n")
        return
    
    # Simulate a quality gate report
    report = QualityGateReport(
        project_id="api-gateway-v2",
        run_id="run-2024-001",
        passed=False,
        quality_score=0.42,
        coverage_percent=65,
        test_count=42,
        failed_test_count=5,
        issues=[
            QualityIssue(
                type=IssueType.SECURITY,
                severity=IssueSeverity.CRITICAL,
                message="Hardcoded API key detected in config.py",
                file_path="src/config.py",
                line_start=45,
                rule_id="bandit-B105",
            ),
            QualityIssue(
                type=IssueType.TEST,
                severity=IssueSeverity.HIGH,
                message="3 tests failing in test_auth.py",
                file_path="tests/test_auth.py",
                test_name="test_token_validation",
            ),
            QualityIssue(
                type=IssueType.PERFORMANCE,
                severity=IssueSeverity.MEDIUM,
                message="O(n²) loop detected in user search",
                file_path="src/users/search.py",
                line_start=88,
                line_end=95,
            ),
        ],
        dashboard_url="https://orchestrator.example.com/runs/run-2024-001",
        logs_url="https://orchestrator.example.com/runs/run-2024-001/logs",
        artifacts_url="https://orchestrator.example.com/runs/run-2024-001/artifacts",
    )
    
    print(f"Project: {report.project_id}")
    print(f"Run ID: {report.run_id}")
    print(f"Quality Score: {report.quality_score}")
    print(f"Issues Found: {len(report.issues)}")
    print()
    
    # This would create tickets in Jira/Linear
    print("Calling on_quality_gate_evaluated()...")
    # issue_id = await hooks.on_quality_gate_evaluated(report)
    # print(f"Created issue: {issue_id}")
    
    print("✅ Quality gate failure would create tickets with:")
    print("   - Title: [Orchestrator] SECURITY: Hardcoded API key...")
    print("   - Labels: ai-orchestrator, quality-gate-failure, security, critical")
    print("   - Links to dashboard, logs, artifacts")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: RICE Scoring Backlog Sync
# ═══════════════════════════════════════════════════════════════════════════════

async def example_rice_sync():
    """
    Demonstrate syncing RICE scores to backlog issues.
    
    This fetches backlog items, calculates RICE scores, and updates
    the issues with scores and priorities.
    """
    print("=" * 60)
    print("Example 2: RICE Scoring Backlog Sync")
    print("=" * 60)
    
    try:
        tracker = IssueTrackerService.from_env()
    except (KeyError, ValueError) as e:
        print(f"Note: {e}")
        print("Skipping actual API calls.\n")
        return
    
    # Configure RICE mapping
    rice_config = RICEMappingConfig(
        reach_field="customfield_10001",  # Jira custom field IDs
        impact_field="customfield_10002",
        confidence_field="customfield_10003",
        effort_field="customfield_10004",
        default_reach=500,
        default_impact=2,
        default_confidence=70,
        default_effort=2.0,
    )
    
    # Create sync service
    sync_service = BacklogSyncService(tracker, rice_config)
    
    print("RICE Configuration:")
    print(f"  Reach field: {rice_config.reach_field}")
    print(f"  Impact field: {rice_config.impact_field}")
    print(f"  Confidence field: {rice_config.confidence_field}")
    print(f"  Effort field: {rice_config.effort_field}")
    print()
    
    # Example: Dry run
    print("Running RICE sync (DRY RUN)...")
    # results = await sync_service.sync_rice_scores(
    #     status_filter=["To Do", "Backlog"],
    #     dry_run=True,
    # )
    # report = sync_service.generate_report(results)
    # print(report)
    
    print("✅ RICE sync would:")
    print("   1. Fetch backlog items from Jira/Linear")
    print("   2. Calculate RICE = (Reach × Impact × Confidence) / Effort")
    print("   3. Update each issue with rice_score custom field")
    print("   4. Update priority based on RICE score ranges")
    print()
    
    # Example RICE calculation
    from orchestrator.issue_tracking import RICECalculator
    
    calculator = RICECalculator()
    result = calculator.calculate(
        reach=1000,    # 1000 users/month
        impact=3,      # Massive impact
        confidence=90, # 90% confident
        effort=2.0,    # 2 person-weeks
    )
    
    print("Sample RICE Calculation:")
    print(f"  Reach: {result.reach} users")
    print(f"  Impact: {result.impact}/3")
    print(f"  Confidence: {result.confidence}%")
    print(f"  Effort: {result.effort} weeks")
    print(f"  Score: {result.score:.2f}")
    print(f"  Priority: {result.priority.upper()}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Knowledge Artifact Sync
# ═══════════════════════════════════════════════════════════════════════════════

async def example_knowledge_sync():
    """
    Demonstrate converting resolved tickets to Knowledge Artifacts.
    
    This scans recently resolved tickets and creates Knowledge Base
    entries for solved problems.
    """
    print("=" * 60)
    print("Example 3: Knowledge Artifact Sync")
    print("=" * 60)
    
    try:
        tracker = IssueTrackerService.from_env()
    except (KeyError, ValueError) as e:
        print(f"Note: {e}")
        print("Skipping actual API calls.\n")
        return
    
    # Mock knowledge base (replace with actual KB instance)
    class MockKnowledgeBase:
        async def add_artifact(self, **kwargs):
            return f"artifact-{datetime.now().timestamp()}"
    
    kb = MockKnowledgeBase()
    
    # Create sync service
    sync_service = TicketKnowledgeSync(
        tracker=tracker,
        knowledge_base=kb,
        auto_create=False,  # Set to True for auto-creation
        min_confidence=0.7,
    )
    
    print("Knowledge Sync Configuration:")
    print(f"  Auto-create: {sync_service.auto_create}")
    print(f"  Min confidence: {sync_service.min_confidence}")
    print()
    
    # Sync resolved tickets from last 7 days
    since = datetime.utcnow() - timedelta(days=7)
    
    print("Syncing resolved tickets (last 7 days)...")
    # drafts = await sync_service.sync_resolved_tickets(
    #     since=since,
    #     label_filter=["ai-orchestrator"],
    # )
    
    print("✅ Knowledge sync would:")
    print("   1. Scan for resolved tickets with 'ai-orchestrator' label")
    print("   2. Extract problem from title/description")
    print("   3. Extract solution from resolution comments")
    print("   4. Create Knowledge Artifact with:")
    print("      - Title: [Resolved] <original title>")
    print("      - Tags: from labels + 'ai-orchestrator'")
    print("      - Links: to source issue and orchestrator runs")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Example 4: CLI Usage
# ═══════════════════════════════════════════════════════════════════════════════

async def example_cli():
    """
    Demonstrate CLI commands for issue tracking.
    """
    print("=" * 60)
    print("Example 4: CLI Commands")
    print("=" * 60)
    
    cli = IssueTrackingCLI()
    
    print("Available commands:")
    print()
    print("1. Sync RICE scores (dry run):")
    print("   $ orchestrator backlog sync-rice --dry-run")
    print("   # or via API: POST /cli/backlog/sync-rice")
    print()
    
    print("2. Sync RICE scores (apply):")
    print("   $ orchestrator backlog sync-rice --apply")
    print("   # Updates all backlog issues with RICE scores")
    print()
    
    print("3. Sync knowledge from tickets:")
    print("   $ orchestrator knowledge sync-from-tickets --since 7d")
    print("   # Converts resolved tickets to Knowledge Artifacts")
    print()
    
    print("4. Sync with auto-create:")
    print("   $ orchestrator knowledge sync-from-tickets --since 7d --auto-create")
    print("   # Auto-creates artifacts without review")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Example 5: Webhook Handlers (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════════

async def example_webhooks():
    """
    Demonstrate webhook handlers for FastAPI integration.
    """
    print("=" * 60)
    print("Example 5: FastAPI Webhook Handlers")
    print("=" * 60)
    
    print("""
# main.py
from fastapi import FastAPI, Request
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketSyncHooks,
    TicketKnowledgeSync,
    QualityGateReport,
)

app = FastAPI()
tracker = IssueTrackerService.from_env()

# Quality gate webhook from orchestrator
@app.post("/webhooks/quality-gate")
async def quality_gate_webhook(request: Request):
    data = await request.json()
    
    report = QualityGateReport(
        project_id=data["project_id"],
        run_id=data["run_id"],
        passed=data["passed"],
        quality_score=data["quality_score"],
        issues=data["issues"],
        dashboard_url=data["dashboard_url"],
    )
    
    hooks = TicketSyncHooks(tracker)
    issue_id = await hooks.on_quality_gate_evaluated(report)
    
    return {"issue_id": issue_id, "created": issue_id is not None}

# Issue tracker webhook (Jira/Linear → Knowledge Base)
@app.post("/webhooks/issue-tracker")
async def issue_tracker_webhook(request: Request):
    data = await request.json()
    
    sync = TicketKnowledgeSync(
        tracker=tracker,
        knowledge_base=get_knowledge_base(),
        auto_create=os.environ.get("KNOWLEDGE_AUTO_CREATE") == "true",
    )
    
    draft = await sync.handle_webhook(
        event_type=data["event_type"],  # "issue.resolved"
        ticket_data=data["ticket"],
    )
    
    return {
        "draft_created": draft is not None,
        "confidence": draft.confidence if draft else None,
    }

# CLI endpoint for RICE sync
@app.post("/cli/backlog/sync-rice")
async def sync_rice(dry_run: bool = True):
    from orchestrator.issue_tracking import IssueTrackingCLI
    
    cli = IssueTrackingCLI()
    report = await cli.sync_rice(dry_run=dry_run)
    
    return {"report": report}
""")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Issue Tracking Integration Examples")
    print("=" * 60 + "\n")
    
    await example_quality_gate_ticket()
    await example_rice_sync()
    await example_knowledge_sync()
    await example_cli()
    await example_webhooks()
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("""
Next steps:
1. Set up Jira/Linear credentials:
   export ISSUE_TRACKER_API_TOKEN="your-token"
   export ISSUE_TRACKER_PROJECT_KEY="YOURPROJECT"

2. Run the orchestrator with hooks:
   tracker = IssueTrackerService.from_env()
   hooks = TicketSyncHooks(tracker)
   orchestrator.register_hook("quality_gate", hooks.on_quality_gate_evaluated)

3. Try CLI commands:
   python -m orchestrator backlog sync-rice --dry-run
""")


if __name__ == "__main__":
    asyncio.run(main())
