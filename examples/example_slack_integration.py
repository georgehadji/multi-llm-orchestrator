"""
Example: Slack Integration for Multi-LLM Orchestrator
======================================================

This example shows how to integrate the orchestrator with Slack for:
1. Alerting (budget, quality gate, circuit breaker)
2. Run summaries
3. Slash commands

Setup:
1. Create a Slack App at https://api.slack.com/apps
2. Enable Incoming Webhooks and note the URL
3. Enable Slash Commands and set the endpoint
4. Set environment variables (see below)

Environment Variables:
    export ORCHESTRATOR_SLACK_WEBHOOK_URL="https://hooks.slack.com/services/xxx/yyy/zzz"
    export ORCHESTRATOR_SLACK_SIGNING_SECRET="your_signing_secret"
    export ORCHESTRATOR_HOST="dashboard.example.com"
"""

import os
import asyncio
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# Import the orchestrator and Slack integration
from orchestrator import Orchestrator, Budget, ProjectState
from orchestrator.slack_integration import (
    SlackIntegrationHooks,
    SlackEndpointHandler,
    SlashCommandHandler,
    TemplateRunner,
    BudgetStats,
    FailedCheck,
    IssueItem,
    CostBreakdown,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configure Environment
# ═══════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("ORCHESTRATOR_SLACK_WEBHOOK_URL", "")
os.environ.setdefault("ORCHESTRATOR_HOST", "localhost:8888")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Implement TemplateRunner for Slash Commands
# ═══════════════════════════════════════════════════════════════════════════════

class MyTemplateRunner(TemplateRunner):
    """
    Custom template runner that integrates with your orchestrator.
    
    This creates and starts projects based on predefined templates.
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self._runs: dict[str, str] = {}  # run_id -> project_id mapping
    
    async def run_template(
        self,
        template_name: str,
        user_id: str,
        overrides: dict[str, Any],
    ) -> str:
        """
        Create and start a project from a template.
        
        Returns the run_id for tracking.
        """
        # Map template names to project configurations
        templates = {
            "secure-api-starter": {
                "description": "FastAPI + JWT authentication API",
                "budget": overrides.get("budget", 8.0),
                "allowed_models": ["gpt-4o", "claude-3-5-sonnet"],
                "security_checks": ["bandit", "safety", "secrets"],
            },
            "internal-dashboard": {
                "description": "Next.js internal dashboard",
                "budget": overrides.get("budget", 3.0),
                "allowed_models": ["gpt-4o-mini", "deepseek-chat"],
                "security_checks": ["basic"],
            },
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        config = templates[template_name]
        
        # Create the budget
        budget = Budget(max_usd=config["budget"])
        
        # Start the project
        state = await self.orchestrator.run_project(
            project_description=config["description"],
            success_criteria="Complete implementation with passing tests",
        )
        
        run_id = f"run-{state.project_id}"
        self._runs[run_id] = state.project_id
        
        return run_id
    
    def get_dashboard_url(self, run_id: str) -> str:
        """Get the dashboard URL for a run."""
        host = os.environ.get("ORCHESTRATOR_HOST", "localhost:8888")
        return f"https://{host}/runs/{run_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Create FastAPI App with Slack Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Orchestrator with Slack Integration")

# Initialize orchestrator and template runner
orch = Orchestrator()
template_runner = MyTemplateRunner(orch)

# Initialize Slack handlers
slack_hooks = SlackIntegrationHooks()
slash_handler = SlashCommandHandler(template_runner=template_runner)
slack_endpoint = SlackEndpointHandler(command_handler=slash_handler)


@app.post("/slack/slash/orchestrator")
async def slack_slash_command(request: Request):
    """
    Slack slash command endpoint.
    
    Configure in Slack App:
    - Command: /orchestrator
    - Request URL: https://your-domain.com/slack/slash/orchestrator
    - Methods: POST
    """
    return await slack_endpoint.handle_request(request)


@app.post("/slack/interactive")
async def slack_interactive(request: Request):
    """
    Handle interactive components (buttons, etc.) from Slack.
    
    This is where button clicks (like "Escalate Budget") are handled.
    """
    body = await request.body()
    payload = await request.json()
    
    # Handle different action types
    action = payload.get("actions", [{}])[0]
    action_id = action.get("action_id")
    
    if action_id == "escalate_budget":
        # Parse project_id from the payload and escalate budget
        project_id = payload.get("callback_id", "").replace("project-", "")
        # Implement budget escalation logic here
        return JSONResponse({"text": f"Budget escalated for project {project_id}"})
    
    return JSONResponse({"text": "Action handled"})


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Orchestrator Lifecycle Hooks
# ═══════════════════════════════════════════════════════════════════════════════

async def on_budget_threshold(
    project_id: str,
    run_id: str,
    threshold: float,
    budget: Budget,
):
    """Called when budget crosses a threshold."""
    stats = BudgetStats(
        total_budget=budget.max_usd,
        spent=budget.spent,
        remaining=budget.remaining,
        percentage_used=(budget.spent / budget.max_usd) * 100 if budget.max_usd else 0,
    )
    
    await slack_hooks.on_budget_threshold_crossed(
        project_id=project_id,
        run_id=run_id,
        threshold=threshold,
        budget_stats=stats,
    )


async def on_quality_gate(
    project_id: str,
    run_id: str,
    passed: bool,
    quality_score: float,
    failed_checks: list,
):
    """Called when quality gate is evaluated."""
    checks = [
        FailedCheck(name=c["name"], expected=c["expected"], actual=c["actual"])
        for c in failed_checks
    ]
    
    await slack_hooks.on_quality_gate_evaluated(
        project_id=project_id,
        run_id=run_id,
        passed=passed,
        quality_score=quality_score,
        failed_checks=checks,
    )


async def on_run_completed_hook(project_id: str, run_id: str, state: ProjectState):
    """Called when a run completes."""
    # Calculate cost breakdown
    cost_breakdown = [
        CostBreakdown(model_name="deepseek-chat", cost_usd=0.45),
        CostBreakdown(model_name="gpt-4o", cost_usd=1.20),
    ]
    
    # Collect top issues
    top_issues = [
        IssueItem(category="SECURITY", description="Hardcoded secret detected", severity="high"),
        IssueItem(category="TEST", description="2 failing tests in test_api.py", severity="medium"),
    ]
    
    await slack_hooks.on_run_completed(
        project_id=project_id,
        run_id=run_id,
        status="PASSED" if state.status.value == "completed" else "FAILED",
        total_cost=sum(c.cost_usd for c in cost_breakdown),
        cost_breakdown=cost_breakdown,
        quality_score=0.85,
        quality_gate_passed=state.status.value == "completed",
        top_issues=top_issues,
        duration_seconds=120.5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Manual Testing Functions
# ═══════════════════════════════════════════════════════════════════════════════

async def test_budget_alert():
    """Test sending a budget alert."""
    from orchestrator.slack_integration import (
        SlackNotifier, BudgetAlertPayload, BudgetStats, AlertSeverity
    )
    
    notifier = SlackNotifier()
    
    payload = BudgetAlertPayload(
        project_id="test-project-123",
        run_id="run-456",
        threshold_crossed=0.8,
        stats=BudgetStats(
            total_budget=10.0,
            spent=8.5,
            remaining=1.5,
            percentage_used=85.0,
        ),
        dashboard_url="https://dashboard.example.com/runs/run-456",
        escalate_url="https://dashboard.example.com/api/escalate",
        severity=AlertSeverity.WARNING,
    )
    
    success = await notifier.notify_budget_alert(payload)
    print(f"Budget alert sent: {success}")


async def test_quality_gate_failure():
    """Test sending a quality gate failure."""
    from orchestrator.slack_integration import (
        SlackNotifier, QualityGateFailurePayload, FailedCheck
    )
    
    notifier = SlackNotifier()
    
    payload = QualityGateFailurePayload(
        project_id="test-project-123",
        run_id="run-456",
        quality_score=0.45,
        failed_checks=[
            FailedCheck(name="Test Coverage", expected=">=80%", actual="65%"),
            FailedCheck(name="Security Scan", expected="0 issues", actual="3 issues"),
        ],
        dashboard_url="https://dashboard.example.com/runs/run-456",
        rerun_url="https://dashboard.example.com/api/rerun",
    )
    
    success = await notifier.notify_quality_gate_failure(payload)
    print(f"Quality gate alert sent: {success}")


async def test_run_summary():
    """Test sending a run summary."""
    from orchestrator.slack_integration import (
        send_run_summary_to_slack, RunSummaryPayload, CostBreakdown, IssueItem
    )
    
    payload = RunSummaryPayload(
        project_id="test-project-123",
        run_id="run-456",
        status="FAILED",
        total_cost_usd=2.35,
        cost_breakdown=[
            CostBreakdown(model_name="deepseek-chat", cost_usd=0.45, tokens_used=15000),
            CostBreakdown(model_name="gpt-4o", cost_usd=1.90, tokens_used=8000),
        ],
        quality_score=0.72,
        quality_gate_passed=False,
        top_issues=[
            IssueItem(category="SECURITY", description="Hardcoded API key", severity="high"),
            IssueItem(category="TEST", description="2 tests failing", severity="medium"),
            IssueItem(category="PERFORMANCE", description="O(n²) loop detected", severity="low"),
        ],
        duration_seconds=245,
        dashboard_url="https://dashboard.example.com/runs/run-456",
        logs_url="https://dashboard.example.com/runs/run-456/logs",
        artifacts_url="https://dashboard.example.com/runs/run-456/artifacts",
    )
    
    success = await send_run_summary_to_slack(payload)
    print(f"Run summary sent: {success}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        print("Testing Slack integration...")
        asyncio.run(test_budget_alert())
        asyncio.run(test_quality_gate_failure())
        asyncio.run(test_run_summary())
    else:
        # Run the FastAPI server
        import uvicorn
        print("Starting server on http://localhost:8000")
        print("Slash command endpoint: POST /slack/slash/orchestrator")
        uvicorn.run(app, host="0.0.0.0", port=8000)
