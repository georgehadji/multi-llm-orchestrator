"""
Example: Integrating Git Integration into Orchestrator Engine
===============================================================

This example shows how to wire the Git integration hooks into the
orchestrator run lifecycle.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

# ============================================================================
# EXAMPLE 1: Basic Integration in Orchestrator.run_project()
# ============================================================================


async def run_project_with_git_integration(
    self,
    project_description: str,
    success_criteria: str,
    git_context: GitHookContext | None = None,
) -> ProjectState:
    """Run project with GitHub/GitLab integration."""

    # Initialize Git hooks
    git_hooks = GitIntegrationHooks.from_config()

    # 1. Run start - create check run
    if git_context:
        await git_hooks.on_run_start(
            context=git_context,
            project_description=project_description,
        )

    # 2. Decompose project
    tasks = await self._decompose(project_description, success_criteria)

    # 3. Run in progress - update check run
    if git_context:
        await git_hooks.on_run_in_progress(
            context=git_context,
            total_tasks=len(tasks),
        )

    # 4. Execute tasks
    state = await self._execute_tasks(tasks)

    # 5. Post code review comments (if PR context)
    if git_context and git_context.pr_number:
        review_results = self._extract_code_reviews(state)
        await git_hooks.post_code_review_comments(
            context=git_context,
            review_results=review_results,
            policy_set=self._active_policies,
        )

    # 6. Run complete - finalize check run
    if git_context:
        dashboard_url = f"{self.config.dashboard_url}/runs/{git_context.run_id}"
        await git_hooks.on_run_complete(
            context=git_context,
            state=state,
            dashboard_url=dashboard_url,
        )

    # 7. Auto-commit / create PR if enabled
    if git_context and state.status == ProjectStatus.SUCCESS:
        generated_files = self._extract_generated_files(state)
        result = await git_hooks.auto_commit_or_create_pr(
            context=git_context,
            state=state,
            files=generated_files,
            policy_set=self._active_policies,
        )
        if result:
            logger.info(f"Git action result: {result}")

    return state


# ============================================================================
# EXAMPLE 2: CI/CD Integration (GitHub Actions)
# ============================================================================

"""
# .github/workflows/ai-orchestrator.yml

name: AI Orchestrator

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  orchestrate:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      checks: write

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Install orchestrator
      run: pip install multi-llm-orchestrator

    - name: Run AI Orchestrator
      env:
        # Git integration config
        GIT_INTEGRATION_ENABLED: "true"
        GIT_PROVIDER: "github"
        GIT_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        GIT_REPOSITORY: ${{ github.repository }}
        DASHBOARD_URL: "https://orchestrator.example.com"

        # LLM API keys
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}

      run: |
        python -m orchestrator \
          --project "${{ github.event.head_commit.message }}" \
          --criteria "All tests pass, code reviewed" \
          --budget 5.0 \
          --git-commit ${{ github.sha }} \
          --git-branch ${{ github.ref_name }} \
          --git-pr ${{ github.event.number }}
"""


# ============================================================================
# EXAMPLE 3: Policy Configuration
# ============================================================================

"""
# policies.yaml - Example policy configuration for Git integration

global:
  - name: git_integration
    allow_auto_commit: false  # Never auto-commit to main
    require_human_approval: true  # Always require PR review
    enable_pr_comments: true  # Post code review comments

  - name: cost_cap
    max_cost_per_task_usd: 0.50

team:
  frontend:
    - name: auto_commit_feature_branches
      allow_auto_commit: true  # Allow auto-commit on feature branches
      require_human_approval: false

    - name: pr_comments
      enable_pr_comments: true
"""


# ============================================================================
# EXAMPLE 4: Custom Git Integration Config
# ============================================================================

from orchestrator.git_service import GitIntegrationConfig, GitIntegrationHooks

# Manual configuration (instead of env vars)
config = GitIntegrationConfig(
    enabled=True,
    provider="github",
    token="ghp_xxxxxxxxxxxxxxxx",
    repository="myorg/myrepo",
    enable_pr_comments=True,
    enable_auto_commit=False,  # Start safe
    require_human_approval=True,
    dashboard_url="https://orchestrator.mycompany.com",
)

git_hooks = GitIntegrationHooks.from_config(config)


# ============================================================================
# EXAMPLE 5: Testing Git Integration Locally
# ============================================================================


async def test_git_integration():
    """Test the Git integration locally."""

    # Mock context for testing
    context = GitHookContext(
        run_id="test-run-123",
        commit_sha="abc123def456",
        branch="feature/test",
        pr_number=42,
        repository="myorg/myrepo",
    )

    # Create hooks
    hooks = GitIntegrationHooks.from_config()

    # Test check run lifecycle
    await hooks.on_run_start(
        context=context,
        project_description="Test project",
    )

    # Simulate work...
    import asyncio

    await asyncio.sleep(2)

    # Complete
    from orchestrator.models import Budget, ProjectState

    state = ProjectState(
        project_description="Test",
        success_criteria="Pass",
        budget=Budget(max_usd=5.0, spent_usd=0.5),
    )

    await hooks.on_run_complete(
        context=context,
        state=state,
        dashboard_url="http://localhost:8888",
    )


# ============================================================================
# USAGE SUMMARY
# ============================================================================

"""
Environment Variables:
---------------------
GIT_INTEGRATION_ENABLED=true      # Enable Git integration
GIT_PROVIDER=github               # github or gitlab
GIT_TOKEN=ghp_xxxxxxxxxxxx        # GitHub/GitLab token
GIT_REPOSITORY=owner/repo         # Repository name
DASHBOARD_URL=https://...         # Dashboard base URL
ENABLE_PR_COMMENTS=true           # Enable PR code review comments
ENABLE_AUTO_COMMIT=false          # Enable auto-commit (careful!)
REQUIRE_HUMAN_APPROVAL=true       # Require PR approval

Policy Flags:
------------
allow_auto_commit: bool           # Allow direct commits
require_human_approval: bool      # Require human PR review
enable_pr_comments: bool          # Post code review comments

Integration Points:
------------------
1. on_run_start()        -> Create "queued" check run
2. on_run_in_progress()  -> Update to "in_progress"
3. post_code_review_comments() -> Post PR review comments
4. on_run_complete()     -> Finalize check run (success/failure)
5. auto_commit_or_create_pr() -> Commit changes or open PR
"""
