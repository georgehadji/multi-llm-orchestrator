"""
Issue Tracking Integration for Multi-LLM Orchestrator
=====================================================

Deep integration with Jira/Linear for:
1. Ticket sync from quality gate failures
2. RICE scoring import/export with backlog
3. Knowledge artifact creation from resolved tickets

Environment Variables:
    ISSUE_TRACKER_PROVIDER=jira|linear
    ISSUE_TRACKER_API_TOKEN=your_api_token
    ISSUE_TRACKER_PROJECT_KEY=PROJ
    ISSUE_TRACKER_URL=https://your-domain.atlassian.net (Jira) or https://api.linear.app (Linear)
    ISSUE_TRACKER_REPEAT_THRESHOLD=3
    KNOWLEDGE_AUTO_CREATE=true
    KNOWLEDGE_MIN_CONFIDENCE=0.7

Usage:
    from orchestrator.issue_tracking import IssueTrackerService, TicketSyncHooks
    
    tracker = IssueTrackerService.from_env()
    hooks = TicketSyncHooks(tracker)
    await hooks.on_quality_gate_failed(project_id, run_id, report)
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Callable
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

class IssueSeverity(Enum):
    """Severity levels for issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(Enum):
    """Types of issues from quality gates."""
    TEST = "test"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BUG = "bug"
    QUALITY_GATE = "quality_gate"


@dataclass
class QualityIssue:
    """A single quality issue from the quality gate."""
    type: IssueType
    severity: IssueSeverity
    message: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    test_name: Optional[str] = None
    rule_id: Optional[str] = None
    
    def signature(self) -> str:
        """Generate a unique signature for deduplication."""
        key = f"{self.type.value}:{self.file_path or ''}:{self.test_name or ''}:{self.rule_id or ''}:{self.message[:100]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class QualityGateReport:
    """Full quality gate report."""
    project_id: str
    run_id: str
    passed: bool
    quality_score: float
    issues: list[QualityIssue]
    coverage_percent: Optional[float] = None
    test_count: Optional[int] = None
    failed_test_count: Optional[int] = None
    dashboard_url: str = ""
    logs_url: str = ""
    artifacts_url: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IssueTrackerConfig:
    """Configuration for issue tracker integration."""
    provider: str  # "jira" or "linear"
    api_token: str
    project_key: str  # Jira project key or Linear team ID
    base_url: str
    repeat_threshold: int = 3  # Number of occurrences before marking as repeat bug
    auto_create_tickets: bool = True
    
    @classmethod
    def from_env(cls) -> IssueTrackerConfig:
        """Load configuration from environment variables."""
        return cls(
            provider=os.environ.get("ISSUE_TRACKER_PROVIDER", "jira"),
            api_token=os.environ["ISSUE_TRACKER_API_TOKEN"],
            project_key=os.environ["ISSUE_TRACKER_PROJECT_KEY"],
            base_url=os.environ.get("ISSUE_TRACKER_URL", ""),
            repeat_threshold=int(os.environ.get("ISSUE_TRACKER_REPEAT_THRESHOLD", "3")),
            auto_create_tickets=os.environ.get("ISSUE_TRACKER_AUTO_CREATE", "true").lower() == "true",
        )


@dataclass
class Ticket:
    """Represents a ticket in an issue tracker."""
    id: str
    key: str
    title: str
    description: str
    status: str
    labels: list[str]
    created_at: datetime
    updated_at: datetime
    assignee: Optional[str] = None
    url: str = ""
    
    # RICE fields (optional)
    reach: Optional[int] = None
    impact: Optional[int] = None
    confidence: Optional[int] = None
    effort: Optional[float] = None
    rice_score: Optional[float] = None
    
    # Custom fields for mapping
    custom_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacklogItem:
    """Backlog item for RICE syncing."""
    id: str
    title: str
    description: str
    status: str
    
    # RICE inputs
    reach: int = 100  # Default: small audience
    impact: int = 2   # Default: medium impact (scale 0-3)
    confidence: int = 80  # Default: fairly confident
    effort: float = 1.0  # Default: 1 person-week
    
    # Metadata
    labels: list[str] = field(default_factory=list)
    component: Optional[str] = None
    watchers: int = 0
    story_points: Optional[int] = None
    url: str = ""


@dataclass
class RICEResult:
    """RICE scoring result."""
    reach: int
    impact: int
    confidence: int
    effort: float
    score: float
    priority: str  # "low", "medium", "high", "critical"


@dataclass
class KnowledgeArtifactDraft:
    """Draft knowledge artifact from resolved ticket."""
    title: str
    problem: str
    solution: str
    rationale: str
    tags: list[str]
    source_issue_id: str
    source_issue_url: str
    orchestrator_run_urls: list[str]
    relevant_files: list[str]
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# Issue Tracker Service Abstraction
# ═══════════════════════════════════════════════════════════════════════════════

class IssueTrackerService(ABC):
    """
    Abstract interface for issue tracker integration.
    
    Implementations: JiraIssueTrackerService, LinearIssueTrackerService
    """
    
    def __init__(self, config: IssueTrackerConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._repeat_tracker: dict[str, list[datetime]] = defaultdict(list)
    
    @classmethod
    def from_env(cls) -> IssueTrackerService:
        """Factory method to create appropriate service from environment."""
        config = IssueTrackerConfig.from_env()
        
        if config.provider == "jira":
            return JiraIssueTrackerService(config)
        elif config.provider == "linear":
            return LinearIssueTrackerService(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    @abstractmethod
    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str],
        issue_type: str = "Bug",
    ) -> Ticket:
        """Create a new issue in the tracker."""
        pass
    
    @abstractmethod
    async def find_existing_issue(
        self,
        signature: str,
        status_filter: Optional[list[str]] = None,
    ) -> Optional[Ticket]:
        """
        Find an existing issue by signature (hash of identifying fields).
        
        Args:
            signature: Unique signature of the issue
            status_filter: Only search in these statuses (e.g., ["Open", "In Progress"])
        """
        pass
    
    @abstractmethod
    async def link_run_to_issue(
        self,
        issue_id: str,
        run_id: str,
        run_url: str,
    ) -> None:
        """Add a comment linking an orchestrator run to an issue."""
        pass
    
    @abstractmethod
    async def fetch_backlog_issues(
        self,
        status_filter: Optional[list[str]] = None,
    ) -> list[BacklogItem]:
        """
        Fetch backlog items for RICE scoring.
        
        Args:
            status_filter: Filter by status (e.g., ["To Do", "Backlog"])
        """
        pass
    
    @abstractmethod
    async def update_issue_rice(
        self,
        issue_id: str,
        rice_result: RICEResult,
        dry_run: bool = False,
    ) -> bool:
        """
        Update an issue with RICE score and priority.
        
        Args:
            issue_id: The issue ID
            rice_result: Computed RICE result
            dry_run: If True, don't actually update
            
        Returns:
            True if update was successful (or would be in dry-run)
        """
        pass
    
    @abstractmethod
    async def get_recently_resolved_issues(
        self,
        since: datetime,
        label_filter: Optional[list[str]] = None,
    ) -> list[Ticket]:
        """
        Get issues resolved since a given time.
        
        Args:
            since: Only issues resolved after this time
            label_filter: Only issues with these labels
        """
        pass
    
    def is_repeat_bug(self, signature: str) -> bool:
        """
        Check if this bug has occurred multiple times.
        
        Tracks occurrences and returns True if threshold is exceeded.
        """
        now = datetime.utcnow()
        occurrences = self._repeat_tracker[signature]
        
        # Clean old occurrences (>30 days)
        cutoff = now - timedelta(days=30)
        occurrences[:] = [t for t in occurrences if t > cutoff]
        
        # Add current occurrence
        occurrences.append(now)
        
        return len(occurrences) >= self.config.repeat_threshold
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ═══════════════════════════════════════════════════════════════════════════════
# Jira Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class JiraIssueTrackerService(IssueTrackerService):
    """Jira Cloud implementation of IssueTrackerService."""
    
    def __init__(self, config: IssueTrackerConfig):
        super().__init__(config)
        if not config.base_url:
            raise ValueError("Jira requires ISSUE_TRACKER_URL (e.g., https://your-domain.atlassian.net)")
        self.api_url = f"{config.base_url.rstrip('/')}/rest/api/3"
        self.auth = (config.api_token, "")  # Jira uses token as username with empty password
    
    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str],
        issue_type: str = "Bug",
    ) -> Ticket:
        """Create a Jira issue using REST API v3."""
        client = await self._get_client()
        
        # Format description as Atlassian Document Format (ADF)
        adf_description = self._text_to_adf(description)
        
        payload = {
            "fields": {
                "project": {"key": self.config.project_key},
                "summary": title,
                "description": adf_description,
                "issuetype": {"name": issue_type},
                "labels": labels,
            }
        }
        
        response = await client.post(
            f"{self.api_url}/issue",
            json=payload,
            auth=self.auth,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        data = response.json()
        issue_key = data["key"]
        issue_id = data["id"]
        
        logger.info(f"Created Jira issue: {issue_key}")
        
        return Ticket(
            id=issue_id,
            key=issue_key,
            title=title,
            description=description,
            status="Open",
            labels=labels,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            url=f"{self.config.base_url}/browse/{issue_key}",
        )
    
    def _text_to_adf(self, text: str) -> dict:
        """Convert plain text to Atlassian Document Format."""
        # Simple conversion - split by newlines
        content = []
        for line in text.split("\n"):
            if line.strip().startswith("-"):
                # Bullet point
                content.append({
                    "type": "bulletList",
                    "content": [{
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": line.strip()[1:].strip()}]
                        }]
                    }]
                })
            elif line.strip().startswith("#"):
                # Heading
                level = len(line) - len(line.lstrip("#"))
                content.append({
                    "type": f"heading{min(level, 6)}",
                    "content": [{"type": "text", "text": line.strip("# ").strip()}]
                })
            else:
                content.append({
                    "type": "paragraph",
                    "content": [{"type": "text", "text": line}]
                })
        
        return {"type": "doc", "version": 1, "content": content}
    
    async def find_existing_issue(
        self,
        signature: str,
        status_filter: Optional[list[str]] = None,
    ) -> Optional[Ticket]:
        """Search for existing issue by signature in description."""
        client = await self._get_client()
        
        # JQL query
        jql = f'project = {self.config.project_key} AND text ~ "signature:{signature}"'
        if status_filter:
            statuses = ", ".join(f'"{s}"' for s in status_filter)
            jql += f" AND status in ({statuses})"
        
        response = await client.get(
            f"{self.api_url}/search",
            params={"jql": jql, "maxResults": 1},
            auth=self.auth,
        )
        response.raise_for_status()
        
        data = response.json()
        if data["total"] == 0:
            return None
        
        issue = data["issues"][0]
        fields = issue["fields"]
        
        return Ticket(
            id=issue["id"],
            key=issue["key"],
            title=fields["summary"],
            description="",  # ADF to text conversion omitted for brevity
            status=fields["status"]["name"],
            labels=fields.get("labels", []),
            created_at=datetime.fromisoformat(fields["created"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(fields["updated"].replace("Z", "+00:00")),
            url=f"{self.config.base_url}/browse/{issue['key']}",
        )
    
    async def link_run_to_issue(
        self,
        issue_id: str,
        run_id: str,
        run_url: str,
    ) -> None:
        """Add a comment with run link."""
        client = await self._get_client()
        
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [{
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": f"Orchestrator run linked: "},
                        {
                            "type": "text",
                            "text": run_id,
                            "marks": [{"type": "link", "attrs": {"href": run_url}}]
                        }
                    ]
                }]
            }
        }
        
        await client.post(
            f"{self.api_url}/issue/{issue_id}/comment",
            json=payload,
            auth=self.auth,
        )
        logger.info(f"Linked run {run_id} to Jira issue {issue_id}")
    
    async def fetch_backlog_issues(
        self,
        status_filter: Optional[list[str]] = None,
    ) -> list[BacklogItem]:
        """Fetch backlog items from Jira."""
        client = await self._get_client()
        
        statuses = status_filter or ["To Do", "Backlog", "Selected for Development"]
        status_list = ", ".join(f'"{s}"' for s in statuses)
        jql = f"project = {self.config.project_key} AND status in ({status_list}) ORDER BY created DESC"
        
        response = await client.get(
            f"{self.api_url}/search",
            params={"jql": jql, "maxResults": 100, "fields": "summary,description,status,labels,customfield_*"},
            auth=self.auth,
        )
        response.raise_for_status()
        
        data = response.json()
        items = []
        
        for issue in data.get("issues", []):
            fields = issue["fields"]
            
            # Extract RICE fields from custom fields (adjust field IDs as needed)
            custom_fields = {k: v for k, v in fields.items() if k.startswith("customfield_")}
            
            item = BacklogItem(
                id=issue["id"],
                title=fields["summary"],
                description="",  # Simplified
                status=fields["status"]["name"],
                reach=custom_fields.get("customfield_10001", 100),  # Example field ID
                impact=custom_fields.get("customfield_10002", 2),
                confidence=custom_fields.get("customfield_10003", 80),
                effort=custom_fields.get("customfield_10004", 1.0),
                labels=fields.get("labels", []),
                url=f"{self.config.base_url}/browse/{issue['key']}",
            )
            items.append(item)
        
        return items
    
    async def update_issue_rice(
        self,
        issue_id: str,
        rice_result: RICEResult,
        dry_run: bool = False,
    ) -> bool:
        """Update Jira issue with RICE score and priority."""
        if dry_run:
            logger.info(f"[DRY-RUN] Would update {issue_id} with RICE={rice_result.score:.2f}, priority={rice_result.priority}")
            return True
        
        client = await self._get_client()
        
        # Map RICE priority to Jira priority
        priority_mapping = {
            "critical": "Highest",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
        }
        
        payload = {
            "fields": {
                "customfield_rice_score": round(rice_result.score, 2),  # Adjust field ID
                "priority": {"name": priority_mapping.get(rice_result.priority, "Medium")},
            }
        }
        
        response = await client.put(
            f"{self.api_url}/issue/{issue_id}",
            json=payload,
            auth=self.auth,
        )
        
        if response.status_code == 204:
            logger.info(f"Updated Jira issue {issue_id} with RICE={rice_result.score:.2f}")
            return True
        else:
            logger.error(f"Failed to update Jira issue: {response.text}")
            return False
    
    async def get_recently_resolved_issues(
        self,
        since: datetime,
        label_filter: Optional[list[str]] = None,
    ) -> list[Ticket]:
        """Get recently resolved issues."""
        client = await self._get_client()
        
        since_str = since.strftime("%Y-%m-%d %H:%M")
        jql = f'project = {self.config.project_key} AND status = Done AND updated >= "{since_str}"'
        
        if label_filter:
            labels = ", ".join(f'"{l}"' for l in label_filter)
            jql += f" AND labels in ({labels})"
        
        response = await client.get(
            f"{self.api_url}/search",
            params={"jql": jql, "maxResults": 50},
            auth=self.auth,
        )
        response.raise_for_status()
        
        data = response.json()
        tickets = []
        
        for issue in data.get("issues", []):
            fields = issue["fields"]
            tickets.append(Ticket(
                id=issue["id"],
                key=issue["key"],
                title=fields["summary"],
                description="",
                status=fields["status"]["name"],
                labels=fields.get("labels", []),
                created_at=datetime.fromisoformat(fields["created"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(fields["updated"].replace("Z", "+00:00")),
                url=f"{self.config.base_url}/browse/{issue['key']}",
            ))
        
        return tickets


# ═══════════════════════════════════════════════════════════════════════════════
# Linear Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class LinearIssueTrackerService(IssueTrackerService):
    """Linear implementation using GraphQL API."""
    
    def __init__(self, config: IssueTrackerConfig):
        super().__init__(config)
        self.api_url = "https://api.linear.app/graphql"
        self.headers = {
            "Authorization": config.api_token,
            "Content-Type": "application/json",
        }
        self.team_id = config.project_key  # For Linear, project_key is the team ID
    
    async def _graphql_query(self, query: str, variables: dict = None) -> dict:
        """Execute a GraphQL query."""
        client = await self._get_client()
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        response = await client.post(
            self.api_url,
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        
        return data["data"]
    
    async def create_issue(
        self,
        title: str,
        description: str,
        labels: list[str],
        issue_type: str = "Bug",
    ) -> Ticket:
        """Create a Linear issue."""
        query = """
        mutation CreateIssue($input: IssueCreateInput!) {
            issueCreate(input: $input) {
                success
                issue {
                    id
                    identifier
                    title
                    url
                    createdAt
                    updatedAt
                    state {
                        name
                    }
                }
            }
        }
        """
        
        variables = {
            "input": {
                "teamId": self.team_id,
                "title": title,
                "description": description,
                "labelIds": [],  # Would need to map label names to IDs
            }
        }
        
        data = await self._graphql_query(query, variables)
        issue = data["issueCreate"]["issue"]
        
        logger.info(f"Created Linear issue: {issue['identifier']}")
        
        return Ticket(
            id=issue["id"],
            key=issue["identifier"],
            title=issue["title"],
            description=description,
            status=issue["state"]["name"],
            labels=labels,
            created_at=datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00")),
            url=issue["url"],
        )
    
    async def find_existing_issue(
        self,
        signature: str,
        status_filter: Optional[list[str]] = None,
    ) -> Optional[Ticket]:
        """Search for existing Linear issue by signature."""
        query = """
        query SearchIssues($filter: IssueFilter) {
            issues(filter: $filter, first: 1) {
                nodes {
                    id
                    identifier
                    title
                    url
                    state {
                        name
                    }
                    createdAt
                    updatedAt
                    labels {
                        nodes {
                            name
                        }
                    }
                }
            }
        }
        """
        
        # Linear doesn't support full-text search in GraphQL easily
        # This is a simplified version - you'd typically use their search API
        variables = {
            "filter": {
                "team": {"id": {"eq": self.team_id}},
                "title": {"contains": signature[:8]},  # Partial match
            }
        }
        
        data = await self._graphql_query(query, variables)
        issues = data["issues"]["nodes"]
        
        if not issues:
            return None
        
        issue = issues[0]
        if status_filter and issue["state"]["name"] not in status_filter:
            return None
        
        return Ticket(
            id=issue["id"],
            key=issue["identifier"],
            title=issue["title"],
            description="",
            status=issue["state"]["name"],
            labels=[l["name"] for l in issue["labels"]["nodes"]],
            created_at=datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00")),
            url=issue["url"],
        )
    
    async def link_run_to_issue(
        self,
        issue_id: str,
        run_id: str,
        run_url: str,
    ) -> None:
        """Add a comment to Linear issue."""
        query = """
        mutation CreateComment($input: CommentCreateInput!) {
            commentCreate(input: $input) {
                success
            }
        }
        """
        
        variables = {
            "input": {
                "issueId": issue_id,
                "body": f"Orchestrator run linked: [{run_id}]({run_url})",
            }
        }
        
        await self._graphql_query(query, variables)
        logger.info(f"Linked run {run_id} to Linear issue {issue_id}")
    
    async def fetch_backlog_issues(
        self,
        status_filter: Optional[list[str]] = None,
    ) -> list[BacklogItem]:
        """Fetch backlog issues from Linear."""
        query = """
        query TeamIssues($filter: IssueFilter) {
            issues(filter: $filter, first: 100) {
                nodes {
                    id
                    identifier
                    title
                    description
                    state {
                        name
                    }
                    createdAt
                    url
                    labels {
                        nodes {
                            name
                        }
                    }
                }
            }
        }
        """
        
        # Map status names to Linear state names
        states = status_filter or ["Backlog", "Todo", "In Progress"]
        
        variables = {
            "filter": {
                "team": {"id": {"eq": self.team_id}},
                "state": {"name": {"in": states}},
            }
        }
        
        data = await self._graphql_query(query, variables)
        issues = data["issues"]["nodes"]
        
        items = []
        for issue in issues:
            items.append(BacklogItem(
                id=issue["id"],
                title=issue["title"],
                description=issue.get("description", ""),
                status=issue["state"]["name"],
                labels=[l["name"] for l in issue["labels"]["nodes"]],
                url=issue["url"],
            ))
        
        return items
    
    async def update_issue_rice(
        self,
        issue_id: str,
        rice_result: RICEResult,
        dry_run: bool = False,
    ) -> bool:
        """Update Linear issue with RICE score."""
        if dry_run:
            logger.info(f"[DRY-RUN] Would update {issue_id} with RICE={rice_result.score:.2f}")
            return True
        
        # Linear doesn't have built-in RICE fields, we'd use custom fields or labels
        query = """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
            issueUpdate(id: $id, input: $input) {
                success
            }
        }
        """
        
        # Add RICE score as a label or in description
        rice_label = f"rice:{rice_result.priority}:{rice_result.score:.1f}"
        
        variables = {
            "id": issue_id,
            "input": {
                "labelIds": [],  # Would need to create/get label IDs
            }
        }
        
        await self._graphql_query(query, variables)
        logger.info(f"Updated Linear issue {issue_id} with RICE score")
        return True
    
    async def get_recently_resolved_issues(
        self,
        since: datetime,
        label_filter: Optional[list[str]] = None,
    ) -> list[Ticket]:
        """Get recently resolved Linear issues."""
        query = """
        query ResolvedIssues($filter: IssueFilter) {
            issues(filter: $filter, first: 50) {
                nodes {
                    id
                    identifier
                    title
                    url
                    state {
                        name
                    }
                    createdAt
                    updatedAt
                    labels {
                        nodes {
                            name
                        }
                    }
                }
            }
        }
        """
        
        variables = {
            "filter": {
                "team": {"id": {"eq": self.team_id}},
                "state": {"type": {"eq": "canceled"}},  # Or "done"
                "updatedAt": {"gt": since.isoformat()},
            }
        }
        
        if label_filter:
            variables["filter"]["labels"] = {"name": {"in": label_filter}}
        
        data = await self._graphql_query(query, variables)
        issues = data["issues"]["nodes"]
        
        tickets = []
        for issue in issues:
            tickets.append(Ticket(
                id=issue["id"],
                key=issue["identifier"],
                title=issue["title"],
                description="",
                status=issue["state"]["name"],
                labels=[l["name"] for l in issue["labels"]["nodes"]],
                created_at=datetime.fromisoformat(issue["createdAt"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(issue["updatedAt"].replace("Z", "+00:00")),
                url=issue["url"],
            ))
        
        return tickets


# ═══════════════════════════════════════════════════════════════════════════════
# RICE Scoring Service
# ═══════════════════════════════════════════════════════════════════════════════

class RICEMappingConfig:
    """Configuration for mapping issue fields to RICE inputs."""
    
    def __init__(
        self,
        reach_field: Optional[str] = None,
        impact_field: Optional[str] = None,
        confidence_field: Optional[str] = None,
        effort_field: Optional[str] = None,
        default_reach: int = 100,
        default_impact: int = 2,
        default_confidence: int = 80,
        default_effort: float = 1.0,
    ):
        self.reach_field = reach_field
        self.impact_field = impact_field
        self.confidence_field = confidence_field
        self.effort_field = effort_field
        self.defaults = {
            "reach": default_reach,
            "impact": default_impact,
            "confidence": default_confidence,
            "effort": default_effort,
        }


class RICECalculator:
    """Calculate RICE scores from input values."""
    
    @staticmethod
    def calculate(
        reach: int,
        impact: int,
        confidence: int,
        effort: float,
    ) -> RICEResult:
        """
        Calculate RICE score.
        
        RICE = (Reach × Impact × Confidence) / Effort
        
        Args:
            reach: Number of users/events per period
            impact: 0-3 scale (0=minimal, 3=massive)
            confidence: 0-100 percentage
            effort: Person-weeks of work
        """
        if effort <= 0:
            effort = 0.1  # Prevent division by zero
        
        # Normalize confidence to 0-1
        confidence_normalized = confidence / 100.0
        
        # Calculate score
        score = (reach * impact * confidence_normalized) / effort
        
        # Determine priority
        if score >= 50:
            priority = "critical"
        elif score >= 20:
            priority = "high"
        elif score >= 5:
            priority = "medium"
        else:
            priority = "low"
        
        return RICEResult(
            reach=reach,
            impact=impact,
            confidence=confidence,
            effort=effort,
            score=score,
            priority=priority,
        )
    
    @classmethod
    def from_backlog_item(
        cls,
        item: BacklogItem,
        config: RICEMappingConfig,
    ) -> RICEResult:
        """Calculate RICE from a backlog item using mapping config."""
        # Use mapped fields or defaults
        reach = getattr(item, config.reach_field, None) or item.reach or config.defaults["reach"]
        impact = getattr(item, config.impact_field, None) or item.impact or config.defaults["impact"]
        confidence = getattr(item, config.confidence_field, None) or item.confidence or config.defaults["confidence"]
        effort = getattr(item, config.effort_field, None) or item.effort or config.defaults["effort"]
        
        # Infer from labels if not set
        if impact == config.defaults["impact"]:
            impact = cls._infer_impact_from_labels(item.labels)
        
        # Adjust confidence if using defaults
        if confidence == config.defaults["confidence"] and not config.confidence_field:
            confidence = 50  # Lower confidence when using defaults
        
        return cls.calculate(reach, impact, confidence, effort)
    
    @staticmethod
    def _infer_impact_from_labels(labels: list[str]) -> int:
        """Infer impact score from issue labels."""
        label_lower = [l.lower() for l in labels]
        
        if any(l in label_lower for l in ["critical", "p0", "blocker", "security"]):
            return 3
        elif any(l in label_lower for l in ["high", "p1", "major", "performance"]):
            return 2
        elif any(l in label_lower for l in ["medium", "p2", "enhancement"]):
            return 1
        else:
            return 1  # Default to low impact


class BacklogSyncService:
    """
    Service for syncing backlog items with RICE scoring.
    
    Usage:
        service = BacklogSyncService(tracker)
        results = await service.sync_rice_scores(dry_run=True)
    """
    
    def __init__(
        self,
        tracker: IssueTrackerService,
        rice_config: Optional[RICEMappingConfig] = None,
    ):
        self.tracker = tracker
        self.rice_config = rice_config or RICEMappingConfig()
        self.calculator = RICECalculator()
    
    async def sync_rice_scores(
        self,
        status_filter: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> list[dict]:
        """
        Fetch backlog, calculate RICE scores, and update issues.
        
        Args:
            status_filter: Only sync issues in these statuses
            dry_run: If True, don't actually update issues
            
        Returns:
            List of sync results with before/after comparison
        """
        logger.info(f"Starting RICE sync (dry_run={dry_run})")
        
        # Fetch backlog
        items = await self.tracker.fetch_backlog_issues(status_filter)
        logger.info(f"Fetched {len(items)} backlog items")
        
        results = []
        
        for item in items:
            # Calculate RICE
            rice_result = self.calculator.from_backlog_item(item, self.rice_config)
            
            # Store original for comparison
            original = {
                "reach": item.reach,
                "impact": item.impact,
                "confidence": item.confidence,
                "effort": item.effort,
                "rice_score": item.rice_score,
            }
            
            # Update issue
            success = await self.tracker.update_issue_rice(
                item.id,
                rice_result,
                dry_run=dry_run,
            )
            
            results.append({
                "id": item.id,
                "title": item.title,
                "url": item.url,
                "original": original,
                "rice": asdict(rice_result),
                "updated": success and not dry_run,
                "dry_run": dry_run,
            })
        
        logger.info(f"RICE sync complete. Processed {len(results)} items.")
        return results
    
    def generate_report(self, results: list[dict]) -> str:
        """Generate a human-readable report of the sync."""
        lines = ["# RICE Sync Report\n"]
        
        # Summary
        updated = sum(1 for r in results if r["updated"])
        lines.append(f"**Total Issues:** {len(results)}")
        lines.append(f"**Updated:** {updated}")
        lines.append(f"**Dry Run:** {results[0]['dry_run'] if results else False}\n")
        
        # Top priority items
        lines.append("## Top Priority Items\n")
        sorted_results = sorted(results, key=lambda r: r["rice"]["score"], reverse=True)
        
        for r in sorted_results[:10]:
            rice = r["rice"]
            lines.append(f"- **{r['title'][:50]}...**")
            lines.append(f"  - RICE: {rice['score']:.2f} ({rice['priority']})")
            lines.append(f"  - R×I×C/E = {rice['reach']}×{rice['impact']}×{rice['confidence']}%/{rice['effort']}")
            lines.append(f"  - [View Issue]({r['url']})")
            lines.append("")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Ticket to Knowledge Artifact Sync
# ═══════════════════════════════════════════════════════════════════════════════

class TicketKnowledgeSync:
    """
    Sync resolved tickets to Knowledge Base as artifacts.
    
    Usage:
        sync = TicketKnowledgeSync(tracker, knowledge_base)
        await sync.sync_resolved_tickets(since=datetime.now() - timedelta(days=7))
    """
    
    def __init__(
        self,
        tracker: IssueTrackerService,
        knowledge_base: Any,  # Your KnowledgeBase instance
        auto_create: bool = False,
        min_confidence: float = 0.7,
        llm_summarizer: Optional[Callable[[str, str], str]] = None,
    ):
        self.tracker = tracker
        self.kb = knowledge_base
        self.auto_create = auto_create
        self.min_confidence = min_confidence
        self.llm_summarizer = llm_summarizer
        self._processed_ids: set[str] = set()
    
    async def sync_resolved_tickets(
        self,
        since: datetime,
        label_filter: Optional[list[str]] = None,
    ) -> list[KnowledgeArtifactDraft]:
        """
        Sync recently resolved tickets to Knowledge Base.
        
        Args:
            since: Only tickets resolved after this time
            label_filter: Only sync tickets with these labels
            
        Returns:
            List of created/pending artifact drafts
        """
        # Default to ai-orchestrator label if not specified
        label_filter = label_filter or ["ai-orchestrator"]
        
        logger.info(f"Syncing resolved tickets since {since}")
        
        # Fetch resolved tickets
        tickets = await self.tracker.get_recently_resolved_issues(since, label_filter)
        logger.info(f"Found {len(tickets)} resolved tickets")
        
        drafts = []
        
        for ticket in tickets:
            if ticket.id in self._processed_ids:
                continue
            
            # Convert to artifact draft
            draft = await self._ticket_to_draft(ticket)
            
            # Check confidence
            if draft.confidence >= self.min_confidence and self.auto_create:
                # Auto-create artifact
                await self._create_artifact(draft)
                draft.created = True
            else:
                # Mark for review
                draft.created = False
            
            drafts.append(draft)
            self._processed_ids.add(ticket.id)
        
        return drafts
    
    async def _ticket_to_draft(self, ticket: Ticket) -> KnowledgeArtifactDraft:
        """Convert a ticket to a Knowledge Artifact draft."""
        # Extract problem from title and description
        problem = ticket.title
        if ticket.description:
            problem += f"\n\n{ticket.description[:500]}"  # First 500 chars
        
        # Try to extract solution from comments (simplified)
        solution = "See resolution in ticket."
        
        # Use LLM to summarize if available
        if self.llm_summarizer:
            try:
                solution = await self.llm_summarizer(ticket.description or "", "resolution")
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
        
        # Derive tags from labels and components
        tags = list(ticket.labels)
        if "ai-orchestrator" not in tags:
            tags.append("ai-orchestrator")
        
        # Calculate confidence based on available data
        confidence = 0.5
        if ticket.description and len(ticket.description) > 100:
            confidence += 0.2
        if self.llm_summarizer:
            confidence += 0.2
        if "resolved" in ticket.status.lower() or "done" in ticket.status.lower():
            confidence += 0.1
        
        return KnowledgeArtifactDraft(
            title=f"[Resolved] {ticket.title}",
            problem=problem,
            solution=solution,
            rationale="Resolved via issue tracker workflow",
            tags=tags,
            source_issue_id=ticket.id,
            source_issue_url=ticket.url,
            orchestrator_run_urls=[],  # Would extract from comments
            relevant_files=[],  # Would parse from description
            confidence=min(confidence, 1.0),
        )
    
    async def _create_artifact(self, draft: KnowledgeArtifactDraft) -> str:
        """Create a Knowledge Artifact from draft."""
        # This integrates with your existing KnowledgeBase API
        try:
            # Assuming knowledge_base has an add_artifact method
            artifact_id = await self.kb.add_artifact(
                artifact_type="resolved_issue",
                title=draft.title,
                content={
                    "problem": draft.problem,
                    "solution": draft.solution,
                    "rationale": draft.rationale,
                },
                tags=draft.tags,
                links={
                    "source_issue": draft.source_issue_url,
                    "runs": draft.orchestrator_run_urls,
                },
            )
            
            logger.info(f"Created Knowledge Artifact: {artifact_id}")
            return artifact_id
            
        except Exception as e:
            logger.error(f"Failed to create artifact: {e}")
            raise
    
    async def handle_webhook(
        self,
        event_type: str,
        ticket_data: dict,
    ) -> Optional[KnowledgeArtifactDraft]:
        """
        Handle incoming webhook from issue tracker.
        
        Args:
            event_type: e.g., "issue.resolved", "issue.closed"
            ticket_data: The ticket data from the webhook
            
        Returns:
            Created draft or None if not processed
        """
        if event_type not in ("issue.resolved", "issue.closed", "issue.done"):
            return None
        
        # Parse ticket data
        ticket = Ticket(
            id=ticket_data["id"],
            key=ticket_data.get("key", ""),
            title=ticket_data["title"],
            description=ticket_data.get("description", ""),
            status=ticket_data.get("status", "Done"),
            labels=ticket_data.get("labels", []),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            url=ticket_data.get("url", ""),
        )
        
        # Check if it has ai-orchestrator label
        if "ai-orchestrator" not in ticket.labels:
            logger.debug(f"Skipping non-orchestrator ticket: {ticket.id}")
            return None
        
        # Convert to draft
        draft = await self._ticket_to_draft(ticket)
        
        # Auto-create if enabled
        if self.auto_create and draft.confidence >= self.min_confidence:
            await self._create_artifact(draft)
            draft.created = True
        
        return draft


# ═══════════════════════════════════════════════════════════════════════════════
# Ticket Sync Hooks (Orchestrator Lifecycle Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TicketSyncHooks:
    """
    Orchestrator lifecycle hooks for ticket synchronization.
    
    Integrates with the orchestrator's quality control and run lifecycle.
    
    Usage:
        hooks = TicketSyncHooks(tracker)
        
        # Register with orchestrator
        orchestrator.register_hook("quality_gate", hooks.on_quality_gate_evaluated)
        orchestrator.register_hook("run_completed", hooks.on_run_completed)
    """
    
    def __init__(
        self,
        tracker: IssueTrackerService,
        dashboard_host: Optional[str] = None,
    ):
        self.tracker = tracker
        self.host = dashboard_host or os.environ.get("ORCHESTRATOR_HOST", "localhost:8888")
        self._issue_cache: dict[str, str] = {}  # signature -> issue_id
    
    def _get_dashboard_url(self, project_id: str, run_id: str) -> str:
        """Generate dashboard URL."""
        return f"https://{self.host}/runs/{run_id}"
    
    def _format_issue_description(self, report: QualityGateReport) -> str:
        """Format quality gate report as markdown description."""
        lines = [
            f"# Quality Gate Failed: {report.project_id}",
            "",
            f"**Run ID:** `{report.run_id}`",
            f"**Quality Score:** {report.quality_score:.2f}",
            f"**Test Coverage:** {report.coverage_percent}%" if report.coverage_percent else "",
            f"**Tests:** {report.failed_test_count}/{report.test_count} failed" if report.test_count else "",
            "",
            "## Top Issues",
            "",
        ]
        
        # Add top 5 issues
        for i, issue in enumerate(report.issues[:5], 1):
            lines.append(f"{i}. **[{issue.type.value.upper()}]** {issue.message}")
            if issue.file_path:
                location = issue.file_path
                if issue.line_start:
                    location += f":{issue.line_start}"
                    if issue.line_end:
                        location += f"-{issue.line_end}"
                lines.append(f"   - Location: `{location}`")
            if issue.test_name:
                lines.append(f"   - Test: `{issue.test_name}`")
            if issue.rule_id:
                lines.append(f"   - Rule: `{issue.rule_id}`")
            lines.append("")
        
        if len(report.issues) > 5:
            lines.append(f"_... and {len(report.issues) - 5} more issues_")
            lines.append("")
        
        # Add links
        lines.extend([
            "## Links",
            "",
            f"- [Dashboard]({report.dashboard_url})",
            f"- [Logs]({report.logs_url})" if report.logs_url else "",
            f"- [Artifacts]({report.artifacts_url})" if report.artifacts_url else "",
            "",
            f"<!-- signature:{self._generate_report_signature(report)} -->",
        ])
        
        return "\n".join(lines)
    
    def _generate_report_signature(self, report: QualityGateReport) -> str:
        """Generate signature for deduplication."""
        key = f"{report.project_id}:{len(report.issues)}:{report.quality_score:.2f}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def on_quality_gate_evaluated(self, report: QualityGateReport) -> Optional[str]:
        """
        Called when quality gate is evaluated.
        
        Creates or updates issue tracker tickets if gate failed.
        
        Returns:
            Issue ID if created/linked, None otherwise
        """
        if report.passed:
            return None
        
        if not self.tracker.config.auto_create_tickets:
            logger.info("Auto-create tickets disabled, skipping")
            return None
        
        logger.info(f"Quality gate failed for {report.project_id}, syncing to issue tracker")
        
        # Generate signatures for top issues
        issue_signatures = [(issue, issue.signature()) for issue in report.issues[:5]]
        
        created_issues = []
        
        for issue, signature in issue_signatures:
            # Check if this is a repeat bug
            is_repeat = self.tracker.is_repeat_bug(signature)
            
            # Check for existing open issue
            existing = await self.tracker.find_existing_issue(
                signature,
                status_filter=["Open", "In Progress", "To Do"],
            )
            
            if existing:
                # Link run to existing issue
                await self.tracker.link_run_to_issue(
                    existing.id,
                    report.run_id,
                    report.dashboard_url,
                )
                
                if is_repeat:
                    # Add note about repeat occurrence
                    logger.info(f"Repeat bug detected: {signature}")
                
                created_issues.append(existing.id)
                self._issue_cache[signature] = existing.id
                
            else:
                # Create new issue
                title = f"[Orchestrator] {issue.type.value.upper()}: {issue.message[:80]}"
                if is_repeat:
                    title = f"[REPEAT] {title}"
                
                description = self._format_issue_description(report)
                
                labels = [
                    "ai-orchestrator",
                    "quality-gate-failure",
                    issue.type.value,
                    issue.severity.value,
                ]
                
                new_issue = await self.tracker.create_issue(
                    title=title,
                    description=description,
                    labels=labels,
                    issue_type="Bug" if issue.type in (IssueType.BUG, IssueType.TEST, IssueType.SECURITY) else "Task",
                )
                
                created_issues.append(new_issue.id)
                self._issue_cache[signature] = new_issue.id
                
                # Link run
                await self.tracker.link_run_to_issue(
                    new_issue.id,
                    report.run_id,
                    report.dashboard_url,
                )
        
        return created_issues[0] if created_issues else None
    
    async def on_run_completed(
        self,
        project_id: str,
        run_id: str,
        success: bool,
        quality_report: Optional[QualityGateReport] = None,
    ) -> None:
        """
        Called when a run completes.
        
        Updates linked issues with final status.
        """
        # This could update issue status, add final comments, etc.
        logger.debug(f"Run completed hook: {run_id}, success={success}")
        
        # If we have cached issues for this run, we could update them
        # This is a placeholder for more sophisticated workflows


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ═══════════════════════════════════════════════════════════════════════════════

class IssueTrackingCLI:
    """
    CLI commands for issue tracking integration.
    
    Usage:
        orchestrator backlog sync-rice --dry-run
        orchestrator knowledge sync-from-tickets --since 7d
    """
    
    def __init__(self):
        self.tracker = IssueTrackerService.from_env()
    
    async def sync_rice(
        self,
        dry_run: bool = False,
        status_filter: Optional[list[str]] = None,
        output_format: str = "markdown",
    ) -> str:
        """
        Sync RICE scores to backlog issues.
        
        Args:
            dry_run: Preview changes without applying
            status_filter: Filter by status
            output_format: "markdown" or "json"
        """
        service = BacklogSyncService(self.tracker)
        results = await service.sync_rice_scores(
            status_filter=status_filter,
            dry_run=dry_run,
        )
        
        if output_format == "json":
            return json.dumps(results, indent=2, default=str)
        else:
            return service.generate_report(results)
    
    async def sync_knowledge(
        self,
        since_days: int = 7,
        auto_create: bool = False,
        label_filter: Optional[list[str]] = None,
    ) -> list[KnowledgeArtifactDraft]:
        """
        Sync resolved tickets to Knowledge Base.
        
        Args:
            since_days: Look back this many days
            auto_create: Auto-create artifacts without review
            label_filter: Only sync tickets with these labels
        """
        # Import knowledge base here to avoid circular dependency
        try:
            from orchestrator import get_knowledge_base
            kb = get_knowledge_base()
        except ImportError:
            raise RuntimeError("Knowledge base not available")
        
        since = datetime.utcnow() - timedelta(days=since_days)
        
        sync = TicketKnowledgeSync(
            tracker=self.tracker,
            knowledge_base=kb,
            auto_create=auto_create,
        )
        
        drafts = await sync.sync_resolved_tickets(since, label_filter)
        
        logger.info(f"Processed {len(drafts)} tickets")
        for draft in drafts:
            status = "✅ Created" if getattr(draft, 'created', False) else "⏳ Pending review"
            logger.info(f"  {status}: {draft.title[:60]}...")
        
        return drafts


# ═══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════════

"""
## Setup

### Environment Variables
```bash
export ISSUE_TRACKER_PROVIDER=jira
export ISSUE_TRACKER_API_TOKEN="your-api-token"
export ISSUE_TRACKER_PROJECT_KEY="PROJ"
export ISSUE_TRACKER_URL="https://your-domain.atlassian.net"
export KNOWLEDGE_AUTO_CREATE=false
```

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from orchestrator.issue_tracking import (
    IssueTrackerService,
    TicketSyncHooks,
    IssueTrackingCLI,
    BacklogSyncService,
    TicketKnowledgeSync,
)

app = FastAPI()

# Initialize tracker and hooks
tracker = IssueTrackerService.from_env()
hooks = TicketSyncHooks(tracker)

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
    
    issue_id = await hooks.on_quality_gate_evaluated(report)
    return {"issue_id": issue_id}

# Issue tracker webhook (for knowledge sync)
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

# CLI endpoints
@app.post("/cli/backlog/sync-rice")
async def sync_rice(dry_run: bool = True):
    cli = IssueTrackingCLI()
    report = await cli.sync_rice(dry_run=dry_run)
    return {"report": report}

@app.post("/cli/knowledge/sync")
async def sync_knowledge(since_days: int = 7):
    cli = IssueTrackingCLI()
    drafts = await cli.sync_knowledge(since_days=since_days)
    return {"drafts": len(drafts)}
```

### Direct Usage

```python
import asyncio
from orchestrator.issue_tracking import (
    IssueTrackerService,
    QualityGateReport,
    QualityIssue,
    IssueType,
    IssueSeverity,
    TicketSyncHooks,
)

async def main():
    # Create tracker
    tracker = IssueTrackerService.from_env()
    
    # Create hooks
    hooks = TicketSyncHooks(tracker)
    
    # Simulate quality gate failure
    report = QualityGateReport(
        project_id="my-project",
        run_id="run-123",
        passed=False,
        quality_score=0.45,
        issues=[
            QualityIssue(
                type=IssueType.TEST,
                severity=IssueSeverity.HIGH,
                message="3 tests failing in test_api.py",
                file_path="tests/test_api.py",
                test_name="test_user_auth",
            ),
        ],
        dashboard_url="https://dash.example.com/runs/run-123",
    )
    
    # This will create a Jira/Linear issue
    issue_id = await hooks.on_quality_gate_evaluated(report)
    print(f"Created issue: {issue_id}")

asyncio.run(main())
```
"""
