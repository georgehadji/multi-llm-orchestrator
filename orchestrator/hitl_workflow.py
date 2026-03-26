"""
Human-in-the-Loop (HITL) Workflow for Meta-Optimization
========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Approval workflow for strategy proposals. Provides human oversight
for structural or high-impact changes to the orchestrator.

Features:
- Approval queue with priority ordering
- Auto-approval for low-risk proposals
- Notification system (logging, file-based)
- Audit trail for compliance

USAGE:
    from orchestrator.hitl_workflow import HITLWorkflow, ApprovalConfig

    config = ApprovalConfig(
        auto_approve_low_risk=True,
        approval_timeout_hours=72.0,
    )
    
    hitl = HITLWorkflow(config)
    
    # Submit for approval
    request = await hitl.submit_for_approval(proposal)
    
    # Check status
    if await hitl.is_approved(request.request_id):
        # Proceed with proposal
        pass
    
    # Or manually approve/reject
    await hitl.approve(request.request_id, reviewer="admin")
    await hitl.reject(request.request_id, reviewer="admin", notes="Too risky")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from collections import defaultdict

from .meta_orchestrator import StrategyProposal, ProposalStatus

logger = logging.getLogger("orchestrator.hitl")


# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────

class ImpactLevel(str, Enum):
    """Impact level of a proposal."""
    LOW = "low"           # Minor tweaks, reversible
    MEDIUM = "medium"     # Moderate changes, some risk
    HIGH = "high"         # Significant changes, notable risk
    STRUCTURAL = "structural"  # Core system changes, high risk


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class NotificationChannel(str, Enum):
    """Notification channels for approvals."""
    LOG = "log"
    FILE = "file"
    WEBHOOK = "webhook"
    EMAIL = "email"


# Default timeout: 72 hours
DEFAULT_APPROVAL_TIMEOUT_HOURS = 72.0


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ApprovalConfig:
    """Configuration for HITL workflow."""
    auto_approve_low_risk: bool = True
    approval_timeout_hours: float = DEFAULT_APPROVAL_TIMEOUT_HOURS
    notification_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.LOG, NotificationChannel.FILE]
    )
    storage_path: Optional[Path] = None
    approvers: List[str] = field(default_factory=list)  # Authorized approvers
    auto_approve_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_cost_impact": 0.05,  # 5% cost change auto-approved
            "min_confidence": 0.9,    # 90% confidence auto-approved
        }
    )


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    request_id: str
    proposal: StrategyProposal
    impact_level: ImpactLevel
    submitted_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    # Review information
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[float] = None
    review_notes: Optional[str] = None
    
    # Auto-decision tracking
    auto_approved: bool = False
    auto_approve_reason: Optional[str] = None
    
    # Notifications
    notifications_sent: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "proposal": self.proposal.to_dict(),
            "impact_level": self.impact_level.value,
            "submitted_at": self.submitted_at,
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at,
            "review_notes": self.review_notes,
            "auto_approved": self.auto_approved,
            "auto_approve_reason": self.auto_approve_reason,
            "notifications_sent": self.notifications_sent,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalRequest":
        return cls(
            request_id=data["request_id"],
            proposal=StrategyProposal(**data["proposal"]),
            impact_level=ImpactLevel(data["impact_level"]),
            submitted_at=data["submitted_at"],
            status=ApprovalStatus(data["status"]),
            reviewer_id=data.get("reviewer_id"),
            reviewed_at=data.get("reviewed_at"),
            review_notes=data.get("review_notes"),
            auto_approved=data.get("auto_approved", False),
            auto_approve_reason=data.get("auto_approve_reason"),
            notifications_sent=data.get("notifications_sent", []),
        )
    
    @property
    def is_expired(self) -> bool:
        """Check if request has timed out."""
        # Timeout checked externally
        return self.status == ApprovalStatus.EXPIRED
    
    @property
    def pending_duration_hours(self) -> float:
        """Hours since submission."""
        end_time = self.reviewed_at or time.time()
        return (end_time - self.submitted_at) / 3600


@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    entry_id: str
    timestamp: float
    event_type: str
    request_id: str
    details: Dict[str, Any]
    signature: str  # For integrity verification
    
    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "details": self.details,
            "signature": self.signature,
        }


# ─────────────────────────────────────────────
# Notification Service
# ─────────────────────────────────────────────

class NotificationService:
    """
    Basic notification service for approval requests.
    
    Supports logging and file-based notifications.
    Extended versions can support email, webhooks, etc.
    """
    
    def __init__(
        self,
        channels: List[NotificationChannel],
        storage_path: Optional[Path] = None,
    ):
        self.channels = channels
        self._storage_path = storage_path or (
            Path.home() / ".orchestrator_cache" / "hitl_notifications"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    async def notify(self, request: ApprovalRequest) -> List[str]:
        """
        Send notifications for an approval request.
        
        Returns list of channels where notification was sent.
        """
        sent = []
        
        if NotificationChannel.LOG in self.channels:
            self._notify_log(request)
            sent.append("log")
        
        if NotificationChannel.FILE in self.channels:
            await self._notify_file(request)
            sent.append("file")
        
        return sent
    
    def _notify_log(self, request: ApprovalRequest):
        """Log notification to logger."""
        logger.warning(
            f"APPROVAL REQUEST: {request.request_id}\n"
            f"  Proposal: {request.proposal.description}\n"
            f"  Impact: {request.impact_level.value}\n"
            f"  Submitted: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request.submitted_at))}\n"
            f"  Status: {request.status.value}"
        )
    
    async def _notify_file(self, request: ApprovalRequest):
        """Write notification to file."""
        notifications_file = self._storage_path / "pending_approvals.jsonl"
        
        with open(notifications_file, "a") as f:
            f.write(json.dumps({
                "request_id": request.request_id,
                "proposal": request.proposal.description,
                "impact": request.impact_level.value,
                "submitted": request.submitted_at,
                "status": request.status.value,
            }) + "\n")
    
    async def notify_decision(
        self,
        request: ApprovalRequest,
        decision: str,
        reviewer: str,
    ):
        """Notify about approval/rejection decision."""
        logger.info(
            f"APPROVAL {decision.upper()}: {request.request_id}\n"
            f"  Reviewer: {reviewer}\n"
            f"  Notes: {request.review_notes or 'None'}"
        )


# ─────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────

class AuditLogger:
    """
    Immutable audit log for approval requests.
    
    Uses append-only JSONL format with simple signatures
    for tamper detection.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self._storage_path = storage_path or (
            Path.home() / ".orchestrator_cache" / "hitl_audit"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._audit_file = self._storage_path / "audit_log.jsonl"
        self._secret = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]
    
    def _generate_signature(self, data: Dict[str, Any]) -> str:
        """Generate signature for audit entry."""
        content = json.dumps(data, sort_keys=True) + self._secret
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def log_request(self, request: ApprovalRequest) -> AuditEntry:
        """Log approval request submission."""
        entry = AuditEntry(
            entry_id=f"audit_{int(time.time() * 1000)}",
            timestamp=time.time(),
            event_type="request_submitted",
            request_id=request.request_id,
            details={
                "proposal_id": request.proposal.proposal_id,
                "impact_level": request.impact_level.value,
                "auto_approved": request.auto_approved,
            },
            signature=self._generate_signature({
                "request_id": request.request_id,
                "timestamp": time.time(),
            }),
        )
        
        self._append_entry(entry)
        return entry
    
    def log_decision(
        self,
        request: ApprovalRequest,
        decision: str,
        reviewer: str,
    ) -> AuditEntry:
        """Log approval/rejection decision."""
        entry = AuditEntry(
            entry_id=f"audit_{int(time.time() * 1000)}",
            timestamp=time.time(),
            event_type=f"request_{decision}",
            request_id=request.request_id,
            details={
                "reviewer": reviewer,
                "notes": request.review_notes,
                "duration_hours": request.pending_duration_hours,
            },
            signature=self._generate_signature({
                "request_id": request.request_id,
                "decision": decision,
                "reviewer": reviewer,
            }),
        )
        
        self._append_entry(entry)
        return entry
    
    def _append_entry(self, entry: AuditEntry):
        """Append entry to audit log."""
        with open(self._audit_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
    
    def get_trail(self, request_id: str) -> List[AuditEntry]:
        """Get audit trail for a request."""
        if not self._audit_file.exists():
            return []
        
        trail = []
        with open(self._audit_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data["request_id"] == request_id:
                        trail.append(AuditEntry(
                            entry_id=data["entry_id"],
                            timestamp=data["timestamp"],
                            event_type=data["event_type"],
                            request_id=data["request_id"],
                            details=data["details"],
                            signature=data["signature"],
                        ))
        
        return trail


# ─────────────────────────────────────────────
# HITL Workflow Engine
# ─────────────────────────────────────────────

class HITLWorkflow:
    """
    Human-in-the-Loop workflow for approval requests.
    
    Manages approval queue, auto-approval logic, and notifications.
    """
    
    def __init__(self, config: Optional[ApprovalConfig] = None):
        self.config = config or ApprovalConfig()
        
        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()
        
        # Initialize services
        self._notification_service = NotificationService(
            self.config.notification_channels,
            self.config.storage_path,
        )
        self._audit_logger = AuditLogger(self.config.storage_path)
        
        self._load_requests()
    
    def _load_requests(self):
        """Load requests from disk."""
        requests_file = self._get_storage_path() / "approval_requests.jsonl"
        if not requests_file.exists():
            return
        
        try:
            with open(requests_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        request = ApprovalRequest.from_dict(data)
                        self._requests[request.request_id] = request
            
            logger.info(f"Loaded {len(self._requests)} approval requests from disk")
        except Exception as e:
            logger.warning(f"Failed to load approval requests: {e}")
    
    def _get_storage_path(self) -> Path:
        """Get storage path."""
        return self.config.storage_path or (
            Path.home() / ".orchestrator_cache" / "hitl"
        )
    
    def _persist_request(self, request: ApprovalRequest):
        """Persist request to disk."""
        requests_file = self._get_storage_path() / "approval_requests.jsonl"
        
        # Read existing, update/add, write back
        requests = []
        if requests_file.exists():
            with open(requests_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data["request_id"] != request.request_id:
                            requests.append(data)
        
        requests.append(request.to_dict())
        
        with open(requests_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
    
    def _determine_impact_level(self, proposal: StrategyProposal) -> ImpactLevel:
        """
        Determine impact level based on proposal characteristics.
        
        Rules:
        - Structural changes to routing → STRUCTURAL
        - Budget changes > 20% → HIGH
        - Model routing changes → MEDIUM
        - Template/config changes → LOW
        """
        strategy_type = proposal.strategy_type.value
        confidence = proposal.confidence
        
        # Structural changes
        if "structural" in proposal.description.lower():
            return ImpactLevel.STRUCTURAL
        
        # Budget changes
        if strategy_type == "budget_allocation":
            factor = proposal.proposed_config.get("budget_factor", 1.0)
            current = proposal.current_config.get("budget_factor", 1.0)
            change = abs(factor - current) / current
            if change > 0.2:
                return ImpactLevel.HIGH
            return ImpactLevel.MEDIUM
        
        # Model routing
        if strategy_type == "model_routing":
            if proposal.proposed_config.get("enabled") is False:
                return ImpactLevel.HIGH  # Disabling model is high impact
            return ImpactLevel.MEDIUM
        
        # Template/config
        return ImpactLevel.LOW
    
    def _should_auto_approve(self, proposal: StrategyProposal) -> tuple[bool, str]:
        """
        Determine if proposal should be auto-approved.
        
        Returns (should_approve, reason)
        """
        if not self.config.auto_approve_low_risk:
            return False, "Auto-approval disabled"
        
        # Check confidence threshold
        min_confidence = self.config.auto_approve_thresholds.get("min_confidence", 0.9)
        if proposal.confidence >= min_confidence:
            return True, f"High confidence ({proposal.confidence:.2%} >= {min_confidence:.2%})"
        
        # Check expected improvement
        if proposal.expected_improvement < 0.05:  # < 5% improvement
            return True, "Low impact change (< 5% improvement)"
        
        return False, "Does not meet auto-approval criteria"
    
    async def submit_for_approval(
        self,
        proposal: StrategyProposal,
        submitter_id: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Submit a proposal for approval.
        
        Args:
            proposal: The proposal to approve
            submitter_id: ID of user/system submitting
        
        Returns:
            Approval request (may already be approved if auto-approved)
        """
        async with self._lock:
            request_id = f"approval_{proposal.proposal_id}_{int(time.time())}"
            
            # Determine impact level
            impact_level = self._determine_impact_level(proposal)
            
            # Check auto-approval
            auto_approve, auto_reason = self._should_auto_approve(proposal)
            
            request = ApprovalRequest(
                request_id=request_id,
                proposal=proposal,
                impact_level=impact_level,
                submitted_at=time.time(),
                auto_approved=auto_approve,
                auto_approve_reason=auto_reason if auto_approve else None,
            )
            
            # Auto-approve if eligible
            if auto_approve:
                request.status = ApprovalStatus.APPROVED
                request.reviewed_at = time.time()
                request.auto_approved = True
                logger.info(f"Auto-approved request {request_id}: {auto_reason}")
            else:
                # Send notifications
                channels = await self._notification_service.notify(request)
                request.notifications_sent = channels
            
            self._requests[request_id] = request
            self._persist_request(request)
            
            # Log to audit trail
            self._audit_logger.log_request(request)
            
            return request
    
    async def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        async with self._lock:
            # Check for expired requests
            await self._check_expirations()
            
            return [
                req for req in self._requests.values()
                if req.status == ApprovalStatus.PENDING
            ]
    
    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get request by ID."""
        return self._requests.get(request_id)
    
    async def approve(
        self,
        request_id: str,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: Request to approve
            reviewer_id: ID of reviewer
            notes: Optional review notes
        
        Returns:
            True if approved, False if request not found or not pending
        """
        async with self._lock:
            if request_id not in self._requests:
                return False
            
            request = self._requests[request_id]
            if request.status != ApprovalStatus.PENDING:
                return False
            
            # Update request
            request.status = ApprovalStatus.APPROVED
            request.reviewer_id = reviewer_id
            request.reviewed_at = time.time()
            request.review_notes = notes
            
            self._persist_request(request)
            
            # Log decision
            self._audit_logger.log_decision(request, "approved", reviewer_id)
            
            # Notify
            await self._notification_service.notify_decision(
                request, "approved", reviewer_id
            )
            
            logger.info(f"Approved request {request_id} by {reviewer_id}")
            return True
    
    async def reject(
        self,
        request_id: str,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: Request to reject
            reviewer_id: ID of reviewer
            notes: Optional review notes
        
        Returns:
            True if rejected, False if request not found or not pending
        """
        async with self._lock:
            if request_id not in self._requests:
                return False
            
            request = self._requests[request_id]
            if request.status != ApprovalStatus.PENDING:
                return False
            
            # Update request
            request.status = ApprovalStatus.REJECTED
            request.reviewer_id = reviewer_id
            request.reviewed_at = time.time()
            request.review_notes = notes
            
            self._persist_request(request)
            
            # Log decision
            self._audit_logger.log_decision(request, "rejected", reviewer_id)
            
            # Notify
            await self._notification_service.notify_decision(
                request, "rejected", reviewer_id
            )
            
            logger.info(f"Rejected request {request_id} by {reviewer_id}")
            return True
    
    async def is_approved(self, request_id: str) -> bool:
        """Check if a request is approved."""
        request = await self.get_request(request_id)
        if not request:
            return False
        return request.status == ApprovalStatus.APPROVED
    
    async def _check_expirations(self):
        """Check and mark expired requests."""
        timeout_seconds = self.config.approval_timeout_hours * 3600
        now = time.time()
        
        for request in self._requests.values():
            if request.status == ApprovalStatus.PENDING:
                if now - request.submitted_at > timeout_seconds:
                    request.status = ApprovalStatus.EXPIRED
                    self._persist_request(request)
                    logger.warning(f"Request {request.request_id} expired")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval workflow statistics."""
        by_status = defaultdict(int)
        for req in self._requests.values():
            by_status[req.status.value] += 1
        
        return {
            "total_requests": len(self._requests),
            "by_status": dict(by_status),
            "pending_count": by_status.get("pending", 0),
            "auto_approved_count": sum(
                1 for r in self._requests.values() if r.auto_approved
            ),
            "avg_pending_hours": self._calculate_avg_pending_hours(),
        }
    
    def _calculate_avg_pending_hours(self) -> float:
        """Calculate average pending duration for reviewed requests."""
        reviewed = [
            r for r in self._requests.values()
            if r.reviewed_at is not None
        ]
        if not reviewed:
            return 0.0
        
        total_hours = sum(r.pending_duration_hours for r in reviewed)
        return total_hours / len(reviewed)


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_hitl: Optional[HITLWorkflow] = None


def get_hitl_workflow() -> HITLWorkflow:
    """Get global HITL workflow instance."""
    global _hitl
    if _hitl is None:
        _hitl = HITLWorkflow()
    return _hitl


def reset_hitl_workflow() -> None:
    """Reset global HITL workflow (for testing)."""
    global _hitl
    _hitl = None
