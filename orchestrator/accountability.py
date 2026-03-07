"""
Accountability Framework — Enhanced audit trail for responsibility tracking
============================================================================

Implements accountability tracking from "Agents of Chaos" paper (arXiv:2602.20021):
- Raises unresolved questions about accountability, delegated authority, and 
  responsibility for downstream harms

This module provides:
1. Action attribution - track who/what initiated each action
2. Delegation chain tracking - trace authority delegation
3. Downstream harm tracking - attribute cascading failures
4. Chain of custody for sensitive operations
5. Integration with existing audit system

Usage:
    from orchestrator.accountability import AccountabilityTracker, Action, DelegationChain
    
    tracker = AccountabilityTracker()
    
    # Track an action with attribution
    action_id = tracker.record_action(
        actor="user:admin",
        action_type="file_write",
        target="src/main.py",
        delegation_chain=["user:admin", "agent:code_writer", "tool:file_write"],
    )
    
    # Track downstream impact
    tracker.track_impact(
        action_id=action_id,
        impact_type="resource_consumption",
        severity="medium",
        description="File write consumed 100ms CPU",
    )
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .log_config import get_logger

logger = get_logger(__name__)


class ActorType(Enum):
    """Types of actors that can initiate actions."""
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"
    PLUGIN = "plugin"
    EXTERNAL = "external"


class ActionType(Enum):
    """Types of actions that can be performed."""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    COMMAND_EXECUTE = "command_execute"
    API_CALL = "api_call"
    NETWORK_REQUEST = "network_request"
    DATA_ACCESS = "data_access"
    STATE_CHANGE = "state_change"
    DELEGATION = "delegation"
    POLICY_CHECK = "policy_check"
    TASK_EXECUTE = "task_execute"
    VERIFICATION = "verification"


class ImpactSeverity(Enum):
    """Severity of downstream impacts."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactType(Enum):
    """Types of downstream impacts."""
    RESOURCE_CONSUMPTION = "resource_consumption"
    DATA_DISCLOSURE = "data_disclosure"
    SYSTEM_MODIFICATION = "system_modification"
    FAILURE_CASCADE = "failure_cascade"
    SECURITY_BREACH = "security_breach"
    PRIVACY_VIOLATION = "privacy_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class Actor:
    """Represents an actor that can perform actions."""
    id: str
    type: ActorType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.type.value}:{self.name}"


@dataclass
class Action:
    """
    Represents a single action performed in the system.
    
    This is the core unit of accountability - every significant
    operation should be recorded as an Action.
    """
    id: str
    timestamp: datetime
    actor: Actor
    action_type: ActionType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    success: bool = True
    error_message: Optional[str] = None
    delegation_chain: List[str] = field(default_factory=list)  # ["user:admin", "agent:writer", "tool:write"]
    parent_action_id: Optional[str] = None  # For tracing causation
    verification_id: Optional[str] = None  # Links to task verification
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor.id,
            "actor_type": self.actor.type.value,
            "actor_name": self.actor.name,
            "action_type": self.action_type.value,
            "target": self.target,
            "parameters": self.parameters,
            "result": str(self.result) if self.result else None,
            "success": self.success,
            "error_message": self.error_message,
            "delegation_chain": self.delegation_chain,
            "parent_action_id": self.parent_action_id,
            "verification_id": self.verification_id,
            "session_id": self.session_id,
        }


@dataclass
class Impact:
    """Represents a downstream impact from an action."""
    id: str
    timestamp: datetime
    action_id: str
    impact_type: ImpactType
    severity: ImpactSeverity
    description: str
    affected_systems: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "impact_type": self.impact_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_systems": self.affected_systems,
            "metadata": self.metadata,
        }


@dataclass
class DelegationRecord:
    """Records a delegation of authority from one actor to another."""
    id: str
    timestamp: datetime
    delegator: str  # Actor ID
    delegatee: str  # Actor ID
    scope: List[str] = field(default_factory=list)  # What actions are delegated
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    revoked: bool = False
    
    def is_valid(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class AccountabilityTracker:
    """
    Tracks accountability for all actions in the system.
    
    Provides:
    - Complete action history with attribution
    - Delegation chain tracking
    - Downstream impact monitoring
    - Chain of custody for sensitive operations
    
    This addresses the paper's concern about accountability for downstream harms.
    """

    def __init__(self):
        self._actions: Dict[str, Action] = {}
        self._impacts: Dict[str, List[Impact]] = {}
        self._delegations: Dict[str, DelegationRecord] = {}
        self._actor_sessions: Dict[str, str] = {}  # session_id -> actor_id
        self._action_sessions: Dict[str, str] = {}  # action_id -> session_id

    def start_session(self, actor_id: str, session_id: Optional[str] = None) -> str:
        """Start a new session for an actor."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        self._actor_sessions[session_id] = actor_id
        logger.debug(f"Started session {session_id} for actor {actor_id}")
        return session_id

    def end_session(self, session_id: str) -> None:
        """End an actor's session."""
        self._actor_sessions.pop(session_id, None)
        logger.debug(f"Ended session {session_id}")

    def get_current_session(self, actor_id: str) -> Optional[str]:
        """Get the current session ID for an actor."""
        for session_id, a_id in self._actor_sessions.items():
            if a_id == actor_id:
                return session_id
        return None

    def record_action(
        self,
        actor_id: str,
        actor_type: ActorType,
        actor_name: str,
        action_type: ActionType,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        delegation_chain: Optional[List[str]] = None,
        parent_action_id: Optional[str] = None,
        verification_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a new action with full attribution.
        
        Returns the action_id for tracking downstream impacts.
        """
        action_id = str(uuid.uuid4())
        
        actor = Actor(
            id=actor_id,
            type=actor_type,
            name=actor_name,
            metadata=metadata or {},
        )
        
        action = Action(
            id=action_id,
            timestamp=datetime.utcnow(),
            actor=actor,
            action_type=action_type,
            target=target,
            parameters=parameters or {},
            result=result,
            success=success,
            error_message=error_message,
            delegation_chain=delegation_chain or [],
            parent_action_id=parent_action_id,
            verification_id=verification_id,
            session_id=session_id,
        )
        
        self._actions[action_id] = action
        
        if session_id:
            self._action_sessions[action_id] = session_id
        
        logger.debug(f"Recorded action {action_id}: {actor_type.value}:{actor_name} -> {action_type.value} on {target}")
        
        return action_id

    def record_delegation(
        self,
        delegator_id: str,
        delegatee_id: str,
        scope: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> str:
        """Record a delegation of authority."""
        delegation_id = str(uuid.uuid4())
        
        delegation = DelegationRecord(
            id=delegation_id,
            timestamp=datetime.utcnow(),
            delegator=delegator_id,
            delegatee=delegatee_id,
            scope=scope or [],
            conditions=conditions or {},
            expires_at=expires_at,
        )
        
        self._delegations[delegation_id] = delegation
        logger.info(f"Recorded delegation: {delegator_id} -> {delegatee_id} (scope: {scope})")
        
        return delegation_id

    def revoke_delegation(self, delegation_id: str) -> bool:
        """Revoke a delegation."""
        delegation = self._delegations.get(delegation_id)
        if delegation:
            delegation.revoked = True
            logger.info(f"Revoked delegation {delegation_id}")
            return True
        return False

    def get_delegation_chain(self, actor_id: str) -> List[DelegationRecord]:
        """Get all delegations for an actor."""
        return [
            d for d in self._delegations.values()
            if d.delegatee == actor_id and d.is_valid()
        ]

    def track_impact(
        self,
        action_id: str,
        impact_type: ImpactType,
        severity: ImpactSeverity,
        description: str,
        affected_systems: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track a downstream impact from an action."""
        impact_id = str(uuid.uuid4())
        
        impact = Impact(
            id=impact_id,
            timestamp=datetime.utcnow(),
            action_id=action_id,
            impact_type=impact_type,
            severity=severity,
            description=description,
            affected_systems=affected_systems or [],
            metadata=metadata or {},
        )
        
        if action_id not in self._impacts:
            self._impacts[action_id] = []
        self._impacts[action_id].append(impact)
        
        logger.debug(f"Tracked impact {impact_id} for action {action_id}: {severity.value} {impact_type.value}")
        
        return impact_id

    def get_action(self, action_id: str) -> Optional[Action]:
        """Get an action by ID."""
        return self._actions.get(action_id)

    def get_impacts_for_action(self, action_id: str) -> List[Impact]:
        """Get all impacts for an action."""
        return self._impacts.get(action_id, [])

    def get_cascading_impacts(self, action_id: str) -> List[Impact]:
        """Get all cascading impacts (impacts from child actions)."""
        all_impacts = []
        
        # Get direct impacts
        direct = self._impacts.get(action_id, [])
        all_impacts.extend(direct)
        
        # Find child actions
        child_actions = [
            a for a in self._actions.values()
            if a.parent_action_id == action_id
        ]
        
        # Recursively get impacts
        for child in child_actions:
            all_impacts.extend(self.get_cascading_impacts(child.id))
        
        return all_impacts

    def get_high_severity_impacts(self) -> List[Impact]:
        """Get all high or critical severity impacts."""
        high_severities = {ImpactSeverity.HIGH, ImpactSeverity.CRITICAL}
        return [
            impact
            for impacts in self._impacts.values()
            for impact in impacts
            if impact.severity in high_severities
        ]

    def get_actions_by_actor(self, actor_id: str) -> List[Action]:
        """Get all actions performed by an actor."""
        return [
            a for a in self._actions.values()
            if a.actor.id == actor_id
        ]

    def get_failed_actions(self) -> List[Action]:
        """Get all failed actions."""
        return [a for a in self._actions.values() if not a.success]

    def get_accountability_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate an accountability report for a time period."""
        actions = list(self._actions.values())
        
        if start_time:
            actions = [a for a in actions if a.timestamp >= start_time]
        if end_time:
            actions = [a for a in actions if a.timestamp <= end_time]
        
        # Group by actor
        actor_stats: Dict[str, Dict[str, Any]] = {}
        for action in actions:
            actor_key = str(action.actor)
            if actor_key not in actor_stats:
                actor_stats[actor_key] = {
                    "total_actions": 0,
                    "successful": 0,
                    "failed": 0,
                    "action_types": {},
                }
            
            actor_stats[actor_key]["total_actions"] += 1
            if action.success:
                actor_stats[actor_key]["successful"] += 1
            else:
                actor_stats[actor_key]["failed"] += 1
            
            at = action.action_type.value
            actor_stats[actor_key]["action_types"][at] = \
                actor_stats[actor_key]["action_types"].get(at, 0) + 1
        
        # Get high severity impacts
        high_impacts = self.get_high_severity_impacts()
        
        return {
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "summary": {
                "total_actions": len(actions),
                "total_impacts": sum(len(i) for i in self._impacts.values()),
                "high_severity_impacts": len(high_impacts),
                "failed_actions": len([a for a in actions if not a.success]),
            },
            "actor_statistics": actor_stats,
            "recent_high_impacts": [
                i.to_dict() for i in high_impacts[-10:]
            ],
        }

    def flush_jsonl(self, path: str | Path) -> None:
        """Flush all actions and impacts to a JSONL file."""
        with open(path, "a", encoding="utf-8") as fh:
            # Write actions
            for action in self._actions.values():
                fh.write(json.dumps({"type": "action", **action.to_dict()}) + "\n")
            
            # Write impacts
            for impacts in self._impacts.values():
                for impact in impacts:
                    fh.write(json.dumps({"type": "impact", **impact.to_dict()}) + "\n")

    def clear(self) -> None:
        """Clear all tracked data."""
        self._actions.clear()
        self._impacts.clear()
        self._delegations.clear()
        self._actor_sessions.clear()
        self._action_sessions.clear()

    def __len__(self) -> int:
        return len(self._actions)