"""
Cross-Agent Propagation Guards — Prevent unsafe practices spreading between agents
=================================================================================

Implements the finding from "Agents of Chaos" paper (arXiv:2602.20021):
- Cross-agent propagation of unsafe practices

This module provides:
1. Safety signal tracking per agent
2. Pattern detection for unsafe behavior propagation
3. Agent quarantine mechanisms
4. Safety barrier enforcement between agents
5. Behavioral anomaly detection

Usage:
    from orchestrator.agent_safety import AgentSafetyMonitor, SafetyLevel

    monitor = AgentSafetyMonitor()

    # Register an agent
    monitor.register_agent("agent_001", "code_writer")

    # Report safety-relevant events
    monitor.report_event(
        agent_id="agent_001",
        event_type="security_bypass_attempt",
        severity=1,
        details={"reason": "skipping validation"},
    )

    # Check if agent can interact with others
    if monitor.can_interact("agent_001", "agent_002"):
        # Allow interaction
        pass
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class SafetyLevel(Enum):
    """Safety classification for agents."""
    TRUSTED = "trusted"           # Fully trusted, can interact freely
    NORMAL = "normal"             # Standard safety, normal interactions
    SUSPICIOUS = "suspicious"     # Exhibited concerning behavior
    QUARANTINED = "quarantined"   # Isolated, no interactions allowed
    COMPROMISED = "compromised"   # Known to be acting unsafely


class SafetyEventType(Enum):
    """Types of safety-relevant events."""
    SECURITY_BYPASS_ATTEMPT = "security_bypass_attempt"
    VALIDATION_SKIP = "validation_skip"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    UNSAFE_DELEGATION = "unsafe_delegation"
    ERROR_SUPPRESSION = "error_suppression"
    OUTPUT_MANIPULATION = "output_manipulation"
    RESOURCE_ABUSE = "resource_abuse"
    DATA_LEAK = "data_leak"
    IDENTITY_SPOOFING = "identity_spoofing"


@dataclass
class SafetyEvent:
    """A safety-relevant event from an agent."""
    id: str
    timestamp: datetime
    agent_id: str
    event_type: SafetyEventType
    severity: int  # 0-10 scale
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None  # For tracing propagation


@dataclass
class AgentSafetyProfile:
    """Safety profile for an individual agent."""
    agent_id: str
    agent_type: str
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    events: list[SafetyEvent] = field(default_factory=list)
    blocked_interactions: set[str] = field(default_factory=set)  # Agent IDs
    trusted_agents: set[str] = field(default_factory=set)  # Agent IDs
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    violation_count: int = 0
    warning_count: int = 0

    @property
    def risk_score(self) -> float:
        """Calculate risk score based on events and violations."""
        score = 0.0

        # Count recent events (last hour)
        recent = datetime.utcnow() - timedelta(hours=1)
        recent_events = [e for e in self.events if e.timestamp > recent]

        for event in recent_events:
            score += event.severity * 0.5

        # Add violation weight
        score += self.violation_count * 2.0
        score += self.warning_count * 0.5

        return min(score, 10.0)  # Cap at 10


@dataclass
class InteractionRequest:
    """Request for inter-agent interaction."""
    id: str
    timestamp: datetime
    source_agent_id: str
    target_agent_id: str
    interaction_type: str
    data: dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    rejection_reason: str | None = None


class AgentSafetyMonitor:
    """
    Monitors agent behavior and prevents unsafe propagation between agents.

    Implements defense against cross-agent unsafe practice propagation by:
    1. Tracking safety events per agent
    2. Calculating risk scores
    3. Enforcing interaction barriers
    4. Quarantining compromised agents
    """

    def __init__(
        self,
        quarantine_threshold: float = 5.0,
        suspicious_threshold: float = 2.0,
        max_events_retain: int = 1000,
    ):
        self._profiles: dict[str, AgentSafetyProfile] = {}
        self._interaction_log: list[InteractionRequest] = []
        self._quarantine_threshold = quarantine_threshold
        self._suspicious_threshold = suspicious_threshold
        self._max_events_retain = max_events_retain

        # Pattern signatures for unsafe behaviors
        self._unsafe_patterns: dict[str, list[str]] = {
            "validation_skip": [
                r"skip.*validation",
                r"bypass.*check",
                r"disable.*security",
            ],
            "error_suppression": [
                r"except.*pass",
                r"try:.*pass.*except",
                r"suppress.*error",
            ],
            "output_manipulation": [
                r"fake.*success",
                r"mock.*result",
                r"return.*True.*always",
            ],
        }

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        trusted_agents: list[str] | None = None,
    ) -> AgentSafetyProfile:
        """Register a new agent in the safety system."""
        profile = AgentSafetyProfile(
            agent_id=agent_id,
            agent_type=agent_type,
            trusted_agents=set(trusted_agents or []),
        )
        self._profiles[agent_id] = profile
        logger.info(f"Registered agent {agent_id} (type: {agent_type}) in safety monitor")
        return profile

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._profiles.pop(agent_id, None)
        logger.info(f"Unregistered agent {agent_id} from safety monitor")

    def report_event(
        self,
        agent_id: str,
        event_type: SafetyEventType,
        severity: int,
        description: str,
        details: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> str:
        """
        Report a safety-relevant event from an agent.

        Returns the event ID.
        """
        profile = self._profiles.get(agent_id)
        if profile is None:
            logger.warning(f"Event from unknown agent {agent_id}, registering new profile")
            profile = self.register_agent(agent_id, "unknown")

        event = SafetyEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            event_type=event_type,
            severity=min(max(severity, 0), 10),  # Clamp to 0-10
            description=description,
            details=details or {},
            parent_event_id=parent_event_id,
        )

        profile.events.append(event)
        profile.last_activity = datetime.utcnow()

        # Trim old events if needed
        if len(profile.events) > self._max_events_retain:
            profile.events = profile.events[-self._max_events_retain:]

        # Update violation/warning counts
        if severity >= 7:
            profile.violation_count += 1
        elif severity >= 4:
            profile.warning_count += 1

        # Update safety level based on risk score
        self._update_safety_level(profile)

        logger.warning(f"Safety event for {agent_id}: {event_type.value} (severity: {severity})")

        return event.id

    def _update_safety_level(self, profile: AgentSafetyProfile) -> None:
        """Update agent's safety level based on risk score."""
        risk = profile.risk_score

        old_level = profile.safety_level

        if risk >= self._quarantine_threshold:
            profile.safety_level = SafetyLevel.QUARANTINED
        elif risk >= self._suspicious_threshold:
            profile.safety_level = SafetyLevel.SUSPICIOUS
        elif risk < 1.0 and profile.violation_count == 0:
            profile.safety_level = SafetyLevel.TRUSTED
        else:
            profile.safety_level = SafetyLevel.NORMAL

        if old_level != profile.safety_level:
            logger.warning(f"Agent {profile.agent_id} safety level changed: {old_level.value} -> {profile.safety_level.value}")

    def can_interact(
        self,
        source_agent_id: str,
        target_agent_id: str,
        interaction_type: str = "default",
    ) -> tuple[bool, str | None]:
        """
        Check if two agents can interact.

        Returns (allowed, reason) tuple.
        """
        source = self._profiles.get(source_agent_id)
        target = self._profiles.get(target_agent_id)

        # Unknown agents are allowed but logged
        if source is None or target is None:
            return True, None

        # Check if source is blocked from target
        if target_agent_id in source.blocked_interactions:
            return False, f"Agent {source_agent_id} is blocked from interacting with {target_agent_id}"

        # Check source's safety level
        if source.safety_level == SafetyLevel.QUARANTINED:
            return False, f"Source agent {source_agent_id} is quarantined"

        if source.safety_level == SafetyLevel.COMPROMISED:
            return False, f"Source agent {source_agent_id} is compromised"

        # Check target's safety level
        if target.safety_level == SafetyLevel.QUARANTINED:
            return False, f"Target agent {target_agent_id} is quarantined"

        if target.safety_level == SafetyLevel.COMPROMISED:
            return False, f"Target agent {target_agent_id} is compromised"

        # Check trust relationship
        if target_agent_id in source.trusted_agents:
            return True, None

        # Suspicious agents have limited interaction
        if source.safety_level == SafetyLevel.SUSPICIOUS:
            # Allow only read-only interactions
            if interaction_type in ("read", "query", "status"):
                return True, None
            return False, f"Suspicious agent {source_agent_id} has limited interaction permissions"

        return True, None

    def request_interaction(
        self,
        source_agent_id: str,
        target_agent_id: str,
        interaction_type: str,
        data: dict[str, Any] | None = None,
    ) -> InteractionRequest:
        """Log and approve/reject an interaction request."""
        request = InteractionRequest(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            interaction_type=interaction_type,
            data=data or {},
        )

        approved, reason = self.can_interact(source_agent_id, target_agent_id, interaction_type)
        request.approved = approved
        request.rejection_reason = reason

        self._interaction_log.append(request)

        if not approved:
            logger.warning(f"Interaction denied: {source_agent_id} -> {target_agent_id}: {reason}")

        return request

    def quarantine_agent(self, agent_id: str, reason: str) -> None:
        """Quarantine an agent, preventing all interactions."""
        profile = self._profiles.get(agent_id)
        if profile:
            profile.safety_level = SafetyLevel.QUARANTINED
            profile.blocked_interactions = set(self._profiles.keys()) - {agent_id}
            logger.critical(f"Agent {agent_id} quarantined: {reason}")

    def unquarantine_agent(self, agent_id: str) -> None:
        """Remove an agent from quarantine."""
        profile = self._profiles.get(agent_id)
        if profile:
            profile.safety_level = SafetyLevel.NORMAL
            profile.blocked_interactions = set()
            logger.info(f"Agent {agent_id} removed from quarantine")

    def mark_compromised(self, agent_id: str, reason: str) -> None:
        """Mark an agent as compromised."""
        profile = self._profiles.get(agent_id)
        if profile:
            profile.safety_level = SafetyLevel.COMPROMISED
            profile.blocked_interactions = set(self._profiles.keys()) - {agent_id}
            logger.critical(f"Agent {agent_id} marked as compromised: {reason}")

    def get_agent_profile(self, agent_id: str) -> AgentSafetyProfile | None:
        """Get safety profile for an agent."""
        return self._profiles.get(agent_id)

    def get_all_quarantined(self) -> list[AgentSafetyProfile]:
        """Get all quarantined agents."""
        return [
            p for p in self._profiles.values()
            if p.safety_level == SafetyLevel.QUARANTINED
        ]

    def get_suspicious_agents(self) -> list[AgentSafetyProfile]:
        """Get all suspicious agents."""
        return [
            p for p in self._profiles.values()
            if p.safety_level == SafetyLevel.SUSPICIOUS
        ]

    def detect_unsafe_patterns(
        self,
        content: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect unsafe patterns in content (e.g., code, prompts).

        Returns list of detected patterns with details.
        """
        import re

        detected = []

        for pattern_type, patterns in self._unsafe_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    detected.append({
                        "pattern_type": pattern_type,
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.start(),
                    })

                    # Optionally report event
                    if agent_id:
                        self.report_event(
                            agent_id=agent_id,
                            event_type=SafetyEventType.SUSPICIOUS_PATTERN,
                            severity=3,
                            description=f"Detected unsafe pattern: {pattern_type}",
                            details={"pattern": pattern, "match": match.group()},
                        )

        return detected

    def get_safety_summary(self) -> dict[str, Any]:
        """Get summary of safety status across all agents."""
        total = len(self._profiles)
        by_level = {}
        for profile in self._profiles.values():
            level = profile.safety_level.value
            by_level[level] = by_level.get(level, 0) + 1

        return {
            "total_agents": total,
            "by_safety_level": by_level,
            "quarantined_count": by_level.get("quarantined", 0),
            "suspicious_count": by_level.get("suspicious", 0),
            "compromised_count": by_level.get("compromised", 0),
            "total_interactions": len(self._interaction_log),
            "denied_interactions": sum(1 for i in self._interaction_log if not i.approved),
        }

    def clear(self) -> None:
        """Clear all tracked data."""
        self._profiles.clear()
        self._interaction_log.clear()
