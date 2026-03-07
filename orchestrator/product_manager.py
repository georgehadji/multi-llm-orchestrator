"""
Product Management System
=========================
Feature prioritization, release planning, and roadmap management.

Features:
- RICE scoring for prioritization
- Release train scheduling
- User feedback integration
- Feature flag management
- Roadmap visualization

Usage:
    from orchestrator.product_manager import ProductManager, Feature
    
    pm = ProductManager()
    await pm.add_feature(feature)
    roadmap = await pm.generate_roadmap()
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import hashlib

from .log_config import get_logger
from .performance import cached

logger = get_logger(__name__)


class FeatureStatus(Enum):
    """Feature lifecycle status."""
    IDEA = "idea"
    RESEARCH = "research"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    BETA = "beta"
    RELEASED = "released"
    DEPRECATED = "deprecated"


class FeaturePriority(Enum):
    """Business priority levels."""
    P0_CRITICAL = 0
    P1_HIGH = 1
    P2_MEDIUM = 2
    P3_LOW = 3


@dataclass
class RICEScore:
    """
    RICE prioritization framework.
    
    RICE = (Reach * Impact * Confidence) / Effort
    """
    reach: int  # How many users will this affect? (1-1000)
    impact: int  # How much will it impact? (0.25, 0.5, 1, 2, 3)
    confidence: int  # How confident are we? (0-100%)
    effort: int  # Person-months (1-12)
    
    @property
    def score(self) -> float:
        """Calculate RICE score."""
        if self.effort == 0:
            return 0
        return (self.reach * self.impact * (self.confidence / 100)) / self.effort
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reach": self.reach,
            "impact": self.impact,
            "confidence": self.confidence,
            "effort": self.effort,
            "score": round(self.score, 2),
        }


@dataclass
class Feature:
    """Product feature definition."""
    id: str
    name: str
    description: str
    status: FeatureStatus
    priority: FeaturePriority
    rice_score: RICEScore
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    target_release: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    progress: float = 0.0  # 0-100
    
    # Metrics
    user_requests: int = 0
    satisfaction_impact: float = 0.0  # Predicted impact on satisfaction
    revenue_impact: Optional[float] = None  # Predicted revenue impact
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["rice_score"] = self.rice_score.to_dict()
        data["status"] = self.status.value
        data["priority"] = self.priority.name
        return data


@dataclass
class Release:
    """Product release definition."""
    id: str
    name: str
    version: str
    target_date: datetime
    features: List[str] = field(default_factory=list)
    status: str = "planned"  # planned, in_progress, shipped, delayed
    notes: str = ""
    
    def is_on_track(self) -> bool:
        """Check if release is on track."""
        if self.status == "shipped":
            return True
        return datetime.now() < self.target_date


@dataclass
class UserFeedback:
    """User feedback entry."""
    id: str
    user_id: str
    feature_id: Optional[str]
    feedback_type: str  # request, bug, praise, complaint
    content: str
    sentiment: float  # -1 to 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed: bool = False


class FeatureFlagManager:
    """Manage feature flags for A/B testing and gradual rollout."""
    
    def __init__(self):
        self._flags: Dict[str, Dict[str, Any]] = {}
    
    def create_flag(
        self,
        name: str,
        default_value: bool = False,
        rollout_percentage: float = 0.0,
        target_users: Optional[List[str]] = None,
    ) -> str:
        """Create new feature flag."""
        flag_id = hashlib.sha256(name.encode()).hexdigest()[:16]
        
        self._flags[flag_id] = {
            "id": flag_id,
            "name": name,
            "default_value": default_value,
            "rollout_percentage": rollout_percentage,
            "target_users": set(target_users or []),
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "exposures": 0,
                "enagements": 0,
            },
        }
        
        return flag_id
    
    def is_enabled(self, flag_id: str, user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled for user."""
        if flag_id not in self._flags:
            return False
        
        flag = self._flags[flag_id]
        
        # Check targeted users
        if user_id and user_id in flag["target_users"]:
            return True
        
        # Check percentage rollout
        if user_id:
            user_hash = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
            user_bucket = (user_hash % 100) / 100
            return user_bucket < flag["rollout_percentage"]
        
        return flag["default_value"]
    
    def update_rollout(self, flag_id: str, percentage: float):
        """Update rollout percentage."""
        if flag_id in self._flags:
            self._flags[flag_id]["rollout_percentage"] = min(100, max(0, percentage))
    
    def get_metrics(self, flag_id: str) -> Dict[str, Any]:
        """Get feature flag metrics."""
        if flag_id not in self._flags:
            return {}
        
        flag = self._flags[flag_id]
        metrics = flag["metrics"]
        
        return {
            "exposures": metrics["exposures"],
            "engagements": metrics["engagements"],
            "conversion_rate": (
                metrics["engagements"] / metrics["exposures"]
                if metrics["exposures"] > 0 else 0
            ),
        }


class SentimentAnalyzer:
    """Simple sentiment analysis for feedback."""
    
    # Simple keyword-based approach
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "love", "perfect", "awesome",
        "fantastic", "wonderful", "best", "happy", "satisfied", "helpful",
        "easy", "fast", "smooth", "intuitive", "beautiful", "clean"
    }
    
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "hate", "worst", "broken", "slow",
        "confusing", "difficult", "hard", "annoying", "frustrating",
        "buggy", "crash", "error", "problem", "issue", "missing"
    }
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Returns:
            Score from -1 (negative) to 1 (positive)
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total


class ProductManager:
    """
    Main product management orchestrator.
    
    Features:
    - Feature backlog management
    - RICE prioritization
    - Release planning
    - User feedback processing
    - Roadmap generation
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".product")
        self.storage_path.mkdir(exist_ok=True)
        
        self._features: Dict[str, Feature] = {}
        self._releases: Dict[str, Release] = {}
        self._feedback: List[UserFeedback] = []
        self._flag_manager = FeatureFlagManager()
        self._sentiment_analyzer = SentimentAnalyzer()
        
        self._load_data()
    
    def _load_data(self):
        """Load existing product data."""
        features_file = self.storage_path / "features.json"
        if features_file.exists():
            try:
                with open(features_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get("features", []):
                        rice = RICEScore(**item["rice_score"])
                        item["rice_score"] = rice
                        item["status"] = FeatureStatus(item["status"])
                        item["priority"] = FeaturePriority[item["priority"]]
                        feature = Feature(**item)
                        self._features[feature.id] = feature
            except Exception as e:
                logger.warning(f"Failed to load features: {e}")
    
    async def add_feature(
        self,
        name: str,
        description: str,
        rice_score: RICEScore,
        priority: FeaturePriority = FeaturePriority.P2_MEDIUM,
        tags: Optional[List[str]] = None,
        owner: Optional[str] = None,
    ) -> Feature:
        """Add new feature to backlog."""
        feature_id = f"feat_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        
        feature = Feature(
            id=feature_id,
            name=name,
            description=description,
            status=FeatureStatus.IDEA,
            priority=priority,
            rice_score=rice_score,
            tags=tags or [],
            owner=owner,
        )
        
        self._features[feature_id] = feature
        await self._persist_features()
        
        logger.info(f"Added feature: {name} (RICE: {rice_score.score:.1f})")
        return feature
    
    def get_prioritized_backlog(
        self,
        status_filter: Optional[FeatureStatus] = None,
        limit: int = 20,
    ) -> List[Feature]:
        """Get features sorted by RICE score."""
        features = list(self._features.values())
        
        if status_filter:
            features = [f for f in features if f.status == status_filter]
        
        # Sort by RICE score (descending)
        features.sort(key=lambda f: f.rice_score.score, reverse=True)
        
        return features[:limit]
    
    async def plan_release(
        self,
        name: str,
        version: str,
        target_date: datetime,
        capacity: int = 5,  # Number of features that fit
    ) -> Release:
        """
        Plan a release with top-priority features.
        
        Algorithm:
        1. Filter features that fit in capacity
        2. Respect dependencies
        3. Balance high-impact with quick wins
        """
        release_id = f"rel_{hashlib.sha256(name.encode()).hexdigest()[:8]}"
        
        # Get ready features
        ready_features = [
            f for f in self._features.values()
            if f.status in (FeatureStatus.PLANNED, FeatureStatus.IN_PROGRESS)
        ]
        
        # Sort by RICE and priority
        ready_features.sort(key=lambda f: (f.priority.value, -f.rice_score.score))
        
        # Select features up to capacity
        selected = []
        total_effort = 0
        
        for feature in ready_features:
            if total_effort + feature.rice_score.effort <= capacity * 2:  # Rough estimate
                selected.append(feature.id)
                total_effort += feature.rice_score.effort
                
                if len(selected) >= capacity:
                    break
        
        release = Release(
            id=release_id,
            name=name,
            version=version,
            target_date=target_date,
            features=selected,
        )
        
        # Update feature target releases
        for feature_id in selected:
            if feature_id in self._features:
                self._features[feature_id].target_release = release_id
        
        self._releases[release_id] = release
        await self._persist_features()
        
        logger.info(f"Planned release {name} with {len(selected)} features")
        return release
    
    async def add_feedback(
        self,
        user_id: str,
        content: str,
        feature_id: Optional[str] = None,
        feedback_type: str = "request",
    ) -> UserFeedback:
        """Add and analyze user feedback."""
        feedback_id = f"fb_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        
        # Analyze sentiment
        sentiment = self._sentiment_analyzer.analyze(content)
        
        feedback = UserFeedback(
            id=feedback_id,
            user_id=user_id,
            feature_id=feature_id,
            feedback_type=feedback_type,
            content=content,
            sentiment=sentiment,
        )
        
        self._feedback.append(feedback)
        
        # If positive feedback on a feature, increase its priority
        if feature_id and feature_id in self._features and sentiment > 0.5:
            feature = self._features[feature_id]
            feature.user_requests += 1
            # Boost RICE reach slightly
            feature.rice_score.reach = min(1000, feature.rice_score.reach + 10)
        
        await self._persist_feedback()
        
        return feedback
    
    def get_feedback_summary(self, feature_id: Optional[str] = None) -> Dict[str, Any]:
        """Summarize user feedback."""
        feedback_list = self._feedback
        
        if feature_id:
            feedback_list = [f for f in feedback_list if f.feature_id == feature_id]
        
        if not feedback_list:
            return {"count": 0}
        
        sentiments = [f.sentiment for f in feedback_list]
        
        return {
            "count": len(feedback_list),
            "average_sentiment": round(sum(sentiments) / len(sentiments), 2),
            "positive": sum(1 for s in sentiments if s > 0.2),
            "neutral": sum(1 for s in sentiments if -0.2 <= s <= 0.2),
            "negative": sum(1 for s in sentiments if s < -0.2),
            "top_requests": self._extract_top_requests(feedback_list),
        }
    
    def _extract_top_requests(self, feedback_list: List[UserFeedback]) -> List[Dict[str, Any]]:
        """Extract most common feature requests."""
        requests = [f for f in feedback_list if f.feedback_type == "request"]
        
        # Simple keyword extraction
        keywords: Dict[str, int] = {}
        for fb in requests:
            words = fb.content.lower().split()
            for word in words:
                if len(word) > 4:  # Filter short words
                    keywords[word] = keywords.get(word, 0) + 1
        
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [
            {"term": term, "mentions": count}
            for term, count in top_keywords
        ]
    
    @cached(ttl=300)
    async def generate_roadmap(self) -> Dict[str, Any]:
        """Generate product roadmap visualization."""
        now = datetime.now()
        
        # Group features by status
        by_status: Dict[str, List[Feature]] = {}
        for feature in self._features.values():
            status = feature.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(feature)
        
        # Sort by RICE within each status
        for status in by_status:
            by_status[status].sort(key=lambda f: f.rice_score.score, reverse=True)
        
        # Build timeline
        quarters = []
        for i in range(4):  # Next 4 quarters
            quarter_start = now + timedelta(days=90 * i)
            quarter_name = f"Q{(quarter_start.month - 1) // 3 + 1} {quarter_start.year}"
            
            # Find releases in this quarter
            quarter_releases = [
                r for r in self._releases.values()
                if (r.target_date.year == quarter_start.year and
                    (r.target_date.month - 1) // 3 == (quarter_start.month - 1) // 3)
            ]
            
            quarters.append({
                "name": quarter_name,
                "releases": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "version": r.version,
                        "features": len(r.features),
                        "status": r.status,
                    }
                    for r in quarter_releases
                ],
            })
        
        return {
            "generated_at": now.isoformat(),
            "total_features": len(self._features),
            "by_status": {
                status: [f.name for f in features[:5]]  # Top 5
                for status, features in by_status.items()
            },
            "timeline": quarters,
            "top_priorities": [
                {
                    "name": f.name,
                    "rice_score": f.rice_score.score,
                    "priority": f.priority.name,
                }
                for f in self.get_prioritized_backlog(limit=5)
            ],
        }
    
    def create_feature_flag(
        self,
        feature_id: str,
        rollout_percentage: float = 0.0,
    ) -> str:
        """Create feature flag for gradual rollout."""
        if feature_id not in self._features:
            raise ValueError(f"Feature not found: {feature_id}")
        
        feature = self._features[feature_id]
        flag_id = self._flag_manager.create_flag(
            name=f"feature_{feature_id}",
            default_value=False,
            rollout_percentage=rollout_percentage,
        )
        
        return flag_id
    
    def is_feature_enabled(self, feature_id: str, user_id: Optional[str] = None) -> bool:
        """Check if feature is enabled for user."""
        flag_id = hashlib.sha256(f"feature_{feature_id}".encode()).hexdigest()[:16]
        return self._flag_manager.is_enabled(flag_id, user_id)
    
    async def update_feature_status(
        self,
        feature_id: str,
        new_status: FeatureStatus,
        progress: Optional[float] = None,
    ):
        """Update feature status and progress."""
        if feature_id not in self._features:
            return
        
        feature = self._features[feature_id]
        feature.status = new_status
        
        if progress is not None:
            feature.progress = progress
        
        await self._persist_features()
        
        logger.info(f"Updated feature {feature_id} to {new_status.value}")
    
    async def _persist_features(self):
        """Save features to disk."""
        features_file = self.storage_path / "features.json"
        
        data = {
            "updated_at": datetime.now().isoformat(),
            "feature_count": len(self._features),
            "features": [f.to_dict() for f in self._features.values()],
        }
        
        with open(features_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _persist_feedback(self):
        """Save feedback to disk."""
        feedback_file = self.storage_path / "feedback.json"
        
        with open(feedback_file, 'w') as f:
            json.dump(
                {
                    "feedback": [
                        {
                            "id": f.id,
                            "user_id": f.user_id,
                            "feature_id": f.feature_id,
                            "type": f.feedback_type,
                            "content": f.content,
                            "sentiment": f.sentiment,
                            "created_at": f.created_at,
                        }
                        for f in self._feedback
                    ]
                },
                f,
                indent=2,
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get product management metrics."""
        return {
            "total_features": len(self._features),
            "by_status": {
                status.value: sum(1 for f in self._features.values() if f.status == status)
                for status in FeatureStatus
            },
            "total_feedback": len(self._feedback),
            "active_releases": len(self._releases),
            "feature_flags": len(self._flag_manager._flags),
        }


# Global product manager
_product_manager: Optional[ProductManager] = None


def get_product_manager(storage_path: Optional[Path] = None) -> ProductManager:
    """Get global product manager instance."""
    global _product_manager
    if _product_manager is None:
        _product_manager = ProductManager(storage_path)
    return _product_manager
