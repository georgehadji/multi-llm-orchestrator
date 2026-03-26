"""
SaaS-Ready Monetization Layer (Multi-Tenant Support)
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Multi-tenant support with usage tracking and billing

Current State: Single-user CLI tool
Future State: Multi-tenant SaaS with plans and quotas

Benefits:
- Ready for paying customers
- Multiple revenue tiers
- Usage-based scaling

Usage:
    from orchestrator.tenancy import TenantManager, Plan
    
    manager = TenantManager()
    tenant = await manager.create_tenant("acme-corp", "pro")
    
    # Check quota before operation
    if await manager.check_quota(tenant.id, "run_project"):
        await manager.record_usage(tenant.id, "run_project", cost=0.50)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .log_config import get_logger

logger = get_logger(__name__)


class PlanTier(str, Enum):
    """Subscription plan tiers."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class Plan:
    """
    Subscription plan definition.
    
    Attributes:
        name: Plan name
        max_projects_per_month: Max projects (-1 for unlimited)
        max_budget_per_project: Max budget per project in USD (-1 for unlimited)
        max_concurrent_tasks: Max concurrent tasks
        allowed_models: List of allowed model names
        features: Set of feature flags
        price_monthly: Monthly price in USD
    """
    name: PlanTier
    max_projects_per_month: int
    max_budget_per_project: float
    max_concurrent_tasks: int
    allowed_models: List[str]
    features: Set[str] = field(default_factory=set)
    price_monthly: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name.value,
            "max_projects_per_month": self.max_projects_per_month,
            "max_budget_per_project": self.max_budget_per_project,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "allowed_models": self.allowed_models,
            "features": list(self.features),
            "price_monthly": self.price_monthly,
        }


# Predefined plans
PLANS = {
    PlanTier.FREE: Plan(
        name=PlanTier.FREE,
        max_projects_per_month=5,
        max_budget_per_project=1.0,
        max_concurrent_tasks=2,
        allowed_models=["deepseek-v3.2", "gemini-2.0-flash"],
        features=set(),
        price_monthly=0.0,
    ),
    PlanTier.STARTER: Plan(
        name=PlanTier.STARTER,
        max_projects_per_month=20,
        max_budget_per_project=5.0,
        max_concurrent_tasks=4,
        allowed_models=["deepseek-v3.2", "gemini-2.0-flash", "claude-3-haiku"],
        features={"basic_support"},
        price_monthly=29.0,
    ),
    PlanTier.PRO: Plan(
        name=PlanTier.PRO,
        max_projects_per_month=50,
        max_budget_per_project=10.0,
        max_concurrent_tasks=8,
        allowed_models=["all"],  # All models
        features={"basic_support", "priority_routing", "benchmark_access"},
        price_monthly=99.0,
    ),
    PlanTier.ENTERPRISE: Plan(
        name=PlanTier.ENTERPRISE,
        max_projects_per_month=-1,  # Unlimited
        max_budget_per_project=-1,  # Unlimited
        max_concurrent_tasks=32,
        allowed_models=["all"],
        features={
            "basic_support",
            "priority_routing",
            "benchmark_access",
            "plugin_support",
            "sso",
            "dedicated_support",
            "custom_models",
        },
        price_monthly=499.0,
    ),
}


@dataclass
class UsageTracker:
    """
    Tracks tenant usage.
    
    Attributes:
        projects_this_month: Projects run this month
        budget_spent_this_month: Budget spent this month
        api_calls_this_month: API calls made
        storage_used_mb: Storage used in MB
    """
    projects_this_month: int = 0
    budget_spent_this_month: float = 0.0
    api_calls_this_month: int = 0
    storage_used_mb: float = 0.0
    
    def reset_monthly(self) -> None:
        """Reset monthly counters."""
        self.projects_this_month = 0
        self.budget_spent_this_month = 0.0
        self.api_calls_this_month = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "projects_this_month": self.projects_this_month,
            "budget_spent_this_month": self.budget_spent_this_month,
            "api_calls_this_month": self.api_calls_this_month,
            "storage_used_mb": self.storage_used_mb,
        }


@dataclass
class Tenant:
    """
    Multi-tenant customer.
    
    Attributes:
        id: Unique tenant ID
        name: Tenant name
        plan: Subscription plan
        usage: Usage tracker
        api_key: API key for authentication
        created_at: Creation timestamp
        expires_at: Subscription expiry
        active: Whether tenant is active
    """
    id: str
    name: str
    plan: Plan
    usage: UsageTracker = field(default_factory=UsageTracker)
    api_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "plan": self.plan.to_dict(),
            "usage": self.usage.to_dict(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "active": self.active,
        }


class TenantManager:
    """
    Manages multi-tenant support with usage tracking and billing.
    
    Features:
    - Create/manage tenants
    - Plan management
    - Usage tracking
    - Quota enforcement
    - API key authentication
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize tenant manager.
        
        Args:
            storage_path: Path to store tenant data (default: .orchestrator/tenants)
        """
        from pathlib import Path
        
        self.storage_path = Path(storage_path or ".orchestrator/tenants")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.tenants: Dict[str, Tenant] = {}
        self.api_keys: Dict[str, str] = {}  # api_key → tenant_id
        
        # Load existing tenants
        self._load_tenants()
        
        logger.info(f"Tenant manager initialized ({len(self.tenants)} tenants)")

    def _load_tenants(self) -> None:
        """Load tenants from disk."""
        import json
        
        tenants_file = self.storage_path / "tenants.json"
        
        if tenants_file.exists():
            try:
                with tenants_file.open("r") as f:
                    data = json.load(f)
                    
                    for tenant_data in data.get("tenants", []):
                        plan_tier = PlanTier(tenant_data["plan"]["name"])
                        plan = PLANS[plan_tier]
                        
                        tenant = Tenant(
                            id=tenant_data["id"],
                            name=tenant_data["name"],
                            plan=plan,
                            usage=UsageTracker(
                                **tenant_data.get("usage", {})
                            ),
                            api_key=tenant_data.get("api_key", ""),
                            created_at=datetime.fromisoformat(tenant_data["created_at"]),
                            expires_at=(
                                datetime.fromisoformat(tenant_data["expires_at"])
                                if tenant_data.get("expires_at")
                                else None
                            ),
                            active=tenant_data.get("active", True),
                        )
                        
                        self.tenants[tenant.id] = tenant
                        self.api_keys[tenant.api_key] = tenant.id
                        
                logger.info(f"Loaded {len(self.tenants)} tenants from disk")
                
            except Exception as e:
                logger.error(f"Failed to load tenants: {e}")

    def _save_tenants(self) -> None:
        """Save tenants to disk."""
        import json
        
        tenants_file = self.storage_path / "tenants.json"
        
        try:
            with tenants_file.open("w") as f:
                json.dump({
                    "tenants": [t.to_dict() for t in self.tenants.values()],
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save tenants: {e}")

    async def create_tenant(
        self,
        name: str,
        plan_name: str = "free",
    ) -> Tenant:
        """
        Create new tenant.
        
        Args:
            name: Tenant name
            plan_name: Plan tier name
            
        Returns:
            Created tenant
        """
        tenant_id = str(uuid.uuid4())
        api_key = self._generate_api_key()
        
        plan_tier = PlanTier(plan_name.lower())
        plan = PLANS[plan_tier]
        
        # Set expiry based on plan
        if plan_tier != PlanTier.ENTERPRISE:
            expires_at = datetime.now() + timedelta(days=30)
        else:
            expires_at = None  # No expiry for enterprise
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            plan=plan,
            api_key=api_key,
            expires_at=expires_at,
        )
        
        self.tenants[tenant_id] = tenant
        self.api_keys[api_key] = tenant_id
        
        self._save_tenants()
        
        logger.info(f"Created tenant: {name} ({plan_name})")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Get tenant by ID.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Tenant or None
        """
        return self.tenants.get(tenant_id)

    async def get_tenant_by_api_key(self, api_key: str) -> Optional[Tenant]:
        """
        Get tenant by API key with constant-time comparison.
        
        FIX-PS-001a: Prevent timing attacks on API key authentication.
        
        Args:
            api_key: API key to look up
            
        Returns:
            Tenant or None
        """
        import hmac
        
        # FIX-PS-001a: Constant-time comparison to prevent timing attacks
        # Standard dict lookup leaks timing information about key presence
        # hmac.compare_digest executes in constant time regardless of match
        
        found_tenant = None
        
        # Iterate through all keys to maintain constant execution time
        for stored_key, tenant_id in self.api_keys.items():
            # Constant-time comparison - always executes full comparison
            # regardless of where in the iteration we are
            is_match = hmac.compare_digest(
                stored_key.encode('utf-8'),
                api_key.encode('utf-8')
            )
            
            if is_match:
                found_tenant = self.tenants.get(tenant_id)
                # Don't break - continue iterating to maintain constant time
        
        # Log failed attempt for security monitoring
        if found_tenant is None:
            logger.warning(f"Failed API key authentication attempt from {api_key[:8]}...")
        
        return found_tenant

    async def check_quota(
        self,
        tenant_id: str,
        operation: str,
        cost: float = 0.0,
    ) -> bool:
        """
        Check if tenant is within quota for operation.
        
        Args:
            tenant_id: Tenant ID
            operation: Operation name (run_project, api_call, etc.)
            cost: Expected cost of operation
            
        Returns:
            True if within quota
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        if not tenant.active:
            return False
        
        # Check expiry
        if tenant.expires_at and datetime.now() > tenant.expires_at:
            logger.warning(f"Tenant {tenant_id} subscription expired")
            return False
        
        plan = tenant.plan
        
        # Check projects per month
        if operation == "run_project":
            if plan.max_projects_per_month > 0:
                if tenant.usage.projects_this_month >= plan.max_projects_per_month:
                    logger.warning(f"Tenant {tenant_id} exceeded monthly project limit")
                    return False
            
            # Check budget per project
            if plan.max_budget_per_project > 0:
                if cost > plan.max_budget_per_project:
                    logger.warning(
                        f"Project cost ${cost:.2f} exceeds plan limit ${plan.max_budget_per_project:.2f}"
                    )
                    return False
        
        # Check concurrent tasks (would need separate tracking)
        # For now, skip this check
        
        return True

    async def record_usage(
        self,
        tenant_id: str,
        operation: str,
        cost: float = 0.0,
        api_calls: int = 0,
    ) -> None:
        """
        Record usage for tenant.
        
        Args:
            tenant_id: Tenant ID
            operation: Operation name
            cost: Cost of operation
            api_calls: Number of API calls
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return
        
        if operation == "run_project":
            tenant.usage.projects_this_month += 1
            tenant.usage.budget_spent_this_month += cost
        
        if api_calls > 0:
            tenant.usage.api_calls_this_month += api_calls
        
        self._save_tenants()

    async def update_plan(
        self,
        tenant_id: str,
        plan_name: str,
    ) -> bool:
        """
        Update tenant plan.
        
        Args:
            tenant_id: Tenant ID
            plan_name: New plan name
            
        Returns:
            True if successful
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        plan_tier = PlanTier(plan_name.lower())
        tenant.plan = PLANS[plan_tier]
        
        # Reset expiry for new plan
        if plan_tier != PlanTier.ENTERPRISE:
            tenant.expires_at = datetime.now() + timedelta(days=30)
        else:
            tenant.expires_at = None
        
        self._save_tenants()
        
        logger.info(f"Updated tenant {tenant_id} to {plan_name}")
        return True

    async def reset_usage(self, tenant_id: str) -> None:
        """
        Reset tenant usage (for monthly reset).
        
        Args:
            tenant_id: Tenant ID
        """
        tenant = self.tenants.get(tenant_id)
        if tenant:
            tenant.usage.reset_monthly()
            self._save_tenants()

    def _generate_api_key(self) -> str:
        """Generate secure API key."""
        # Generate random key
        key = secrets.token_urlsafe(32)
        
        # Add tenant prefix for easy identification
        prefix = "orch_"
        
        return prefix + key

    def get_statistics(self) -> Dict[str, Any]:
        """Get tenant statistics."""
        total_tenants = len(self.tenants)
        active_tenants = sum(1 for t in self.tenants.values() if t.active)
        
        by_plan = {}
        for tenant in self.tenants.values():
            plan_name = tenant.plan.name.value
            by_plan[plan_name] = by_plan.get(plan_name, 0) + 1
        
        total_revenue = sum(
            t.plan.price_monthly for t in self.tenants.values()
            if t.active
        )
        
        return {
            "total_tenants": total_tenants,
            "active_tenants": active_tenants,
            "by_plan": by_plan,
            "monthly_recurring_revenue": total_revenue,
        }


__all__ = [
    "TenantManager",
    "Tenant",
    "Plan",
    "PlanTier",
    "UsageTracker",
    "PLANS",
]
