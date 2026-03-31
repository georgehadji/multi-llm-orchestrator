"""
MultiTenantGateway — Multi-tenant API gateway
============================================
Module for providing a multi-tenant API gateway with JWT/API-key authentication
for SaaS deployment.

Pattern: Facade
Async: Yes — for I/O-bound operations
Layer: L4 Supervisor

Usage:
    from orchestrator.multi_tenant_gateway import MultiTenantGateway
    gateway = MultiTenantGateway(jwt_secret="secret_key")
    await gateway.start_server()
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import jwt
from aiohttp import web

logger = logging.getLogger("orchestrator.multi_tenant_gateway")


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system."""

    id: str
    name: str
    api_keys: list[str]
    jwt_secret: str
    quota_monthly: int  # Monthly request quota
    quota_reset_day: int  # Day of month when quota resets
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    metadata: dict[str, Any] | None = None


@dataclass
class QuotaUsage:
    """Represents quota usage for a tenant."""

    tenant_id: str
    month: str  # Format: YYYY-MM
    requests_count: int
    tokens_count: int
    last_reset: datetime
    remaining_quota: int


class MultiTenantGateway:
    """Multi-tenant API gateway with JWT/API-key authentication."""

    def __init__(
        self,
        jwt_secret: str = None,
        default_quota: int = 10000,
        port: int = 8000,
        host: str = "localhost",
    ):
        """
        Initialize the multi-tenant gateway.

        Args:
            jwt_secret: Secret key for JWT signing (will generate if not provided)
            default_quota: Default monthly quota for tenants
            port: Port to run the server on
            host: Host to bind to
        """
        self.jwt_secret = jwt_secret or os.urandom(32).hex()
        self.default_quota = default_quota
        self.port = port
        self.host = host

        # In-memory storage (in production, use a database)
        self.tenants: dict[str, Tenant] = {}
        self.quota_usage: dict[str, QuotaUsage] = {}  # Key: f"{tenant_id}:{month}"

        # API key to tenant mapping
        self.api_key_to_tenant: dict[str, str] = {}

        # Initialize web app
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # Register routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""
        # Public routes
        self.app.router.add_get("/", self.health_check)
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_post("/register", self.register_tenant)
        self.app.router.add_post("/login", self.login)

        # Protected routes
        self.app.router.add_post("/execute", self.execute_task)
        self.app.router.add_get("/usage", self.get_usage)
        self.app.router.add_get("/models", self.list_models)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {"status": "healthy", "timestamp": datetime.now().isoformat(), "multi_tenant": True}
        )

    async def register_tenant(self, request: web.Request) -> web.Response:
        """Register a new tenant."""
        try:
            data = await request.json()

            if "name" not in data:
                return web.json_response({"error": "Tenant name is required"}, status=400)

            tenant_name = data["name"]

            # Generate tenant ID
            tenant_id = hashlib.sha256(f"{tenant_name}_{time.time()}".encode()).hexdigest()[:16]

            # Generate API key
            api_key = f"orch_{tenant_id}_{hashlib.sha256(os.urandom(32)).hexdigest()[:32]}"

            # Generate JWT secret for this tenant
            jwt_secret = os.urandom(32).hex()

            # Create tenant
            tenant = Tenant(
                id=tenant_id,
                name=tenant_name,
                api_keys=[api_key],
                jwt_secret=jwt_secret,
                quota_monthly=self.default_quota,
                quota_reset_day=1,  # Reset on the 1st of each month
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata=data.get("metadata", {}),
            )

            self.tenants[tenant_id] = tenant
            self.api_key_to_tenant[api_key] = tenant_id

            # Initialize quota usage
            current_month = datetime.now().strftime("%Y-%m")
            self.quota_usage[f"{tenant_id}:{current_month}"] = QuotaUsage(
                tenant_id=tenant_id,
                month=current_month,
                requests_count=0,
                tokens_count=0,
                last_reset=datetime.now(),
                remaining_quota=self.default_quota,
            )

            logger.info(f"Registered new tenant: {tenant_name} (ID: {tenant_id})")

            return web.json_response(
                {
                    "tenant_id": tenant_id,
                    "api_key": api_key,
                    "jwt_secret_hint": "Store this securely. It's needed for client-side JWT generation.",
                    "message": "Tenant registered successfully",
                }
            )

        except Exception as e:
            logger.error(f"Error registering tenant: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def login(self, request: web.Request) -> web.Response:
        """Login endpoint to get JWT token."""
        try:
            data = await request.json()

            if "api_key" not in data:
                return web.json_response({"error": "API key is required"}, status=400)

            api_key = data["api_key"]

            # Find tenant by API key
            tenant_id = self.api_key_to_tenant.get(api_key)
            if not tenant_id:
                return web.json_response({"error": "Invalid API key"}, status=401)

            tenant = self.tenants.get(tenant_id)
            if not tenant or not tenant.is_active:
                return web.json_response({"error": "Tenant not active"}, status=401)

            # Generate JWT token
            payload = {
                "tenant_id": tenant_id,
                "exp": datetime.utcnow() + timedelta(hours=24),  # 24-hour token
                "iat": datetime.utcnow(),
            }

            token = jwt.encode(payload, tenant.jwt_secret, algorithm="HS256")

            return web.json_response(
                {
                    "token": token,
                    "tenant_id": tenant_id,
                    "expires_in": 24 * 60 * 60,  # 24 hours in seconds
                }
            )

        except Exception as e:
            logger.error(f"Error during login: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def _authenticate_request(self, request: web.Request) -> Tenant | None:
        """Authenticate a request using JWT or API key."""
        # Check for JWT token in Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix

            # Find tenant by decoding token with known secrets
            for tenant in self.tenants.values():
                try:
                    payload = jwt.decode(token, tenant.jwt_secret, algorithms=["HS256"])
                    tenant_id = payload.get("tenant_id")

                    if tenant_id == tenant.id:
                        return tenant
                except jwt.ExpiredSignatureError:
                    continue  # Try next tenant
                except jwt.InvalidTokenError:
                    continue  # Try next tenant

            return None

        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            tenant_id = self.api_key_to_tenant.get(api_key)
            if tenant_id:
                tenant = self.tenants.get(tenant_id)
                if tenant and tenant.is_active:
                    return tenant

        return None

    async def _check_quota(self, tenant: Tenant) -> bool:
        """Check if tenant has quota available."""
        current_month = datetime.now().strftime("%Y-%m")
        quota_key = f"{tenant.id}:{current_month}"

        if quota_key not in self.quota_usage:
            # Initialize quota for this month
            self.quota_usage[quota_key] = QuotaUsage(
                tenant_id=tenant.id,
                month=current_month,
                requests_count=0,
                tokens_count=0,
                last_reset=datetime.now(),
                remaining_quota=tenant.quota_monthly,
            )

        quota = self.quota_usage[quota_key]
        return quota.remaining_quota > 0

    async def _update_quota(self, tenant: Tenant, tokens_used: int = 0):
        """Update quota usage for a tenant."""
        current_month = datetime.now().strftime("%Y-%m")
        quota_key = f"{tenant.id}:{current_month}"

        if quota_key not in self.quota_usage:
            # Initialize if not exists
            self.quota_usage[quota_key] = QuotaUsage(
                tenant_id=tenant.id,
                month=current_month,
                requests_count=0,
                tokens_count=0,
                last_reset=datetime.now(),
                remaining_quota=tenant.quota_monthly,
            )

        quota = self.quota_usage[quota_key]
        quota.requests_count += 1
        quota.tokens_count += tokens_used
        quota.remaining_quota = max(0, quota.remaining_quota - 1)
        quota.updated_at = datetime.now()

    async def execute_task(self, request: web.Request) -> web.Response:
        """Execute a task for an authenticated tenant."""
        # Authenticate request
        tenant = await self._authenticate_request(request)
        if not tenant:
            return web.json_response({"error": "Authentication required"}, status=401)

        # Check quota
        if not await self._check_quota(tenant):
            return web.json_response({"error": "Quota exceeded"}, status=429)

        try:
            data = await request.json()

            # Validate required fields
            if "task" not in data:
                return web.json_response({"error": "Task definition required"}, status=400)

            # Extract task parameters
            task_description = data["task"]
            data.get("criteria", "")
            data.get("budget", 1.0)
            data.get("model", None)

            # In a real implementation, we would call the orchestrator here
            # For now, we'll simulate the execution
            import uuid

            task_id = str(uuid.uuid4())

            # Simulate task execution
            result = {
                "task_id": task_id,
                "status": "completed",
                "result": f"Simulated execution of: {task_description}",
                "cost": 0.05,
                "tokens_used": 150,
                "execution_time": 2.5,
            }

            # Update quota
            await self._update_quota(tenant, tokens_used=result["tokens_used"])

            return web.json_response(result)

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def get_usage(self, request: web.Request) -> web.Response:
        """Get usage information for the authenticated tenant."""
        tenant = await self._authenticate_request(request)
        if not tenant:
            return web.json_response({"error": "Authentication required"}, status=401)

        current_month = datetime.now().strftime("%Y-%m")
        quota_key = f"{tenant.id}:{current_month}"

        if quota_key in self.quota_usage:
            quota = self.quota_usage[quota_key]
            usage_info = {
                "tenant_id": tenant.id,
                "tenant_name": tenant.name,
                "month": quota.month,
                "requests_made": quota.requests_count,
                "tokens_used": quota.tokens_count,
                "quota_remaining": quota.remaining_quota,
                "quota_total": tenant.quota_monthly,
                "quota_percentage_used": (tenant.quota_monthly - quota.remaining_quota)
                / tenant.quota_monthly
                * 100,
            }
        else:
            usage_info = {
                "tenant_id": tenant.id,
                "tenant_name": tenant.name,
                "month": current_month,
                "requests_made": 0,
                "tokens_used": 0,
                "quota_remaining": tenant.quota_monthly,
                "quota_total": tenant.quota_monthly,
                "quota_percentage_used": 0,
            }

        return web.json_response(usage_info)

    async def list_models(self, request: web.Request) -> web.Response:
        """List available models."""
        # Authenticate request
        tenant = await self._authenticate_request(request)
        if not tenant:
            return web.json_response({"error": "Authentication required"}, status=401)

        # Check quota
        if not await self._check_quota(tenant):
            return web.json_response({"error": "Quota exceeded"}, status=429)

        # Update quota for this request
        await self._update_quota(tenant)

        # In a real implementation, we would get the actual list of models
        # For now, we'll return a simulated list
        models = [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "capabilities": ["text", "code"],
                "cost_per_mil_tokens": 30.0,
            },
            {
                "id": "claude-3-5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "capabilities": ["text", "code"],
                "cost_per_mil_tokens": 15.0,
            },
            {
                "id": "deepseek-chat",
                "name": "DeepSeek Chat",
                "capabilities": ["text", "code"],
                "cost_per_mil_tokens": 2.0,
            },
            {
                "id": "deepseek-reasoner",
                "name": "DeepSeek Reasoner",
                "capabilities": ["reasoning", "math"],
                "cost_per_mil_tokens": 12.0,
            },
            {
                "id": "gemini-pro",
                "name": "Gemini Pro",
                "capabilities": ["text", "multimodal"],
                "cost_per_mil_tokens": 15.0,
            },
        ]

        return web.json_response(models)

    async def start_server(self):
        """Start the multi-tenant API gateway server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        logger.info(f"Multi-tenant API Gateway started at http://{self.host}:{self.port}")

    async def stop_server(self):
        """Stop the multi-tenant API gateway server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        logger.info("Multi-tenant API Gateway stopped")

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.site is not None and self.runner is not None

    def get_tenants_count(self) -> int:
        """Get the number of registered tenants."""
        return len(self.tenants)

    def get_usage_stats(self) -> dict[str, Any]:
        """Get overall usage statistics."""
        total_requests = sum(quota.requests_count for quota in self.quota_usage.values())
        total_tokens = sum(quota.tokens_count for quota in self.quota_usage.values())

        active_tenants = sum(1 for tenant in self.tenants.values() if tenant.is_active)

        return {
            "total_tenants": len(self.tenants),
            "active_tenants": active_tenants,
            "total_requests_served": total_requests,
            "total_tokens_processed": total_tokens,
            "quota_usage_records": len(self.quota_usage),
        }

    async def reset_quota_for_tenant(self, tenant_id: str):
        """Manually reset quota for a specific tenant."""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        current_month = datetime.now().strftime("%Y-%m")
        quota_key = f"{tenant_id}:{current_month}"

        tenant = self.tenants[tenant_id]
        self.quota_usage[quota_key] = QuotaUsage(
            tenant_id=tenant_id,
            month=current_month,
            requests_count=0,
            tokens_count=0,
            last_reset=datetime.now(),
            remaining_quota=tenant.quota_monthly,
        )

        logger.info(f"Reset quota for tenant {tenant_id}")


# Global multi-tenant gateway instance for convenience
_global_gateway: MultiTenantGateway | None = None


async def get_global_gateway(
    jwt_secret: str = None, default_quota: int = 10000, port: int = 8000, host: str = "localhost"
) -> MultiTenantGateway:
    """
    Get the global multi-tenant gateway instance, creating it if needed.

    Args:
        jwt_secret: Secret key for JWT signing
        default_quota: Default monthly quota for tenants
        port: Port to run the server on
        host: Host to bind to

    Returns:
        MultiTenantGateway instance
    """
    global _global_gateway
    if _global_gateway is None:
        _global_gateway = MultiTenantGateway(
            jwt_secret=jwt_secret, default_quota=default_quota, port=port, host=host
        )
    return _global_gateway


async def start_global_gateway(
    jwt_secret: str = None, default_quota: int = 10000, port: int = 8000, host: str = "localhost"
):
    """
    Start the global multi-tenant gateway.

    Args:
        jwt_secret: Secret key for JWT signing
        default_quota: Default monthly quota for tenants
        port: Port to run the server on
        host: Host to bind to
    """
    gateway = await get_global_gateway(jwt_secret, default_quota, port, host)
    await gateway.start_server()


async def stop_global_gateway():
    """Stop the global multi-tenant gateway."""
    global _global_gateway
    if _global_gateway:
        await _global_gateway.stop_server()
        _global_gateway = None
