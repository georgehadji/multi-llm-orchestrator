"""
Gateway — API gateway
====================
Module for handling API gateway functionality, including routing, authentication,
and request/response transformation.

Pattern: Facade
Async: Yes — for I/O-bound operations
Layer: L3 Agents

Usage:
    from orchestrator.gateway import APIGateway
    gateway = APIGateway()
    await gateway.route_request(request_data, target_service="orchestrator")
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("orchestrator.gateway")


class APIRequest:
    """Represents an incoming API request."""

    def __init__(self, method: str, url: str, headers: dict[str, str],
                 body: str | None = None, query_params: dict[str, str] | None = None):
        self.method = method
        self.url = url
        self.headers = headers
        self.body = body
        self.query_params = query_params or {}
        self.timestamp = datetime.now()
        self.client_ip: str | None = None
        self.api_key: str | None = None
        self.authenticated = False


class APIResponse:
    """Represents an outgoing API response."""

    def __init__(self, status_code: int, headers: dict[str, str],
                 body: str | None = None, error: str | None = None):
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.error = error
        self.timestamp = datetime.now()


class RateLimitInfo:
    """Information about rate limiting for a client."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests_made = 0
        self.window_start = datetime.now()


class APIGateway:
    """Handles API gateway functionality including routing, authentication, and transformations."""

    def __init__(self, rate_limit_default: int = 100, rate_window_seconds: int = 60):
        """Initialize the API gateway."""
        self.rate_limit_default = rate_limit_default
        self.rate_window_seconds = rate_window_seconds
        self.rate_limits: dict[str, RateLimitInfo] = {}  # client_id -> RateLimitInfo
        self.api_keys: dict[str, dict[str, Any]] = {}  # hashed_key -> {user_id, permissions, etc}
        self.routes: dict[str, str] = {}  # path_pattern -> target_service
        self.middlewares: list[callable] = []
        self.request_log: list[tuple[APIRequest, APIResponse]] = []
        self.max_log_entries = 1000

        # Add default routes
        self.add_route("/orchestrator/*", "orchestrator_service")
        self.add_route("/models/*", "model_service")
        self.add_route("/health", "health_service")

    def add_route(self, path_pattern: str, target_service: str):
        """Add a route to the gateway."""
        self.routes[path_pattern] = target_service
        logger.info(f"Added route: {path_pattern} -> {target_service}")

    def add_middleware(self, middleware_func: callable):
        """Add a middleware function to process requests/responses."""
        self.middlewares.append(middleware_func)
        logger.info(f"Added middleware: {middleware_func.__name__}")

    def register_api_key(self, api_key: str, user_id: str, permissions: list[str] = None) -> str:
        """
        Register a new API key.

        Args:
            api_key: The API key to register
            user_id: The user ID associated with the key
            permissions: List of permissions for the key

        Returns:
            str: Hashed version of the API key
        """
        # Hash the API key for security
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()

        self.api_keys[hashed_key] = {
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.now(),
            "last_used": None
        }

        logger.info(f"Registered API key for user: {user_id}")
        return hashed_key

    def verify_api_key(self, api_key: str) -> dict[str, Any] | None:
        """
        Verify an API key and return user info if valid.

        Args:
            api_key: The API key to verify

        Returns:
            Dict with user info if valid, None if invalid
        """
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        user_info = self.api_keys.get(hashed_key)

        if user_info:
            # Update last used timestamp
            user_info["last_used"] = datetime.now()
            return user_info

        return None

    async def authenticate_request(self, request: APIRequest) -> bool:
        """
        Authenticate an incoming request.

        Args:
            request: The incoming API request

        Returns:
            bool: True if authenticated, False otherwise
        """
        # Check for API key in header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        else:
            # Check for API key in custom header
            api_key = request.headers.get("X-API-Key")

        if not api_key:
            logger.warning("No API key provided in request")
            return False

        user_info = self.verify_api_key(api_key)
        if not user_info:
            logger.warning("Invalid API key provided")
            return False

        # Set authentication info on request
        request.api_key = api_key
        request.authenticated = True

        return True

    def check_rate_limit(self, client_id: str) -> tuple[bool, int, int]:
        """
        Check if a client has exceeded their rate limit.

        Args:
            client_id: The client identifier

        Returns:
            Tuple of (allowed, remaining_requests, reset_time_seconds)
        """
        now = datetime.now()

        if client_id not in self.rate_limits:
            # Initialize rate limit for this client
            self.rate_limits[client_id] = RateLimitInfo(
                self.rate_limit_default,
                self.rate_window_seconds
            )

        rate_info = self.rate_limits[client_id]

        # Check if we're in a new window
        elapsed = (now - rate_info.window_start).total_seconds()
        if elapsed > rate_info.window_seconds:
            # Reset the window
            rate_info.requests_made = 0
            rate_info.window_start = now
        elif rate_info.requests_made >= rate_info.limit:
            # Rate limit exceeded
            reset_in = int(rate_info.window_seconds - elapsed)
            return False, 0, reset_in

        # Increment request count
        rate_info.requests_made += 1

        remaining = rate_info.limit - rate_info.requests_made
        reset_in = int(rate_info.window_seconds - elapsed)

        return True, remaining, reset_in

    async def transform_request(self, request: APIRequest, target_service: str) -> APIRequest:
        """
        Transform an incoming request before forwarding to target service.

        Args:
            request: The incoming request
            target_service: The target service

        Returns:
            APIRequest: The transformed request
        """
        # Apply middlewares
        for middleware in self.middlewares:
            request = await middleware(request, "request")

        # Add gateway-specific headers
        request.headers["X-Gateway-Timestamp"] = request.timestamp.isoformat()
        request.headers["X-Gateway-Service"] = target_service

        # Sanitize headers to remove sensitive information
        sanitized_headers = {}
        for key, value in request.headers.items():
            if key.lower() not in ["authorization", "cookie"]:
                sanitized_headers[key] = value
        request.headers = sanitized_headers

        return request

    async def transform_response(self, response: APIResponse, target_service: str) -> APIResponse:
        """
        Transform an outgoing response before sending to client.

        Args:
            response: The outgoing response
            target_service: The target service

        Returns:
            APIResponse: The transformed response
        """
        # Add gateway-specific headers
        response.headers["X-Gateway-Timestamp"] = response.timestamp.isoformat()
        response.headers["X-Gateway-Service"] = target_service

        # Apply middlewares
        for middleware in reversed(self.middlewares):
            response = await middleware(response, "response")

        return response

    async def route_request(self, request_data: dict[str, Any],
                           target_service: str | None = None) -> APIResponse:
        """
        Route an incoming request to the appropriate service.

        Args:
            request_data: Dictionary containing request information
            target_service: Optional target service (if known)

        Returns:
            APIResponse: The response from the target service
        """
        # Create APIRequest object from request data
        request = APIRequest(
            method=request_data.get("method", "GET"),
            url=request_data.get("url", "/"),
            headers=request_data.get("headers", {}),
            body=request_data.get("body"),
            query_params=request_data.get("query_params", {})
        )

        # Set client IP if available
        request.client_ip = request_data.get("client_ip")

        # Authenticate request
        if not await self.authenticate_request(request):
            response = APIResponse(
                status_code=401,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"error": "Unauthorized"}),
                error="Authentication failed"
            )
            self._log_request_response(request, response)
            return response

        # Check rate limit
        client_id = request.api_key or request.client_ip or "anonymous"
        allowed, remaining, reset_time = self.check_rate_limit(client_id)

        if not allowed:
            response = APIResponse(
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(self.rate_limit_default),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time)
                },
                body=json.dumps({"error": "Rate limit exceeded"}),
                error="Rate limit exceeded"
            )
            self._log_request_response(request, response)
            return response

        # Determine target service if not provided
        if not target_service:
            target_service = self._match_route(request.url)

        if not target_service:
            response = APIResponse(
                status_code=404,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"error": "Service not found"}),
                error="No matching route found"
            )
            self._log_request_response(request, response)
            return response

        # Transform request
        request = await self.transform_request(request, target_service)

        # Forward request to target service
        try:
            service_response = await self._forward_request(request, target_service)
        except Exception as e:
            logger.error(f"Error forwarding request to {target_service}: {e}")
            response = APIResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"error": "Internal server error"}),
                error=str(e)
            )
            self._log_request_response(request, response)
            return response

        # Transform response
        response = await self.transform_response(service_response, target_service)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit_default)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)

        self._log_request_response(request, response)
        return response

    def _match_route(self, url: str) -> str | None:
        """Match a URL to a registered route."""
        for pattern, service in self.routes.items():
            # Simple pattern matching (supports wildcard at the end)
            if pattern.endswith("*"):
                prefix = pattern[:-1]  # Remove the '*'
                if url.startswith(prefix):
                    return service
            elif pattern == url:
                return service

        return None

    async def _forward_request(self, request: APIRequest, target_service: str) -> APIResponse:
        """Forward the request to the target service."""
        # In a real implementation, this would make an HTTP call to the actual service
        # For now, we'll simulate different services

        if target_service == "health_service":
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"status": "healthy", "timestamp": datetime.now().isoformat()})
            )
        elif target_service == "orchestrator_service":
            # Simulate orchestrator service response
            # In a real implementation, this would call the actual orchestrator
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps({
                    "message": "Request processed by orchestrator",
                    "service": target_service,
                    "request_id": hashlib.sha256((request.url + str(request.timestamp)).encode()).hexdigest()[:16]
                })
            )
        elif target_service == "model_service":
            # Simulate model service response
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps({
                    "models": ["gpt-4", "claude-3", "gemini-pro"],
                    "service": target_service
                })
            )
        else:
            # Unknown service
            return APIResponse(
                status_code=501,
                headers={"Content-Type": "application/json"},
                body=json.dumps({"error": f"Service {target_service} not implemented"})
            )

    def _log_request_response(self, request: APIRequest, response: APIResponse):
        """Log the request and response."""
        self.request_log.append((request, response))

        # Trim log if it gets too long
        if len(self.request_log) > self.max_log_entries:
            self.request_log = self.request_log[-self.max_log_entries:]

    def get_request_logs(self, limit: int = 100) -> list[tuple[APIRequest, APIResponse]]:
        """Get recent request logs."""
        return self.request_log[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        total_requests = len(self.request_log)
        successful_requests = sum(1 for req, resp in self.request_log if 200 <= resp.status_code < 300)
        error_requests = sum(1 for req, resp in self.request_log if resp.status_code >= 400)

        # Calculate requests per minute
        if self.request_log:
            time_diff = (self.request_log[-1][0].timestamp - self.request_log[0][0].timestamp).total_seconds()
            if time_diff > 0:
                requests_per_minute = (total_requests / time_diff) * 60
            else:
                requests_per_minute = total_requests
        else:
            requests_per_minute = 0

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_requests": error_requests,
            "requests_per_minute": round(requests_per_minute, 2),
            "active_rate_limits": len(self.rate_limits),
            "registered_api_keys": len(self.api_keys),
            "routes_count": len(self.routes)
        }

    async def invalidate_rate_limit(self, client_id: str):
        """Invalidate/reset rate limit for a specific client."""
        if client_id in self.rate_limits:
            del self.rate_limits[client_id]
            logger.info(f"Invalidated rate limit for client: {client_id}")

    async def reload_routes(self):
        """Reload routes from configuration."""
        # In a real implementation, this would load from a config file or database
        logger.info("Routes reloaded")
