"""
APIServer — REST API server
==========================
Module for providing a REST API server for the orchestrator.

Pattern: Facade
Async: Yes — for I/O-bound operations
Layer: L4 Supervisor

SECURITY FIXES:
- Configurable CORS allowlist (not wildcard)
- Rate limiting on all endpoints
- Secure API key authentication
- Request size limits

Usage:
    from orchestrator.api_server import APIServer
    server = APIServer(port=8000, cors_origins=["https://trusted-domain.com"])
    await server.start()
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import web

logger = logging.getLogger("orchestrator.api_server")


# ─────────────────────────────────────────────
# Rate Limiter (Token Bucket)
# ─────────────────────────────────────────────

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    
    SECURITY: Prevents DoS and brute force attacks.
    """
    
    def __init__(
        self,
        rate: int = 100,  # requests per window
        window_seconds: int = 60,
    ):
        self.rate = rate
        self.window_seconds = window_seconds
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": rate, "last_update": time.time()}
        )
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier (IP, API key, etc.)
        
        Returns:
            True if allowed
        """
        async with self._lock:
            bucket = self._buckets[client_id]
            now = time.time()
            
            # Refill tokens
            elapsed = now - bucket["last_update"]
            tokens_to_add = (elapsed / self.window_seconds) * self.rate
            bucket["tokens"] = min(self.rate, bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = now
            
            # Check if allowed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            else:
                return False
    
    def get_retry_after(self, client_id: str) -> float:
        """Get seconds until next request is allowed."""
        bucket = self._buckets.get(client_id)
        if not bucket:
            return 0.0
        
        if bucket["tokens"] >= 1:
            return 0.0
        
        tokens_needed = 1 - bucket["tokens"]
        return (tokens_needed / self.rate) * self.window_seconds


# ─────────────────────────────────────────────
# API Server
# ─────────────────────────────────────────────

class APIServer:
    """REST API server for the orchestrator."""

    def __init__(
        self,
        port: int = 8000,
        host: str = "localhost",
        cors_origins: Optional[List[str]] = None,
        auth_required: bool = True,
        rate_limit: int = 100,
        rate_window: int = 60,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """
        Initialize the API server.
        
        SECURITY:
        - cors_origins: Configurable allowlist (not wildcard)
        - rate_limit: Rate limiting to prevent DoS
        - max_request_size: Limit request size
        """
        self.port = port
        self.host = host
        self.cors_origins = cors_origins or []  # Empty = same origin only
        self.auth_required = auth_required
        self.max_request_size = max_request_size
        
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # API keys: hashed_key -> {user_id, permissions, created_at}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            rate=rate_limit,
            window_seconds=rate_window,
        )
        
        # Request stats
        self.request_stats: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "start_time": datetime.now()
        }

        # Register routes
        self._setup_routes()

        # Enable CORS with allowlist (SECURITY FIX)
        if self.cors_origins:
            self._enable_cors()

        # Add rate limiting middleware
        self.app.middlewares.append(self._rate_limit_middleware)
        
        # Add request size limit
        self.app.middlewares.append(self._request_size_middleware)
    
    def _setup_routes(self):
        """Setup API routes."""
        self.app.router.add_get("/", self.health_check)
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_post("/execute", self.execute_task)
        self.app.router.add_get("/status/{task_id}", self.get_task_status)
        self.app.router.add_get("/models", self.list_models)
        self.app.router.add_post("/register_key", self.register_api_key)
        self.app.router.add_get("/stats", self.get_stats)

    def _enable_cors(self):
        """
        Enable CORS with configurable allowlist.
        
        SECURITY FIX: No longer uses wildcard '*' which allows any origin.
        Now uses configurable allowlist for trusted domains only.
        """
        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                response = await handler(request)
                
                # Check origin against allowlist
                origin = request.headers.get('Origin', '')
                
                if origin in self.cors_origins:
                    response.headers['Access-Control-Allow-Origin'] = origin
                elif '*' in self.cors_origins:
                    # Only allow wildcard if explicitly configured
                    response.headers['Access-Control-Allow-Origin'] = '*'
                
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
                response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
                
                return response
            return middleware_handler

        self.app.middlewares.append(cors_middleware)
    
    async def _rate_limit_middleware(self, app, handler):
        """
        Rate limiting middleware.
        
        SECURITY: Prevents DoS and brute force attacks.
        """
        async def middleware_handler(request):
            # Get client identifier (IP address or API key)
            client_id = request.headers.get('X-API-Key', request.remote or 'unknown')
            
            # Check rate limit
            if not await self.rate_limiter.is_allowed(client_id):
                self.request_stats["rate_limited_requests"] += 1
                retry_after = self.rate_limiter.get_retry_after(client_id)
                
                return web.json_response(
                    {
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                    },
                    status=429,
                    headers={'Retry-After': str(int(retry_after))}
                )
            
            return await handler(request)
        
        return middleware_handler
    
    async def _request_size_middleware(self, app, handler):
        """
        Request size limit middleware.
        
        SECURITY: Prevents large payload DoS attacks.
        """
        async def middleware_handler(request):
            content_length = request.content_length
            
            if content_length and content_length > self.max_request_size:
                return web.json_response(
                    {
                        "error": f"Request too large. Max size: {self.max_request_size} bytes",
                    },
                    status=413
                )
            
            return await handler(request)
        
        return middleware_handler
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        self._update_request_stats(success=True)
        
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.request_stats["start_time"])
        })
    
    async def execute_task(self, request: web.Request) -> web.Response:
        """Execute a task via the orchestrator."""
        # Update request stats
        self._update_request_stats()
        
        # Authenticate request if required
        if self.auth_required:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "Authorization header required"}, 
                    status=401
                )
            
            api_key = auth_header[7:]  # Remove "Bearer " prefix
            if not self._verify_api_key(api_key):
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "Invalid API key"}, 
                    status=401
                )
        
        try:
            # Get request data
            data = await request.json()
            
            # Validate required fields
            if "task" not in data:
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "Task definition required"}, 
                    status=400
                )
            
            # Extract task parameters
            task_description = data["task"]
            criteria = data.get("criteria", "")
            budget = data.get("budget", 1.0)
            model_preference = data.get("model", None)
            
            # In a real implementation, we would call the orchestrator here
            # For now, we'll simulate the execution
            task_id = hashlib.sha256(f"{task_description}{datetime.now()}".encode()).hexdigest()[:16]
            
            # Simulate task execution
            result = {
                "task_id": task_id,
                "status": "completed",
                "result": f"Simulated execution of: {task_description}",
                "cost": 0.05,
                "tokens_used": 150,
                "execution_time": 2.5
            }
            
            self._update_request_stats(success=True)
            return web.json_response(result)
            
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response(
                {"error": "Invalid JSON in request body"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            self._update_request_stats(success=False)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )
    
    async def get_task_status(self, request: web.Request) -> web.Response:
        """Get the status of a task."""
        self._update_request_stats(success=True)
        
        task_id = request.match_info["task_id"]
        
        # In a real implementation, we would check the actual task status
        # For now, we'll return a simulated status
        status = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "result_available": True,
            "estimated_completion": None
        }
        
        return web.json_response(status)
    
    async def list_models(self, request: web.Request) -> web.Response:
        """List available models."""
        self._update_request_stats(success=True)
        
        # In a real implementation, we would get the actual list of models
        # For now, we'll return a simulated list
        models = [
            {"id": "gpt-4", "name": "GPT-4", "capabilities": ["text", "code"], "cost_per_mil_tokens": 30.0},
            {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "capabilities": ["text", "code"], "cost_per_mil_tokens": 15.0},
            {"id": "deepseek-chat", "name": "DeepSeek Chat", "capabilities": ["text", "code"], "cost_per_mil_tokens": 2.0},
            {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "capabilities": ["reasoning", "math"], "cost_per_mil_tokens": 12.0},
            {"id": "gemini-pro", "name": "Gemini Pro", "capabilities": ["text", "multimodal"], "cost_per_mil_tokens": 15.0}
        ]
        
        return web.json_response(models)
    
    async def register_api_key(self, request: web.Request) -> web.Response:
        """Register a new API key."""
        self._update_request_stats()
        
        try:
            data = await request.json()
            
            if "user_id" not in data:
                self._update_request_stats(success=False)
                return web.json_response(
                    {"error": "user_id required"}, 
                    status=400
                )
            
            user_id = data["user_id"]
            permissions = data.get("permissions", ["read", "execute"])
            
            # Generate a new API key
            raw_key = f"orchestrator_{user_id}_{datetime.now().isoformat()}_{hashlib.sha256(str(hash(str(data))).encode()).hexdigest()[:16]}"
            hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()
            
            # Store the API key
            self.api_keys[hashed_key] = {
                "user_id": user_id,
                "permissions": permissions,
                "created_at": datetime.now().isoformat(),
                "last_used": None
            }
            
            self._update_request_stats(success=True)
            return web.json_response({
                "api_key": raw_key,
                "message": "API key registered successfully"
            })
            
        except json.JSONDecodeError:
            self._update_request_stats(success=False)
            return web.json_response(
                {"error": "Invalid JSON in request body"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Error registering API key: {e}")
            self._update_request_stats(success=False)
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )
    
    async def get_stats(self, request: web.Request) -> web.Response:
        """Get server statistics."""
        self._update_request_stats(success=True)
        
        uptime = datetime.now() - self.request_stats["start_time"]
        
        stats = {
            "total_requests": self.request_stats["total_requests"],
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "success_rate": (
                self.request_stats["successful_requests"] / self.request_stats["total_requests"] 
                if self.request_stats["total_requests"] > 0 
                else 0
            ),
            "uptime": str(uptime),
            "server_time": datetime.now().isoformat(),
            "registered_api_keys": len(self.api_keys)
        }
        
        return web.json_response(stats)
    
    def _verify_api_key(self, api_key: str) -> bool:
        """Verify an API key."""
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        if hashed_key in self.api_keys:
            # Update last used timestamp
            self.api_keys[hashed_key]["last_used"] = datetime.now().isoformat()
            return True
        return False
    
    def _update_request_stats(self, success: bool = True):
        """Update request statistics."""
        self.request_stats["total_requests"] += 1
        if success:
            self.request_stats["successful_requests"] += 1
        else:
            self.request_stats["failed_requests"] += 1
    
    async def start(self):
        """Start the API server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"API Server started at http://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the API server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("API Server stopped")
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self.site is not None and self.runner is not None


# Global server instance for convenience
_global_server: Optional[APIServer] = None


async def get_server_instance(port: int = 8000, host: str = "localhost") -> APIServer:
    """
    Get the global server instance, creating it if it doesn't exist.
    
    Args:
        port: Port to run the server on
        host: Host to bind to
        
    Returns:
        APIServer instance
    """
    global _global_server
    if _global_server is None:
        _global_server = APIServer(port=port, host=host)
    return _global_server


async def start_server(port: int = 8000, host: str = "localhost"):
    """
    Start the API server.
    
    Args:
        port: Port to run the server on
        host: Host to bind to
    """
    server = await get_server_instance(port, host)
    await server.start()


async def stop_server():
    """Stop the API server."""
    global _global_server
    if _global_server:
        await _global_server.stop()
        _global_server = None