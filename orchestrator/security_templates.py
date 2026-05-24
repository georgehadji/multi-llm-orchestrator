"""
Security Templates Generator — Template Method + Functional Composition
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Security template generation using Template Method Pattern for structure
and Functional Composition for pure validation functions.

Paradigm: Hybrid (OOP for templates, Functional for utilities)
Patterns: Template Method, Functional Composition, Factory Method

Usage:
    from orchestrator.security_templates import AuthTemplate, SecurityConfig

    config = SecurityConfig(jwt_expiry=900, bcrypt_cost=12)
    template = AuthTemplate()
    auth_code = template.generate(config)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any
from functools import reduce

# ═══════════════════════════════════════════════════════════════════
# FUNCTIONAL COMPOSITION UTILITIES
# ═══════════════════════════════════════════════════════════════════


def compose(*functions: Callable) -> Callable:
    """
    Compose multiple functions: f(g(h(x))).

    Functional Programming: Pure function composition.

    Args:
        *functions: Functions to compose (right to left)

    Returns:
        Composed function

    Example:
        >>> add_one = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> composed = compose(add_one, double)
        >>> composed(5)  # double(add_one(5)) = 12
        12
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(*functions: Callable) -> Callable:
    """
    Pipe functions left to right: h(g(f(x))).

    Functional Programming: More readable than compose for some cases.

    Args:
        *functions: Functions to pipe (left to right)

    Returns:
        Piped function
    """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def validate_not_empty(value: Any) -> bool:
    """Pure function: Validate value is not empty."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def validate_min_length(min_len: int) -> Callable[[str], bool]:
    """Pure function factory: Validate minimum length."""

    def validator(value: str) -> bool:
        return len(value) >= min_len

    return validator


def validate_max_length(max_len: int) -> Callable[[str], bool]:
    """Pure function factory: Validate maximum length."""

    def validator(value: str) -> bool:
        return len(value) <= max_len

    return validator


def validate_regex(pattern: str) -> Callable[[str], bool]:
    """Pure function factory: Validate against regex pattern."""
    import re

    compiled = re.compile(pattern)

    def validator(value: str) -> bool:
        return bool(compiled.match(value))

    return validator


# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SecurityConfig:
    """
    Immutable security configuration.

    Frozen dataclass ensures immutability (Functional Programming principle).

    Attributes:
        jwt_expiry: JWT token expiry in seconds (default: 15 min)
        refresh_expiry: Refresh token expiry (default: 7 days)
        rate_limit_window: Rate limit window in seconds
        rate_limit_max: Max requests per window
        bcrypt_cost: Bcrypt cost factor (≥ 10 recommended)
        session_timeout: Session timeout in seconds
        password_min_length: Minimum password length
        require_special_chars: Require special characters in password
    """

    jwt_expiry: int = 900  # 15 minutes
    refresh_expiry: int = 604800  # 7 days
    rate_limit_window: int = 60
    rate_limit_max: int = 100
    bcrypt_cost: int = 12
    session_timeout: int = 1800  # 30 minutes
    password_min_length: int = 8
    require_special_chars: bool = True
    cors_origins: tuple = field(default_factory=lambda: ("*",))
    csrf_enabled: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Immutable validation (runs once during init)
        if self.bcrypt_cost < 10:
            object.__setattr__(self, "bcrypt_cost", 10)
        if self.jwt_expiry < 300:
            object.__setattr__(self, "jwt_expiry", 300)
        if self.password_min_length < 6:
            object.__setattr__(self, "password_min_length", 6)


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE METHOD PATTERN — BASE CLASS
# ═══════════════════════════════════════════════════════════════════


class SecurityTemplate(ABC):
    """
    Template Method Pattern for security templates.

    Defines the skeleton of security template generation, allowing
    subclasses to customize specific steps without changing the algorithm.

    Usage:
        class AuthTemplate(SecurityTemplate):
            def _define_template(self) -> str:
                # Custom implementation
                pass

            def _validate(self, config: SecurityConfig) -> bool:
                # Custom validation
                pass

        template = AuthTemplate()
        result = template.generate(config)
    """

    def __init__(self):
        self._hooks: Dict[str, Callable] = {}

    def register_hook(self, name: str, callback: Callable) -> None:
        """
        Register a hook for customization.

        Args:
            name: Hook name
            callback: Callback function
        """
        self._hooks[name] = callback

    def _trigger_hook(self, name: str, *args, **kwargs) -> Any:
        """Trigger a registered hook."""
        if name in self._hooks:
            return self._hooks[name](*args, **kwargs)
        return None

    @abstractmethod
    def _define_template(self, config: SecurityConfig) -> str:
        """
        Hook: Define the template structure.

        Subclasses must implement this to define their specific template.

        Args:
            config: Security configuration

        Returns:
            Template string
        """
        pass

    @abstractmethod
    def _validate(self, config: SecurityConfig) -> bool:
        """
        Hook: Validate configuration (pure function).

        Subclasses implement validation logic.

        Args:
            config: Security configuration

        Returns:
            True if valid, False otherwise
        """
        pass

    def generate(self, config: SecurityConfig) -> str:
        """
        Template Method: Final algorithm for template generation.

        This is the final algorithm that cannot be changed by subclasses.

        Steps:
        1. Validate configuration
        2. Trigger pre-generation hooks
        3. Generate template
        4. Trigger post-generation hooks
        5. Return final template

        Args:
            config: Security configuration

        Returns:
            Generated security template

        Raises:
            ValueError: If configuration is invalid
        """
        # Step 1: Validate
        if not self._validate(config):
            raise ValueError(f"Invalid security configuration for {self.__class__.__name__}")

        # Step 2: Pre-generation hooks
        self._trigger_hook("pre_generate", config)

        # Step 3: Generate template
        template = self._define_template(config)

        # Step 4: Post-generation hooks
        self._trigger_hook("post_generate", template)

        # Step 5: Return
        return template


# ═══════════════════════════════════════════════════════════════════
# AUTHENTICATION TEMPLATE
# ═══════════════════════════════════════════════════════════════════


class AuthTemplate(SecurityTemplate):
    """
    Authentication template generator.

    Generates JWT-based authentication code with refresh tokens.
    """

    def _validate(self, config: SecurityConfig) -> bool:
        """Validate auth configuration (pure function composition)."""
        # Compose validation functions
        validations = compose(
            lambda c: c.jwt_expiry >= 300,  # Min 5 minutes
            lambda c: c.refresh_expiry >= 3600,  # Min 1 hour
            lambda c: c.bcrypt_cost >= 10,  # Min cost factor
        )
        return validations(config)

    def _define_template(self, config: SecurityConfig) -> str:
        """Generate authentication template."""
        return f'''
# ═══════════════════════════════════════════════════════
# Authentication Module — JWT with Refresh Tokens
# ═══════════════════════════════════════════════════════
# Author: Georgios-Chrysovalantis Chatzivantsidis
# Generated by AI Orchestrator Security Templates
# ═══════════════════════════════════════════════════════

"""
Authentication module with JWT and refresh tokens.

Security Configuration:
- JWT Expiry: {config.jwt_expiry}s ({config.jwt_expiry // 60} minutes)
- Refresh Token Expiry: {config.refresh_expiry}s ({config.refresh_expiry // 86400} days)
- Password Hashing: bcrypt (cost={config.bcrypt_cost})
- Session Timeout: {config.session_timeout}s ({config.session_timeout // 60} minutes)
"""

import jwt
import bcrypt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

# ═══════════════════════════════════════════════════════
# Configuration (from environment variables)
# ═══════════════════════════════════════════════════════

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = {config.jwt_expiry}
REFRESH_EXPIRY_SECONDS = {config.refresh_expiry}
BCRYPT_COST = {config.bcrypt_cost}


@dataclass(frozen=True)
class TokenPair:
    """Immutable token pair (access + refresh)."""
    access_token: str
    refresh_token: str
    expires_at: datetime


def _get_jwt_secret() -> str:
    """Get JWT secret from environment (pure function)."""
    if not JWT_SECRET:
        raise ValueError("JWT_SECRET environment variable not set")
    return JWT_SECRET


@lru_cache(maxsize=128)
def hash_password(password: str) -> str:
    """
    Hash password with bcrypt (cached for performance).
    
    Functional Programming: Pure function with memoization.
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=BCRYPT_COST)
    ).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash (pure function).
    
    Args:
        password: Plain text password
        hashed: Hashed password
    
    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed.encode('utf-8')
    )


def create_access_token(user_id: str, additional_claims: Dict[str, Any] = None) -> str:
    """
    Create JWT access token (pure function).
    
    Args:
        user_id: User identifier
        additional_claims: Additional JWT claims
    
    Returns:
        JWT access token
    """
    now = datetime.utcnow()
    payload = {{
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(seconds=JWT_EXPIRY_SECONDS),
        "type": "access",
    }}
    
    if additional_claims:
        payload.update(additional_claims)
    
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token (pure function).
    
    Args:
        user_id: User identifier
    
    Returns:
        JWT refresh token
    """
    now = datetime.utcnow()
    payload = {{
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(seconds=REFRESH_EXPIRY_SECONDS),
        "type": "refresh",
    }}
    
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def create_token_pair(user_id: str) -> TokenPair:
    """
    Create token pair (access + refresh).
    
    Args:
        user_id: User identifier
    
    Returns:
        TokenPair with access and refresh tokens
    """
    access = create_access_token(user_id)
    refresh = create_refresh_token(user_id)
    expires = datetime.utcnow() + timedelta(seconds=JWT_EXPIRY_SECONDS)
    
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires
    )


def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """
    Verify and decode JWT token (pure function).
    
    Args:
        token: JWT token
        token_type: Expected token type ("access" or "refresh")
    
    Returns:
        User ID if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        
        # Verify token type
        if payload.get("type") != token_type:
            return None
        
        # Verify not expired (jwt.decode does this automatically)
        return payload.get("sub")
        
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def refresh_access_token(refresh_token: str) -> Optional[TokenPair]:
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        New TokenPair if refresh token is valid, None otherwise
    """
    user_id = verify_token(refresh_token, token_type="refresh")
    
    if not user_id:
        return None
    
    return create_token_pair(user_id)


# ═══════════════════════════════════════════════════════
# Middleware for FastAPI/Flask
# ═══════════════════════════════════════════════════════

def auth_middleware(get_current_user):
    """
    Authentication middleware decorator.
    
    Usage:
        @app.get("/protected")
        @auth_middleware(get_current_user)
        def protected_route(user: User):
            return {{"user": user}}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get token from request
            token = get_token_from_request()
            
            if not token:
                raise HTTPException(401, "Missing authentication token")
            
            user_id = verify_token(token)
            
            if not user_id:
                raise HTTPException(401, "Invalid or expired token")
            
            # Get user from database
            user = get_current_user(user_id)
            
            if not user:
                raise HTTPException(404, "User not found")
            
            # Inject user into function
            kwargs['user'] = user
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_token_from_request() -> Optional[str]:
    """Extract token from Authorization header."""
    from fastapi import Request
    
    # Implementation depends on framework
    # This is a placeholder
    return None
'''


# ═══════════════════════════════════════════════════════
# RBAC (ROLE-BASED ACCESS CONTROL) TEMPLATE
# ═══════════════════════════════════════════════════════


class RBACTemplate(SecurityTemplate):
    """
    Role-Based Access Control template generator.

    Generates RBAC middleware and permission system.
    """

    def _validate(self, config: SecurityConfig) -> bool:
        """Validate RBAC configuration."""
        return len(config.cors_origins) > 0

    def _define_template(self, config: SecurityConfig) -> str:
        """Generate RBAC template."""
        cors_origins_str = ", ".join(f'"{origin}"' for origin in config.cors_origins)

        return f'''
# ═══════════════════════════════════════════════════════
# RBAC Middleware — Role-Based Access Control
# ═══════════════════════════════════════════════════════
# Author: Georgios-Chrysovalantis Chatzivantsidis
# Generated by AI Orchestrator Security Templates
# ═══════════════════════════════════════════════════════

"""
Role-Based Access Control (RBAC) middleware.

Configuration:
- CORS Origins: [{cors_origins_str}]
- CSRF Protection: {"Enabled" if config.csrf_enabled else "Disabled"}
"""

from enum import Enum
from typing import List, Set, Callable
from functools import wraps
from dataclasses import dataclass


class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    MODERATOR = "moderator"


class Permission(Enum):
    """Fine-grained permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MODERATE = "moderate"


# Role-Permission mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {{
    Role.ADMIN: {{Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}},
    Role.MODERATOR: {{Permission.READ, Permission.WRITE, Permission.MODERATE}},
    Role.USER: {{Permission.READ, Permission.WRITE}},
    Role.GUEST: {{Permission.READ}},
}}


@dataclass(frozen=True)
class UserContext:
    """Immutable user context."""
    user_id: str
    roles: List[Role]
    permissions: Set[Permission] = field(default_factory=set)
    
    def __post_init__(self):
        """Derive permissions from roles."""
        all_permissions = set()
        for role in self.roles:
            all_permissions.update(ROLE_PERMISSIONS.get(role, set()))
        object.__setattr__(self, 'permissions', all_permissions)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has permission (pure method)."""
        return permission in self.permissions
    
    def has_role(self, role: Role) -> bool:
        """Check if user has role (pure method)."""
        return role in self.roles


def require_permission(permission: Permission):
    """
    Decorator: Require specific permission.
    
    Usage:
        @app.delete("/users/<id>")
        @require_permission(Permission.DELETE)
        def delete_user(user: UserContext):
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user: UserContext, **kwargs):
            if not user.has_permission(permission):
                raise HTTPException(403, f"Missing permission: {{permission.value}}")
            return func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """
    Decorator: Require specific role.
    
    Usage:
        @app.get("/admin/dashboard")
        @require_role(Role.ADMIN)
        def admin_dashboard(user: UserContext):
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user: UserContext, **kwargs):
            if not user.has_role(role):
                raise HTTPException(403, f"Missing role: {{role.value}}")
            return func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_any_role(roles: List[Role]):
    """
    Decorator: Require any of the specified roles.
    
    Usage:
        @app.get("/moderate")
        @require_any_role([Role.ADMIN, Role.MODERATOR])
        def moderate_content(user: UserContext):
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user: UserContext, **kwargs):
            if not any(user.has_role(role) for role in roles):
                raise HTTPException(403, f"Missing required roles: {{roles}}")
            return func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_all_roles(roles: List[Role]):
    """
    Decorator: Require all specified roles.
    
    Usage:
        @app.post("/admin/users")
        @require_all_roles([Role.ADMIN])
        def create_user(user: UserContext):
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user: UserContext, **kwargs):
            if not all(user.has_role(role) for role in roles):
                raise HTTPException(403, f"Missing required roles: {{roles}}")
            return func(*args, user=user, **kwargs)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════
# CORS Middleware
# ═══════════════════════════════════════════════════════

CORS_ORIGINS = [{cors_origins_str}]

def cors_middleware(response):
    """Add CORS headers to response."""
    response.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS[0] if len(CORS_ORIGINS) == 1 else "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# ═══════════════════════════════════════════════════════
# CSRF Protection
# ═══════════════════════════════════════════════════════

{"# CSRF protection enabled" if config.csrf_enabled else "# CSRF protection disabled"}

import secrets

def generate_csrf_token() -> str:
    """Generate CSRF token (pure function)."""
    return secrets.token_hex(32)

def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token (pure function)."""
    return secrets.compare_digest(token, session_token)

'''


# ═══════════════════════════════════════════════════════
# RATE LIMITING TEMPLATE
# ═══════════════════════════════════════════════════════


class RateLimitTemplate(SecurityTemplate):
    """
    Rate limiting template generator.

    Generates token bucket rate limiting implementation.
    """

    def _validate(self, config: SecurityConfig) -> bool:
        """Validate rate limit configuration."""
        return config.rate_limit_window > 0 and config.rate_limit_max > 0

    def _define_template(self, config: SecurityConfig) -> str:
        """Generate rate limiting template."""
        return f'''
# ═══════════════════════════════════════════════════════
# Rate Limiting — Token Bucket Algorithm
# ═══════════════════════════════════════════════════════
# Author: Georgios-Chrysovalantis Chatzivantsidis
# Generated by AI Orchestrator Security Templates
# ═══════════════════════════════════════════════════════

"""
Rate limiting using token bucket algorithm.

Configuration:
- Window: {config.rate_limit_window}s
- Max Requests: {config.rate_limit_max} per window
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
from functools import wraps
import threading


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int = {config.rate_limit_max}
    tokens: float = field(default={config.rate_limit_max})
    last_update: float = field(default_factory=time.time)
    refill_rate: float = {config.rate_limit_max / config.rate_limit_window}  # tokens per second
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if tokens consumed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update
        
        # Refill tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now
        
        # Check if enough tokens available
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Rate limiter with per-key buckets."""
    
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = defaultdict(TokenBucket)
        self._lock = threading.Lock()
    
    def get_bucket(self, key: str) -> TokenBucket:
        """Get or create bucket for key."""
        with self._lock:
            return self._buckets[key]
    
    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., user_id, IP address)
            tokens: Number of tokens to consume
        
        Returns:
            True if allowed, False if rate limited
        """
        bucket = self.get_bucket(key)
        return bucket.consume(tokens)


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limit(key_func: Callable, max_requests: int = {config.rate_limit_max}, window: int = {config.rate_limit_window}):
    """
    Rate limiting decorator.
    
    Usage:
        @app.get("/api/data")
        @rate_limit(key_func=lambda: get_current_user().id)
        def get_data(user: User):
            pass
    
    Args:
        key_func: Function to extract rate limit key from request
        max_requests: Max requests per window
        window: Window in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func()
            
            if not _rate_limiter.is_allowed(key):
                raise HTTPException(429, "Rate limit exceeded")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit_ip(max_requests: int = {config.rate_limit_max}, window: int = {config.rate_limit_window}):
    """Rate limit by IP address."""
    return rate_limit(
        key_func=lambda: get_request_ip(),
        max_requests=max_requests,
        window=window
    )


def rate_limit_user(max_requests: int = {config.rate_limit_max}, window: int = {config.rate_limit_window}):
    """Rate limit by user ID."""
    return rate_limit(
        key_func=lambda: get_current_user().id,
        max_requests=max_requests,
        window=window
    )


def get_request_ip() -> str:
    """Get client IP address from request."""
    # Implementation depends on framework
    return "127.0.0.1"


def get_current_user():
    """Get current authenticated user."""
    # Implementation depends on auth system
    pass

'''


# ═══════════════════════════════════════════════════════
# FACTORY FOR TEMPLATES
# ═══════════════════════════════════════════════════════


class SecurityTemplateFactory:
    """
    Factory Method for creating security templates.

    Usage:
        factory = SecurityTemplateFactory()
        auth_template = factory.create("auth")
        rbac_template = factory.create("rbac")
        rate_limit_template = factory.create("rate_limit")
    """

    _templates = {
        "auth": AuthTemplate,
        "rbac": RBACTemplate,
        "rate_limit": RateLimitTemplate,
    }

    def create(self, template_type: str) -> SecurityTemplate:
        """
        Create security template.

        Args:
            template_type: Type of template ("auth", "rbac", "rate_limit")

        Returns:
            SecurityTemplate instance

        Raises:
            ValueError: If template type is unknown
        """
        if template_type not in self._templates:
            raise ValueError(f"Unknown template type: {template_type}")

        return self._templates[template_type]()

    def register_template(self, name: str, template_class: type) -> None:
        """
        Register custom template.

        Args:
            name: Template name
            template_class: Template class (must extend SecurityTemplate)
        """
        if not issubclass(template_class, SecurityTemplate):
            raise TypeError("Template must extend SecurityTemplate")

        self._templates[name] = template_class


# ═══════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════


def generate_auth_template(config: SecurityConfig = None) -> str:
    """
    Generate authentication template.

    Args:
        config: Security configuration (uses defaults if None)

    Returns:
        Generated authentication code
    """
    if config is None:
        config = SecurityConfig()

    factory = SecurityTemplateFactory()
    template = factory.create("auth")
    return template.generate(config)


def generate_rbac_template(config: SecurityConfig = None) -> str:
    """
    Generate RBAC template.

    Args:
        config: Security configuration

    Returns:
        Generated RBAC code
    """
    if config is None:
        config = SecurityConfig()

    factory = SecurityTemplateFactory()
    template = factory.create("rbac")
    return template.generate(config)


def generate_rate_limit_template(config: SecurityConfig = None) -> str:
    """
    Generate rate limiting template.

    Args:
        config: Security configuration

    Returns:
        Generated rate limiting code
    """
    if config is None:
        config = SecurityConfig()

    factory = SecurityTemplateFactory()
    template = factory.create("rate_limit")
    return template.generate(config)
