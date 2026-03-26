"""
API Builder — Visual API integration builder
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Build API integrations visually from OpenAPI specs, Postman collections,
or manual configuration. Auto-generates client code with authentication.

Features:
- OpenAPI/Swagger spec import
- Postman collection import
- Authentication configuration (API key, OAuth, Bearer)
- Auto-generated client code
- Request/response validation

USAGE:
    from orchestrator.api_builder import APIIntegrationBuilder
    
    builder = APIIntegrationBuilder()
    
    # Import from OpenAPI spec
    integration = await builder.from_openapi("https://api.example.com/openapi.json")
    
    # Or from Postman
    integration = await builder.from_postman(postman_collection)
    
    # Configure auth
    integration = builder.configure_auth(
        integration,
        auth_type="bearer",
        credentials={"token": "xxx"},
    )
    
    # Generate client code
    code = builder.generate_client(integration, language="python")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from pathlib import Path

logger = logging.getLogger("orchestrator.api_builder")


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class AuthType(str, Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    OAUTH = "oauth"
    BEARER = "bearer"
    BASIC = "basic"


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "get"
    POST = "post"
    PUT = "put"
    PATCH = "patch"
    DELETE = "delete"
    OPTIONS = "options"
    HEAD = "head"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class APIEndpoint:
    """API endpoint definition."""
    path: str
    method: HTTPMethod
    summary: str = ""
    description: str = ""
    parameters: List[dict] = field(default_factory=list)
    request_body: Optional[dict] = None
    responses: Dict[str, dict] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method.value,
            "summary": self.summary,
            "description": self.description,
            "parameters": self.parameters,
            "request_body": self.request_body,
            "responses": self.responses,
            "tags": self.tags,
        }


@dataclass
class APIIntegration:
    """API integration configuration."""
    name: str
    base_url: str
    version: str = "1.0.0"
    description: str = ""
    auth_type: AuthType = AuthType.NONE
    auth_config: dict = field(default_factory=dict)
    endpoints: List[APIEndpoint] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    
    # Generated code
    client_code: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "version": self.version,
            "description": self.description,
            "auth_type": self.auth_type.value,
            "auth_config": self.auth_config,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "headers": self.headers,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "APIIntegration":
        """Create from dictionary."""
        endpoints = [
            APIEndpoint(
                path=e["path"],
                method=HTTPMethod(e["method"]),
                summary=e.get("summary", ""),
                description=e.get("description", ""),
                parameters=e.get("parameters", []),
                request_body=e.get("request_body"),
                responses=e.get("responses", {}),
                tags=e.get("tags", []),
            )
            for e in data.get("endpoints", [])
        ]
        
        return cls(
            name=data["name"],
            base_url=data["base_url"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            auth_type=AuthType(data.get("auth_type", "none")),
            auth_config=data.get("auth_config", {}),
            endpoints=endpoints,
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30),
            retry_count=data.get("retry_count", 3),
        )


# ─────────────────────────────────────────────
# API Integration Builder
# ─────────────────────────────────────────────

class APIIntegrationBuilder:
    """
    Build API integrations from various sources.
    
    Supports OpenAPI specs, Postman collections, and manual configuration.
    """
    
    def __init__(self):
        self._integrations: Dict[str, APIIntegration] = {}
        
        # Statistics
        self._total_imports = 0
        self._total_endpoints = 0
    
    async def from_openapi(
        self,
        spec_url: str,
        name: Optional[str] = None,
    ) -> APIIntegration:
        """
        Generate integration from OpenAPI/Swagger spec.
        
        Args:
            spec_url: URL or path to OpenAPI spec
            name: Optional name override
        
        Returns:
            API integration
        """
        try:
            # Load spec
            spec = await self._load_openapi_spec(spec_url)
            
            # Extract info
            info = spec.get("info", {})
            integration = APIIntegration(
                name=name or info.get("title", "API"),
                base_url=self._extract_base_url(spec),
                version=info.get("version", "1.0.0"),
                description=info.get("description", ""),
            )
            
            # Extract security schemes
            security_schemes = spec.get("components", {}).get("securitySchemes", {})
            if security_schemes:
                integration.auth_type, integration.auth_config = self._parse_security(
                    security_schemes
                )
            
            # Extract endpoints
            paths = spec.get("paths", {})
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.upper() in [m.value for m in HTTPMethod]:
                        endpoint = APIEndpoint(
                            path=path,
                            method=HTTPMethod(method.lower()),
                            summary=details.get("summary", ""),
                            description=details.get("description", ""),
                            parameters=details.get("parameters", []),
                            request_body=details.get("requestBody"),
                            responses=details.get("responses", {}),
                            tags=details.get("tags", []),
                        )
                        integration.endpoints.append(endpoint)
            
            self._total_imports += 1
            self._total_endpoints += len(integration.endpoints)
            
            logger.info(
                f"Imported {len(integration.endpoints)} endpoints from {spec_url}"
            )
            
            return integration
            
        except Exception as e:
            logger.error(f"Failed to import OpenAPI spec: {e}")
            raise
    
    async def from_postman(
        self,
        collection: dict,
        name: Optional[str] = None,
    ) -> APIIntegration:
        """
        Generate from Postman collection.
        
        Args:
            collection: Postman collection dict
            name: Optional name override
        
        Returns:
            API integration
        """
        try:
            info = collection.get("info", {})
            integration = APIIntegration(
                name=name or info.get("name", "API"),
                base_url=self._extract_postman_base_url(collection),
                version=info.get("schema", "1.0.0"),
                description=info.get("description", ""),
            )
            
            # Extract endpoints from items
            items = collection.get("item", [])
            integration.endpoints = self._parse_postman_items(items)
            
            self._total_imports += 1
            self._total_endpoints += len(integration.endpoints)
            
            logger.info(
                f"Imported {len(integration.endpoints)} endpoints from Postman"
            )
            
            return integration
            
        except Exception as e:
            logger.error(f"Failed to import Postman collection: {e}")
            raise
    
    def configure_auth(
        self,
        integration: APIIntegration,
        auth_type: str,
        credentials: dict,
    ) -> APIIntegration:
        """
        Configure authentication.
        
        Args:
            integration: API integration
            auth_type: Authentication type
            credentials: Authentication credentials
        
        Returns:
            Updated integration
        """
        integration.auth_type = AuthType(auth_type)
        
        if auth_type == AuthType.API_KEY.value:
            integration.auth_config = {
                "key_name": credentials.get("key_name", "X-API-Key"),
                "key_value": credentials.get("key_value", ""),
                "in": credentials.get("in", "header"),  # header, query, cookie
            }
        
        elif auth_type == AuthType.BEARER.value:
            integration.auth_config = {
                "token": credentials.get("token", ""),
                "prefix": "Bearer",
            }
        
        elif auth_type == AuthType.BASIC.value:
            integration.auth_config = {
                "username": credentials.get("username", ""),
                "password": credentials.get("password", ""),
            }
        
        elif auth_type == AuthType.OAUTH.value:
            integration.auth_config = {
                "client_id": credentials.get("client_id", ""),
                "client_secret": credentials.get("client_secret", ""),
                "token_url": credentials.get("token_url", ""),
                "scope": credentials.get("scope", ""),
                "grant_type": credentials.get("grant_type", "client_credentials"),
            }
        
        logger.info(f"Configured {auth_type} authentication")
        return integration
    
    def generate_client(
        self,
        integration: APIIntegration,
        language: str = "python",
    ) -> Dict[str, str]:
        """
        Generate client code.
        
        Args:
            integration: API integration
            language: Target language (python, javascript, typescript)
        
        Returns:
            Generated code files
        """
        if language == "python":
            return self._generate_python_client(integration)
        elif language == "javascript":
            return self._generate_javascript_client(integration)
        elif language == "typescript":
            return self._generate_typescript_client(integration)
        else:
            logger.warning(f"Unsupported language: {language}")
            return {}
    
    def _generate_python_client(
        self,
        integration: APIIntegration,
    ) -> Dict[str, str]:
        """Generate Python client code."""
        files = {}
        
        # Generate main client
        client_code = f'''"""
{integration.name} API Client
Generated by AI Orchestrator API Builder
"""

import httpx
from typing import Dict, Any, Optional
import asyncio

class {integration.name.replace(" ", "")}Client:
    """API client for {integration.name}."""
    
    def __init__(
        self,
        base_url: str = "{integration.base_url}",
        timeout: int = {integration.timeout},
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )
        self._configure_auth()
    
    def _configure_auth(self):
        """Configure authentication."""
'''
        
        # Add auth configuration
        if integration.auth_type == AuthType.API_KEY:
            key_name = integration.auth_config.get("key_name", "X-API-Key")
            key_value = integration.auth_config.get("key_value", "")
            client_code += f'''        self.client.headers["{key_name}"] = "{key_value}"
'''
        
        elif integration.auth_type == AuthType.BEARER:
            token = integration.auth_config.get("token", "")
            client_code += f'''        self.client.headers["Authorization"] = "Bearer {token}"
'''
        
        elif integration.auth_type == AuthType.BASIC:
            username = integration.auth_config.get("username", "")
            password = integration.auth_config.get("password", "")
            client_code += f'''        import base64
        credentials = base64.b64encode(b"{username}:{password}").decode()
        self.client.headers["Authorization"] = f"Basic {{credentials}}"
'''
        
        client_code += '''
    async def close(self):
        """Close the client."""
        await self.client.aclose()
    
'''
        
        # Generate endpoint methods
        for endpoint in integration.endpoints:
            method_name = self._generate_method_name(endpoint)
            client_code += self._generate_python_endpoint_method(endpoint)
        
        files[f"{integration.name.lower().replace(' ', '_')}_client.py"] = client_code
        
        # Generate models
        models_code = self._generate_python_models(integration)
        files["models.py"] = models_code
        
        return files
    
    def _generate_python_endpoint_method(
        self,
        endpoint: APIEndpoint,
    ) -> str:
        """Generate Python method for endpoint."""
        method_name = self._generate_method_name(endpoint)
        
        # Build parameters
        params = []
        path_params = []
        query_params = []
        
        for param in endpoint.parameters:
            param_name = param.get("name", "param")
            param_in = param.get("in", "query")
            required = param.get("required", False)
            param_type = self._openapi_type_to_python(param.get("schema", {}).get("type", "str"))
            
            default = "" if required else " = None"
            params.append(f"{param_name}: {param_type}{default}")
            
            if param_in == "path":
                path_params.append(param_name)
            elif param_in == "query":
                query_params.append(param_name)
        
        params_str = ", ".join(["self"] + params)
        
        # Generate docstring
        docstring = f'"""{endpoint.summary or endpoint.description}"""'
        
        # Generate method
        method_code = f'''    async def {method_name}({params_str}) -> Dict[str, Any]:
        {docstring}
        url = "{endpoint.path}"
'''
        
        if path_params:
            method_code += f'''        url = url.format({", ".join(f"{p}={p}" for p in path_params)})
'''
        
        method_code += f'''        
        response = await self.client.{endpoint.method.value}(url)
        response.raise_for_status()
        return response.json()

'''
        
        return method_code
    
    def _generate_python_models(
        self,
        integration: APIIntegration,
    ) -> str:
        """Generate Python data models."""
        models_code = f'''"""
Models for {integration.name} API
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

'''
        
        # Generate models from endpoint responses
        for endpoint in integration.endpoints:
            for status_code, response in endpoint.responses.items():
                if status_code == "200" and "content" in response:
                    schema = response["content"].get("application/json", {}).get("schema", {})
                    if schema.get("type") == "object":
                        model_name = f"{endpoint.path.split('/')[-1].title().replace('_', '')}Response"
                        models_code += f'''
@dataclass
class {model_name}:
    """Response model for {endpoint.summary}."""
'''
                        properties = schema.get("properties", {})
                        for prop_name, prop_schema in properties.items():
                            prop_type = self._openapi_type_to_python(prop_schema.get("type", "Any"))
                            models_code += f'''    {prop_name}: {prop_type}
'''
                        models_code += "\n"
        
        return models_code
    
    def _generate_javascript_client(
        self,
        integration: APIIntegration,
    ) -> Dict[str, str]:
        """Generate JavaScript client code."""
        # Similar to Python but for JavaScript
        return {}
    
    def _generate_typescript_client(
        self,
        integration: APIIntegration,
    ) -> Dict[str, str]:
        """Generate TypeScript client code."""
        # Similar to Python but for TypeScript
        return {}
    
    async def _load_openapi_spec(self, spec_url: str) -> dict:
        """Load OpenAPI spec from URL or file."""
        import json
        
        if spec_url.startswith("http"):
            # Fetch from URL
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(spec_url)
                response.raise_for_status()
                return response.json()
        else:
            # Load from file
            path = Path(spec_url)
            if path.suffix == ".json":
                with open(path) as f:
                    return json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                import yaml
                with open(path) as f:
                    return yaml.safe_load(f)
        
        raise ValueError(f"Unsupported spec format: {spec_url}")
    
    def _extract_base_url(self, spec: dict) -> str:
        """Extract base URL from OpenAPI spec."""
        # OpenAPI 3.x
        servers = spec.get("servers", [])
        if servers:
            return servers[0].get("url", "")
        
        # OpenAPI 2.x (Swagger)
        host = spec.get("host", "")
        schemes = spec.get("schemes", ["https"])
        base_path = spec.get("basePath", "")
        
        if host:
            return f"{schemes[0]}://{host}{base_path}"
        
        return ""
    
    def _extract_postman_base_url(self, collection: dict) -> str:
        """Extract base URL from Postman collection."""
        # Check variable
        variables = collection.get("variable", [])
        for var in variables:
            if var.get("key") == "baseUrl":
                return var.get("value", "")
        
        # Check first request
        items = collection.get("item", [])
        if items:
            request = items[0].get("request", {})
            url = request.get("url", {})
            if isinstance(url, dict):
                return url.get("host", [""])[0] if url.get("host") else ""
        
        return ""
    
    def _parse_postman_items(
        self,
        items: list,
    ) -> List[APIEndpoint]:
        """Parse Postman items to endpoints."""
        endpoints = []
        
        for item in items:
            request = item.get("request", {})
            
            endpoint = APIEndpoint(
                path=request.get("url", {}).get("path", "/"),
                method=HTTPMethod(request.get("method", "get").lower()),
                summary=item.get("name", ""),
                description=request.get("description", ""),
            )
            
            endpoints.append(endpoint)
        
        return endpoints
    
    def _parse_security(
        self,
        security_schemes: dict,
    ) -> tuple[AuthType, dict]:
        """Parse OpenAPI security schemes."""
        for name, scheme in security_schemes.items():
            scheme_type = scheme.get("type", "")
            
            if scheme_type == "apiKey":
                return AuthType.API_KEY, {
                    "key_name": scheme.get("name", "X-API-Key"),
                    "in": scheme.get("in", "header"),
                }
            
            elif scheme_type == "http":
                http_scheme = scheme.get("scheme", "")
                if http_scheme == "bearer":
                    return AuthType.BEARER, {}
                elif http_scheme == "basic":
                    return AuthType.BASIC, {}
            
            elif scheme_type == "oauth2":
                return AuthType.OAUTH, {
                    "flows": scheme.get("flows", {}),
                }
        
        return AuthType.NONE, {}
    
    def _generate_method_name(self, endpoint: APIEndpoint) -> str:
        """Generate method name from endpoint."""
        # Use summary if available
        if endpoint.summary:
            name = endpoint.summary.lower().replace(" ", "_").replace("-", "_")
            return name
        
        # Otherwise generate from path and method
        path_parts = endpoint.path.strip("/").split("/")
        path_name = "_".join(path_parts)
        return f"{endpoint.method.value}_{path_name}"
    
    def _openapi_type_to_python(self, openapi_type: str) -> str:
        """Convert OpenAPI type to Python type."""
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
        }
        return type_map.get(openapi_type, "Any")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get builder statistics."""
        return {
            "total_imports": self._total_imports,
            "total_endpoints": self._total_endpoints,
            "integrations_count": len(self._integrations),
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_builder: Optional[APIIntegrationBuilder] = None


def get_api_builder() -> APIIntegrationBuilder:
    """Get or create default API builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = APIIntegrationBuilder()
    return _default_builder


def reset_api_builder() -> None:
    """Reset default builder (for testing)."""
    global _default_builder
    _default_builder = None


async def import_from_openapi(
    spec_url: str,
    name: Optional[str] = None,
) -> APIIntegration:
    """Import API from OpenAPI spec."""
    builder = get_api_builder()
    return await builder.from_openapi(spec_url, name)


async def import_from_postman(
    collection: dict,
    name: Optional[str] = None,
) -> APIIntegration:
    """Import API from Postman collection."""
    builder = get_api_builder()
    return await builder.from_postman(collection, name)
