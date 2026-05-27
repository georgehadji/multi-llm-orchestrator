"""
Configuration-as-Code — Entity, Auth, and Agent schemas.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 5, Phase B1 (Base44-inspired): Entity schemas as JSON/YAML
that generate typed infrastructure (models, auth, agents, API endpoints).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    JSON = "json"


class AuthMethod(str, Enum):
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SESSION = "session"
    NONE = "none"


@dataclass
class EntityField:
    """A field definition for an entity schema."""

    name: str
    type: FieldType = FieldType.STRING
    required: bool = False
    unique: bool = False
    default: Any = None
    description: str = ""
    max_length: int = 0
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type.value}
        if self.required:
            d["required"] = True
        if self.unique:
            d["unique"] = True
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntityField:
        return cls(
            name=d["name"],
            type=FieldType(d.get("type", "string")),
            required=d.get("required", False),
            unique=d.get("unique", False),
            description=d.get("description", ""),
        )


@dataclass
class EntitySchema:
    """Schema for a data entity (table/model)."""

    name: str
    description: str = ""
    fields: list[EntityField] = field(default_factory=list)
    auth_required: bool = False
    api_endpoints: list[str] = field(default_factory=list)  # GET, POST, PUT, DELETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "auth_required": self.auth_required,
            "api_endpoints": self.api_endpoints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EntitySchema:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            fields=[EntityField.from_dict(f) for f in d.get("fields", [])],
            auth_required=d.get("auth_required", False),
            api_endpoints=d.get("api_endpoints", []),
        )

    def to_pydantic(self) -> str:
        """Generate Pydantic model code from schema."""
        lines = [f"from pydantic import BaseModel, Field", "from typing import Optional", ""]
        lines.append(f"class {self.name}(BaseModel):")
        if self.description:
            lines.append(f'    """{self.description}"""')
        for fld in self.fields:
            py_type = self._py_type(fld.type)
            if not fld.required:
                py_type = f"Optional[{py_type}]"
                default = "None"
            else:
                default = ""
            extra = ""
            if fld.description:
                extra += f', description="{fld.description}"'
            if default:
                lines.append(f"    {fld.name}: {py_type} = {default}{extra}")
            else:
                lines.append(f"    {fld.name}: {py_type}{extra}")
        return "\n".join(lines)

    def to_typescript(self) -> str:
        """Generate TypeScript interface from schema."""
        lines = [f"// Generated from entity schema: {self.name}"]
        if self.description:
            lines.append(f"// {self.description}")
        lines.append(f"export interface {self.name} {{")
        for fld in self.fields:
            ts_type = self._ts_type(fld.type)
            optional = "?" if not fld.required else ""
            jsdoc = f"/** {fld.description} */" if fld.description else ""
            if jsdoc:
                lines.append(f"  {jsdoc}")
            lines.append(f"  {fld.name}{optional}: {ts_type};")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _py_type(ft: FieldType) -> str:
        mapping = {
            FieldType.STRING: "str",
            FieldType.INTEGER: "int",
            FieldType.FLOAT: "float",
            FieldType.BOOLEAN: "bool",
            FieldType.DATE: "date",
            FieldType.DATETIME: "datetime",
            FieldType.EMAIL: "str",
            FieldType.URL: "str",
            FieldType.UUID: "str",
            FieldType.JSON: "dict",
        }
        return mapping.get(ft, "str")

    @staticmethod
    def _ts_type(ft: FieldType) -> str:
        mapping = {
            FieldType.STRING: "string",
            FieldType.INTEGER: "number",
            FieldType.FLOAT: "number",
            FieldType.BOOLEAN: "boolean",
            FieldType.DATE: "string",
            FieldType.DATETIME: "string",
            FieldType.EMAIL: "string",
            FieldType.URL: "string",
            FieldType.UUID: "string",
            FieldType.JSON: "Record<string, any>",
        }
        return mapping.get(ft, "string")


@dataclass
class AuthConfig:
    """Authentication configuration schema."""

    method: AuthMethod = AuthMethod.JWT
    jwt_secret_env: str = "JWT_SECRET"
    token_expiry_minutes: int = 1440
    refresh_enabled: bool = False
    api_key_header: str = "X-API-Key"
    oauth2_providers: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "token_expiry_minutes": self.token_expiry_minutes,
            "refresh_enabled": self.refresh_enabled,
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }


@dataclass
class AppConfig:
    """Complete application configuration schema (B1)."""

    app_name: str = ""
    description: str = ""
    version: str = "0.1.0"
    entities: list[EntitySchema] = field(default_factory=list)
    auth: AuthConfig = field(default_factory=AuthConfig)
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    database_url_env: str = "DATABASE_URL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "description": self.description,
            "version": self.version,
            "entities": [e.to_dict() for e in self.entities],
            "auth": self.auth.to_dict(),
            "api_prefix": self.api_prefix,
            "cors_origins": self.cors_origins,
        }

    def save(self, path: str) -> None:
        """Save config to JSON or YAML file."""
        fp = Path(path)
        data = self.to_dict()
        if fp.suffix in (".yaml", ".yml"):
            import yaml

            fp.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        else:
            fp.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> AppConfig:
        """Load config from JSON or YAML file."""
        fp = Path(path)
        if not fp.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = fp.read_text(encoding="utf-8")
        if fp.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        entities = [EntitySchema.from_dict(e) for e in data.get("entities", [])]
        auth_data = data.get("auth", {})
        auth = AuthConfig(
            method=AuthMethod(auth_data.get("method", "jwt")),
            token_expiry_minutes=auth_data.get("token_expiry_minutes", 1440),
            refresh_enabled=auth_data.get("refresh_enabled", False),
            rate_limit_per_minute=auth_data.get("rate_limit_per_minute", 60),
        )
        return cls(
            app_name=data.get("app_name", ""),
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            entities=entities,
            auth=auth,
            api_prefix=data.get("api_prefix", "/api/v1"),
        )

    def generate_all_types(self) -> dict[str, str]:
        """Generate Pydantic + TypeScript types for all entities."""
        result = {}
        for entity in self.entities:
            result[f"{entity.name}_pydantic.py"] = entity.to_pydantic()
            result[f"{entity.name}.ts"] = entity.to_typescript()
        return result
