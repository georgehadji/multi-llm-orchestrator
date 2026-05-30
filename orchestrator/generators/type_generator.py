"""
DynamicTypeGenerator - TypeScript, Pydantic, and SQL from entity schemas.
==========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 5, Phase B2 (Base44-inspired).
Extends B1 with runtime type generation from AppConfig entities.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class TargetLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    SQL = "sql"
    GRAPHQL = "graphql"
    PROTOBUF = "protobuf"


@dataclass
class TypeOutput:
    language: TargetLanguage
    code: str
    filename: str


class DynamicTypeGenerator:
    """Generates typed code from entity schemas in multiple languages."""

    def generate(self, entity, language: TargetLanguage) -> TypeOutput:
        methods = {
            TargetLanguage.PYTHON: self._to_pydantic,
            TargetLanguage.TYPESCRIPT: self._to_typescript,
            TargetLanguage.SQL: self._to_sql,
            TargetLanguage.GRAPHQL: self._to_graphql,
        }
        fn = methods.get(language, self._to_pydantic)
        code = fn(entity)
        exts = {
            TargetLanguage.PYTHON: "py",
            TargetLanguage.TYPESCRIPT: "ts",
            TargetLanguage.SQL: "sql",
            TargetLanguage.GRAPHQL: "graphql",
        }
        return TypeOutput(
            language=language,
            code=code,
            filename=f"{entity.name.lower()}.{exts.get(language, 'txt')}",
        )

    def _to_pydantic(self, entity) -> str:
        lines = [
            "from pydantic import BaseModel, Field",
            "from typing import Optional",
            "",
            f"class {entity.name}(BaseModel):",
        ]
        for f in entity.fields:
            py_type = self._map_py(f.type if hasattr(f, "type") else "string")
            optional = (
                "Optional[" + py_type + "] = None" if not getattr(f, "required", False) else py_type
            )
            desc = f', description="{f.description}"' if getattr(f, "description", "") else ""
            lines.append(f"    {f.name}: {optional}{desc}")
        return "\n".join(lines)

    def _to_typescript(self, entity) -> str:
        lines = [f"export interface {entity.name} {{"]
        for f in entity.fields:
            ts_type = self._map_ts(f.type if hasattr(f, "type") else "string")
            optional = "?" if not getattr(f, "required", False) else ""
            lines.append(f"  {f.name}{optional}: {ts_type};")
        lines.append("}")
        return "\n".join(lines)

    def _to_sql(self, entity) -> str:
        lines = [f"CREATE TABLE {entity.name.lower()} ("]
        cols = []
        for f in entity.fields:
            sql_type = self._map_sql(f.type if hasattr(f, "type") else "string")
            constraints = []
            if getattr(f, "required", False):
                constraints.append("NOT NULL")
            if getattr(f, "unique", False):
                constraints.append("UNIQUE")
            cols.append(f"    {f.name} {sql_type} {' '.join(constraints)}")
        cols.append(f"    id SERIAL PRIMARY KEY")
        lines.append(",\n".join(cols))
        lines.append(");")
        return "\n".join(lines)

    def _to_graphql(self, entity) -> str:
        lines = [f"type {entity.name} {{"]
        for f in entity.fields:
            gql_type = self._map_gql(f.type if hasattr(f, "type") else "string")
            required = "!" if getattr(f, "required", False) else ""
            lines.append(f"  {f.name}: {gql_type}{required}")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _map_py(ft):
        return {
            "string": "str",
            "integer": "int",
            "float": "float",
            "boolean": "bool",
            "date": "date",
            "email": "str",
            "json": "dict",
            "uuid": "str",
        }.get(str(ft), "str")

    @staticmethod
    def _map_ts(ft):
        return {
            "string": "string",
            "integer": "number",
            "float": "number",
            "boolean": "boolean",
            "date": "string",
            "email": "string",
            "json": "Record<string, any>",
            "uuid": "string",
        }.get(str(ft), "string")

    @staticmethod
    def _map_sql(ft):
        return {
            "string": "TEXT",
            "integer": "INTEGER",
            "float": "REAL",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "email": "TEXT",
            "json": "JSONB",
            "uuid": "UUID",
        }.get(str(ft), "TEXT")

    @staticmethod
    def _map_gql(ft):
        return {
            "string": "String",
            "integer": "Int",
            "float": "Float",
            "boolean": "Boolean",
            "date": "String",
            "email": "String",
            "json": "JSON",
            "uuid": "ID",
        }.get(str(ft), "String")
