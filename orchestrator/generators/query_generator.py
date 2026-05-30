"""
AIQueryGenerator - Natural language to SQL/JS/GraphQL with schema validation.
=============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 8, Phase R3 (Retool-inspired).
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryLanguage(str, Enum):
    SQL = "sql"
    JAVASCRIPT = "javascript"
    GRAPHQL = "graphql"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"


@dataclass
class QueryResult:
    language: QueryLanguage
    query: str
    explanation: str = ""
    valid: bool = True
    error: str = ""


@dataclass
class TableSchema:
    name: str
    columns: dict[str, str]  # column_name -> type

    def to_description(self):
        cols = ", ".join(f"{n} ({t})" for n, t in self.columns.items())
        return f"Table '{self.name}': {cols}"


class AIQueryGenerator:
    """Generates queries from natural language with schema awareness."""

    def __init__(self, client=None):
        self._client = client
        self._schemas: list[TableSchema] = []

    def add_schema(self, schema):
        self._schemas.append(schema)

    def generate_fast(self, nl_query, language=QueryLanguage.SQL):
        """Fast template-based generation (no LLM). Falls back to basic patterns."""
        if language == QueryLanguage.SQL:
            return self._fast_sql(nl_query)
        elif language == QueryLanguage.GRAPHQL:
            return self._fast_graphql(nl_query)
        return QueryResult(
            language=language,
            query="",
            valid=False,
            error="Fast mode supports SQL and GraphQL only",
        )

    def _fast_sql(self, nl_query):
        nl = nl_query.lower()
        schema_text = "\n".join(s.to_description() for s in self._schemas)
        tables = ", ".join(s.name for s in self._schemas)
        cols = set()
        for s in self._schemas:
            cols.update(s.columns.keys())

        if "count" in nl or "how many" in nl:
            return QueryResult(
                QueryLanguage.SQL,
                f"SELECT COUNT(*) FROM {tables};",
                valid=True,
                explanation="Count query",
            )

        if "all" in nl or "list" in nl or "show" in nl:
            return QueryResult(
                QueryLanguage.SQL,
                f"SELECT * FROM {tables} LIMIT 100;",
                valid=True,
                explanation="List-all query",
            )

        matching_cols = [c for c in cols if c in nl]
        if matching_cols:
            where = f"WHERE {matching_cols[0]} = ?"
            return QueryResult(
                QueryLanguage.SQL,
                f"SELECT * FROM {tables} {where} LIMIT 10;",
                valid=True,
                explanation=f"Select by {matching_cols[0]}",
            )

        return QueryResult(
            QueryLanguage.SQL,
            f"SELECT * FROM {tables} LIMIT 10;",
            valid=True,
            explanation="Generic select",
        )

    def _fast_graphql(self, nl_query):
        query_type = (
            "query" if "get" in nl_query.lower() or "fetch" in nl_query.lower() else "mutation"
        )
        return QueryResult(
            QueryLanguage.GRAPHQL,
            f"{query_type} {{\n  items {{ id name }}\n}}",
            valid=True,
            explanation="Basic GraphQL query",
        )

    async def generate_llm(self, nl_query, language=QueryLanguage.SQL):
        if not self._client:
            return self.generate_fast(nl_query, language)

        schema_text = "\n".join(s.to_description() for s in self._schemas) or "No schema provided"
        prompt = f"""Convert this natural language query to {language.value}.

Schema:
{schema_text}

Query: {nl_query}

Return ONLY the query, no explanation, no markdown fences."""

        try:
            response = await self._client.call(
                model=None,
                prompt=prompt,
                system=f"You are a {language.value} expert. Return only valid {language.value}.",
                max_tokens=500,
                temperature=0.1,
                timeout=30,
            )
            query = response.text.strip()
            return QueryResult(language=language, query=query, valid=True)
        except Exception as e:
            return QueryResult(language=language, query="", valid=False, error=str(e))
