"""
JSON Schema Verifier — validates that response contains valid JSON
matching a given schema.

Uses the ``jsonschema`` library when available; falls back to
stdlib ``json.loads`` for basic structural checks when the library
is absent (graceful degradation).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from orchestrator.models import TaskType, Verdict

logger = logging.getLogger(__name__)

# Lazy-loaded optional dependency
_JSONSCHEMA: Any | None = None


def _get_validator(schema: dict[str, Any]) -> Any | None:
    """Lazy-import jsonschema; return None if not installed."""
    global _JSONSCHEMA
    if _JSONSCHEMA is None:
        try:
            import jsonschema

            _JSONSCHEMA = jsonschema
        except ImportError:
            logger.info("jsonschema not installed — falling back to json.loads only")
            _JSONSCHEMA = False  # sentinel: checked once
    return _JSONSCHEMA if _JSONSCHEMA is not False else None


def _detect_schema(prompt: str) -> dict[str, Any] | None:
    """Simple heuristics to guess a JSON schema from the prompt.

    Looks for inline JSON Schema snippets or ``response_format``
    hints.  Returns ``None`` when no schema can be inferred
    (verifier falls back to basic JSON-parse only).
    """
    # Check for explicit "json_schema" or "response_format" keywords
    # in the prompt — a basic signal that structured output was
    # requested.
    markers = [
        '"type": "object"',
        "'type': 'object'",
        '"properties"',
        "'properties'",
        "json_schema",
        "response_format",
    ]
    if any(marker in prompt for marker in markers):
        # Attempt a crude JSON object extraction from the prompt
        # to find an inline schema.
        try:
            import re

            for match in re.finditer(r"\{[^{}]*\}", prompt, re.DOTALL):
                candidate = match.group()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "type" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception:
            pass
    return None


class JSONSchemaVerifier:
    """Verify that response is valid JSON (optionally against a schema).

    Usage:
        verifier = JSONSchemaVerifier()
        verdict = await verifier.verify(
            prompt="Return JSON with name and age",
            response='{"name": "Alice", "age": 30}',
            task_type=TaskType.DATA_EXTRACT,
        )
    """

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        """Initialize verifier.

        Args:
            schema: Optional JSON schema to validate against.
                If ``None``, only basic JSON-parse validation is performed.
        """
        self._schema = schema

    async def verify(
        self,
        *,
        prompt: str,
        response: str,
        task_type: TaskType,
    ) -> Verdict:
        """Validate *response* as JSON."""
        # Detect schema from prompt if none was explicitly provided
        schema = self._schema or _detect_schema(prompt)

        signals: list[str] = []

        # Phase 1: basic parse
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            return Verdict(
                passed=False,
                score=0.0,
                signals=("json_parse_failed",),
                detail=f"Invalid JSON: {exc.msg}",
            )

        signals.append("json_valid")

        # Phase 2: validate against schema (if available)
        if schema is not None:
            validator = _get_validator(schema)
            if validator is not None:
                try:
                    validator.validate(parsed, schema)
                    signals.append("json_schema_valid")
                except validator.exceptions.ValidationError as exc:
                    return Verdict(
                        passed=False,
                        score=0.0,
                        signals=tuple(signals) + ("json_schema_failed",),
                        detail=f"Schema validation failed: {exc.message}",
                    )
            else:
                # Without the library, fall back to basic structural checks
                if "properties" in schema:
                    required = schema.get("required", [])
                    missing = [k for k in required if k not in parsed]
                    if missing:
                        return Verdict(
                            passed=False,
                            score=0.0,
                            signals=tuple(signals) + ("json_schema_failed",),
                            detail=f"Missing required fields: {missing}",
                        )
                    signals.append("json_schema_valid")

        # Pass — compute a quality score based on structural depth
        score = 0.8
        if isinstance(parsed, dict) and len(parsed) > 0:
            score = min(1.0, 0.8 + 0.05 * min(len(parsed), 4))

        return Verdict(
            passed=True,
            score=score,
            signals=tuple(signals),
            detail="JSON is valid",
        )
