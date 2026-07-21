"""Unit tests for TaskType -> OpenRouter schema resolution.

Regression coverage for two bugs that left the structured-output (and therefore
the response-healing) path unreachable:

  Bug #1: the registry is keyed by enum *name* ("DATA_EXTRACT") but the live
          caller looked it up by enum *value* ("data_extraction"), so lookups
          returned None and ``response_format`` was never attached.
  Bug #2: the emitted schema used ``strict: True``, which OpenAI/Azure reject
          for these models because the schemas contain free-form ``dict``
          fields. Non-strict json_schema is the lenient path response-healing
          is meant to repair.
"""

import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import TaskType
from orchestrator.task_schemas import (
    generate_openrouter_schema,
    get_schema_for_task_type,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("key", ["DATA_EXTRACT", "data_extraction", TaskType.DATA_EXTRACT])
def test_resolves_by_name_value_and_enum(key):
    """Lookup must work whether given the enum name, value, or the enum itself."""
    assert get_schema_for_task_type(key) is not None
    schema = generate_openrouter_schema(key)
    assert schema is not None
    assert schema["type"] == "json_schema"


def test_every_task_type_value_resolves():
    """The live caller passes enum values; all schema-backed types must resolve."""
    schema_backed = {
        TaskType.CODE_GEN,
        TaskType.CODE_REVIEW,
        TaskType.REASONING,
        TaskType.WRITING,
        TaskType.DATA_EXTRACT,
        TaskType.SUMMARIZE,
        TaskType.EVALUATE,
    }
    for t in schema_backed:
        assert generate_openrouter_schema(t.value) is not None, t.value


def test_schema_is_non_strict():
    """strict must be False — these schemas use free-form dicts strict mode forbids."""
    schema = generate_openrouter_schema(TaskType.DATA_EXTRACT)
    assert schema["json_schema"]["strict"] is False


def test_schema_name_uses_canonical_key():
    schema = generate_openrouter_schema(TaskType.DATA_EXTRACT.value)
    assert schema["json_schema"]["name"] == "data_extract_output"


def test_unknown_task_type_returns_none():
    assert get_schema_for_task_type("nonsense") is None
    assert generate_openrouter_schema("nonsense") is None
    # IMAGE_GEN has no output schema registered.
    assert generate_openrouter_schema(TaskType.IMAGE_GEN) is None
