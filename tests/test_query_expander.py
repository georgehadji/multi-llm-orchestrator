"""Tests for QueryExpander — LLM-based query expansion."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator.query_expander import QueryExpander


@pytest.mark.asyncio
async def test_expand_returns_list_including_original():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = ["find python code", "search python examples", "python source code"]
        result = await expander.expand("python code")
    assert "python code" in result  # original always included
    assert isinstance(result, list)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_expand_falls_back_on_llm_error():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = Exception("LLM unavailable")
        result = await expander.expand("python code")
    assert result == ["python code"]  # graceful fallback = original only


@pytest.mark.asyncio
async def test_expand_deduplicates_and_preserves_original():
    expander = QueryExpander()
    with patch.object(expander, "_call_llm", new_callable=AsyncMock) as mock_llm:
        # LLM returns original as one of its variants — should not duplicate
        mock_llm.return_value = ["python code", "python snippets"]
        result = await expander.expand("python code")
    assert result.count("python code") == 1  # deduplicated
    assert "python snippets" in result


@pytest.mark.asyncio
async def test_call_llm_uses_client_call_not_chat_completion():
    """_call_llm must use UnifiedClient.call(), not the non-existent chat_completion()."""
    from unittest.mock import patch as _patch

    expander = QueryExpander(model="deepseek-chat")
    mock_response = MagicMock()
    mock_response.text = '["variant one", "variant two"]'

    with _patch("orchestrator.api_clients.UnifiedClient") as MockClient:
        instance = MockClient.return_value
        instance.call = AsyncMock(return_value=mock_response)
        result = await expander._call_llm("test query")

    instance.call.assert_called_once()
    assert "variant one" in result
    assert "variant two" in result
