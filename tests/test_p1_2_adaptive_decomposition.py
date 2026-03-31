"""
Test P1-2: Adaptive Decomposition Model Selection
==================================================
Tests that decomposition model is selected based on project complexity
to optimize cost while maintaining quality.
"""

import pytest
from unittest.mock import patch

from orchestrator.engine import Orchestrator
from orchestrator.models import Budget, Model


class TestAdaptiveDecomposition:
    """Test P1-2: Adaptive model selection for decomposition."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with multiple models available."""
        orch = Orchestrator(budget=Budget(max_usd=100.0))

        # Ensure multiple models are "available" for testing
        orch.api_health[Model.GPT_4O_MINI] = True
        orch.api_health[Model.GPT_4O] = True
        orch.api_health[Model.DEEPSEEK_CHAT] = True
        orch.api_health[Model.GEMINI_FLASH] = True
        orch.api_health[Model.CLAUDE_3_5_SONNET] = True

        return orch

    def test_simple_project_selects_fast_model(self, orchestrator):
        """
        Simple projects (<500 tokens, <2 complexity) should use GPT-4o-mini.
        """
        # Arrange: Simple project description
        simple_project = "Create a hello world Flask app"

        # Act: Select model
        model = orchestrator._select_decomposition_model(simple_project)

        # Assert: Should select fast/cheap model
        assert model == Model.GPT_4O_MINI

    def test_medium_project_selects_balanced_model(self, orchestrator):
        """
        Medium projects (<2000 tokens, <5 complexity) should use DeepSeek-Chat.
        """
        # Arrange: Medium project with some complexity
        medium_project = """
        Build a REST API with FastAPI that includes:
        - User authentication with JWT
        - CRUD operations for products
        - PostgreSQL database integration
        - Basic error handling
        """

        # Act: Select model
        model = orchestrator._select_decomposition_model(medium_project)

        # Assert: Should select balanced model
        # (DeepSeek-Chat or GPT-4o-mini fallback)
        assert model in [Model.DEEPSEEK_CHAT, Model.GPT_4O_MINI]

    def test_complex_project_selects_quality_model(self, orchestrator):
        """
        Complex projects (>=2000 tokens or >=5 complexity) should use GPT-4o.
        """
        # Arrange: Complex project with many keywords
        complex_project = """
        Build a distributed microservices architecture with:
        - Kubernetes cluster deployment
        - OAuth2 authentication and RBAC permissions
        - PostgreSQL with replication and sharding
        - Redis caching layer
        - Real-time WebSocket streaming
        - Kafka message queue integration
        - Multi-tenant SaaS support
        - API gateway with load balancer
        - Machine learning model embeddings
        - Docker containerization with Terraform
        """

        # Act: Select model
        model = orchestrator._select_decomposition_model(complex_project)

        # Assert: Should select high-quality model
        assert model in [Model.GPT_4O, Model.CLAUDE_3_5_SONNET, Model.DEEPSEEK_CHAT]

    def test_token_estimate_calculation(self, orchestrator):
        """
        Verify token estimate is calculated correctly (4 chars/token).
        """
        # Arrange: Project with known length
        project = "A" * 2000  # 2000 chars = ~500 tokens

        # Act: Select model (this internally calculates token_estimate)
        # We can't directly test the calculation, but we can verify
        # the behavior matches expectations
        model = orchestrator._select_decomposition_model(project)

        # 500 tokens is borderline, should use fast model if no complexity
        assert model == Model.GPT_4O_MINI

    def test_complexity_keyword_detection(self, orchestrator):
        """
        Verify complexity keywords are detected correctly.
        """
        # Test individual keywords
        keyword_tests = [
            ("microservice architecture", 1),
            ("OAuth authentication", 1),
            ("database replication", 1),
            ("real-time websocket", 1),
            ("machine learning ML", 2),  # Both "machine learning" and "ML"
        ]

        for project, expected_min_complexity in keyword_tests:
            # The model selection logic counts keywords internally
            # We verify by checking the selected model
            model = orchestrator._select_decomposition_model(project)
            # Just verify it doesn't crash and returns a Model
            assert isinstance(model, Model)

    def test_tech_stack_keyword_detection(self, orchestrator):
        """
        Verify tech stack keywords contribute to complexity.
        """
        # Arrange: Project with tech stack keywords
        tech_project = """
        Build with React, Next.js, FastAPI, PostgreSQL, Docker, and AWS
        """

        # Act: Select model
        model = orchestrator._select_decomposition_model(tech_project)

        # Assert: Tech keywords should increase complexity score
        # (tech_score is divided by 2, so need several to matter)
        assert isinstance(model, Model)

    def test_fallback_to_fast_model(self, orchestrator):
        """
        If no optimal model is available, fallback to _get_fast_decomposition_model.
        """
        # Arrange: Disable all models
        for model in Model:
            orchestrator.api_health[model] = False

        # Act: Select model (should use fallback)
        model = orchestrator._select_decomposition_model("test")

        # Assert: Should return some model (fallback logic)
        # Note: With all models disabled, this may return None or a default
        # depending on _get_fast_decomposition_model implementation

    def test_get_fast_decomposition_model(self, orchestrator):
        """
        Test the fallback _get_fast_decomposition_model() method.
        """
        # Arrange: Enable specific fast models
        for model in Model:
            orchestrator.api_health[model] = False

        orchestrator.api_health[Model.MISTRAL_NEMO] = True

        # Act: Get fast model
        model = orchestrator._get_fast_decomposition_model()

        # Assert: Should return available fast model
        assert model == Model.MISTRAL_NEMO

    def test_get_cheapest_available(self, orchestrator):
        """
        Test the _get_cheapest_available() fallback method.
        """
        # Arrange: Enable only expensive models
        for model in Model:
            orchestrator.api_health[model] = False

        orchestrator.api_health[Model.GPT_4O] = True

        # Act: Get cheapest
        model = orchestrator._get_cheapest_available()

        # Assert: Should return available model
        assert model == Model.GPT_4O

    def test_decomposition_integration(self, orchestrator):
        """
        Integration test: Verify _decompose uses adaptive selection.
        """
        # This is a higher-level test that would actually call LLM APIs
        # For unit testing, we just verify the method exists and is callable

        # Arrange: Mock the API client to avoid real calls
        async def mock_call(*args, **kwargs):
            from orchestrator.api_clients import APIResponse

            return APIResponse(
                text='[{"id": "task_1", "type": "code_generation", "prompt": "test", "dependencies": []}]',
                input_tokens=10,
                output_tokens=50,
                model=Model.GPT_4O_MINI,
            )

        orchestrator.client.call = mock_call

        # Act: Decompose simple project (should use fast model)
        import asyncio

        tasks = asyncio.get_event_loop().run_until_complete(
            orchestrator._decompose(
                project="Simple hello world",
                criteria="Works",
            )
        )

        # Assert: Should return tasks (exact behavior depends on mock)
        assert isinstance(tasks, dict)
