"""
Tests for Plugin System
"""

import pytest
from dataclasses import dataclass

from orchestrator.plugins import (
    PluginRegistry,
    ValidatorPlugin,
    IntegrationPlugin,
    RouterPlugin,
    FeedbackPlugin,
    PluginMetadata,
    PluginType,
    ValidationResult,
    RoutingSuggestion,
    FeedbackPayload,
    get_plugin_registry,
    reset_plugin_registry,
)
from orchestrator.models import Model, TaskType


# Test Plugins
class TestValidator(ValidatorPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-validator",
            version="1.0.0",
            author="test",
            description="Test validator",
            plugin_type=PluginType.VALIDATOR,
        )
    
    def can_validate(self, file_path, language):
        return language == "python"
    
    def validate(self, code, context):
        return ValidationResult(passed=True, score=1.0)


class TestIntegration(IntegrationPlugin):
    def __init__(self):
        self.configured = False
    
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-integration",
            version="1.0.0",
            author="test",
            description="Test integration",
            plugin_type=PluginType.INTEGRATION,
        )
    
    def initialize(self, config):
        self.configured = config.get("enabled", False)
    
    def is_configured(self):
        return self.configured
    
    async def send_notification(self, event_type, payload):
        return True


class TestRouter(RouterPlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test-router",
            version="1.0.0",
            author="test",
            description="Test router",
            plugin_type=PluginType.ROUTER,
        )
    
    def suggest_models(self, task, available_models, context):
        return [RoutingSuggestion(
            model=Model.GPT_4O,
            confidence=0.9,
            reason="Test suggestion",
        )]


class TestPluginRegistry:
    def setup_method(self):
        reset_plugin_registry()
        self.registry = PluginRegistry()
    
    def test_register_validator(self):
        plugin = TestValidator()
        self.registry.register(plugin)
        
        assert self.registry.get("test-validator") is plugin
        assert len(self.registry.get_validators()) == 1
    
    def test_register_duplicate_ignored(self):
        plugin = TestValidator()
        self.registry.register(plugin)
        self.registry.register(plugin)  # Second register should be ignored
        
        assert len(self.registry.get_validators()) == 1
    
    def test_unregister(self):
        plugin = TestValidator()
        self.registry.register(plugin)
        self.registry.unregister("test-validator")
        
        assert self.registry.get("test-validator") is None
        assert len(self.registry.get_validators()) == 0
    
    def test_get_by_type(self):
        self.registry.register(TestValidator())
        self.registry.register(TestIntegration())
        
        validators = self.registry.get_by_type(PluginType.VALIDATOR)
        integrations = self.registry.get_by_type(PluginType.INTEGRATION)
        
        assert len(validators) == 1
        assert len(integrations) == 1
    
    def test_list_plugins(self):
        self.registry.register(TestValidator())
        self.registry.register(TestIntegration())
        
        plugins = self.registry.list_plugins()
        names = {p.name for p in plugins}
        
        assert "test-validator" in names
        assert "test-integration" in names
    
    def test_health_checks(self):
        self.registry.register(TestValidator())
        
        results = self.registry.run_health_checks()
        
        assert "test-validator" in results
        healthy, error = results["test-validator"]
        assert healthy is True
        assert error is None


class TestGlobalRegistry:
    def setup_method(self):
        reset_plugin_registry()
    
    def test_get_plugin_registry(self):
        registry = get_plugin_registry()
        assert registry is not None
        
        # Should return same instance
        registry2 = get_plugin_registry()
        assert registry is registry2


class TestValidatorPlugin:
    def test_base_metadata(self):
        validator = ValidatorPlugin()
        assert validator.metadata.plugin_type == PluginType.VALIDATOR
    
    def test_custom_validator(self):
        validator = TestValidator()
        
        assert validator.can_validate("test.py", "python") is True
        assert validator.can_validate("test.js", "javascript") is False
        
        result = validator.validate("print('hello')", {})
        assert result.passed is True
        assert result.score == 1.0


class TestIntegrationPlugin:
    @pytest.mark.asyncio
    async def test_integration_lifecycle(self):
        plugin = TestIntegration()
        
        assert plugin.is_configured() is False
        
        plugin.initialize({"enabled": True})
        assert plugin.is_configured() is True
        
        result = await plugin.send_notification("test", {})
        assert result is True


class TestRouterPlugin:
    def test_router_suggestion(self):
        router = TestRouter()
        
        from orchestrator.models import Task
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")
        
        suggestions = router.suggest_models(
            task,
            [Model.GPT_4O, Model.GPT_4O_MINI],
            {},
        )
        
        assert len(suggestions) == 1
        assert suggestions[0].model == Model.GPT_4O
        assert suggestions[0].confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
