"""
TDD (Test-First Generation) Model Configuration
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Date: 2026-03-30

Optimal model selection for each TDD phase with quality tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TDDModelConfig:
    """
    Model configuration for TDD phases.

    Three quality tiers available:
    - budget: Cost-effective models for simple tasks
    - balanced: Best value/quality ratio (default)
    - premium: Maximum quality for critical tasks

    Usage:
        config = TDDModelConfig()
        test_model = config.get_model("test_generation", tier="balanced")
    """

    # ═══════════════════════════════════════════════════════
    # Balanced Tier (Default) - Best Value/Quality Ratio
    # ═══════════════════════════════════════════════════════

    # Test Generation: Best test design capability
    test_generation: str = "anthropic/claude-sonnet-4-6"  # $3.00/$15.00
    # Implementation: Cost-effective coding
    implementation: str = "qwen/qwen-3-coder-next"  # $0.12/$0.75
    # Test Review: Best analysis capability
    test_review: str = "anthropic/claude-sonnet-4-6"  # $3.00/$15.00
    # Refactoring: Cost-effective improvements
    refactoring: str = "qwen/qwen-3-coder-next"  # $0.12/$0.75

    # ═══════════════════════════════════════════════════════
    # Budget Tier - Best value with good test quality
    # DeepSeek V3 is used for test generation/review as it has strong
    # instruction-following and test writing ability at low cost.
    # ═══════════════════════════════════════════════════════

    budget_test_generation: str = "deepseek/deepseek-v3.2"  # $0.27/$1.10 — better at tests than qwen-coder
    budget_implementation: str = "qwen/qwen-3-coder-next"  # $0.12/$0.75
    budget_test_review: str = "deepseek/deepseek-v3.2"  # $0.27/$1.10
    budget_refactoring: str = "qwen/qwen-3-coder-next"  # $0.12/$0.75

    # ═══════════════════════════════════════════════════════
    # Premium Tier - Maximum Quality
    # ═══════════════════════════════════════════════════════

    premium_test_generation: str = "openai/gpt-5.4-pro"  # $30.00/$180.00
    premium_implementation: str = "openai/gpt-5.4-pro"  # $30.00/$180.00
    premium_test_review: str = "openai/gpt-5.4-pro"  # $30.00/$180.00
    premium_refactoring: str = "openai/gpt-5.4-pro"  # $30.00/$180.00

    # ═══════════════════════════════════════════════════════
    # Framework-Specific Overrides (Optional)
    # ═══════════════════════════════════════════════════════

    # Python + pytest (most common)
    python_test_generation: str | None = None  # Use default if None
    python_implementation: str | None = None

    # JavaScript + Jest
    javascript_test_generation: str | None = None
    javascript_implementation: str | None = None

    # TypeScript + Vitest
    typescript_test_generation: str | None = None
    typescript_implementation: str | None = None

    # Go + testing
    go_test_generation: str | None = None
    go_implementation: str | None = None

    # Rust + cargo test
    rust_test_generation: str | None = None
    rust_implementation: str | None = None

    def get_model(self, phase: str, tier: str = "balanced", language: str | None = None) -> str:
        """
        Get optimal model for a TDD phase.

        Args:
            phase: TDD phase name (test_generation, implementation, test_review, refactoring)
            tier: Quality tier (budget, balanced, premium)
            language: Optional language-specific override (python, javascript, typescript, go, rust)

        Returns:
            Model ID string (e.g., "anthropic/claude-sonnet-4-6")

        Examples:
            >>> config = TDDModelConfig()
            >>> config.get_model("test_generation", "balanced")
            'anthropic/claude-sonnet-4-6'
            >>> config.get_model("implementation", "budget")
            'qwen/qwen-3-coder-next'
            >>> config.get_model("test_generation", "balanced", "python")
            'anthropic/claude-sonnet-4-6'  # or language-specific if configured
        """
        # Check for language-specific override first
        if language:
            lang_phase_key = f"{language}_{phase}"
            lang_model = getattr(self, lang_phase_key, None)
            if lang_model is not None:
                return lang_model

        # Get model based on tier
        if tier == "budget":
            key = f"budget_{phase}"
        elif tier == "premium":
            key = f"premium_{phase}"
        else:  # balanced (default)
            key = phase

        model = getattr(self, key, None)
        if model is None:
            # Fallback to balanced tier default
            model = getattr(self, phase, "qwen/qwen-3-coder-next")

        return model

    def get_all_models(self, tier: str = "balanced") -> dict[str, str]:
        """
        Get all models for a quality tier.

        Args:
            tier: Quality tier (budget, balanced, premium)

        Returns:
            Dictionary mapping phase names to model IDs
        """
        phases = ["test_generation", "implementation", "test_review", "refactoring"]

        if tier == "budget":
            prefix = "budget_"
        elif tier == "premium":
            prefix = "premium_"
        else:
            prefix = ""

        return {phase: self.get_model(phase, tier) for phase in phases}

    def estimate_cost(self, tier: str = "balanced") -> dict[str, float]:
        """
        Estimate TDD cost per phase (per 1M tokens).

        Args:
            tier: Quality tier

        Returns:
            Dictionary with cost estimates per phase and total
        """
        from .models import COST_TABLE, Model

        models = self.get_all_models(tier)
        costs = {}

        for phase, model_id in models.items():
            try:
                model = Model(model_id)
                cost_data = COST_TABLE.get(model, {"input": 0.0, "output": 0.0})
                # Average of input/output costs for estimation
                costs[f"{phase}_cost"] = (cost_data["input"] + cost_data["output"]) / 2
            except (ValueError, KeyError):
                costs[f"{phase}_cost"] = 0.0

        # Estimate total cost for typical TDD task
        # Test gen: 2K tokens, Implementation: 5K tokens, Review: 1K tokens
        costs["estimated_total_per_task"] = (
            costs.get("test_generation_cost", 0.0) * 0.002
            + costs.get("implementation_cost", 0.0) * 0.005
            + costs.get("test_review_cost", 0.0) * 0.001
        )

        return costs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_generation": self.test_generation,
            "implementation": self.implementation,
            "test_review": self.test_review,
            "refactoring": self.refactoring,
            "budget_test_generation": self.budget_test_generation,
            "budget_implementation": self.budget_implementation,
            "budget_test_review": self.budget_test_review,
            "budget_refactoring": self.budget_refactoring,
            "premium_test_generation": self.premium_test_generation,
            "premium_implementation": self.premium_implementation,
            "premium_test_review": self.premium_test_review,
            "premium_refactoring": self.premium_refactoring,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TDDModelConfig:
        """Create from dictionary."""
        return cls(**data)


# ═══════════════════════════════════════════════════════
# Pre-configured TDD Profiles
# ═══════════════════════════════════════════════════════

# Budget Profile: Best value — DeepSeek for tests (strong test-writing), Qwen for implementation
TDD_BUDGET_PROFILE = TDDModelConfig(
    test_generation="deepseek/deepseek-v3.2",  # $0.27/$1.10 — strong test writer
    implementation="qwen/qwen-3-coder-next",   # $0.12/$0.75 — fast coder
    test_review="deepseek/deepseek-v3.2",      # $0.27/$1.10 — good analyser
    refactoring="qwen/qwen-3-coder-next",      # $0.12/$0.75
)

# Balanced Profile: Best value (default)
TDD_BALANCED_PROFILE = TDDModelConfig(
    test_generation="anthropic/claude-sonnet-4-6",  # $3.00/$15.00
    implementation="qwen/qwen-3-coder-next",  # $0.12/$0.75
    test_review="anthropic/claude-sonnet-4-6",  # $3.00/$15.00
    refactoring="qwen/qwen-3-coder-next",  # $0.12/$0.75
)

# Premium Profile: Maximum quality
TDD_PREMIUM_PROFILE = TDDModelConfig(
    test_generation="openai/gpt-5.4-pro",  # $30.00/$180.00
    implementation="openai/gpt-5.4-pro",  # $30.00/$180.00
    test_review="openai/gpt-5.4-pro",  # $30.00/$180.00
    refactoring="openai/gpt-5.4-pro",  # $30.00/$180.00
)

# Python-Specialized Profile (pytest)
TDD_PYTHON_PROFILE = TDDModelConfig(
    test_generation="anthropic/claude-sonnet-4-6",  # Best pytest knowledge
    implementation="qwen/qwen-3-coder-next",  # Cost-effective
    test_review="anthropic/claude-sonnet-4-6",  # Best test analysis
    refactoring="qwen/qwen-3-coder-next",
    python_test_generation="anthropic/claude-sonnet-4-6",
    python_implementation="qwen/qwen-3-coder-next",
)

# JavaScript-Specialized Profile (Jest)
TDD_JAVASCRIPT_PROFILE = TDDModelConfig(
    test_generation="anthropic/claude-sonnet-4-6",  # Best Jest knowledge
    implementation="qwen/qwen-3-coder-next",  # Cost-effective
    test_review="anthropic/claude-sonnet-4-6",  # Best test analysis
    refactoring="qwen/qwen-3-coder-next",
    javascript_test_generation="anthropic/claude-sonnet-4-6",
    javascript_implementation="qwen/qwen-3-coder-next",
)


# ═══════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════


def get_tdd_profile(tier: str = "balanced", language: str | None = None) -> TDDModelConfig:
    """
    Get pre-configured TDD profile.

    Args:
        tier: Quality tier (budget, balanced, premium)
        language: Optional language-specific profile (python, javascript, typescript, go, rust)

    Returns:
        TDDModelConfig instance

    Examples:
        >>> config = get_tdd_profile("balanced")
        >>> config = get_tdd_profile("budget")
        >>> config = get_tdd_profile("balanced", "python")
    """
    # Language-specific profiles take precedence
    if language:
        lang_profiles = {
            "python": TDD_PYTHON_PROFILE,
            "javascript": TDD_JAVASCRIPT_PROFILE,
        }
        if language.lower() in lang_profiles:
            return lang_profiles[language.lower()]

    # Tier-based profiles
    profiles = {
        "budget": TDD_BUDGET_PROFILE,
        "balanced": TDD_BALANCED_PROFILE,
        "premium": TDD_PREMIUM_PROFILE,
    }

    return profiles.get(tier.lower(), TDD_BALANCED_PROFILE)


def estimate_tdd_cost(tier: str = "balanced") -> dict[str, Any]:
    """
    Estimate TDD cost for different quality tiers.

    Args:
        tier: Quality tier

    Returns:
        Dictionary with cost estimates

    Examples:
        >>> costs = estimate_tdd_cost("balanced")
        >>> print(f"Estimated cost per task: ${costs['estimated_total_per_task']:.4f}")
    """
    config = get_tdd_profile(tier)
    costs = config.estimate_cost(tier)
    costs["tier"] = tier
    return costs


# ═══════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════


def example():
    """Example usage of TDD model configuration."""
    print("=" * 70)
    print("TDD Model Configuration - Examples")
    print("=" * 70)

    # Example 1: Get models for balanced tier
    print("\n1. Balanced Tier Models:")
    config = get_tdd_profile("balanced")
    for phase, model in config.get_all_models("balanced").items():
        print(f"   {phase:20} → {model}")

    # Example 2: Get models for budget tier
    print("\n2. Budget Tier Models:")
    config = get_tdd_profile("budget")
    for phase, model in config.get_all_models("budget").items():
        print(f"   {phase:20} → {model}")

    # Example 3: Cost estimation
    print("\n3. Cost Estimates (per task):")
    for tier in ["budget", "balanced", "premium"]:
        costs = estimate_tdd_cost(tier)
        print(f"   {tier:10}: ${costs['estimated_total_per_task']:.4f}")

    # Example 4: Language-specific profile
    print("\n4. Python-Specific Profile:")
    config = get_tdd_profile("balanced", "python")
    print(f"   Test Generation: {config.get_model('test_generation', 'balanced', 'python')}")
    print(f"   Implementation:  {config.get_model('implementation', 'balanced', 'python')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    example()
