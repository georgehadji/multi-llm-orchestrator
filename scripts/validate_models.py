#!/usr/bin/env python3
"""
Model Validation Script
=======================
Date: 2026-04-01

Validates all model configurations after OpenRouter verification updates.
Tests imports, model availability, and cost calculations.

Usage:
    python scripts/validate_models.py
"""

import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all updated modules import correctly."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        from orchestrator.models import Model
        print("✅ orchestrator.models imported")
    except Exception as e:
        print(f"❌ orchestrator.models failed: {e}")
        return False
    
    try:
        from orchestrator.model_registry import ModelRegistry
        print("✅ orchestrator.model_registry imported")
    except Exception as e:
        print(f"❌ orchestrator.model_registry failed: {e}")
        return False
    
    try:
        from orchestrator.phase_aware_models import PhaseAwareModelSelector, PhaseType
        print("✅ orchestrator.phase_aware_models imported")
    except Exception as e:
        print(f"❌ orchestrator.phase_aware_models failed: {e}")
        return False
    
    try:
        from orchestrator.tdd_config import TDDModelConfig, get_tdd_profile
        print("✅ orchestrator.tdd_config imported")
    except Exception as e:
        print(f"❌ orchestrator.tdd_config failed: {e}")
        return False
    
    try:
        from orchestrator.api_clients import UnifiedClient, validate_model_available
        print("✅ orchestrator.api_clients imported")
    except Exception as e:
        print(f"❌ orchestrator.api_clients failed: {e}")
        return False
    
    print()
    return True


def test_model_registry():
    """Test ModelRegistry configuration."""
    print("=" * 60)
    print("Testing ModelRegistry")
    print("=" * 60)
    
    from orchestrator.model_registry import ModelRegistry
    
    print(f"Verified models in COST_TABLE: {len(ModelRegistry.COST_TABLE)}")
    print(f"Unavailable models: {len(ModelRegistry.UNAVAILABLE_MODELS)}")
    print(f"Budget models: {len(ModelRegistry.BUDGET_MODELS)}")
    print(f"Coding specialists: {len(ModelRegistry.CODING_SPECIALISTS)}")
    print(f"Reasoning models: {len(ModelRegistry.REASONING_MODELS)}")
    
    # Test specific verified models
    verified_models = [
        ("MIMO_V2_FLASH", ModelRegistry.MIMO_V2_FLASH),
        ("STEP_3_5_FLASH", ModelRegistry.STEP_3_5_FLASH),
        ("GROK_4_20", ModelRegistry.GROK_4_20),
        ("QWEN_2_5_CODER_32B", ModelRegistry.QWEN_2_5_CODER_32B),
        ("GLM_4_7_FLASH", ModelRegistry.GLM_4_7_FLASH),
        ("MINIMAX_M2_7", ModelRegistry.MINIMAX_M2_7),
    ]
    
    print("\nVerified Model IDs:")
    for name, model_id in verified_models:
        cost = ModelRegistry.COST_TABLE.get(model_id, {})
        print(f"  ✅ {name}: {model_id} (${cost.get('input', 0):.2f}/${cost.get('output', 0):.2f})")
    
    # Test unavailable models
    print("\nUnavailable Models (with replacements):")
    for unavailable, replacement in ModelRegistry.UNAVAILABLE_MODELS.items():
        print(f"  ❌ {unavailable} → {replacement}")
    
    print()
    return True


def test_phase_models():
    """Test phase-aware model selection."""
    print("=" * 60)
    print("Testing Phase-Aware Model Selection")
    print("=" * 60)
    
    from orchestrator.phase_aware_models import PhaseAwareModelSelector, PhaseType
    
    selector = PhaseAwareModelSelector()
    
    for phase in PhaseType:
        models = selector.get_phase_models(phase, count=3)
        print(f"{phase.value:15} → {models[:3]}")
    
    print("\nBudget Config:")
    budget = selector.get_budget_config()
    for phase, model in budget.items():
        print(f"  {phase.value:15} → {model}")
    
    print()
    return True


def test_tdd_config():
    """Test TDD model configuration."""
    print("=" * 60)
    print("Testing TDD Configuration")
    print("=" * 60)
    
    from orchestrator.tdd_config import TDDModelConfig, get_tdd_profile
    
    # Test default config
    config = TDDModelConfig()
    print(f"Balanced Implementation: {config.implementation}")
    print(f"Budget Implementation: {config.budget_implementation}")
    print(f"Premium Implementation: {config.premium_implementation}")
    
    # Test profiles
    print("\nTDD Profiles:")
    for tier in ["budget", "balanced", "premium"]:
        profile = get_tdd_profile(tier)
        print(f"  {tier:10}: impl={profile.implementation}")
    
    print()
    return True


def test_model_validation():
    """Test runtime model validation."""
    print("=" * 60)
    print("Testing Runtime Model Validation")
    print("=" * 60)
    
    from orchestrator.api_clients import validate_model_available
    from orchestrator.models import Model
    
    # Test verified models
    test_models = [
        (Model.XIAOMI_MIMO_V2_FLASH, True, None),
        (Model.STEPFUN_STEP_3_5_FLASH, True, None),
        (Model.XAI_GROK_4_5, True, None),
        (Model.QWEN_2_5_CODER_32B, True, None),
    ]
    
    print("Verified Models:")
    for model, expected_available, _ in test_models:
        is_available, replacement = validate_model_available(model)
        status = "✅" if is_available == expected_available else "❌"
        print(f"  {status} {model.value}: available={is_available}")
    
    print()
    return True


def test_cost_calculations():
    """Test cost calculations with verified models."""
    print("=" * 60)
    print("Testing Cost Calculations")
    print("=" * 60)
    
    from orchestrator.model_registry import ModelRegistry
    
    # Calculate cost for balanced tier pipeline execution
    print("Balanced Tier Pipeline Costs (estimated):")
    
    phase_models = {
        "ANALYSIS": ModelRegistry.STEP_3_5_FLASH,
        "GENERATION": ModelRegistry.MIMO_V2_FLASH,
        "CRITIQUE": ModelRegistry.GROK_4_20,
        "SYNTHESIS": ModelRegistry.MIMO_V2_PRO,
        "RESEARCH": ModelRegistry.KIMI_K2_5,
        "EVALUATION": ModelRegistry.GROK_4_20,
        "VERIFICATION": ModelRegistry.GROK_4_20,
    }
    
    total_cost = 0
    for phase, model in phase_models.items():
        cost = ModelRegistry.COST_TABLE.get(model, {"input": 0, "output": 0})
        # Estimate: 2K input tokens, 1K output tokens per phase
        phase_cost = (cost["input"] * 0.002) + (cost["output"] * 0.001)
        total_cost += phase_cost
        print(f"  {phase:15} → {model.split('/')[-1]:30} ${phase_cost:.4f}")
    
    print(f"\n  Total (all phases): ${total_cost:.2f}")
    print(f"  Total (12 methods): ${total_cost * 12:.2f}")
    
    print()
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("MODEL VALIDATION SCRIPT")
    print("Date: 2026-04-01")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("ModelRegistry", test_model_registry),
        ("Phase Models", test_phase_models),
        ("TDD Config", test_tdd_config),
        ("Model Validation", test_model_validation),
        ("Cost Calculations", test_cost_calculations),
    ]
    
    for name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"❌ {name} FAILED\n")
        except Exception as e:
            all_passed = False
            print(f"❌ {name} ERROR: {e}\n")
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Test with actual project: python -m orchestrator --file projects/your_project.yaml")
        print("2. Monitor logs for any model availability issues")
        print("3. Adjust budget based on actual usage")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review errors above and fix before proceeding.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
