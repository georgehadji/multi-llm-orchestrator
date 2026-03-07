#!/usr/bin/env python3
"""Test script to verify all models are properly configured."""

from orchestrator.models import Model, COST_TABLE, ROUTING_TABLE, get_provider, TaskType

def main():
    print("=" * 60)
    print("TESTING ORCHESTRATOR MODELS")
    print("=" * 60)
    
    # Test that all models are defined
    print("\n1. Testing Model enum...")
    models = list(Model)
    print(f"   Total models: {len(models)}")
    
    # Test that all models have costs
    print("\n2. Testing COST_TABLE...")
    models_without_cost = [m for m in models if m not in COST_TABLE]
    if models_without_cost:
        print(f"   ❌ Models without cost: {models_without_cost}")
        return False
    else:
        print(f"   ✅ All {len(models)} models have cost information")
    
    # Test get_provider
    print("\n3. Testing get_provider...")
    providers = set()
    for m in models:
        providers.add(get_provider(m))
    print(f"   Providers: {sorted(providers)}")
    
    # Count models per provider
    from collections import Counter
    provider_counts = Counter(get_provider(m) for m in models)
    print("\n   Models per provider:")
    for provider, count in sorted(provider_counts.items()):
        print(f"     {provider}: {count}")
    
    # Test ROUTING_TABLE
    print("\n4. Testing ROUTING_TABLE...")
    for task_type in ROUTING_TABLE:
        model_count = len(ROUTING_TABLE[task_type])
        print(f"   {task_type.value}: {model_count} models")
    
    # Test api_clients imports
    print("\n5. Testing api_clients...")
    try:
        from orchestrator.api_clients import UnifiedClient
        print("   ✅ UnifiedClient imports successfully")
    except Exception as e:
        print(f"   ❌ Error importing UnifiedClient: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
