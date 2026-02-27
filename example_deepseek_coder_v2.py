#!/usr/bin/env python3
"""
Example: Using DeepSeek-Coder for Code Generation
==================================================

DeepSeek-Coder is a code-specialized model optimized for:
- Code generation and completion
- Code review and refactoring  
- Bug fixing and debugging
- Algorithm implementation

Prerequisites:
    export DEEPSEEK_API_KEY="your-api-key"

Usage:
    python example_deepseek_coder.py
"""

import asyncio
import os


async def model_routing_info():
    """Display model routing information for code generation"""
    from orchestrator.models import ROUTING_TABLE, COST_TABLE, FALLBACK_CHAIN, TaskType, Model
    
    print("=" * 60)
    print("📋 Code Generation Model Priority (ROUTING_TABLE)")
    print("=" * 60)
    
    for i, model in enumerate(ROUTING_TABLE[TaskType.CODE_GEN], 1):
        cost = COST_TABLE.get(model, {"input": 0, "output": 0})
        marker = "⭐" if model == Model.DEEPSEEK_CODER else "  "
        print(f"{marker} {i}. {model.value}")
        print(f"      Cost: ${cost['input']:.2f} in / ${cost['output']:.2f} out per 1M tokens")
        fallback = FALLBACK_CHAIN.get(model)
        if fallback:
            print(f"      Fallback: {fallback.value}")
    
    print("\n🎯 DeepSeek-Coder is #1 priority for CODE_GEN tasks!")


async def direct_api_call_example():
    """Direct API call using DeepSeek-Coder"""
    from orchestrator.models import Model
    from orchestrator.api_clients import UnifiedClient
    
    print("\n" + "=" * 60)
    print("Example: Direct API Call to DeepSeek-Coder")
    print("=" * 60)
    
    client = UnifiedClient()
    
    if not client.is_available(Model.DEEPSEEK_CODER):
        print("❌ DeepSeek-Coder not available. Check DEEPSEEK_API_KEY.")
        return
    
    print("✅ DeepSeek-Coder is available")
    
    prompt = """
Write a Python function that implements binary search with:
- Type hints
- Docstring with examples
- Edge case handling
"""
    
    print(f"\nPrompt: {prompt.strip()}")
    print("\nCalling DeepSeek-Coder...")
    
    response = await client.call(
        model=Model.DEEPSEEK_CODER,
        prompt=prompt,
        system="You are an expert Python programmer.",
        max_tokens=2048,
        temperature=0.3
    )
    
    print(f"\n✅ Response!")
    print(f"   Tokens: {response.input_tokens} in / {response.output_tokens} out")
    print(f"   Cost: ${response.cost_usd:.6f}")
    print(f"\nGenerated Code:\n{'-' * 40}")
    print(response.text[:1000])


async def orchestrator_project_example():
    """Using Orchestrator with DeepSeek-Coder"""
    from orchestrator import Orchestrator
    from orchestrator.models import Budget
    
    print("\n" + "=" * 60)
    print("Example: Orchestrator Project with DeepSeek-Coder")
    print("=" * 60)
    
    budget = Budget(max_usd=2.0, max_time_seconds=600)
    orchestrator = Orchestrator(budget=budget)
    
    project = "Build a Python CLI tool that converts JSON to CSV"
    criteria = "Handle nested objects, include tests, type hints"
    
    print(f"Project: {project}")
    print(f"Criteria: {criteria}")
    print("\nRunning (DeepSeek-Coder will be used for code tasks)...")
    
    state = await orchestrator.run_project(
        project_description=project,
        success_criteria=criteria
    )
    
    print(f"\n✅ Status: {state.status.value}")
    print(f"   Budget: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}")
    
    for task_id, result in state.results.items():
        print(f"\n   {task_id}: {result.model_used.value}")
        print(f"   Score: {result.score:.3f} | Cost: ${result.cost_usd:.4f}")


async def main():
    print("🚀 DeepSeek-Coder Examples")
    
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\n⚠️  DEEPSEEK_API_KEY not set!")
        print("   Set with: export DEEPSEEK_API_KEY='your-key'")
    
    # Always show routing info
    await model_routing_info()
    
    # Uncomment to run actual API calls:
    # await direct_api_call_example()
    # await orchestrator_project_example()
    
    print("\n" + "=" * 60)
    print("Examples complete! Uncomment API calls to test.")


if __name__ == "__main__":
    asyncio.run(main())
