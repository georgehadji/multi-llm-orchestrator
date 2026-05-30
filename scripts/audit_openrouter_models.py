#!/usr/bin/env python3
"""
OpenRouter Model Audit Script
=============================
Query OpenRouter API to check model variant availability.
Part of Phase 0: Foundation for OpenRouter Optimizations.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

import asyncio
import json
import os
from typing import Any

import aiohttp

# Models from current ROUTING_TABLE that we want to check
MODELS_TO_CHECK = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "google/gemini-3.1-pro",
    "google/gemini-3.1-flash",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1",
    "meta/llama-4-maverick",
    "meta/llama-4-scout",
    "x-ai/grok-4.20",
    "qwen/qwen-2.5-coder-32b",
    "xiaomi/mimo-v2-flash",
    "moonshot/kimi-k2.5",
    "stepfun/step-3.5-flash",
    "z-ai/glm-4.7-flash",
]

# Variants to check availability
VARIANTS = [":free", ":nitro", ":floor", ":thinking", ":extended", ":exacto"]

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


async def fetch_models(session: aiohttp.ClientSession) -> list[dict[str, Any]]:
    """Fetch all available models from OpenRouter API."""
    url = f"{OPENROUTER_API_BASE}/models"
    
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return data.get("data", [])
        else:
            print(f"Error fetching models: {response.status}")
            return []


def check_variant_availability(
    model_id: str, 
    available_models: list[dict[str, Any]]
) -> dict[str, bool]:
    """Check which variants are available for a given model."""
    results = {}
    
    for variant in VARIANTS:
        variant_id = f"{model_id}{variant}"
        # Check if any model in the list matches this variant ID
        is_available = any(
            m.get("id") == variant_id or 
            m.get("canonical_slug") == variant_id
            for m in available_models
        )
        results[variant] = is_available
    
    return results


def get_model_info(model_id: str, available_models: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Get detailed info for a specific model."""
    for model in available_models:
        if model.get("id") == model_id or model.get("canonical_slug") == model_id:
            return {
                "id": model.get("id"),
                "name": model.get("name"),
                "context_length": model.get("context_length"),
                "pricing": model.get("pricing"),
                "supported_parameters": model.get("supported_parameters", []),
            }
    return None


async def main():
    """Main audit function."""
    print("=" * 70)
    print("OpenRouter Model Variant Availability Audit")
    print("=" * 70)
    print()
    
    async with aiohttp.ClientSession() as session:
        print("Fetching available models from OpenRouter...")
        available_models = await fetch_models(session)
        print(f"Found {len(available_models)} models\n")
        
        # Build lookup sets for quick checking
        available_ids = {m.get("id") for m in available_models}
        available_slugs = {m.get("canonical_slug") for m in available_models if m.get("canonical_slug")}
        
        results = {
            "audit_date": "2026-04-05",
            "total_available_models": len(available_models),
            "models_checked": {},
            "variant_summary": {v: {"available": 0, "unavailable": 0} for v in VARIANTS}
        }
        
        print("Checking model availability and variants...")
        print("-" * 70)
        
        for model_id in MODELS_TO_CHECK:
            # Check base model availability
            base_available = model_id in available_ids or model_id in available_slugs
            
            if not base_available:
                print(f"\n⚠️  {model_id}: NOT FOUND in OpenRouter")
                results["models_checked"][model_id] = {
                    "available": False,
                    "variants": {}
                }
                continue
            
            # Get model info
            info = get_model_info(model_id, available_models)
            
            # Check variant availability
            variant_status = check_variant_availability(model_id, available_models)
            
            # Update summary
            for variant, available in variant_status.items():
                if available:
                    results["variant_summary"][variant]["available"] += 1
                else:
                    results["variant_summary"][variant]["unavailable"] += 1
            
            results["models_checked"][model_id] = {
                "available": True,
                "info": info,
                "variants": variant_status
            }
            
            # Print summary
            print(f"\n✅ {model_id}")
            if info:
                print(f"   Context: {info.get('context_length', 'N/A')} tokens")
                pricing = info.get('pricing', {})
                prompt_price = pricing.get('prompt', 'N/A')
                completion_price = pricing.get('completion', 'N/A')
                print(f"   Pricing: ${prompt_price}/1M prompt, ${completion_price}/1M completion")
            
            # Print variant status
            available_variants = [v for v, avail in variant_status.items() if avail]
            unavailable_variants = [v for v, avail in variant_status.items() if not avail]
            
            if available_variants:
                print(f"   Available variants: {', '.join(available_variants)}")
            if unavailable_variants:
                print(f"   ❌ Missing variants: {', '.join(unavailable_variants)}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        total_checked = len(MODELS_TO_CHECK)
        available_count = sum(1 for m in results["models_checked"].values() if m["available"])
        
        print(f"\nBase Models: {available_count}/{total_checked} available")
        print("\nVariant Availability Summary:")
        print(f"{'Variant':<15} {'Available':<12} {'%':<8}")
        print("-" * 35)
        
        for variant, counts in results["variant_summary"].items():
            total = counts["available"] + counts["unavailable"]
            pct = (counts["available"] / total * 100) if total > 0 else 0
            print(f"{variant:<15} {counts['available']}/{total:<8} {pct:.1f}%")
        
        # Save results to JSON
        output_file = "openrouter_model_audit.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Detailed results saved to: {output_file}")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        for variant in VARIANTS:
            available = results["variant_summary"][variant]["available"]
            total = available + results["variant_summary"][variant]["unavailable"]
            pct = (available / total * 100) if total > 0 else 0
            
            if pct >= 80:
                print(f"✅ {variant}: High availability ({pct:.0f}%) - Safe to implement")
            elif pct >= 50:
                print(f"⚠️  {variant}: Medium availability ({pct:.0f}%) - Implement with fallback")
            else:
                print(f"❌ {variant}: Low availability ({pct:.0f}%) - Not recommended yet")


if __name__ == "__main__":
    asyncio.run(main())
