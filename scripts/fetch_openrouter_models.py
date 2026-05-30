"""
Fetch and analyze OpenRouter models API
"""
import urllib.request
import json
import ssl

def fetch_models():
    ctx = ssl.create_default_context()
    
    try:
        with urllib.request.urlopen(
            'https://openrouter.ai/api/v1/models', 
            context=ctx, 
            timeout=30
        ) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        print(f'Error fetching models: {e}')
        return None

def analyze_models(data):
    if not data or 'data' not in data:
        print("No data received")
        return
    
    models = data['data']
    print(f"\nTotal models available: {len(models)}\n")
    
    # Collect all model IDs
    model_ids = [m.get('id', '') for m in models]
    
    # Check for variant support by looking at model IDs
    variants = {
        ':nitro': [m for m in model_ids if ':nitro' in m],
        ':thinking': [m for m in model_ids if ':thinking' in m],
        ':floor': [m for m in model_ids if ':floor' in m],
        ':extended': [m for m in model_ids if ':extended' in m],
        ':exacto': [m for m in model_ids if ':exacto' in m],
        ':free': [m for m in model_ids if ':free' in m],
    }
    
    print("=" * 60)
    print("VARIANT SUPPORT ANALYSIS")
    print("=" * 60)
    
    for variant, matches in variants.items():
        print(f"\n{variant}: {len(matches)} models")
        if matches:
            print("  Examples:")
            for m in matches[:5]:
                print(f"    - {m}")
    
    # Check models in our ROUTING_TABLE
    print("\n" + "=" * 60)
    print("MODELS IN OUR ROUTING_TABLE")
    print("=" * 60)
    
    our_models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-5",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-v3.2",
        "qwen/qwen-2.5-coder-32b",
        "xiaomi/mimo-v2-flash",
        "moonshot/kimi-k2.5",
    ]
    
    for model in our_models:
        found = model in model_ids
        status = "[AVAILABLE]" if found else "[NOT FOUND]"
        print(f"  {model}: {status}")
        
        # Check for variants
        if found:
            for variant in [':nitro', ':thinking', ':floor', ':free']:
                variant_id = model + variant
                if variant_id in model_ids:
                    print(f"    +-- {variant} [SUPPORTED]")
    
    # Providers
    providers = set()
    for model_id in model_ids:
        if '/' in model_id:
            providers.add(model_id.split('/')[0])
    
    print(f"\n\nTotal providers: {len(providers)}")
    print(f"Top providers: {', '.join(sorted(list(providers))[:15])}")
    
    # Save full data for reference
    with open('openrouter_models.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\n\nFull model list saved to: openrouter_models.json")

if __name__ == '__main__':
    print("Fetching OpenRouter models...")
    data = fetch_models()
    if data:
        analyze_models(data)
