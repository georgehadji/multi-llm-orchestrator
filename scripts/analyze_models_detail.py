import json

with open('openrouter_models.json', 'r') as f:
    data = json.load(f)

models = data['data']

# Find anthropic models
print('ANTHROPIC MODELS:')
for m in models:
    if m['id'].startswith('anthropic/'):
        print(f"  {m['id']}")

print()
print('QWEN MODELS:')
for m in models:
    if m['id'].startswith('qwen/'):
        print(f"  {m['id']}")

print()
print('MOONSHOT MODELS:')
for m in models:
    if m['id'].startswith('moonshot/'):
        print(f"  {m['id']}")

print()
print('DEEPSEEK MODELS:')
for m in models:
    if m['id'].startswith('deepseek/'):
        print(f"  {m['id']}")
