"""
Check API Keys - Ελεγχος αν τα API keys φορτώθηκαν σωστά
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✅ Loaded .env file")
except ImportError:
    print("❌ python-dotenv not installed")
    print("   Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Failed to load .env: {e}")

print()
print("=" * 60)
print("🔑 API KEY STATUS")
print("=" * 60)

providers = [
    ("OpenAI", "OPENAI_API_KEY"),
    ("DeepSeek", "DEEPSEEK_API_KEY"),
    ("Google", "GOOGLE_API_KEY"),
    ("Anthropic", "ANTHROPIC_API_KEY"),
    ("Minimax", "MINIMAX_API_KEY"),
]

for name, env_var in providers:
    value = os.getenv(env_var)
    if value:
        # Mask the key for security
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"✅ {name:12} | {env_var:20} | Set ({masked})")
    else:
        print(f"❌ {name:12} | {env_var:20} | NOT SET")

print("=" * 60)
print()
print("💡 If keys show as NOT SET but you have a .env file:")
print("   1. Check that the .env file is in the project root")
print("   2. Check that variable names are correct (e.g., OPENAI_API_KEY)")
print("   3. Check that there are no spaces around the = sign")
print()
