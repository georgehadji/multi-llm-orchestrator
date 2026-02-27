"""Quick DeepSeek API test."""
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from orchestrator.api_clients import UnifiedClient
from orchestrator.cache import DiskCache
from orchestrator.models import Model


async def main():
    cache = DiskCache()
    client = UnifiedClient(cache=cache, max_concurrency=1)

    for m in [Model.DEEPSEEK_CHAT, Model.DEEPSEEK_REASONER]:
        print(f"Testing {m.value}...")
        resp = await client.call(
            m, "What is 7 * 8? Reply with just the number.",
            max_tokens=50, bypass_cache=True,
        )
        print(f"  Response: {resp.text.strip()!r}")
        print(f"  Tokens: in={resp.input_tokens} out={resp.output_tokens}  cost=${resp.cost_usd:.6f}")

    await cache.close()


asyncio.run(main())
