"""
Pricing Cache with Graceful Degradation
=======================================

Ensures users always have pricing information even when:
- Dashboard API is unavailable
- Provider APIs return errors
- Network is flaky

Implements minimax regret: worst-case scenario is using
stale but reasonable prices, not running blind.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class PricingTier(Enum):
    """Pricing tiers for LLM providers."""
    FREE = "free"
    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass(frozen=True)
class ModelPricing:
    """Immutable pricing for a specific model."""
    model_id: str
    tier: PricingTier
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    currency: str = "USD"
    source: str = "unknown"  # "live_api", "cache", "fallback"
    fetched_at: Optional[datetime] = None
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given token count."""
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        return round(input_cost + output_cost, 6)


@dataclass
class PricingHealth:
    """Health status of the pricing system."""
    is_live: bool
    cache_age_hours: float
    stale_models: list[str]
    fallback_active: bool
    last_error: Optional[str] = None


class PricingCache:
    """
    Multi-tier pricing cache with automatic fallback.
    
    Resolution order:
    1. Live dashboard API (freshest)
    2. Local disk cache (up to 24h old)
    3. Compiled fallback prices (guaranteed available)
    
    This ensures minimax regret: even in worst case (no network),
    users have reasonable pricing estimates.
    """
    
    # Fallback prices - updated quarterly via release
    FALLBACK_PRICES: dict[str, ModelPricing] = {
        # Economy tier
        "gemini-flash-lite": ModelPricing(
            model_id="gemini-flash-lite",
            tier=PricingTier.ECONOMY,
            input_price_per_1m=0.075,
            output_price_per_1m=0.30,
            source="fallback",
        ),
        "deepseek-chat": ModelPricing(
            model_id="deepseek-chat",
            tier=PricingTier.ECONOMY,
            input_price_per_1m=0.28,
            output_price_per_1m=0.42,
            source="fallback",
        ),
        # Standard tier
        "deepseek-reasoner": ModelPricing(
            model_id="deepseek-reasoner",
            tier=PricingTier.STANDARD,
            input_price_per_1m=0.28,
            output_price_per_1m=0.42,
            source="fallback",
        ),
        "gpt-4o-mini": ModelPricing(
            model_id="gpt-4o-mini",
            tier=PricingTier.STANDARD,
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            source="fallback",
        ),
        "kimi-k2-5": ModelPricing(
            model_id="kimi-k2-5",
            tier=PricingTier.STANDARD,
            input_price_per_1m=0.56,
            output_price_per_1m=2.92,
            source="fallback",
        ),
        # Premium tier
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            tier=PricingTier.PREMIUM,
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            source="fallback",
        ),
        "o3-mini": ModelPricing(
            model_id="o3-mini",
            tier=PricingTier.PREMIUM,
            input_price_per_1m=1.10,
            output_price_per_1m=4.40,
            source="fallback",
        ),
        "gemini-pro": ModelPricing(
            model_id="gemini-pro",
            tier=PricingTier.PREMIUM,
            input_price_per_1m=3.50,
            output_price_per_1m=10.50,
            source="fallback",
        ),
    }
    
    CACHE_TTL_HOURS = 24
    CACHE_FILE = Path.home() / ".orchestrator" / "pricing_cache.json"
    
    def __init__(
        self,
        dashboard_url: Optional[str] = None,
        cache_file: Optional[Path] = None,
    ):
        self.dashboard_url = dashboard_url or "http://localhost:8888"
        self.cache_file = cache_file or self.CACHE_FILE
        self._memory_cache: dict[str, ModelPricing] = {}
        self._last_fetch_error: Optional[str] = None
        
        # Ensure cache directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_disk_cache(self) -> dict[str, ModelPricing]:
        """Load pricing from local disk cache."""
        if not self.cache_file.exists():
            return {}
        
        try:
            data = json.loads(self.cache_file.read_text())
            prices = {}
            for model_id, pricing_data in data.items():
                pricing_data["fetched_at"] = datetime.fromisoformat(
                    pricing_data["fetched_at"]
                ) if pricing_data.get("fetched_at") else None
                prices[model_id] = ModelPricing(**pricing_data)
            return prices
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load pricing cache: {e}")
            return {}
    
    def _save_disk_cache(self, prices: dict[str, ModelPricing]) -> None:
        """Save pricing to local disk cache."""
        try:
            data = {}
            for model_id, pricing in prices.items():
                d = asdict(pricing)
                if d["fetched_at"]:
                    d["fetched_at"] = d["fetched_at"].isoformat()
                data[model_id] = d
            
            self.cache_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save pricing cache: {e}")
    
    async def _fetch_live_prices(self) -> Optional[dict[str, ModelPricing]]:
        """Fetch live prices from dashboard API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.dashboard_url}/api/providers/pricing"
                )
                response.raise_for_status()
                data = response.json()
                
                prices = {}
                for item in data.get("models", []):
                    pricing = ModelPricing(
                        model_id=item["id"],
                        tier=PricingTier(item.get("tier", "standard")),
                        input_price_per_1m=item["input_price_per_1m"],
                        output_price_per_1m=item["output_price_per_1m"],
                        currency=item.get("currency", "USD"),
                        source="live_api",
                        fetched_at=datetime.utcnow(),
                    )
                    prices[pricing.model_id] = pricing
                
                self._last_fetch_error = None
                return prices
                
        except Exception as e:
            self._last_fetch_error = str(e)
            logger.warning(f"Failed to fetch live pricing: {e}")
            return None
    
    async def get_pricing(self, model_id: str) -> ModelPricing:
        """
        Get pricing for a model with automatic fallback.
        
        Resolution order:
        1. Memory cache (fastest)
        2. Live API (freshest)
        3. Disk cache (stale but recent)
        4. Compiled fallback (guaranteed)
        
        Returns:
            ModelPricing with source indicator
        """
        model_id = model_id.lower().replace("_", "-")
        
        # 1. Check memory cache
        if model_id in self._memory_cache:
            cached = self._memory_cache[model_id]
            age = datetime.utcnow() - (cached.fetched_at or datetime.min)
            if age < timedelta(hours=self.CACHE_TTL_HOURS):
                return cached
        
        # 2. Try live API
        live_prices = await self._fetch_live_prices()
        if live_prices:
            self._memory_cache.update(live_prices)
            self._save_disk_cache(live_prices)
            if model_id in live_prices:
                return live_prices[model_id]
        
        # 3. Try disk cache
        disk_cache = self._load_disk_cache()
        if model_id in disk_cache:
            age = datetime.utcnow() - (disk_cache[model_id].fetched_at or datetime.min)
            if age < timedelta(hours=self.CACHE_TTL_HOURS * 7):  # 1 week grace period
                self._memory_cache[model_id] = disk_cache[model_id]
                return disk_cache[model_id]
        
        # 4. Fallback to compiled prices (guaranteed)
        if model_id in self.FALLBACK_PRICES:
            logger.warning(
                f"Using fallback pricing for {model_id}. "
                f"Live prices unavailable. Last error: {self._last_fetch_error}"
            )
            return self.FALLBACK_PRICES[model_id]
        
        # 5. Unknown model - estimate as premium
        logger.error(f"Unknown model {model_id}, using premium tier estimate")
        return ModelPricing(
            model_id=model_id,
            tier=PricingTier.PREMIUM,
            input_price_per_1m=5.0,  # Conservative estimate
            output_price_per_1m=15.0,
            source="estimate",
            fetched_at=datetime.utcnow(),
        )
    
    def get_health(self) -> PricingHealth:
        """Get health status of pricing system."""
        disk_cache = self._load_disk_cache()
        
        stale_models = []
        for model_id, pricing in disk_cache.items():
            age = datetime.utcnow() - (pricing.fetched_at or datetime.min)
            if age > timedelta(hours=self.CACHE_TTL_HOURS):
                stale_models.append(model_id)
        
        oldest_cache_age = 0.0
        if disk_cache:
            ages = [
                (datetime.utcnow() - (p.fetched_at or datetime.min)).total_seconds() / 3600
                for p in disk_cache.values()
            ]
            oldest_cache_age = max(ages)
        
        return PricingHealth(
            is_live=self._last_fetch_error is None,
            cache_age_hours=oldest_cache_age,
            stale_models=stale_models,
            fallback_active=len(disk_cache) == 0 and self._last_fetch_error is not None,
            last_error=self._last_fetch_error,
        )
    
    async def refresh_cache(self) -> bool:
        """Manually trigger cache refresh. Returns True if successful."""
        live_prices = await self._fetch_live_prices()
        if live_prices:
            self._memory_cache.update(live_prices)
            self._save_disk_cache(live_prices)
            return True
        return False


# Global instance for convenience
_pricing_cache: Optional[PricingCache] = None


def get_pricing_cache() -> PricingCache:
    """Get the global pricing cache instance."""
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = PricingCache()
    return _pricing_cache


async def estimate_task_cost(model_id: str, estimated_input_tokens: int, 
                             estimated_output_tokens: int) -> tuple[float, str]:
    """
    Estimate cost for a task with source indication.
    
    Returns:
        (cost_usd, price_source)
    """
    cache = get_pricing_cache()
    pricing = await cache.get_pricing(model_id)
    cost = pricing.estimate_cost(estimated_input_tokens, estimated_output_tokens)
    return cost, pricing.source
