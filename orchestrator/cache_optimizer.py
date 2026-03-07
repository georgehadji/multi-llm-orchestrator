"""
Multi-Level Cache Optimizer
============================

Advanced caching system with:
- L1: In-memory cache (hot data)
- L2: Disk cache (SQLite with TTL)
- L3: Semantic cache (intent-based matching)
- Cache warming for common patterns
- Compression for large responses
- Statistics and monitoring

Usage:
    from orchestrator.cache_optimizer import CacheOptimizer
    
    optimizer = CacheOptimizer()
    result = await optimizer.get_with_cache(model, prompt, task_type)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from collections import OrderedDict
from functools import lru_cache

import aiosqlite

from .models import Task, TaskType, Model
from .semantic_cache import SemanticCache

logger = logging.getLogger("orchestrator.cache_optimizer")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheConfig:
    """Configuration for multi-level cache."""
    # L1 Memory Cache
    l1_max_size: int = 100  # Max entries in memory
    l1_ttl_seconds: int = 3600  # 1 hour
    
    # L2 Disk Cache
    l2_db_path: Path = field(default_factory=lambda: Path.home() / ".orchestrator_cache" / "cache_l2.db")
    l2_ttl_hours: int = 24
    l2_max_size_mb: int = 1000
    l2_compression_threshold: int = 1000  # Compress responses > 1KB
    
    # L3 Semantic Cache
    l3_quality_threshold: float = 0.85
    l3_min_use_count: int = 2
    
    # Cache Warming
    warm_cache_on_startup: bool = True
    warm_patterns_file: Optional[Path] = None
    
    # Statistics
    track_stats: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# L1: In-Memory Cache (LRU with TTL)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryCacheEntry:
    """Entry in L1 memory cache with TTL."""
    value: Any
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class L1MemoryCache:
    """
    L1 Cache: In-memory LRU cache with TTL.
    Fastest but limited size.
    """
    
    def __init__(self, max_size: int = 100, default_ttl: int = 3600):
        self._cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access stats and move to end (LRU)
            entry.access_count += 1
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in memory cache."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._cache.popitem(last=False)
            
            entry = MemoryCacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl or self._default_ttl
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)
    
    async def clear(self) -> None:
        """Clear memory cache."""
        async with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "level": "L1 (Memory)",
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "max_size": self._max_size,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L2: Disk Cache with TTL and Compression
# ═══════════════════════════════════════════════════════════════════════════════

class L2DiskCache:
    """
    L2 Cache: SQLite-based disk cache with TTL and compression.
    Persistent across restarts.
    """
    
    def __init__(self, config: CacheConfig):
        self._config = config
        self._db_path = config.l2_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._cost_saved = 0.0
    
    async def _get_conn(self) -> aiosqlite.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self._db_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    ttl_hours INTEGER DEFAULT 24,
                    compressed INTEGER DEFAULT 0
                )
            """)
            await self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)
            """)
            await self._conn.commit()
        return self._conn
    
    def _compress(self, data: str) -> bytes:
        """Compress string data."""
        return zlib.compress(data.encode('utf-8'), level=6)
    
    def _decompress(self, data: bytes) -> str:
        """Decompress bytes to string."""
        return zlib.decompress(data).decode('utf-8')
    
    def _generate_key(self, model: str, prompt: str, max_tokens: int, 
                      system: str = "", temperature: float = 0.3) -> str:
        """Generate deterministic cache key."""
        payload = f"{model}:{prompt}:{max_tokens}:{system}:{temperature}"
        return hashlib.sha256(payload.encode()).hexdigest()
    
    async def get(self, model: str, prompt: str, max_tokens: int,
                  system: str = "", temperature: float = 0.3) -> Optional[Dict[str, Any]]:
        """Get from disk cache with TTL check."""
        key = self._generate_key(model, prompt, max_tokens, system, temperature)
        
        async with self._lock:
            conn = await self._get_conn()
            
            # Check for expired entries and delete them
            await conn.execute(
                "DELETE FROM cache WHERE (strftime('%s', 'now') - created_at) > ttl_hours * 3600"
            )
            
            async with conn.execute(
                "SELECT value, tokens_input, tokens_output, cost, compressed FROM cache WHERE key = ?",
                (key,)
            ) as cursor:
                row = await cursor.fetchone()
            
            if row is None:
                self._misses += 1
                return None
            
            value, tokens_in, tokens_out, cost, compressed = row
            
            # Decompress if needed
            if compressed:
                try:
                    value = self._decompress(value)
                except Exception as e:
                    logger.warning(f"Failed to decompress cache entry: {e}")
                    return None
            else:
                value = value.decode('utf-8') if isinstance(value, bytes) else value
            
            self._hits += 1
            self._tokens_saved += tokens_in + tokens_out
            self._cost_saved += cost
            
            return {
                "response": value,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "cost": cost,
                "cached": True,
            }
    
    async def put(self, model: str, prompt: str, max_tokens: int,
                  response: str, tokens_input: int = 0, tokens_output: int = 0,
                  cost: float = 0.0, system: str = "", temperature: float = 0.3,
                  ttl_hours: Optional[int] = None) -> None:
        """Store in disk cache with optional compression."""
        key = self._generate_key(model, prompt, max_tokens, system, temperature)
        ttl = ttl_hours or self._config.l2_ttl_hours
        
        # Compress if response is large
        compressed = 0
        if len(response) > self._config.l2_compression_threshold:
            try:
                response = self._compress(response)
                compressed = 1
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
        
        async with self._lock:
            conn = await self._get_conn()
            await conn.execute(
                """INSERT OR REPLACE INTO cache 
                   (key, value, tokens_input, tokens_output, cost, created_at, ttl_hours, compressed)
                   VALUES (?, ?, ?, ?, ?, strftime('%s', 'now'), ?, ?)""",
                (key, response, tokens_input, tokens_output, cost, ttl, compressed)
            )
            await conn.commit()
    
    async def clear(self) -> None:
        """Clear disk cache."""
        async with self._lock:
            conn = await self._get_conn()
            await conn.execute("DELETE FROM cache")
            await conn.commit()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count deleted."""
        async with self._lock:
            conn = await self._get_conn()
            cursor = await conn.execute(
                "DELETE FROM cache WHERE (strftime('%s', 'now') - created_at) > ttl_hours * 3600"
            )
            await conn.commit()
            return cursor.rowcount
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            conn = await self._get_conn()
            
            async with conn.execute("SELECT COUNT(*), SUM(tokens_input + tokens_output), SUM(cost) FROM cache") as cursor:
                row = await cursor.fetchone()
                total_entries = row[0] or 0
                total_tokens = row[1] or 0
                total_cost = row[2] or 0.0
            
            total_requests = self._hits + self._misses
            
            return {
                "level": "L2 (Disk)",
                "entries": total_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
                "tokens_saved": self._tokens_saved,
                "cost_saved": self._cost_saved,
                "db_size_mb": self._db_path.stat().st_size / (1024 * 1024) if self._db_path.exists() else 0,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Smart Cache Key Generator
# ═══════════════════════════════════════════════════════════════════════════════

class SmartCacheKeyGenerator:
    """
    Generates cache keys that maximize hits by normalizing variable content.
    """
    
    # Patterns to normalize
    PATTERNS = [
        (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),           # Dates
        (r'\b\d{2}/\d{2}/\d{4}\b', '<DATE>'),          # US dates
        (r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>'),  # UUIDs
        (r'\b0x[0-9a-fA-F]+\b', '<HEX>'),              # Hex numbers
        (r'\b\d+\.\d+\.\d+\.\d+\b', '<IP>'),          # IP addresses
        (r'\buser\d+\b', '<USER>'),                    # user123 → <USER>
        (r'\bitem\d+\b', '<ITEM>'),                    # item456 → <ITEM>
        (r'\btask_\d+\b', '<TASK_ID>'),                # task_001 → <TASK_ID>
        (r'"[^"]*"', '"<STRING>"'),                    # String literals
        (r"'[^']*'", "'<STRING>'"),                    # String literals
    ]
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize text for better cache hits."""
        # Convert to lowercase
        text = text.lower()
        
        # Apply normalization patterns
        for pattern, replacement in cls.PATTERNS:
            text = re.sub(pattern, replacement, text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @classmethod
    def generate_key(cls, model: str, prompt: str, max_tokens: int,
                     system: str = "", temperature: float = 0.3) -> str:
        """Generate smart cache key."""
        normalized_prompt = cls.normalize(prompt)
        normalized_system = cls.normalize(system)
        
        payload = f"{model}:{normalized_prompt}:{max_tokens}:{normalized_system}:{temperature}"
        return hashlib.sha256(payload.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Warming
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WarmPattern:
    """Pattern for cache warming."""
    task_type: TaskType
    prompt_template: str
    variables: Dict[str, List[str]] = field(default_factory=dict)


DEFAULT_WARM_PATTERNS: List[WarmPattern] = [
    WarmPattern(
        TaskType.CODE_GEN,
        "Generate a Python function to {operation} with type hints and docstring",
        {"operation": ["validate email", "parse JSON", "hash password", "format date"]}
    ),
    WarmPattern(
        TaskType.CODE_REVIEW,
        "Review this {language} code for {issues}",
        {
            "language": ["Python", "JavaScript", "TypeScript"],
            "issues": ["security vulnerabilities", "performance issues", "code smells"]
        }
    ),
    WarmPattern(
        TaskType.EVALUATE,
        "Evaluate the quality of this code on a scale of 0 to 1",
        {}
    ),
    WarmPattern(
        TaskType.SUMMARIZE,
        "Summarize the following {content_type} in {length} sentences",
        {
            "content_type": ["documentation", "code", "error logs"],
            "length": ["2-3", "3-5", "5-7"]
        }
    ),
]


class CacheWarmer:
    """Warms cache with common patterns at startup."""
    
    def __init__(self, client, patterns: Optional[List[WarmPattern]] = None):
        self._client = client
        self._patterns = patterns or DEFAULT_WARM_PATTERNS
    
    async def warm(self, models: List[Model] = None) -> Dict[str, Any]:
        """
        Warm cache with common patterns.
        
        Returns:
            Statistics about warming process
        """
        models = models or [Model.GPT_4O_MINI, Model.GEMINI_FLASH]
        stats = {"patterns_generated": 0, "cache_entries": 0, "cost": 0.0}
        
        logger.info("Starting cache warming...")
        
        for pattern in self._patterns:
            # Generate all combinations of variables
            combinations = self._generate_combinations(pattern.variables)
            
            for combo in combinations:
                prompt = pattern.prompt_template.format(**combo)
                
                for model in models:
                    try:
                        # Check if already cached
                        cached = await self._client.cache.get(
                            model.value, prompt, max_tokens=1000
                        )
                        
                        if cached is None:
                            # Generate and cache
                            response = await self._client.call_model(
                                model=model,
                                prompt=prompt,
                                max_tokens=1000,
                                temperature=0.3,
                            )
                            
                            await self._client.cache.put(
                                model=model.value,
                                prompt=prompt,
                                max_tokens=1000,
                                response=response.text,
                                tokens_input=response.input_tokens,
                                tokens_output=response.output_tokens,
                                cost=response.cost_usd,
                            )
                            
                            stats["patterns_generated"] += 1
                            stats["cache_entries"] += 1
                            stats["cost"] += response.cost_usd
                            
                    except Exception as e:
                        logger.warning(f"Failed to warm cache for {model.value}: {e}")
        
        logger.info(f"Cache warming complete: {stats['cache_entries']} entries, ${stats['cost']:.4f}")
        return stats
    
    def _generate_combinations(self, variables: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate all combinations of variable values."""
        if not variables:
            return [{}]
        
        keys = list(variables.keys())
        values = list(variables.values())
        
        import itertools
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations


# ═══════════════════════════════════════════════════════════════════════════════
# Main Cache Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class CacheOptimizer:
    """
    Multi-level cache optimizer combining L1, L2, and L3 caches.
    
    Usage:
        optimizer = CacheOptimizer()
        
        # Get with automatic cache lookup
        result = await optimizer.get(model, prompt, task_type)
        
        # Put with automatic cache storage
        await optimizer.put(model, prompt, response, task_type)
        
        # Get statistics
        stats = optimizer.get_stats()
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self._config = config or CacheConfig()
        
        # L1: Memory cache
        self._l1 = L1MemoryCache(
            max_size=self._config.l1_max_size,
            default_ttl=self._config.l1_ttl_seconds
        )
        
        # L2: Disk cache
        self._l2 = L2DiskCache(self._config)
        
        # L3: Semantic cache
        self._l3 = SemanticCache(
            quality_threshold=self._config.l3_quality_threshold,
            min_use_count=self._config.l3_min_use_count
        )
        
        # Statistics
        self._total_hits_l1 = 0
        self._total_hits_l2 = 0
        self._total_hits_l3 = 0
        self._total_misses = 0
        self._tokens_saved = 0
        self._cost_saved = 0.0
    
    def _generate_key(self, model: str, prompt: str, max_tokens: int,
                      system: str = "", temperature: float = 0.3) -> str:
        """Generate smart cache key."""
        return SmartCacheKeyGenerator.generate_key(
            model, prompt, max_tokens, system, temperature
        )
    
    async def get(self, model: str, prompt: str, max_tokens: int,
                  system: str = "", temperature: float = 0.3,
                  task_type: Optional[TaskType] = None) -> Optional[Dict[str, Any]]:
        """
        Get from cache with multi-level lookup.
        
        Order: L1 → L2 → L3
        """
        # Generate smart key
        key = self._generate_key(model, prompt, max_tokens, system, temperature)
        
        # L1: Memory cache
        result = await self._l1.get(key)
        if result is not None:
            self._total_hits_l1 += 1
            logger.debug(f"L1 cache hit: {key[:16]}...")
            return result
        
        # L2: Disk cache
        result = await self._l2.get(model, prompt, max_tokens, system, temperature)
        if result is not None:
            self._total_hits_l2 += 1
            # Promote to L1
            await self._l1.put(key, result)
            logger.debug(f"L2 cache hit: {key[:16]}...")
            return result
        
        # L3: Semantic cache (if task type provided)
        if task_type:
            # Create dummy task for semantic cache lookup
            dummy_task = Task(
                id="cache_lookup",
                type=task_type,
                prompt=prompt,
            )
            semantic_result = self._l3.get_cached_pattern(dummy_task)
            if semantic_result:
                self._total_hits_l3 += 1
                result = {
                    "response": semantic_result,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost": 0.0,
                    "cached": True,
                    "semantic": True,
                }
                # Promote to L1 and L2
                await self._l1.put(key, result)
                logger.debug(f"L3 cache hit: {key[:16]}...")
                return result
        
        self._total_misses += 1
        return None
    
    async def put(self, model: str, prompt: str, max_tokens: int,
                  response: str, tokens_input: int, tokens_output: int,
                  cost: float, system: str = "", temperature: float = 0.3,
                  task_type: Optional[TaskType] = None,
                  quality_score: Optional[float] = None) -> None:
        """
        Store in cache at all levels.
        """
        key = self._generate_key(model, prompt, max_tokens, system, temperature)
        
        result = {
            "response": response,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost": cost,
            "cached": True,
        }
        
        # L1: Memory cache
        await self._l1.put(key, result)
        
        # L2: Disk cache
        await self._l2.put(
            model, prompt, max_tokens, response,
            tokens_input, tokens_output, cost,
            system, temperature
        )
        
        # L3: Semantic cache (if quality score provided)
        if task_type and quality_score:
            dummy_task = Task(
                id="cache_store",
                type=task_type,
                prompt=prompt,
            )
            self._l3.cache_pattern(dummy_task, response, quality_score)
        
        logger.debug(f"Cached at all levels: {key[:16]}...")
    
    async def warm(self, client, models: Optional[List[Model]] = None) -> Dict[str, Any]:
        """Warm cache with common patterns."""
        warmer = CacheWarmer(client)
        return await warmer.warm(models)
    
    async def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries."""
        l1_expired = 0
        # L1 cleanup happens automatically on access
        
        # L2 cleanup
        l2_deleted = await self._l2.cleanup_expired()
        
        return {
            "l1_expired": l1_expired,
            "l2_deleted": l2_deleted,
        }
    
    async def clear(self, level: Optional[str] = None) -> None:
        """
        Clear cache levels.
        
        Args:
            level: 'l1', 'l2', 'l3', or None for all
        """
        if level is None or level == 'l1':
            await self._l1.clear()
        if level is None or level == 'l2':
            await self._l2.clear()
        if level is None or level == 'l3':
            self._l3._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = self._total_hits_l1 + self._total_hits_l2 + self._total_hits_l3
        total_requests = total_hits + self._total_misses
        
        l1_stats = self._l1.get_stats()
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": self._total_misses,
            "overall_hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "l1_hits": self._total_hits_l1,
            "l2_hits": self._total_hits_l2,
            "l3_hits": self._total_hits_l3,
            "l1_stats": l1_stats,
            "tokens_saved": self._tokens_saved,
            "cost_saved": self._cost_saved,
        }
    
    def print_stats(self) -> None:
        """Print cache statistics in readable format."""
        stats = self.get_stats()
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    CACHE STATISTICS                          ║
╠══════════════════════════════════════════════════════════════╣
║ Total Requests:    {total_requests:>10,}                              ║
║ Total Hits:        {total_hits:>10,}  ({hit_rate:.1%})                      ║
║ Total Misses:      {total_misses:>10,}                              ║
╠══════════════════════════════════════════════════════════════╣
║ Breakdown by Level:                                          ║
║   L1 (Memory):     {l1_hits:>10,} hits                           ║
║   L2 (Disk):       {l2_hits:>10,} hits                           ║
║   L3 (Semantic):   {l3_hits:>10,} hits                           ║
╠══════════════════════════════════════════════════════════════╣
║ Savings:                                                     ║
║   Tokens Saved:    {tokens_saved:>10,}                              ║
║   Cost Saved:      ${cost_saved:>9.2f}                              ║
╚══════════════════════════════════════════════════════════════╝
        """.format(
            total_requests=stats['total_requests'],
            total_hits=stats['total_hits'],
            hit_rate=stats['overall_hit_rate'],
            total_misses=stats['total_misses'],
            l1_hits=stats['l1_hits'],
            l2_hits=stats['l2_hits'],
            l3_hits=stats['l3_hits'],
            tokens_saved=stats['tokens_saved'],
            cost_saved=stats['cost_saved'],
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

# Global cache optimizer instance
_cache_optimizer: Optional[CacheOptimizer] = None


def get_cache_optimizer() -> CacheOptimizer:
    """Get global cache optimizer instance."""
    global _cache_optimizer
    if _cache_optimizer is None:
        _cache_optimizer = CacheOptimizer()
    return _cache_optimizer


def reset_cache_optimizer() -> None:
    """Reset global cache optimizer."""
    global _cache_optimizer
    _cache_optimizer = None


# Example usage
if __name__ == "__main__":
    # Test cache optimizer
    optimizer = CacheOptimizer()
    print("Cache optimizer initialized")
    optimizer.print_stats()
