# Multi-Level Cache Optimization Summary

## 🚀 Overview

The AI Orchestrator now features a **3-level caching system** designed to maximize token savings and minimize API costs.

## 📊 Cache Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW                                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1: MEMORY CACHE (Hot Data)                                     │
│ • Max 200 entries                                               │
│ • 1 hour TTL                                                    │
│ • < 1ms latency                                                 │
│ • In-memory LRU with automatic eviction                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Miss
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2: DISK CACHE (Persistent)                                     │
│ • SQLite with WAL mode                                          │
│ • 48 hour TTL                                                   │
│ • ~5ms latency                                                  │
│ • Automatic compression for responses > 1KB                     │
│ • Smart key normalization (dates, UUIDs, etc.)                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Miss
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3: SEMANTIC CACHE (Intent-Based)                               │
│ • Pattern matching on semantic intent                           │
│ • Quality threshold: 0.85                                       │
│ • Minimum use count: 2                                          │
│ • Normalizes variable names, literals, whitespace               │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Miss
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ API CALL (Last Resort)                                          │
│ • Actual LLM request                                            │
│ • Full cost                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 💡 Key Features

### 1. Smart Cache Key Generation
```python
# Variable content is normalized for better cache hits
"Generate validator for user123"     →  "Generate validator for <USER>"
"Generate validator for user456"     →  "Generate validator for <USER>"
# Same cache key = Cache hit!
```

### 2. Automatic Compression
- Responses > 1KB are automatically compressed with zlib
- Saves disk space without sacrificing speed
- Transparent decompression on retrieval

### 3. Multi-Level TTL
- **L1**: 1 hour (memory)
- **L2**: 48 hours (disk)
- **L3**: Permanent (until manually cleared)

### 4. Cache Warming
Pre-populate cache with common patterns at startup:
```python
# Examples of warmed patterns
- "Generate Python function with type hints"
- "Review code for security vulnerabilities"
- "Evaluate code quality on scale 0-1"
```

## 📈 Expected Performance

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| Hit Rate | 0% | 60-85% | +60-85% |
| Avg Latency | 5-30s | <10ms (hit) | 99.9% faster |
| Token Cost | $100 | $15-40 | 60-85% savings |
| API Calls | 1000 | 150-400 | 60-85% reduction |

## 🔧 Usage

### Python API
```python
from orchestrator import CacheOptimizer, CacheConfig

# Create optimizer with custom config
optimizer = CacheOptimizer(CacheConfig(
    l1_max_size=200,
    l1_ttl_seconds=3600,
    l2_ttl_hours=48,
    l3_quality_threshold=0.85,
))

# Get from cache (checks L1 → L2 → L3)
result = await optimizer.get(
    model="gpt-4o",
    prompt="Generate email validator",
    max_tokens=1000,
    task_type=TaskType.CODE_GEN,
)

# Store in cache (stores to L1, L2, L3)
await optimizer.put(
    model="gpt-4o",
    prompt="Generate email validator",
    response=generated_code,
    tokens_input=50,
    tokens_output=200,
    cost=0.005,
    task_type=TaskType.CODE_GEN,
    quality_score=0.95,
)

# Show statistics
optimizer.print_stats()
```

### CLI Commands
```bash
# Show cache statistics
python -m orchestrator cache-stats

# Clear all cache levels
python -m orchestrator cache-stats --clear

# Clear specific level
python -m orchestrator cache-stats --clear --level l1

# Cleanup expired entries
python -m orchestrator cache-stats --cleanup
```

## 📊 Cache Statistics Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    CACHE STATISTICS                              ║
╠══════════════════════════════════════════════════════════════════╣
║ Total Requests:          1,234                                   ║
║ Total Hits:              1,052  (85.3%)                          ║
║ Total Misses:              182                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ By Level:                                                        ║
║   L1 (Memory):           523 hits                                ║
║   L2 (Disk):             315 hits                                ║
║   L3 (Semantic):         214 hits                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Savings:                                                         ║
║   Tokens Saved:          45,230                                  ║
║   Cost Saved:            $12.45                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

## 🔬 Smart Normalization

The cache normalizes prompts to maximize hits:

| Pattern | Example | Normalized |
|---------|---------|------------|
| Dates | `2024-01-15` | `<DATE>` |
| UUIDs | `a3f5d2...` | `<UUID>` |
| IPs | `192.168.1.1` | `<IP>` |
| Users | `user123` | `<USER>` |
| Items | `item456` | `<ITEM>` |
| Task IDs | `task_001` | `<TASK_ID>` |
| Hex | `0xDEADBEEF` | `<HEX>` |
| Strings | `"hello"` | `"<STRING>"` |

## 🛠️ Files Modified

| File | Changes |
|------|---------|
| `orchestrator/cache_optimizer.py` | NEW: Complete multi-level cache implementation |
| `orchestrator/__init__.py` | Added CacheOptimizer exports |
| `orchestrator/engine.py` | Integrated CacheOptimizer into task execution |
| `orchestrator/cli.py` | Added `cache-stats` command |

## 🎯 Future Enhancements

1. **Distributed Cache**: Redis support for multi-node deployments
2. **ML-Based Cache Warming**: Predict which patterns to pre-cache
3. **Adaptive TTL**: Adjust TTL based on hit patterns
4. **Cache Analytics**: Detailed reports on cache effectiveness per model/task

## 📚 References

- See `orchestrator/cache_optimizer.py` for implementation details
- Run `python -m orchestrator cache-stats --help` for CLI options
- Check `output.log` for cache hit/miss logs during execution
