# Αρχιτεκτονικές Βελτιώσεις: Από Monolith σε Modern Distributed

## Επισκόπηση

Η τρέχουσα αρχιτεκτονική είναι **modular monolith** με καλή οργάνωση αλλά ευκαιρίες για βελτίωση σε:
- **Loose coupling** (σφιχτά coupled components)
- **Observability** (tracing σε όλα τα επίπεδα)
- **Scalability** (horizontal scaling για enterprise)
- **Resilience** (failure isolation)

---

## 1. Event-Driven Architecture (EDA)

### Τρέχουσα Κατάσταση
```python
# hooks.py - Fire-and-forget, sync callbacks
class HookRegistry:
    def fire(self, event, **kwargs):
        for cb in self._hooks.get(key, []):
            cb(**kwargs)  # Sync, no persistence, no replay
```

### Προτεινόμενη Βελτίωση
```python
# orchestrator/events.py
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
import json

@dataclass(frozen=True)
class DomainEvent:
    event_id: str
    event_type: str
    aggregate_id: str
    timestamp: datetime
    payload: dict
    metadata: dict  # tracing, correlation ids

class EventBus(Protocol):
    async def publish(self, event: DomainEvent) -> None: ...
    async def subscribe(self, event_type: str, handler: Callable) -> None: ...

# Implementations:
# - InMemoryEventBus (dev/testing)
# - RedisEventBus (production, single instance)
# - KafkaEventBus (enterprise, multi-instance)
# - SQLiteEventBus (edge/embedded)
```

### Οφέλη
1. **Persistence**: Events αποθηκεύονται → replay για debugging
2. **Async Processing**: Handlers τρέχουν παράλληλα
3. **Loose Coupling**: Components δεν ξέρουν ο ένας για τον άλλον
4. **Audit Trail**: Πλήρες ιστορικό όλων των αποφάσεων

---

## 2. CQRS (Command Query Responsibility Segregation)

### Τρέχουσα Κατάσταση
Το ίδιο μοντέλο για writes και reads:
```python
class FeedbackLoop:
    async def record_outcome(self, outcome):  # Write
        ...
    def get_model_score(self, model, task_type):  # Read
        ...  # Ίδιο data structure
```

### Προτεινόμενη Βελτίωση
```python
# Write Model (Optimized for consistency)
class FeedbackCommandHandler:
    async def handle(self, cmd: RecordOutcomeCommand):
        outcome = ProductionOutcome(...)
        await self.event_store.append(outcome)
        await self.projector.project(outcome)  # Update read models

# Read Models (Optimized for queries)
class ModelPerformanceReadModel:
    """Pre-computed, materialized view"""
    def get_score(self, model, task_type) -> float:
        return self.cache.get(f"{model}:{task_type}")
    
    def get_leaderboard(self) -> List[LeaderboardEntry]:
        return self.db.query("SELECT * FROM model_leaderboard ORDER BY score DESC")
```

### Read Models να Δημιουργούνται

| Read Model | Data Source | Update Frequency |
|------------|-------------|------------------|
| `ModelLeaderboardView` | Production outcomes | Real-time (event-driven) |
| `CodebaseSimilarityView` | Codebase fingerprints | On-demand |
| `CostEfficiencyView` | Telemetry + costs | Every 5 minutes |
| `UserActivityView` | All events | Every minute |

---

## 3. Saga Pattern για Distributed Transactions

### Πρόβλημα
Ένα project έχει πολλά tasks που εξαρτώνται το ένα από το άλλο. Αν ένα αποτύχει, τι γίνεται με τα προηγούμενα;

### Λύση
```python
# orchestrator/sagas.py
from enum import Enum, auto

class SagaState(Enum):
    STARTED = auto()
    TASK_COMPLETED = auto()
    COMPENSATING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class Saga:
    """Orchestrates a multi-step process with rollback capability"""
    saga_id: str
    project_id: str
    steps: List[SagaStep]
    state: SagaState
    compensation_log: List[CompensationAction]
    
    async def execute(self) -> SagaResult:
        for step in self.steps:
            try:
                result = await step.execute()
                self.compensation_log.append(step.compensation_action)
            except Exception as e:
                await self.compensate()  # Rollback
                return SagaResult.failed(e)
        return SagaResult.success()
    
    async def compensate(self):
        """Undo all completed steps in reverse order"""
        for action in reversed(self.compensation_log):
            await action.execute()
```

### Saga για Project Execution
```python
class ProjectExecutionSaga:
    steps = [
        SagaStep(
            name="enhance_project",
            action=EnhanceProjectAction(),
            compensation=DeleteEnhancementAction(),
        ),
        SagaStep(
            name="decompose",
            action=DecomposeAction(),
            compensation=DeleteTasksAction(),
        ),
        SagaStep(
            name="execute_tasks",
            action=ExecuteTasksAction(),
            compensation=MarkTasksFailedAction(),
        ),
    ]
```

---

## 4. Dependency Injection Container

### Τρέχουσα Κατάσταση
```python
class Orchestrator:
    def __init__(self, budget=None, cache=None, state_manager=None, ...):
        self.budget = budget or Budget()
        self.cache = cache or DiskCache()
        # 15+ dependencies manually wired
```

### Προτεινόμενη Βελτίωση
```python
# orchestrator/container.py
from dataclasses import dataclass
from typing import TypeVar, Type

T = TypeVar('T')

class Container:
    """Simple DI container with lifecycle management"""
    
    def __init__(self):
        self._registrations: Dict[Type, Registration] = {}
        self._singletons: Dict[Type, Any] = {}
    
    def register_singleton(self, interface: Type[T], implementation: T):
        self._registrations[interface] = Registration(
            implementation=implementation,
            lifecycle=Lifecycle.SINGLETON
        )
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]):
        self._registrations[interface] = Registration(
            factory=factory,
            lifecycle=Lifecycle.TRANSIENT
        )
    
    def resolve(self, interface: Type[T]) -> T:
        reg = self._registrations[interface]
        if reg.lifecycle == Lifecycle.SINGLETON:
            if interface not in self._singletons:
                self._singletons[interface] = reg.factory()
            return self._singletons[interface]
        return reg.factory()

# Usage
container = Container()
container.register_singleton(Cache, lambda: DiskCache())
container.register_singleton(EventBus, lambda: RedisEventBus())
container.register_factory(Orchestrator, lambda: Orchestrator(container))

orch = container.resolve(Orchestrator)
```

### Οφέλη
- **Testability**: Easy mocking με `container.register_mock(Cache, MockCache())`
- **Configuration**: DI based on environment
- **Lifecycle Management**: Singletons vs Transient vs Scoped

---

## 5. Plugin Isolation & Sandboxing

### Τρέχουσα Κατάσταση
Plugins τρέχουν στο ίδιο process:
```python
# plugins.py
class PluginRegistry:
    def register(self, plugin: Plugin):
        self._plugins.append(plugin)  # Same memory space
```

### Προτεινόμενη Βελτίωση
```python
# orchestrator/plugin_runtime.py
import multiprocessing
import tempfile

class IsolatedPluginRuntime:
    """Runs plugins in separate processes with resource limits"""
    
    def __init__(self):
        self._pool = multiprocessing.Pool(processes=4)
        self._resource_limits = {
            'memory_mb': 512,
            'cpu_percent': 50,
            'timeout_seconds': 30,
        }
    
    async def execute_plugin(
        self,
        plugin: Plugin,
        method: str,
        *args,
        **kwargs
    ) -> PluginResult:
        """Execute plugin method in isolated process"""
        
        # Create temporary sandbox
        with tempfile.TemporaryDirectory() as sandbox:
            # Apply seccomp/SELinux restrictions
            # Limit filesystem access to sandbox
            # Limit network access (whitelist only)
            
            result = await asyncio.wait_for(
                self._pool.apply_async(
                    _run_plugin_method,
                    (plugin, method, args, kwargs)
                ).get(),
                timeout=self._resource_limits['timeout_seconds']
            )
            
            return result
    
    def _run_plugin_method(plugin, method, args, kwargs):
        """Runs in separate process"""
        import resource
        # Apply resource limits
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))
        
        handler = getattr(plugin, method)
        return handler(*args, **kwargs)
```

### Security Benefits
- **Memory isolation**: Plugin crash δεν καταρρέει το core
- **Resource limits**: Plugin δεν μπορεί να καταναλώσει όλους τους πόρους
- **Filesystem isolation**: Plugin βλέπει μόνο temp directory
- **Network isolation**: Whitelist-based network access

---

## 6. Multi-Layer Caching Strategy

### Τρέχουσα Κατάσταση
```python
# cache.py - Single layer
class DiskCache:
    def get(self, key): ...
    def set(self, key, value): ...
```

### Προτεινόμενη Βελτίωη
```python
# orchestrator/caching.py
from enum import Enum

class CacheLevel(Enum):
    L1_MEMORY = 1    # In-process dict (fastest, smallest)
    L2_REDIS = 2     # Shared memory (fast, medium)
    L3_DISK = 3      # Local disk (slow, large)
    L4_S3 = 4        # Object storage (slowest, unlimited)

class MultiLayerCache:
    """Hierarchical cache with automatic promotion/demotion"""
    
    def __init__(self):
        self.l1 = LRUCache(maxsize=1000)      # 1MB
        self.l2 = RedisCache(maxsize=10000)   # 100MB
        self.l3 = DiskCache(maxsize=1_000_000)  # 1GB
    
    async def get(self, key: str) -> Optional[Any]:
        # Try L1 → L2 → L3
        for cache in [self.l1, self.l2, self.l3]:
            value = await cache.get(key)
            if value is not None:
                # Promote to faster cache
                await self._promote(key, value, cache)
                return value
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        level: CacheLevel = CacheLevel.L2_REDIS
    ):
        # Write to specified level and all slower levels
        if level.value <= CacheLevel.L1_MEMORY.value:
            await self.l1.set(key, value, ttl=timedelta(minutes=1))
        if level.value <= CacheLevel.L2_REDIS.value:
            await self.l2.set(key, value, ttl=ttl or timedelta(hours=1))
        if level.value <= CacheLevel.L3_DISK.value:
            await self.l3.set(key, value, ttl=ttl or timedelta(days=7))
    
    async def invalidate(self, pattern: str):
        """Invalidate all keys matching pattern (for cache coherence)"""
        await asyncio.gather(
            self.l1.invalidate(pattern),
            self.l2.invalidate(pattern),
            self.l3.invalidate(pattern),
        )
```

---

## 7. Streaming Pipeline για Large Projects

### Πρόβλημα
Projects με 50+ tasks block το UI. Χρειάζεται streaming progress.

### Λύση
```python
# orchestrator/streaming.py
from typing import AsyncIterator

class ProjectPipeline:
    """Async streaming pipeline for project execution"""
    
    def __init__(self):
        self.stages = [
            DecomposeStage(),
            RouteStage(),
            ExecuteStage(max_parallel=5),
            ValidateStage(),
            FeedbackStage(),
        ]
    
    async def execute_streaming(
        self,
        project: ProjectSpec
    ) -> AsyncIterator[PipelineEvent]:
        """Yield events as they happen"""
        
        context = PipelineContext(project)
        
        for stage in self.stages:
            yield StageStartedEvent(stage=stage.name)
            
            async for event in stage.process_streaming(context):
                yield event
                
                if isinstance(event, TaskCompletedEvent):
                    # Check if we can trigger dependent tasks
                    await self._trigger_ready_tasks(context)
            
            yield StageCompletedEvent(stage=stage.name)
    
    async def _trigger_ready_tasks(self, context):
        """Trigger tasks whose dependencies are complete"""
        ready = [
            task for task in context.pending_tasks
            if all(dep in context.completed_tasks for dep in task.dependencies)
        ]
        
        for task in ready:
            asyncio.create_task(self._execute_task(task, context))

# Usage in WebSocket handler
async def project_websocket(websocket, project_id):
    pipeline = ProjectPipeline()
    project = await load_project(project_id)
    
    async for event in pipeline.execute_streaming(project):
        await websocket.send_json({
            "type": event.type,
            "data": event.to_dict(),
        })
```

---

## 8. Configuration Management με Validation

### Τρέχουσα Κατάσταση
```python
# .env files, scattered config
budget = float(os.environ.get("BUDGET", "5.0"))
```

### Προτεινόμενη Βελτίωση
```python
# orchestrator/config.py
from pydantic import BaseSettings, Field, validator
from typing import Literal

class OrchestratorConfig(BaseSettings):
    """Type-safe, validated configuration"""
    
    # Core settings
    default_budget: float = Field(5.0, gt=0, le=1000)
    max_concurrency: int = Field(3, ge=1, le=50)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Feature flags
    enable_feedback_loop: bool = True
    enable_outcome_router: bool = True
    enable_plugin_isolation: bool = False  # Experimental
    
    # Cache configuration
    cache_backend: Literal["memory", "redis", "disk"] = "disk"
    cache_ttl_seconds: int = Field(3600, ge=60)
    
    # Plugin security
    plugin_allow_network: bool = True
    plugin_allow_filesystem: bool = True
    plugin_timeout_seconds: int = Field(30, ge=1, le=300)
    
    # Provider settings
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    @validator("deepseek_api_key", "openai_api_key")
    def validate_key_format(cls, v):
        if v and not v.startswith(("sk-", "sk-proj-")):
            raise ValueError("Invalid API key format")
        return v
    
    class Config:
        env_prefix = "ORCHESTRATOR_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
config = OrchestratorConfig()
if config.enable_feedback_loop:
    feedback_loop = FeedbackLoop()
```

---

## 9. Health Checks & Readiness Probes

```python
# orchestrator/health.py
from enum import Enum
from typing import Dict, List
import asyncio

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    check: Callable[[], Awaitable[HealthStatus]]
    timeout: float = 5.0

class HealthMonitor:
    """Kubernetes-style health checks"""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self._status: Dict[str, HealthStatus] = {}
    
    def register(self, check: HealthCheck):
        self.checks.append(check)
    
    async def run_checks(self) -> Dict[str, HealthStatus]:
        results = await asyncio.gather(*[
            self._run_check(check)
            for check in self.checks
        ], return_exceptions=True)
        
        return {
            check.name: status if not isinstance(status, Exception) else HealthStatus.UNHEALTHY
            for check, status in zip(self.checks, results)
        }
    
    async def _run_check(self, check: HealthCheck) -> HealthStatus:
        try:
            return await asyncio.wait_for(check.check(), timeout=check.timeout)
        except asyncio.TimeoutError:
            return HealthStatus.UNHEALTHY

# Register checks
health = HealthMonitor()
health.register(HealthCheck("cache", lambda: check_cache()))
health.register(HealthCheck("event_bus", lambda: check_event_bus()))
health.register(HealthCheck("providers", lambda: check_providers()))
```

---

## 10. Unified Observability (OpenTelemetry)

```python
# orchestrator/observability.py
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from functools import wraps

class Observability:
    """Unified logging, metrics, and tracing"""
    
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Custom metrics
        self.task_counter = self.meter.create_counter(
            "orchestrator.tasks",
            description="Number of tasks executed",
        )
        self.cost_histogram = self.meter.create_histogram(
            "orchestrator.cost",
            description="Cost per task",
            unit="USD",
        )
    
    def trace(self, name: str, attributes: dict = None):
        """Decorator for automatic tracing"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    if attributes:
                        span.set_attributes(attributes)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        raise
            return wrapper
        return decorator

# Usage
obs = Observability()

class Orchestrator:
    @obs.trace("execute_task", {"task_type": "code_gen"})
    async def _execute_task(self, task: Task) -> TaskResult:
        self.observability.task_counter.add(1, {"type": task.task_type.value})
        ...
```

---

## Συνοπτικός Πίνακας Βελτιώσεων

| Βελτίωση | Complexity | Impact | Priority |
|----------|-----------|--------|----------|
| Event-Driven Architecture | High | Very High | 🔴 Critical |
| CQRS | Medium | High | 🟡 High |
| Saga Pattern | Medium | High | 🟡 High |
| Dependency Injection | Low | Medium | 🟢 Medium |
| Plugin Isolation | High | High | 🟡 High |
| Multi-Layer Cache | Low | Medium | 🟢 Medium |
| Streaming Pipeline | Medium | Very High | 🔴 Critical |
| Config Management | Low | Medium | 🟢 Medium |
| Health Checks | Low | High | 🟡 High |
| OpenTelemetry | Low | High | 🟡 High |

---

## Επόμενα Βήματα

1. **Week 1**: Implement Event Bus + migrate hooks
2. **Week 2**: Add CQRS read models for leaderboard
3. **Week 3**: Streaming pipeline for project execution
4. **Week 4**: OpenTelemetry integration
5. **Week 5-6**: Plugin isolation (security hardening)

Αυτές οι αλλαγές μετατρέπουν το orchestrator από **modular monolith** σε **modern event-driven system** με enterprise-grade observability, resilience, και scalability.
