"""
Capability Usage Logger
=======================
Tracks usage of system capabilities for analytics and debugging.

Logs capability invocations with timestamps, context, and results.
Integrates with the existing telemetry system.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional
from threading import Lock

from .log_config import get_logger

logger = get_logger(__name__)


class CapabilityType(Enum):
    """Types of capabilities that can be logged."""
    # Model Routing
    ROUTING_DECISION = auto()
    FALLBACK_TRIGGERED = auto()
    COST_OPTIMIZATION = auto()
    
    # Task Execution
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    
    # Quality Assurance
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    CRITIQUE_ROUND = auto()
    REVISION_ROUND = auto()
    
    # Project Management
    PROJECT_CREATED = auto()
    PROJECT_RESUMED = auto()
    PROJECT_EXPORTED = auto()
    
    # Analysis
    CODEBASE_ANALYSIS = auto()
    IMPROVEMENT_SUGGESTION = auto()
    ARCHITECTURE_CHECK = auto()
    
    # Dashboard
    DASHBOARD_START = auto()
    DASHBOARD_COMMAND = auto()
    WEBSOCKET_EVENT = auto()
    
    # Cache & Performance
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    SEMANTIC_CACHE_USED = auto()


@dataclass
class CapabilityEvent:
    """A single capability usage event."""
    timestamp: str
    capability: str
    task_type: Optional[str]
    model: Optional[str]
    project_id: Optional[str]
    duration_ms: float
    success: bool
    details: dict[str, Any]
    
    @classmethod
    def create(
        cls,
        capability: CapabilityType,
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        details: Optional[dict] = None,
    ) -> CapabilityEvent:
        """Factory method to create an event with current timestamp."""
        return cls(
            timestamp=datetime.utcnow().isoformat() + "Z",
            capability=capability.name,
            task_type=task_type,
            model=model,
            project_id=project_id,
            duration_ms=duration_ms,
            success=success,
            details=details or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CapabilityLogger:
    """Thread-safe logger for capability usage events."""
    
    _instance: Optional[CapabilityLogger] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> CapabilityLogger:
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the capability logger.
        
        Args:
            log_dir: Directory for capability logs. Defaults to ./logs/capabilities
        """
        if self._initialized:
            return
            
        self._initialized = True
        self._buffer: list[CapabilityEvent] = []
        self._buffer_lock = Lock()
        self._flush_interval = 10  # Flush every N events
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path.cwd() / "logs" / "capabilities"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file (rotates daily)
        self._current_file = self._get_log_file()
        
        logger.info(f"CapabilityLogger initialized: {self.log_dir}")
    
    def _get_log_file(self) -> Path:
        """Get current log file path (date-based rotation)."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"capabilities_{date_str}.jsonl"
    
    def log(
        self,
        capability: CapabilityType,
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        details: Optional[dict] = None,
    ) -> None:
        """Log a capability usage event.
        
        Args:
            capability: Type of capability used
            task_type: Optional task type (e.g., 'code_generation')
            model: Optional model name (e.g., 'GPT_4O')
            project_id: Optional project identifier
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            details: Additional context-specific details
        """
        event = CapabilityEvent.create(
            capability=capability,
            task_type=task_type,
            model=model,
            project_id=project_id,
            duration_ms=duration_ms,
            success=success,
            details=details,
        )
        
        # Add to buffer
        with self._buffer_lock:
            self._buffer.append(event)
            should_flush = len(self._buffer) >= self._flush_interval
        
        # Also log to main logger at debug level
        logger.debug(f"Capability: {event.capability} | Task: {task_type} | Model: {model}")
        
        # Flush if needed
        if should_flush:
            self.flush()
    
    def flush(self) -> None:
        """Flush buffered events to disk."""
        with self._buffer_lock:
            if not self._buffer:
                return
            events_to_write = self._buffer.copy()
            self._buffer.clear()
        
        # Check if we need to rotate file
        current_file = self._get_log_file()
        
        try:
            with open(current_file, "a", encoding="utf-8") as f:
                for event in events_to_write:
                    json_line = json.dumps(event.to_dict(), default=str)
                    f.write(json_line + "\n")
        except Exception as e:
            logger.error(f"Failed to write capability log: {e}")
    
    def get_stats(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> dict[str, Any]:
        """Get usage statistics for capabilities.
        
        Args:
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            
        Returns:
            Dictionary with capability usage statistics
        """
        stats = {
            "total_events": 0,
            "by_capability": {},
            "by_model": {},
            "by_task_type": {},
            "success_rate": 0.0,
            "avg_duration_ms": 0.0,
        }
        
        total_duration = 0.0
        success_count = 0
        
        # Read all log files
        for log_file in self.log_dir.glob("capabilities_*.jsonl"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event_data = json.loads(line)
                            
                            # Time filtering
                            event_time = datetime.fromisoformat(
                                event_data["timestamp"].replace("Z", "+00:00")
                            ).timestamp()
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue
                            
                            # Update stats
                            stats["total_events"] += 1
                            
                            cap = event_data["capability"]
                            stats["by_capability"][cap] = stats["by_capability"].get(cap, 0) + 1
                            
                            model = event_data.get("model")
                            if model:
                                stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
                            
                            task = event_data.get("task_type")
                            if task:
                                stats["by_task_type"][task] = stats["by_task_type"].get(task, 0) + 1
                            
                            if event_data.get("success"):
                                success_count += 1
                            
                            total_duration += event_data.get("duration_ms", 0)
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
            except Exception as e:
                logger.warning(f"Error reading {log_file}: {e}")
        
        # Calculate aggregates
        if stats["total_events"] > 0:
            stats["success_rate"] = success_count / stats["total_events"]
            stats["avg_duration_ms"] = total_duration / stats["total_events"]
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """Remove log files older than specified days.
        
        Args:
            days_to_keep: Number of days to retain
            
        Returns:
            Number of files deleted
        """
        cutoff = time.time() - (days_to_keep * 24 * 3600)
        deleted = 0
        
        for log_file in self.log_dir.glob("capabilities_*.jsonl"):
            try:
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {log_file}: {e}")
        
        logger.info(f"Cleaned up {deleted} old capability log files")
        return deleted


# Global singleton instance
_capability_logger: Optional[CapabilityLogger] = None


def get_capability_logger() -> CapabilityLogger:
    """Get the global capability logger instance."""
    global _capability_logger
    if _capability_logger is None:
        _capability_logger = CapabilityLogger()
    return _capability_logger


def log_capability(
    capability: CapabilityType,
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    project_id: Optional[str] = None,
    duration_ms: float = 0.0,
    success: bool = True,
    details: Optional[dict] = None,
) -> None:
    """Convenience function to log a capability event."""
    get_capability_logger().log(
        capability=capability,
        task_type=task_type,
        model=model,
        project_id=project_id,
        duration_ms=duration_ms,
        success=success,
        details=details,
    )


# Decorator for automatic capability logging
def log_capability_use(
    capability: CapabilityType,
    task_type_arg: Optional[str] = None,
    model_arg: Optional[str] = None,
):
    """Decorator to automatically log capability usage.
    
    Args:
        capability: The capability being used
        task_type_arg: Argument name containing task type
        model_arg: Argument name containing model name
        
    Example:
        @log_capability_use(CapabilityType.TASK_STARTED, task_type_arg="task_type")
        def start_task(task_type: str, model: str):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract task_type and model from kwargs if specified
                task_type = kwargs.get(task_type_arg) if task_type_arg else None
                model = kwargs.get(model_arg) if model_arg else None
                
                log_capability(
                    capability=capability,
                    task_type=task_type,
                    model=model,
                    duration_ms=duration_ms,
                    success=success,
                )
        
        return wrapper
    return decorator
