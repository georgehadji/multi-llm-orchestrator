"""
Tracing — OpenTelemetry tracing integration
==========================================
Module for integrating OpenTelemetry tracing to monitor and analyze orchestrator performance.

Pattern: Observer
Async: Yes — for I/O-bound tracing operations
Layer: L6 Observability

Usage:
    from orchestrator.tracing import Tracer
    tracer = Tracer(service_name="orchestrator")
    with tracer.trace("task_execution", {"task_id": "123"}):
        # Execute task
        result = await orchestrator.run_task(task)
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

from .models import Model

logger = logging.getLogger("orchestrator.tracing")


class Span:
    """Represents a single tracing span."""
    
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.start_time = None
        self.end_time = None
        self.parent_span = None
        self.status = "UNSET"  # UNSET, OK, ERROR
        self.events = []
    
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": self._get_timestamp()
        })
    
    def set_status(self, status: str):
        """Set the status of the span."""
        self.status = status
    
    def _get_timestamp(self) -> float:
        """Get the current timestamp."""
        import time
        return time.time()


class Tracer:
    """OpenTelemetry-compatible tracer for the orchestrator."""

    def __init__(self, service_name: str = "orchestrator", enabled: bool = True):
        """Initialize the tracer."""
        self.service_name = service_name
        self.enabled = enabled
        self.active_spans: list[Span] = []
        self.span_buffer: list[Span] = []  # Completed spans waiting to be exported
        self.exporter = None  # Will be set when exporter is configured
    
    @contextmanager
    def trace(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator[Span, None, None]:
        """
        Create and manage a tracing span.
        
        Args:
            name: Name of the span
            attributes: Attributes to attach to the span
            
        Yields:
            Span: The created span
        """
        if not self.enabled:
            # If tracing is disabled, yield a dummy span
            dummy_span = Span(name, attributes)
            yield dummy_span
            return
        
        # Create a new span
        span = Span(name, attributes)
        span.start_time = span._get_timestamp()
        
        # Set parent span if there's an active one
        if self.active_spans:
            span.parent_span = self.active_spans[-1]
        
        # Add to active spans
        self.active_spans.append(span)
        
        try:
            yield span
            span.set_status("OK")
        except Exception as e:
            span.set_status("ERROR")
            span.add_event("exception", {"exception.message": str(e)})
            raise
        finally:
            # End the span
            span.end_time = span._get_timestamp()
            
            # Remove from active spans
            self.active_spans.pop()
            
            # Add to buffer for export
            self.span_buffer.append(span)
            
            # Export if buffer is full (simple batching)
            if len(self.span_buffer) >= 10:
                self._export_spans()
    
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Span:
        """
        Start a new span without using a context manager.
        
        Args:
            name: Name of the span
            attributes: Attributes to attach to the span
            
        Returns:
            Span: The started span
        """
        if not self.enabled:
            return Span(name, attributes)
        
        span = Span(name, attributes)
        span.start_time = span._get_timestamp()
        
        # Set parent span if there's an active one
        if self.active_spans:
            span.parent_span = self.active_spans[-1]
        
        self.active_spans.append(span)
        return span
    
    def end_span(self, span: Span, status: str = "OK"):
        """
        End a span that was started with start_span.

        Args:
            span: The span to end
            status: Status to set on the span
        """
        if not self.enabled:
            return

        span.end_time = span._get_timestamp()
        span.set_status(status)

        # Remove from active spans if it's the current one
        if self.active_spans and self.active_spans[-1] == span:
            self.active_spans.pop()

        # Add to buffer for export
        self.span_buffer.append(span)

        # Export if buffer is full
        if len(self.span_buffer) >= 10:
            self._export_spans()

    @contextmanager
    def start_as_current_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Generator[Span, None, None]:
        """
        Start a span and make it the current active span (OpenTelemetry-style API).
        
        This is an alias for trace() to maintain compatibility with OpenTelemetry patterns.

        Args:
            name: Name of the span
            attributes: Attributes to attach to the span

        Yields:
            Span: The created span
        """
        with self.trace(name, attributes) as span:
            yield span
    
    def add_tracing_attributes(self, **kwargs):
        """
        Add attributes to the currently active span.
        
        Args:
            **kwargs: Attributes to add
        """
        if not self.enabled or not self.active_spans:
            return
        
        current_span = self.active_spans[-1]
        for key, value in kwargs.items():
            current_span.set_attribute(key, value)
    
    def record_exception(self, exception: Exception, attributes: Optional[Dict[str, Any]] = None):
        """
        Record an exception in the currently active span.
        
        Args:
            exception: The exception to record
            attributes: Additional attributes
        """
        if not self.enabled or not self.active_spans:
            return
        
        current_span = self.active_spans[-1]
        attrs = attributes or {}
        attrs["exception.message"] = str(exception)
        attrs["exception.type"] = type(exception).__name__
        
        current_span.add_event("exception", attrs)
        current_span.set_status("ERROR")
    
    def configure_exporter(self, exporter):
        """
        Configure the span exporter.
        
        Args:
            exporter: An exporter object with an export(spans) method
        """
        self.exporter = exporter
        logger.info("Tracing exporter configured")
    
    def _export_spans(self):
        """Export buffered spans using the configured exporter."""
        if not self.exporter or not self.span_buffer:
            return
        
        try:
            # Export spans
            self.exporter.export(self.span_buffer.copy())
            
            # Clear the buffer
            self.span_buffer.clear()
            
            logger.debug(f"Exported {len(self.span_buffer)} spans")
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
    
    async def flush(self):
        """Flush any remaining spans in the buffer."""
        if self.span_buffer:
            self._export_spans()
    
    def get_active_span_count(self) -> int:
        """Get the number of currently active spans."""
        return len(self.active_spans)
    
    def get_buffered_span_count(self) -> int:
        """Get the number of spans in the buffer waiting for export."""
        return len(self.span_buffer)
    
    def trace_model_call(self, model: Model, prompt_tokens: int, response_tokens: int):
        """
        Convenience method to trace a model call.
        
        Args:
            model: The model that was called
            prompt_tokens: Number of input tokens
            response_tokens: Number of output tokens
        """
        if not self.enabled:
            return
        
        if self.active_spans:
            current_span = self.active_spans[-1]
            current_span.set_attribute("llm.model", model.value)
            current_span.set_attribute("llm.request.token_count", prompt_tokens)
            current_span.set_attribute("llm.response.token_count", response_tokens)
            current_span.set_attribute("llm.vendor", model.get_provider().value)


# Global tracer instance for convenience
_global_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "orchestrator") -> Tracer:
    """
    Get the global tracer instance.
    
    Args:
        service_name: Name of the service for the tracer
        
    Returns:
        Tracer: The global tracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(service_name)
    return _global_tracer


def set_global_tracer(tracer: Tracer):
    """
    Set the global tracer instance.
    
    Args:
        tracer: The tracer instance to set as global
    """
    global _global_tracer
    _global_tracer = tracer


def trace_function(tracer: Optional[Tracer] = None, span_name: Optional[str] = None):
    """
    Decorator to trace a function.
    
    Args:
        tracer: Tracer instance to use (defaults to global tracer)
        span_name: Name for the span (defaults to function name)
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            t = tracer or get_tracer()
            name = span_name or func.__name__
            
            with t.trace(name, {"function.args.count": len(args), "function.kwargs.count": len(kwargs)}):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            t = tracer or get_tracer()
            name = span_name or func.__name__
            
            with t.trace(name, {"function.args.count": len(args), "function.kwargs.count": len(kwargs)}):
                return func(*args, **kwargs)
        
        # Return the appropriate wrapper based on whether the function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ConsoleExporter:
    """Simple console exporter for debugging purposes."""
    
    def export(self, spans: list[Span]):
        """Export spans to console."""
        for span in spans:
            duration = (span.end_time - span.start_time) * 1000 if span.end_time and span.start_time else 0
            print(f"[TRACE] {span.name} - {duration:.2f}ms - Status: {span.status}")
            for attr_key, attr_val in span.attributes.items():
                print(f"  {attr_key}: {attr_val}")


class InMemoryExporter:
    """In-memory exporter for testing purposes."""
    
    def __init__(self):
        self.spans = []
    
    def export(self, spans: list[Span]):
        """Export spans to memory."""
        self.spans.extend(spans)
    
    def get_spans(self) -> list[Span]:
        """Get all exported spans."""
        return self.spans
    
    def clear(self):
        """Clear all exported spans."""
        self.spans.clear()


# ---------------------------------------------------------------------------
# BUG-IMPORT-001 FIX: stubs that engine.py and api_clients.py import but were
# missing from this module, causing an ImportError at startup.
# ---------------------------------------------------------------------------

@dataclass
class TracingConfig:
    """Minimal tracing configuration passed to Orchestrator.__init__."""
    enabled: bool = True
    service_name: str = "orchestrator"
    export_endpoint: str = ""


def configure_tracing(cfg: TracingConfig) -> None:
    """Apply a TracingConfig to the global tracer."""
    tracer = get_tracer(cfg.service_name)
    tracer.enabled = cfg.enabled


def traced_task(task_id: str, task_type: str = ""):
    """Context manager that wraps a task execution in a tracing span."""
    return get_tracer().trace(f"task.{task_type}", {"task.id": task_id, "task.type": task_type})


def traced_llm_call(model: str, operation: str = "api_call"):
    """Context manager that wraps an LLM API call in a tracing span."""
    return get_tracer().trace(f"llm.{operation}", {"llm.model": model, "llm.operation": operation})


def traced_policy_check(policy_id: str, check_type: str = ""):
    """Context manager that wraps a policy check in a tracing span."""
    return get_tracer().trace(
        f"policy.{check_type}", {"policy.id": policy_id, "policy.check_type": check_type}
    )


def traced_remediation(remediation_id: str, action: str = ""):
    """Context manager that wraps a remediation action in a tracing span."""
    return get_tracer().trace(
        f"remediation.{action}",
        {"remediation.id": remediation_id, "remediation.action": action},
    )
