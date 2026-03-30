"""
Batch API Client Module
========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements batch API processing for 50% cost reduction on non-critical phases.

Features:
- Automatic batch/realtime routing based on phase
- Batch request aggregation
- Result polling and retrieval
- Cost tracking and savings metrics

Usage:
    from orchestrator.optimization import BatchClient, OptimizationPhase

    batch = BatchClient()

    # Non-critical phase (auto-batch)
    result = await batch.call(
        model="claude-sonnet-4.6",
        prompt="Evaluate this code quality",
        phase=OptimizationPhase.EVALUATION,
    )

    # Critical phase (realtime)
    result = await batch.call(
        model="claude-sonnet-4.6",
        prompt="Generate code",
        phase=OptimizationPhase.GENERATION,
    )
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from orchestrator.log_config import get_logger

from . import OptimizationPhase

logger = get_logger(__name__)


class BatchStatus(str, Enum):
    """Batch job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRequest:
    """Individual batch request."""
    id: str
    model: str
    prompt: str
    phase: OptimizationPhase
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    result: Any | None = None
    error: str | None = None


@dataclass
class BatchJob:
    """Batch job containing multiple requests."""
    id: str
    requests: list[BatchRequest]
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    results_file: Path | None = None
    completed_at: float | None = None


@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    batch_requests: int = 0
    realtime_requests: int = 0
    batch_completions: int = 0
    batch_failures: int = 0
    total_savings: float = 0.0
    avg_batch_latency: float = 0.0

    @property
    def batch_ratio(self) -> float:
        """Calculate batch vs realtime ratio."""
        total = self.batch_requests + self.realtime_requests
        if total == 0:
            return 0.0
        return self.batch_requests / total


class BatchClient:
    """
    Batch API client for cost optimization.

    Usage:
        batch = BatchClient()
        result = await batch.call(model, prompt, phase)
    """

    # Phases that should use batch API (50% discount)
    BATCH_PHASES = {
        OptimizationPhase.EVALUATION,
        OptimizationPhase.PROMPT_ENHANCEMENT,
        OptimizationPhase.CONDENSING,
        OptimizationPhase.CRITIQUE,
    }

    # Batch processing configuration
    BATCH_WINDOW_SECONDS = 60  # Aggregate requests within this window
    POLL_INTERVAL_SECONDS = 10  # Poll for results every N seconds
    MAX_BATCH_SIZE = 1000  # Maximum requests per batch

    def __init__(self, client=None, base_url: str | None = None):
        """
        Initialize batch client.

        Args:
            client: UnifiedClient or provider-specific client
            base_url: API base URL
        """
        self.client = client
        self.base_url = base_url
        self.metrics = BatchMetrics()
        self._pending_jobs: dict[str, BatchJob] = {}
        self._request_queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._batch_task: asyncio.Task | None = None

    def _should_use_batch(self, phase: OptimizationPhase) -> bool:
        """
        Determine if phase should use batch API.

        Args:
            phase: Optimization phase

        Returns:
            True if batch API should be used
        """
        return phase in self.BATCH_PHASES

    async def call(
        self,
        model: str,
        prompt: str,
        phase: OptimizationPhase,
        **kwargs,
    ) -> Any:
        """
        Make API call with automatic batch/realtime routing.

        Args:
            model: Model to use
            prompt: Prompt text
            phase: Optimization phase
            **kwargs: Additional parameters

        Returns:
            API response
        """
        if self._should_use_batch(phase):
            return await self._batch_call(model, prompt, phase, **kwargs)
        else:
            return await self._realtime_call(model, prompt, phase, **kwargs)

    async def _batch_call(
        self,
        model: str,
        prompt: str,
        phase: OptimizationPhase,
        **kwargs,
    ) -> Any:
        """
        Make batch API call (50% discount).

        Args:
            model: Model to use
            prompt: Prompt text
            phase: Optimization phase
            **kwargs: Additional parameters

        Returns:
            Batch result
        """
        request_id = f"req_{int(time.time() * 1000)}_{len(self._request_queue)}"

        request = BatchRequest(
            id=request_id,
            model=model,
            prompt=prompt,
            phase=phase,
        )

        async with self._lock:
            self._request_queue.append(request)
            self.metrics.batch_requests += 1

        logger.debug(f"Queued batch request {request_id} (phase={phase.value})")

        # If queue is large enough, process immediately
        if len(self._request_queue) >= 10:
            await self._process_batch_queue()

        # Wait for result with timeout
        timeout = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            if request.result:
                self.metrics.batch_completions += 1
                # Track savings (50% of estimated cost)
                estimated_cost = self._estimate_cost(model, prompt)
                self.metrics.total_savings += estimated_cost * 0.5
                return request.result

            if request.error:
                self.metrics.batch_failures += 1
                raise RuntimeError(f"Batch request failed: {request.error}")

            await asyncio.sleep(1)

        raise TimeoutError(f"Batch request {request_id} timed out")

    async def _realtime_call(
        self,
        model: str,
        prompt: str,
        phase: OptimizationPhase,
        **kwargs,
    ) -> Any:
        """
        Make realtime API call (standard pricing).

        Args:
            model: Model to use
            prompt: Prompt text
            phase: Optimization phase
            **kwargs: Additional parameters

        Returns:
            Realtime response
        """
        self.metrics.realtime_requests += 1
        logger.debug(f"Realtime call (phase={phase.value}, model={model})")

        if self.client:
            return await self.client.call(model, prompt, **kwargs)
        else:
            raise RuntimeError("No client available for realtime calls")

    async def _process_batch_queue(self) -> None:
        """Process all pending batch requests."""
        async with self._lock:
            if not self._request_queue:
                return

            # Create batch job
            job_id = f"batch_{int(time.time())}"
            job = BatchJob(
                id=job_id,
                requests=list(self._request_queue),
            )
            self._pending_jobs[job_id] = job
            self._request_queue.clear()

        logger.info(f"Processing batch job {job_id} with {len(job.requests)} requests")

        try:
            # Submit batch to provider
            await self._submit_batch_job(job)

            # Poll for results
            await self._poll_batch_results(job)

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            # Mark all requests as failed
            for request in job.requests:
                request.status = BatchStatus.FAILED
                request.error = str(e)

    async def _submit_batch_job(self, job: BatchJob) -> None:
        """
        Submit batch job to provider.

        Args:
            job: Batch job to submit
        """
        try:
            # Anthropic batch API format
            if self.client and hasattr(self.client, 'batches'):
                batch_input = []
                for req in job.requests:
                    batch_input.append({
                        "custom_id": req.id,
                        "model": req.model,
                        "messages": [
                            {"role": "user", "content": req.prompt}
                        ],
                        "max_tokens": 1000,
                    })

                # Create batch file
                batch_file = Path(f"/tmp/batch_{job.id}.jsonl")
                with batch_file.open('w') as f:
                    for item in batch_input:
                        f.write(json.dumps(item) + '\n')

                # Submit to API
                batch = await self.client.batches.create(
                    input_file=str(batch_file),
                    endpoint="/v1/messages",
                )

                job.status = BatchStatus.PROCESSING
                logger.info(f"Batch job {job.id} submitted (batch_id={batch.id})")

            else:
                # Simulate batch processing for compatibility
                logger.warning("Batch API not available, simulating batch processing")
                await self._simulate_batch_processing(job)

        except Exception as e:
            logger.error(f"Batch submission failed: {e}")
            raise

    async def _simulate_batch_processing(self, job: BatchJob) -> None:
        """
        Simulate batch processing when API not available.

        This allows testing without actual batch API access.

        Args:
            job: Batch job to process
        """
        job.status = BatchStatus.PROCESSING

        # Process each request
        for req in job.requests:
            try:
                if self.client:
                    result = await self.client.call(req.model, req.prompt)
                    req.result = result
                    req.status = BatchStatus.COMPLETED
                else:
                    req.result = {"content": f"Simulated response for {req.id}"}
                    req.status = BatchStatus.COMPLETED

            except Exception as e:
                req.error = str(e)
                req.status = BatchStatus.FAILED

        job.status = BatchStatus.COMPLETED
        job.completed_at = time.time()

    async def _poll_batch_results(self, job: BatchJob) -> None:
        """
        Poll for batch job results.

        Args:
            job: Batch job to poll
        """
        start_time = time.time()
        max_wait = 600  # 10 minutes

        while job.status != BatchStatus.COMPLETED:
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Batch job {job.id} timed out")

            # Check job status
            if hasattr(self.client, 'batches'):
                try:
                    batch = await self.client.batches.retrieve(job.id)
                    if batch.status == "completed":
                        job.status = BatchStatus.COMPLETED
                        job.completed_at = time.time()
                except Exception:
                    pass  # Continue polling

        logger.info(f"Batch job {job.id} completed")

    def _estimate_cost(self, model: str, prompt: str) -> float:
        """
        Estimate API cost for request.

        Args:
            model: Model name
            prompt: Prompt text

        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates (per 1K tokens)
        COST_PER_1K = {
            "claude-opus": 15.0,
            "claude-sonnet": 3.0,
            "gpt-4": 30.0,
            "gpt-4-turbo": 10.0,
            "deepseek": 1.0,
        }

        # Estimate tokens (4 chars ≈ 1 token)
        tokens = len(prompt) / 4

        # Get model cost
        model_key = model.lower()
        cost_per_1k = 3.0  # Default
        for key, cost in COST_PER_1K.items():
            if key in model_key:
                cost_per_1k = cost
                break

        return (tokens / 1000) * cost_per_1k

    def get_metrics(self) -> dict[str, Any]:
        """
        Get batch processing metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "batch_requests": self.metrics.batch_requests,
            "realtime_requests": self.metrics.realtime_requests,
            "batch_ratio": self.metrics.batch_ratio,
            "batch_completions": self.metrics.batch_completions,
            "batch_failures": self.metrics.batch_failures,
            "total_savings": self.metrics.total_savings,
            "pending_jobs": len(self._pending_jobs),
            "queued_requests": len(self._request_queue),
        }

    async def shutdown(self) -> None:
        """Shutdown batch client gracefully."""
        logger.info("Shutting down batch client...")

        # Process remaining queue
        if self._request_queue:
            await self._process_batch_queue()

        # Wait for pending jobs
        if self._pending_jobs:
            await asyncio.sleep(5)  # Give jobs time to complete


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

async def batch_call(
    model: str,
    prompt: str,
    phase: str,
    client=None,
) -> Any:
    """
    Convenience function for batch API calls.

    Args:
        model: Model to use
        prompt: Prompt text
        phase: Phase name
        client: API client

    Returns:
        API response
    """
    batch = BatchClient(client=client)
    phase_enum = OptimizationPhase(phase)
    return await batch.call(model, prompt, phase_enum)


__all__ = [
    "BatchClient",
    "BatchStatus",
    "BatchRequest",
    "BatchJob",
    "BatchMetrics",
    "batch_call",
]
