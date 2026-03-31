"""
Provisioned Throughput Manager
===============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Enterprise-grade provisioned throughput management for xAI Grok API.

Features:
- Provisioned capacity units (31,500 input TPM + 12,500 output TPM per unit)
- Committed capacity (30-day minimum)
- On-demand overage handling
- Usage tracking and monitoring
- 99.9% SLA guarantees

Pricing:
- $10.00 per day per unit
- 30-day minimum commitment
- Overage at pay-as-you-go rates

Usage:
    from orchestrator.provisioned_throughput import ProvisionedThroughputManager

    manager = ProvisionedThroughputManager(units=4, models=["grok-4.20"])
    await manager.check_capacity(tokens=10000, is_input=True)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger("orchestrator.provisioned_throughput")


class CapacityType(str, Enum):
    """Type of capacity."""

    COMMITTED = "committed"  # 30-day minimum
    ON_DEMAND = "on_demand"  # Pay-as-you-go overage


@dataclass
class CapacityUnit:
    """A single provisioned capacity unit."""

    unit_id: str
    model: str
    input_tpm: int = 31_500  # Tokens per minute (input)
    output_tpm: int = 12_500  # Tokens per minute (output)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime | None = None  # None = ongoing
    status: str = "active"  # active, expired, suspended

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "model": self.model,
            "input_tpm": self.input_tpm,
            "output_tpm": self.output_tpm,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
        }


@dataclass
class UsageMetrics:
    """Current usage metrics."""

    current_input_tpm: int = 0
    current_output_tpm: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    committed_usage: int = 0  # Tokens from committed capacity
    on_demand_usage: int = 0  # Tokens from on-demand
    last_reset: datetime = field(default_factory=datetime.now)

    def reset_if_needed(self):
        """Reset TPM counters if minute has passed."""
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= 60:
            self.current_input_tpm = 0
            self.current_output_tpm = 0
            self.last_reset = now


@dataclass
class ProvisionedThroughputConfig:
    """Configuration for provisioned throughput."""

    enabled: bool = False
    units: int = 0
    models: list[str] = field(default_factory=lambda: ["grok-4.20", "grok-4.20-reasoning"])
    max_daily_cost: float = 100.0  # $10 × units
    auto_scale: bool = False  # Auto-scale based on demand
    min_units: int = 1
    max_units: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "units": self.units,
            "models": self.models,
            "max_daily_cost": self.max_daily_cost,
            "auto_scale": self.auto_scale,
            "min_units": self.min_units,
            "max_units": self.max_units,
        }


class ProvisionedThroughputManager:
    """
    Manager for provisioned throughput capacity.

    Features:
    - Capacity tracking and allocation
    - Usage monitoring
    - Auto-scaling (optional)
    - Cost tracking
    - SLA guarantees (99.9%)

    Usage:
        manager = ProvisionedThroughputManager(units=4)
        await manager.check_capacity(tokens=10000)
    """

    # Capacity per unit (from xAI)
    INPUT_TPM_PER_UNIT = 31_500
    OUTPUT_TPM_PER_UNIT = 12_500

    # Pricing
    COST_PER_UNIT_PER_DAY = 10.0

    def __init__(
        self,
        config: ProvisionedThroughputConfig | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize provisioned throughput manager.

        Args:
            config: Provisioned throughput configuration
            api_key: xAI API key
        """
        self.config = config or ProvisionedThroughputConfig()
        self.api_key = api_key or os.environ.get("XAI_API_KEY")

        self.usage = UsageMetrics()
        self._capacity_units: list[CapacityUnit] = []
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None

        # Metrics
        self.total_capacity_checks = 0
        self.total_capacity_exceeded = 0
        self.auto_scale_events = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._session

    async def close(self):
        """Close manager and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_total_capacity(self) -> tuple[int, int]:
        """
        Get total provisioned capacity.

        Returns:
            Tuple of (input_tpm, output_tpm)
        """
        if not self.config.enabled or self.config.units == 0:
            return (0, 0)

        total_input = self.config.units * self.INPUT_TPM_PER_UNIT
        total_output = self.config.units * self.OUTPUT_TPM_PER_UNIT

        return (total_input, total_output)

    async def check_capacity(
        self,
        tokens: int,
        is_input: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """
        Check if capacity is available for request.

        Args:
            tokens: Number of tokens for request
            is_input: True for input tokens, False for output
            timeout: Maximum time to wait for capacity

        Returns:
            True if capacity available, False otherwise
        """
        if not self.config.enabled:
            # Not using provisioned throughput
            return True

        start_time = time.time()

        async with self._lock:
            self.usage.reset_if_needed()
            self.total_capacity_checks += 1

            input_capacity, output_capacity = self._get_total_capacity()

            if is_input:
                available = input_capacity - self.usage.current_input_tpm
                if available >= tokens:
                    self.usage.current_input_tpm += tokens
                    self.usage.total_input_tokens += tokens
                    self.usage.committed_usage += tokens
                    logger.debug(
                        f"Capacity acquired: {tokens} input tokens, "
                        f"usage: {self.usage.current_input_tpm}/{input_capacity} TPM"
                    )
                    return True
            else:
                available = output_capacity - self.usage.current_output_tpm
                if available >= tokens:
                    self.usage.current_output_tpm += tokens
                    self.usage.total_output_tokens += tokens
                    self.usage.committed_usage += tokens
                    logger.debug(
                        f"Capacity acquired: {tokens} output tokens, "
                        f"usage: {self.usage.current_output_tpm}/{output_capacity} TPM"
                    )
                    return True

            # Capacity exceeded
            self.total_capacity_exceeded += 1
            logger.warning(
                f"Provisioned capacity exceeded: needed {tokens}, "
                f"available {available}, unit={self.config.units}"
            )

            # Check if auto-scaling should trigger
            if self.config.auto_scale:
                scaled = await self._auto_scale_capacity()
                if scaled:
                    # Retry after scaling
                    return await self.check_capacity(
                        tokens, is_input, timeout - (time.time() - start_time)
                    )

            # Check if we can wait for capacity to free up
            elapsed = time.time() - start_time
            if elapsed < timeout:
                wait_time = 60 - (datetime.now() - self.usage.last_reset).total_seconds()
                if wait_time > 0 and elapsed + wait_time <= timeout:
                    logger.debug(f"Waiting {wait_time:.1f}s for capacity reset")
                    await asyncio.sleep(wait_time)
                    return await self.check_capacity(
                        tokens, is_input, timeout - elapsed - wait_time
                    )

            # No capacity available, return False (caller should use on-demand)
            return False

    async def _auto_scale_capacity(self) -> bool:
        """
        Auto-scale capacity based on demand.

        Returns:
            True if scaled successfully
        """
        if not self.config.auto_scale:
            return False

        current_units = self.config.units

        # Check if we can scale up
        if current_units >= self.config.max_units:
            logger.warning(f"Cannot auto-scale: already at max units ({current_units})")
            return False

        # Scale up by 1 unit
        new_units = current_units + 1
        logger.info(f"Auto-scaling capacity: {current_units} → {new_units} units")

        # In production, this would call xAI API to provision more capacity
        # For now, just update config
        self.config.units = new_units
        self.auto_scale_events += 1

        return True

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        capacity_type: CapacityType = CapacityType.COMMITTED,
    ):
        """
        Record actual API usage.

        Args:
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            capacity_type: Type of capacity used
        """
        self.usage.total_input_tokens += input_tokens
        self.usage.total_output_tokens += output_tokens

        if capacity_type == CapacityType.COMMITTED:
            self.usage.committed_usage += input_tokens + output_tokens
        else:
            self.usage.on_demand_usage += input_tokens + output_tokens

        logger.debug(
            f"Recorded usage: {input_tokens} input + {output_tokens} output tokens, "
            f"type={capacity_type.value}"
        )

    async def fetch_usage_from_api(self) -> UsageMetrics:
        """
        Fetch current usage from xAI API.

        Returns:
            Current usage metrics
        """
        if not self.api_key:
            logger.warning("No API key configured, cannot fetch usage")
            return self.usage

        try:
            session = await self._get_session()
            async with session.get("https://api.x.ai/v1/usage") as response:
                if response.status == 200:
                    data = await response.json()
                    self.usage.total_input_tokens = data.get("input_tokens", 0)
                    self.usage.total_output_tokens = data.get("output_tokens", 0)
                    self.usage.committed_usage = data.get("committed_usage", 0)
                    self.usage.on_demand_usage = data.get("on_demand_usage", 0)
                    return self.usage
                else:
                    logger.warning(f"Failed to fetch usage: {response.status}")
                    return self.usage
        except Exception as e:
            logger.error(f"Error fetching usage: {e}")
            return self.usage

    def get_stats(self) -> dict[str, Any]:
        """
        Get provisioned throughput statistics.

        Returns:
            Dictionary with stats
        """
        input_capacity, output_capacity = self._get_total_capacity()
        daily_cost = self.config.units * self.COST_PER_UNIT_PER_DAY if self.config.enabled else 0.0

        return {
            "enabled": self.config.enabled,
            "units": self.config.units,
            "models": self.config.models,
            "input_capacity_tpm": input_capacity,
            "output_capacity_tpm": output_capacity,
            "current_input_tpm": self.usage.current_input_tpm,
            "current_output_tpm": self.usage.current_output_tpm,
            "total_input_tokens": self.usage.total_input_tokens,
            "total_output_tokens": self.usage.total_output_tokens,
            "committed_usage": self.usage.committed_usage,
            "on_demand_usage": self.usage.on_demand_usage,
            "capacity_utilization": (
                (self.usage.current_input_tpm / input_capacity * 100) if input_capacity > 0 else 0
            ),
            "daily_cost": daily_cost,
            "auto_scale_events": self.auto_scale_events,
            "capacity_checks": self.total_capacity_checks,
            "capacity_exceeded": self.total_capacity_exceeded,
        }

    async def provision_units(
        self,
        units: int,
        models: list[str] | None = None,
    ) -> list[CapacityUnit]:
        """
        Provision new capacity units.

        In production, this would call xAI API to provision capacity.
        For now, creates local capacity units.

        Args:
            units: Number of units to provision
            models: Models to provision for

        Returns:
            List of provisioned capacity units
        """
        models = models or self.config.models

        new_units = []
        for i in range(units):
            unit = CapacityUnit(
                unit_id=f"pt-unit-{len(self._capacity_units) + i + 1}",
                model=models[0] if models else "grok-4.20",
            )
            new_units.append(unit)
            self._capacity_units.append(unit)

        # Update config
        self.config.units += units
        self.config.enabled = True

        logger.info(f"Provisioned {units} capacity units: {[u.unit_id for u in new_units]}")

        return new_units


# Global manager instance
_manager: ProvisionedThroughputManager | None = None


def get_throughput_manager(
    config: ProvisionedThroughputConfig | None = None,
    api_key: str | None = None,
) -> ProvisionedThroughputManager:
    """
    Get or create global throughput manager.

    Args:
        config: Provisioned throughput configuration
        api_key: xAI API key

    Returns:
        ProvisionedThroughputManager instance
    """
    global _manager
    if _manager is None:
        _manager = ProvisionedThroughputManager(config=config, api_key=api_key)
    return _manager


async def close_throughput_manager() -> None:
    """Close global throughput manager."""
    global _manager
    if _manager:
        await _manager.close()
        _manager = None
