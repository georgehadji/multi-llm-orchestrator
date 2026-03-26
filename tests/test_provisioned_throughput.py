"""
Tests for Provisioned Throughput Manager
=========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_provisioned_throughput.py -v
"""

import pytest
import asyncio

from orchestrator.provisioned_throughput import (
    ProvisionedThroughputManager,
    ProvisionedThroughputConfig,
    CapacityUnit,
    UsageMetrics,
    CapacityType,
    get_throughput_manager,
)


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def pt_config():
    """Create provisioned throughput config."""
    return ProvisionedThroughputConfig(
        enabled=True,
        units=4,
        models=["grok-4.20"],
        max_daily_cost=40.0,
    )


@pytest.fixture
def pt_manager(pt_config):
    """Create ProvisionedThroughputManager instance."""
    return ProvisionedThroughputManager(config=pt_config, api_key="test-key")


# ─────────────────────────────────────────────
# Test CapacityUnit
# ─────────────────────────────────────────────

class TestCapacityUnit:
    """Test CapacityUnit dataclass."""
    
    def test_capacity_unit_creation(self):
        """Test capacity unit creation."""
        unit = CapacityUnit(
            unit_id="test-unit-1",
            model="grok-4.20",
        )
        
        assert unit.unit_id == "test-unit-1"
        assert unit.model == "grok-4.20"
        assert unit.input_tpm == 31_500
        assert unit.output_tpm == 12_500
        assert unit.status == "active"
    
    def test_capacity_unit_to_dict(self):
        """Test capacity unit serialization."""
        unit = CapacityUnit(unit_id="test-1", model="grok-4.20")
        unit_dict = unit.to_dict()
        
        assert unit_dict["unit_id"] == "test-1"
        assert unit_dict["input_tpm"] == 31_500
        assert unit_dict["output_tpm"] == 12_500


# ─────────────────────────────────────────────
# Test UsageMetrics
# ─────────────────────────────────────────────

class TestUsageMetrics:
    """Test UsageMetrics dataclass."""
    
    def test_usage_metrics_creation(self):
        """Test usage metrics creation."""
        metrics = UsageMetrics()
        
        assert metrics.current_input_tpm == 0
        assert metrics.current_output_tpm == 0
        assert metrics.total_input_tokens == 0
        assert metrics.committed_usage == 0
    
    def test_usage_metrics_reset(self):
        """Test usage metrics reset."""
        metrics = UsageMetrics()
        metrics.current_input_tpm = 1000
        metrics.current_output_tpm = 500
        
        # Reset manually
        metrics.current_input_tpm = 0
        metrics.current_output_tpm = 0
        
        assert metrics.current_input_tpm == 0
        assert metrics.current_output_tpm == 0


# ─────────────────────────────────────────────
# Test ProvisionedThroughputConfig
# ─────────────────────────────────────────────

class TestProvisionedThroughputConfig:
    """Test ProvisionedThroughputConfig dataclass."""
    
    def test_config_creation(self, pt_config):
        """Test config creation."""
        assert pt_config.enabled is True
        assert pt_config.units == 4
        assert "grok-4.20" in pt_config.models
        assert pt_config.max_daily_cost == 40.0
    
    def test_config_to_dict(self, pt_config):
        """Test config serialization."""
        config_dict = pt_config.to_dict()
        
        assert config_dict["enabled"] is True
        assert config_dict["units"] == 4
        assert config_dict["max_daily_cost"] == 40.0


# ─────────────────────────────────────────────
# Test ProvisionedThroughputManager
# ─────────────────────────────────────────────

class TestProvisionedThroughputManager:
    """Test ProvisionedThroughputManager class."""
    
    def test_manager_initialization(self, pt_manager, pt_config):
        """Test manager initializes correctly."""
        assert pt_manager.config.enabled is True
        assert pt_manager.config.units == 4
        assert pt_manager.usage.total_input_tokens == 0
    
    def test_get_total_capacity(self, pt_manager):
        """Test getting total capacity."""
        input_cap, output_cap = pt_manager._get_total_capacity()
        
        # 4 units × 31,500 TPM = 126,000 TPM input
        # 4 units × 12,500 TPM = 50,000 TPM output
        assert input_cap == 4 * 31_500
        assert output_cap == 4 * 12_500
    
    def test_get_total_capacity_disabled(self):
        """Test capacity when disabled."""
        config = ProvisionedThroughputConfig(enabled=False, units=0)
        manager = ProvisionedThroughputManager(config=config)
        
        input_cap, output_cap = manager._get_total_capacity()
        
        assert input_cap == 0
        assert output_cap == 0
    
    @pytest.mark.asyncio
    async def test_check_capacity_success(self, pt_manager):
        """Test successful capacity check."""
        acquired = await pt_manager.check_capacity(tokens=10000, is_input=True)
        
        assert acquired is True
        assert pt_manager.usage.total_input_tokens == 10000
    
    @pytest.mark.asyncio
    async def test_check_capacity_exceeded(self, pt_manager):
        """Test capacity exceeded scenario."""
        # Try to acquire more than capacity
        input_cap, _ = pt_manager._get_total_capacity()
        
        acquired = await pt_manager.check_capacity(
            tokens=input_cap + 1,
            is_input=True,
            timeout=0.1,
        )
        
        # Should fail (capacity exceeded)
        assert acquired is False
        assert pt_manager.total_capacity_exceeded >= 1
    
    @pytest.mark.asyncio
    async def test_check_capacity_multiple_requests(self, pt_manager):
        """Test multiple capacity requests."""
        # Make several small requests
        for i in range(5):
            acquired = await pt_manager.check_capacity(tokens=1000, is_input=True)
            assert acquired is True
        
        assert pt_manager.usage.total_input_tokens == 5000
        assert pt_manager.total_capacity_checks == 5
    
    def test_record_usage(self, pt_manager):
        """Test recording usage."""
        pt_manager.record_usage(
            input_tokens=5000,
            output_tokens=3000,
            capacity_type=CapacityType.COMMITTED,
        )
        
        assert pt_manager.usage.total_input_tokens == 5000
        assert pt_manager.usage.total_output_tokens == 3000
        assert pt_manager.usage.committed_usage == 8000
    
    def test_record_usage_on_demand(self, pt_manager):
        """Test recording on-demand usage."""
        pt_manager.record_usage(
            input_tokens=2000,
            output_tokens=1000,
            capacity_type=CapacityType.ON_DEMAND,
        )
        
        assert pt_manager.usage.on_demand_usage == 3000
    
    def test_get_stats(self, pt_manager):
        """Test getting statistics."""
        # Make some requests
        asyncio.run(pt_manager.check_capacity(tokens=10000, is_input=True))
        
        stats = pt_manager.get_stats()
        
        assert "enabled" in stats
        assert "units" in stats
        assert "input_capacity_tpm" in stats
        assert "capacity_utilization" in stats
        assert stats["units"] == 4
        assert stats["total_input_tokens"] == 10000
    
    @pytest.mark.asyncio
    async def test_auto_scale_capacity(self, pt_manager):
        """Test auto-scaling."""
        # Enable auto-scaling
        pt_manager.config.auto_scale = True
        pt_manager.config.max_units = 10
        
        # Trigger auto-scale
        scaled = await pt_manager._auto_scale_capacity()
        
        assert scaled is True
        assert pt_manager.config.units == 5  # Scaled from 4 to 5
        assert pt_manager.auto_scale_events == 1
    
    @pytest.mark.asyncio
    async def test_auto_scale_at_max(self, pt_manager):
        """Test auto-scaling at max capacity."""
        pt_manager.config.auto_scale = True
        pt_manager.config.max_units = 4  # Already at max
        pt_manager.config.units = 4
        
        # Try to scale
        scaled = await pt_manager._auto_scale_capacity()
        
        assert scaled is False
        assert pt_manager.config.units == 4  # Unchanged
    
    @pytest.mark.asyncio
    async def test_provision_units(self, pt_manager):
        """Test provisioning new units."""
        units = await pt_manager.provision_units(2, ["grok-4.20"])
        
        assert len(units) == 2
        assert pt_manager.config.units == 6  # 4 + 2
        assert pt_manager.config.enabled is True
    
    @pytest.mark.asyncio
    async def test_close_manager(self, pt_manager):
        """Test closing manager."""
        await pt_manager.close()
        
        assert pt_manager._session is None


# ─────────────────────────────────────────────
# Test Global Instance
# ─────────────────────────────────────────────

class TestGlobalInstance:
    """Test global throughput manager instance."""
    
    def test_get_throughput_manager(self, pt_config):
        """Test getting global manager."""
        manager1 = get_throughput_manager(config=pt_config)
        manager2 = get_throughput_manager(config=pt_config)
        
        # Should return same instance
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_close_throughput_manager(self, pt_config):
        """Test closing global manager."""
        manager = get_throughput_manager(config=pt_config)
        await manager.check_capacity(tokens=1000)
        
        await close_throughput_manager()
        
        # Should be able to create new one
        new_manager = get_throughput_manager(config=pt_config)
        assert new_manager is not manager


# Import for cleanup test
from orchestrator.provisioned_throughput import close_throughput_manager


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
