"""
Tests for Resilient Improvements (Black Swan Scenarios)
========================================================

Tests the minimax regret improvements:
1. Event Store Corruption Resistance
2. Plugin Sandbox Security
3. Streaming Backpressure

Run with: pytest tests/test_resilient_improvements.py -v
"""

import pytest
import asyncio
import tempfile
import sqlite3
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from orchestrator.events_resilient import ResilientEventStore
from orchestrator.plugin_isolation_secure import (
    SecureIsolatedRuntime,
    SecureIsolationConfig,
    TrustedPluginRegistry,
)
from orchestrator.streaming_resilient import (
    ResilientStreamingPipeline,
    MemoryPressureConfig,
    CircuitBreakerConfig,
    MemoryMonitor,
    CircuitBreaker,
    MemoryPressure,
    CircuitState,
    BackpressureStrategy,
)

# =============================================================================
# Test Event Store Resilience
# =============================================================================


class TestResilientEventStore:
    """Test resilient event store with corruption resistance."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for stores."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as replica1_dir:
                with tempfile.TemporaryDirectory() as replica2_dir:
                    yield {
                        "primary": Path(primary_dir) / "primary.db",
                        "replica1": Path(replica1_dir) / "replica1.db",
                        "replica2": Path(replica2_dir) / "replica2.db",
                    }

    @pytest.fixture
    async def store(self, temp_dirs):
        """Create resilient event store."""
        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[
                str(temp_dirs["replica1"]),
                str(temp_dirs["replica2"]),
            ],
        )
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_write_to_primary(self, temp_dirs):
        """Test that writes go to primary."""
        from orchestrator.events import TaskCompletedEvent

        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[str(temp_dirs["replica1"])],
        )

        event = TaskCompletedEvent(
            task_id="test-1",
            project_id="proj-1",
            result={"output": "hello"},
        )

        await store.append(event)
        await asyncio.sleep(0.1)  # Let replication complete

        # Verify in primary
        events = await store.primary.get_events()
        assert len(events) == 1
        assert events[0].task_id == "test-1"

        await store.close()

    @pytest.mark.asyncio
    async def test_checksum_validation(self, temp_dirs):
        """Test that checksums are calculated and stored."""
        from orchestrator.events import TaskCompletedEvent

        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[],
        )

        event = TaskCompletedEvent(
            task_id="test-1",
            project_id="proj-1",
            result={"output": "hello"},
        )

        await store.append(event)

        # Verify checksum stored
        assert event.event_id in store.checksums
        assert len(store.checksums[event.event_id]) == 64  # SHA-256 hex

        await store.close()

    @pytest.mark.asyncio
    async def test_async_replication(self, temp_dirs):
        """Test async replication to secondaries."""
        from orchestrator.events import TaskCompletedEvent

        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[
                str(temp_dirs["replica1"]),
                str(temp_dirs["replica2"]),
            ],
        )

        event = TaskCompletedEvent(
            task_id="test-1",
            project_id="proj-1",
            result={"output": "hello"},
        )

        await store.append(event)
        await asyncio.sleep(0.2)  # Let replication complete

        # Verify in replicas
        for replica in store.replicas:
            events = await replica.get_events()
            assert len(events) == 1
            assert events[0].task_id == "test-1"

        await store.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, temp_dirs):
        """Test that WAL mode is enabled on primary."""
        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[],
        )

        # Check journal mode
        conn = sqlite3.connect(str(temp_dirs["primary"]))
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        assert mode.upper() == "WAL"

        await store.close()

    @pytest.mark.asyncio
    async def test_integrity_verification(self, temp_dirs):
        """Test integrity verification."""
        from orchestrator.events import TaskCompletedEvent

        store = ResilientEventStore(
            primary_path=str(temp_dirs["primary"]),
            replica_paths=[str(temp_dirs["replica1"])],
        )

        # Add event
        event = TaskCompletedEvent(
            task_id="test-1",
            project_id="proj-1",
            result={"output": "hello"},
        )
        await store.append(event)
        await asyncio.sleep(0.1)

        # Verify integrity
        report = await store.verify_integrity()

        assert report["primary_healthy"] is True
        assert len(report["replica_health"]) == 1
        assert report["replica_health"][0]["healthy"] is True

        await store.close()


# =============================================================================
# Test Secure Plugin Isolation
# =============================================================================


class TestSecurePluginIsolation:
    """Test secure plugin isolation with defense in depth."""

    @pytest.fixture
    def secure_config(self):
        """Create secure isolation config."""
        return SecureIsolationConfig(
            memory_limit_mb=256,
            cpu_limit_percent=50,
            timeout_seconds=5.0,
            allow_network=False,
            enable_seccomp=False,  # Skip for testing (not available everywhere)
            enable_landlock=False,  # Skip for testing
            enable_capabilities=False,  # Skip for testing
        )

    @pytest.fixture
    def runtime(self, secure_config):
        """Create secure runtime."""
        return SecureIsolatedRuntime(secure_config)

    def test_security_feature_detection(self, runtime):
        """Test that security features are detected."""
        # Should detect what's available
        assert SecurityLayer.PROCESS in runtime.features
        # Other layers may or may not be available depending on system

    def test_sandbox_creation(self, runtime):
        """Test sandbox directory creation."""
        sandbox = runtime._create_sandbox()

        assert sandbox.exists()
        assert (sandbox / "workspace").exists()
        assert (sandbox / "temp").exists()

        runtime._cleanup_sandbox(sandbox)
        assert not sandbox.exists()

    def test_sandbox_cleanup(self, runtime):
        """Test sandbox cleanup."""
        sandbox = runtime._create_sandbox()

        # Create a file
        (sandbox / "workspace" / "test.txt").write_text("test")

        runtime._cleanup_sandbox(sandbox)

        assert not sandbox.exists()

    @pytest.mark.asyncio
    async def test_simple_plugin_execution(self, runtime):
        """Test basic plugin execution."""

        # Create a simple test plugin
        class TestPlugin:
            def validate(self, code: str) -> dict:
                return {"valid": True, "code": code}

        plugin = TestPlugin()
        result = await runtime.execute(plugin, "validate", "test code")

        assert result.success is True
        assert result.result["valid"] is True
        assert result.result["code"] == "test code"

    @pytest.mark.asyncio
    async def test_plugin_timeout(self, runtime):
        """Test that plugins timeout correctly."""

        class SlowPlugin:
            def validate(self, code: str) -> dict:
                import time

                time.sleep(10)  # Will timeout
                return {"valid": True}

        plugin = SlowPlugin()
        result = await runtime.execute(plugin, "validate", "test")

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_plugin_exception_handling(self, runtime):
        """Test that plugin exceptions are handled."""

        class FailingPlugin:
            def validate(self, code: str) -> dict:
                raise ValueError("Test error")

        plugin = FailingPlugin()
        result = await runtime.execute(plugin, "validate", "test")

        assert result.success is False
        assert "ValueError" in result.error


class TestTrustedPluginRegistry:
    """Test trusted plugin registry."""

    @pytest.fixture
    def registry(self):
        """Create trusted registry."""
        return TrustedPluginRegistry()

    @pytest.fixture
    def temp_plugin(self, tmp_path):
        """Create temporary plugin file."""
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text("# Test plugin\nprint('hello')")
        return plugin_file

    def test_add_trusted_plugin(self, registry):
        """Test adding trusted plugin."""
        registry.add_trusted_plugin("test_plugin", "abc123")

        assert "test_plugin" in registry.trusted_hashes
        assert registry.trusted_hashes["test_plugin"] == "abc123"

    def test_verify_valid_plugin(self, registry, temp_plugin):
        """Test verification of valid plugin."""
        # Calculate hash
        content = temp_plugin.read_bytes()
        expected_hash = hashlib.sha256(content).hexdigest()

        registry.add_trusted_plugin("test_plugin", expected_hash)

        assert registry.verify_plugin(temp_plugin, "test_plugin") is True

    def test_verify_modified_plugin(self, registry, temp_plugin):
        """Test detection of modified plugin."""
        # Add wrong hash
        registry.add_trusted_plugin("test_plugin", "wrong_hash")

        assert registry.verify_plugin(temp_plugin, "test_plugin") is False

    def test_verify_unknown_plugin(self, registry, temp_plugin):
        """Test rejection of unknown plugin."""
        assert registry.verify_plugin(temp_plugin, "unknown_plugin") is False


# =============================================================================
# Test Resilient Streaming
# =============================================================================


class TestResilientStreaming:
    """Test resilient streaming with backpressure."""

    @pytest.fixture
    def memory_config(self):
        """Create memory pressure config."""
        return MemoryPressureConfig(
            max_queue_size=10,
            max_memory_mb=1024,
            backpressure_strategy=BackpressureStrategy.SAMPLE,
            sampling_rate=2,
        )

    @pytest.fixture
    def circuit_config(self):
        """Create circuit breaker config."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
        )

    @pytest.fixture
    def pipeline(self, memory_config, circuit_config):
        """Create resilient pipeline."""
        return ResilientStreamingPipeline(
            max_parallel=2,
            memory_config=memory_config,
            circuit_config=circuit_config,
        )

    def test_memory_monitor_without_psutil(self):
        """Test memory monitor fallback without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            monitor = MemoryMonitor()
            assert monitor._psutil_available is False
            # Should not crash
            assert monitor.get_usage_percent() >= 0
            assert monitor.get_available_mb() > 0

    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        assert cb.is_open() is False
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        # Record failures
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is False  # Not yet

        cb.record_failure()
        assert cb.is_open() is True  # Now open
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_half_open(self):
        """Test circuit breaker enters half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        assert cb.is_open() is True

        # Wait for recovery timeout
        import time

        time.sleep(0.15)

        assert cb.is_open() is False  # Half-open now
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes after success."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
        )
        cb = CircuitBreaker(config)

        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_pipeline_circuit_breaker(self, pipeline):
        """Test that pipeline respects circuit breaker."""
        # Open the circuit
        for _ in range(5):
            pipeline.circuit.record_failure()

        assert pipeline.circuit.is_open()

        # Should reject new execution
        with pytest.raises(Exception) as exc_info:
            async for _ in pipeline.execute_streaming(
                "test project",
                "success criteria",
            ):
                pass

        assert "circuit" in str(exc_info.value).lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple improvements."""

    @pytest.mark.asyncio
    async def test_end_to_end_resilience(self, tmp_path):
        """Test end-to-end resilience with all components."""
        from orchestrator.events import TaskCompletedEvent

        # 1. Setup resilient event store
        primary = tmp_path / "primary.db"
        replica = tmp_path / "replica.db"

        store = ResilientEventStore(
            primary_path=str(primary),
            replica_paths=[str(replica)],
        )

        # 2. Add events
        for i in range(10):
            event = TaskCompletedEvent(
                task_id=f"task-{i}",
                project_id="test-proj",
                result={"iteration": i},
            )
            await store.append(event)

        await asyncio.sleep(0.2)

        # 3. Verify integrity
        report = await store.verify_integrity()
        assert report["primary_healthy"]

        # 4. Read events back
        events = await store.get_events()
        assert len(events) == 10

        # 5. Cleanup
        await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
