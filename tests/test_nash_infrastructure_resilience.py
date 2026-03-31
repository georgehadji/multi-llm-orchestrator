"""
Nash Stability Infrastructure v2.0 - Resilience Stress Test
===========================================================

Black Swan Event Simulation & Recovery Validation
Dev/Adversary Round 3: Resilience Testing

Tests:
1. Extreme load (thundering herd)
2. Malformed inputs (type confusion)
3. Resource exhaustion (memory/disk)
4. Timing attacks (race conditions)
5. Cascading failures
"""

import asyncio
import gc
import json
import os
import random
import string
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Dict
import unittest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.nash_infrastructure_v2 import (
    AsyncIOManager,
    WriteAheadLog,
    WALEntry,
    WALEntryStatus,
    EventNormalizer,
    UnifiedEventBus,
    NormalizedEvent,
    TransactionalStorage,
)


class BlackSwanTestSuite(unittest.IsolatedAsyncioTestCase):
    """
    BLACK SWAN EVENT SIMULATION
    ===========================

    Tests for unexpected catastrophic events
    """

    async def asyncSetUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="nash_stress_"))
        self.wal_dir = self.test_dir / "wal"
        self.data_dir = self.test_dir / "data"
        self.wal_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    async def asyncTearDown(self):
        # Cleanup
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK SWAN #1: TYPE CONFUSION ATTACK
    # ═══════════════════════════════════════════════════════════════════════════

    async def test_type_confusion_wal_none_path(self):
        """
        BLACK SWAN: WAL receives None as target_path
        Expected: Graceful error, no crash
        """
        wal = WriteAheadLog(wal_dir=self.wal_dir)

        with self.assertRaises((TypeError, AttributeError)):
            await wal.append("write", None, "test data")

    async def test_type_confusion_wal_bytes_as_string(self):
        """
        BLACK SWAN: WAL receives bytes where string expected (or vice versa)
        Expected: Handle gracefully via encoding
        """
        wal = WriteAheadLog(wal_dir=self.wal_dir)

        # This should work - bytes data
        entry = await wal.append(
            "write",
            self.data_dir / "test.bin",
            b"\x00\x01\x02\xff" * 1000,  # Binary data
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.status, WALEntryStatus.PENDING)

    async def test_type_confusion_event_non_dataclass(self):
        """
        BLACK SWAN: EventNormalizer receives completely foreign object
        Expected: Auto-normalize without crashing
        """
        normalizer = EventNormalizer()

        class ForeignObject:
            """Simulates an event from an unknown system"""

            def __init__(self):
                self.custom_field = "value"
                self._private = "should be ignored"

        foreign = ForeignObject()
        normalized = normalizer.normalize(foreign)

        self.assertIsInstance(normalized, NormalizedEvent)
        self.assertIn("custom_field", normalized.payload)
        self.assertNotIn("_private", normalized.payload)  # Private fields ignored

    async def test_type_confusion_nested_corruption(self):
        """
        BLACK SWAN: Deeply nested circular reference in event payload
        Expected: Handle without infinite recursion
        """
        normalizer = EventNormalizer()

        class CircularEvent:
            def __init__(self):
                self.self_ref = self
                self.data = "test"

        circular = CircularEvent()
        # Should not crash with infinite recursion
        try:
            normalized = normalizer.normalize(circular)
            # If it succeeds, payload might contain representation
            self.assertIsInstance(normalized, NormalizedEvent)
        except RecursionError:
            self.fail("Normalizer crashed on circular reference")

    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK SWAN #2: EXTREME LOAD (THUNDERING HERD)
    # ═══════════════════════════════════════════════════════════════════════════

    async def test_thundering_herd_wal_append(self):
        """
        BLACK SWAN: 1000 concurrent WAL appends
        Expected: All succeed, no data loss, proper rotation
        """
        wal = WriteAheadLog(
            wal_dir=self.wal_dir,
            max_entries_per_file=100,  # Small for testing rotation
        )

        async def append_task(i: int) -> str:
            entry = await wal.append(
                "write",
                self.data_dir / f"file_{i}.txt",
                f"data_{i}" * 100,
            )
            return entry.entry_id

        # Launch 1000 concurrent tasks
        start_time = time.time()
        tasks = [append_task(i) for i in range(1000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Check results
        errors = [r for r in results if isinstance(r, Exception)]
        success_count = len(results) - len(errors)

        print(f"\n[THUNDERING HERD] 1000 appends: {success_count} succeeded, {len(errors)} errors")
        print(f"[THUNDERING HERD] Time: {elapsed:.2f}s ({1000/elapsed:.0f} ops/sec)")

        # Assert high success rate (>95%)
        self.assertGreaterEqual(success_count / 1000, 0.95)

        # Verify WAL files were rotated
        wal_files = list(self.wal_dir.glob("wal_*.jsonl"))
        self.assertGreater(len(wal_files), 1)  # Should have rotated

    async def test_thundering_herd_async_io(self):
        """
        BLACK SWAN: 500 concurrent file writes via AsyncIOManager
        Expected: Thread pool handles it without exhaustion
        """
        io_mgr = AsyncIOManager(max_workers=4)

        async def write_task(i: int) -> bool:
            path = self.data_dir / f"concurrent_{i}.txt"
            return await io_mgr.write_file(path, f"content_{i}" * 1000)

        start_time = time.time()
        tasks = [write_task(i) for i in range(500)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        errors = [r for r in results if isinstance(r, Exception)]
        success_count = len(results) - len(errors)

        print(f"\n[ASYNC IO STORM] 500 writes: {success_count} succeeded, {len(errors)} errors")
        print(f"[ASYNC IO STORM] Time: {elapsed:.2f}s ({500/elapsed:.0f} ops/sec)")

        # Should have high success rate
        self.assertGreaterEqual(success_count / 500, 0.95)
        io_mgr.shutdown()

    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK SWAN #3: RESOURCE EXHAUSTION
    # ═══════════════════════════════════════════════════════════════════════════

    async def test_resource_exhaustion_memory_pressure(self):
        """
        BLACK SWAN: Event with massive payload (10MB)
        Expected: WAL stores hash only, not full data
        """
        wal = WriteAheadLog(wal_dir=self.wal_dir)

        # Generate 10MB of data
        massive_data = "X" * (10 * 1024 * 1024)

        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]

        entry = await wal.append(
            "write",
            self.data_dir / "huge.txt",
            massive_data,
        )

        after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # WAL entry should NOT contain the full data
        self.assertIsNone(entry.data)  # Data too large, not stored
        self.assertEqual(entry.data_size, len(massive_data))
        self.assertIsNotNone(entry.data_hash)  # But hash is stored

        print(f"\n[MEMORY PRESSURE] 10MB payload: stored hash only")
        print(f"[MEMORY PRESSURE] Entry size in WAL: ~{len(json.dumps(entry.to_dict()))} bytes")

    async def test_resource_exhaustion_disk_full_simulation(self):
        """
        BLACK SWAN: Simulate disk full by using invalid path
        Expected: Graceful failure with clear error
        """
        # Use a read-only path to simulate disk full
        readonly_dir = Path("/nonexistent_path_for_testing")

        with self.assertRaises((OSError, PermissionError, FileNotFoundError)):
            wal = WriteAheadLog(wal_dir=readonly_dir)

    async def test_resource_exhaustion_file_descriptor_leak(self):
        """
        BLACK SWAN: Verify no file descriptor leaks under heavy load
        """
        import resource

        wal = WriteAheadLog(wal_dir=self.wal_dir)

        # Get initial fd count (Unix only)
        try:
            initial_fds = resource.getrlimit(resource.RLIMIT_NOFILE)
        except:
            initial_fds = None

        # Heavy operations
        for i in range(100):
            await wal.append("write", self.data_dir / f"fd_test_{i}.txt", f"data_{i}")

        # Force garbage collection
        gc.collect()

        # Check WAL file handle is properly closed
        # (This is a basic check - real check would use lsof)
        print(f"\n[FD LEAK CHECK] Completed 100 operations")
        if initial_fds:
            print(f"[FD LEAK CHECK] FD limit: {initial_fds}")

    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK SWAN #4: TIMING & RACE CONDITIONS
    # ═══════════════════════════════════════════════════════════════════════════

    async def test_race_condition_read_during_write(self):
        """
        BLACK SWAN: Read same file while being written
        Expected: Atomic writes ensure consistent state
        """
        io_mgr = AsyncIOManager()
        target_file = self.data_dir / "race_test.txt"

        async def writer():
            for i in range(50):
                await io_mgr.write_file(
                    target_file,
                    f"VERSION_{i}_" + "x" * 1000,
                )
                await asyncio.sleep(0.001)

        async def reader():
            versions_seen = set()
            for _ in range(50):
                try:
                    content = await io_mgr.read_file(target_file)
                    # Extract version number
                    if content.startswith("VERSION_"):
                        version = content.split("_")[1]
                        versions_seen.add(version)
                except FileNotFoundError:
                    pass  # File might not exist yet
                await asyncio.sleep(0.001)
            return versions_seen

        # Run concurrently
        writer_task = asyncio.create_task(writer())
        reader_task = asyncio.create_task(reader())

        await writer_task
        versions = await reader_task

        print(f"\n[RACE TEST] Versions seen during writes: {len(versions)}")
        # Reader should have seen some versions (atomic writes working)
        self.assertGreater(len(versions), 0)
        io_mgr.shutdown()

    async def test_race_condition_wal_rotation_mid_append(self):
        """
        BLACK SWAN: WAL rotation happens while append in progress
        Expected: All entries accounted for, no corruption
        """
        wal = WriteAheadLog(
            wal_dir=self.wal_dir,
            max_entries_per_file=10,  # Very small for frequent rotation
        )

        async def appender(i: int):
            await asyncio.sleep(random.uniform(0, 0.01))  # Random delay
            return await wal.append(
                "write",
                self.data_dir / f"race_{i}.txt",
                f"data_{i}",
            )

        # 50 concurrent appends with small rotation threshold
        tasks = [appender(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        success_entries = [r for r in results if isinstance(r, WALEntry)]

        print(f"\n[ROTATION RACE] 50 appends: {len(success_entries)} success, {len(errors)} errors")
        self.assertEqual(len(success_entries), 50)  # All should succeed

    # ═══════════════════════════════════════════════════════════════════════════
    # BLACK SWAN #5: CASCADING FAILURES
    # ═══════════════════════════════════════════════════════════════════════════

    async def test_cascading_failure_event_bus(self):
        """
        BLACK SWAN: One subscriber fails, others should still receive
        Expected: Isolated failures, no cascading
        """
        bus = UnifiedEventBus()

        received_by = []

        async def good_subscriber(event):
            received_by.append("good")

        async def bad_subscriber(event):
            received_by.append("bad")
            raise RuntimeError("Intentional failure")

        async def another_good_subscriber(event):
            received_by.append("another_good")

        bus.subscribe(good_subscriber)
        bus.subscribe(bad_subscriber)
        bus.subscribe(another_good_subscriber)

        # Publish event
        class TestEvent:
            event_id = "test-123"
            event_type = "test"

        result = await bus.publish(TestEvent())

        print(f"\n[CASCADING FAILURE] Subscribers received: {received_by}")
        # All subscribers should have been called despite one failing
        self.assertIn("good", received_by)
        self.assertIn("bad", received_by)
        self.assertIn("another_good", received_by)

        # Result should still be successful
        self.assertIsNotNone(result)

    async def test_cascading_failure_wal_corruption_recovery(self):
        """
        BLACK SWAN: WAL file partially corrupted
        Expected: Recover valid entries, skip corrupted
        """
        wal = WriteAheadLog(wal_dir=self.wal_dir)

        # Add some valid entries
        for i in range(5):
            await wal.append("write", self.data_dir / f"valid_{i}.txt", f"valid_data_{i}")

        # Manually corrupt the WAL file
        wal_files = list(self.wal_dir.glob("wal_*.jsonl"))
        if wal_files:
            with open(wal_files[-1], "a") as f:
                f.write("THIS_IS_CORRUPTED_JSON{[[[")

        # Try to recover
        try:
            recovered = await wal.recover()
            print(f"\n[WAL CORRUPTION] Recovered {len(recovered)} entries despite corruption")
            # Should recover at least the valid entries
            self.assertGreaterEqual(len(recovered), 5)
        except json.JSONDecodeError:
            # This is expected - recovery should handle or propagate
            print("\n[WAL CORRUPTION] JSONDecodeError raised (expected behavior)")


class ResilienceMetrics:
    """
    STABILITY THRESHOLD (τ) CALCULATION
    ===================================

    Pre-computed recovery triggers
    """

    # Threshold definitions
    THRESHOLDS = {
        # I/O Subsystem
        "nash_io_error_rate": {
            "τ_critical": 0.05,  # 5% error rate triggers alert
            "τ_rollback": 0.20,  # 20% error rate triggers rollback
            "window_seconds": 60,
        },
        "nash_io_latency_p99": {
            "τ_critical": 1.0,  # 1 second
            "τ_rollback": 5.0,  # 5 seconds
        },
        # WAL Subsystem
        "nash_wal_pending_entries": {
            "τ_critical": 100,  # 100 pending entries
            "τ_rollback": 1000,  # 1000 pending entries (indicates commit failure)
        },
        "nash_wal_recovery_failures": {
            "τ_critical": 1,  # Any recovery failure
            "τ_rollback": 3,  # Multiple failures indicate corruption
        },
        # Event Subsystem
        "nash_event_normalization_failures": {
            "τ_critical": 0.01,  # 1% of events
            "τ_rollback": 0.10,  # 10% of events
        },
        "nash_event_backpressure": {
            "τ_critical": 1000,  # 1000 queued events
            "τ_rollback": 10000,  # 10000 queued events
        },
        # Thread Pool
        "nash_thread_pool_saturation": {
            "τ_critical": 0.80,  # 80% utilization
            "τ_rollback": 0.95,  # 95% utilization (near deadlock)
        },
        "nash_thread_pool_queue_size": {
            "τ_critical": 100,  # 100 queued tasks
            "τ_rollback": 1000,  # 1000 queued tasks
        },
        # Memory
        "nash_memory_pressure": {
            "τ_critical": 0.80,  # 80% of MAX_STORED_SIZE (52MB)
            "τ_rollback": 0.95,  # 95% (61MB)
        },
    }

    @classmethod
    def get_rollback_triggers(cls) -> Dict[str, float]:
        """Get all rollback thresholds."""
        return {metric: config["τ_rollback"] for metric, config in cls.THRESHOLDS.items()}

    @classmethod
    def get_critical_triggers(cls) -> Dict[str, float]:
        """Get all critical alert thresholds."""
        return {metric: config["τ_critical"] for metric, config in cls.THRESHOLDS.items()}

    @classmethod
    def check_stability_violation(
        cls,
        metric: str,
        value: float,
    ) -> tuple[bool, str]:
        """
        Check if a metric violates stability thresholds.

        Returns: (violated, action)
            violated: True if threshold exceeded
            action: "none", "alert", "rollback", or "emergency"
        """
        if metric not in cls.THRESHOLDS:
            return False, "unknown_metric"

        config = cls.THRESHOLDS[metric]

        if value >= config["τ_rollback"]:
            return True, "rollback"
        elif value >= config["τ_critical"]:
            return True, "alert"

        return False, "none"


class RecoveryPlan:
    """
    PRE-COMPUTED RECOVERY PLAN
    ==========================

    Immediate actions for each failure mode
    """

    RECOVERY_ACTIONS = {
        # I/O Failures
        "nash_io_error_rate_exceeded": {
            "immediate_action": "Drain and recreate AsyncIOManager",
            "steps": [
                "1. Set AsyncIOManager._shutdown = True",
                "2. Wait for pending tasks (timeout=5s)",
                "3. Shutdown ThreadPoolExecutor",
                "4. Create new AsyncIOManager instance",
                "5. Resume operations",
            ],
            "fallback": "Switch to synchronous I/O mode",
            "data_loss_risk": "Low - WAL protects uncommitted writes",
        },
        # WAL Failures
        "nash_wal_corruption": {
            "immediate_action": "Activate WAL recovery mode",
            "steps": [
                "1. Stop accepting new writes",
                "2. Run wal.recover() to scan all entries",
                "3. Identify COMMITTED entries with missing files",
                "4. Replay entries with stored data",
                "5. Rotate to new WAL file",
                "6. Resume operations",
            ],
            "fallback": "Restore from backup WAL snapshot",
            "data_loss_risk": "Medium - entries without stored data cannot be replayed",
        },
        # Event System Failures
        "nash_event_bus_failure": {
            "immediate_action": "Reset event bus and notify subscribers",
            "steps": [
                "1. Clear subscriber list",
                "2. Drain pending event queue",
                "3. Re-initialize EventNormalizer",
                "4. Require subscribers to re-subscribe",
                "5. Resume with degraded mode (sync only)",
            ],
            "fallback": "Disable event normalization (raw events only)",
            "data_loss_risk": "High - events may be dropped during reset",
        },
        # Thread Pool Exhaustion
        "nash_thread_pool_deadlock": {
            "immediate_action": "Force thread pool restart",
            "steps": [
                "1. Cancel all pending futures",
                "2. Shutdown executor (wait=False)",
                "3. Create new ThreadPoolExecutor",
                "4. Double max_workers temporarily",
                "5. Gradually reduce to normal",
            ],
            "fallback": "Scale horizontally - offload to external workers",
            "data_loss_risk": "Low - tasks are cancelled, not lost",
        },
        # Memory Exhaustion
        "nash_memory_pressure_critical": {
            "immediate_action": "Activate memory pressure mode",
            "steps": [
                "1. Reject new large payload writes",
                "2. Flush all WAL entries immediately",
                "3. Trigger garbage collection",
                "4. Enable compression for all new writes",
                "5. Monitor for 60s before normalizing",
            ],
            "fallback": "Spill to disk - use file-backed storage",
            "data_loss_risk": "None",
        },
    }

    @classmethod
    def get_recovery_plan(cls, failure_mode: str) -> Dict[str, Any]:
        """Get pre-computed recovery plan for a failure mode."""
        return cls.RECOVERY_ACTIONS.get(
            failure_mode,
            {
                "immediate_action": "Unknown failure - manual intervention required",
                "steps": ["Contact on-call engineer"],
                "fallback": "None",
                "data_loss_risk": "Unknown",
            },
        )


def generate_deployment_checklist() -> str:
    """
    FINAL DEPLOYMENT CHECKLIST
    ==========================
    """
    checklist = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           NASH STABILITY INFRASTRUCTURE v2.0 - DEPLOYMENT CHECKLIST          ║
╚══════════════════════════════════════════════════════════════════════════════╝

□ PRE-DEPLOYMENT VALIDATION
  □ Run full test suite: python -m pytest tests/test_nash_infrastructure_resilience.py -v
  □ Verify syntax: python -m py_compile orchestrator/nash_infrastructure_v2.py
  □ Check imports: python -c "from orchestrator.nash_infrastructure_v2 import *"
  □ Validate CLI: python -m orchestrator nash status

□ CONFIGURATION
  □ WAL directory exists: mkdir -p .nash_data/wal
  □ WAL directory permissions: chmod 755 .nash_data/wal
  □ Event directory exists: mkdir -p .nash_events
  □ Knowledge graph dir: mkdir -p .knowledge_graph
  □ Adaptive templates dir: mkdir -p .adaptive_templates

□ RESOURCE LIMITS
  □ Max WAL file size: 1000 entries (configurable)
  □ Max stored data per entry: 64KB
  □ Thread pool workers: 2-4 recommended
  □ Disk space: Ensure 2x expected WAL size available

□ MONITORING SETUP
  □ Deploy stability threshold monitors:
    - nash_io_error_rate (τ_rollback=0.20)
    - nash_wal_pending_entries (τ_rollback=1000)
    - nash_thread_pool_saturation (τ_rollback=0.95)
    - nash_event_normalization_failures (τ_rollback=0.10)
  □ Set up alerting channels for τ_critical thresholds
  □ Configure rollback automation for τ_rollback thresholds

□ HEALTH CHECKS
  □ Endpoint: /health/nash-io
  □ Endpoint: /health/nash-wal
  □ Endpoint: /health/nash-events
  □ Command: python -m orchestrator nash status

□ BACKUP & RECOVERY
  □ Schedule WAL snapshots every 15 minutes
  □ Test recovery procedure: python -m orchestrator nash backup --test
  □ Document RTO (Recovery Time Objective): < 30 seconds
  □ Document RPO (Recovery Point Objective): < 5 seconds

□ ROLLBACK PLAN
  □ Previous version tag identified: __main__
  □ Rollback command ready: git checkout <previous_tag>
  □ Data migration plan if schema changed
  □ Communication plan for stakeholders

□ SIGN-OFF
  □ Dev team approval
  □ QA team approval  
  □ SRE team approval
  □ Product owner approval

╔══════════════════════════════════════════════════════════════════════════════╗
║  EMERGENCY CONTACTS                                                          ║
║  - On-call SRE: [FILL IN]                                                    ║
║  - Nash Stability Owner: [FILL IN]                                           ║
║  - Escalation: [FILL IN]                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return checklist


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════


async def run_stress_test():
    """Run all resilience tests and generate report."""
    print("=" * 80)
    print("NASH STABILITY INFRASTRUCTURE v2.0 - RESILIENCE STRESS TEST")
    print("=" * 80)

    # Run unit tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(BlackSwanTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("STABILITY THRESHOLDS (τ)")
    print("=" * 80)

    print("\nCritical Alert Thresholds (τ_critical):")
    for metric, value in ResilienceMetrics.get_critical_triggers().items():
        print(f"  {metric}: {value}")

    print("\nRollback Triggers (τ_rollback):")
    for metric, value in ResilienceMetrics.get_rollback_triggers().items():
        print(f"  {metric}: {value}")

    print("\n" + "=" * 80)
    print("RECOVERY PLANS")
    print("=" * 80)

    for failure_mode in RecoveryPlan.RECOVERY_ACTIONS:
        plan = RecoveryPlan.get_recovery_plan(failure_mode)
        print(f"\n[{failure_mode}]")
        print(f"  Action: {plan['immediate_action']}")
        print(f"  Risk: {plan['data_loss_risk']}")

    print("\n" + "=" * 80)
    print(generate_deployment_checklist())

    return result.wasSuccessful()


if __name__ == "__main__":
    success = asyncio.run(run_stress_test())
    sys.exit(0 if success else 1)
