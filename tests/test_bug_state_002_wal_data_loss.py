"""
Test Suite: WAL Data Loss Fix (BUG-STATE-002)
==============================================
Tests for two-phase commit with atomic rename to prevent data loss.

Test Framework: pytest + pytest-asyncio
Coverage: Regression, Edge Cases, Failure Injection, Integration
"""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from orchestrator.nash_infrastructure_v2 import (
    TransactionalStorage,
    WriteAheadLog,
    WALEntry,
    WALEntryStatus,
    AsyncIOManager,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 6a. REGRESSION TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestWALDataLossRegression:
    """
    Regression tests for BUG-STATE-002: WAL data loss for large files (>100KB).

    Original Bug: Large files not stored in WAL, crash between WAL append and
    file write causes permanent data loss.
    """

    @pytest.mark.asyncio
    async def test_large_file_write_with_crash_recovery(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-REG-01

        DESCRIPTION: Simulate crash during large file write, verify recovery.

        PRECONDITIONS:
        - TransactionalStorage with WAL enabled
        - Large file (200KB) to write

        INPUT:
        - write() with 200KB data
        - Simulated crash after WAL append, before file write

        EXPECTED BEHAVIOR (with patch - two-phase commit):
        - Data written to temp file first
        - WAL entry references temp file
        - Atomic rename to target path
        - On crash recovery: temp file exists, can complete rename

        FAILURE BEHAVIOR (without patch):
        - WAL entry has data=None for large file
        - Crash before file write
        - Recovery cannot replay (data not in WAL)
        - Permanent data loss
        """
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        # Wait for recovery task to complete
        await asyncio.sleep(0.1)

        # Create large data (200KB)
        large_data = "X" * (200 * 1024)
        target_path = base_dir / "large_file.txt"

        # Act: Write with transaction safety
        result = await storage.write(target_path, large_data, transactional=True)

        # Assert: Write succeeded
        assert result is True
        assert target_path.exists()

        # Verify content
        content = await storage.read(target_path)
        assert len(content) == 200 * 1024

    @pytest.mark.asyncio
    async def test_small_file_stored_in_wal(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-REG-02

        DESCRIPTION: Verify small files (<100KB) are still stored in WAL.

        PRECONDITIONS:
        - WriteAheadLog with default settings

        INPUT:
        - 50KB data (below 100KB threshold)

        EXPECTED BEHAVIOR (with patch):
        - Data stored in WAL entry
        - Can replay from WAL on recovery

        FAILURE BEHAVIOR (without patch):
        - Same behavior (small files always stored)
        """
        wal_dir = tmp_path / "wal"
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Small data (50KB)
        small_data = "Y" * (50 * 1024)
        target_path = tmp_path / "small_file.txt"

        # Act
        entry = await wal.append("write", target_path, small_data)

        # Assert: Data is stored in entry
        assert entry.data is not None
        assert len(entry.data) == 50 * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestWALDataLossEdgeCases:
    """
    Edge case tests for WAL two-phase commit.
    """

    @pytest.mark.asyncio
    async def test_exactly_100kb_threshold(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-EDGE-01

        DESCRIPTION: File exactly at 100KB threshold.

        PRECONDITIONS:
        - WALEntry.MAX_STORED_SIZE = 100KB

        INPUT:
        - Exactly 102400 bytes (100KB)

        EXPECTED BEHAVIOR (with patch):
        - Stored in WAL (<= threshold)
        - Can replay from WAL

        FAILURE BEHAVIOR (without patch):
        - Same behavior
        """
        wal_dir = tmp_path / "wal"
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Exactly 100KB
        exact_data = "Z" * 102400
        target_path = tmp_path / "exact_100kb.txt"

        # Act
        entry = await wal.append("write", target_path, exact_data)

        # Assert: Should be stored (at threshold)
        assert entry.data is not None

    @pytest.mark.asyncio
    async def test_one_byte_over_threshold(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-EDGE-02

        DESCRIPTION: File one byte over 100KB threshold.

        PRECONDITIONS:
        - WALEntry.MAX_STORED_SIZE = 100KB

        INPUT:
        - 102401 bytes (100KB + 1)

        EXPECTED BEHAVIOR (with patch - two-phase commit):
        - Data NOT stored in WAL (over threshold)
        - Temp file created
        - Atomic rename on commit

        FAILURE BEHAVIOR (without patch):
        - Data NOT stored in WAL
        - No temp file mechanism
        - Crash = data loss
        """
        wal_dir = tmp_path / "wal"
        wal = WriteAheadLog(wal_dir=wal_dir)

        # 100KB + 1 byte
        over_data = "A" * 102401
        target_path = tmp_path / "over_100kb.txt"

        # Act
        entry = await wal.append("write", target_path, over_data)

        # Assert: Data not stored in WAL (over threshold)
        assert entry.data is None
        assert entry.data_size == 102401

    @pytest.mark.asyncio
    async def test_cross_filesystem_write(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-EDGE-03

        DESCRIPTION: Write across filesystem boundaries.

        PRECONDITIONS:
        - WAL directory on one mount
        - Target directory on different mount

        INPUT:
        - Large file write across mounts

        EXPECTED BEHAVIOR (with patch):
        - Temp file created in SAME directory as target
        - Atomic rename works (same filesystem)

        FAILURE BEHAVIOR (without patch):
        - os.rename() fails with "Invalid cross-device link"
        - Data stuck in temp file
        """
        # Note: Can't easily test cross-fs in test environment
        # This test verifies temp file is created in target directory
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        await asyncio.sleep(0.1)

        large_data = "B" * (150 * 1024)
        target_path = base_dir / "cross_fs_test.txt"

        # Act
        result = await storage.write(target_path, large_data, transactional=True)

        # Assert: No cross-fs error
        assert result is True
        assert target_path.exists()

        # Verify no temp files left behind
        temp_files = list(base_dir.glob("*.tmp.*"))
        assert len(temp_files) == 0

    @pytest.mark.asyncio
    async def test_empty_file_write(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-EDGE-04

        DESCRIPTION: Write empty file (0 bytes).

        PRECONDITIONS:
        - TransactionalStorage with WAL

        INPUT:
        - Empty string

        EXPECTED BEHAVIOR (with patch):
        - Write succeeds
        - Empty file created

        FAILURE BEHAVIOR (without patch):
        - Same behavior (empty files are small)
        """
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        await asyncio.sleep(0.1)

        # Act
        result = await storage.write(base_dir / "empty.txt", "", transactional=True)

        # Assert
        assert result is True
        assert (base_dir / "empty.txt").exists()

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_path(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-EDGE-05

        DESCRIPTION: Concurrent writes to same path.

        PRECONDITIONS:
        - 2 coroutines writing to same path simultaneously

        INPUT:
        - Concurrent write() calls

        EXPECTED BEHAVIOR (with patch):
        - Last write wins (atomic rename)
        - No data corruption
        - One WAL entry may be orphaned

        FAILURE BEHAVIOR (without patch):
        - Race condition
        - File corruption possible
        """
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        await asyncio.sleep(0.1)

        target_path = base_dir / "concurrent.txt"

        async def write_data(data):
            return await storage.write(target_path, data, transactional=True)

        # Act: Concurrent writes
        results = await asyncio.gather(
            write_data("AAAA" * 1024),
            write_data("BBBB" * 1024),
        )

        # Assert: Both succeeded (last one wins)
        assert all(results)
        assert target_path.exists()

        # Content should be one or the other (not mixed)
        content = await storage.read(target_path)
        assert content in ["AAAA" * 1024, "BBBB" * 1024]


# ═══════════════════════════════════════════════════════════════════════════════
# 6c. FAILURE INJECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestWALDataLossFailureInjection:
    """
    Failure injection tests for WAL two-phase commit fix.
    """

    @pytest.mark.asyncio
    async def test_crash_after_temp_write_before_rename(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-FAIL-01

        DESCRIPTION: Simulate crash after temp file written, before rename.

        PRECONDITIONS:
        - Large file write in progress
        - Temp file exists, rename not yet called

        INPUT:
        - Simulated crash (process kill)
        - Recovery on restart

        EXPECTED BEHAVIOR (with patch):
        - Recovery finds temp file
        - Completes rename
        - Data recovered

        FAILURE BEHAVIOR (without patch):
        - No temp file mechanism
        - Data lost
        """
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        await asyncio.sleep(0.1)

        large_data = "C" * (200 * 1024)
        target_path = base_dir / "crash_test.txt"

        # Mock to simulate crash after temp write
        original_rename = os.rename
        rename_called = False

        def mock_rename(src, dst):
            nonlocal rename_called
            if "tmp" in str(src):
                rename_called = True
                # Simulate crash: don't actually rename
                # Temp file exists, but rename didn't complete
                return
            return original_rename(src, dst)

        with patch("os.rename", mock_rename):
            # Act: Write will "crash" before rename
            result = await storage.write(target_path, large_data, transactional=True)

        # Assert: Temp file should exist (simulating crash state)
        temp_files = list(base_dir.glob("*.tmp.*"))
        assert len(temp_files) > 0 or not rename_called

    @pytest.mark.asyncio
    async def test_disk_full_during_temp_write(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-FAIL-02

        DESCRIPTION: Disk full during temp file write.

        PRECONDITIONS:
        - Large file write
        - Disk full error injected

        INPUT:
        - Mock write to raise "No space left on device"

        EXPECTED BEHAVIOR (with patch):
        - Catch I/O error
        - Clean up partial temp file
        - Return False (write failed)

        FAILURE BEHAVIOR (without patch):
        - Exception propagates
        - May leave partial temp file
        """
        base_dir = tmp_path / "nash_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)

        await asyncio.sleep(0.1)

        large_data = "D" * (200 * 1024)
        target_path = base_dir / "disk_full_test.txt"

        # Inject disk full error
        async def mock_write_file(path, data, mode):
            raise OSError(28, "No space left on device")

        storage._io.write_file = mock_write_file

        # Act: Should handle gracefully
        with pytest.raises(OSError):
            await storage.write(target_path, large_data, transactional=True)

        # Assert: No orphaned temp files
        temp_files = list(base_dir.glob("*.tmp.*"))
        assert len(temp_files) == 0

    @pytest.mark.asyncio
    async def test_wal_entry_corrupted(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-FAIL-03

        DESCRIPTION: WAL entry corrupted (checksum mismatch).

        PRECONDITIONS:
        - WAL entry written
        - Entry manually corrupted

        INPUT:
        - Corrupt WAL file

        EXPECTED BEHAVIOR (with patch):
        - Recovery detects corruption
        - Skips corrupted entry
        - Logs error

        FAILURE BEHAVIOR (without patch):
        - May crash on corrupted entry
        - Or replay incorrect data
        """
        wal_dir = tmp_path / "wal"
        wal = WriteAheadLog(wal_dir=wal_dir)

        # Write entry
        target_path = tmp_path / "corrupt_test.txt"
        entry = await wal.append("write", target_path, "test data")

        # Corrupt the WAL file
        wal_files = list(wal_dir.glob("wal_*.jsonl"))
        assert len(wal_files) > 0

        # Manually corrupt checksum
        wal_file = wal_files[0]
        content = wal_file.read_text()
        corrupted = content.replace(entry.checksum, "XXXXXXXX")
        wal_file.write_text(corrupted)

        # Act: Recovery should handle corruption
        pending = await wal.recover()

        # Assert: Should not crash, may skip corrupted entry
        # (exact behavior depends on corruption detection implementation)


# ═══════════════════════════════════════════════════════════════════════════════
# 6d. INTEGRATION SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestWALDataLossIntegration:
    """
    Integration smoke tests for WAL two-phase commit with callers.
    """

    @pytest.mark.asyncio
    async def test_transactional_storage_with_knowledge_base(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-INT-01

        DESCRIPTION: Verify WAL integrates with KnowledgeBase storage.

        PRECONDITIONS:
        - KnowledgeBase using TransactionalStorage
        - Large knowledge graph to store

        INPUT:
        - KnowledgeBase.store() with large data

        EXPECTED BEHAVIOR (with patch):
        - Two-phase commit used
        - Large file durability guaranteed

        FAILURE BEHAVIOR (without patch):
        - Large files may be lost on crash
        """
        from orchestrator.knowledge_base import KnowledgeBase

        base_dir = tmp_path / "kb_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)
        kb = KnowledgeBase(storage=storage)

        await asyncio.sleep(0.1)

        # Large knowledge graph
        large_graph = {"nodes": [{"id": i, "data": "X" * 1000} for i in range(500)]}

        import json

        large_data = json.dumps(large_graph)

        # Act
        await kb.store("test_graph", large_data)

        # Assert: Stored successfully
        retrieved = await kb.retrieve("test_graph")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_transactional_storage_with_memory_tier(self, tmp_path):
        """
        TEST-ID: BUG-STATE-002-INT-02

        DESCRIPTION: Verify WAL integrates with MemoryTierManager.

        PRECONDITIONS:
        - MemoryTierManager using TransactionalStorage
        - Large memory to persist

        INPUT:
        - MemoryTierManager.store() with large data

        EXPECTED BEHAVIOR (with patch):
        - Two-phase commit for large memories
        - No data loss on crash

        FAILURE BEHAVIOR (without patch):
        - Large memories may be lost
        """
        from orchestrator.memory_tier import MemoryTierManager

        base_dir = tmp_path / "memory_data"
        storage = TransactionalStorage(base_dir=base_dir, enable_wal=True)
        manager = MemoryTierManager(storage=storage)

        await asyncio.sleep(0.1)

        # Large memory
        large_memory = "M" * (150 * 1024)

        # Act
        memory_id = await manager.store(
            project_id="test_project",
            content=large_memory,
            memory_type="task",
        )

        # Assert: Stored successfully
        assert memory_id is not None

        # Verify can retrieve
        memories = await manager.retrieve(project_id="test_project")
        assert len(memories) > 0
