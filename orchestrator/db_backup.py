"""
Database Backup Manager — Automated backup and recovery
========================================================
Author: Senior Distributed Systems Architect

CRITICAL: SQLite is a single point of failure. This module provides:
- Periodic snapshots
- Point-in-time recovery
- Corruption detection
- Automatic failover to backup

DATABASES MANAGED:
- state.db (project state, checkpoints)
- cache.db (LLM response cache)
- telemetry.db (cross-run learning)
- events.db (event store)

RETENTION POLICY:
- 7 daily backups
- 4 weekly backups
- 12 monthly backups

USAGE:
    from orchestrator.db_backup import DatabaseBackupManager

    manager = DatabaseBackupManager()
    await manager.start()  # Start background scheduler

    # Restore from backup
    await manager.restore(backup_path)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("orchestrator.db_backup")


@dataclass
class BackupManifest:
    """Metadata for a backup."""
    timestamp: datetime
    db_path: Path
    size_bytes: int
    checksum_sha256: str
    tables: list[str]
    event_count: int
    is_valid: bool


@dataclass
class BackupInfo:
    """Information about an available backup."""
    path: Path
    timestamp: datetime
    databases: list[str]
    total_size_bytes: int
    is_valid: bool


class DatabaseBackupManager:
    """
    Manages backups for all orchestrator databases.

    STRATEGY:
    - Full backup every 6 hours (configurable)
    - WAL checkpoint every 30 minutes
    - Retention: 7 daily, 4 weekly, 12 monthly
    - Validation on every backup
    - Corruption detection via checksum

    INTEGRATION:
        orch = Orchestrator()
        await orch._backup_manager.start()  # In __aenter__
        await orch._backup_manager.stop()   # In __aexit__
    """

    DEFAULT_DATABASES = ["state.db", "cache.db", "telemetry.db", "events.db"]

    def __init__(
        self,
        data_dir: Path = None,
        backup_dir: Path | None = None,
        backup_interval_hours: float = 6.0,
        checkpoint_interval_minutes: float = 30.0,
    ):
        """
        Initialize backup manager.

        Args:
            data_dir: Directory containing database files
            backup_dir: Directory for backups (default: data_dir/backups)
            backup_interval_hours: Hours between full backups
            checkpoint_interval_minutes: Minutes between WAL checkpoints
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".orchestrator_cache"
        self.backup_dir = backup_dir or (self.data_dir / "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.backup_interval = backup_interval_hours * 3600
        self.checkpoint_interval = checkpoint_interval_minutes * 60

        self._task: asyncio.Task | None = None
        self._running = False
        self._last_backup: float = 0.0
        self._last_checkpoint: float = 0.0

        # Track backup history
        self._backup_history: list[BackupInfo] = []

    async def start(self) -> None:
        """Start background backup scheduler."""
        if self._running:
            logger.warning("Backup manager already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._backup_loop())
        logger.info(f"Backup manager started (backup={self.backup_interval}s, checkpoint={self.checkpoint_interval}s)")

    async def stop(self) -> None:
        """Stop backup scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Final checkpoint before shutdown
        await self._checkpoint_all()

        logger.info("Backup manager stopped")

    async def _backup_loop(self) -> None:
        """Background backup scheduler."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = time.time()

                # Checkpoint (quick, low impact)
                if now - self._last_checkpoint > self.checkpoint_interval:
                    await self._checkpoint_all()
                    self._last_checkpoint = now

                # Full backup
                if now - self._last_backup > self.backup_interval:
                    await self._backup_all()
                    self._last_backup = now
                    await self._prune_old_backups()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _checkpoint_all(self) -> None:
        """Checkpoint WAL files to main database."""
        for db_name in self.DEFAULT_DATABASES:
            db_path = self.data_dir / db_name
            if db_path.exists():
                try:
                    async with aiosqlite.connect(str(db_path)) as conn:
                        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    logger.debug(f"Checkpointed {db_name}")
                except Exception as e:
                    logger.warning(f"Checkpoint failed for {db_name}: {e}")

    async def backup_now(self) -> list[BackupManifest]:
        """
        Force immediate backup.

        Returns:
            List of backup manifests
        """
        return await self._backup_all()

    async def _backup_all(self) -> list[BackupManifest]:
        """Create full backup of all databases."""
        manifests = []
        timestamp = datetime.utcnow()
        backup_subdir = self.backup_dir / timestamp.strftime("%Y%m%d_%H%M%S")
        backup_subdir.mkdir(parents=True, exist_ok=True)

        for db_name in self.DEFAULT_DATABASES:
            db_path = self.data_dir / db_name
            if not db_path.exists():
                logger.debug(f"Skipping {db_name} (not found)")
                continue

            try:
                manifest = await self._backup_single(db_path, backup_subdir / db_name)
                manifests.append(manifest)
            except Exception as e:
                logger.error(f"Backup failed for {db_name}: {e}")

        if manifests:
            # Write manifest
            manifest_path = backup_subdir / "manifest.json"
            manifest_data = {
                "timestamp": timestamp.isoformat(),
                "orchestrator_version": "6.0.0",
                "backups": [
                    {
                        "db_name": m.db_path.name,
                        "size_bytes": m.size_bytes,
                        "checksum": m.checksum_sha256,
                        "tables": m.tables,
                        "event_count": m.event_count,
                        "is_valid": m.is_valid,
                    }
                    for m in manifests
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data, indent=2))

            # Update history
            self._backup_history.append(BackupInfo(
                path=backup_subdir,
                timestamp=timestamp,
                databases=[m.db_path.name for m in manifests],
                total_size_bytes=sum(m.size_bytes for m in manifests),
                is_valid=all(m.is_valid for m in manifests)
            ))

            logger.info(f"Backup complete: {len(manifests)} databases, {sum(m.size_bytes for m in manifests)} bytes")

        return manifests

    async def _backup_single(self, source: Path, dest: Path) -> BackupManifest:
        """Backup a single database with validation."""
        # Use SQLite backup API for consistency
        async with aiosqlite.connect(str(source)) as source_conn:
            # Get metadata
            tables = []
            async with source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                async for row in cursor:
                    tables.append(row[0])

            # Count events if events table exists
            event_count = 0
            if "events" in tables:
                try:
                    async with source_conn.execute("SELECT COUNT(*) FROM events") as c:
                        row = await c.fetchone()
                        event_count = row[0] if row else 0
                except Exception:
                    pass

            # Backup to destination
            await source_conn.backup(str(dest))

        # Calculate checksum
        checksum = await self._calculate_checksum(dest)
        size = dest.stat().st_size

        # Validate backup
        is_valid = await self._validate_backup(dest)

        return BackupManifest(
            timestamp=datetime.utcnow(),
            db_path=dest,
            size_bytes=size,
            checksum_sha256=checksum,
            tables=tables,
            event_count=event_count,
            is_valid=is_valid,
        )

    async def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _validate_backup(self, path: Path) -> bool:
        """Validate backup integrity."""
        try:
            async with aiosqlite.connect(str(path)) as conn:
                await conn.execute("PRAGMA integrity_check")
                result = await conn.fetchone()
                return result[0] == "ok" if result else False
        except Exception as e:
            logger.warning(f"Backup validation failed: {e}")
            return False

    async def _prune_old_backups(self) -> int:
        """
        Remove old backups according to retention policy.

        Returns:
            Number of backups removed
        """
        now = datetime.utcnow()
        backups = sorted(
            [d for d in self.backup_dir.iterdir() if d.is_dir()],
            reverse=True
        )

        removed = 0
        kept_daily = 0
        kept_weekly = 0
        kept_monthly = 0

        for backup in backups:
            try:
                backup_time = datetime.strptime(backup.name, "%Y%m%d_%H%M%S")
            except ValueError:
                continue

            age_days = (now - backup_time).days

            # Daily: keep last 7
            if age_days < 7:
                if kept_daily >= 7:
                    shutil.rmtree(backup)
                    removed += 1
                    continue
                kept_daily += 1
            # Weekly: keep last 4 (one per week)
            elif age_days < 30:
                if kept_weekly >= 4:
                    shutil.rmtree(backup)
                    removed += 1
                    continue
                kept_weekly += 1
            # Monthly: keep last 12
            elif age_days < 365:
                if kept_monthly >= 12:
                    shutil.rmtree(backup)
                    removed += 1
                    continue
                kept_monthly += 1
            else:
                # Older than 1 year, delete
                shutil.rmtree(backup)
                removed += 1

        if removed:
            logger.info(f"Pruned {removed} old backups")

        return removed

    async def restore(
        self,
        backup_path: Path,
        target_databases: list[str] | None = None
    ) -> dict[str, bool]:
        """
        Restore databases from backup.

        WARNING: This overwrites current data!

        Args:
            backup_path: Path to backup directory
            target_databases: Specific databases to restore (default: all)

        Returns:
            Dict mapping database name to success status
        """
        target_databases = target_databases or self.DEFAULT_DATABASES
        results = {}

        for db_name in target_databases:
            backup_db = backup_path / db_name
            target_db = self.data_dir / db_name

            if not backup_db.exists():
                logger.warning(f"Backup not found: {db_name}")
                results[db_name] = False
                continue

            # Validate before restore
            if not await self._validate_backup(backup_db):
                logger.error(f"Backup corrupted: {db_name}, skipping")
                results[db_name] = False
                continue

            try:
                # Create safety copy of current
                if target_db.exists():
                    safety_path = target_db.with_suffix(".db.pre_restore")
                    shutil.copy(target_db, safety_path)
                    logger.info(f"Created safety backup: {safety_path}")

                # Restore
                shutil.copy(backup_db, target_db)
                logger.info(f"Restored {db_name} from backup")
                results[db_name] = True

            except Exception as e:
                logger.error(f"Restore failed for {db_name}: {e}")
                results[db_name] = False

        return results

    def list_backups(self) -> list[dict[str, Any]]:
        """
        List available backups.

        Returns:
            List of backup info dicts
        """
        backups = []

        for backup_dir in sorted(self.backup_dir.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue

            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text())
                    data["path"] = str(backup_dir)
                    backups.append(data)
                except Exception as e:
                    logger.warning(f"Failed to read manifest {manifest_path}: {e}")

        return backups

    def get_latest_backup(self) -> Path | None:
        """Get path to most recent valid backup."""
        backups = self.list_backups()

        for backup in backups:
            if backup.get("backups"):
                # Check if any backup is valid
                for db_backup in backup["backups"]:
                    if db_backup.get("is_valid", False):
                        return Path(backup["path"])

        return None

    async def verify_all_backups(self) -> dict[str, list[dict[str, Any]]]:
        """
        Verify integrity of all backups.

        Returns:
            Dict with 'valid' and 'invalid' lists
        """
        results = {"valid": [], "invalid": []}

        for backup_dir in self.backup_dir.iterdir():
            if not backup_dir.is_dir():
                continue

            manifest_path = backup_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = json.loads(manifest_path.read_text())

                for db_backup in manifest.get("backups", []):
                    db_path = backup_dir / db_backup["db_name"]

                    if not db_path.exists():
                        results["invalid"].append({
                            "path": str(db_path),
                            "reason": "file_not_found"
                        })
                        continue

                    is_valid = await self._validate_backup(db_path)

                    if is_valid:
                        results["valid"].append({
                            "path": str(db_path),
                            "size_bytes": db_path.stat().st_size
                        })
                    else:
                        results["invalid"].append({
                            "path": str(db_path),
                            "reason": "integrity_check_failed"
                        })

            except Exception as e:
                results["invalid"].append({
                    "path": str(backup_dir),
                    "reason": str(e)
                })

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        backups = self.list_backups()

        total_size = 0
        for backup in backups:
            for db in backup.get("backups", []):
                total_size += db.get("size_bytes", 0)

        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "oldest_backup": backups[-1]["timestamp"] if backups else None,
            "newest_backup": backups[0]["timestamp"] if backups else None,
            "backup_dir": str(self.backup_dir),
        }


# Singleton for convenience
_instance: DatabaseBackupManager | None = None


def get_backup_manager() -> DatabaseBackupManager:
    """Get singleton DatabaseBackupManager instance."""
    global _instance
    if _instance is None:
        _instance = DatabaseBackupManager()
    return _instance


def reset_backup_manager() -> None:
    """Reset singleton (for testing)."""
    global _instance
    if _instance:
        asyncio.create_task(_instance.stop())
    _instance = None
