"""
MemoryBank — Persistent cross-run memory
=======================================
Module for maintaining persistent memory across different runs of the orchestrator.

Pattern: Repository
Async: Yes — for I/O-bound storage operations
Layer: L1 Infrastructure

Usage:
    from orchestrator.memory_bank import MemoryBank
    bank = MemoryBank(memory_dir="./memory")
    await bank.store(key="project_context", value={"info": "..."}, tags=["project", "context"])
    retrieved_value = await bank.retrieve(key="project_context")
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.memory_bank")


class MemoryEntry:
    """Represents a single memory entry."""

    def __init__(self, key: str, value: Any, tags: list[str], timestamp: datetime,
                 expiry: datetime | None = None, importance: float = 0.5):
        self.key = key
        self.value = value
        self.tags = tags
        self.timestamp = timestamp
        self.expiry = expiry
        self.importance = importance  # 0.0-1.0, higher is more important
        self.access_count = 0
        self.checksum = self._calculate_checksum(value)

    def _calculate_checksum(self, value: Any) -> str:
        """Calculate a checksum for the memory value."""
        value_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(value_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert the memory entry to a dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "importance": self.importance,
            "access_count": self.access_count,
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create a memory entry from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        expiry = None
        if data.get("expiry"):
            expiry = datetime.fromisoformat(data["expiry"])

        entry = cls(
            key=data["key"],
            value=data["value"],
            tags=data["tags"],
            timestamp=timestamp,
            expiry=expiry,
            importance=data.get("importance", 0.5)
        )
        entry.access_count = data.get("access_count", 0)

        # Verify checksum
        if entry.checksum != data["checksum"]:
            raise ValueError("Memory entry corrupted: checksum mismatch")

        return entry

    def is_expired(self) -> bool:
        """Check if the memory entry is expired."""
        if self.expiry:
            return datetime.now() > self.expiry
        return False


class MemoryBank:
    """Maintains persistent memory across different runs of the orchestrator."""

    def __init__(self, memory_dir: str = "./memory", retention_days: int = 30):
        """Initialize the memory bank."""
        self.memory_dir = Path(memory_dir)
        self.retention_days = retention_days
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    async def store(self, key: str, value: Any, tags: list[str] = None,
                    ttl_minutes: int | None = None, importance: float = 0.5) -> bool:
        """
        Store a value in memory with optional tags and expiration.

        Args:
            key: The key to store the value under
            value: The value to store
            tags: Optional tags for categorizing the memory
            ttl_minutes: Optional time-to-live in minutes
            importance: Importance of this memory (0.0-1.0)

        Returns:
            bool: True if stored successfully, False otherwise
        """
        # Determine expiry time
        expiry = None
        if ttl_minutes:
            expiry = datetime.now() + timedelta(minutes=ttl_minutes)

        # Create memory entry
        entry = MemoryEntry(
            key=key,
            value=value,
            tags=tags or [],
            timestamp=datetime.now(),
            expiry=expiry,
            importance=importance
        )

        # Create filename with hash of key to avoid invalid characters
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"memory_{key_hash}.json"
        filepath = self.memory_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)

            logger.info(f"Stored memory entry: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to store memory entry {key}: {e}")
            return False

    async def retrieve(self, key: str) -> Any | None:
        """
        Retrieve a value from memory by key.

        Args:
            key: The key to retrieve the value for

        Returns:
            The stored value or None if not found/expired
        """
        # Find the memory file for this key
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"memory_{key_hash}.json"
        filepath = self.memory_dir / filename

        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)

            entry = MemoryEntry.from_dict(data)

            # Check if expired
            if entry.is_expired():
                await self._delete_file(filepath)
                logger.info(f"Expired memory entry deleted: {key}")
                return None

            # Increment access count
            entry.access_count += 1

            # Update the file with incremented access count
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)

            logger.info(f"Retrieved memory entry: {key}")
            return entry.value
        except FileNotFoundError:
            logger.info(f"Memory entry not found: {key}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory entry {key}: {e}")
            return None

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for memory entries by tags.

        Args:
            tags: Tags to search for
            limit: Maximum number of results to return

        Returns:
            List of memory entries matching the tags
        """
        results = []

        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                entry = MemoryEntry.from_dict(data)

                # Check if expired
                if entry.is_expired():
                    await self._delete_file(filepath)
                    continue

                # Check if entry has any of the requested tags
                if any(tag in entry.tags for tag in tags):
                    results.append({
                        "key": entry.key,
                        "value": entry.value,
                        "tags": entry.tags,
                        "timestamp": entry.timestamp,
                        "importance": entry.importance,
                        "access_count": entry.access_count
                    })
            except Exception as e:
                logger.error(f"Failed to read memory file {filepath}: {e}")

        # Sort by importance and access count (descending)
        results.sort(key=lambda x: (x["importance"], x["access_count"]), reverse=True)

        return results[:limit]

    async def search_by_content(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for memory entries by content.

        Args:
            query: Text to search for in memory values
            limit: Maximum number of results to return

        Returns:
            List of memory entries containing the query text
        """
        results = []
        query_lower = query.lower()

        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                entry = MemoryEntry.from_dict(data)

                # Check if expired
                if entry.is_expired():
                    await self._delete_file(filepath)
                    continue

                # Convert entry value to string for searching
                value_str = json.dumps(entry.value, default=str).lower()

                if query_lower in value_str:
                    results.append({
                        "key": entry.key,
                        "value": entry.value,
                        "tags": entry.tags,
                        "timestamp": entry.timestamp,
                        "importance": entry.importance,
                        "access_count": entry.access_count
                    })
            except Exception as e:
                logger.error(f"Failed to read memory file {filepath}: {e}")

        # Sort by importance and access count (descending)
        results.sort(key=lambda x: (x["importance"], x["access_count"]), reverse=True)

        return results[:limit]

    async def update_importance(self, key: str, importance: float) -> bool:
        """
        Update the importance of a memory entry.

        Args:
            key: The key of the entry to update
            importance: New importance value (0.0-1.0)

        Returns:
            bool: True if updated successfully, False otherwise
        """
        # Retrieve the current entry
        current_value = await self.retrieve(key)
        if current_value is None:
            return False

        # Find the file and update importance
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"memory_{key_hash}.json"
        filepath = self.memory_dir / filename

        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)

            # Update importance
            data["importance"] = max(0.0, min(1.0, importance))  # Clamp between 0.0 and 1.0

            # Write back to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Updated importance for memory entry: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update importance for {key}: {e}")
            return False

    async def add_tags(self, key: str, tags: list[str]) -> bool:
        """
        Add tags to an existing memory entry.

        Args:
            key: The key of the entry to update
            tags: Tags to add

        Returns:
            bool: True if updated successfully, False otherwise
        """
        # Retrieve the current entry
        current_value = await self.retrieve(key)
        if current_value is None:
            return False

        # Find the file and update tags
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"memory_{key_hash}.json"
        filepath = self.memory_dir / filename

        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)

            # Add new tags, avoiding duplicates
            current_tags = data.get("tags", [])
            for tag in tags:
                if tag not in current_tags:
                    current_tags.append(tag)

            data["tags"] = current_tags

            # Write back to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Added tags to memory entry: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to add tags to {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a memory entry.

        Args:
            key: The key of the entry to delete

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        filename = f"memory_{key_hash}.json"
        filepath = self.memory_dir / filename

        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted memory entry: {key}")
                return True
            else:
                logger.info(f"Memory entry not found for deletion: {key}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete memory entry {key}: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired memory entries.

        Returns:
            int: Number of entries cleaned up
        """
        cleaned_count = 0

        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                entry = MemoryEntry.from_dict(data)

                if entry.is_expired():
                    await self._delete_file(filepath)
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired memory entry: {entry.key}")
            except Exception as e:
                logger.error(f"Failed to check expiration for {filepath}: {e}")

        return cleaned_count

    async def cleanup_low_importance(self, importance_threshold: float = 0.2,
                                     min_age_days: int = 7) -> int:
        """
        Clean up low-importance memory entries that are older than min_age_days.

        Args:
            importance_threshold: Entries with importance below this will be considered for cleanup
            min_age_days: Minimum age in days for entries to be eligible for cleanup

        Returns:
            int: Number of entries cleaned up
        """
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=min_age_days)

        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                entry = MemoryEntry.from_dict(data)

                # Check if entry is old enough and has low importance
                if (entry.timestamp < cutoff_date and
                    entry.importance < importance_threshold and
                    not entry.is_expired()):  # Don't delete already expired entries here
                    await self._delete_file(filepath)
                    cleaned_count += 1
                    logger.info(f"Cleaned up low-importance memory entry: {entry.key}")
            except Exception as e:
                logger.error(f"Failed to check importance for {filepath}: {e}")

        return cleaned_count

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the memory bank.

        Returns:
            Dict with statistics about the memory bank
        """
        total_entries = 0
        expired_entries = 0
        total_size = 0
        importance_sum = 0.0
        access_count_sum = 0

        for filepath in self.memory_dir.glob("memory_*.json"):
            try:
                with open(filepath, encoding='utf-8') as f:
                    data = json.load(f)

                entry = MemoryEntry.from_dict(data)
                total_entries += 1
                total_size += filepath.stat().st_size
                importance_sum += entry.importance
                access_count_sum += entry.access_count

                if entry.is_expired():
                    expired_entries += 1
            except Exception as e:
                logger.error(f"Failed to read memory file {filepath} for stats: {e}")

        avg_importance = importance_sum / total_entries if total_entries > 0 else 0.0
        avg_access_count = access_count_sum / total_entries if total_entries > 0 else 0.0

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_size_bytes": total_size,
            "average_importance": avg_importance,
            "average_access_count": avg_access_count,
            "memory_dir": str(self.memory_dir)
        }

    async def _delete_file(self, filepath: Path) -> bool:
        """Delete a memory file."""
        try:
            filepath.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory file {filepath}: {e}")
            return False

    async def export_memory(self, export_path: str) -> bool:
        """
        Export all memory entries to a single JSON file.

        Args:
            export_path: Path to export the memory to

        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            all_entries = []

            for filepath in self.memory_dir.glob("memory_*.json"):
                try:
                    with open(filepath, encoding='utf-8') as f:
                        data = json.load(f)

                    entry = MemoryEntry.from_dict(data)

                    # Only export non-expired entries
                    if not entry.is_expired():
                        all_entries.append(entry.to_dict())
                except Exception as e:
                    logger.error(f"Failed to read memory file {filepath} for export: {e}")

            # Write all entries to export file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(all_entries, f, indent=2, default=str)

            logger.info(f"Exported {len(all_entries)} memory entries to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export memory: {e}")
            return False

    async def import_memory(self, import_path: str) -> bool:
        """
        Import memory entries from a JSON file.

        Args:
            import_path: Path to import the memory from

        Returns:
            bool: True if imported successfully, False otherwise
        """
        try:
            with open(import_path, encoding='utf-8') as f:
                entries_data = json.load(f)

            import_count = 0

            for entry_data in entries_data:
                try:
                    entry = MemoryEntry.from_dict(entry_data)

                    # Store the entry using the regular store method
                    success = await self.store(
                        key=entry.key,
                        value=entry.value,
                        tags=entry.tags,
                        importance=entry.importance
                    )

                    if success:
                        import_count += 1
                except Exception as e:
                    logger.error(f"Failed to import memory entry {entry_data.get('key', 'unknown')}: {e}")

            logger.info(f"Imported {import_count} memory entries from {import_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to import memory: {e}")
            return False
