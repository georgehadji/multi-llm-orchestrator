"""
Multi-tier Memory — HOT/WARM/COLD Memory Hierarchy
==================================================

Implements Mnemo Cortex memory tiers:
- HOT (Days 1-3): Raw JSONL, instant keyword search
- WARM (Days 4-30): Summarized + embedded, semantic search
- COLD (Day 30+): Compressed archive, full scan

Usage:
    from orchestrator.memory_tier import MemoryTierManager, MemoryEntry

    manager = MemoryTierManager()

    # Store a memory
    await manager.store(
        project_id="project_001",
        content="User asked for fibonacci function",
        memory_type="task",
    )

    # Retrieve memories (searches all tiers)
    memories = await manager.retrieve(
        project_id="project_001",
        query="fibonacci",
        limit=5,
    )

    # Run tier migration
    await manager.migrate_tiers()
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .log_config import get_logger

# Import BM25 search for hybrid retrieval
try:
    from .bm25_search import BM25Search, get_bm25_search
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Search = None

logger = get_logger(__name__)


class MemoryTier(Enum):
    """Memory tier based on age."""
    HOT = "hot"     # Days 1-3: Raw JSONL, instant search
    WARM = "warm"   # Days 4-30: Summarized + embedded
    COLD = "cold"   # Day 30+: Compressed archive


class MemoryType(Enum):
    """Type of memory."""
    TASK = "task"
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    SUMMARY = "summary"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    project_id: str
    content: str
    memory_type: MemoryType
    tier: MemoryTier
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    embedding: list[float] | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> int:
        return (datetime.utcnow() - self.created_at).days

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=d["id"],
            project_id=d["project_id"],
            content=d["content"],
            memory_type=MemoryType(d["memory_type"]),
            tier=MemoryTier(d["tier"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            last_accessed=datetime.fromisoformat(d["last_accessed"]),
            access_count=d.get("access_count", 0),
            summary=d.get("summary"),
            metadata=d.get("metadata", {}),
        )


class MemoryTierManager:
    """
    Manages multi-tier memory system.

    Implements HOT/WARM/COLD hierarchy:
    - HOT: Recent memories, raw storage, fast access
    - WARM: Older memories, summarized, indexed for search
    - COLD: Archive storage, compressed, slower access
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        hot_ttl_days: int = 3,
        warm_ttl_days: int = 30,
        enable_auto_migration: bool = True,
        enable_bm25: bool = True,
    ):
        self.storage_path = storage_path or Path.home() / ".orchestrator_cache" / "memory"
        self.hot_ttl_days = hot_ttl_days
        self.warm_ttl_days = warm_ttl_days
        self.enable_auto_migration = enable_auto_migration
        self.enable_bm25 = enable_bm25 and HAS_BM25

        # Create tier directories
        self.tier_paths = {
            MemoryTier.HOT: self.storage_path / "hot",
            MemoryTier.WARM: self.storage_path / "warm",
            MemoryTier.COLD: self.storage_path / "cold",
        }

        for path in self.tier_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookup
        self._hot_index: dict[str, MemoryEntry] = {}
        self._warm_index: dict[str, MemoryEntry] = {}

        # BM25 search index for hybrid retrieval
        self._bm25_search: BM25Search | None = None
        if self.enable_bm25:
            self._bm25_search = get_bm25_search(str(self.storage_path / "search.db"))

        # Load existing memories
        self._load_index()

    def _memory_file_path(self, tier: MemoryTier, memory_id: str) -> Path:
        """Get file path for memory storage."""
        return self.tier_paths[tier] / f"{memory_id}.json"

    def _load_index(self) -> None:
        """Load memory index from disk."""
        # Load HOT tier
        for file_path in self.tier_paths[MemoryTier.HOT].glob("*.json"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    entry = MemoryEntry.from_dict(data)
                    self._hot_index[entry.id] = entry
            except Exception as e:
                logger.warning(f"Failed to load memory {file_path}: {e}")

        # Load WARM tier
        for file_path in self.tier_paths[MemoryTier.WARM].glob("*.json"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    entry = MemoryEntry.from_dict(data)
                    self._warm_index[entry.id] = entry
            except Exception as e:
                logger.warning(f"Failed to load memory {file_path}: {e}")

        logger.info(f"Loaded {len(self._hot_index)} hot, {len(self._warm_index)} warm memories")

    def _save_memory(self, entry: MemoryEntry) -> None:
        """Save memory to disk."""
        file_path = self._memory_file_path(entry.tier, entry.id)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entry.to_dict(), f, indent=2)

    def _get_tier_for_age(self, age_days: int) -> MemoryTier:
        """Determine tier based on age."""
        if age_days < self.hot_ttl_days:
            return MemoryTier.HOT
        elif age_days < self.warm_ttl_days:
            return MemoryTier.WARM
        else:
            return MemoryTier.COLD

    async def store(
        self,
        project_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.TASK,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a new memory entry.

        Returns the memory ID.
        """
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        entry = MemoryEntry(
            id=memory_id,
            project_id=project_id,
            content=content,
            memory_type=memory_type,
            tier=MemoryTier.HOT,  # New memories go to HOT
            created_at=now,
            last_accessed=now,
            metadata=metadata or {},
        )

        # Store in index
        self._hot_index[memory_id] = entry

        # Save to disk
        self._save_memory(entry)

        # Add to BM25 search index
        if self._bm25_search:
            await self._bm25_search.add_document(
                doc_id=memory_id,
                project_id=project_id,
                content=content,
                metadata={"memory_type": memory_type.value, **metadata} if metadata else {"memory_type": memory_type.value},
            )

        logger.debug(f"Stored memory {memory_id} in HOT tier")

        return memory_id

    async def retrieve(
        self,
        project_id: str,
        query: str | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 5,
        use_hybrid: bool = True,
    ) -> list[MemoryEntry]:
        """
        Retrieve memories for a project.

        Searches HOT first, then WARM, then COLD.
        If BM25 is enabled and query is provided, uses hybrid search.

        Args:
            project_id: Project to retrieve from
            query: Search query (optional)
            memory_type: Filter by memory type
            limit: Maximum results
            use_hybrid: Use BM25 hybrid search if available

        Returns:
            List of MemoryEntry ordered by relevance
        """
        # Use BM25 hybrid search if available and query provided
        if self._bm25_search and query and use_hybrid:
            return await self._retrieve_with_bm25(project_id, query, limit, memory_type)

        # Fall back to basic keyword search
        results: list[MemoryEntry] = []

        # Search HOT tier
        for entry in self._hot_index.values():
            if entry.project_id != project_id:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            if query and query.lower() not in entry.content.lower():
                continue
            results.append(entry)

        # Search WARM tier
        for entry in self._warm_index.values():
            if entry.project_id != project_id:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            if query and query.lower() not in entry.content.lower():
                continue
            # Check summary if available
            if query and entry.summary and query.lower() not in entry.summary.lower():
                continue
            results.append(entry)

        # Sort by last accessed (most recent first)
        results.sort(key=lambda e: e.last_accessed, reverse=True)

        return results[:limit]

    async def _retrieve_with_bm25(
        self,
        project_id: str,
        query: str,
        limit: int,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve memories using BM25 hybrid search."""
        # Search using BM25
        bm25_results = await self._bm25_search.bm25_search(
            query=query,
            project_id=project_id,
            limit=limit * 2,  # Get more to filter
        )

        # Convert BM25 results to MemoryEntry
        results = []
        for result in bm25_results:
            # Check memory type filter
            if memory_type:
                mt = result.metadata.get("memory_type")
                if mt != memory_type.value:
                    continue

            # Try to find in hot/warm index first
            entry = self._hot_index.get(result.doc_id) or self._warm_index.get(result.doc_id)

            if entry:
                results.append(entry)
            else:
                # Create entry from BM25 result
                entry = MemoryEntry(
                    id=result.doc_id,
                    project_id=result.project_id,
                    content=result.content,
                    memory_type=MemoryType(result.metadata.get("memory_type", "task")),
                    tier=MemoryTier.HOT,  # Assume HOT for search results
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    metadata=result.metadata,
                )
                results.append(entry)

        return results[:limit]

    async def get_recent(
        self,
        project_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get most recent memories."""
        all_entries = list(self._hot_index.values()) + list(self._warm_index.values())

        # Filter by project
        entries = [e for e in all_entries if e.project_id == project_id]

        # Sort by created_at
        entries.sort(key=lambda e: e.created_at, reverse=True)

        return entries[:limit]

    async def update_access(self, memory_id: str) -> None:
        """Update last accessed time for a memory."""
        # Check HOT
        if memory_id in self._hot_index:
            entry = self._hot_index[memory_id]
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self._save_memory(entry)
            return

        # Check WARM
        if memory_id in self._warm_index:
            entry = self._warm_index[memory_id]
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self._save_memory(entry)
            return

        # Check COLD (load if found)
        for file_path in self.tier_paths[MemoryTier.COLD].glob("*.json"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    if data["id"] == memory_id:
                        entry = MemoryEntry.from_dict(data)
                        entry.last_accessed = datetime.utcnow()
                        entry.access_count += 1
                        # Move to WARM if accessed
                        entry.tier = MemoryTier.WARM
                        self._warm_index[memory_id] = entry
                        self._save_memory(entry)
                        return
            except Exception:
                continue

    async def summarize_memory(self, memory_id: str, summary: str) -> None:
        """Add or update summary for a memory."""
        # Check HOT
        if memory_id in self._hot_index:
            entry = self._hot_index[memory_id]
            entry.summary = summary
            self._save_memory(entry)
            return

        # Check WARM
        if memory_id in self._warm_index:
            entry = self._warm_index[memory_id]
            entry.summary = summary
            self._save_memory(entry)
            return

    async def migrate_tiers(self) -> dict[str, int]:
        """
        Migrate memories between tiers based on age.

        Returns migration counts.
        """
        counts = {"hot_to_warm": 0, "warm_to_cold": 0}

        now = datetime.utcnow()

        # Migrate HOT -> WARM
        to_migrate = []
        for entry in self._hot_index.values():
            age = (now - entry.created_at).days
            if age >= self.hot_ttl_days:
                to_migrate.append(entry)

        for entry in to_migrate:
            # Remove from HOT
            del self._hot_index[entry.id]

            # Update tier
            entry.tier = MemoryTier.WARM
            self._warm_index[entry.id] = entry

            # Save
            self._save_memory(entry)

            counts["hot_to_warm"] += 1

        # Migrate WARM -> COLD
        to_migrate = []
        for entry in self._warm_index.values():
            age = (now - entry.created_at).days
            if age >= self.warm_ttl_days:
                to_migrate.append(entry)

        for entry in to_migrate:
            # Remove from WARM
            del self._warm_index[entry.id]

            # Update tier
            entry.tier = MemoryTier.COLD

            # Save to cold path
            self._save_memory(entry)

            counts["warm_to_cold"] += 1

        if sum(counts.values()) > 0:
            logger.info(f"Migrated memories: {counts}")

        return counts

    def get_stats(self, project_id: str | None = None) -> dict[str, Any]:
        """Get memory statistics."""
        hot_count = len(self._hot_index)
        warm_count = len(self._warm_index)

        # Count cold (files on disk)
        cold_count = len(list(self.tier_paths[MemoryTier.COLD].glob("*.json")))

        # Filter by project if specified
        if project_id:
            hot_count = sum(1 for e in self._hot_index.values() if e.project_id == project_id)
            warm_count = sum(1 for e in self._warm_index.values() if e.project_id == project_id)

        stats = {
            "hot_count": hot_count,
            "warm_count": warm_count,
            "cold_count": cold_count,
            "total": hot_count + warm_count + cold_count,
            "hot_ttl_days": self.hot_ttl_days,
            "warm_ttl_days": self.warm_ttl_days,
        }

        # Add BM25 stats if available
        if self._bm25_search:
            stats["bm25"] = self._bm25_search.get_stats()
            stats["hybrid_search_enabled"] = True
        else:
            stats["hybrid_search_enabled"] = False

        return stats

    async def delete_project_memories(self, project_id: str) -> int:
        """Delete all memories for a project."""
        deleted = 0

        # Delete from HOT
        to_delete = [id for id, e in self._hot_index.items() if e.project_id == project_id]
        for memory_id in to_delete:
            self._hot_index.pop(memory_id)
            file_path = self._memory_file_path(MemoryTier.HOT, memory_id)
            if file_path.exists():
                file_path.unlink()
            deleted += 1

        # Delete from WARM
        to_delete = [id for id, e in self._warm_index.items() if e.project_id == project_id]
        for memory_id in to_delete:
            self._warm_index.pop(memory_id)
            file_path = self._memory_file_path(MemoryTier.WARM, memory_id)
            if file_path.exists():
                file_path.unlink()
            deleted += 1

        # Delete from COLD
        for file_path in self.tier_paths[MemoryTier.COLD].glob("*.json"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    if data["project_id"] == project_id:
                        file_path.unlink()
                        deleted += 1
            except Exception:
                continue

        logger.info(f"Deleted {deleted} memories for project {project_id}")
        return deleted


# Global manager instance
_default_manager: MemoryTierManager | None = None


def get_memory_manager() -> MemoryTierManager:
    """Get the default memory tier manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = MemoryTierManager()
    return _default_manager
