"""
ContextSources — Multiple context sources
=======================================
Module for managing multiple context sources like documents, databases, APIs, etc.

Pattern: Strategy
Async: Yes — for I/O-bound operations
Layer: L2 Verification

Usage:
    from orchestrator.context_sources import ContextSourceManager
    source_manager = ContextSourceManager()
    doc_source = source_manager.add_document_source("docs", "./documents/")
    context = await source_manager.get_context(query="What is AI?", sources=["docs"])
"""
from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.context_sources")


@dataclass
class ContextChunk:
    """Represents a chunk of context with metadata."""

    id: str
    content: str
    source_id: str
    source_type: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None  # For vector search


class BaseContextSource(ABC):
    """Base class for all context sources."""

    def __init__(self, source_id: str, source_type: str, config: dict[str, Any]):
        self.source_id = source_id
        self.source_type = source_type
        self.config = config
        self.chunks: list[ContextChunk] = []
        self.loaded = False

    @abstractmethod
    async def load(self):
        """Load context from the source."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[ContextChunk]:
        """Search for relevant context chunks."""
        pass

    @abstractmethod
    async def update(self, new_content: str, metadata: dict[str, Any] = None):
        """Update the context source with new content."""
        pass


class DocumentSource(BaseContextSource):
    """Context source for document files."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        super().__init__(source_id, "document", config)
        self.document_dir = Path(config["path"])
        self.supported_formats = config.get("formats", [".txt", ".md", ".pdf", ".docx"])

    async def load(self):
        """Load documents from the specified directory."""
        if not self.document_dir.exists():
            logger.warning(f"Document directory does not exist: {self.document_dir}")
            return

        for file_path in self.document_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    content = await self._read_file(file_path)
                    chunk_id = hashlib.sha256(f"{self.source_id}:{file_path}".encode()).hexdigest()[:16]

                    chunk = ContextChunk(
                        id=chunk_id,
                        content=content,
                        source_id=self.source_id,
                        source_type=self.source_type,
                        metadata={
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "file_size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime
                        }
                    )

                    self.chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Failed to load document {file_path}: {e}")

        self.loaded = True
        logger.info(f"Loaded {len(self.chunks)} document chunks from {self.source_id}")

    async def _read_file(self, file_path: Path) -> str:
        """Read content from a file."""
        if file_path.suffix.lower() == ".pdf":
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        elif file_path.suffix.lower() in [".docx"]:
            from docx import Document
            doc = Document(str(file_path))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            # For text-based formats
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return f.read()

    async def search(self, query: str, limit: int = 10) -> list[ContextChunk]:
        """Search for relevant context chunks in documents."""
        if not self.loaded:
            await self.load()

        # Simple keyword search for now
        # In a more advanced implementation, we would use embeddings or full-text search
        results = []
        query_lower = query.lower()

        for chunk in self.chunks:
            if query_lower in chunk.content.lower():
                results.append(chunk)

        # Sort by relevance (for now, just by content length as a proxy)
        results.sort(key=lambda c: len(c.content), reverse=True)
        return results[:limit]

    async def update(self, new_content: str, metadata: dict[str, Any] = None):
        """Add new content to the document source."""
        chunk_id = hashlib.sha256(f"{self.source_id}:{new_content[:50]}".encode()).hexdigest()[:16]

        chunk = ContextChunk(
            id=chunk_id,
            content=new_content,
            source_id=self.source_id,
            source_type=self.source_type,
            metadata=metadata or {}
        )

        self.chunks.append(chunk)
        logger.info(f"Added new content to document source {self.source_id}")


class DatabaseSource(BaseContextSource):
    """Context source for database queries."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        super().__init__(source_id, "database", config)
        self.db_config = config
        self.connection = None

    async def load(self):
        """Load context from database."""
        # Connect to database based on configuration
        # This is a simplified implementation - in reality, we'd use the specific DB connector
        logger.info(f"Connecting to database source: {self.source_id}")

        # For now, we'll just mark as loaded
        # In a real implementation, we would connect to the database
        # and possibly preload some context
        self.loaded = True

    async def search(self, query: str, limit: int = 10) -> list[ContextChunk]:
        """Search database for relevant context."""
        if not self.loaded:
            await self.load()

        # This is a simplified implementation
        # In a real implementation, we would execute the query against the database
        # and return the results as context chunks
        logger.info(f"Searching database source {self.source_id} with query: {query}")

        # For demonstration, return a mock result
        chunk_id = hashlib.sha256(f"{self.source_id}:{query}".encode()).hexdigest()[:16]
        chunk = ContextChunk(
            id=chunk_id,
            content=f"Mock database result for query: {query}",
            source_id=self.source_id,
            source_type=self.source_type,
            metadata={"query": query, "result_count": 1}
        )

        return [chunk]

    async def update(self, new_content: str, metadata: dict[str, Any] = None):
        """Update database with new content."""
        # In a real implementation, we would insert/update data in the database
        logger.info(f"Updating database source {self.source_id} with new content")


class APISource(BaseContextSource):
    """Context source for API endpoints."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        super().__init__(source_id, "api", config)
        self.api_config = config
        self.session = None

    async def load(self):
        """Initialize API session."""
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.loaded = True
        logger.info(f"Initialized API source: {self.source_id}")

    async def search(self, query: str, limit: int = 10) -> list[ContextChunk]:
        """Search API for relevant context."""
        if not self.loaded:
            await self.load()

        # Construct API request based on configuration
        api_endpoint = self.api_config["endpoint"]
        headers = self.api_config.get("headers", {})
        params = {self.api_config.get("query_param", "q"): query}

        try:
            async with self.session.get(api_endpoint, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Convert API response to context chunks
                    chunks = []
                    for item in data if isinstance(data, list) else [data]:
                        content = json.dumps(item) if isinstance(item, (dict, list)) else str(item)
                        chunk_id = hashlib.sha256(f"{self.source_id}:{content[:50]}".encode()).hexdigest()[:16]

                        chunk = ContextChunk(
                            id=chunk_id,
                            content=content,
                            source_id=self.source_id,
                            source_type=self.source_type,
                            metadata={"api_response": True}
                        )
                        chunks.append(chunk)

                    return chunks[:limit]
                else:
                    logger.error(f"API request failed with status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return []

    async def update(self, new_content: str, metadata: dict[str, Any] = None):
        """Update API with new content."""
        # In a real implementation, we would make a POST/PUT request to the API
        logger.info(f"Updating API source {self.source_id} with new content")

    async def close(self):
        """Close the API session."""
        if self.session:
            await self.session.close()


class MemorySource(BaseContextSource):
    """Context source for in-memory data."""

    def __init__(self, source_id: str, config: dict[str, Any] = None):
        super().__init__(source_id, "memory", config or {})
        self.data: dict[str, str] = {}

    async def load(self):
        """Load context from memory."""
        self.loaded = True
        logger.info(f"Loaded memory source: {self.source_id}")

    async def search(self, query: str, limit: int = 10) -> list[ContextChunk]:
        """Search in-memory data for relevant context."""
        if not self.loaded:
            await self.load()

        results = []
        query_lower = query.lower()

        for key, value in self.data.items():
            if query_lower in key.lower() or query_lower in value.lower():
                chunk_id = hashlib.sha256(f"{self.source_id}:{key}".encode()).hexdigest()[:16]

                chunk = ContextChunk(
                    id=chunk_id,
                    content=value,
                    source_id=self.source_id,
                    source_type=self.source_type,
                    metadata={"key": key}
                )
                results.append(chunk)

        return results[:limit]

    async def update(self, new_content: str, metadata: dict[str, Any] = None):
        """Update in-memory data with new content."""
        key = metadata.get("key") if metadata else hashlib.sha256(new_content.encode()).hexdigest()[:16]
        self.data[key] = new_content
        logger.info(f"Updated memory source {self.source_id} with key: {key}")


class ContextSourceManager:
    """Manages multiple context sources."""

    def __init__(self):
        """Initialize the context source manager."""
        self.sources: dict[str, BaseContextSource] = {}
        self.default_sources: list[str] = []

    def add_document_source(self, source_id: str, path: str,
                           formats: list[str] = None) -> DocumentSource:
        """
        Add a document source.

        Args:
            source_id: Unique identifier for the source
            path: Path to the document directory
            formats: Supported document formats

        Returns:
            DocumentSource instance
        """
        config = {
            "path": path,
            "formats": formats or [".txt", ".md", ".pdf", ".docx"]
        }

        source = DocumentSource(source_id, config)
        self.sources[source_id] = source
        logger.info(f"Added document source: {source_id}")
        return source

    def add_database_source(self, source_id: str, config: dict[str, Any]) -> DatabaseSource:
        """
        Add a database source.

        Args:
            source_id: Unique identifier for the source
            config: Database configuration

        Returns:
            DatabaseSource instance
        """
        source = DatabaseSource(source_id, config)
        self.sources[source_id] = source
        logger.info(f"Added database source: {source_id}")
        return source

    def add_api_source(self, source_id: str, config: dict[str, Any]) -> APISource:
        """
        Add an API source.

        Args:
            source_id: Unique identifier for the source
            config: API configuration

        Returns:
            APISource instance
        """
        source = APISource(source_id, config)
        self.sources[source_id] = source
        logger.info(f"Added API source: {source_id}")
        return source

    def add_memory_source(self, source_id: str, config: dict[str, Any] = None) -> MemorySource:
        """
        Add a memory source.

        Args:
            source_id: Unique identifier for the source
            config: Optional configuration

        Returns:
            MemorySource instance
        """
        source = MemorySource(source_id, config)
        self.sources[source_id] = source
        logger.info(f"Added memory source: {source_id}")
        return source

    async def load_all_sources(self):
        """Load all registered context sources."""
        for source_id, source in self.sources.items():
            try:
                await source.load()
            except Exception as e:
                logger.error(f"Failed to load source {source_id}: {e}")

    async def get_context(self, query: str, sources: list[str] = None,
                         limit: int = 10) -> list[ContextChunk]:
        """
        Get context from specified sources.

        Args:
            query: Query to search for
            sources: List of source IDs to search (if None, use default sources)
            limit: Maximum number of results per source

        Returns:
            List of context chunks
        """
        if sources is None:
            sources = self.default_sources

        if not sources:
            sources = list(self.sources.keys())

        all_results = []

        for source_id in sources:
            if source_id in self.sources:
                try:
                    source = self.sources[source_id]
                    results = await source.search(query, limit)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Failed to search source {source_id}: {e}")

        # Sort results by source type priority and return
        # For now, just return all results
        return all_results

    async def update_source(self, source_id: str, new_content: str,
                           metadata: dict[str, Any] = None):
        """
        Update a specific context source with new content.

        Args:
            source_id: ID of the source to update
            new_content: New content to add
            metadata: Optional metadata for the content
        """
        if source_id in self.sources:
            await self.sources[source_id].update(new_content, metadata)
        else:
            logger.error(f"Source {source_id} not found")

    def set_default_sources(self, source_ids: list[str]):
        """Set the default sources to use when none are specified."""
        self.default_sources = source_ids
        logger.info(f"Set default sources: {source_ids}")

    async def get_source_stats(self) -> dict[str, Any]:
        """
        Get statistics about all context sources.

        Returns:
            Dict with source statistics
        """
        stats = {
            "total_sources": len(self.sources),
            "source_types": {},
            "total_chunks": 0
        }

        for _source_id, source in self.sources.items():
            source_type = type(source).__name__
            stats["source_types"][source_type] = stats["source_types"].get(source_type, 0) + 1
            if hasattr(source, 'chunks'):
                stats["total_chunks"] += len(source.chunks)

        return stats

    async def close(self):
        """Close resources used by context sources."""
        for source in self.sources.values():
            if isinstance(source, APISource):
                await source.close()
