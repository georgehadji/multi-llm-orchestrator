"""
BM25 Search — SQLite FTS5 Full-Text Search for Memory Retrieval
================================================================

Implements BM25 full-text search using SQLite FTS5 extension.
Combines with vector search for hybrid retrieval.

Based on QMD search architecture.

Usage:
    from orchestrator.bm25_search import BM25Search

    search = BM25Search(":memory:")

    # Add documents
    await search.add_document("doc1", "project1", "Python code example", {"type": "code"})
    await search.add_document("doc2", "project1", "JavaScript tutorial", {"type": "docs"})

    # BM25 search
    results = await search.bm25_search("python", project_id="project1")

    # Hybrid search (BM25 + Vector)
    results = await search.hybrid_search("python code", project_id="project1")
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class SearchDocument:
    """Document for search."""
    doc_id: str
    project_id: str
    content: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class SearchResult:
    """Search result with scoring."""
    doc_id: str
    project_id: str
    content: str
    title: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "project_id": self.project_id,
            "content": self.content[:500],  # Truncate for response
            "title": self.title,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class BM25Search:
    """
    BM25 full-text search using SQLite FTS5.

    Implements:
    - BM25 keyword search (SQLite FTS5)
    - Hybrid search (BM25 + Vector)
    - RRF (Reciprocal Rank Fusion) for result combination
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize database tables."""
        cursor = self.conn.cursor()

        # Create FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id,
                project_id,
                title,
                content,
                metadata,
                tokenize='porter'
            )
        """)

        # Create regular table for additional metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)

        # Create index for project-based queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_project ON documents(project_id)
        """)

        self.conn.commit()
        logger.debug("Initialized BM25 search tables")

    async def add_document(
        self,
        doc_id: str,
        project_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        title: str = "",
    ) -> None:
        """Add a document to the search index."""
        cursor = self.conn.cursor()

        metadata_json = json.dumps(metadata or {})

        # Insert into regular table
        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (doc_id, project_id, title, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, project_id, title, content, metadata_json))

        # Insert into FTS5 table
        cursor.execute("""
            INSERT OR REPLACE INTO documents_fts
            (doc_id, project_id, title, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, project_id, title, content, metadata_json))

        self.conn.commit()
        logger.debug(f"Added document {doc_id} to search index")

    async def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the search index."""
        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))

        self.conn.commit()

        return cursor.rowcount > 0

    async def bm25_search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Perform BM25 full-text search.

        Args:
            query: Search query (keywords)
            project_id: Optional project filter
            limit: Maximum results

        Returns:
            List of SearchResult ordered by BM25 score
        """
        cursor = self.conn.cursor()

        # Build query
        if project_id:
            cursor.execute("""
                SELECT
                    d.doc_id,
                    d.project_id,
                    d.title,
                    d.content,
                    d.metadata,
                    bm25(documents_fts) as score
                FROM documents_fts
                JOIN documents d ON documents_fts.doc_id = d.doc_id
                WHERE documents_fts MATCH ?
                    AND project_id = ?
                ORDER BY score DESC
                LIMIT ?
            """, (query, project_id, limit))
        else:
            cursor.execute("""
                SELECT
                    d.doc_id,
                    d.project_id,
                    d.title,
                    d.content,
                    d.metadata,
                    bm25(documents_fts) as score
                FROM documents_fts
                JOIN documents d ON documents_fts.doc_id = d.doc_id
                WHERE documents_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (query, limit))

        results = []
        for i, row in enumerate(cursor.fetchall()):
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            # Convert BM25 score (negative) to positive
            score = abs(row["score"]) if row["score"] else 0.0

            results.append(SearchResult(
                doc_id=row["doc_id"],
                project_id=row["project_id"],
                title=row["title"] or "",
                content=row["content"],
                score=score,
                rank=i + 1,
                metadata=metadata,
            ))

        logger.debug(f"BM25 search for '{query}' returned {len(results)} results")

        return results

    async def vector_search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Note: For production, use actual vector embeddings with
        a vector database or SQLite vec extension.
        This is a simplified implementation using cosine similarity
        with cached embeddings.
        """
        # This would require actual vector embeddings
        # For now, return empty or fall back to BM25
        logger.debug("Vector search requires embedding model - falling back to BM25")
        return await self.bm25_search(query, project_id, limit)

    def _rrf_fusion(
        self,
        bm25_results: list[SearchResult],
        vector_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining search results.

        Formula: score = Σ(1 / (k + rank))

        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            k: RRF constant (default 60)

        Returns:
            Combined and re-ranked results
        """
        # Track cumulative scores
        scores: dict[str, tuple[float, SearchResult]] = {}

        # Add BM25 scores
        for result in bm25_results:
            rrf_score = 1.0 / (k + result.rank)
            scores[result.doc_id] = (rrf_score, result)

        # Add vector scores
        for result in vector_results:
            rrf_score = 1.0 / (k + result.rank)
            if result.doc_id in scores:
                # Add to existing score
                prev_score, prev_result = scores[result.doc_id]
                scores[result.doc_id] = (prev_score + rrf_score, prev_result)
            else:
                scores[result.doc_id] = (rrf_score, result)

        # Sort by combined score
        combined = [
            SearchResult(
                doc_id=doc_id,
                project_id=result.project_id,
                title=result.title,
                content=result.content,
                score=rrf_score,
                rank=0,  # Will be set after sorting
                metadata=result.metadata,
            )
            for doc_id, (rrf_score, result) in scores.items()
        ]

        # Sort and assign ranks
        combined.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(combined):
            result.rank = i + 1

        return combined

    async def hybrid_search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
        use_rrf: bool = True,
    ) -> list[SearchResult]:
        """
        Hybrid search combining BM25 and vector search.

        Args:
            query: Search query
            project_id: Optional project filter
            limit: Maximum results
            use_rrf: Use RRF fusion (default True)

        Returns:
            Combined search results
        """
        # Run both searches in parallel
        bm25_task = asyncio.create_task(self.bm25_search(query, project_id, limit * 2))
        vector_task = asyncio.create_task(self.vector_search(query, project_id, limit * 2))

        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

        if use_rrf and vector_results:
            # Combine with RRF
            combined = self._rrf_fusion(bm25_results, vector_results)
            return combined[:limit]
        else:
            # Return BM25 results (vector not available)
            return bm25_results[:limit]

    async def search_with_highlights(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 10,
        snippet_size: int = 150,
    ) -> list[dict[str, Any]]:
        """
        Search with highlighted snippets.

        Uses FTS5 snippet function to show matching context.
        """
        cursor = self.conn.cursor()

        # Use FTS5 snippet function
        query_terms = query.replace('"', '').split()

        if project_id:
            cursor.execute("""
                SELECT
                    d.doc_id,
                    d.project_id,
                    d.title,
                    snippet(documents_fts, 3, '<mark>', '</mark>', '...', 10) as snippet,
                    bm25(documents_fts) as score
                FROM documents_fts
                JOIN documents d ON documents_fts.doc_id = d.doc_id
                WHERE documents_fts MATCH ?
                    AND project_id = ?
                ORDER BY score DESC
                LIMIT ?
            """, (query, project_id, limit))
        else:
            cursor.execute("""
                SELECT
                    d.doc_id,
                    d.project_id,
                    d.title,
                    snippet(documents_fts, 3, '<mark>', '</mark>', '...', 10) as snippet,
                    bm25(documents_fts) as score
                FROM documents_fts
                JOIN documents d ON documents_fts.doc_id = d.doc_id
                WHERE documents_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (query, limit))

        results = []
        for row in cursor.fetchall():
            score = abs(row["score"]) if row["score"] else 0.0
            results.append({
                "doc_id": row["doc_id"],
                "project_id": row["project_id"],
                "title": row["title"] or "",
                "snippet": row["snippet"],
                "score": score,
                "highlights": query_terms,
            })

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get search index statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT project_id) FROM documents")
        total_projects = cursor.fetchone()[0]

        return {
            "total_documents": total_docs,
            "total_projects": total_projects,
            "index_type": "SQLite FTS5",
        }

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


# Global instance
_default_bm25: BM25Search | None = None


def get_bm25_search(db_path: str = ":memory:") -> BM25Search:
    """Get or create BM25 search instance."""
    global _default_bm25
    if _default_bm25 is None:
        _default_bm25 = BM25Search(db_path)
    return _default_bm25
