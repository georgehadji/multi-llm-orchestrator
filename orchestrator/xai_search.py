"""
X Search Integration — Real-time X/Twitter Data
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Provides access to X (Twitter) search via xAI Grok's X Search tool.
Unique capability: Real-time access to X posts, trends, and breaking news.

Usage:
    from orchestrator.xai_search import XSearchClient

    client = XSearchClient(api_key="xai-...")
    posts = await client.search_posts("AI trends 2026", count=10)
    trends = await client.get_trends(location="US")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from .log_config import get_logger

logger = get_logger(__name__)


@dataclass
class XPost:
    """A single X post."""
    id: str
    text: str
    author: str
    author_handle: str
    created_at: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    url: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "author_handle": self.author_handle,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "url": self.url,
            "verified": self.verified,
        }


@dataclass
class XSearchResult:
    """Result from X search."""
    query: str
    posts: list[XPost] = field(default_factory=list)
    total_count: int = 0
    search_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "posts": [p.to_dict() for p in self.posts],
            "total_count": self.total_count,
            "search_metadata": self.search_metadata,
        }


@dataclass
class TrendTopic:
    """A trending topic."""
    name: str
    tweet_volume: int = 0
    url: str = ""
    promoted: bool = False


@dataclass
class TrendsResult:
    """Result from trends query."""
    location: str
    trends: list[TrendTopic] = field(default_factory=list)
    as_of: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "trends": [
                {"name": t.name, "tweet_volume": t.tweet_volume, "url": t.url}
                for t in self.trends
            ],
            "as_of": self.as_of.isoformat(),
        }


class XSearchClient:
    """
    Client for X Search API via xAI.

    Features:
    - Search X posts in real-time
    - Get trending topics by location
    - Filter by sort (latest/top)
    - Cost tracking ($5 per 1k calls)

    Usage:
        client = XSearchClient(api_key="xai-...")
        posts = await client.search_posts("AI trends", count=10)
    """

    BASE_URL = "https://api.x.ai/v1"
    SEARCH_COST_PER_CALL = 0.005  # $5 per 1000 calls

    def __init__(self, api_key: str | None = None):
        """
        Initialize X Search client.

        Args:
            api_key: xAI API key (or set XAI_API_KEY env var)
        """
        import os
        self.api_key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")

        if not self.api_key:
            logger.warning("X Search: No API key provided. Set XAI_API_KEY or pass api_key parameter.")

        self._client: httpx.AsyncClient | None = None
        self.total_calls = 0
        self.total_cost = 0.0

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_posts(
        self,
        query: str,
        count: int = 10,
        sort: str = "latest",  # or "top"
    ) -> XSearchResult:
        """
        Search X posts.

        Args:
            query: Search query
            count: Number of posts (max 100)
            sort: Sort order ("latest" or "top")

        Returns:
            XSearchResult with matching posts

        Cost: $0.005 per call ($5 per 1000 calls)
        """
        await self._ensure_client()

        try:
            response = await self._client.post(
                "/search",
                json={
                    "query": query,
                    "count": min(count, 100),  # API limit
                    "sort": sort,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Track usage
            self.total_calls += 1
            self.total_cost += self.SEARCH_COST_PER_CALL

            # Parse results
            posts = []
            for post_data in data.get("results", []):
                post = XPost(
                    id=post_data.get("id", ""),
                    text=post_data.get("text", ""),
                    author=post_data.get("author", {}).get("name", ""),
                    author_handle=post_data.get("author", {}).get("username", ""),
                    created_at=datetime.fromisoformat(post_data["created_at"].replace("Z", "+00:00")) if post_data.get("created_at") else None,
                    likes=post_data.get("likes", 0),
                    retweets=post_data.get("retweets", 0),
                    replies=post_data.get("replies", 0),
                    url=post_data.get("url", ""),
                    verified=post_data.get("author", {}).get("verified", False),
                )
                posts.append(post)

            return XSearchResult(
                query=query,
                posts=posts,
                total_count=len(posts),
                search_metadata=data.get("metadata", {}),
            )

        except httpx.HTTPError as e:
            logger.error(f"X Search failed: {e}")
            return XSearchResult(query=query, posts=[])
        except Exception as e:
            logger.error(f"X Search error: {e}")
            return XSearchResult(query=query, posts=[])

    async def get_trends(
        self,
        location: str = "US",
    ) -> TrendsResult:
        """
        Get trending topics.

        Args:
            location: Location code (US, GB, etc.)

        Returns:
            TrendsResult with trending topics
        """
        await self._ensure_client()

        try:
            response = await self._client.post(
                "/trends",
                json={"location": location},
            )
            response.raise_for_status()
            data = response.json()

            self.total_calls += 1
            self.total_cost += self.SEARCH_COST_PER_CALL

            trends = []
            for trend_data in data.get("trends", []):
                trend = TrendTopic(
                    name=trend_data.get("name", ""),
                    tweet_volume=trend_data.get("tweet_volume", 0),
                    url=trend_data.get("url", ""),
                    promoted=trend_data.get("promoted", False),
                )
                trends.append(trend)

            return TrendsResult(
                location=location,
                trends=trends,
                as_of=datetime.now(),
            )

        except httpx.HTTPError as e:
            logger.error(f"X Trends failed: {e}")
            return TrendsResult(location=location, trends=[])
        except Exception as e:
            logger.error(f"X Trends error: {e}")
            return TrendsResult(location=location, trends=[])

    async def search_with_filters(
        self,
        query: str,
        min_likes: int = 0,
        min_retweets: int = 0,
        verified_only: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        count: int = 10,
    ) -> XSearchResult:
        """
        Search X posts with advanced filters.

        Args:
            query: Base search query
            min_likes: Minimum likes
            min_retweets: Minimum retweets
            verified_only: Only verified authors
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            count: Number of results

        Returns:
            Filtered XSearchResult
        """
        result = await self.search_posts(query, count=count * 3)  # Get more to filter

        # Apply filters
        filtered_posts = []
        for post in result.posts:
            if post.likes < min_likes:
                continue
            if post.retweets < min_retweets:
                continue
            if verified_only and not post.verified:
                continue
            if date_from and post.created_at:
                from_date = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                if post.created_at < from_date:
                    continue
            if date_to and post.created_at:
                to_date = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                if post.created_at > to_date:
                    continue

            filtered_posts.append(post)

            if len(filtered_posts) >= count:
                break

        return XSearchResult(
            query=query,
            posts=filtered_posts,
            total_count=len(filtered_posts),
            search_metadata={**result.search_metadata, "filtered": True},
        )

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": self.total_cost,
            "cost_per_call": self.SEARCH_COST_PER_CALL,
        }


async def search_x(
    query: str,
    count: int = 10,
    sort: str = "latest",
    api_key: str | None = None,
) -> XSearchResult:
    """
    Convenience function to search X.

    Usage:
        results = await search_x("AI trends 2026", count=10)
        for post in results.posts:
            print(f"{post.author}: {post.text}")
    """
    async with XSearchClient(api_key) as client:
        return await client.search_posts(query, count, sort)


async def get_x_trends(
    location: str = "US",
    api_key: str | None = None,
) -> TrendsResult:
    """
    Convenience function to get X trends.

    Usage:
        trends = await get_x_trends("US")
        for trend in trends.trends:
            print(f"{trend.name}: {trend.tweet_volume} tweets")
    """
    async with XSearchClient(api_key) as client:
        return await client.get_trends(location)


def format_x_results(results: XSearchResult, max_posts: int = 5) -> str:
    """
    Format X search results as text summary.

    Usage:
        summary = format_x_results(results)
        print(summary)
    """
    if not results.posts:
        return f"No posts found for: {results.query}"

    lines = [f"X Search Results for: {results.query}", "=" * 50]

    for i, post in enumerate(results.posts[:max_posts], 1):
        verified = "✓" if post.verified else " "
        lines.append(f"\n{i}. [{verified}] {post.author} (@{post.author_handle})")
        lines.append(f"   {post.text[:200]}...")
        lines.append(f"   👍 {post.likes} | 🔁 {post.retweets} | 💬 {post.replies}")
        if post.url:
            lines.append(f"   🔗 {post.url}")

    if results.total_count > max_posts:
        lines.append(f"\n... and {results.total_count - max_posts} more posts")

    return "\n".join(lines)
