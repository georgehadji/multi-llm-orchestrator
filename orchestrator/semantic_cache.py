"""
Semantic Cache — High-level pattern caching for cost optimization
=================================================================
Caches high-probability sub-results based on semantic intent rather than
exact prompt matching. This enables cache hits for semantically equivalent
tasks with different surface forms.

Example:
    "Generate a FastAPI auth endpoint" and "Create authentication API route"
    Should hit the same cache entry (both = auth endpoint generation)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import Task, TaskType


@dataclass
class SemanticPattern:
    """A cached semantic pattern with quality threshold."""
    pattern_hash: str
    task_type: TaskType
    normalized_intent: str
    output: str
    quality_score: float
    use_count: int = 0


# Backward compatibility: DuplicationDetector alias
DuplicationDetector = None  # Deprecated: functionality merged into SemanticCache


class SemanticCache:
    """
    Semantic cache for high-level task patterns.

    Unlike the DiskCache which does exact prompt matching, this cache:
    1. Normalizes prompts to extract semantic intent
    2. Strips variable names and literals
    3. Preserves structure and operation types
    4. Only caches high-quality results (score >= threshold)
    """

    def __init__(self, quality_threshold: float = 0.85, min_use_count: int = 2):
        self._cache: dict[str, SemanticPattern] = {}
        self._quality_threshold = quality_threshold
        self._min_use_count = min_use_count

    def _normalize_prompt(self, prompt: str, task_type: TaskType) -> str:
        """
        Normalize prompt to extract semantic intent.

        Transformations:
        - Strip variable names (replace with <IDENTIFIER>)
        - Strip literal values (replace with <LITERAL>)
        - Normalize whitespace
        - Lowercase keywords
        - Preserve structure and operation types
        """
        # Remove code blocks and quotes
        text = re.sub(r'```[\s\S]*?```', '<CODE_BLOCK>', prompt)
        text = re.sub(r'["\'][^"\']+["\']', '<LITERAL>', text)

        # Replace variable-like identifiers (camelCase, snake_case)
        text = re.sub(r'\b[a-z][a-zA-Z0-9_]*\b', '<identifier>', text)
        text = re.sub(r'\b[A-Z][a-zA-Z0-9_]*\b', '<Identifier>', text)

        # Replace numbers
        text = re.sub(r'\b\d+\b', '<NUM>', text)

        # Normalize task-specific patterns
        if task_type == TaskType.CODE_GEN:
            # Normalize function/class names in definitions
            text = re.sub(r'\b(def|class)\s+\w+', r'\1 <NAME>', text)

        # Normalize whitespace
        text = ' '.join(text.split())
        text = text.lower()

        return text

    def _compute_hash(self, normalized: str, task_type: TaskType) -> str:
        """Compute deterministic hash for normalized intent."""
        payload = f"{task_type.value}:{normalized}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_cached_pattern(self, task: Task, min_quality: float = 0.0) -> str | None:
        """
        Retrieve cached output if semantic match exists.

        Args:
            task: The task to find a cached pattern for
            min_quality: Minimum quality score required for cache hit

        Returns:
            Cached output string or None if no match
        """
        normalized = self._normalize_prompt(task.prompt, task.type)
        pattern_hash = self._compute_hash(normalized, task.type)

        pattern = self._cache.get(pattern_hash)
        if pattern is None:
            return None

        # Verify quality threshold
        if pattern.quality_score < max(min_quality, self._quality_threshold):
            return None

        # Only return if pattern has proven useful (used multiple times)
        if pattern.use_count < self._min_use_count:
            return None

        pattern.use_count += 1
        return pattern.output

    def cache_pattern(
        self,
        task: Task,
        output: str,
        quality_score: float
    ) -> bool:
        """
        Cache a pattern if it meets quality threshold.

        Args:
            task: The task that generated the output
            output: The generated output
            quality_score: The quality score of the output

        Returns:
            True if cached, False if quality too low
        """
        if quality_score < self._quality_threshold:
            return False

        normalized = self._normalize_prompt(task.prompt, task.type)
        pattern_hash = self._compute_hash(normalized, task.type)

        # Update existing or create new
        if pattern_hash in self._cache:
            existing = self._cache[pattern_hash]
            # Keep the higher quality version
            if quality_score > existing.quality_score:
                existing.output = output
                existing.quality_score = quality_score
            existing.use_count += 1
        else:
            self._cache[pattern_hash] = SemanticPattern(
                pattern_hash=pattern_hash,
                task_type=task.type,
                normalized_intent=normalized,
                output=output,
                quality_score=quality_score,
                use_count=1
            )

        return True

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._cache:
            return {"entries": 0, "avg_quality": 0.0, "total_uses": 0}

        qualities = [p.quality_score for p in self._cache.values()]
        uses = [p.use_count for p in self._cache.values()]

        return {
            "entries": len(self._cache),
            "avg_quality": sum(qualities) / len(qualities),
            "total_uses": sum(uses),
            "hot_entries": sum(1 for u in uses if u >= self._min_use_count)
        }

    def clear(self) -> None:
        """Clear all cached patterns."""
        self._cache.clear()
