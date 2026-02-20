"""
Semantic output cache and duplication detector.

SemanticCache stores (task_type, prompt_tokens, output) entries and
retrieves outputs when a new prompt is sufficiently similar (Jaccard
similarity on word tokens — no embedding model required).

DuplicationDetector scans a task list for near-duplicate prompts so the
engine can reuse outputs instead of generating twice.
"""
from __future__ import annotations
import re
from typing import Optional


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Standard Jaccard similarity on token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _soft_jaccard(a: set[str], b: set[str]) -> float:
    """
    Jaccard with prefix-match softening: a token in 'a' is counted as
    matching a token in 'b' when one is a prefix of the other (minimum
    prefix length 3). This handles morphological variants such as
    'app' / 'application' or 'auth' / 'authentication'.

    Soft-matched token pairs contribute 0.5 to the intersection instead
    of a full 1.0, so identical prompts still score 1.0 and truly
    unrelated prompts stay near 0.
    """
    if not a and not b:
        return 1.0
    union_size = len(a | b)
    if union_size == 0:
        return 0.0

    exact_inter = a & b
    # find tokens that match via prefix but are not exact matches
    soft_matched_a: set[str] = set()
    soft_matched_b: set[str] = set()
    for ta in a - exact_inter:
        for tb in b - exact_inter:
            min_len = min(len(ta), len(tb))
            if min_len >= 3 and (ta.startswith(tb[:min_len]) or tb.startswith(ta[:min_len])):
                soft_matched_a.add(ta)
                soft_matched_b.add(tb)

    # exact matches contribute 1.0 each; soft matches contribute 0.5 each
    soft_inter = len(exact_inter) + 0.5 * len(soft_matched_a | soft_matched_b)
    return soft_inter / union_size


class SemanticCache:
    """
    In-memory semantic output cache keyed by (task_type, prompt_tokens).
    """

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        self.threshold = similarity_threshold
        # entries: list of (task_type, tokens, output)
        self._entries: list[tuple[str, set[str], str]] = []

    async def get(
        self, task_id: str, task_type: str, prompt: str
    ) -> Optional[str]:
        tokens = _tokenize(prompt)
        best_score = 0.0
        best_output: Optional[str] = None
        for (etype, etokens, eoutput) in self._entries:
            if etype != task_type:
                continue
            score = _jaccard(tokens, etokens)
            if score > best_score:
                best_score = score
                best_output = eoutput
        if best_score >= self.threshold:
            return best_output
        return None

    async def put(
        self,
        task_id: str,
        task_type: str,
        output: str,
        prompt: str = "",
    ) -> None:
        tokens = _tokenize(prompt)
        self._entries.append((task_type, tokens, output))

    def size(self) -> int:
        return len(self._entries)


class DuplicationDetector:
    """
    Scans a dict of {task_id: {prompt: str}} and returns groups of
    near-duplicate task IDs based on soft Jaccard similarity of prompts.
    Soft Jaccard handles morphological variants (e.g. 'app'/'application')
    via prefix matching so that near-paraphrases are correctly grouped.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.threshold = similarity_threshold

    def find_duplicate_groups(
        self, tasks: dict[str, dict]
    ) -> list[list[str]]:
        ids = list(tasks.keys())
        token_map = {
            tid: _tokenize(tasks[tid].get("prompt", ""))
            for tid in ids
        }
        visited: set[str] = set()
        groups: list[list[str]] = []
        for i, tid_a in enumerate(ids):
            if tid_a in visited:
                continue
            group = [tid_a]
            for tid_b in ids[i + 1:]:
                if tid_b in visited:
                    continue
                score = _soft_jaccard(token_map[tid_a], token_map[tid_b])
                if score >= self.threshold:
                    group.append(tid_b)
                    visited.add(tid_b)
            if len(group) > 1:
                groups.append(group)
                visited.add(tid_a)
        return groups
