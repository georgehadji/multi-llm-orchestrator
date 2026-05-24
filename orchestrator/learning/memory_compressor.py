"""Summarizes many similar patterns into lessons."""
from __future__ import annotations
from typing import Any

class MemoryCompressor:
    THRESHOLD = 10

    def compress(self, buffer: Any) -> list[str]:
        lessons = []
        if not hasattr(buffer, 'successes'):
            return lessons

        from collections import defaultdict
        by_type = defaultdict(list)
        for p in buffer.successes:
            by_type[p.task_type].append(p)

        for task_type, patterns in by_type.items():
            if len(patterns) >= self.THRESHOLD:
                avg = sum(p.score for p in patterns) / len(patterns)
                methods = {}
                for p in patterns:
                    methods[p.method] = methods.get(p.method, 0) + 1
                best_method = max(methods, key=methods.get) if methods else "unknown"
                lessons.append(
                    f"[Lesson] {task_type}: {best_method} method achieves "
                    f"{avg:.2f} avg score over {len(patterns)} runs"
                )
        return lessons
