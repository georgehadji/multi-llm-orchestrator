"""ProductBacklog — Requirements backlog."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UserStory:
    id: str
    title: str
    description: str
    priority: str = "P2"
    module: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = "backlog"


class ProductBacklog:
    def __init__(self, path=None):
        self.stories: list[UserStory] = []
        self._path = path

    def add(self, story):
        self.stories.append(story)

    def by_priority(self):
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return sorted(self.stories, key=lambda s: order.get(s.priority, 99))

    def completion_status(self):
        result = {}
        for s in self.stories:
            k = s.priority
            done = sum(1 for x in self.stories if x.priority == k and x.status == "done")
            total = sum(1 for x in self.stories if x.priority == k)
            result[k] = f"{done}/{total}"
        return result
