"""KnowledgeBase — Cross-project knowledge store."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

@dataclass
class KnowledgeEntry:
    id: str
    category: str
    content: str
    tags: list[str] = field(default_factory=list)
    date: str = ""
    project: str = ""
    source_agent: str = ""

class KnowledgeBase:
    def __init__(self, path=None):
        self.entries: list[KnowledgeEntry] = []
        self._path = path
        if path and path.exists():
            self._load()
    def add(self, entry):
        if not entry.date:
            entry.date = datetime.now().isoformat()
        self.entries.append(entry)
    def search(self, query, tags=None):
        results = []
        q = query.lower()
        for e in self.entries:
            if q in e.content.lower() or q in e.category.lower():
                if not tags or any(t in e.tags for t in tags):
                    results.append(e)
        return results
    def _save(self):
        if self._path:
            data = [{"id": e.id, "category": e.category, "content": e.content} for e in self.entries]
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    def _load(self):
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for item in data:
                self.entries.append(KnowledgeEntry(**item))
        except:
            pass
