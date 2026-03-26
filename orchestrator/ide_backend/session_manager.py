"""
Session Manager - Manages IDE session state
"""
from __future__ import annotations

import asyncio
import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from .log_config import get_logger

logger = get_logger(__name__)


class SessionMode(str, Enum):
    """Session operation modes."""
    BUILD = "build"
    PLAN = "plan"
    CHAT = "chat"
    DEBUG = "debug"


class AutonomyLevel(str, Enum):
    """Autonomy levels for code generation."""
    LITE = "lite"
    STANDARD = "standard"
    AUTONOMOUS = "autonomous"
    MAX = "max"


@dataclass
class FileNode:
    """File tree node."""
    name: str
    type: str  # "file" or "folder"
    children: List["FileNode"] = field(default_factory=list)
    content: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "type": self.type}
        if self.type == "folder" and self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.type == "file":
            result["language"] = self.language or "text"
        return result


@dataclass
class ChatMessage:
    """Chat message in session."""
    role: str  # "user" or "assistant"
    content: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    thinking: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    cost: Optional[float] = None
    quality: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskProgress:
    """Task progress tracking."""
    task_id: str
    name: str
    status: str = "pending"  # pending, running, completed, failed
    model: Optional[str] = None
    score: Optional[float] = None
    cost: float = 0.0
    elapsed: float = 0.0
    repairs: int = 0


@dataclass
class SessionState:
    """Complete session state."""
    id: str
    project_name: str = "Untitled Project"
    description: str = ""
    mode: SessionMode = SessionMode.BUILD
    autonomy: AutonomyLevel = AutonomyLevel.STANDARD
    model: str = "auto"
    budget: float = 5.0
    spent: float = 0.0
    created_at: float = field(default_factory=datetime.now().timestamp)
    started_at: Optional[float] = None
    messages: List[ChatMessage] = field(default_factory=list)
    files: List[FileNode] = field(default_factory=list)
    tasks: List[TaskProgress] = field(default_factory=list)
    terminal_lines: List[Dict[str, str]] = field(default_factory=list)
    quality_score: float = 0.0
    cache_hit_rate: float = 0.0
    status: str = "idle"  # idle, running, paused, completed, error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        elapsed = (datetime.now().timestamp() - self.started_at) if self.started_at else 0
        completed_tasks = sum(1 for t in self.tasks if t.status == "completed")
        total_tasks = len(self.tasks) if self.tasks else 1
        
        return {
            "id": self.id,
            "project": {
                "name": self.project_name,
                "description": self.description,
            },
            "settings": {
                "mode": self.mode.value,
                "autonomy": self.autonomy.value,
                "model": self.model,
            },
            "budget": {
                "total": round(self.budget, 2),
                "spent": round(self.spent, 2),
                "remaining": round(self.budget - self.spent, 2),
                "percent": round(self.spent / self.budget * 100, 1) if self.budget > 0 else 0,
            },
            "progress": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "percent": round(completed_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0,
            },
            "time": {
                "elapsed": self._format_time(elapsed),
                "elapsed_seconds": elapsed,
            },
            "metrics": {
                "quality_score": round(self.quality_score, 2),
                "cache_hit_rate": round(self.cache_hit_rate, 2),
            },
            "status": self.status,
            "messages": [m.to_dict() for m in self.messages[-50:]],  # Last 50 messages
            "files": [f.to_dict() for f in self.files],
            "tasks": [asdict(t) for t in self.tasks],
            "terminal_lines": self.terminal_lines[-100:],  # Last 100 lines
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"


class SessionManager:
    """Manages all IDE sessions."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self._sessions: Dict[str, SessionState] = {}
        self._storage_path = storage_path or Path.home() / ".orchestrator" / "ide_sessions"
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"SessionManager initialized, storage: {self._storage_path}")
    
    async def create_session(
        self,
        project_name: str = "Untitled Project",
        description: str = "",
        mode: str = "build",
        autonomy: str = "standard",
        budget: float = 5.0,
        model: str = "auto",
    ) -> SessionState:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        session = SessionState(
            id=session_id,
            project_name=project_name,
            description=description,
            mode=SessionMode(mode.lower()),
            autonomy=AutonomyLevel(autonomy.lower()),
            budget=budget,
            model=model,
            started_at=datetime.now().timestamp(),
        )
        
        async with self._lock:
            self._sessions[session_id] = session
        
        # Persist session
        await self._save_session(session)
        
        logger.info(f"Session created: {session_id}, project={project_name}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)
    
    async def update_session(self, session_id: str, **updates) -> Optional[SessionState]:
        """Update session fields."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            await self._save_session(session)
            return session
    
    async def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add chat message to session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.messages.append(message)
            await self._save_session(session)
            return True
    
    async def add_file(self, session_id: str, file_node: FileNode) -> bool:
        """Add file to session file tree."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.files.append(file_node)
            await self._save_session(session)
            return True
    
    async def update_file(self, session_id: str, path: str, content: str) -> bool:
        """Update file content."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            # Find and update file
            def update_in_tree(nodes: List[FileNode], parts: List[str], depth: int = 0) -> bool:
                for node in nodes:
                    if node.name == parts[depth]:
                        if depth == len(parts) - 1:
                            node.content = content
                            return True
                        if node.type == "folder":
                            return update_in_tree(node.children, parts, depth + 1)
                return False
            
            path_parts = path.split("/")
            update_in_tree(session.files, path_parts)
            await self._save_session(session)
            return True
    
    async def add_task(self, session_id: str, task: TaskProgress) -> bool:
        """Add task to session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.tasks.append(task)
            await self._save_session(session)
            return True
    
    async def update_task(self, session_id: str, task_id: str, **updates) -> bool:
        """Update task progress."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            for task in session.tasks:
                if task.task_id == task_id:
                    for key, value in updates.items():
                        if hasattr(task, key):
                            setattr(task, key, value)
                    await self._save_session(session)
                    return True
            return False
    
    async def add_terminal_line(self, session_id: str, line_type: str, text: str):
        """Add terminal output line."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.terminal_lines.append({"type": line_type, "text": text})
                # Keep only last 200 lines
                if len(session.terminal_lines) > 200:
                    session.terminal_lines = session.terminal_lines[-200:]
                await self._save_session(session)
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        async with self._lock:
            return [
                {
                    "id": s.id,
                    "project_name": s.project_name,
                    "status": s.status,
                    "created_at": s.created_at,
                }
                for s in self._sessions.values()
            ]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                # Remove persisted file
                session_file = self._storage_path / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
                logger.info(f"Session deleted: {session_id}")
                return True
            return False
    
    async def _save_session(self, session: SessionState):
        """Persist session to disk."""
        try:
            session_file = self._storage_path / f"{session.id}.json"
            with open(session_file, "w") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")
    
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session from disk."""
        session_file = self._storage_path / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            
            # Reconstruct session from dict
            session = SessionState(
                id=data["id"],
                project_name=data["project"]["name"],
                description=data["project"]["description"],
                budget=data["budget"]["total"],
                spent=data["budget"]["spent"],
                quality_score=data["metrics"]["quality_score"],
                cache_hit_rate=data["metrics"]["cache_hit_rate"],
                status=data["status"],
            )
            
            # Restore messages
            for msg_data in data.get("messages", []):
                session.messages.append(ChatMessage(**msg_data))
            
            # Restore file tree
            def build_file_tree(data: Dict) -> FileNode:
                return FileNode(
                    name=data["name"],
                    type=data["type"],
                    children=[build_file_tree(c) for c in data.get("children", [])],
                    content=data.get("content"),
                    language=data.get("language"),
                )
            
            session.files = [build_file_tree(f) for f in data.get("files", [])]
            
            # Restore tasks
            for task_data in data.get("tasks", []):
                session.tasks.append(TaskProgress(**task_data))
            
            # Restore terminal lines
            session.terminal_lines = data.get("terminal_lines", [])
            
            async with self._lock:
                self._sessions[session_id] = session
            
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None


# Global session manager instance
_manager: Optional[SessionManager] = None


def get_session_manager(storage_path: Optional[Path] = None) -> SessionManager:
    """Get or create the global session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager(storage_path)
    return _manager


def reset_session_manager():
    """Reset the session manager (for testing)."""
    global _manager
    _manager = None
