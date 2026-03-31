"""
API Routes - REST endpoints for IDE
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from orchestrator.ide_backend.log_config import get_logger
from orchestrator.ide_backend.session_manager import (
    ChatMessage,
    FileNode,
    get_session_manager,
)

logger = get_logger(__name__)

router = APIRouter(tags=["IDE"])


# Request/Response models
class CreateSessionRequest(BaseModel):
    project_name: str = "Untitled Project"
    description: str = ""
    mode: str = "build"
    autonomy: str = "standard"
    budget: float = 5.0
    model: str = "auto"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class FileUpdateRequest(BaseModel):
    content: str
    language: str | None = None


class TaskUpdateRequest(BaseModel):
    status: str | None = None
    score: float | None = None
    cost: float | None = None
    repairs: int | None = None


# Session endpoints
@router.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new IDE session."""
    session_manager = get_session_manager()
    session = await session_manager.create_session(
        project_name=request.project_name,
        description=request.description,
        mode=request.mode,
        autonomy=request.autonomy,
        budget=request.budget,
        model=request.model,
    )
    return {"session": session.to_dict()}


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    session_manager = get_session_manager()
    sessions = await session_manager.list_sessions()
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session by ID."""
    session_manager = get_session_manager()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.to_dict()}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager = get_session_manager()
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


# Chat endpoints
@router.post("/chat")
async def send_message(request: ChatRequest):
    """Send a chat message to the session."""
    session_manager = get_session_manager()

    session_id = request.session_id
    if not session_id:
        # Get or create default session
        sessions = await session_manager.list_sessions()
        if sessions:
            session_id = sessions[0]["id"]
        else:
            session = await session_manager.create_session()
            session_id = session.id

    # Add user message
    from datetime import datetime

    user_message = ChatMessage(
        role="user",
        content=request.message,
        timestamp=datetime.now().strftime("%H:%M"),
    )
    await session_manager.add_message(session_id, user_message)

    # TODO: Send to orchestrator and get response
    # For now, add a placeholder assistant message
    assistant_message = ChatMessage(
        role="assistant",
        content=None,
        thinking=True,
        steps=[{"label": "Processing request...", "done": False}],
    )
    await session_manager.add_message(session_id, assistant_message)

    session = await session_manager.get_session(session_id)
    return {"session": session.to_dict(), "message_id": len(session.messages) - 1}


# File endpoints
@router.get("/sessions/{session_id}/files")
async def get_files(session_id: str):
    """Get file tree for session."""
    session_manager = get_session_manager()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"files": [f.to_dict() for f in session.files]}


@router.get("/sessions/{session_id}/files/{file_path:path}")
async def get_file_content(session_id: str, file_path: str):
    """Get file content."""
    session_manager = get_session_manager()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find file in tree
    def find_file(nodes: list[FileNode], parts: list[str], depth: int = 0) -> FileNode | None:
        for node in nodes:
            if node.name == parts[depth]:
                if depth == len(parts) - 1:
                    return node if node.type == "file" else None
                if node.type == "folder":
                    return find_file(node.children, parts, depth + 1)
        return None

    path_parts = file_path.split("/")
    file_node = find_file(session.files, path_parts)

    if not file_node:
        raise HTTPException(status_code=404, detail="File not found")

    return {"content": file_node.content or "", "language": file_node.language}


@router.put("/sessions/{session_id}/files/{file_path:path}")
async def update_file(session_id: str, file_path: str, request: FileUpdateRequest):
    """Update file content."""
    session_manager = get_session_manager()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    success = await session_manager.update_file(session_id, file_path, request.content)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")

    return {"success": True}


# Task endpoints
@router.get("/sessions/{session_id}/tasks")
async def get_tasks(session_id: str):
    """Get tasks for session."""
    session_manager = get_session_manager()
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"tasks": [t.__dict__ for t in session.tasks]}


@router.put("/sessions/{session_id}/tasks/{task_id}")
async def update_task(session_id: str, task_id: str, request: TaskUpdateRequest):
    """Update task progress."""
    session_manager = get_session_manager()

    updates = {k: v for k, v in request.dict().items() if v is not None}
    success = await session_manager.update_task(session_id, task_id, **updates)

    if not success:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"success": True}


# Model endpoints
@router.get("/models")
async def get_models():
    """Get available models."""

    return {
        "models": [
            {"id": "auto", "name": "Auto (Tiered)", "desc": "Smart routing", "icon": "⚡"},
            {"id": "opus", "name": "Claude Opus 4.6", "desc": "Reasoning", "icon": "◆"},
            {"id": "sonnet", "name": "Claude Sonnet 4.6", "desc": "Balanced", "icon": "◇"},
            {"id": "deepseek", "name": "DeepSeek V3.2", "desc": "Budget", "icon": "●"},
            {"id": "gpt54", "name": "GPT-5.4", "desc": "Tools", "icon": "○"},
            {"id": "gemini", "name": "Gemini 3.1 Pro", "desc": "Long context", "icon": "◈"},
        ]
    }


# Health endpoint
@router.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": "now"}
