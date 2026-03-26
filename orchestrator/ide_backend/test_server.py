"""
Standalone IDE Server Test - No orchestrator package dependency
"""
import sys
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ide_test")
logger.info("Starting standalone test...")

# Add paths
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path))

# Import required modules directly
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logger.info("FastAPI imported")

# Create app
app = FastAPI(title="IDE Test", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions = {}

# Request models
class CreateSessionRequest(BaseModel):
    project_name: str = "Untitled Project"
    description: str = ""
    mode: str = "build"
    autonomy: str = "standard"
    budget: float = 5.0
    model: str = "auto"

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

# API Routes
@app.get("/health")
def health():
    return {"status": "healthy", "sessions": len(sessions)}

@app.get("/api/sessions")
def list_sessions():
    return {"sessions": [{"id": sid, "project_name": s.get("project_name", "Untitled")} for sid, s in sessions.items()]}

@app.post("/api/sessions")
def create_session(request: CreateSessionRequest):
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "id": session_id,
        "project_name": request.project_name,
        "description": request.description,
        "mode": request.mode,
        "autonomy": request.autonomy,
        "budget": request.budget,
        "model": request.model,
        "created_at": datetime.now().timestamp(),
        "messages": [],
        "files": [],
        "tasks": [],
        "terminal_lines": [],
        "spent": 0.0,
        "quality_score": 0.0,
    }
    logger.info(f"Session created: {session_id}")
    return {"session": sessions[session_id]}

@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": sessions[session_id]}

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"success": True}

@app.post("/api/chat")
def send_message(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        # Get first session or create one
        if not sessions:
            session = create_session(CreateSessionRequest())
            session_id = session["session"]["id"]
        else:
            session_id = list(sessions.keys())[0]
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message
    sessions[session_id]["messages"].append({
        "role": "user",
        "content": request.message,
        "ts": datetime.now().strftime("%H:%M"),
    })
    
    return {"session": sessions[session_id]}

@app.get("/api/models")
def get_models():
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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sessions[session_id] = {"connected": True, "messages": []}
    logger.info(f"WebSocket connected: {session_id}")

    try:
        # Send initial session state
        await websocket.send_json({
            "event": "session_state",
            "data": {
                "id": session_id,
                "project": {"name": "Untitled Project"},
                "settings": {"mode": "build", "autonomy": "standard", "model": "auto"},
                "budget": {"total": 5.0, "spent": 0.0},
                "progress": {"total_tasks": 0, "completed_tasks": 0},
                "messages": [],
                "files": [],
                "terminal_lines": [],
                "status": "idle"
            }
        })
        
        while True:
            data = await websocket.receive_json()
            event = data.get("event")
            payload = data.get("data", {})
            logger.info(f"Received event: {event}")
            
            if event == "ping":
                await websocket.send_json({"event": "pong", "data": {"ts": "now"}})
            
            elif event == "chat_message":
                message = payload.get("message", "")
                logger.info(f"Chat message: {message[:50]}...")
                
                # Add user message to session
                if session_id in sessions:
                    sessions[session_id]["messages"].append({
                        "role": "user",
                        "content": message,
                        "ts": "now"
                    })
                
                # Send thinking response
                await websocket.send_json({
                    "event": "messages_update",
                    "data": {
                        "messages": [{
                            "role": "assistant",
                            "content": None,
                            "thinking": True,
                            "steps": [
                                {"label": "Analyzing request...", "done": True},
                                {"label": "Planning approach...", "done": False}
                            ],
                            "ts": "now"
                        }]
                    }
                })
                
                # Simulate processing delay
                import asyncio
                await asyncio.sleep(1.5)
                
                # Send completed response
                response_text = f"""I've received your request: "{message[:100]}"

Here's what I'll do:
1. **Analyze** the requirements
2. **Plan** the architecture
3. **Generate** the code
4. **Test** the implementation

Starting now..."""
                
                await websocket.send_json({
                    "event": "messages_update",
                    "data": {
                        "messages": [{
                            "role": "assistant",
                            "content": response_text,
                            "thinking": False,
                            "steps": [
                                {"label": "Analyzing request...", "done": True},
                                {"label": "Planning approach...", "done": True},
                                {"label": "Creating tasks...", "done": True}
                            ],
                            "ts": "now",
                            "quality": 0.92
                        }]
                    }
                })
                
                # Update session
                if session_id in sessions:
                    sessions[session_id]["messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "thinking": False
                    })
            
            elif event == "session_update":
                logger.info(f"Session update: {payload}")
                # Acknowledge update
                await websocket.send_json({
                    "event": "session_state",
                    "data": {"settings": payload}
                })
            
            elif event == "terminal_command":
                command = payload.get("command", "")
                logger.info(f"Terminal command: {command}")
                await websocket.send_json({
                    "event": "terminal_update",
                    "data": {
                        "lines": [
                            {"type": "cmd", "text": f"$ {command}"},
                            {"type": "info", "text": f"Executing: {command}"},
                            {"type": "success", "text": "Command completed"}
                        ]
                    }
                })
            
            else:
                logger.warning(f"Unknown event: {event}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        sessions.pop(session_id, None)

# Serve frontend
frontend_dist = base_path.parent.parent / "ide_frontend" / "dist"
if frontend_dist.exists():
    logger.info(f"Serving frontend from {frontend_dist}")
    app.mount("/ide", StaticFiles(directory=str(frontend_dist), html=True), name="ide")
    
    @app.get("/")
    def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ide")
else:
    logger.warning(f"Frontend not found at {frontend_dist}")

if __name__ == "__main__":
    logger.info("Starting server on port 8765...")
    print("\n" + "=" * 60)
    print("  AI Orchestrator IDE - Test Server")
    print("=" * 60)
    print("  🌐 http://localhost:8765")
    print("  📁 Frontend:", "Yes" if frontend_dist.exists() else "No")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
