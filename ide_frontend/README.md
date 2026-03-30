# AI Orchestrator IDE

Real-time IDE dashboard for the AI Orchestrator system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Orchestrator IDE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐         ┌─────────────────┐            │
│  │   React IDE     │◀───────▶│   FastAPI       │            │
│  │   (Vite)        │  WS     │   Server        │            │
│  │   Port 3000     │  REST   │   Port 8765     │            │
│  └─────────────────┘         └────────┬────────┘            │
│                                       │                      │
│                          ┌────────────┼────────────┐         │
│                          │            │            │         │
│                          ▼            ▼            ▼         │
│                   ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│                   │Orchestr- │ │  SQLite  │ │   LLM    │    │
│                   │ator Core │ │  Events  │ │ Providers│    │
│                   └──────────┘ └──────────┘ └──────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Frontend Dependencies

```bash
cd ide_frontend
npm install
```

### 2. Start the IDE Server

```bash
# From project root
python -m orchestrator.ide_backend.launch --reload

# Or with specific port
python -m orchestrator.ide_backend.launch --port 9000
```

### 3. Development Mode (Optional)

For frontend development with hot reload:

```bash
# Terminal 1: Start backend
python -m orchestrator.ide_backend.launch --no-frontend

# Terminal 2: Start frontend dev server
cd ide_frontend
npm run dev
```

Then open http://localhost:3000

## Features

### Real-time Communication
- **WebSocket** for live updates (chat, terminal, file changes)
- **REST API** for CRUD operations (sessions, files, tasks)

### Session Management
- Create and manage multiple project sessions
- Persist session state to disk
- Restore sessions after restart

### Chat Interface
- Send messages to the orchestrator
- View thinking steps and progress
- See cost and quality metrics

### File Explorer
- Browse generated file tree
- View and edit file contents
- Syntax highlighting support

### Terminal
- View command output
- Real-time log streaming
- Test results and linting output

### Settings
- **Mode**: Build, Plan, Chat, Debug
- **Model**: Auto, Claude Opus/Sonnet, DeepSeek, GPT, Gemini
- **Autonomy**: Lite, Standard, Autonomous, Max

## API Reference

### Sessions

```bash
# Create session
POST /api/sessions
{
  "project_name": "My Project",
  "mode": "build",
  "autonomy": "standard",
  "budget": 5.0
}

# Get session
GET /api/sessions/{session_id}

# List sessions
GET /api/sessions

# Delete session
DELETE /api/sessions/{session_id}
```

### Chat

```bash
# Send message
POST /api/chat
{
  "message": "Build a FastAPI REST API",
  "session_id": "abc123"
}
```

### Files

```bash
# Get file tree
GET /api/sessions/{session_id}/files

# Get file content
GET /api/sessions/{session_id}/files/src/main.py

# Update file
PUT /api/sessions/{session_id}/files/src/main.py
{
  "content": "print('Hello')",
  "language": "python"
}
```

### WebSocket Events

#### Client → Server
- `chat_message`: Send chat message
- `session_update`: Update settings
- `file_request`: Request file content
- `file_update`: Update file content
- `terminal_command`: Execute terminal command
- `ping`: Keep-alive ping

#### Server → Client
- `session_state`: Full session state
- `messages_update`: New chat messages
- `files_update`: File tree changed
- `terminal_update`: New terminal output
- `task_progress`: Task progress update
- `budget_update`: Budget changed
- `pong`: Ping response

## Project Structure

```
orchestrator/ide_backend/
├── __init__.py              # Package exports
├── server.py                # FastAPI application
├── launch.py                # Launcher script
├── log_config.py            # Logging configuration
├── websocket_manager.py     # WebSocket connection manager
├── session_manager.py       # Session state management
├── api/
│   ├── __init__.py
│   └── routes.py            # REST API routes
├── websocket/
│   ├── __init__.py
│   └── handlers.py          # WebSocket event handlers
└── integration/
    ├── __init__.py
    └── orchestrator_bridge.py  # Orchestrator integration

ide_frontend/
├── package.json
├── vite.config.js
├── index.html
└── src/
    ├── main.jsx             # React entry point
    ├── App.jsx              # Main IDE component
    ├── index.css            # Global styles
    ├── hooks/
    │   ├── useWebSocket.js  # WebSocket hook
    │   └── useSession.js    # Session management hook
    └── services/
        ├── websocket.js     # WebSocket service
        └── api.js           # REST API client
```

## Configuration

### Environment Variables

```bash
# Backend
PORT=8765
HOST=0.0.0.0
LOG_LEVEL=INFO

# Frontend (via Vite)
VITE_API_URL=http://localhost:8765/api
VITE_WS_URL=ws://localhost:8765/ws
```

### Session Storage

Sessions are persisted to:
- Windows: `C:\Users\<user>\.orchestrator\ide_sessions\`
- Linux/Mac: `~/.orchestrator/ide_sessions/`

## Integration with Orchestrator Core

To integrate with the main Orchestrator:

```python
from orchestrator import Orchestrator
from orchestrator.ide_backend import run_ide_server

# Create orchestrator
orchestrator = Orchestrator(...)

# Start IDE with orchestrator integration
run_ide_server(orchestrator=orchestrator, port=8765)
```

The `OrchestratorBridge` will automatically subscribe to orchestrator events and broadcast them to connected IDE clients.

## Troubleshooting

### Frontend not loading
1. Check if frontend build exists: `ide_frontend/dist/index.html`
2. Run `npm run build` in `ide_frontend/`
3. Restart server with `--reload`

### WebSocket disconnects
1. Check firewall settings for port 8765
2. Verify no proxy interference
3. Check browser console for errors

### Session not persisting
1. Check write permissions for `~/.orchestrator/ide_sessions/`
2. Verify disk space
3. Check logs for errors

## Development

### Running Tests

```bash
# Backend tests
pytest orchestrator/ide_backend/tests/

# Frontend tests
cd ide_frontend
npm test
```

### Linting

```bash
# Backend
ruff check orchestrator/ide_backend/

# Frontend
cd ide_frontend
npm run lint
```

## License

MIT
