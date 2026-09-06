"""
Dashboard Chat View — /chat page and /ws/chat WebSocket handler.

Serves a self-contained chat UI that talks to ConversationAgent over WebSocket.
When the spec is ready and the user confirms, it fires the build and streams
progress back to the browser via the same WebSocket.

Protocol (JSON frames):
  Client → Server:
    { "type": "message", "text": "..." }
    { "type": "accept_enhancement", "text": "..." }
    { "type": "build" }         ← user clicked "Build it"

  Server → Client:
    { "type": "agent",       "text": "...", "suggestions": [...], "confidence": 0.7, "ready": false }
    { "type": "build_start" }
    { "type": "build_log",   "text": "..." }
    { "type": "build_done",  "status": "SUCCESS" }
    { "type": "error",       "text": "..." }
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import WebSocket, WebSocketDisconnect

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = object  # type: ignore[misc,assignment]
    WebSocketDisconnect = Exception  # type: ignore[misc,assignment]


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket handler
# ─────────────────────────────────────────────────────────────────────────────


async def handle_chat_websocket(websocket: WebSocket) -> None:  # type: ignore[valid-type]
    """Manage one chat session over a WebSocket connection."""
    await websocket.accept()

    try:
        from ..api_clients import UnifiedClient
        from ..cache import DiskCache
        from ..application.conversation_agent import ConversationAgent

        cache = DiskCache()
        client = UnifiedClient(cache=cache)
        agent = ConversationAgent(client=client)

        # Send opening message
        opening = await agent.start()
        await websocket.send_json(
            {
                "type": "agent",
                "text": opening.content,
                "suggestions": opening.suggestions,
                "confidence": opening.confidence,
                "ready": opening.ready_to_build,
            }
        )

        # Conversation loop
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                frame = {"type": "message", "text": raw}

            frame_type = frame.get("type", "message")

            if frame_type == "accept_enhancement":
                agent.accept_enhancement(frame.get("text", ""))
                continue

            if frame_type == "build" or (frame_type == "message" and agent.ready):
                await _run_build(websocket, agent)
                return

            if frame_type == "message":
                text = frame.get("text", "").strip()
                if not text:
                    continue

                turn = await agent.process_turn(text)
                await websocket.send_json(
                    {
                        "type": "agent",
                        "text": turn.content,
                        "suggestions": turn.suggestions,
                        "confidence": turn.confidence,
                        "ready": turn.ready_to_build,
                    }
                )

                if turn.ready_to_build:
                    # Wait for explicit "build" frame or next message
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Chat WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "text": str(exc)})
        except Exception:
            pass


async def _run_build(websocket: WebSocket, agent) -> None:  # type: ignore[valid-type]
    """Launch the orchestrator build and stream progress back."""
    from ..engine import Orchestrator
    from ..budget import Budget

    spec = agent.spec
    args = spec.to_orchestrator_args()

    await websocket.send_json({"type": "build_start"})

    try:
        orch = Orchestrator(budget=Budget(max_usd=spec.budget_usd))
        async with orch:
            state = await orch.run_project(
                project_description=args["project"],
                success_criteria=args["criteria"],
            )
        status = state.status.value if hasattr(state.status, "value") else str(state.status)
        await websocket.send_json({"type": "build_done", "status": status})
    except Exception as exc:
        await websocket.send_json({"type": "error", "text": f"Build failed: {exc}"})


# ─────────────────────────────────────────────────────────────────────────────
# HTML page
# ─────────────────────────────────────────────────────────────────────────────


def render_chat_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Build something — Orchestrator Chat</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0d0f14;
      --surface: #161b22;
      --surface2: #1e2430;
      --border: #30363d;
      --accent: #58a6ff;
      --accent2: #3fb950;
      --warn: #d29922;
      --text: #e6edf3;
      --text-dim: #8b949e;
      --radius: 12px;
      --font: 'Inter', system-ui, sans-serif;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      height: 100dvh;
      display: flex;
      flex-direction: column;
    }

    /* ── Header ───────────────────────────────────────── */
    header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
      background: var(--surface);
      flex-shrink: 0;
    }
    header .logo { font-size: 20px; font-weight: 700; color: var(--accent); }
    header .sub  { font-size: 13px; color: var(--text-dim); }
    header a { margin-left: auto; color: var(--text-dim); font-size: 13px; text-decoration: none; }
    header a:hover { color: var(--accent); }

    /* ── Confidence bar ───────────────────────────────── */
    #conf-bar-wrap {
      padding: 8px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    #conf-label { font-size: 12px; color: var(--text-dim); white-space: nowrap; }
    #conf-track {
      flex: 1;
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }
    #conf-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      border-radius: 2px;
      transition: width 0.5s ease;
    }
    #conf-pct { font-size: 12px; color: var(--text-dim); width: 32px; text-align: right; }

    /* ── Messages ─────────────────────────────────────── */
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
    }

    .msg { display: flex; gap: 12px; max-width: 780px; }
    .msg.user  { align-self: flex-end; flex-direction: row-reverse; }
    .msg.agent { align-self: flex-start; }

    .avatar {
      width: 32px; height: 32px;
      border-radius: 50%;
      display: grid; place-items: center;
      font-size: 14px;
      flex-shrink: 0;
    }
    .msg.agent .avatar { background: var(--accent); color: #000; }
    .msg.user  .avatar { background: var(--surface2); color: var(--text-dim); }

    .bubble {
      padding: 12px 16px;
      border-radius: var(--radius);
      font-size: 14px;
      line-height: 1.65;
      max-width: 640px;
      white-space: pre-wrap;
    }
    .msg.agent .bubble {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-top-left-radius: 4px;
    }
    .msg.user .bubble {
      background: var(--accent);
      color: #000;
      border-top-right-radius: 4px;
    }

    /* ── Suggestions ──────────────────────────────────── */
    .suggestions {
      display: flex; flex-wrap: wrap; gap: 8px;
      margin-top: 8px;
      margin-left: 44px;
    }
    .suggestion-chip {
      display: flex; align-items: center; gap: 6px;
      padding: 5px 12px;
      background: transparent;
      border: 1px solid var(--warn);
      color: var(--warn);
      border-radius: 999px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
    }
    .suggestion-chip:hover { background: var(--warn); color: #000; }
    .suggestion-chip.accepted { border-color: var(--accent2); color: var(--accent2); }
    .suggestion-chip.accepted::before { content: "✓ "; }

    /* ── Build ready banner ───────────────────────────── */
    #ready-banner {
      display: none;
      margin: 0 24px 16px;
      padding: 14px 20px;
      background: linear-gradient(135deg, #1a2e1a, #1e2e1a);
      border: 1px solid var(--accent2);
      border-radius: var(--radius);
      display: none;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }
    #ready-banner .ready-text { flex: 1; font-size: 14px; color: var(--accent2); }
    #build-btn {
      padding: 10px 22px;
      background: var(--accent2);
      color: #000;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    #build-btn:hover { opacity: 0.85; }
    #build-btn:disabled { opacity: 0.4; cursor: default; }

    /* ── Build progress ───────────────────────────────── */
    #build-log {
      display: none;
      margin: 0 24px 16px;
      padding: 14px 20px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      font-family: monospace;
      font-size: 12px;
      color: var(--text-dim);
      max-height: 160px;
      overflow-y: auto;
      flex-shrink: 0;
    }

    /* ── Input bar ────────────────────────────────────── */
    #input-area {
      padding: 16px 24px;
      border-top: 1px solid var(--border);
      background: var(--surface);
      display: flex;
      gap: 10px;
      flex-shrink: 0;
    }
    #user-input {
      flex: 1;
      padding: 12px 16px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 14px;
      font-family: var(--font);
      resize: none;
      min-height: 44px;
      max-height: 140px;
      overflow-y: auto;
      transition: border-color 0.15s;
    }
    #user-input:focus { outline: none; border-color: var(--accent); }
    #user-input::placeholder { color: var(--text-dim); }
    #send-btn {
      padding: 0 18px;
      background: var(--accent);
      color: #000;
      border: none;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
      white-space: nowrap;
    }
    #send-btn:hover { opacity: 0.85; }
    #send-btn:disabled { opacity: 0.4; cursor: default; }

    /* ── Typing indicator ─────────────────────────────── */
    .typing-indicator .bubble {
      display: flex; gap: 4px; align-items: center;
      padding: 14px 18px;
    }
    .dot {
      width: 6px; height: 6px;
      background: var(--text-dim);
      border-radius: 50%;
      animation: bounce 1.2s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); }
      40%            { transform: translateY(-6px); }
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>

<header>
  <div class="logo">⚡ Orchestrator</div>
  <div class="sub">Interactive build mode</div>
  <a href="/">← Dashboard</a>
</header>

<div id="conf-bar-wrap">
  <span id="conf-label">Spec completeness</span>
  <div id="conf-track"><div id="conf-fill"></div></div>
  <span id="conf-pct">0%</span>
</div>

<div id="messages"></div>

<div id="ready-banner">
  <span class="ready-text">✓ Spec is complete — ready to build.</span>
  <button id="build-btn" onclick="startBuild()">🚀 Build it</button>
</div>

<div id="build-log"></div>

<div id="input-area">
  <textarea
    id="user-input"
    rows="1"
    placeholder="Describe what you want to build..."
    onkeydown="handleKey(event)"
    oninput="autoResize(this)"
  ></textarea>
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>

<script>
  const ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
  const messages = document.getElementById('messages');
  const input = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const readyBanner = document.getElementById('ready-banner');
  const buildLog = document.getElementById('build-log');

  let isReady = false;
  let typingEl = null;

  ws.onopen = () => {
    sendBtn.disabled = false;
  };

  ws.onmessage = (e) => {
    const frame = JSON.parse(e.data);
    removeTyping();

    if (frame.type === 'agent') {
      appendMessage('agent', frame.text);
      if (frame.suggestions && frame.suggestions.length > 0) {
        appendSuggestions(frame.suggestions);
      }
      updateConfidence(frame.confidence || 0);
      if (frame.ready && !isReady) {
        isReady = true;
        readyBanner.style.display = 'flex';
      }
    } else if (frame.type === 'build_start') {
      readyBanner.style.display = 'none';
      buildLog.style.display = 'block';
      buildLog.textContent = '⚙ Build started...\\n';
      input.disabled = true;
      sendBtn.disabled = true;
    } else if (frame.type === 'build_log') {
      buildLog.textContent += frame.text + '\\n';
      buildLog.scrollTop = buildLog.scrollHeight;
    } else if (frame.type === 'build_done') {
      buildLog.textContent += `\\n✅ Done — status: ${frame.status}`;
      appendMessage('agent', `Build complete! Status: **${frame.status}**`);
    } else if (frame.type === 'error') {
      appendMessage('agent', `⚠ Error: ${frame.text}`);
    }
  };

  ws.onerror = () => {
    appendMessage('agent', '⚠ Connection lost. Please refresh the page.');
  };

  function sendMessage() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;
    appendMessage('user', text);
    ws.send(JSON.stringify({ type: 'message', text }));
    input.value = '';
    autoResize(input);
    showTyping();
  }

  function startBuild() {
    document.getElementById('build-btn').disabled = true;
    ws.send(JSON.stringify({ type: 'build' }));
  }

  function acceptSuggestion(chip, text) {
    if (chip.classList.contains('accepted')) return;
    chip.classList.add('accepted');
    ws.send(JSON.stringify({ type: 'accept_enhancement', text }));
  }

  function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
      <div class="avatar">${role === 'agent' ? '⚡' : '👤'}</div>
      <div class="bubble">${escHtml(text)}</div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function appendSuggestions(suggestions) {
    const wrap = document.createElement('div');
    wrap.className = 'suggestions';
    suggestions.forEach(s => {
      const chip = document.createElement('button');
      chip.className = 'suggestion-chip';
      chip.textContent = '✦ ' + s;
      chip.onclick = () => acceptSuggestion(chip, s);
      wrap.appendChild(chip);
    });
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
  }

  function updateConfidence(val) {
    const pct = Math.round(val * 100);
    document.getElementById('conf-fill').style.width = pct + '%';
    document.getElementById('conf-pct').textContent = pct + '%';
  }

  function showTyping() {
    removeTyping();
    typingEl = document.createElement('div');
    typingEl.className = 'msg agent typing-indicator';
    typingEl.innerHTML = `
      <div class="avatar">⚡</div>
      <div class="bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
    `;
    messages.appendChild(typingEl);
    messages.scrollTop = messages.scrollHeight;
  }

  function removeTyping() {
    if (typingEl) { typingEl.remove(); typingEl = null; }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
             .replace(/"/g,'&quot;').replace(/\\n/g,'<br>');
  }
</script>
</body>
</html>"""
