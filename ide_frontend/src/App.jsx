import { useState, useEffect, useRef, useCallback } from "react";
import { 
  Camera, Send, Paperclip, Play, Eye, Code2, Terminal, FolderTree, Settings, 
  ChevronDown, ChevronRight, X, Maximize2, Minimize2, RotateCcw, ExternalLink, 
  Zap, Shield, Brain, GitBranch, Upload, Mic, Sparkles, Search, LayoutGrid, 
  PanelLeftClose, PanelLeft, MoreVertical, Check, Loader2, AlertTriangle, Copy, 
  Download, Globe, Cpu, DollarSign, Clock, FileCode, FileText, Folder, ChevronUp, 
  Bot, User, ArrowRight, RefreshCw, Pause, MessageSquare, Layers 
} from "lucide-react";
import { useSession } from "./hooks/useSession";
import { getWebSocketService } from "./services/websocket";

/* ════════════════════════════════════════
   DESIGN SYSTEM — Obsidian Theme
   ════════════════════════════════════════ */
const DS = {
  bg: "#09090b",
  bgElevated: "#0c0c0f",
  surface: "#111113",
  surfaceHover: "#18181b",
  surfaceActive: "#1f1f23",
  card: "#131316",
  border: "#1c1c21",
  borderHover: "#27272a",
  borderActive: "#3f3f46",
  text: "#fafafa",
  textSecondary: "#a1a1aa",
  textMuted: "#52525b",
  textDim: "#3f3f46",
  accent: "#818cf8",
  accentHover: "#6366f1",
  accentMuted: "#818cf815",
  success: "#34d399",
  successMuted: "#34d39915",
  warning: "#fbbf24",
  warningMuted: "#fbbf2415",
  error: "#f87171",
  errorMuted: "#f8717115",
  info: "#38bdf8",
  infoMuted: "#38bdf815",
  mono: "'JetBrains Mono', 'SF Mono', 'Cascadia Code', monospace",
  sans: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
  radius: { sm: 6, md: 8, lg: 12 },
};

/* ════════════════════════════════════════
   CONSTANTS
   ════════════════════════════════════════ */
const MODELS = [
  { id: "auto", name: "Auto (Tiered)", desc: "Smart routing", icon: "⚡" },
  { id: "opus", name: "Claude Opus 4.6", desc: "Reasoning", icon: "◆" },
  { id: "sonnet", name: "Claude Sonnet 4.6", desc: "Balanced", icon: "◇" },
  { id: "deepseek", name: "DeepSeek V3.2", desc: "Budget", icon: "●" },
  { id: "gpt54", name: "GPT-5.4", desc: "Tools", icon: "○" },
  { id: "gemini", name: "Gemini 3.1 Pro", desc: "Long context", icon: "◈" },
];

const MODES = [
  { id: "build", name: "Build", icon: Zap, desc: "Generate & execute code" },
  { id: "plan", name: "Plan", icon: Layers, desc: "Plan without executing" },
  { id: "chat", name: "Chat", icon: MessageSquare, desc: "Discuss & brainstorm" },
  { id: "debug", name: "Debug", icon: Search, desc: "Find & fix issues" },
];

const AUTONOMY = [
  { id: "lite", name: "Lite", desc: "Quick, no repair", color: DS.info },
  { id: "standard", name: "Standard", desc: "1 repair, unit tests", color: DS.success },
  { id: "autonomous", name: "Autonomous", desc: "3 repairs, integration", color: DS.warning },
  { id: "max", name: "Max", desc: "5 repairs, 200min", color: DS.error },
];

/* ════════════════════════════════════════
   MICRO COMPONENTS
   ════════════════════════════════════════ */
const Btn = ({ children, variant = "ghost", size = "sm", active, style, ...props }) => {
  const variants = {
    ghost: { bg: "transparent", hover: DS.surfaceHover, border: "transparent", color: DS.textSecondary },
    outline: { bg: "transparent", hover: DS.surfaceHover, border: DS.border, color: DS.textSecondary },
    filled: { bg: DS.accent, hover: DS.accentHover, border: DS.accent, color: "#fff" },
    surface: { bg: DS.surface, hover: DS.surfaceHover, border: DS.border, color: DS.text },
  };
  const v = variants[variant];
  const sizes = { xs: { px: 6, py: 3, fs: 11 }, sm: { px: 10, py: 5, fs: 12 }, md: { px: 14, py: 7, fs: 13 } };
  const s = sizes[size];
  return (
    <button {...props} style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: `${s.py}px ${s.px}px`, borderRadius: DS.radius.sm,
      background: active ? DS.surfaceActive : v.bg,
      border: `1px solid ${active ? DS.borderActive : v.border}`,
      color: active ? DS.text : v.color, fontSize: s.fs,
      fontFamily: DS.sans, fontWeight: 500, cursor: "pointer",
      transition: "all 0.15s ease", lineHeight: 1, whiteSpace: "nowrap",
      ...style,
    }}
      onMouseEnter={e => { if (!active) e.currentTarget.style.background = v.hover; }}
      onMouseLeave={e => { if (!active) e.currentTarget.style.background = active ? DS.surfaceActive : v.bg; }}
    >{children}</button>
  );
};

const Divider = ({ vertical }) => (
  <div style={vertical
    ? { width: 1, background: DS.border, alignSelf: "stretch", margin: "0 2px" }
    : { height: 1, background: DS.border, width: "100%" }
  } />
);

const Badge = ({ children, color = DS.textMuted }) => (
  <span style={{
    padding: "1px 6px", borderRadius: 4, fontSize: 10,
    fontFamily: DS.mono, fontWeight: 500, color,
    background: `${color}12`, border: `1px solid ${color}20`,
    letterSpacing: "0.02em",
  }}>{children}</span>
);

const Dot = ({ color = DS.success, pulse }) => (
  <span style={{ position: "relative", display: "inline-flex", width: 8, height: 8 }}>
    <span style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
    {pulse && <span style={{
      position: "absolute", inset: -2, borderRadius: "50%",
      border: `1.5px solid ${color}`, opacity: 0.5,
      animation: "ping 2s ease-in-out infinite",
    }} />}
  </span>
);

const Kbd = ({ children }) => (
  <kbd style={{
    padding: "1px 5px", borderRadius: 3, fontSize: 10,
    fontFamily: DS.mono, color: DS.textMuted,
    background: DS.surfaceHover, border: `1px solid ${DS.border}`,
  }}>{children}</kbd>
);

/* ════════════════════════════════════════
   FILE TREE COMPONENT
   ════════════════════════════════════════ */
const FileTreeItem = ({ item, depth = 0, selected, onSelect, parentPath = "" }) => {
  const [open, setOpen] = useState(true);
  const isFolder = item.type === "folder";
  const Icon = isFolder ? (open ? ChevronDown : ChevronRight) : FileCode;
  
  // Build full path for files
  const fullPath = parentPath ? `${parentPath}/${item.name}` : item.name;

  return (
    <>
      <div onClick={() => isFolder ? setOpen(!open) : onSelect(fullPath)}
        style={{
          display: "flex", alignItems: "center", gap: 5,
          padding: `3px 8px 3px ${12 + depth * 14}px`,
          cursor: "pointer", fontSize: 12, fontFamily: DS.mono,
          color: selected === fullPath ? DS.text : DS.textSecondary,
          background: selected === fullPath ? DS.surfaceActive : "transparent",
          borderRadius: 4, margin: "1px 4px",
          transition: "background 0.1s",
        }}
        onMouseEnter={e => { if (selected !== fullPath) e.currentTarget.style.background = DS.surfaceHover; }}
        onMouseLeave={e => { if (selected !== fullPath) e.currentTarget.style.background = "transparent"; }}
      >
        <Icon size={13} style={{ color: isFolder ? DS.warning : DS.textMuted, flexShrink: 0 }} />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.name}</span>
      </div>
      {isFolder && open && item.children?.map((child, i) => (
        <FileTreeItem key={i} item={child} depth={depth + 1} selected={selected} onSelect={onSelect} parentPath={fullPath} />
      ))}
    </>
  );
};

/* ════════════════════════════════════════
   CHAT MESSAGE COMPONENT
   ════════════════════════════════════════ */
const ChatMessage = ({ msg }) => {
  const isUser = msg.role === "user";

  if (msg.thinking) {
    return (
      <div style={{ padding: "12px 16px", display: "flex", gap: 10 }}>
        <div style={{
          width: 24, height: 24, borderRadius: DS.radius.sm, flexShrink: 0,
          background: `linear-gradient(135deg, ${DS.accent}, ${DS.info})`,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}><Bot size={13} color="#fff" /></div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: DS.textMuted, marginBottom: 8, fontFamily: DS.mono }}>Thinking...</div>
          {msg.steps?.map((step, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "4px 0", fontSize: 12, fontFamily: DS.mono,
              color: step.done ? DS.textSecondary : DS.textMuted,
            }}>
              {step.done ? <Check size={12} color={DS.success} /> : <Loader2 size={12} color={DS.accent} style={{ animation: "spin 1s linear infinite" }} />}
              {step.label}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      padding: "12px 16px", display: "flex", gap: 10,
      background: isUser ? "transparent" : `${DS.accent}04`,
    }}>
      <div style={{
        width: 24, height: 24, borderRadius: DS.radius.sm, flexShrink: 0,
        background: isUser ? DS.surfaceActive : `linear-gradient(135deg, ${DS.accent}, ${DS.info})`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        {isUser ? <User size={13} color={DS.textSecondary} /> : <Bot size={13} color="#fff" />}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: DS.text }}>{isUser ? "You" : "Orchestrator"}</span>
          <span style={{ fontSize: 10, fontFamily: DS.mono, color: DS.textDim }}>{msg.ts}</span>
          {msg.cost != null && <Badge color={DS.success}>${msg.cost.toFixed(2)}</Badge>}
          {msg.quality != null && <Badge color={DS.accent}>Q: {msg.quality}</Badge>}
        </div>
        <div style={{
          fontSize: 13, lineHeight: 1.65, color: DS.textSecondary,
          fontFamily: DS.sans, whiteSpace: "pre-wrap",
        }}>
          {msg.content?.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
            part.startsWith("**") && part.endsWith("**")
              ? <strong key={i} style={{ color: DS.text, fontWeight: 600 }}>{part.slice(2, -2)}</strong>
              : <span key={i}>{part}</span>
          )}
        </div>
        {msg.files && (
          <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
            {msg.files.map((f, i) => (
              <span key={i} style={{
                display: "inline-flex", alignItems: "center", gap: 4,
                padding: "3px 8px", borderRadius: 4, fontSize: 11,
                fontFamily: DS.mono, color: DS.accent,
                background: DS.accentMuted, border: `1px solid ${DS.accent}20`,
                cursor: "pointer",
              }}><FileCode size={11} />{f}</span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/* ════════════════════════════════════════
   MAIN APP COMPONENT
   ════════════════════════════════════════ */
export default function App() {
  const {
    session,
    sessionId,
    isConnected,
    loading,
    sendChatMessage,
    updateSettings,
    sendMessage,
  } = useSession();

  const [input, setInput] = useState("");
  const [rightPanel, setRightPanel] = useState("preview");
  const [showFiles, setShowFiles] = useState(true);
  const [showTerminal, setShowTerminal] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState("");
  const selectedFileRef = useRef(null);  // Keep track of current selected file
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [showAutonomy, setShowAutonomy] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Update ref when selectedFile changes
  useEffect(() => {
    selectedFileRef.current = selectedFile;
  }, [selectedFile]);

  // Fetch file content when selectedFile changes
  useEffect(() => {
    console.log('File selected:', selectedFile, 'Session:', sessionId, 'SendMessage:', typeof sendMessage);
    if (selectedFile && sessionId && sendMessage) {
      // Request file content via WebSocket
      console.log('Requesting file content for:', selectedFile);
      sendMessage('file_request', { path: selectedFile, session_id: sessionId });
    }
  }, [selectedFile, sessionId, sendMessage]);

  // Listen for file content response - use ref to avoid stale closure
  useEffect(() => {
    if (!isConnected) return;

    console.log('Setting up file_content listener, isConnected:', isConnected);
    const ws = getWebSocketService();
    console.log('WebSocket service:', ws, 'isConnected:', ws.isConnected());

    const unsubscribe = ws.on('file_content', (data) => {
      console.log('*** FILE CONTENT RECEIVED ***', data.path, data.content?.length, 'bytes');
      console.log('Current selected file (from ref):', selectedFileRef.current, 'Match:', data.path === selectedFileRef.current);
      if (data.path === selectedFileRef.current) {
        console.log('Setting file content!');
        setFileContent(data.content);
      } else {
        console.log('File path mismatch, current selection:', selectedFileRef.current);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [isConnected]);

  // Listen for terminal updates
  useEffect(() => {
    if (!isConnected) return;

    const ws = getWebSocketService();
    const unsubscribe = ws.on('terminal_update', (data) => {
      console.log('*** TERMINAL UPDATE RECEIVED ***', data);
      // Terminal lines are updated via session_state event
      // This listener is just for debugging
    });

    return () => {
      unsubscribe();
    };
  }, [isConnected]);

  // Auto-refresh preview when files are modified
  const [previewKey, setPreviewKey] = useState(0);
  
  useEffect(() => {
    if (!isConnected) return;

    const ws = getWebSocketService();
    const unsubscribe = ws.on('terminal_update', (data) => {
      // Check if this is a file modification
      const lines = data.lines || [];
      const hasUpdate = lines.some(line => 
        line.content?.includes('✓ Updated') || 
        line.content?.includes('modified')
      );
      
      if (hasUpdate) {
        console.log('*** Files modified - refreshing preview ***');
        setPreviewKey(k => k + 1); // Force iframe refresh
      }
    });

    return () => {
      unsubscribe();
    };
  }, [isConnected]);

  // Extract session state
  const mode = session?.settings?.mode || "build";
  const model = session?.settings?.model || "auto";
  const autonomy = session?.settings?.autonomy || "standard";
  const messages = session?.messages || [];
  const files = session?.files || [];
  const terminalLines = session?.terminal_lines || [];
  const budget = session?.budget || { spent: 0, total: 5 };
  const progress = session?.progress || { completed: 0, total: 0 };
  
  // Determine preview URL based on project type
  const getPreviewUrl = () => {
    // Check terminal output for server URL
    const serverLine = terminalLines.find(line =>
      line.content?.includes('localhost:3000') ||
      line.content?.includes('localhost:8000')
    );
    if (serverLine) {
      const match = serverLine.content.match(/https?:\/\/localhost:\d+/);
      if (match) return match[0];
    }
    // Default: prefer 3000 for static sites, 8000 for FastAPI
    const hasPackageJson = files.some(f => f.name === 'package.json');
    const hasMainPy = files.some(f => f.name === 'main.py' || f.path?.includes('main.py'));
    if (hasMainPy) return 'http://localhost:8000';
    return 'http://localhost:3000'; // Default to 3000
  };
  
  const previewUrl = getPreviewUrl();

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleNewSession = () => {
    if (window.confirm("Start a new project? Current progress will be saved.")) {
      window.location.reload();
    }
  };

  const handleCopy = async () => {
    const code = fileContent || messages[messages.length - 1]?.content || "";
    if (code) {
      await navigator.clipboard.writeText(code);
      alert("Copied to clipboard!");
    }
  };

  const handleRefresh = () => {
    if (selectedFile && sessionId && sendMessage) {
      sendMessage('file_request', { path: selectedFile, session_id: sessionId });
      alert(`Refreshing ${selectedFile}...`);
    }
  };

  const handleAttach = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        alert(`File selected: ${file.name} (upload not implemented)`);
      }
    };
    input.click();
  };

  const handleURL = () => {
    const url = prompt("Enter URL to analyze:");
    if (url) {
      sendChatMessage(`Analyze this URL: ${url}`);
    }
  };

  const handleCommit = () => {
    alert("Git integration coming soon!");
  };

  const handleScreenshot = () => {
    alert("Screenshot feature coming soon!");
  };

  const handleEnhance = () => {
    if (input.trim()) {
      setInput(`Improve this: ${input.trim()}`);
    }
  };

  const handleSend = () => {
    if (!input.trim()) return;
    sendChatMessage(input.trim());
    setInput("");
  };

  const handleModeChange = (newMode) => {
    updateSettings({ mode: newMode });
  };

  const handleModelChange = (newModel) => {
    updateSettings({ model: newModel });
    setShowModelPicker(false);
  };

  const handleAutonomyChange = (newAutonomy) => {
    updateSettings({ autonomy: newAutonomy });
    setShowAutonomy(false);
  };

  const currentModel = MODELS.find(m => m.id === model);
  const currentAutonomy = AUTONOMY.find(a => a.id === autonomy);

  if (loading) {
    return (
      <div style={{
        width: "100vw", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
        background: DS.bg, color: DS.text,
      }}>
        <Loader2 size={32} color={DS.accent} style={{ animation: "spin 1s linear infinite" }} />
      </div>
    );
  }

  return (
    <div style={{
      width: "100vw", height: "100vh", display: "flex", flexDirection: "column",
      background: DS.bg, color: DS.text, fontFamily: DS.sans,
      overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        @keyframes ping { 0% { transform:scale(1); opacity:.7; } 100% { transform:scale(2.5); opacity:0; } }
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
        ::-webkit-scrollbar { width:5px; height:5px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:${DS.border}; border-radius:3px; }
        ::-webkit-scrollbar-thumb:hover { background:${DS.borderHover}; }
        textarea:focus, input:focus { outline:none; }
      `}</style>

      {/* ═══════════ TOP BAR ═══════════ */}
      <div style={{
        height: 44, flexShrink: 0, display: "flex", alignItems: "center",
        padding: "0 12px", gap: 8,
        background: DS.bgElevated, borderBottom: `1px solid ${DS.border}`,
      }}>
        {/* Logo */}
        <div style={{
          width: 26, height: 26, borderRadius: 6,
          background: `linear-gradient(135deg, ${DS.accent}, #a78bfa)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 13, fontWeight: 800, color: "#fff",
        }}>◆</div>
        <span style={{ fontSize: 13, fontWeight: 700, letterSpacing: "-0.02em", marginRight: 4 }}>AI Orchestrator</span>
        <Badge color={DS.textMuted}>v1.0</Badge>
        {!isConnected && <Badge color={DS.error}>Disconnected</Badge>}

        <Divider vertical />

        {/* Project Name */}
        <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "4px 8px", borderRadius: DS.radius.sm, background: DS.surface }}>
          <Folder size={12} color={DS.warning} />
          <span style={{ fontSize: 12, fontFamily: DS.mono, color: DS.text }}>{session?.project?.name || "Untitled"}</span>
          <ChevronDown size={12} color={DS.textMuted} />
        </div>
        <Btn size="xs" variant="outline" onClick={handleNewSession} title="New Project">
          <RefreshCw size={11} />
          New
        </Btn>

        <Divider vertical />

        {/* Mode Selector */}
        <div style={{ display: "flex", gap: 2 }}>
          {MODES.map(m => (
            <Btn key={m.id} size="xs" active={mode === m.id} onClick={() => handleModeChange(m.id)}>
              <m.icon size={12} />
              {m.name}
            </Btn>
          ))}
        </div>

        <div style={{ flex: 1 }} />

        {/* Model Picker */}
        <div style={{ position: "relative" }}>
          <Btn size="xs" variant="outline" onClick={() => setShowModelPicker(!showModelPicker)}>
            <Cpu size={11} />
            <span style={{ fontFamily: DS.mono }}>{currentModel?.name}</span>
            <ChevronDown size={11} />
          </Btn>
          {showModelPicker && (
            <div style={{
              position: "absolute", top: "100%", right: 0, marginTop: 4,
              background: DS.surface, border: `1px solid ${DS.border}`,
              borderRadius: DS.radius.md, padding: 4, width: 220, zIndex: 100,
              boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
              animation: "fadeIn 0.15s ease-out",
            }}>
              {MODELS.map(m => (
                <div key={m.id} onClick={() => handleModelChange(m.id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "7px 10px", borderRadius: DS.radius.sm,
                    cursor: "pointer", background: model === m.id ? DS.surfaceActive : "transparent",
                    transition: "background 0.1s",
                  }}
                  onMouseEnter={e => { if (model !== m.id) e.currentTarget.style.background = DS.surfaceHover; }}
                  onMouseLeave={e => { if (model !== m.id) e.currentTarget.style.background = "transparent"; }}
                >
                  <span style={{ fontSize: 14, width: 20, textAlign: "center" }}>{m.icon}</span>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 500, color: DS.text }}>{m.name}</div>
                    <div style={{ fontSize: 10, color: DS.textMuted, fontFamily: DS.mono }}>{m.desc}</div>
                  </div>
                  {model === m.id && <Check size={13} color={DS.accent} style={{ marginLeft: "auto" }} />}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Autonomy */}
        <div style={{ position: "relative" }}>
          <Btn size="xs" variant="outline" onClick={() => setShowAutonomy(!showAutonomy)}>
            <Shield size={11} color={currentAutonomy?.color} />
            <span style={{ fontFamily: DS.mono }}>{currentAutonomy?.name}</span>
          </Btn>
          {showAutonomy && (
            <div style={{
              position: "absolute", top: "100%", right: 0, marginTop: 4,
              background: DS.surface, border: `1px solid ${DS.border}`,
              borderRadius: DS.radius.md, padding: 4, width: 240, zIndex: 100,
              boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
              animation: "fadeIn 0.15s ease-out",
            }}>
              {AUTONOMY.map(a => (
                <div key={a.id} onClick={() => handleAutonomyChange(a.id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "7px 10px", borderRadius: DS.radius.sm,
                    cursor: "pointer", background: autonomy === a.id ? DS.surfaceActive : "transparent",
                  }}
                  onMouseEnter={e => { if (autonomy !== a.id) e.currentTarget.style.background = DS.surfaceHover; }}
                  onMouseLeave={e => { if (autonomy !== a.id) e.currentTarget.style.background = "transparent"; }}
                >
                  <Dot color={a.color} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, color: DS.text }}>{a.name}</div>
                    <div style={{ fontSize: 10, color: DS.textMuted, fontFamily: DS.mono }}>{a.desc}</div>
                  </div>
                  {autonomy === a.id && <Check size={13} color={DS.accent} />}
                </div>
              ))}
            </div>
          )}
        </div>

        <Divider vertical />

        {/* Actions */}
        <Btn size="xs" variant="outline"><GitBranch size={12} /> main</Btn>
        <Btn size="sm" variant="filled" style={{ gap: 6 }}><Play size={12} /> Deploy</Btn>
      </div>

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* ──── LEFT: Chat Panel ──── */}
        <div style={{
          width: 420, flexShrink: 0, display: "flex", flexDirection: "column",
          borderRight: `1px solid ${DS.border}`,
          background: DS.bgElevated,
        }}>
          {/* Chat Header */}
          <div style={{
            padding: "8px 12px", display: "flex", alignItems: "center", justifyContent: "space-between",
            borderBottom: `1px solid ${DS.border}`,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Dot color={isConnected ? DS.success : DS.error} pulse={isConnected} />
              <span style={{ fontSize: 12, fontWeight: 600 }}>Chat</span>
              <Badge color={DS.textMuted}>{messages.length} messages</Badge>
            </div>
            <div style={{ display: "flex", gap: 2 }}>
              <Btn size="xs"><RotateCcw size={11} /> New</Btn>
            </div>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
            {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}
            <div ref={chatEndRef} />
          </div>

          {/* Input Area */}
          <div style={{
            padding: 10, borderTop: `1px solid ${DS.border}`,
            background: DS.bg,
          }}>
            {/* Context Pills */}
            <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
              <Btn size="xs" variant="outline" onClick={handleAttach}><Paperclip size={10} /> Attach</Btn>
              <Btn size="xs" variant="outline" onClick={handleURL}><Globe size={10} /> URL</Btn>
              <Btn size="xs" variant="outline" onClick={handleCommit}><GitBranch size={10} /> Commit</Btn>
              <Btn size="xs" variant="outline" onClick={handleScreenshot}><Camera size={10} /> Screenshot</Btn>
              <Btn size="xs" variant="outline" onClick={handleEnhance}><Sparkles size={10} /> Enhance</Btn>
            </div>

            {/* Text Input */}
            <div style={{
              display: "flex", alignItems: "flex-end", gap: 6,
              background: DS.surface, border: `1px solid ${DS.border}`,
              borderRadius: DS.radius.md, padding: "8px 10px",
              transition: "border-color 0.15s",
            }}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder={mode === "plan" ? "Describe what you want to plan..." : mode === "chat" ? "Ask a question..." : "Describe what to build..."}
                rows={2}
                style={{
                  flex: 1, background: "transparent", border: "none",
                  color: DS.text, fontSize: 13, fontFamily: DS.sans,
                  resize: "none", lineHeight: 1.5,
                }}
              />
              <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
                <Btn size="xs" style={{ color: DS.textMuted }}><Mic size={13} /></Btn>
                <Btn size="xs" variant="filled" onClick={handleSend}
                  style={{ opacity: input.trim() ? 1 : 0.4 }}
                ><Send size={12} /></Btn>
              </div>
            </div>

            {/* Footer hints */}
            <div style={{
              display: "flex", justifyContent: "space-between", marginTop: 6,
              fontSize: 10, color: DS.textDim, fontFamily: DS.mono,
            }}>
              <span><Kbd>Enter</Kbd> send · <Kbd>Shift+Enter</Kbd> newline</span>
              <span>${budget.spent?.toFixed(2)} / ${budget.total?.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* ──── MIDDLE: File Tree ──── */}
        {showFiles && (
          <div style={{
            width: 200, flexShrink: 0, display: "flex", flexDirection: "column",
            borderRight: `1px solid ${DS.border}`,
            background: DS.bg,
          }}>
            <div style={{
              padding: "8px 10px", display: "flex", alignItems: "center", justifyContent: "space-between",
              borderBottom: `1px solid ${DS.border}`,
            }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: DS.textSecondary, letterSpacing: "0.04em", textTransform: "uppercase" }}>Files</span>
              <Btn size="xs" onClick={() => setShowFiles(false)}><PanelLeftClose size={12} /></Btn>
            </div>
            <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
              {files.map((item, i) => (
                <FileTreeItem key={i} item={item} selected={selectedFile} onSelect={setSelectedFile} />
              ))}
            </div>
          </div>
        )}

        {/* ──── RIGHT: Preview / Code ──── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Right Panel Header */}
          <div style={{
            height: 38, display: "flex", alignItems: "center", gap: 4,
            padding: "0 10px", borderBottom: `1px solid ${DS.border}`,
            background: DS.bgElevated,
          }}>
            {!showFiles && (
              <Btn size="xs" onClick={() => setShowFiles(true)}><PanelLeft size={12} /></Btn>
            )}
            <Btn size="xs" active={rightPanel === "preview"} onClick={() => setRightPanel("preview")}>
              <Eye size={12} /> Preview
            </Btn>
            <Btn size="xs" active={rightPanel === "code"} onClick={() => setRightPanel("code")}>
              <Code2 size={12} /> Code
            </Btn>

            {rightPanel === "preview" && (
              <>
                <Divider vertical />
                <div style={{
                  flex: 1, display: "flex", alignItems: "center", gap: 6,
                  background: DS.surface, borderRadius: 5, padding: "3px 10px",
                  marginLeft: 4, marginRight: 4,
                }}>
                  <Globe size={11} color={DS.textMuted} />
                  <span style={{ fontSize: 11, fontFamily: DS.mono, color: DS.textSecondary }}>
                    localhost:8765
                  </span>
                  <Dot color={isConnected ? DS.success : DS.error} />
                </div>
              </>
            )}

            {rightPanel === "code" && selectedFile && (
              <>
                <Divider vertical />
                <div style={{ display: "flex", alignItems: "center", gap: 4, flex: 1 }}>
                  <FileCode size={12} color={DS.accent} />
                  <span style={{ fontSize: 12, fontFamily: DS.mono, color: DS.text }}>{selectedFile}</span>
                </div>
              </>
            )}

            <div style={{ display: "flex", gap: 2 }}>
              <Btn size="xs" onClick={handleRefresh} title="Refresh file"><RefreshCw size={11} /></Btn>
              <Btn size="xs" onClick={() => selectedFile && window.open(`http://localhost:3000/${selectedFile}`, '_blank')} title="Open in new tab"><ExternalLink size={11} /></Btn>
              <Btn size="xs" onClick={handleCopy} title="Copy to clipboard"><Copy size={11} /></Btn>
            </div>
          </div>

          {/* Content Area */}
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            {rightPanel === "preview" ? (
              <div style={{
                flex: 1, display: "flex", flexDirection: "column",
                background: "#fafafa", position: "relative",
              }}>
                {/* Preview URL bar */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 12px", background: "white",
                  borderBottom: `1px solid ${DS.border}`,
                }}>
                  <Globe size={14} color={DS.textMuted} />
                  <input
                    type="text"
                    value={previewUrl}
                    readOnly
                    style={{
                      flex: 1, padding: "4px 8px",
                      border: `1px solid ${DS.border}`,
                      borderRadius: 4, fontSize: 12,
                      fontFamily: DS.mono, color: DS.text,
                    }}
                  />
                  <button
                    onClick={() => window.open(previewUrl, '_blank')}
                    style={{
                      padding: "4px 8px", background: DS.accent,
                      color: "white", border: "none", borderRadius: 4,
                      cursor: "pointer", fontSize: 12,
                    }}
                  >
                    <ExternalLink size={12} />
                  </button>
                </div>
                
                {/* Website Preview iframe */}
                <iframe
                  key={previewKey}
                  src={previewUrl}
                  style={{
                    flex: 1, border: "none", width: "100%", height: "100%",
                    background: "white",
                  }}
                  title="Website Preview"
                  sandbox="allow-scripts allow-same-origin allow-forms"
                  onLoad={() => console.log('Preview loaded:', previewUrl)}
                  onError={() => console.error('Preview error:', previewUrl)}
                />
              </div>
            ) : (
              <div style={{
                flex: 1, overflow: "auto", padding: "12px 0",
                background: DS.bg, fontFamily: DS.mono, fontSize: 12.5,
                lineHeight: 1.7, counterReset: "line",
              }}>
                {/* Debug info */}
                <div style={{ padding: "16px", color: DS.info, borderBottom: `1px solid ${DS.border}` }}>
                  <div>Selected: {selectedFile || 'none'}</div>
                  <div>File content length: {fileContent?.length || 0}</div>
                  <div>Session: {sessionId}</div>
                </div>
                
                {fileContent ? (
                  fileContent.split("\n").map((line, i) => (
                    <div key={i} style={{
                      display: "flex", padding: "0 16px",
                      background: i % 2 === 0 ? "transparent" : `${DS.text}02`,
                    }}>
                      <span style={{
                        width: 40, flexShrink: 0, textAlign: "right",
                        color: DS.textDim, paddingRight: 16,
                        userSelect: "none", fontSize: 11,
                      }}>{i + 1}</span>
                      <span style={{
                        color: line.startsWith("#") || line.startsWith('"""') || line.startsWith("'''") ? DS.textMuted
                          : line.includes("def ") || line.includes("class ") || line.includes("import ") || line.includes("from ") ? DS.accent
                          : line.includes("return ") ? DS.warning
                          : line.includes('"') || line.includes("'") ? DS.success
                          : DS.textSecondary,
                        whiteSpace: "pre",
                      }}>{line || " "}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ padding: "16px", color: DS.textMuted, textAlign: "center" }}>
                    Select a file from the tree to view its content
                  </div>
                )}
              </div>
            )}

            {/* ── Terminal ── */}
            {showTerminal && (
              <div style={{
                height: 180, flexShrink: 0,
                borderTop: `1px solid ${DS.border}`,
                display: "flex", flexDirection: "column",
                background: DS.bg,
              }}>
                <div style={{
                  height: 30, display: "flex", alignItems: "center", gap: 6,
                  padding: "0 10px", borderBottom: `1px solid ${DS.border}`,
                  background: DS.bgElevated,
                }}>
                  <Terminal size={12} color={DS.textMuted} />
                  <span style={{ fontSize: 11, fontWeight: 600, color: DS.textSecondary }}>Terminal</span>
                  <Badge color={DS.success}>Ready</Badge>
                  <div style={{ flex: 1 }} />
                  <Btn size="xs" onClick={() => setShowTerminal(false)}><ChevronDown size={11} /></Btn>
                </div>
                <div style={{ flex: 1, overflow: "auto", padding: "6px 12px", fontFamily: DS.mono, fontSize: 11.5, lineHeight: 1.7 }}>
                  {terminalLines.length > 0 ? (
                    terminalLines.map((line, i) => (
                      <div key={i} style={{
                        color: line.type === "cmd" ? DS.text
                          : line.type === "success" ? DS.success
                          : line.type === "error" ? DS.error
                          : line.type === "info" ? DS.info
                          : DS.textSecondary,
                        fontWeight: line.type === "cmd" || line.type === "success" ? 500 : 400,
                      }}>{line.text}</div>
                    ))
                  ) : (
                    <div style={{ color: DS.textMuted }}>Terminal ready. Commands will appear here...</div>
                  )}
                </div>
              </div>
            )}

            {!showTerminal && (
              <div style={{
                height: 30, display: "flex", alignItems: "center", gap: 6,
                padding: "0 10px", borderTop: `1px solid ${DS.border}`,
                background: DS.bgElevated, cursor: "pointer",
              }} onClick={() => setShowTerminal(true)}>
                <Terminal size={12} color={DS.textMuted} />
                <span style={{ fontSize: 11, color: DS.textSecondary }}>Terminal</span>
                <Badge color={DS.success}>Ready</Badge>
                <div style={{ flex: 1 }} />
                <ChevronUp size={11} color={DS.textMuted} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ═══════════ BOTTOM STATUS BAR ═══════════ */}
      <div style={{
        height: 24, flexShrink: 0, display: "flex", alignItems: "center",
        padding: "0 10px", gap: 12,
        background: DS.bgElevated, borderTop: `1px solid ${DS.border}`,
        fontSize: 10, fontFamily: DS.mono, color: DS.textMuted,
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <Dot color={isConnected ? DS.success : DS.error} /> {isConnected ? "Connected" : "Disconnected"}
        </span>
        <span>Mode: {mode}</span>
        <span>Model: {currentModel?.name}</span>
        <span>Autonomy: {currentAutonomy?.name}</span>
        <Divider vertical />
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <DollarSign size={10} /> ${budget.spent?.toFixed(2)} / ${budget.total?.toFixed(2)}
        </span>
        <span>Tasks: {progress.completed_tasks}/{progress.total_tasks}</span>
        <span>Quality: {(session?.metrics?.quality_score || 0).toFixed(2)}</span>
        <div style={{ flex: 1 }} />
        <span>Session: {sessionId}</span>
      </div>

      {/* Click-away for dropdowns */}
      {(showModelPicker || showAutonomy) && (
        <div style={{ position: "fixed", inset: 0, zIndex: 50 }}
          onClick={() => { setShowModelPicker(false); setShowAutonomy(false); }} />
      )}
    </div>
  );
}
