import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadialBarChart, RadialBar } from "recharts";
import * as d3 from "d3";

// ─── Design Tokens ───
const T = {
  bg: "#0a0b0f",
  surface: "#12131a",
  surfaceHover: "#1a1b24",
  card: "#15161f",
  cardHover: "#1c1d28",
  border: "#1e2030",
  borderActive: "#2a2d42",
  text: "#e2e4ed",
  textSecondary: "#6b7089",
  textMuted: "#444766",
  accent: "#6c5ce7",
  accentGlow: "rgba(108,92,231,0.15)",
  success: "#00d68f",
  successGlow: "rgba(0,214,143,0.12)",
  warning: "#ffaa00",
  warningGlow: "rgba(255,170,0,0.10)",
  error: "#ff4757",
  errorGlow: "rgba(255,71,87,0.10)",
  info: "#0abde3",
  infoGlow: "rgba(10,189,227,0.10)",
  providers: {
    anthropic: "#d4a574",
    openai: "#74b9ff",
    google: "#55efc4",
    deepseek: "#fd79a8",
    minimax: "#a29bfe",
    moonshot: "#fdcb6e",
  },
  font: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
  fontSans: "'DM Sans', -apple-system, sans-serif",
};

// ─── Simulated Live Data ───
const genTimeline = () => Array.from({ length: 24 }, (_, i) => ({
  hour: `${String(i).padStart(2, "0")}:00`,
  cost: +(Math.random() * 0.4 + 0.05).toFixed(3),
  tasks: Math.floor(Math.random() * 8 + 1),
  quality: +(Math.random() * 0.15 + 0.82).toFixed(3),
  latency: Math.floor(Math.random() * 3000 + 800),
}));

const genModelCost = () => [
  { name: "DeepSeek V3.2", cost: 0.42, tasks: 34, quality: 0.91, color: T.providers.deepseek },
  { name: "Claude Sonnet 4.6", cost: 1.85, tasks: 18, quality: 0.94, color: T.providers.anthropic },
  { name: "GPT-4o-mini", cost: 0.38, tasks: 22, quality: 0.87, color: T.providers.openai },
  { name: "Gemini Flash", cost: 0.12, tasks: 15, quality: 0.85, color: T.providers.google },
  { name: "Kimi K2.5", cost: 0.09, tasks: 8, quality: 0.83, color: T.providers.moonshot },
];

const genTasks = () => [
  { id: "task_001", name: "Auth Module", status: "completed", model: "DeepSeek V3.2", score: 0.94, cost: 0.08, time: 12.4, repairs: 0 },
  { id: "task_002", name: "Rate Limiter", status: "completed", model: "Claude Sonnet", score: 0.97, cost: 0.22, time: 18.7, repairs: 1 },
  { id: "task_003", name: "DB Schema", status: "running", model: "DeepSeek V3.2", score: null, cost: 0.04, time: 6.2, repairs: 0 },
  { id: "task_004", name: "API Routes", status: "queued", model: "GPT-4o-mini", score: null, cost: 0, time: 0, repairs: 0 },
  { id: "task_005", name: "WebSocket Handler", status: "queued", model: "DeepSeek V3.2", score: null, cost: 0, time: 0, repairs: 0 },
  { id: "task_006", name: "Test Suite", status: "queued", model: "Claude Sonnet", score: null, cost: 0, time: 0, repairs: 0 },
  { id: "task_007", name: "Error Handling", status: "completed", model: "Gemini Flash", score: 0.89, cost: 0.03, time: 4.1, repairs: 0 },
  { id: "task_008", name: "Middleware", status: "completed", model: "DeepSeek V3.2", score: 0.92, cost: 0.06, time: 8.3, repairs: 0 },
];

const genAgentLog = () => [
  { ts: "14:32:07", agent: "Supervisor", action: "Delegated task_003 to Specialist-DB", level: "info" },
  { ts: "14:32:05", agent: "Verifier", action: "task_002 passed UNIT verification (17/17 tests)", level: "success" },
  { ts: "14:31:58", agent: "Chairman", action: "Selected DeepSeek output (score: 0.94 vs 0.87)", level: "info" },
  { ts: "14:31:42", agent: "Specialist-API", action: "Self-healing: repaired NameError in auth.py", level: "warning" },
  { ts: "14:31:30", agent: "Router", action: "Budget pressure: downgraded to BUDGET tier", level: "warning" },
  { ts: "14:31:15", agent: "Circuit Breaker", action: "moonshot-v1 → OPEN (3 consecutive timeouts)", level: "error" },
  { ts: "14:31:02", agent: "Memory Bank", action: "Saved 3 architectural decisions from run", level: "info" },
  { ts: "14:30:48", agent: "Condenser", action: "Context compressed: 12,400 → 3,200 tokens (74%)", level: "info" },
];

const genCircuitBreakers = () => [
  { provider: "Anthropic", state: "CLOSED", failures: 0, lastSuccess: "2s ago", health: 100 },
  { provider: "OpenAI", state: "CLOSED", failures: 1, lastSuccess: "8s ago", health: 95 },
  { provider: "Google", state: "CLOSED", failures: 0, lastSuccess: "4s ago", health: 100 },
  { provider: "DeepSeek", state: "CLOSED", failures: 0, lastSuccess: "1s ago", health: 100 },
  { provider: "Moonshot", state: "OPEN", failures: 3, lastSuccess: "4m ago", health: 0 },
  { provider: "MiniMax", state: "HALF_OPEN", failures: 2, lastSuccess: "62s ago", health: 40 },
];

// ─── Micro Components ───
const Glow = ({ color, size = 120, style }) => (
  <div style={{
    position: "absolute", width: size, height: size,
    borderRadius: "50%", background: color, filter: `blur(${size / 2}px)`,
    opacity: 0.4, pointerEvents: "none", ...style,
  }} />
);

const Badge = ({ children, color = T.textSecondary, bg }) => (
  <span style={{
    display: "inline-flex", alignItems: "center", gap: 4,
    padding: "2px 8px", borderRadius: 4, fontSize: 11,
    fontFamily: T.font, fontWeight: 500, letterSpacing: "0.02em",
    color, background: bg || `${color}15`,
    border: `1px solid ${color}25`,
  }}>{children}</span>
);

const StatusDot = ({ status }) => {
  const colors = { completed: T.success, running: T.info, queued: T.textMuted, failed: T.error, blocked: T.warning };
  const c = colors[status] || T.textMuted;
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center", justifyContent: "center", width: 10, height: 10 }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: c }} />
      {status === "running" && <span style={{
        position: "absolute", width: 10, height: 10, borderRadius: "50%",
        border: `1.5px solid ${c}`, animation: "ping 1.5s ease-in-out infinite", opacity: 0.6,
      }} />}
    </span>
  );
};

const KPICard = ({ label, value, sub, trend, icon, color = T.accent }) => (
  <div style={{
    position: "relative", overflow: "hidden",
    background: T.card, border: `1px solid ${T.border}`,
    borderRadius: 10, padding: "16px 18px", flex: 1, minWidth: 170,
    transition: "border-color 0.2s, box-shadow 0.2s",
  }}
    onMouseEnter={e => { e.currentTarget.style.borderColor = `${color}40`; e.currentTarget.style.boxShadow = `0 0 20px ${color}08`; }}
    onMouseLeave={e => { e.currentTarget.style.borderColor = T.border; e.currentTarget.style.boxShadow = "none"; }}
  >
    <Glow color={color} size={80} style={{ top: -30, right: -30, opacity: 0.15 }} />
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
      <span style={{ fontSize: 11, fontFamily: T.font, color: T.textSecondary, letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</span>
      <span style={{ fontSize: 16, opacity: 0.6 }}>{icon}</span>
    </div>
    <div style={{ fontSize: 26, fontWeight: 700, fontFamily: T.font, color: T.text, letterSpacing: "-0.02em" }}>{value}</div>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 }}>
      <span style={{ fontSize: 11, fontFamily: T.font, color: T.textMuted }}>{sub}</span>
      {trend && <Badge color={trend > 0 ? T.success : T.error}>{trend > 0 ? "↑" : "↓"} {Math.abs(trend)}%</Badge>}
    </div>
  </div>
);

const SectionHeader = ({ title, subtitle, action }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
    <div>
      <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, fontFamily: T.fontSans, color: T.text, letterSpacing: "-0.01em" }}>{title}</h3>
      {subtitle && <span style={{ fontSize: 11, fontFamily: T.font, color: T.textMuted, marginTop: 2, display: "block" }}>{subtitle}</span>}
    </div>
    {action}
  </div>
);

const Card = ({ children, style, glow }) => (
  <div style={{
    position: "relative", overflow: "hidden",
    background: T.card, border: `1px solid ${T.border}`, borderRadius: 12,
    padding: 18, ...style,
  }}>
    {glow && <Glow color={glow} size={100} style={{ bottom: -40, left: -40, opacity: 0.12 }} />}
    {children}
  </div>
);

const Tab = ({ active, children, onClick }) => (
  <button onClick={onClick} style={{
    background: active ? `${T.accent}15` : "transparent",
    border: `1px solid ${active ? T.accent + "40" : "transparent"}`,
    color: active ? T.accent : T.textSecondary,
    borderRadius: 6, padding: "5px 12px", fontSize: 12,
    fontFamily: T.font, fontWeight: 500, cursor: "pointer",
    transition: "all 0.2s",
  }}>{children}</button>
);

// ─── Custom Tooltip ───
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 8, padding: "10px 14px", fontSize: 12,
      fontFamily: T.font, boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    }}>
      <div style={{ color: T.textSecondary, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || T.text, display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: p.color }} />
          <span>{p.name}:</span>
          <span style={{ fontWeight: 600 }}>{typeof p.value === "number" && p.name?.includes("cost") ? `$${p.value.toFixed(3)}` : p.value}</span>
        </div>
      ))}
    </div>
  );
};

// ─── Agent Hierarchy Viz (D3 Tree) ───
const AgentTree = () => {
  const ref = useRef();
  useEffect(() => {
    if (!ref.current) return;
    const w = ref.current.offsetWidth, h = 180;
    d3.select(ref.current).selectAll("*").remove();
    const svg = d3.select(ref.current).append("svg").attr("width", w).attr("height", h);

    const data = {
      name: "Supervisor", role: "reasoning", children: [
        { name: "Specialist-API", role: "code", children: [{ name: "task_001", role: "task" }, { name: "task_004", role: "task" }] },
        { name: "Specialist-DB", role: "code", children: [{ name: "task_003", role: "task" }, { name: "task_005", role: "task" }] },
        { name: "Reviewer", role: "review", children: [{ name: "task_002", role: "task" }, { name: "task_006", role: "task" }] },
      ],
    };

    const root = d3.hierarchy(data);
    const treeLayout = d3.tree().size([w - 60, h - 60]);
    treeLayout(root);

    const roleColor = { reasoning: T.accent, code: T.info, review: T.warning, task: T.textMuted };

    svg.selectAll("line").data(root.links()).join("line")
      .attr("x1", d => d.source.x + 30).attr("y1", d => d.source.y + 25)
      .attr("x2", d => d.target.x + 30).attr("y2", d => d.target.y + 25)
      .attr("stroke", T.border).attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

    const nodes = svg.selectAll("g").data(root.descendants()).join("g")
      .attr("transform", d => `translate(${d.x + 30},${d.y + 25})`);

    nodes.append("circle")
      .attr("r", d => d.data.role === "task" ? 4 : 8)
      .attr("fill", d => roleColor[d.data.role] || T.textMuted)
      .attr("stroke", d => `${roleColor[d.data.role]}40`).attr("stroke-width", 3);

    nodes.filter(d => d.data.role !== "task").append("text")
      .attr("dy", -14).attr("text-anchor", "middle")
      .attr("fill", T.textSecondary).attr("font-size", 10)
      .attr("font-family", T.font).text(d => d.data.name);
  }, []);

  return <div ref={ref} style={{ width: "100%", minHeight: 180 }} />;
};

// ─── Main Dashboard ───
export default function OrchestratorDashboard() {
  const [timeline] = useState(genTimeline);
  const [models] = useState(genModelCost);
  const [tasks] = useState(genTasks);
  const [logs] = useState(genAgentLog);
  const [breakers] = useState(genCircuitBreakers);
  const [activeTab, setActiveTab] = useState("overview");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const iv = setInterval(() => setTick(t => t + 1), 3000);
    return () => clearInterval(iv);
  }, []);

  const totalCost = models.reduce((s, m) => s + m.cost, 0);
  const totalTasks = tasks.length;
  const completedTasks = tasks.filter(t => t.status === "completed").length;
  const avgQuality = tasks.filter(t => t.score).reduce((s, t, _, a) => s + t.score / a.length, 0);
  const runningTask = tasks.find(t => t.status === "running");

  const costByModel = models.map(m => ({ name: m.name.split(" ")[0], value: m.cost, color: m.color }));

  return (
    <div style={{
      minHeight: "100vh", background: T.bg, color: T.text,
      fontFamily: T.fontSans, position: "relative", overflow: "hidden",
    }}>
      {/* Background effects */}
      <Glow color={T.accent} size={400} style={{ top: -200, left: -100 }} />
      <Glow color={T.info} size={300} style={{ bottom: -100, right: -50 }} />

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        @keyframes ping { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(2.5); opacity: 0; } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${T.bg}; }
        ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: ${T.borderActive}; }
      `}</style>

      {/* ═══ Top Bar ═══ */}
      <div style={{
        position: "sticky", top: 0, zIndex: 50,
        background: `${T.bg}e6`, backdropFilter: "blur(16px)",
        borderBottom: `1px solid ${T.border}`,
        padding: "12px 24px", display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: `linear-gradient(135deg, ${T.accent}, ${T.info})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 700, color: "#fff",
          }}>◆</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.02em" }}>AI Orchestrator</div>
            <div style={{ fontSize: 10, fontFamily: T.font, color: T.textMuted, letterSpacing: "0.04em" }}>v6.2 · MISSION CONTROL</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {["overview", "tasks", "agents", "cost"].map(tab => (
            <Tab key={tab} active={activeTab === tab} onClick={() => setActiveTab(tab)}>
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </Tab>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: T.success, animation: "pulse 2s infinite" }} />
            <span style={{ fontSize: 11, fontFamily: T.font, color: T.textSecondary }}>LIVE</span>
          </div>
          <Badge color={T.success}>Autonomy: STANDARD</Badge>
        </div>
      </div>

      {/* ═══ Content ═══ */}
      <div style={{ padding: "20px 24px", maxWidth: 1440, margin: "0 auto" }}>

        {/* ─── KPI Row ─── */}
        <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap", animation: "slideIn 0.3s ease-out" }}>
          <KPICard label="Budget Spent" value={`$${totalCost.toFixed(2)}`} sub="of $5.00 budget" trend={-22} icon="💰" color={T.success} />
          <KPICard label="Tasks" value={`${completedTasks}/${totalTasks}`} sub={runningTask ? `Running: ${runningTask.name}` : "All complete"} trend={12} icon="⚡" color={T.info} />
          <KPICard label="Avg Quality" value={avgQuality.toFixed(3)} sub="target: 0.850" trend={8} icon="◆" color={T.accent} />
          <KPICard label="Repair Cycles" value="1" sub="3 available (STANDARD)" icon="🔧" color={T.warning} />
          <KPICard label="Cache Hit Rate" value="73%" sub="semantic + exact" trend={15} icon="⚡" color={T.success} />
        </div>

        {/* ─── Main Grid ─── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 340px", gap: 14, marginBottom: 14 }}>

          {/* Cost Timeline */}
          <Card glow={T.accent}>
            <SectionHeader title="Cost & Quality Timeline" subtitle="24h rolling window" />
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={timeline}>
                <defs>
                  <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={T.accent} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={T.accent} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="qualGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={T.success} stopOpacity={0.2} />
                    <stop offset="100%" stopColor={T.success} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="hour" tick={{ fill: T.textMuted, fontSize: 10, fontFamily: T.font }} tickLine={false} axisLine={false} interval={3} />
                <YAxis tick={{ fill: T.textMuted, fontSize: 10, fontFamily: T.font }} tickLine={false} axisLine={false} width={35} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="cost" stroke={T.accent} fill="url(#costGrad)" strokeWidth={2} name="cost ($)" dot={false} />
                <Area type="monotone" dataKey="quality" stroke={T.success} fill="url(#qualGrad)" strokeWidth={1.5} name="quality" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          {/* Model Distribution */}
          <Card glow={T.info}>
            <SectionHeader title="Cost by Model" subtitle={`Total: $${totalCost.toFixed(2)}`} />
            <div style={{ display: "flex", gap: 12, height: 200 }}>
              <ResponsiveContainer width="45%" height="100%">
                <PieChart>
                  <Pie data={costByModel} cx="50%" cy="50%" innerRadius={40} outerRadius={65}
                    dataKey="value" stroke={T.bg} strokeWidth={2}>
                    {costByModel.map((m, i) => <Cell key={i} fill={m.color} />)}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", gap: 8 }}>
                {models.map((m, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, fontFamily: T.font }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: m.color, flexShrink: 0 }} />
                    <span style={{ color: T.textSecondary, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.name}</span>
                    <span style={{ color: T.text, fontWeight: 600 }}>${m.cost.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Agent Activity Log */}
          <Card style={{ maxHeight: 300, display: "flex", flexDirection: "column" }}>
            <SectionHeader title="Agent Activity" subtitle={`${logs.length} events`} />
            <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
              {logs.map((log, i) => {
                const levelColors = { info: T.textSecondary, success: T.success, warning: T.warning, error: T.error };
                return (
                  <div key={i} style={{
                    display: "flex", gap: 8, padding: "6px 8px",
                    borderRadius: 6, background: i === 0 ? `${T.surfaceHover}` : "transparent",
                    borderLeft: `2px solid ${levelColors[log.level]}`,
                    animation: i === 0 ? "slideIn 0.4s ease-out" : undefined,
                  }}>
                    <span style={{ fontSize: 10, fontFamily: T.font, color: T.textMuted, flexShrink: 0, paddingTop: 1 }}>{log.ts}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <span style={{ fontSize: 10, fontFamily: T.font, fontWeight: 600, color: levelColors[log.level] }}>{log.agent}</span>
                      <div style={{ fontSize: 11, fontFamily: T.font, color: T.textSecondary, lineHeight: 1.4, marginTop: 1 }}>{log.action}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* ─── Second Row ─── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>

          {/* Task Execution Table */}
          <Card>
            <SectionHeader title="Task Execution" subtitle={`${completedTasks} completed · ${tasks.filter(t => t.status === "running").length} running`} />
            <div style={{ overflow: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: T.font }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                    {["Status", "Task", "Model", "Score", "Cost", "Time", "Repairs"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: T.textMuted, fontWeight: 500, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tasks.map((task, i) => (
                    <tr key={task.id} style={{
                      borderBottom: `1px solid ${T.border}08`,
                      background: task.status === "running" ? `${T.info}08` : "transparent",
                      animation: task.status === "running" ? "pulse 3s infinite" : undefined,
                    }}>
                      <td style={{ padding: "8px 10px" }}><StatusDot status={task.status} /></td>
                      <td style={{ padding: "8px 10px", color: T.text, fontWeight: 500 }}>{task.name}</td>
                      <td style={{ padding: "8px 10px", color: T.textSecondary }}>{task.model}</td>
                      <td style={{ padding: "8px 10px" }}>
                        {task.score ? (
                          <span style={{ color: task.score >= 0.9 ? T.success : task.score >= 0.8 ? T.warning : T.error, fontWeight: 600 }}>
                            {task.score.toFixed(3)}
                          </span>
                        ) : <span style={{ color: T.textMuted }}>—</span>}
                      </td>
                      <td style={{ padding: "8px 10px", color: T.textSecondary }}>${task.cost.toFixed(3)}</td>
                      <td style={{ padding: "8px 10px", color: T.textSecondary }}>{task.time > 0 ? `${task.time.toFixed(1)}s` : "—"}</td>
                      <td style={{ padding: "8px 10px" }}>
                        {task.repairs > 0 ? <Badge color={T.warning}>🔧 {task.repairs}</Badge> : <span style={{ color: T.textMuted }}>0</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Agent Hierarchy + Circuit Breakers */}
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card glow={T.accent}>
              <SectionHeader title="Agent Hierarchy" subtitle="Supervisor → Specialists → Tasks" />
              <AgentTree />
            </Card>

            <Card>
              <SectionHeader title="Circuit Breakers" subtitle="Provider health status" />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {breakers.map((b, i) => {
                  const stateColor = b.state === "CLOSED" ? T.success : b.state === "OPEN" ? T.error : T.warning;
                  return (
                    <div key={i} style={{
                      padding: "10px 12px", borderRadius: 8,
                      background: `${stateColor}08`, border: `1px solid ${stateColor}20`,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                        <span style={{ fontSize: 11, fontFamily: T.font, fontWeight: 600, color: T.text }}>{b.provider}</span>
                        <Badge color={stateColor}>{b.state}</Badge>
                      </div>
                      <div style={{ fontSize: 10, fontFamily: T.font, color: T.textMuted }}>
                        Failures: {b.failures} · {b.lastSuccess}
                      </div>
                      {/* Health bar */}
                      <div style={{ marginTop: 6, height: 3, borderRadius: 2, background: `${T.border}`, overflow: "hidden" }}>
                        <div style={{ width: `${b.health}%`, height: "100%", borderRadius: 2, background: stateColor, transition: "width 0.5s" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </div>

        {/* ─── Bottom Row: Latency + ARA Methods ─── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <Card>
            <SectionHeader title="Latency Distribution" subtitle="Per model response time (ms)" />
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={models} barSize={20}>
                <XAxis dataKey="name" tick={{ fill: T.textMuted, fontSize: 9, fontFamily: T.font }} tickLine={false} axisLine={false}
                  tickFormatter={v => v.split(" ")[0]} />
                <YAxis tick={{ fill: T.textMuted, fontSize: 10, fontFamily: T.font }} tickLine={false} axisLine={false} width={40} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="tasks" name="Tasks executed" radius={[4, 4, 0, 0]}>
                  {models.map((m, i) => <Cell key={i} fill={m.color} fillOpacity={0.8} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card>
            <SectionHeader title="ARA Pipeline Methods" subtitle="Usage across current project" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                { name: "Multi-Perspective", uses: 4, score: 0.91, color: T.accent },
                { name: "Iterative", uses: 3, score: 0.88, color: T.info },
                { name: "Pre-Mortem", uses: 2, score: 0.94, color: T.error },
                { name: "Debate", uses: 2, score: 0.90, color: T.warning },
                { name: "Scientific", uses: 1, score: 0.87, color: T.success },
                { name: "Jury", uses: 1, score: 0.96, color: T.providers.deepseek },
              ].map((m, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "8px 10px", borderRadius: 6, background: `${m.color}08`,
                  border: `1px solid ${m.color}15`,
                }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: 6,
                    background: `${m.color}20`, display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 12, fontFamily: T.font, fontWeight: 700, color: m.color,
                  }}>{m.uses}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, fontFamily: T.font, fontWeight: 500, color: T.text }}>{m.name}</div>
                    <div style={{ fontSize: 10, fontFamily: T.font, color: T.textMuted }}>avg: {m.score.toFixed(3)}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ─── Footer ─── */}
        <div style={{
          marginTop: 20, padding: "12px 0",
          borderTop: `1px solid ${T.border}`,
          display: "flex", justifyContent: "space-between",
          fontSize: 10, fontFamily: T.font, color: T.textMuted,
        }}>
          <span>AI Orchestrator v6.2 · {completedTasks}/{totalTasks} tasks · ${totalCost.toFixed(2)} spent · Cache: 73% hit rate</span>
          <span>Last update: {tick * 3}s ago · Tick #{tick}</span>
        </div>
      </div>
    </div>
  );
}
