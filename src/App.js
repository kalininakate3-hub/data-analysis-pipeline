import { useState, useEffect, useRef, useCallback } from "react";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────

const MODELS = [
  { id: "claude-haiku-4-5-20251001", label: "Haiku 4.5 — Fastest" },
  { id: "claude-sonnet-4-6", label: "Sonnet 4.6 — Balanced" },
  { id: "claude-opus-4-6", label: "Opus 4.6 — Most Capable" },
];

const AGENT_CONFIG = [
  { key: "agent1", label: "Analyst", emoji: "📊", desc: "Runs Python on your data" },
  { key: "agent2", label: "Hypothesis", emoji: "💡", desc: "Generates hypotheses from findings" },
  { key: "agent3", label: "Critic", emoji: "🔍", desc: "Validates hypotheses against data" },
  { key: "agent4", label: "Root Cause", emoji: "🌳", desc: "Identifies root causes" },
  { key: "agent5", label: "Initiatives", emoji: "🎯", desc: "Proposes prioritised initiatives" },
  { key: "agent6", label: "Narrative", emoji: "📝", desc: "Writes executive summary" },
];

// ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────

const SYS = {
  agent1: `You are an expert data analyst with access to a run_python tool.
The CSV files have been loaded into Pyodide's filesystem. Load them with pandas using pd.read_csv('filename.csv').
Steps: explore structure → analyse relevant to the question → generate charts where useful → summarise findings.

For charts, use this exact pattern at the end of any code block that generates a chart:
import io, base64
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
buf.seek(0)
print("CHART:" + base64.b64encode(buf.read()).decode())
plt.close()

End with a clear structured written summary of findings with specific numbers and patterns.`,

  agent2: `You are a hypothesis generator. Generate hypotheses ONLY based on the analysis findings.
Every claim must cite specific evidence from the analysis. No invention.

Respond ONLY with valid JSON (no markdown, no preamble):
{"hypotheses":[{"id":1,"hypothesis":"...","evidence":"specific cited finding","confidence":"high|medium|low","how_to_test":"..."}]}`,

  agent3: `You are a rigorous hypothesis critic. For each hypothesis, check it strictly against the analysis findings.
Flag anything that overreaches what the data actually shows.

Respond ONLY with valid JSON (no markdown):
{"verdicts":[{"hypothesis_id":1,"verdict":"supported|partially_supported|unsupported","reasoning":"...","suggested_revision":"only if partially_supported or unsupported, else null"}]}`,

  agent4: `You are a root cause analyst. For each selected hypothesis, trace the causal chain back to the root cause.
Be honest about confidence and what data would confirm or disprove this.

Respond ONLY with valid JSON (no markdown):
{"root_causes":[{"hypothesis_id":1,"root_cause":"...","causal_chain":"A → B → C → observed pattern","confidence":"high|medium|low","missing_evidence":"..."}]}`,

  agent4Critic: `You are a critic of root cause analyses. Flag any causal claims that go beyond the evidence.

Respond ONLY with valid JSON (no markdown):
{"verdicts":[{"hypothesis_id":1,"verdict":"supported|partially_supported|unsupported","reasoning":"...","suggested_revision":"..."}]}`,

  agent5: `You are a strategic initiative planner. Generate concrete initiatives to address the root causes.
Score: Impact 1=low 2=medium 3=high. Effort 1=easy 2=medium 3=hard. Sort by impact descending.

Respond ONLY with valid JSON (no markdown):
{"initiatives":[{"id":1,"hypothesis_id":1,"initiative":"...","rationale":"...","impact":3,"impact_reasoning":"...","effort":2,"effort_reasoning":"..."}]}`,

  agent5Critic: `You are a critic of strategic initiatives. Flag initiatives not grounded in the analysis or with questionable impact/effort scores.

Respond ONLY with valid JSON (no markdown):
{"verdicts":[{"initiative_id":1,"verdict":"sound|questionable|unsupported","reasoning":"...","suggested_revision":"..."}]}`,

  agent6: `You are a professional business writer. Write two executive summaries (~500-600 words each) based on the full pipeline results.
Tell a cohesive story: what we found → why it matters → what we do about it.
Executive version: for CEO, COO, Legal, Finance — strategic, no technical jargon, business impact.
Product Team version: for PM and Engineering — technical, implementation-focused, actionable.

Respond ONLY with valid JSON (no markdown):
{"executive":"full executive text here...","product_team":"full product team text here..."}`,
};

// ─── UTILITIES ────────────────────────────────────────────────────────────────

async function callClaude({ model, messages, system, tools = [], mcpServers = [] }) {
  const body = { model, max_tokens: 4000, system, messages };
  if (tools.length) body.tools = tools;
  if (mcpServers.length) body.mcp_servers = mcpServers;

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err.slice(0, 300)}`);
  }
  return res.json();
}

function extractText(content) {
  return content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

function parseJSON(text) {
  try {
    const clean = text.replace(/```json\s*|\s*```/g, "").trim();
    return JSON.parse(clean);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw new Error("Could not parse JSON from agent response");
  }
}

const PYTHON_TOOL = {
  name: "run_python",
  description:
    "Execute Python code in the browser. Has pandas, numpy, matplotlib. Use print() for output. Files are pre-loaded in the filesystem.",
  input_schema: {
    type: "object",
    properties: { code: { type: "string", description: "Python code to execute" } },
    required: ["code"],
  },
};

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────

export default function DataAnalysisPipeline() {
  const [phase, setPhase] = useState("setup");
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState("");
  const [agentModels, setAgentModels] = useState({
    agent1: "claude-sonnet-4-6", agent2: "claude-sonnet-4-6",
    agent3: "claude-sonnet-4-6", agent4: "claude-sonnet-4-6",
    agent5: "claude-sonnet-4-6", agent6: "claude-sonnet-4-6",
  });
  const [status, setStatus] = useState({
    agent1: "waiting", agent2: "waiting", agent3: "waiting",
    agent4: "waiting", agent5: "waiting", agent6: "waiting",
  });
  const [outputs, setOutputs] = useState({});
  const [codeLogs, setCodeLogs] = useState([]);
  const [charts, setCharts] = useState([]);
  const [selectedHyps, setSelectedHyps] = useState([]);
  const [agent4Critic, setAgent4Critic] = useState(false);
  const [agent5Critic, setAgent5Critic] = useState(false);
  const [adjustedScores, setAdjustedScores] = useState({});
  const [editedNarratives, setEditedNarratives] = useState({ executive: "", product_team: "" });
  const [driveLink, setDriveLink] = useState("");
  const [errors, setErrors] = useState({});
  const [valErrors, setValErrors] = useState([]);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [pyodideLoading, setPyodideLoading] = useState(true);
  const pyRef = useRef(null);
  const bottomRef = useRef(null);

  // Scroll to bottom as content appears
  useEffect(() => {
    if (phase !== "setup") {
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  }, [outputs, phase, codeLogs, charts]);

  // Load Pyodide
  useEffect(() => {
    const load = async () => {
      try {
        if (!document.getElementById("pyodide-script")) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.id = "pyodide-script";
            s.src = "https://cdn.jsdelivr.net/pyodide/v0.26.0/full/pyodide.js";
            s.onload = res; s.onerror = rej;
            document.head.appendChild(s);
          });
        }
        const py = await window.loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.0/full/" });
        await py.loadPackage(["pandas", "matplotlib", "numpy"]);
        // Init matplotlib Agg backend
        py.runPython(`import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt`);
        pyRef.current = py;
        setPyodideReady(true);
      } catch (e) {
        console.error("Pyodide load error:", e);
      } finally {
        setPyodideLoading(false);
      }
    };
    load();
  }, []);

  // Execute Python code
  const runPython = async (code) => {
    const py = pyRef.current;
    if (!py) return { success: false, error: "Python not loaded", output: "" };
    try {
      py.runPython(`
import sys
from io import StringIO
_buf = StringIO()
sys.stdout = _buf
`);
      try { py.runPython(code); } catch (e) {
        const out = py.runPython(`_buf.getvalue()`);
        py.runPython(`sys.stdout = sys.__stdout__`);
        return { success: false, error: e.message, output: out };
      }
      const output = py.runPython(`_buf.getvalue()`);
      py.runPython(`sys.stdout = sys.__stdout__`);
      // Extract chart data
      const newCharts = [];
      const clean = output.replace(/CHART:([A-Za-z0-9+/=\n]+)/g, (_, b64) => {
        newCharts.push(b64.replace(/\n/g, ""));
        return "[chart generated]";
      });
      if (newCharts.length) setCharts((p) => [...p, ...newCharts]);
      return { success: true, output: clean };
    } catch (e) {
      return { success: false, error: e.message, output: "" };
    }
  };

  const setAgentStatus = (a, s) => setStatus((p) => ({ ...p, [a]: s }));
  const setOutput = (k, v) => setOutputs((p) => ({ ...p, [k]: v }));
  const setErr = (a, m) => setErrors((p) => ({ ...p, [a]: m }));

  const readFile = (f) => new Promise((res, rej) => {
    const r = new FileReader(); r.onload = (e) => res(e.target.result); r.onerror = rej; r.readAsText(f);
  });

  const validateFiles = (fl) => {
    const errs = [];
    if (!fl.length) errs.push("Upload at least one CSV file");
    if (fl.length > 3) errs.push("Maximum 3 CSV files allowed");
    fl.forEach((f) => {
      if (!f.name.toLowerCase().endsWith(".csv")) errs.push(`${f.name} is not a CSV`);
      if (f.size === 0) errs.push(`${f.name} is empty`);
      if (f.size > 10 * 1024 * 1024) errs.push(`${f.name} exceeds 10MB`);
    });
    return errs;
  };

  // Agentic loop for Agent 1
  const runAgent1 = async (csvContents) => {
    // Write CSVs to Pyodide filesystem
    const py = pyRef.current;
    for (const f of csvContents) {
      py.FS.writeFile(f.name, f.content);
    }

    const fileList = csvContents.map((f) => f.name).join(", ");
    const messages = [{
      role: "user",
      content: `Question: ${question}\n\nAvailable CSV files: ${fileList}\nLoad them with: pd.read_csv('filename.csv')\n\nAnalyse the data to answer the question.`,
    }];

    const logs = [];
    let iterations = 0;

    while (iterations < 12) {
      iterations++;
      const res = await callClaude({
        model: agentModels.agent1,
        messages,
        system: SYS.agent1,
        tools: [PYTHON_TOOL],
      });

      messages.push({ role: "assistant", content: res.content });
      const toolUses = res.content.filter((b) => b.type === "tool_use");

      if (!toolUses.length) {
        const summary = extractText(res.content);
        setCodeLogs(logs);
        return { summary, logs };
      }

      const toolResults = [];
      for (const tu of toolUses) {
        const result = await runPython(tu.input.code);
        const log = { code: tu.input.code, output: result.output, error: result.error, success: result.success };
        logs.push(log);
        setCodeLogs([...logs]);
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: result.success
            ? result.output || "Executed with no output"
            : `ERROR: ${result.error}\n${result.output}`,
        });
      }
      messages.push({ role: "user", content: toolResults });
    }
    throw new Error("Agent 1 hit maximum iterations");
  };

  // Run with single retry
  const withRetry = async (fn) => {
    try { return await fn(); } catch {
      return await fn();
    }
  };

  const runPipeline = async () => {
    const ve = validateFiles(files);
    if (ve.length) { setValErrors(ve); return; }
    if (!question.trim()) { setValErrors(["Please enter a question"]); return; }
    setValErrors([]); setPhase("running"); setCharts([]); setErrors({});
    setOutputs({}); setSelectedHyps([]); setCodeLogs([]);
    setStatus({ agent1: "waiting", agent2: "waiting", agent3: "waiting", agent4: "waiting", agent5: "waiting", agent6: "waiting" });

    try {
      const csvContents = await Promise.all(files.map(async (f) => ({ name: f.name, content: await readFile(f) })));

      // Agent 1
      setAgentStatus("agent1", "running");
      let a1;
      try {
        a1 = await withRetry(() => runAgent1(csvContents));
        setOutput("agent1", a1);
        setAgentStatus("agent1", "complete");
      } catch (e) { setAgentStatus("agent1", "error"); setErr("agent1", e.message); return; }

      // Agent 2
      setAgentStatus("agent2", "running");
      let a2;
      try {
        a2 = await withRetry(async () => {
          const res = await callClaude({ model: agentModels.agent2, system: SYS.agent2, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nGenerate hypotheses based only on these findings.` }] });
          return parseJSON(extractText(res.content));
        });
        setOutput("agent2", a2);
        setAgentStatus("agent2", "complete");
      } catch (e) { setAgentStatus("agent2", "error"); setErr("agent2", e.message); return; }

      // Agent 3
      setAgentStatus("agent3", "running");
      let a3;
      try {
        a3 = await withRetry(async () => {
          const res = await callClaude({ model: agentModels.agent3, system: SYS.agent3, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nHypotheses:\n${JSON.stringify(a2.hypotheses, null, 2)}` }] });
          return parseJSON(extractText(res.content));
        });
        setOutput("agent3", a3);
        setAgentStatus("agent3", "complete");
      } catch (e) { setAgentStatus("agent3", "error"); setErr("agent3", e.message); return; }

      setPhase("cp1");
    } catch (e) { console.error(e); }
  };

  const continueFromCP1 = async () => {
    if (!selectedHyps.length) return;
    setPhase("running");
    const a1 = outputs.agent1, a2 = outputs.agent2, a3 = outputs.agent3;
    const selHyps = a2.hypotheses.filter((h) => selectedHyps.includes(h.id));
    const selVerdicts = a3.verdicts.filter((v) => selectedHyps.includes(v.hypothesis_id));

    // Agent 4
    setAgentStatus("agent4", "running");
    let a4;
    try {
      a4 = await withRetry(async () => {
        const res = await callClaude({ model: agentModels.agent4, system: SYS.agent4, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nSelected hypotheses:\n${JSON.stringify(selHyps, null, 2)}\n\nCritic verdicts:\n${JSON.stringify(selVerdicts, null, 2)}` }] });
        const parsed = parseJSON(extractText(res.content));
        if (agent4Critic) {
          const cRes = await callClaude({ model: agentModels.agent4, system: SYS.agent4Critic, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nRoot causes:\n${JSON.stringify(parsed.root_causes, null, 2)}` }] });
          parsed.critic = parseJSON(extractText(cRes.content));
        }
        return parsed;
      });
      setOutput("agent4", a4);
      setAgentStatus("agent4", "complete");
    } catch (e) { setAgentStatus("agent4", "error"); setErr("agent4", e.message); return; }

    // Agent 5
    setAgentStatus("agent5", "running");
    let a5;
    try {
      a5 = await withRetry(async () => {
        const res = await callClaude({ model: agentModels.agent5, system: SYS.agent5, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nValidated hypotheses:\n${JSON.stringify(selHyps, null, 2)}\n\nRoot causes:\n${JSON.stringify(a4.root_causes, null, 2)}` }] });
        const parsed = parseJSON(extractText(res.content));
        if (agent5Critic) {
          const cRes = await callClaude({ model: agentModels.agent5, system: SYS.agent5Critic, messages: [{ role: "user", content: `Analysis findings:\n${a1.summary}\n\nInitiatives:\n${JSON.stringify(parsed.initiatives, null, 2)}` }] });
          parsed.critic = parseJSON(extractText(cRes.content));
        }
        return parsed;
      });
      setOutput("agent5", a5);
      setAgentStatus("agent5", "complete");
      const scores = {};
      a5.initiatives.forEach((i) => { scores[i.id] = { impact: i.impact, effort: i.effort }; });
      setAdjustedScores(scores);
    } catch (e) { setAgentStatus("agent5", "error"); setErr("agent5", e.message); return; }

    setPhase("cp2");
  };

  const continueFromCP2 = async () => {
    setPhase("running");
    const a1 = outputs.agent1, a2 = outputs.agent2, a4 = outputs.agent4, a5 = outputs.agent5;
    const selHyps = a2.hypotheses.filter((h) => selectedHyps.includes(h.id));
    const initiativesWithScores = [...a5.initiatives]
      .map((i) => ({ ...i, impact: adjustedScores[i.id]?.impact ?? i.impact, effort: adjustedScores[i.id]?.effort ?? i.effort }))
      .sort((a, b) => b.impact - a.impact);

    setAgentStatus("agent6", "running");
    try {
      const context = `Analysis findings:\n${a1.summary}\n\nValidated hypotheses:\n${JSON.stringify(selHyps, null, 2)}\n\nRoot causes:\n${JSON.stringify(a4.root_causes, null, 2)}\n\nPrioritised initiatives:\n${JSON.stringify(initiativesWithScores, null, 2)}`;
      const res = await callClaude({ model: agentModels.agent6, system: SYS.agent6, messages: [{ role: "user", content: context }] });
      const parsed = parseJSON(extractText(res.content));
      setOutput("agent6", parsed);
      setEditedNarratives({ executive: parsed.executive, product_team: parsed.product_team });
      setAgentStatus("agent6", "complete");
      setPhase("cp3");
    } catch (e) { setAgentStatus("agent6", "error"); setErr("agent6", e.message); }
  };

  const createGoogleDoc = async () => {
    const now = new Date();
    const month = now.toLocaleString("en", { month: "long" });
    const folderName = `Data-analysis-${now.getFullYear()}-${month}-v1`;
    try {
      const res = await callClaude({
        model: agentModels.agent6,
        system: "You are a Google Drive assistant. Use the available tools to create a folder and document as instructed.",
        messages: [{
          role: "user",
          content: `1. Create a folder in My Drive named: "${folderName}"\n2. Create a Google Doc inside it titled "Data Analysis Report" with these two sections:\n\n# EXECUTIVE SUMMARY\n\n${editedNarratives.executive}\n\n---\n\n# PAYMENTS PRODUCT TEAM BRIEFING\n\n${editedNarratives.product_team}\n\n3. Return the URL of the created document.`,
        }],
        mcpServers: [{ type: "url", url: "https://drivemcp.googleapis.com/mcp/v1", name: "google-drive" }],
      });
      const text = res.content.filter((b) => b.type === "text").map((b) => b.text).join("");
      const link = text.match(/https:\/\/docs\.google\.com\/[^\s)"]+/)?.[0] || "https://drive.google.com";
      setDriveLink(link);
      setPhase("complete");
    } catch (e) {
      setErr("agent6", `Google Drive error: ${e.message}`);
    }
  };

  // ─── STYLE HELPERS ────────────────────────────────────────────────────────

  const statusColor = (s) => ({ waiting: "#64748b", running: "#38bdf8", complete: "#34d399", error: "#f87171" }[s] || "#64748b");
  const statusIcon = (s) => ({ waiting: "○", running: "◌", complete: "✓", error: "✗" }[s] || "○");

  const verdictStyle = (v) => ({
    supported: { bg: "rgba(52,211,153,0.08)", border: "rgba(52,211,153,0.25)", color: "#6ee7b7" },
    partially_supported: { bg: "rgba(251,191,36,0.08)", border: "rgba(251,191,36,0.25)", color: "#fcd34d" },
    unsupported: { bg: "rgba(248,113,113,0.08)", border: "rgba(248,113,113,0.25)", color: "#fca5a5" },
    sound: { bg: "rgba(52,211,153,0.08)", border: "rgba(52,211,153,0.25)", color: "#6ee7b7" },
    questionable: { bg: "rgba(251,191,36,0.08)", border: "rgba(251,191,36,0.25)", color: "#fcd34d" },
    unsound: { bg: "rgba(248,113,113,0.08)", border: "rgba(248,113,113,0.25)", color: "#fca5a5" },
  }[v] || { bg: "rgba(100,116,139,0.1)", border: "rgba(100,116,139,0.2)", color: "#94a3b8" });

  const confColor = (c) => ({ high: "#6ee7b7", medium: "#fcd34d", low: "#94a3b8" }[c] || "#94a3b8");
  const scoreStyle = (n) => ({ 1: { bg: "rgba(100,116,139,0.15)", color: "#94a3b8" }, 2: { bg: "rgba(251,191,36,0.12)", color: "#fcd34d" }, 3: { bg: "rgba(52,211,153,0.12)", color: "#6ee7b7" } }[n] || {});

  const card = { background: "#111827", border: "1px solid #1f2937", borderRadius: 16, overflow: "hidden", marginBottom: 24 };
  const cardHeader = { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 24px", borderBottom: "1px solid #1f2937" };

  // ─── RENDER ───────────────────────────────────────────────────────────────

  return (
    <div style={{ minHeight: "100vh", background: "#0a0e1a", color: "#e2e8f0", fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ borderBottom: "1px solid #1f2937", padding: "16px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0d1117" }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.02em" }}>Data Analysis Pipeline</div>
          <div style={{ fontSize: 11, color: "#4b5563", marginTop: 2 }}>6-agent • hypothesis validation • root cause • initiatives</div>
        </div>
        <div style={{ fontSize: 11, color: pyodideReady ? "#34d399" : "#4b5563" }}>
          {pyodideLoading ? "◌ Loading Python..." : pyodideReady ? "● Python ready" : "○ Python unavailable"}
        </div>
      </div>

      {/* Pipeline Status Bar */}
      {phase !== "setup" && (
        <div style={{ borderBottom: "1px solid #1f2937", background: "#0d1117", padding: "10px 32px", overflowX: "auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4, whiteSpace: "nowrap" }}>
            {AGENT_CONFIG.map((a, i) => (
              <span key={a.key} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span style={{ fontSize: 11, color: statusColor(status[a.key]), display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <span style={{ animation: status[a.key] === "running" ? "spin 1s linear infinite" : "none" }}>{statusIcon(status[a.key])}</span>
                  {a.label}
                </span>
                {i < AGENT_CONFIG.length - 1 && <span style={{ color: "#374151", margin: "0 4px" }}>→</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ maxWidth: 860, margin: "0 auto", padding: "32px 24px" }}>

        {/* ── SETUP ─────────────────────────────────────────────────────── */}
        {phase === "setup" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Files */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Data Files</div>
              <div
                onClick={() => document.getElementById("file-in").click()}
                onDrop={(e) => { e.preventDefault(); setFiles(Array.from(e.dataTransfer.files)); setValErrors([]); }}
                onDragOver={(e) => e.preventDefault()}
                style={{ border: "2px dashed #1f2937", borderRadius: 14, padding: "40px 24px", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s" }}
                onMouseEnter={(e) => e.currentTarget.style.borderColor = "#374151"}
                onMouseLeave={(e) => e.currentTarget.style.borderColor = "#1f2937"}
              >
                <div style={{ fontSize: 28, marginBottom: 8 }}>📂</div>
                <div style={{ fontSize: 13, color: "#9ca3af", fontWeight: 500 }}>Drop CSV files here or click to browse</div>
                <div style={{ fontSize: 11, color: "#374151", marginTop: 4 }}>Up to 3 files · max 10MB each</div>
                <input id="file-in" type="file" multiple accept=".csv" style={{ display: "none" }} onChange={(e) => { setFiles(Array.from(e.target.files)); setValErrors([]); }} />
              </div>
              {files.length > 0 && (
                <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                  {files.map((f, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, background: "#111827", border: "1px solid #1f2937", borderRadius: 10, padding: "10px 14px", fontSize: 12 }}>
                      <span style={{ color: "#34d399" }}>✓</span>
                      <span style={{ flex: 1, color: "#e2e8f0" }}>{f.name}</span>
                      <span style={{ color: "#4b5563" }}>{(f.size / 1024).toFixed(1)} KB</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Question */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Your Question</div>
              <textarea
                style={{ width: "100%", background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: "12px 16px", fontSize: 13, color: "#e2e8f0", resize: "none", outline: "none", lineHeight: 1.6, boxSizing: "border-box", fontFamily: "inherit" }}
                rows={3}
                placeholder="e.g. What is driving the increase in payment failures in Q1?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
            </div>

            {/* Model selectors */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Models per Agent</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {AGENT_CONFIG.map((a) => (
                  <div key={a.key} style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 14 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                      <span>{a.emoji}</span>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>{a.label}</span>
                    </div>
                    <div style={{ fontSize: 11, color: "#4b5563", marginBottom: 10 }}>{a.desc}</div>
                    <select
                      style={{ width: "100%", background: "#0d1117", border: "1px solid #1f2937", borderRadius: 8, padding: "6px 10px", fontSize: 11, color: "#9ca3af", outline: "none", fontFamily: "inherit" }}
                      value={agentModels[a.key]}
                      onChange={(e) => setAgentModels((p) => ({ ...p, [a.key]: e.target.value }))}
                    >
                      {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Critic toggles */}
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>Optional Critic Passes</div>
              <div style={{ display: "flex", gap: 10 }}>
                {[{ label: "Critic for Root Cause", state: agent4Critic, set: setAgent4Critic }, { label: "Critic for Initiatives", state: agent5Critic, set: setAgent5Critic }].map(({ label, state, set }) => (
                  <button key={label} onClick={() => set((p) => !p)} style={{ padding: "8px 14px", borderRadius: 10, border: `1px solid ${state ? "rgba(245,158,11,0.4)" : "#1f2937"}`, background: state ? "rgba(245,158,11,0.08)" : "#111827", color: state ? "#fbbf24" : "#6b7280", fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                    <span>{state ? "●" : "○"}</span>{label}
                  </button>
                ))}
              </div>
            </div>

            {/* Errors */}
            {valErrors.length > 0 && (
              <div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 12, padding: "12px 16px" }}>
                {valErrors.map((e, i) => <div key={i} style={{ fontSize: 12, color: "#fca5a5" }}>{e}</div>)}
              </div>
            )}

            {/* Run button */}
            <button
              onClick={runPipeline}
              disabled={!pyodideReady}
              style={{ background: pyodideReady ? "#f59e0b" : "#374151", color: pyodideReady ? "#0a0e1a" : "#6b7280", border: "none", borderRadius: 12, padding: "14px 24px", fontSize: 13, fontWeight: 700, cursor: pyodideReady ? "pointer" : "not-allowed", width: "100%", fontFamily: "inherit", transition: "background 0.2s" }}
            >
              {pyodideLoading ? "Loading Python runtime..." : "Run Pipeline →"}
            </button>
          </div>
        )}

        {/* ── PIPELINE OUTPUT ────────────────────────────────────────────── */}
        {phase !== "setup" && (
          <div>
            {/* Agent 1 */}
            {(status.agent1 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>📊</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 1 — Analyst</span>
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent1) }}>{statusIcon(status.agent1)} {status.agent1}</span>
                </div>
                {status.agent1 === "running" && !codeLogs.length && (
                  <div style={{ padding: "32px", textAlign: "center", color: "#4b5563", fontSize: 12 }}>Running Python analysis...</div>
                )}
                {codeLogs.map((log, i) => (
                  <div key={i} style={{ borderBottom: "1px solid #1f2937", padding: "16px 24px" }}>
                    <div style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace", marginBottom: 8 }}>Code block {i + 1}</div>
                    <pre style={{ background: "#0d1117", borderRadius: 8, padding: "12px", fontSize: 11, fontFamily: "monospace", color: "#94a3b8", overflowX: "auto", margin: 0, whiteSpace: "pre-wrap" }}>{log.code}</pre>
                    {log.output && <pre style={{ background: "#0d1117", borderRadius: 8, padding: "12px", fontSize: 11, fontFamily: "monospace", color: "#6ee7b7", overflowX: "auto", margin: "8px 0 0", whiteSpace: "pre-wrap" }}>{log.output}</pre>}
                    {log.error && <pre style={{ background: "rgba(248,113,113,0.05)", borderRadius: 8, padding: "12px", fontSize: 11, fontFamily: "monospace", color: "#fca5a5", overflowX: "auto", margin: "8px 0 0", whiteSpace: "pre-wrap" }}>{log.error}</pre>}
                  </div>
                ))}
                {charts.length > 0 && (
                  <div style={{ padding: "16px 24px", borderBottom: "1px solid #1f2937" }}>
                    <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>Generated Charts</div>
                    {charts.map((c, i) => <img key={i} src={`data:image/png;base64,${c}`} alt={`Chart ${i + 1}`} style={{ maxWidth: "100%", borderRadius: 8, marginBottom: 8 }} />)}
                  </div>
                )}
                {outputs.agent1?.summary && (
                  <div style={{ padding: "20px 24px" }}>
                    <div style={{ fontSize: 10, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12 }}>Findings Summary</div>
                    <div style={{ fontSize: 13, color: "#d1d5db", lineHeight: 1.7, whiteSpace: "pre-wrap" }}>{outputs.agent1.summary}</div>
                  </div>
                )}
                {errors.agent1 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: "12px", fontSize: 12, color: "#fca5a5" }}>{errors.agent1}</div></div>}
              </div>
            )}

            {/* Agent 2 */}
            {(status.agent2 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>💡</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 2 — Hypothesis Generator</span>
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent2) }}>{statusIcon(status.agent2)} {status.agent2}</span>
                </div>
                {status.agent2 === "running" && <div style={{ padding: 32, textAlign: "center", color: "#4b5563", fontSize: 12 }}>Generating hypotheses...</div>}
                {outputs.agent2?.hypotheses?.map((h) => (
                  <div key={h.id} style={{ borderBottom: "1px solid #1f2937", padding: "16px 24px", display: "flex", gap: 14 }}>
                    <span style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace", marginTop: 2 }}>H{h.id}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0", marginBottom: 6 }}>{h.hypothesis}</div>
                      <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}><span style={{ color: "#4b5563" }}>Evidence: </span>{h.evidence}</div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, border: `1px solid rgba(0,0,0,0.1)`, background: `${confColor(h.confidence)}18`, color: confColor(h.confidence) }}>{h.confidence} confidence</span>
                        <span style={{ fontSize: 11, color: "#4b5563" }}>Test: {h.how_to_test}</span>
                      </div>
                    </div>
                  </div>
                ))}
                {errors.agent2 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: 12, fontSize: 12, color: "#fca5a5" }}>{errors.agent2}</div></div>}
              </div>
            )}

            {/* Agent 3 */}
            {(status.agent3 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>🔍</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 3 — Critic</span>
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent3) }}>{statusIcon(status.agent3)} {status.agent3}</span>
                </div>
                {status.agent3 === "running" && <div style={{ padding: 32, textAlign: "center", color: "#4b5563", fontSize: 12 }}>Reviewing hypotheses...</div>}
                {outputs.agent3?.verdicts?.map((v) => {
                  const hyp = outputs.agent2?.hypotheses?.find((h) => h.id === v.hypothesis_id);
                  const vs = verdictStyle(v.verdict);
                  return (
                    <div key={v.hypothesis_id} style={{ borderBottom: "1px solid #1f2937", padding: "16px 24px", display: "flex", gap: 14 }}>
                      <span style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace", marginTop: 2 }}>H{v.hypothesis_id}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 8 }}>{hyp?.hypothesis}</div>
                        <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, background: vs.bg, border: `1px solid ${vs.border}`, color: vs.color, display: "inline-block", marginBottom: 6 }}>
                          {v.verdict.replace(/_/g, " ")}
                        </span>
                        <div style={{ fontSize: 12, color: "#d1d5db" }}>{v.reasoning}</div>
                        {v.suggested_revision && v.suggested_revision !== "null" && (
                          <div style={{ fontSize: 11, color: "#fcd34d", marginTop: 6 }}><span style={{ color: "#92400e" }}>Revision: </span>{v.suggested_revision}</div>
                        )}
                      </div>
                    </div>
                  );
                })}
                {errors.agent3 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: 12, fontSize: 12, color: "#fca5a5" }}>{errors.agent3}</div></div>}
              </div>
            )}

            {/* Checkpoint 1 */}
            {phase === "cp1" && outputs.agent2?.hypotheses && (
              <div style={{ background: "rgba(245,158,11,0.04)", border: "2px solid rgba(245,158,11,0.25)", borderRadius: 16, padding: 24, marginBottom: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ color: "#f59e0b" }}>⚡</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#fbbf24" }}>Checkpoint 1 — Select Hypotheses for Root Cause Analysis</span>
                </div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 16 }}>Use your domain knowledge. Unsupported hypotheses can still be worth investigating.</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 20 }}>
                  {outputs.agent2.hypotheses.map((h) => {
                    const verdict = outputs.agent3?.verdicts?.find((v) => v.hypothesis_id === h.id);
                    const vs = verdict ? verdictStyle(verdict.verdict) : null;
                    const sel = selectedHyps.includes(h.id);
                    return (
                      <button key={h.id} onClick={() => setSelectedHyps((p) => sel ? p.filter((id) => id !== h.id) : [...p, h.id])} style={{ textAlign: "left", padding: "12px 16px", borderRadius: 12, border: `1px solid ${sel ? "rgba(245,158,11,0.4)" : "#1f2937"}`, background: sel ? "rgba(245,158,11,0.06)" : "#111827", cursor: "pointer", display: "flex", alignItems: "flex-start", gap: 12 }}>
                        <span style={{ width: 18, height: 18, borderRadius: 6, border: `1px solid ${sel ? "#f59e0b" : "#374151"}`, background: sel ? "#f59e0b" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: sel ? "#0a0e1a" : "transparent", flexShrink: 0, marginTop: 1 }}>✓</span>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 500, color: sel ? "#fef3c7" : "#d1d5db" }}>{h.hypothesis}</div>
                          {vs && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 20, background: vs.bg, border: `1px solid ${vs.border}`, color: vs.color, display: "inline-block", marginTop: 4 }}>{verdict.verdict.replace(/_/g, " ")}</span>}
                        </div>
                      </button>
                    );
                  })}
                </div>
                <button onClick={continueFromCP1} disabled={!selectedHyps.length} style={{ width: "100%", background: selectedHyps.length ? "#f59e0b" : "#374151", color: selectedHyps.length ? "#0a0e1a" : "#6b7280", border: "none", borderRadius: 12, padding: "12px", fontSize: 13, fontWeight: 700, cursor: selectedHyps.length ? "pointer" : "not-allowed", fontFamily: "inherit" }}>
                  Continue with {selectedHyps.length} hypothesis{selectedHyps.length !== 1 ? "es" : ""} →
                </button>
              </div>
            )}

            {/* Agent 4 */}
            {(status.agent4 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>🌳</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 4 — Root Cause Analyst</span>
                    {agent4Critic && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 20, background: "rgba(245,158,11,0.1)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.2)" }}>+ Critic</span>}
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent4) }}>{statusIcon(status.agent4)} {status.agent4}</span>
                </div>
                {status.agent4 === "running" && <div style={{ padding: 32, textAlign: "center", color: "#4b5563", fontSize: 12 }}>Analysing root causes...</div>}
                {outputs.agent4?.root_causes?.map((rc) => {
                  const cv = outputs.agent4?.critic?.verdicts?.find((v) => v.hypothesis_id === rc.hypothesis_id);
                  return (
                    <div key={rc.hypothesis_id} style={{ borderBottom: "1px solid #1f2937", padding: "16px 24px" }}>
                      <div style={{ display: "flex", gap: 14 }}>
                        <span style={{ fontSize: 10, color: "#4b5563", fontFamily: "monospace", marginTop: 2 }}>H{rc.hypothesis_id}</span>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0", marginBottom: 8 }}>{rc.root_cause}</div>
                          <div style={{ background: "#0d1117", borderRadius: 8, padding: "8px 12px", fontSize: 11, fontFamily: "monospace", color: "#94a3b8", marginBottom: 8 }}>{rc.causal_chain}</div>
                          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                            <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 20, background: `${confColor(rc.confidence)}18`, color: confColor(rc.confidence), border: `1px solid ${confColor(rc.confidence)}40` }}>{rc.confidence} confidence</span>
                          </div>
                          {rc.missing_evidence && <div style={{ fontSize: 11, color: "#6b7280" }}><span style={{ color: "#374151" }}>Missing evidence: </span>{rc.missing_evidence}</div>}
                          {cv && (() => { const vs = verdictStyle(cv.verdict); return <div style={{ marginTop: 10, padding: "8px 12px", borderRadius: 8, background: vs.bg, border: `1px solid ${vs.border}`, fontSize: 11, color: vs.color }}><span style={{ fontWeight: 600 }}>Critic ({cv.verdict.replace(/_/g, " ")}): </span>{cv.reasoning}</div>; })()}
                        </div>
                      </div>
                    </div>
                  );
                })}
                {errors.agent4 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: 12, fontSize: 12, color: "#fca5a5" }}>{errors.agent4}</div></div>}
              </div>
            )}

            {/* Agent 5 */}
            {(status.agent5 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>🎯</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 5 — Initiative Planner</span>
                    {agent5Critic && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 20, background: "rgba(245,158,11,0.1)", color: "#f59e0b", border: "1px solid rgba(245,158,11,0.2)" }}>+ Critic</span>}
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent5) }}>{statusIcon(status.agent5)} {status.agent5}</span>
                </div>
                {status.agent5 === "running" && <div style={{ padding: 32, textAlign: "center", color: "#4b5563", fontSize: 12 }}>Planning initiatives...</div>}
                {outputs.agent5?.initiatives && phase !== "cp2" && [...outputs.agent5.initiatives].sort((a, b) => b.impact - a.impact).map((init) => (
                  <div key={init.id} style={{ borderBottom: "1px solid #1f2937", padding: "14px 24px", display: "flex", gap: 16, alignItems: "flex-start" }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0", marginBottom: 4 }}>{init.initiative}</div>
                      <div style={{ fontSize: 11, color: "#6b7280" }}>{init.rationale}</div>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                      <span style={{ padding: "3px 10px", borderRadius: 8, fontSize: 11, fontFamily: "monospace", fontWeight: 600, ...scoreStyle(init.impact) }}>I:{init.impact}</span>
                      <span style={{ padding: "3px 10px", borderRadius: 8, fontSize: 11, fontFamily: "monospace", fontWeight: 600, ...scoreStyle(init.effort) }}>E:{init.effort}</span>
                    </div>
                  </div>
                ))}
                {errors.agent5 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: 12, fontSize: 12, color: "#fca5a5" }}>{errors.agent5}</div></div>}
              </div>
            )}

            {/* Checkpoint 2 */}
            {phase === "cp2" && outputs.agent5?.initiatives && (
              <div style={{ background: "rgba(245,158,11,0.04)", border: "2px solid rgba(245,158,11,0.25)", borderRadius: 16, padding: 24, marginBottom: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ color: "#f59e0b" }}>⚡</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#fbbf24" }}>Checkpoint 2 — Review & Adjust Initiative Scores</span>
                </div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 16 }}>Adjust based on your knowledge. Impact: 1=low 2=medium 3=high. Effort: 1=easy 2=medium 3=hard.</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 20 }}>
                  {[...outputs.agent5.initiatives].sort((a, b) => b.impact - a.impact).map((init) => {
                    const sc = adjustedScores[init.id] || { impact: init.impact, effort: init.effort };
                    const cv = outputs.agent5?.critic?.verdicts?.find((v) => v.initiative_id === init.id);
                    return (
                      <div key={init.id} style={{ background: "#111827", border: "1px solid #1f2937", borderRadius: 12, padding: 14 }}>
                        <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0", marginBottom: 4 }}>{init.initiative}</div>
                        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: cv ? 8 : 12 }}>{init.rationale}</div>
                        {cv && (() => { const vs = verdictStyle(cv.verdict); return <div style={{ marginBottom: 12, padding: "6px 10px", borderRadius: 8, background: vs.bg, border: `1px solid ${vs.border}`, fontSize: 11, color: vs.color }}><span style={{ fontWeight: 600 }}>Critic: </span>{cv.reasoning}</div>; })()}
                        <div style={{ display: "flex", gap: 20 }}>
                          {[{ label: "Impact", key: "impact" }, { label: "Effort", key: "effort" }].map(({ label, key }) => (
                            <div key={key}>
                              <div style={{ fontSize: 10, color: "#4b5563", marginBottom: 6 }}>{label}</div>
                              <div style={{ display: "flex", gap: 4 }}>
                                {[1, 2, 3].map((n) => (
                                  <button key={n} onClick={() => setAdjustedScores((p) => ({ ...p, [init.id]: { ...p[init.id], [key]: n } }))} style={{ width: 32, height: 32, borderRadius: 8, border: `1px solid ${sc[key] === n ? "transparent" : "#374151"}`, ...(sc[key] === n ? scoreStyle(n) : { background: "#1f2937", color: "#4b5563" }), fontSize: 11, fontWeight: 700, cursor: "pointer", fontFamily: "monospace" }}>{n}</button>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <button onClick={continueFromCP2} style={{ width: "100%", background: "#f59e0b", color: "#0a0e1a", border: "none", borderRadius: 12, padding: 12, fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                  Generate Executive Summary →
                </button>
              </div>
            )}

            {/* Agent 6 */}
            {(status.agent6 !== "waiting") && (
              <div style={card}>
                <div style={cardHeader}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 18 }}>📝</span>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>Agent 6 — Narrative Writer</span>
                  </div>
                  <span style={{ fontSize: 11, color: statusColor(status.agent6) }}>{statusIcon(status.agent6)} {status.agent6}</span>
                </div>
                {status.agent6 === "running" && <div style={{ padding: 32, textAlign: "center", color: "#4b5563", fontSize: 12 }}>Writing executive summaries...</div>}
                {errors.agent6 && <div style={{ padding: "16px 24px" }}><div style={{ background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.2)", borderRadius: 10, padding: 12, fontSize: 12, color: "#fca5a5" }}>{errors.agent6}</div></div>}
              </div>
            )}

            {/* Checkpoint 3 */}
            {phase === "cp3" && (
              <div style={{ background: "rgba(245,158,11,0.04)", border: "2px solid rgba(245,158,11,0.25)", borderRadius: 16, padding: 24, marginBottom: 24 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ color: "#f59e0b" }}>⚡</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#fbbf24" }}>Checkpoint 3 — Review & Edit Before Export</span>
                </div>
                <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 20 }}>You are the final critic. Edit directly, then create the Google Doc.</div>
                {[{ key: "executive", label: "Executive Version", desc: "For CEO, COO, Legal, Finance" }, { key: "product_team", label: "Product Team Version", desc: "For PM and Engineering" }].map(({ key, label, desc }) => (
                  <div key={key} style={{ marginBottom: 20 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0" }}>{label}</div>
                        <div style={{ fontSize: 10, color: "#4b5563" }}>{desc}</div>
                      </div>
                      <div style={{ fontSize: 10, color: "#4b5563" }}>{editedNarratives[key]?.split(" ").filter(Boolean).length || 0} words</div>
                    </div>
                    <textarea
                      style={{ width: "100%", background: "#0d1117", border: "1px solid #1f2937", borderRadius: 12, padding: "14px 16px", fontSize: 13, color: "#e2e8f0", resize: "vertical", outline: "none", lineHeight: 1.8, boxSizing: "border-box", fontFamily: "inherit", minHeight: 240 }}
                      value={editedNarratives[key]}
                      onChange={(e) => setEditedNarratives((p) => ({ ...p, [key]: e.target.value }))}
                    />
                  </div>
                ))}
                <button onClick={createGoogleDoc} style={{ width: "100%", background: "#f59e0b", color: "#0a0e1a", border: "none", borderRadius: 12, padding: 12, fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                  Create Google Doc →
                </button>
              </div>
            )}

            {/* Complete */}
            {phase === "complete" && (
              <div style={{ background: "rgba(52,211,153,0.05)", border: "2px solid rgba(52,211,153,0.2)", borderRadius: 16, padding: 40, textAlign: "center" }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>✓</div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#34d399", marginBottom: 8 }}>Pipeline Complete</div>
                <div style={{ fontSize: 13, color: "#4b5563", marginBottom: 24 }}>Your report has been saved to Google Drive.</div>
                {driveLink && (
                  <a href={driveLink} target="_blank" rel="noopener noreferrer" style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#34d399", color: "#0a0e1a", textDecoration: "none", padding: "10px 20px", borderRadius: 10, fontSize: 13, fontWeight: 700 }}>
                    Open in Google Docs →
                  </a>
                )}
                <div>
                  <button onClick={() => { setPhase("setup"); setFiles([]); setQuestion(""); setOutputs({}); setCharts([]); setCodeLogs([]); setErrors({}); setSelectedHyps([]); setStatus({ agent1: "waiting", agent2: "waiting", agent3: "waiting", agent4: "waiting", agent5: "waiting", agent6: "waiting" }); }} style={{ marginTop: 16, background: "none", border: "none", color: "#4b5563", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>
                    Run new analysis
                  </button>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
