"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, body) => {
  const r = await fetch("/api/" + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
};
const pct = (x) => (x * 100).toFixed(0) + "%";
const STEP_NAMES = [
  ["Optimize", "Compress, prune, decompose, budget"], ["Register & sign", "Root hash, embedding, SPT"],
  ["Orchestrate", "Pick agent, child token"], ["Execute", "Sandboxed tool call"],
  ["Verify", "Stage A crypto, Stage B intent"], ["Enforce & adapt", "Sanitize, quarantine, re-plan"],
];
let CASES = [], MODES = [], HEALTH = {}, currentTokens = [];

/* ---------- tabs ---------- */
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
}));

let CURRENT_TAU = 0.13;

function setTau(val) {
  const num = Number(val);
  CURRENT_TAU = isNaN(num) ? (HEALTH ? HEALTH.default_tau : 0.13) : num;
  const formatted = CURRENT_TAU.toFixed(2);
  if ($("#tau")) $("#tau").value = formatted;
  if ($("#tauOut")) $("#tauOut").textContent = formatted;
  if ($("#alTau")) $("#alTau").value = formatted;
  if ($("#alTauOut")) $("#alTauOut").textContent = formatted;
  if ($("#evTau")) $("#evTau").value = formatted;
  updateChips();
}

function updateChips(backendOverride) {
  let backendLabel = "ollama (llama3.2)";
  if (backendOverride) {
    backendLabel = backendOverride.toLowerCase().includes("llama") ? backendOverride : `${backendOverride} (llama3.2)`;
  } else if ($("#backend") && $("#backend").selectedOptions && $("#backend").selectedOptions.length) {
    backendLabel = $("#backend").selectedOptions[0].textContent;
  } else if (HEALTH && HEALTH.backend) {
    backendLabel = HEALTH.backend.toLowerCase().includes("llama") ? HEALTH.backend : `${HEALTH.backend} (llama3.2)`;
  }
  const tauVal = CURRENT_TAU != null ? CURRENT_TAU : (HEALTH ? HEALTH.default_tau : 0.13);
  const isDefault = HEALTH && Math.abs(tauVal - HEALTH.default_tau) < 0.005;
  const tauText = isDefault ? `τ ${tauVal.toFixed(2)}` : `τ ${tauVal.toFixed(2)} (custom)`;
  $("#chips").innerHTML = [`backend ${backendLabel}`, `embedder ${HEALTH.embedder}`, tauText, HEALTH.algorithm, `token TTL ${HEALTH.spt_ttl}s`, `executor ${HEALTH.executor}`, `${HEALTH.embedder_backend}`]
    .map((t) => `<span class="chip">${esc(t)}</span>`).join("");
}

/* ---------- init ---------- */
async function init() {
  let backends;
  [HEALTH, MODES, { cases: CASES }, backends] = await Promise.all([api("health"), api("modes"), api("cases"), api("backends")]);
  if (HEALTH.backend === "ollama" && !HEALTH.ollama.ready) { $("#banner").className = "banner bad"; $("#banner").textContent = "Ollama not ready: " + HEALTH.ollama.error; }
  if ($("#ollamaUrl") && HEALTH.ollama && HEALTH.ollama.url) $("#ollamaUrl").value = HEALTH.ollama.url;
  if ($("#reconnectOllamaBtn")) {
    $("#reconnectOllamaBtn").addEventListener("click", async () => {
      const url = $("#ollamaUrl").value.trim();
      if (!url) return;
      $("#reconnectOllamaBtn").textContent = "...";
      try {
        await api("ollama/configure", { url });
        HEALTH = await api("health");
        if (HEALTH.ollama && HEALTH.ollama.ready) {
          $("#banner").className = "banner idle";
          $("#banner").textContent = `Connected to Ollama at ${HEALTH.ollama.url} (${HEALTH.ollama.model})! Pick a scenario and run it.`;
        } else {
          $("#banner").className = "banner bad";
          $("#banner").textContent = "Ollama not ready: " + (HEALTH.ollama ? HEALTH.ollama.error : "Unknown error");
        }
        updateChips();
      } catch (err) {
        $("#banner").className = "banner bad";
        $("#banner").textContent = "Failed to connect: " + err.message;
      } finally {
        $("#reconnectOllamaBtn").textContent = "Connect";
      }
    });
  }
  $("#backend").innerHTML = backends.available.map((b) => `<option value="${esc(b.key)}">${esc(b.label)}</option>`).join("");
  $("#backend").value = backends.default;
  CURRENT_TAU = HEALTH.default_tau;
  setTau(HEALTH.default_tau);
  const updateBackendVisibility = () => {
    if ($("#ollamaConfigRow")) $("#ollamaConfigRow").style.display = $("#backend").value === "ollama" ? "block" : "none";
  };
  $("#backend").addEventListener("change", () => {
    updateChips($("#backend").value);
    updateBackendVisibility();
  });
  updateBackendVisibility();
  $("#tau").addEventListener("input", (e) => setTau(e.target.value));
  $("#alTau").addEventListener("input", (e) => setTau(e.target.value));
  if ($("#evTau")) $("#evTau").addEventListener("input", (e) => setTau(e.target.value));
  if ($("#resetTauBtn")) $("#resetTauBtn").addEventListener("click", () => setTau(HEALTH.default_tau));
  if ($("#resetAlTauBtn")) $("#resetAlTauBtn").addEventListener("click", () => setTau(HEALTH.default_tau));
  const groups = { benign: "Benign tasks", write_capable: "Write-capable (read_only: false)", static: "Static injections", adaptive: "Adaptive and camouflaged", token: "Forged, tampered, expired, replayed tokens", propagation: "Propagation through handoffs" };
  $("#preset").innerHTML = Object.entries(groups).map(([k, label]) =>
    `<optgroup label="${label}">${CASES.filter((c) => c.scenario === k).map((c) => `<option value="${c.id}">${c.id}: ${esc(c.label || c.goal.slice(0, 54))}</option>`).join("")}</optgroup>`).join("");
  $("#mode").innerHTML = MODES.map((m) => `<option value="${m.key}">${m.key}: ${esc(m.label)}</option>`).join("") + `<option value="custom">Custom</option>`;
  $("#mode").value = HEALTH.default_mode || "FULL";
  loadPreset(); applyMode();
}

function loadPreset() {
  const c = CASES.find((x) => x.id === $("#preset").value); if (!c) return;
  $("#query").value = c.query; $("#fixtures").value = JSON.stringify(c.fixtures || {}, null, 2);
  $("#injection").value = c.injection || ""; $("#rogue").value = c.rogue ? JSON.stringify(c.rogue, null, 2) : "";
  $("#rogue").closest("details").open = !!c.rogue;
}
$("#preset").addEventListener("change", loadPreset);
function applyMode() {
  const m = MODES.find((x) => x.key === $("#mode").value); if (!m) return;
  $("#fA").checked = m.stage_a; $("#fB").checked = m.stage_b; $("#fS").checked = m.scope;
  $("#fArgs").checked = m.argument_checks; $("#fT").checked = m.token_opt;
  $("#fI").checked = m.trusted_plan_only;
}
$("#mode").addEventListener("change", applyMode);
["#fA", "#fB", "#fS", "#fArgs", "#fT", "#fI"].forEach((id) => $(id).addEventListener("change", () => { $("#mode").value = "custom"; }));

function buildCase() {
  const base = CASES.find((x) => x.id === $("#preset").value) || {};
  let fixtures = {}, rogue = null;
  try { fixtures = JSON.parse($("#fixtures").value || "{}"); } catch { throw new Error("Tool outputs must be valid JSON, for example {\"read_email\": \"...\"}"); }
  if ($("#rogue").value.trim()) { try { rogue = JSON.parse($("#rogue").value); } catch { throw new Error("Forged-token request must be valid JSON"); } }
  const injection = $("#injection").value.trim() || null;
  return { id: "custom", goal: base.goal && $("#query").value === base.query ? base.goal : $("#query").value, query: $("#query").value,
    fixtures, injection, rogue, context: base.context || [], scenario: base.scenario || "static", is_attack: !!(injection || rogue) };
}
function currentMode() {
  const k = $("#mode").value;
  return k === "custom" ? {
    key: "custom", label: "Custom", stage_a: $("#fA").checked, stage_b: $("#fB").checked,
    scope: $("#fS").checked, argument_checks: $("#fArgs").checked, token_opt: $("#fT").checked,
    trusted_plan_only: $("#fI").checked,
  } : k;
}
async function busy(btn, fn) {
  const t = btn.textContent; btn.disabled = true; btn.textContent = "Working…";
  try { await fn(); } catch (e) { alert(e.message); } finally { btn.disabled = false; btn.textContent = t; }
}

/* ---------- live run ---------- */
$("#runBtn").addEventListener("click", () => busy($("#runBtn"), async () => {
  const c = buildCase();
  render(await api("run", { case: c, mode: currentMode(), tau: Number($("#tau").value), backend: $("#backend").value }), c);
}));
$("#cmpBtn").addEventListener("click", () => busy($("#cmpBtn"), async () => {
  const r = await api("compare", { case: buildCase(), tau: Number($("#tau").value), backend: $("#backend").value });
  $("#cmpPanel").hidden = false;
  $("#cmp").innerHTML = `<tr><th>Configuration</th><th>Outcome</th><th>Injection detected</th><th>Goal blocked</th><th>Benign block</th><th>First block</th><th class="num">Verify ms</th><th class="num">Token cut</th></tr>` +
    r.results.map((x) => {
      const o = x.outcome || {};
      const badge = o.badge ? tag(o.badge, o.color === "bad" ? "no" : o.color === "good" ? "ok" : "mid") : "-";
      return `<tr><td><b>${esc(x.mode)}</b><br><span class="hint">${esc(x.label)}</span></td>
        <td>${badge}</td>
        <td>${tag(x.injection_detected ? "yes" : "no", x.injection_detected ? "ok" : "mid")}</td>
        <td>${tag(x.attacker_goal_blocked ? "yes" : "no", x.attacker_goal_blocked ? "ok" : "mid")}</td>
        <td>${tag(x.benign_blocked ? "yes" : "no", x.benign_blocked ? "no" : "ok")}</td>
        <td>${esc(x.blocked_at || "none")}</td>
        <td class="num">${x.verify_ms.toFixed(2)}</td><td class="num">${pct(x.r_tok)}</td></tr>`;
    }).join("");
  $("#cmpPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}));
const tag = (t, kind) => `<span class="tag ${kind}">${esc(t)}</span>`;
const originTag = (origin, malicious) => {
  if (malicious) return tag("malicious", "no");
  if (origin === "agent_initiated") return tag("agent initiated", "mid");
  if (origin === "rogue_token") return tag("rogue token", "no");
  return tag("user plan", "ok");
};

function render(r, c) {
  updateChips(r.llm || $("#backend").value);
  const m = r.metrics, o = r.outcome || {};
  const cls = o.color || (c.is_attack ? (m.attack_success ? "bad" : "good") : (m.detected ? "warn" : "good"));
  const head = o.badge || (c.is_attack ? (m.attack_success ? "Attack succeeded" : (m.detected ? "Attack blocked" : "Attack had no effect")) : (m.detected ? "Benign task flagged" : "Benign task completed"));
  const sub = o.sub || (c.is_attack ? (m.attack_success ? "A malicious action reached the Data Plane and executed." : (m.detected ? "The gateway stopped it before anything malicious executed." : "Nothing malicious ran.")) : (m.detected ? "Safe content was quarantined." : "No verification failures."));
  $("#banner").className = "banner " + cls;
  const replanNote = m.includes_replan ? " (including re-plan)" : "";
  $("#banner").innerHTML = `${esc(head)}<small>${esc(sub)} ${m.completed_subtasks}/${m.total_subtasks} sub-tasks done, ${m.verify_ms.toFixed(2)} ms spent verifying, ${pct(m.r_tok)} token reduction${replanNote}.</small>`;

  const byStep = {}; r.steps.forEach((e) => { (byStep[e.step] ||= []).push(e); });
  $("#strip").innerHTML = STEP_NAMES.map(([n, d], i) => {
    const ev = byStep[i + 1] || [];
    const st = !ev.length || ev.every((e) => e.status === "skipped") ? "skipped" : ev.some((e) => e.status === "blocked") ? "blocked" : ev.some((e) => e.status === "warn") ? "warn" : "ok";
    const mark = { skipped: "off", blocked: "blocked", warn: "!", ok: "done" }[st];
    return `<div class="node ${i === 3 ? "dp" : ""} ${st}"><span class="st">${mark}</span><b>${i + 1}. ${n}</b><span>${d}</span><span>${ev.length ? ev.length + " event" + (ev.length > 1 ? "s" : "") : ""}</span></div>`;
  }).join("");

  $("#checksPanel").hidden = false;
  $("#checks").innerHTML = `<tr><th>Point</th><th>Result</th><th>Stage</th><th>Similarity vs Threshold</th><th>Reason / Rule</th></tr>` +
    (r.checks.length ? r.checks.map((k) => `<tr><td>${esc(k.point)}${k.tool ? `<br><span class="hint">${esc(k.tool)}</span>` : ""}</td>
      <td>${tag(k.allowed ? "allowed" : "blocked", k.allowed ? "ok" : "no")}</td><td>${esc(k.stage || "-")}</td>
      <td>${scoreCell(k, r.tau)}</td><td>${esc(k.reason)}${k.rule ? `<br><span class="hint"><b>Rule:</b> ${esc(k.rule)}</span>` : ""}${k.hidden_content ? ' <span class="tag mid">hidden content found</span>' : ""}</td></tr>`).join("")
      : `<tr><td colspan="5" class="hint">Verification is off in this configuration, so nothing was checked.</td></tr>`);

  currentTokens = r.tokens || []; $("#tokPanel").hidden = !currentTokens.length;
  $("#tokList").innerHTML = r.tokens.map((t, i) => `<button type="button" class="tokrow" data-i="${i}" aria-pressed="${i === 0}"><b>${esc(t.agent)}</b> depth ${t.depth}
    <span class="hint">tools: ${esc(t.tools.join(", ") || "none")}</span> <code>lin ${esc(t.lineage)}</code></button>`).join("");
  document.querySelectorAll(".tokrow").forEach((b) => b.addEventListener("click", () => showToken(+b.dataset.i)));
  if (currentTokens.length) showToken(0);
  else $("#tokClaims").textContent = "No SPT issued by this baseline.";

  $("#ledgerPanel").hidden = false;
  $("#ledger").innerHTML = `<tr><th>Tool</th><th>Arguments</th><th>Origin</th></tr>` + r.ledger.map((l) =>
    `<tr><td>${esc(l.tool)}</td><td class="mono small">${esc(JSON.stringify(l.args))}</td><td>${originTag(l.origin, l.malicious)}</td></tr>`).join("");
  $("#final").textContent = r.final_response || "(nothing delivered)";

  $("#logPanel").hidden = false;
  $("#log").innerHTML = r.steps.map((e) => `<li class="${e.status}"><span class="s">step ${e.step}</span>${esc(e.detail)}</li>`).join("");
}
function scoreCell(k, tau) {
  if (k.stage === "drift" || k.point === "semantic_drift") {
    const driftTau = k.threshold != null ? k.threshold : 0.35;
    const scoreVal = k.score != null ? k.score : (k.drift_score != null ? k.drift_score : 0);
    const pass = scoreVal >= driftTau;
    return `<div title="Drift score ${scoreVal.toFixed(4)} vs drift threshold ${driftTau.toFixed(2)}">
      <div class="bar ${pass ? "" : "low"}"><i style="width:${Math.max(0, Math.min(1, scoreVal)) * 100}%"></i><u style="left:${driftTau * 100}%"></u></div>
      <small>${scoreVal.toFixed(4)} vs τ_drift ${driftTau.toFixed(2)} (${esc(k.drift_metric || "ema")})</small></div>`;
  }
  const cs = k.candidates || [];
  if (!cs.length) return k.score == null ? '<span class="hint">no candidate action</span>' : String(k.score);
  return cs.map((c) => `<div title="${esc(c.text)}"><div class="bar ${c.allowed ? "" : "low"}"><i style="width:${Math.max(0, Math.min(1, c.score)) * 100}%"></i><u style="left:${tau * 100}%"></u></div>
    <small>${c.score.toFixed(3)} · ${esc(c.text.slice(0, 70))}${c.text.length > 70 ? "…" : ""}</small></div>`).join("");
}
function showToken(i) {
  if (!currentTokens[i]) return;
  document.querySelectorAll(".tokrow").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.i === i));
  const t = currentTokens[i];
  $("#tokClaims").textContent = JSON.stringify(t.claims, null, 2) + "\n\nJWT (EdDSA)\n" + t.token.slice(0, 96) + "…";
}

/* ---------- evaluation ---------- */
$("#evBtn").addEventListener("click", () => busy($("#evBtn"), async () => {
  const r = await api("evaluate", { split: $("#evSplit").value, tau: Number($("#evTau").value), repeats: Number($("#evRep").value) });
  $("#evOut").hidden = false;
  $("#evTable").innerHTML = `<tr><th>Config</th><th class="num">ASR</th><th class="num">TCR</th><th class="num">TCR under attack</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">FPR</th>
    <th class="num">Check p50 / p95 ms</th><th class="num">Token cut</th></tr>` + r.rows.map((x) => `<tr><td><b>${x.mode}</b><br><span class="hint">${esc(x.label)}</span></td>
    <td class="num">${pct(x.asr)}</td><td class="num">${pct(x.tcr)}</td><td class="num">${pct(x.tcr_under_attack)}</td><td class="num">${pct(x.precision)}</td><td class="num">${pct(x.recall)}</td>
    <td class="num">${x.f1.toFixed(2)}</td><td class="num">${pct(x.fpr)}</td><td class="num">${x.latency_check_ms.median} / ${x.latency_check_ms.p95}</td><td class="num">${pct(x.r_tok)}</td></tr>`).join("");
  const scns = ["static", "adaptive", "token", "propagation"];
  $("#evScn").innerHTML = `<tr><th>Config</th>${scns.map((s) => `<th class="num">${s}</th>`).join("")}<th class="num">benign flagged</th></tr>` + r.rows.map((x) =>
    `<tr><td><b>${x.mode}</b></td>${scns.map((s) => `<td class="num">${x.by_scenario[s] ? pct(x.by_scenario[s].asr) : "-"}</td>`).join("")}<td class="num">${x.by_scenario.benign ? pct(x.by_scenario.benign.fpr) : "-"}</td></tr>`).join("");
}));
$("#swBtn").addEventListener("click", () => busy($("#swBtn"), async () => {
  const r = await api("sweep", { mode: "G-AB", split: "val" });
  $("#swOut").hidden = false; $("#swChart").innerHTML = chart(r.points);
  const b = r.best; $("#swBest").textContent = b ? `Best F1 under the validation FPR cap: τ = ${b.tau} (F1 ${b.f1.toFixed(2)}, FPR ${pct(b.fpr)}, ASR ${pct(b.asr)}). Fix this before the test runs.` : r.selection_status;
}));
function chart(pts) {
  const W = 760, H = 260, L = 44, B = 34, T = 12, R = 12;
  const x = (t) => L + ((t - pts[0].tau) / (pts[pts.length - 1].tau - pts[0].tau)) * (W - L - R), y = (v) => H - B - v * (H - B - T);
  const series = [["f1", "F1", "var(--trust)"], ["fpr", "False-positive rate", "var(--bad)"], ["asr", "Attack success", "var(--data)"]];
  const grid = [0, .25, .5, .75, 1].map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v * 100}%</text>`).join("");
  const xt = pts.filter((_, i) => i % 2 === 0).map((p) => `<text x="${x(p.tau)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--muted)">${p.tau.toFixed(2)}</text>`).join("");
  const lines = series.map(([k, , c]) => `<polyline fill="none" stroke="${c}" stroke-width="2.5" points="${pts.map((p) => `${x(p.tau)},${y(p[k])}`).join(" ")}"/>` + pts.map((p) => `<circle cx="${x(p.tau)}" cy="${y(p[k])}" r="3" fill="${c}"/>`).join("")).join("");
  const leg = series.map(([, n, c], i) => `<g transform="translate(${L + i * 190},${T})"><rect width="12" height="4" y="4" fill="${c}"/><text x="18" y="10" font-size="12" fill="var(--ink)">${n}</text></g>`).join("");
  return `<svg viewBox="0 0 ${W} ${H + 14}" width="100%" role="img" aria-label="F1, false-positive rate and attack success as the threshold varies">${grid}${xt}${lines}${leg}</svg>`;
}

/* ---------- token optimizer ---------- */
$("#tkBtn").addEventListener("click", () => busy($("#tkBtn"), async () => {
  const r = await api("optimize", { query: $("#tkQuery").value, goal: $("#tkGoal").value, context: $("#tkCtx").value.split("\n") });
  $("#tkOut").innerHTML = `<h2>${r.tokens_before} → ${r.tokens_after} tokens (${pct(r.r_tok)} reduction)</h2>
    <p class="hint">Counted with ${esc(r.counter)} counter. The intent used for hashing and embedding is the original, not the compressed text.</p>
    <h3>Compressed prompt</h3><pre class="small">${esc(r.query)}</pre>
    <h3>Sub-tasks and token budgets</h3><table>${r.subtasks.map((s, i) => `<tr><td>${esc(s)}</td><td class="num">${r.budget["subtask_" + i]}</td></tr>`).join("")}</table>
    <h3>Context kept (${r.kept_context.length})</h3>${r.kept_context.map((c) => `<div class="cand">${esc(c)}</div>`).join("") || '<p class="hint">none</p>'}
    <h3>Context pruned (${r.dropped_context.length})</h3>${r.dropped_context.map((c) => `<div class="cand no">${esc(c)}</div>`).join("") || '<p class="hint">none</p>'}`;
}));

/* ---------- alignment lab ---------- */
$("#alBtn").addEventListener("click", () => busy($("#alBtn"), async () => {
  const r = await api("align", { intent: $("#alIntent").value, text: $("#alText").value, tau: Number($("#alTau").value) });
  const candTable = r.candidates && r.candidates.length ? `
    <div class="scroll"><table style="width:100%;margin-top:0.6rem">
      <tr><th>Candidate Action</th><th class="num">Similarity</th><th class="num">Margin vs τ</th><th>Status</th><th>Blocked by / Rule</th></tr>
      ${r.candidates.map((c) => {
        const margin = c.margin != null ? c.margin : (c.score != null ? c.score - r.tau : 0);
        const isNear = Math.abs(margin) < 0.03;
        const marginStr = (margin >= 0 ? "+" : "") + margin.toFixed(3);
        const nearTag = isNear ? ' <span class="tag mid">near threshold</span>' : "";
        const statusBadge = tag(c.allowed ? "Allowed (sanitized)" : "Blocked", c.allowed ? "ok" : "no");
        const blockedByText = c.blocked_by || (c.allowed ? "none (aligned)" : (c.reason || "Stage B rule"));
        const blockedBadge = c.allowed ? '<span class="hint">none (aligned)</span>' : `<span class="tag no">${esc(blockedByText)}</span>`;
        return `<tr>
          <td><span class="mono">${esc(c.text)}</span></td>
          <td class="num">${c.score != null ? c.score.toFixed(3) : "-"}</td>
          <td class="num">${marginStr}${nearTag}</td>
          <td>${statusBadge}</td>
          <td>${blockedBadge}</td>
        </tr>`;
      }).join("")}
    </table></div>` : '<p class="hint">No instruction-like text. The payload is plain data.</p>';

  $("#alOut").innerHTML = `<h2>${r.candidates.length} candidate action${r.candidates.length === 1 ? "" : "s"} found</h2>
    <p class="hint">Root hash ${esc(r.root_hash.slice(0, 16))}… ${r.hidden_content ? '<span class="tag mid">hidden comment or zero-width text exposed</span>' : ""}</p>
    ${candTable}
    <h3 style="margin-top:1.2rem">What the agent receives</h3><pre class="small">${esc(r.sanitized)}</pre>`;
}));

init().catch((e) => { $("#banner").className = "banner bad"; $("#banner").textContent = "Could not reach the backend: " + e.message; });


/* ---------- stored real-data results ---------- */
async function loadReal() {
  const R = await api("results"); const parts = [];
  const row = (x) => `<tr><td><b>${x.mode}</b></td><td class="num">${pct(x.asr)}</td><td class="num">${pct(x.recall)}</td><td class="num">${pct(x.precision)}</td><td class="num">${x.f1.toFixed(2)}</td><td class="num">${pct(x.fpr)}</td><td class="num">${x.latency_check_ms.median}</td></tr>`;
  if (R.results && R.results.header) {
    const h = R.results.header;
    const trials = R.results.trials || [];
    const configs = [...new Set(trials.map((t) => t.config))];
    const rows = configs.map((cfg) => {
      const cfgTrials = trials.filter((t) => t.config === cfg);
      const atk = cfgTrials.filter((t) => t.kind === "attack");
      const executed = atk.filter((t) => t.executed).length;
      const detected = atk.filter((t) => t.detected).length;
      const blocked = atk.filter((t) => t.goal_blocked).length;
      const benign = cfgTrials.filter((t) => t.kind === "benign");
      const benBlocked = benign.filter((t) => t.fp || t.benign_block).length;
      const asr = atk.length ? (executed / atk.length) : 0;
      const vMs = cfgTrials.map((t) => t.verify_ms || 0).sort((a,b)=>a-b);
      const p50 = vMs.length ? vMs[Math.floor(vMs.length / 2)] : 0;
      return `<tr><td><b>${esc(cfg)}</b></td><td class="num">${pct(asr)} (${executed}/${atk.length})</td><td class="num">${detected}/${atk.length}</td><td class="num">${blocked}/${atk.length}</td><td class="num">${benBlocked}/${benign.length}</td><td class="num">${p50.toFixed(2)} ms</td></tr>`;
    }).join("");
    parts.push(`<h3>Ollama Matrix (${esc(h.model)}, temp ${h.temperature}, τ ${h.tau}, N=${h.n}) — ${esc(h.timestamp)}</h3>
      <div class="scroll"><table><tr><th>Config</th><th class="num">ASR (Goal Executed)</th><th class="num">Injection Detected</th><th class="num">Goal Blocked</th><th class="num">Benign False Positives</th><th class="num">Verify p50</th></tr>${rows}</table></div>`);
  }
  if (R.injecagent) for (const [s, v] of Object.entries(R.injecagent.settings))
    parts.push(`<h3>InjecAgent ${s} — test split: ${v.n_attack} real attacks + ${v.n_benign} benign (MiniLM, τ ${R.injecagent.tau})</h3><div class="scroll"><table><tr><th>Config</th><th class="num">ASR</th><th class="num">Recall</th><th class="num">Precision</th><th class="num">F1</th><th class="num">FPR</th><th class="num">check p50 ms</th></tr>${v.rows.map(row).join("")}</table></div>`);
  if (R.agentdojo && R.agentdojo.rows && R.agentdojo.rows.some((r) => r.unauthorized_blocked > 0 || r.asr > 0)) {
    parts.push(`<h3>${esc(R.agentdojo.benchmark)} — ${esc(R.agentdojo.attack)} attack, Ollama agent</h3><div class="scroll"><table><tr><th>Config</th><th class="num">ASR</th><th class="num">Utility</th><th class="num">Utility under attack</th><th class="num">Clean runs flagged</th><th class="num">Injected runs flagged</th></tr>${R.agentdojo.rows.map((x) => `<tr><td><b>${x.mode}</b></td><td class="num">${pct(x.asr)}</td><td class="num">${pct(x.utility_clean)}</td><td class="num">${pct(x.utility_under_attack)}</td><td class="num">${pct(x.clean_runs_flagged)}</td><td class="num">${pct(x.injected_runs_flagged)}</td></tr>`).join("")}</table></div>`);
  }
  if (R.plan_gating_utility) {
    const pg = R.plan_gating_utility;
    const nTotal = pg.rows && pg.rows[0] ? (pg.rows[0].benign_cases + pg.rows[0].attack_cases) : 54;
    parts.push(`<h3>In-house AgentDojo-style dynamic set (${nTotal} cases, data/dynamic_tool_use.json — not official AgentDojo)</h3><div class="scroll"><table><tr><th>Config</th><th class="num">Dynamic Utility (TCR)</th><th class="num">ASR</th><th class="num">Benign Cases</th><th class="num">Attack Cases</th></tr>${(pg.rows || []).map((x) => `<tr><td><b>${esc(x.mode)}</b></td><td class="num">${pct(x.dynamic_utility_tcr != null ? x.dynamic_utility_tcr : x.tcr)}</td><td class="num">${pct(x.asr)}</td><td class="num">${x.benign_cases}</td><td class="num">${x.attack_cases}</td></tr>`).join("")}</table></div>`);
  }
  if (parts.length) parts.push(`<p class="hint"><b>Read with care:</b> results depend on the Ollama model you ran (named in each results file). \"Hard benign\" false positives and the 35-case set are author-written.</p>`);
  if (parts.length) { $("#realOut").hidden = false; $("#realBody").innerHTML = parts.join(""); }
}
loadReal().catch(() => {});
