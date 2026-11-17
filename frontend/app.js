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

/* ---------- init ---------- */
async function init() {
  let backends;
  [HEALTH, MODES, { cases: CASES }, backends] = await Promise.all([api("health"), api("modes"), api("cases"), api("backends")]);
  $("#chips").innerHTML = [`backend ${HEALTH.backend}`, `embedder ${HEALTH.embedder}`, `τ ${HEALTH.default_tau}`, HEALTH.algorithm, `token TTL ${HEALTH.spt_ttl}s`, `executor ${HEALTH.executor}`, `${HEALTH.embedder_backend}`]
    .map((t) => `<span class="chip">${esc(t)}</span>`).join("");
  if (HEALTH.backend === "ollama" && !HEALTH.ollama.ready) { $("#banner").className = "banner bad"; $("#banner").textContent = "Ollama not ready: " + HEALTH.ollama.error; }
  $("#backend").innerHTML = backends.available.map((b) => `<option value="${esc(b.key)}">${esc(b.label)}</option>`).join("");
  $("#backend").value = backends.default;
  const groups = { benign: "Benign tasks", static: "Static injections", adaptive: "Adaptive and camouflaged", token: "Forged, tampered, expired, replayed tokens", propagation: "Propagation through handoffs" };
  $("#preset").innerHTML = Object.entries(groups).map(([k, label]) =>
    `<optgroup label="${label}">${CASES.filter((c) => c.scenario === k).map((c) => `<option value="${c.id}">${c.id}: ${esc(c.goal.slice(0, 54))}</option>`).join("")}</optgroup>`).join("");
  $("#mode").innerHTML = MODES.map((m) => `<option value="${m.key}">${m.key}: ${esc(m.label)}</option>`).join("") + `<option value="custom">Custom</option>`;
  $("#mode").value = HEALTH.default_mode || "FULL";
  $("#tau").value = HEALTH.default_tau; $("#alTau").value = HEALTH.default_tau; $("#evTau").value = HEALTH.default_tau;
  syncTau(); loadPreset(); applyMode();
}
const syncTau = () => { $("#tauOut").textContent = Number($("#tau").value).toFixed(2); $("#alTauOut").textContent = Number($("#alTau").value).toFixed(2); };
$("#tau").addEventListener("input", syncTau); $("#alTau").addEventListener("input", syncTau);

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
  $("#cmp").innerHTML = `<tr><th>Configuration</th><th>Attack executed</th><th>Detected</th><th>Task completed</th><th>First block</th><th class="num">Verify ms</th><th class="num">Token cut</th></tr>` +
    r.results.map((x) => `<tr><td><b>${esc(x.mode)}</b><br><span class="hint">${esc(x.label)}</span></td>
      <td>${tag(x.attack_success ? "yes" : "no", x.attack_success ? "no" : "ok")}</td><td>${tag(x.detected ? "yes" : "no", x.detected ? "ok" : "mid")}</td>
      <td>${tag(x.completed ? "yes" : "no", x.completed ? "ok" : "no")}</td><td>${esc(x.blocked_at || "none")}</td>
      <td class="num">${x.verify_ms.toFixed(2)}</td><td class="num">${pct(x.r_tok)}</td></tr>`).join("");
  $("#cmpPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}));
const tag = (t, kind) => `<span class="tag ${kind}">${esc(t)}</span>`;

function render(r, c) {
  const m = r.metrics, isAtk = c.is_attack;
  let cls, head, sub;
  if (isAtk && m.attack_success) { cls = "bad"; head = "Attack succeeded"; sub = "A malicious action reached the Data Plane and executed."; }
  else if (isAtk) { cls = "good"; head = m.detected ? "Attack blocked" : "Attack had no effect"; sub = m.detected ? "The gateway stopped it before anything malicious executed." : "Nothing malicious ran, but the gateway did not flag it."; }
  else if (m.detected) { cls = "warn"; head = "Benign task flagged (false positive)"; sub = "Safe content was quarantined, which can reduce task quality."; }
  else { cls = "good"; head = "Benign task completed"; sub = "No verification failures."; }
  $("#banner").className = "banner " + cls;
  $("#banner").innerHTML = `${head}<small>${sub} ${m.completed_subtasks}/${m.total_subtasks} sub-tasks done, ${m.verify_ms.toFixed(2)} ms spent verifying, ${pct(m.r_tok)} token reduction.</small>`;

  const byStep = {}; r.steps.forEach((e) => { (byStep[e.step] ||= []).push(e); });
  $("#strip").innerHTML = STEP_NAMES.map(([n, d], i) => {
    const ev = byStep[i + 1] || [];
    const st = !ev.length || ev.every((e) => e.status === "skipped") ? "skipped" : ev.some((e) => e.status === "blocked") ? "blocked" : ev.some((e) => e.status === "warn") ? "warn" : "ok";
    const mark = { skipped: "off", blocked: "blocked", warn: "!", ok: "done" }[st];
    return `<div class="node ${i === 3 ? "dp" : ""} ${st}"><span class="st">${mark}</span><b>${i + 1}. ${n}</b><span>${d}</span><span>${ev.length ? ev.length + " event" + (ev.length > 1 ? "s" : "") : ""}</span></div>`;
  }).join("");

  $("#checksPanel").hidden = false;
  $("#checks").innerHTML = `<tr><th>Point</th><th>Result</th><th>Stage</th><th>Similarity vs τ ${r.tau.toFixed(2)}</th><th>Reason</th></tr>` +
    (r.checks.length ? r.checks.map((k) => `<tr><td>${esc(k.point)}${k.tool ? `<br><span class="hint">${esc(k.tool)}</span>` : ""}</td>
      <td>${tag(k.allowed ? "allowed" : "blocked", k.allowed ? "ok" : "no")}</td><td>${esc(k.stage || "-")}</td>
      <td>${scoreCell(k, r.tau)}</td><td>${esc(k.reason)}${k.hidden_content ? ' <span class="tag mid">hidden content found</span>' : ""}</td></tr>`).join("")
      : `<tr><td colspan="5" class="hint">Verification is off in this configuration, so nothing was checked.</td></tr>`);

  currentTokens = r.tokens || []; $("#tokPanel").hidden = !currentTokens.length;
  $("#tokList").innerHTML = r.tokens.map((t, i) => `<button type="button" class="tokrow" data-i="${i}" aria-pressed="${i === 0}"><b>${esc(t.agent)}</b> depth ${t.depth}
    <span class="hint">tools: ${esc(t.tools.join(", ") || "none")}</span> <code>lin ${esc(t.lineage)}</code></button>`).join("");
  document.querySelectorAll(".tokrow").forEach((b) => b.addEventListener("click", () => showToken(+b.dataset.i)));
  if (currentTokens.length) showToken(0);
  else $("#tokClaims").textContent = "No SPT issued by this baseline.";

  $("#ledgerPanel").hidden = false;
  $("#ledger").innerHTML = `<tr><th>Tool</th><th>Arguments</th><th>Origin</th></tr>` + r.ledger.map((l) =>
    `<tr><td>${esc(l.tool)}</td><td class="mono small">${esc(JSON.stringify(l.args))}</td><td>${l.malicious ? tag("malicious", "no") : tag("user task", "ok")}</td></tr>`).join("");
  $("#final").textContent = r.final_response || "(nothing delivered)";

  $("#logPanel").hidden = false;
  $("#log").innerHTML = r.steps.map((e) => `<li class="${e.status}"><span class="s">step ${e.step}</span>${esc(e.detail)}</li>`).join("");
}
function scoreCell(k, tau) {
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
  $("#alOut").innerHTML = `<h2>${r.candidates.length} candidate action${r.candidates.length === 1 ? "" : "s"} found</h2>
    <p class="hint">Root hash ${esc(r.root_hash.slice(0, 16))}… ${r.hidden_content ? '<span class="tag mid">hidden comment or zero-width text exposed</span>' : ""}</p>` +
    (r.candidates.map((c) => `<div class="cand ${c.allowed ? "" : "no"}"><b>${c.allowed ? "Aligned" : "Blocked"}</b> · similarity ${c.score.toFixed(3)} vs τ ${r.tau.toFixed(2)}<br>${esc(c.text)}</div>`).join("") || '<p class="hint">No instruction-like text. The payload is plain data.</p>') +
    `<h3>What the agent receives</h3><pre class="small">${esc(r.sanitized)}</pre>`;
}));

init().catch((e) => { $("#banner").className = "banner bad"; $("#banner").textContent = "Could not reach the backend: " + e.message; });


/* ---------- stored real-data results ---------- */
async function loadReal() {
  const R = await api("results"); const parts = [];
  const row = (x) => `<tr><td><b>${x.mode}</b></td><td class="num">${pct(x.asr)}</td><td class="num">${pct(x.recall)}</td><td class="num">${pct(x.precision)}</td><td class="num">${x.f1.toFixed(2)}</td><td class="num">${pct(x.fpr)}</td><td class="num">${x.latency_check_ms.median}</td></tr>`;
  if (R.injecagent) for (const [s, v] of Object.entries(R.injecagent.settings))
    parts.push(`<h3>InjecAgent ${s} — test split: ${v.n_attack} real attacks + ${v.n_benign} benign (MiniLM, τ ${R.injecagent.tau})</h3><div class="scroll"><table><tr><th>Config</th><th class="num">ASR</th><th class="num">Recall</th><th class="num">Precision</th><th class="num">F1</th><th class="num">FPR</th><th class="num">check p50 ms</th></tr>${v.rows.map(row).join("")}</table></div>`);
  if (R.agentdojo) parts.push(`<h3>${esc(R.agentdojo.benchmark)} — ${esc(R.agentdojo.attack)} attack, Ollama agent</h3><div class="scroll"><table><tr><th>Config</th><th class="num">ASR</th><th class="num">Utility</th><th class="num">Utility under attack</th><th class="num">Clean runs flagged</th><th class="num">Injected runs flagged</th></tr>${R.agentdojo.rows.map((x) => `<tr><td><b>${x.mode}</b></td><td class="num">${pct(x.asr)}</td><td class="num">${pct(x.utility_clean)}</td><td class="num">${pct(x.utility_under_attack)}</td><td class="num">${pct(x.clean_runs_flagged)}</td><td class="num">${pct(x.injected_runs_flagged)}</td></tr>`).join("")}</table></div>`);
  if (parts.length) parts.push(`<p class="hint"><b>Read with care:</b> results depend on the Ollama model you ran (named in each results file). \"Hard benign\" false positives and the 35-case set are author-written.</p>`);
  if (parts.length) { $("#realOut").hidden = false; $("#realBody").innerHTML = parts.join(""); }
}
loadReal().catch(() => {});
