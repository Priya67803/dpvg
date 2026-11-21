# Dual-Plane Runtime Verification Gateway (DPVG)

A zero-trust, model-agnostic runtime verification gateway designed to protect tool-augmented and multi-agent LLM systems against indirect prompt injection, privilege escalation, and unauthorized dynamic tool execution without model retraining or fine-tuning.

Evaluated exclusively on Meta's **Llama 3.2 (3B)** via native local **Ollama** (`DPVG_BACKEND=ollama`) and an adversarial **Compromised Agent** mode (`DPVG_BACKEND=compromised`) for deterministic defense verification.

---

## Quick Start & Installation

DPVG runs locally on Python 3.12 with minimal dependencies and local ONNX embedding inference (zero GPU or PyTorch requirement):

```bash
# 1. Environment Setup (Python 3.12)
python -m venv .venv312
.venv312\Scripts\activate            # Linux/macOS: source .venv312/bin/activate

# 2. Install Core Dependencies (FastAPI, Uvicorn, ONNX Runtime, LangGraph, python-docx)
pip install -r requirements.txt

# 3. Optional: Exact Token Counting via tiktoken
pip install -r requirements-ml.txt

# 4. Start Local Ollama (in a separate terminal)
ollama serve
ollama run llama3.2

# 5. Launch DPVG Server & Interactive UI Dashboard
python run.py                        # Served at http://127.0.0.1:8000
```

### Running Tests & Benchmarks
```bash
# Run isolated ablation and security unit tests (39 tests in < 0.2s)
python -m pytest tests/test_ablation_isolation.py -v

# Run 250-iteration latency benchmark with warm-up
python scripts/benchmark_latency.py

# Generate formal Word documentation (.docx)
python scripts/generate_objectives_report.py
python scripts/generate_overview_report.py
```

---

## Architectural Foundations

DPVG establishes a strict architectural boundary between two operational planes:

```
+-----------------------------------------------------------------------------------+
|                              TRUSTED CONTROL PLANE                                |
|  - Asymmetric Ed25519 Keypair (generated in-memory at startup)                    |
|  - Root Intent Vector Registration: e_root = E(q) via all-MiniLM-L6-v2 (ONNX)     |
|  - Semantic Provenance Tokens (SPT): signed JWS tokens with privilege attenuation |
|  - Trajectory Drift Monitoring (EMA) & Structured Re-planning Controller           |
+-----------------------------------------------------------------------------------+
                                         │
                         [ Intercepted Tool Request ]
                         [ + Cryptographic Token    ]
                                         ▼
+-----------------------------------------------------------------------------------+
|                        RUNTIME VERIFICATION GATEWAY                               |
|                                                                                   |
|  [Stage A: Cryptographic Gate]                                                    |
|   • Ed25519 Signature Verification                                                |
|   • Expiration & Lineage Ancestry Validation                                      |
|   • Tool Whitelist Scope Check (tools in scope)                                   |
|   • Read-Only Mutation Invariant (blocks write side-effects when read_only=True)  |
|                                                                                   |
|  [Stage B: Semantic Alignment Engine]                                            |
|   • Action-Class Escalation Check (blocks read -> write/delete/exfiltrate jumps)  |
|   • Hidden-Comment & Override Marker Neutralization                               |
|   • Exfiltration Flow Detection (URL / external IP parameter checks)              |
|   • Cosine Similarity vs e_root (admitted if s >= tau, default tau = 0.13)        |
+-----------------------------------------------------------------------------------+
                                         │
                                [ Admitted / Blocked ]
                                         ▼
+-----------------------------------------------------------------------------------+
|                               UNTRUSTED DATA PLANE                                |
|  - Autonomous LLM Agent (Llama 3.2 3B via Ollama / Compromised Agent)              |
|  - Quarantined Tool Execution Sandbox (read_email, get_calendar, send_email, etc.)|
|  - Tool Results tagged as Untrusted Data (never fed as raw system instructions)   |
+-----------------------------------------------------------------------------------+
```

### Semantic Provenance Tokens (SPT)
Each delegation token is formatted as an RFC 7515 JWS signed via Ed25519 (EdDSA). Tokens encapsulate immutable claims:
* `root_intent_hash`: SHA-256 digest of the user's root query $q$.
* `root_embedding_ref`: Reference to $e_{\text{root}} = E(q)$ stored in `VectorRegistry`.
* `lineage`: Ordered cryptographic ancestry list $[jti_0, \dots, jti_k]$.
* `scope`: Explicit whitelist of permitted tools for the subtask.
* `read_only`: Boolean flag preventing side-effecting operations.
* `depth`: Monotonically increasing delegation index ($d \le 6$).
* **Privilege Attenuation**: $\text{Tools}(\text{Child}) \subseteq \text{Tools}(\text{Parent})$. Child tokens inherit parent constraints; untrusted payloads cannot forge valid tokens.

---

## Defense Configuration Matrix

| Mode | Stage A Crypto | Stage B Rules | Stage B Semantic | Scope Whitelist | Trusted Plan | Argument Check | Plan Extension | Token Opt. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `B0` | No | No | No | No | No | No | No | No |
| `B1` | No | No | No | No | No | No | No | Datamark |
| `G-A` | Yes | No | No | Yes | No | No | No | No |
| `G-B-rules` | No | Yes | No | No | No | No | No | No |
| `G-B-sem` | No | No | Yes | No | No | No | No | No |
| `G-B` | No | Yes | Yes | No | No | No | No | No |
| `G-AB` | Yes | Yes | Yes | No | No | No | No | No |
| `G-AB+P` | Yes | Yes | Yes | No | Yes | No | No | No |
| `G-AB+P+E` | Yes | Yes | Yes | No | Yes | No | Yes | No |
| `G-AB+S` | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| `G-AB+T` | Yes | Yes | Yes | No | No | No | No | Yes |
| `FULL` | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **No** | **Yes** |

---

## Key Empirical Findings & Benchmarks

### 1. InjecAgent Full Benchmark (3,162 Cases)
Evaluated across Direct Harm (DH) and Data Stealing (DS) under Base and Enhanced threat settings:
* **B0 (No Defense)**: $100.0\%$ ASR, $100.0\%$ TCR.
* **B1 (Spotlighting)**: $52.4\%$ ASR, $47.6\%$ Recall.
* **FULL (DPVG)**: **$0.0\%$ ASR**, **$100.0\%$ Recall**, **$98.8\%$ Benign Utility (TCR)**, **$0.3\%$ FPR**.

### 2. Dynamic Tool Use & Plan Extension (54 Cases)
Evaluated on the in-house AgentDojo-style dynamic suite (`data/dynamic_tool_use.json`):
* `G-AB`: $100.0\%$ utility, but $60.0\%$ attack leakage.
* `G-AB+P` (Static Plan Gating): Destroys dynamic utility ($0.0\%$ TCR).
* `G-AB+P+E` (Redesigned Plan Extension): **$100.0\%$ Dynamic Utility (39/39)**, **$0.0\%$ ASR (0/15)**.

### 3. Repeated Llama 3.2 Trials ($N=20$ per case)
Repeated trials with fixed temperature on local Llama 3.2 (3B) with Wilson 95% confidence intervals:
* Tier 1 Static Injection (`s01`): B0 ASR $35.0\%$ $\rightarrow$ FULL ASR **$0.0\%$** (Stage B rules).
* Tier 2 In-Domain Action (`p02`): B0 ASR $25.0\%$ $\rightarrow$ FULL ASR **$0.0\%$** (Stage B action-class).
* Tier 3 Exfiltration Flow: B0 ASR $50.0\%$ $\rightarrow$ FULL ASR **$0.0\%$** (Stage A scope bounds).
* Token Tampering (`t01`): B0 ASR $50.0\%$ $\rightarrow$ FULL ASR **$0.0\%$** (Stage A Ed25519 cryptography).
* Susceptible Subset Overall: B0 ASR $44.4\%$ (32/72) $\rightarrow$ FULL ASR **$0.0\%$** (0/72 admitted).

### 4. Write-Capable Tool Scenarios (`w01`, `w02`)
When user intent explicitly authorizes write capabilities (`read_only: false`, `allowed_tools: ["read_email", "send_email"]`):
* **Baseline B0**: Injected malicious directive (`delete_calendar_events`) succeeds ($100\%$ ASR, `ATTACK SUCCEEDED`).
* **FULL DPVG**: Blocks attack ($0.0\%$ ASR) via Stage A write-whitelist containment and Stage B exfiltration rules.

### 5. Deterministic Injection Testing ("Compromised Agent" Mode)
Small 3B models frequently exhibit instruction neglect (ignoring injections and giving a false impression of security in undefended baselines). Under scripted `CompromisedBackend` ("Compromised Agent"):
* **Baseline B0**: $100.0\%$ ASR (all injected directives executed).
* **FULL DPVG**: **$0.0\%$ ASR** (100% defense rate across all test suites).

---

## Latency Overhead Profile ($N=250$, 50 Warm-Up Runs)

Verification overhead is sub-millisecond per stage and less than $0.1\%$ of local LLM inference time (`results/latency_benchmark.json`, `results/latency.md`):

| Operation / Stage | N | Warmup | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | % of Llama 3.2 Inference |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Token issuance (per task) | 250 | 50 | 0.245 | 0.386 | 0.474 | 0.740 | 0.270 | < 0.02% |
| Stage A verify (per call) | 250 | 50 | 0.575 | 0.850 | 1.097 | 1.689 | 0.630 | 0.03% |
| Stage B action-class check | 250 | 50 | 0.028 | 0.043 | 0.049 | 0.097 | 0.030 | < 0.01% |
| Stage B embedding & cosine | 250 | 50 | 0.034 | 0.065 | 0.075 | 0.154 | 0.042 | < 0.01% |
| Stage B total payload verify | 250 | 50 | 0.704 | 0.941 | 1.123 | 1.862 | 0.733 | 0.04% |
| **Single-plan total verify** | **250** | **50** | **2.992** | **4.018** | **4.999** | **7.059** | **2.994** | **0.08%** |
| **Re-plan total verify** | **250** | **50** | **4.605** | **6.227** | **7.609** | **10.870** | **4.626** | **0.08%** |

> **Cold-Start Latency Jitter Eliminated:** Un-warmed single-run tests showed swings between $1.97\text{ ms}$ and $34\text{ ms}$ due to initial ONNX buffer allocation and Ed25519 table construction. DPVG includes an automated `warmup()` routine in `Gateway.__init__()`, eliminating jitter and stabilizing post-warmup latency at $p50 < 3.0\text{ ms}$.

---

## Diagnostic Explainability & Ground-Truth Transparency

* **First Mechanism Attribution**: Execution banners report the exact causal mechanism that triggered the block (e.g. `Stage B rule: override marker at tool_result`), displaying multi-step trajectory drift as subsequent telemetry rather than conflating cause and effect.
* **Benign False Positive Distinction**: Benign actions blocked by strict threshold policies are labeled `Benign action blocked (false positive)` rather than `Attacker goal blocked`.
* **Transparent Drift Thresholding**: Explains why actions like `read_email` (scoring $0.2647 > \tau=0.13$) are quarantined: they violate trajectory drift threshold $\tau_{\text{drift}} = 0.35$ as the `"2nd consecutive low-alignment step after quarantine"`.
* **Alignment Lab Rule Visibility**: Dedicated "Blocked by / Rule" column reports exact deterministic filters (`action-class escalation`, `hidden-comment rule`, `exfiltration rule`, `override marker rule`).
* **Dynamic Tool Gating (`p02`)**:
  * When `"Only authorize actions in trusted plan"` is disabled: dynamic tool calls within semantic intent scope are admitted with an audit log (`"agent-originated action admitted: duplicate of planned tool, empty args, in scope"`).
  * When `"Only authorize actions in trusted plan"` (`trusted_plan_only: true`) is enabled: dynamic calls not explicitly pre-approved are blocked at runtime as `unplanned_action`.

---

## Context Optimization & Token Pruning (`TokenOptimizer`)

* Prunes low-relevance context chunks scoring below $\tau_{\text{cutoff}} = 0.30$.
* Strictly protects root user goals, constraints (`"do not share"`), and user preferences (formatting/tone directives).
* At cutoff $0.30$, DPVG achieves $R_{\text{tok}} = 0.6243$ (37.6% token reduction) with $100\%$ utility preserved relative to unpruned baselines.
* Banner arithmetic transparently accumulates token reduction and latency across initial plan and re-plan cycles.

---

## Documentation & Comprehensive Manuals

The repository generates two complete, professionally styled Word documents (`.docx`):
1. **[DPVG_Objectives_Evaluation_Report.docx](DPVG_Objectives_Evaluation_Report.docx)**:
   Audit report empirically validating all three project objectives with full benchmark tables, confidence intervals, and overhead profiles.
2. **[DPVG_System_Architecture_and_Overview.docx](DPVG_System_Architecture_and_Overview.docx)**:
   Definitive architectural manual detailing repository structure, cryptographic token specifications, two-stage verification algorithms, and the 11-step request lifecycle.

---

## Repository Layout

```
dpvg/
├── app/
│   ├── gateway.py            # Gateway core, Stage A/B orchestration, drift EMA, admission
│   ├── spt.py                # Ed25519 Semantic Provenance Token engine (RFC 7515 JWS)
│   ├── validator.py          # Stage A cryptographic validator (signatures, lineage, scopes)
│   ├── registry.py           # Vector registry for root intent embeddings & lineage state
│   ├── alignment.py          # Stage B semantic alignment & action-class escalation rules
│   ├── token_opt.py          # Intent-preserving context optimization & token pruning
│   ├── llm_backends.py       # Ollama (Llama 3.2 3B) and Compromised Agent backends
│   ├── agents.py             # PlannerAgent and DelegatedAgent role implementations
│   ├── dataplane.py          # Quarantined execution sandbox & tool registry
│   ├── dataset.py            # Case definitions (s01, p02, w01, w02, t01, hard benign)
│   ├── embeddings.py         # Local all-MiniLM-L6-v2 ONNX embedding inference (~0.22 ms)
│   ├── evaluation.py         # Multi-configuration matrix sweep runner
│   ├── main.py               # FastAPI REST API endpoints
│   └── config.py             # Global security settings and calibrated thresholds
├── frontend/
│   ├── index.html            # Single Page Application (Live Run, Evaluation, Optimizer, Lab)
│   ├── app.js                # Frontend state management, REST client, synchronized tau slider
│   └── style.css             # Professional Deep Navy / Slate responsive design system
├── scripts/
│   ├── benchmark_latency.py  # 250-iteration latency benchmark with warm-up
│   ├── generate_objectives_report.py # Builds DPVG_Objectives_Evaluation_Report.docx
│   ├── generate_overview_report.py   # Builds DPVG_System_Architecture_and_Overview.docx
│   ├── calibrate.py          # ROC / AUC validation split threshold calibration
│   └── evaluate.py           # Benchmark execution suite
├── data/
│   ├── dynamic_tool_use.json # In-house dynamic set (54 cases)
│   ├── hard_benign.json      # Hard benign false-positive test suite
│   ├── drift_scenarios.json  # Multi-step drift evaluation corpus
│   └── test_cases_*.json     # InjecAgent DH/DS Base and Enhanced splits
├── results/
│   ├── latency.md            # Benchmark percentile table (p50, p90, p95, p99, mean)
│   ├── latency_benchmark.json# Raw benchmark timing distributions
│   ├── summary_table.md      # Publication-ready matrix evaluation summary
│   └── *.json                # Empirical evaluation results
├── tests/
│   ├── test_ablation_isolation.py # 39 isolated unit & ablation security tests
│   └── cases_example.json    # Example test cases
├── run.py                    # Server bootstrap (FastAPI on http://127.0.0.1:8000)
├── requirements.txt          # Production dependencies
└── README.md                 # Project documentation
```

---

## Scientific Limitations & Boundaries

1. **Model Scope**: All live evaluation was conducted on Meta's Llama 3.2 (3B). Larger frontier models (70B+) may exhibit different baseline instruction following and injection adherence.
2. **Stage B Soft Semantic Boundary**: Dense cosine similarity alone achieves only $\text{AUC} = 0.2591$ against in-domain attacks because malicious in-domain directives score high similarity (~0.405) with root intent. Separation relies on deterministic **Action-Class Escalation Checks**.
3. **Stage A as Primary Invariant**: Asymmetric cryptographic token scoping (Stage A) and read-only invariants provide the decisive mathematical containment guarantee preventing unauthorized execution even if soft semantic filters are bypassed.
4. **Simulated Tool Side Effects**: Tool execution within `DataPlane` is strictly simulated in memory; no live emails are dispatched or external production systems mutated.

---

## Security Notice

Private Ed25519 signing keys are generated once at startup in memory and never exposed to the Data Plane or LLM context. Do not expose `/api/run` publicly without external authentication: it accepts untrusted user and adversary text by design.
