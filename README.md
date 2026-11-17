# Dual-Plane Runtime Verification Gateway (DPVG)

Implementation of *"A Zero-Trust, Token-Optimized Defense Against Indirect Prompt Injection using Multi-Agent LLM Systems"*:
FastAPI gateway, browser dashboard, and evaluation harness with InjecAgent data and an AgentDojo adapter. The default agent backend is a local Ollama model; the vulnerable baseline and an optional OpenAI-compatible hosted backend can also be selected. All built-in tool actions remain simulated.

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # real MiniLM (bundled, ONNX) + LangGraph, no torch
python run.py                                              # dashboard + API -> http://127.0.0.1:8000
python -m pytest -q                                        # test suite
python scripts/evaluate.py all                             # regenerate results/*.json  (AgentDojo needs requirements-bench.txt)
# Full Phase 6 command and service prerequisites are listed below.
```

## What changed versus the previous build (the six gaps)

| Gap | Now | Status |
|---|---|---|
| Agent backends | `OllamaBackend` is the default (`DPVG_BACKEND=ollama`). `VulnerableBackend` deterministically follows instruction-like tool output for the B0 no-defense baseline. `HostedAPIBackend` supports an OpenAI-compatible API configured through environment variables. The dashboard can select among them. Hosted prompts/results are sent to the configured endpoint; only use it with data approved for that service. | Ollama and hosted HTTP behavior are tested with mock transports; hosted calls are not made by the test suite. |
| Fallback embedder | **Removed.** The hashing embedder and the automatic `onnx -> sentence-transformers -> hash` chain are gone. `DPVG_EMBEDDER=onnx` (default, bundled MiniLM, SHA-256 checked at load) or `st`; if it cannot load, startup fails. It reproduces the sentence-transformers reference similarities (0.6660 / 0.1046 / 0.1411) and is tested for it. | Done |
| No LangGraph | The pipeline is now explicit steps run by a **LangGraph `StateGraph`** (`optimize -> register -> (orchestrate -> execute_verify -> advance)* -> adversary -> finalize`). `DPVG_EXECUTOR=plain` runs the same steps without it, but only when you ask for it: a missing `langgraph` package is now an error, not a silent downgrade. | Done and verified |
| Scope and ablations | `G-AB+P`, `G-AB+S`, and `G-AB+T` isolate plan, scope/argument, and token-optimization controls. `G-AB+P+E` adds an optional trusted-input plan-extension attempt. | Implemented; dynamic utility numbers are a two-case simulated fixture only. |
| No spotlighting baseline | **B1** implemented (`app/spotlight.py`: delimiting, datamarking, encoding, Hines et al.). A prior standalone Ollama evaluation command is available; the complete Phase 6 matrix was not run in this environment. | Implemented |
| Loaders never run on real data | **InjecAgent:** real files bundled in `data/` and a loader is available. **AgentDojo v1.2.1:** an adapter is available for its message format and Ollama `/v1` endpoint. Full Phase 6 runs require external services and optional packages. | Implemented; current full run unavailable |

## Configuration matrix

| Mode | A | B rules | B semantic | Scope | Trusted plan | Arguments | Plan extension | Token opt. | Spotlight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0` | No | No | No | No | No | No | No | No | No |
| `B1` | No | No | No | No | No | No | No | No | Datamark |
| `G-A` | Yes | No | No | No | No | No | No | No | No |
| `G-B-rules` | No | Yes | No | No | No | No | No | No | No |
| `G-B-sem` | No | No | Yes | No | No | No | No | No | No |
| `G-B` | No | Yes | Yes | No | No | No | No | No | No |
| `G-AB` | Yes | Yes | Yes | No | No | No | No | No | No |
| `G-AB+P` | Yes | Yes | Yes | No | Yes | No | No | No | No |
| `G-AB+P+E` | Yes | Yes | Yes | No | Yes | No | Yes | No | No |
| `G-AB+S` | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No |
| `G-AB+T` | Yes | Yes | Yes | No | No | No | No | Yes | No |
| `FULL` | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | No |

`G-AB+T` is the token-optimization-only ablation. Older Phase 3 reports used an earlier mapping in which that mode
also enabled trusted-plan gating; their token-optimization comparisons are confounded and should not be read as
isolating token optimization.

## Results
The numbers in earlier versions of this README were produced with the removed simulated agent (obeys any readable instruction). Those files are kept for reference in
`results/legacy_simulated/` and are **not** loaded by the dashboard. Regenerate everything with the Ollama model:

```bash
python scripts/calibrate.py                 # tau on the InjecAgent validation split (does not need the LLM for tau selection of Stage B)
python scripts/evaluate.py all              # proto, InjecAgent, spotlight (B0 vs B1), AgentDojo (needs requirements-bench.txt)
```
Each results file records the Ollama model it was produced with. Runs make one or more model calls per case, so a full InjecAgent pass is slow on a laptop; use `--limit N` to sample.

## Stage B: two components, not a hard guarantee
`G-B-rules` detects the explicit override-pattern list and strips/exposes selected obfuscation; `G-B-sem` scores extracted candidates against the root intent and `tau`; `G-B` combines them. The configured patterns are documented in [Stage B decomposition](docs/STAGE_B_DECOMPOSITION.md) and defined in `app/config.py`.

On the grouped InjecAgent base validation split (527 attacks, 1,054 benign), `G-B-rules` alone had recall 0.000/FPR 0.000; B-sem alone at the selected $\tau=0.13$ had precision 0.7275, recall 0.5066, F1 0.5973, FPR 0.0949. On enhanced validation, B-rules recall was 1.000/FPR 0.000, while B-sem recall was 0.6736/FPR 0.0949. These are validation findings, not test results. The enhanced set's explicit markers make rule-only recall unusually high; that does not generalize to unmarked attacks.

Stage B is a **soft filter**: semantic similarity can miss instructions and the rule list is finite. Plan, scope, and argument checks provide deterministic hard denials only for calls routed through the instrumented simulated gateway path.

`python scripts/evaluate.py spotlight` now runs B1 with delimit, datamark and encode next to B0, G-AB and FULL.

## Token optimization
`G-AB+T` (token optimization only) and `FULL` rank context separately for each sub-task and drop chunks below the configurable cosine cutoff
(`DPVG_CONTEXT_SIMILARITY_CUTOFF`, default `0.30`). `DPVG_TOKEN_BUDGET` (default `3000`) is distributed across
sub-tasks dynamically from remaining budget and estimated task size; prompt content is truncated to each allocation and Ollama's generation limit is capped accordingly.
Run `python scripts/run_phase4_token_opt_ablation.py --split test` to compare G-AB with G-AB+T on ASR, TCR and
`R_tok = 1 - T_opt/T_base`. Ollama usage uses `prompt_eval_count`/`eval_count`, hosted usage fields are preferred, and fallbacks are labeled as `tiktoken`, `wordpiece`, or `documented-estimate`. The context-cutoff sweep covered 0.00–0.60 on 19 prototype validation cases: ASR stayed 0 and benign/attack TCR stayed 1.00 at every tested cutoff; at cutoff 0.30, `R_tok=0.7637`, 18 chunks were kept, and mean kept-chunk relevance was 0.3801. This small prototype sweep does not establish a general optimum.

## Phase 4: Semantic drift detection, re-planning feedback loop, and strict mode
Action alignment across multi-step trajectories is tracked against the root user intent using an Exponential Moving Average (EMA) or running mean. The validation sweep on `data/drift_scenarios.json` used 11 synthetic validation scenarios (6 benign, 5 attacks), searched `tau_drift=0.05..0.60`, `N=1..4`, and `alpha=0.2..0.8`, and selected `tau_drift=0.40`, `N=2`, `alpha=0.80` at detection 1.00 and false-block rate 0.00. The sample is synthetic and small; these are calibrated defaults for the prototype, not field-validated thresholds:
$$\text{DriftScore}_t = \alpha \cdot \text{Sim}(a_t, \text{goal}) + (1-\alpha) \cdot \text{DriftScore}_{t-1}$$
- **Drift Detection:** If the score falls below `DPVG_DRIFT_THRESHOLD` (default `0.40`) for `DPVG_DRIFT_CONSECUTIVE_STEPS` (default `2`) consecutive steps, execution is blocked. Configure via `DPVG_DRIFT_METRIC=ema|mean` and `DPVG_DRIFT_EMA_ALPHA=0.80`.
- **Bounded Re-Planning:** On a verification block, the gateway initiates a bounded re-plan cycle (up to `DPVG_MAX_REPLANS`, default `1` retry) through intent optimization (Step 1), root re-registration (Step 2), and orchestration (Step 3).
  - *Trusted Input Invariant:* Re-planning operates **strictly on trusted inputs** (the original user prompt and authorized catalog); quarantined tool outputs or adversarial payloads are never admitted into the planner.
  - *Safe Fallback:* If re-planning retries are exhausted without successful verification, the gateway falls back to safe fallback remainder delivery.
- **Strict Mode (`DPVG_STRICT_MODE=true`):** In contrast to the default candidate-level quarantine (which redacts only the malicious directive and passes safe surrounding text), strict mode enforces fail-stop zero-trust:
  - Any detected misaligned directive or prompt injection marker quarantines the **entire payload** (`⟦redacted: whole payload quarantined (strict mode)⟧`).
  - Appends an audit record with detection point, stage, rejection code, reason, and similarity score.
  - Immediately raises `RuntimeBlockException`, halting execution to eliminate second-order leakage.

## Phase 5: LLM planner, tool catalog schema gating, and per-role models
The orchestration engine supports both an LLM-backed planner (`LLMPlanner`) and a deterministic rule-based planner (`RulePlanner`), selectable via `DPVG_PLANNER_TYPE=llm|rule`:
- **Strict Tool Catalog Gating:** The LLM planner receives ONLY the root user request and the declared catalog of authorized tools (`ALL_TOOLS` or case-specific `allowed_tools`). If the plan references a tool outside the catalog, the gateway strictly raises `PlanValidationError` and fails closed without falling back to rule planning.
- **Rule Fallback:** On transient JSON formatting or parsing errors (distinct from catalog violations), the planner falls back to `RulePlanner` when configured.
- **Prompt Isolation Invariant:** The planner **never receives tool output or observation data**, preventing indirect prompt injections from influencing subtask planning.
- **Least-Privilege Scope Derivation:** Root token scope $S(\text{root})$ is derived strictly from the validated plan ($S(\text{root}) = \text{tools}(\text{Plan}_{\text{validated}}) \cap \text{Catalog}$). Invocations of ungranted tools (e.g., `send_email` when the user only asked to search the web) fail closed at Stage A (`StageAResult(False, "scope")`).
- **Per-Role Models:** Each operational role can be mapped to an independent backend model via environment variables or runtime configuration:
  1. `planner` (`llm-d`): Subtask planning (`DPVG_BACKEND_PLANNER`)
  2. `tool` (`llm-a`): Standard tool operations (`DPVG_BACKEND_TOOL`)
  3. `reasoning` (`llm-b`): Text summarization and handoffs (`DPVG_BACKEND_REASONING`)
  4. `specialized` (`llm-c`): High-privilege side-effect execution (`DPVG_BACKEND_SPECIALIZED`)
  All roles default to the shared `DPVG_BACKEND` (`ollama|vulnerable|hosted`).

### Plan-gating utility cost and trusted extension
The two-case dynamic-tool fixture is evaluated by `python scripts/evaluate_plan_gating.py`. At `tau=0.0` to isolate plan gating, standard/dynamic utility TCR was 1.00/1.00 for G-AB and 0.00/0.00 for G-AB+P and G-AB+S (2 cases per mode). Initial plans omitted a required follow-up tool in both cases. Plan-extension mode `G-AB+P+E` re-runs the planner on the original user request only; it approved 0/2 requests with the deterministic rule planner and had utility TCR 0.00. These fixture measurements are not representative production estimates. See [POSITIONING.md](docs/POSITIONING.md) for scope and related work.

## Phase 6: Reproducible evaluation harness (`scripts/run_matrix.py`)
`scripts/run_matrix.py` coordinates multi-seed, multi-backend, multi-workload matrix evaluation:
```bash
python scripts/run_matrix.py --modes B0 B1 G-A G-B-rules G-B-sem G-B G-AB G-AB+P G-AB+P+E G-AB+S G-AB+T FULL --backends ollama hosted vulnerable --workloads demo injecagent_base injecagent_enhanced adaptive drift agentdojo --seeds 17 29 43
python scripts/run_matrix.py --generate-tables-only results/matrix_evaluation.json
```
- **Dimensions:** Twelve modes (the core defenses, Stage B ablations, and `G-AB+P+E`) $\times$ three backends $\times$ six workloads, including held-out multi-step drift scenarios $\times$ three seeds.
- **Statistics and provenance:** Reports mean $\pm$ standard deviation, Wilson intervals for ratio metrics (the F1 interval is explicitly labeled as an effective-count approximation), latency deltas, `R_tok`, prompt/completion tokens and counter sources. Each run records backend/model, tau, drift parameters, seed, git revision or `unversioned`, and timestamp.
- **Fine-grained results:** Emits scenario, attack-type, and per-layer ("who stopped it") Markdown/CSV tables. Adaptive cases preserve individual `executed` and `stopped_by` outcomes, including successful attacks.
- **Skipped services:** The full matrix was not run here: Ollama was unreachable, hosted API credentials/model were unset, and `agentdojo` was not installed. Run it only after satisfying the prerequisites; unavailable workloads are recorded as `skipped: reason`.
- **Run Artifacts & Clean Skipping:** Saves individual run JSONs in `results/runs/` with git commit, seed, backend, model, tau, and timestamp. Skips unavailable workloads (e.g., uninstalled `agentdojo` or unconfigured endpoints) with clear `"skipped: reason"` records instead of failing or writing placeholder data.
- **Summary Tables:** Compiles publication-ready tables in Markdown and CSV (`results/summary_table.{md,csv}` and `results/scenarios_table.{md,csv}`).

## Validation calibration method ($\tau = 0.13$)
The threshold is selected on the leak-free InjecAgent base validation split only: 1,581 harness cases (527 attacks, 1,054 benign controls), grouped by attacker tool. The `[0.00, 1.00]` grid uses 0.01 increments and maximum `FPR=0.10`; best validation F1 satisfying the cap selects $\tau=0.13$ (precision 0.7275, recall 0.5066, F1 0.5973, FPR 0.0949). The test split is untouched. InjecAgent has 1,054 raw attacks per setting; the loader creates two matched benign variants per raw row for 3,162 harness cases. See [DATASET_AUDIT.md](docs/DATASET_AUDIT.md) and `calibration.json`.

## What remains untested
To maintain rigorous scientific standards, the following empirical boundaries must be explicitly noted:
1. **Full Phase 6 matrix:** Not run: Ollama was unreachable, `DPVG_HOSTED_API_KEY` / `DPVG_HOSTED_API_MODEL` were unavailable, and AgentDojo was not installed. Do not report absent runs as measured; check `results/matrix_evaluation.json` only after a real run.
2. **Live AgentDojo benchmark suite:** The adapter is implemented, but live `banking`, `slack`, and `travel` runs were skipped because the optional package and model services were unavailable.
3. **Hosted API backend in production:** `HostedAPIBackend` is tested with mock HTTP transports but has not been benchmarked against a live hosted endpoint.
4. **Real-world physical side effects:** All tool executions in `DataPlane` are strictly simulated in memory; no real emails are dispatched, no real file systems altered, and no actual financial transactions executed.

## Caveats you should state in the paper
1. **Attack-success numbers depend on the selected backend.** The deterministic vulnerable backend is intended for reproducible B0 measurements; model-backed runs should report the backend and model name.
2. The Phase 6 adaptive set is a deterministic, task-framed augmentation of InjecAgent, not a human-authored or LLM-generated adversarial red-team set; report it as an augmentation, not a new held-out benchmark.
3. The "benign_hard" set and the 35-case set are author-written. FPR numbers on them are illustrative.
4. The Stage B rules were developed after examining AgentDojo, so AgentDojo is not an untouched evaluation. Tau selection uses InjecAgent base validation data only; do not tune against the test split.
5. "Safe fallback" in gateway messages means the defense's own behaviour (quarantine the misaligned sentence, deliver the redacted remainder), not a substitute model.
6. The test suite uses `tests/fake_ollama.py`, an HTTP-level stand-in that obeys injected text. It tests the plumbing and the gateway, not any real model.

## Layout
```
app/  registry spt validator alignment token_opt orchestrator dataplane gateway(+LangGraph) agents llm_backends(Ollama, vulnerable, hosted) spotlight
      embeddings(MiniLM ONNX) benchmarks(InjecAgent loader, AgentDojo adapter+runner) evaluation main(API)
models/all-MiniLM-L6-v2/  bundled Apache-2.0 weights (fp32 ONNX, SHA-256 verified)     data/  real InjecAgent files (MIT)
results/*.json  calibration.json  frontend/  scripts/{evaluate,calibrate,download_injecagent,fetch_model,run_phase4_token_opt_ablation,run_matrix}.py  tests/
```
Env vars: `DPVG_BACKEND=ollama|vulnerable|hosted` (default `ollama`), `DPVG_OLLAMA_URL` (default `http://localhost:11434`), `DPVG_OLLAMA_MODEL` (default `llama3.2`), `DPVG_HOSTED_API_URL`, `DPVG_HOSTED_API_KEY`, `DPVG_HOSTED_API_MODEL`, `DPVG_LLM_TIMEOUT`, `DPVG_LLM_TEMPERATURE` (default `0`), `DPVG_LLM_SEED`, `DPVG_TAU`, `DPVG_TOKEN_BUDGET`, `DPVG_CONTEXT_SIMILARITY_CUTOFF`, `DPVG_DRIFT_THRESHOLD`, `DPVG_DRIFT_CONSECUTIVE_STEPS`, `DPVG_DRIFT_EMA_ALPHA`, `DPVG_DRIFT_METRIC=ema|mean`, `DPVG_MAX_REPLANS`, `DPVG_STRICT_MODE=true|false`, `DPVG_PLANNER_TYPE=llm|rule`, `DPVG_BACKEND_PLANNER`, `DPVG_BACKEND_TOOL`, `DPVG_BACKEND_REASONING`, `DPVG_BACKEND_SPECIALIZED`, `DPVG_EMBEDDER=onnx|st`, `DPVG_EXECUTOR=langgraph|plain`.

## Security notes
Signing keys are generated per run in memory. Ollama runs locally by default; the hosted backend reads its optional API key from the environment. Tools are simulated: nothing real is sent, deleted or paid.
Do not expose `/api/run` publicly without authentication: it accepts attacker-controlled text by design.
