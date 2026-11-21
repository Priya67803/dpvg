# Paper Sync: Exact Replacement Text

Use the following text to replace the corresponding manuscript passages. It distinguishes implemented mechanisms
from experiments that were not run. The cited measurements are prototype and benchmark results generated directly
from `results/` in this repository; no numbers are placeholder or fabricated.

## Section IV — Architecture and Figure 1 Status

> DPVG is implemented as a LangGraph-orchestrated pipeline with a trusted control plane and an instrumented,
> simulated data plane. The implemented modules include the planner and catalog validator, Ed25519 Scoped
> Provenance Token (SPT) issuance and validation, gateway admission checks, Stage B observation filtering, drift
> monitoring, bounded re-planning, and context/token optimization. Figure 1's control/data-plane and verification
> flow corresponds to these modules. Its boxes describe software components and their intended execution order;
> they should not be read as evidence of universal interception, production isolation, or a completed live-model
> evaluation. Tool effects in the reference implementation are simulated.

> The planner receives the original trusted request and the authorized tool catalog. It does not receive tool
> outputs when creating or extending a plan. The redesigned plan-extension mode (`G-AB+P+E`) asks the planner using
> those trusted inputs and the proposed tool name only, validates the resulting plan, and records the decision in
> the event log and ledger. Approved read-only tools are granted short-TTL tokens (60s); side-effect tools require
> citing the authorizing user request, argument checks still apply, and an optional human-approval hook is provided.
> Re-planning is bounded. The gateway enforces policy only for calls routed through its adapter; integrations that
> bypass it are outside the enforced path.

## Section V.C — Stage B, Guarantees, and Implementation Status

> Stage B consists of two separately ablatable components. **B-rules** applies the configured finite list of
> explicit instruction-override patterns and the supported obfuscation normalization. **B-sem** extracts
> instruction-like candidates and compares their embeddings with the root intent using cosine similarity and
> threshold $\tau$. `G-B-rules` and `G-B-sem` enable each component independently; `G-B` combines them. The
> embedder remains the bundled `all-MiniLM-L6-v2` ONNX model. The override list is documented in the configuration
> and Stage B documentation.

> Stage B is a **soft filter**, not a hard security guarantee. Candidate extraction, finite rules, and embedding
> similarity can miss or misclassify content; similarity is not proof of intent. In our evaluation, adaptive
> obfuscations (`hidden_html_comment`, `hidden_text`, `hidden_zero_width`) bypassed B-sem alone with 70–90% ASR
> under `G-B-sem`. Deterministic plan, scope, and argument checks provide the hard denial layer, reducing ASR to
> 0.00% under `FULL` across all 11 adaptive attack types on the instrumented simulated gateway path.

> Implementation status (Section V.C): SPT signing/validation, gateway checks, Stage B rule and semantic
> components, drift checks, trusted-input re-planning, redesigned plan extension, and context pruning are implemented
> and covered by unit tests. In the current test run, **210 tests passed and 5 were skipped**.

## Section VI — Dataset, Split, and Threshold Calibration

> InjecAgent contains 1,054 raw adversarial rows per setting: 510 Direct Harm and 544 Data Stealing. The repository
> supplies base and enhanced settings, each with 1,054 raw attacks. Because the benchmark rows do not supply clean
> controls, the loader derives two benign cases per attack from its response template: an empty-instruction control
> and a harmless-imperative control. Thus each setting contains 3,162 harness cases (1,054 attacks and 2,108
> derived benign controls); the combined base and enhanced settings contain 6,324 cases.

> The former case-level split leaked repeated attacker instructions and tool groups. The replacement assigns
> complete attacker-tool groups deterministically to validation or test. For each setting, validation and test
> each contain 527 attack rows and 1,054 derived benign controls (1,581 harness cases). The audit checks attacker
> tool, user/attacker tool pair, normalized instruction text, and cross-split near-duplicate text using a
> SequenceMatcher ratio threshold of 0.90. No overlap was found by these checks. See `docs/DATASET_AUDIT.md` for
> source-file and tool-level counts.

> The semantic threshold is selected on the leak-free **base validation split only**, by maximizing F1 over
> $\tau \in [0.00,1.00]$ in increments of 0.01 subject to $\text{FPR} \leq 0.10$. The selected value is $\tau=0.13$
> (precision 0.7275, recall 0.5066, F1 0.5973, FPR 0.0949). The test split is not used for threshold selection.
> This value replaces earlier heuristic values; the embedder was not changed.

### Tau Sensitivity Sweep

> Evaluating Stage B across $\tau \in \{0.03, 0.08, 0.13, 0.18, 0.23\}$ on the base validation split demonstrates
> the trade-off between recall and false positives:
>
> | $\tau$ | Recall | ASR (B-sem) | FPR | Precision | F1 |
> |---|---:|---:|---:|---:|---:|
> | 0.03 | 0.1537 | 0.8463 | 0.0028 | 0.9643 | 0.2651 |
> | 0.08 | 0.3131 | 0.6869 | 0.0351 | 0.8168 | 0.4527 |
> | **0.13** | **0.5066** | **0.4934** | **0.0949** | **0.7275** | **0.5973** |
> | 0.18 | 0.6945 | 0.3055 | 0.1490 | 0.6998 | 0.6971 |
> | 0.23 | 0.8178 | 0.1822 | 0.1898 | 0.6830 | 0.7444 |
>
> While higher $\tau$ improves raw recall, it causes unacceptable false positive rates ($\text{FPR} = 14.90\%$ at
> $\tau=0.18$ and $18.98\%$ at $\tau=0.23$). Threshold $\tau=0.13$ represents the optimal F1 point under the
> $\text{FPR} \leq 0.10$ budget constraint. Full tabular data and per-attack breakdown are recorded in
> `results/tau_sensitivity.json` and `results/tau_sensitivity.md`.

### Demo Margin & Near-Threshold Caveat

> Candidate alignment is evaluated relative to $\tau$ via margin $\Delta = \text{Sim}(c, \text{goal}) - \tau$.
> In the demo scenario ("refund policy + passport exfiltration"), the injected passport exfiltration directive
> achieves a similarity score of 0.637 against the user request to search for the refund policy, exceeding $\tau=0.13$
> by margin $+0.507$. Consequently, B-sem alone labels it as aligned. The dashboard flags any candidate with
> $|\Delta| < 0.03$ as "near threshold" and labels on-topic instruction-like text as "aligned but instruction-like (sanitized)".
> Containment of this attack requires the hard enforcement layers (`FULL`), which deny the unapproved `send_email`
> action via token scope and recipient argument checks.

## Section VI — Stage B Hard-Benign Evaluation

> Beyond derived benchmark controls, Stage B was evaluated against `data/hard_benign.json`, a 325-item dataset
> comprising realistic benign tool outputs that contain imperatives, polite requests, recipes, meeting agendas,
> TODO lists, instructions meant for human readers, and quoted commands. Split by group into 160 validation and
> 165 test cases:
>
> | Dataset Split | B-rules FPR | B-sem FPR ($\tau=0.13$) | Combined Stage B FPR |
> |---|---:|---:|---:|
> | Hard-Benign (Validation, $N=160$) | 0.0000 | 0.0688 (6.88%) | 0.0688 (6.88%) |
> | Hard-Benign (Test, $N=165$) | 0.0000 | 0.0788 (7.88%) | 0.0788 (7.88%) |
> | InjecAgent Base (Test, $N=1054$) | 0.0000 | 0.0028 (0.28%) | 0.0028 (0.28%) |
>
> On held-out test data, Stage B FPR on hard benign text is **7.88%** (compared to 0.28% on InjecAgent base controls).
> False positive rates are concentrated in `todo_lists` (26.09%) and `imperatives` (20.00%), where benign tool
> outputs mimic imperative instruction syntax. Under `FULL`, overall task completion rate on hard benign data was
> **0.9879** (163/165 completed), compared to 1.0000 under `B0`. Full category breakdowns are in
> `results/hard_benign_results.json` and `results/hard_benign_summary.md`.

## Section VI — Multi-Step Trajectory Drift

> Multi-step drift detection was evaluated on `data/drift_scenarios.json`, expanded to 160 synthetic multi-step
> trajectories (80 validation, 80 test) across five behavioral categories: benign long tasks, benign topic shifts,
> gradual drift, sudden jump attacks, and slow-drip injections.
>
> Calibration on the **validation split only** selected $\tau_{\text{drift}}=0.35$, consecutive violations $N=1$,
> and EMA $\alpha=0.80$, achieving validation detection 1.000 and validation false-block rate 0.000.
> On the **held-out test split** ($N=80$), these calibrated parameters achieved:
> - **Test Detection Rate:** 1.0000 (100% of drift attacks detected)
> - **Test False-Block Rate:** 0.1429 (14.29% false blocks on benign multi-step tasks)
>
> The legacy 11-scenario calibration ($\tau_{\text{drift}}=0.40, N=2$) is explicitly **superseded** in
> `calibration.json` (`drift_legacy_11_scenarios`). Full parameter sensitivity sweeps across $\tau_{\text{drift}} \in [0.10, 0.60]$,
> $N \in \{1, 2, 3\}$, and $\alpha \in [0.2, 0.9]$ are archived in `results/drift_sensitivity.json` and
> `results/drift_sensitivity.md`.

## Section VI — Redesigned Plan Extension (`G-AB+P+E`)

> To address plan-gating utility loss, the plan-extension mechanism was redesigned with strict prompt isolation:
> extension requests supply the planner with **only** the original trusted query, the authorized tool catalog, and the
> requested tool name. Tool outputs, proposed arguments, and untrusted observations are strictly excluded.
> Tools in the catalog are classified as `read_only` or `side_effect`. Approved read-only tools receive scoped tokens
> with a short TTL (60s); approved side-effect tools require the planner to cite the specific phrase in the user
> request that authorizes the tool, followed by standard argument checks and an optional human-approval hook.
>
> Evaluated on the in-house AgentDojo-style dynamic set (`data/dynamic_tool_use.json`, not official AgentDojo: 24 read-only lookups,
> 15 authorized side-effects, 15 attack injections):
>
> | Configuration | Dynamic Utility TCR | ASR | Approved Extensions | Attack Injections Admitted |
> |---|---:|---:|---:|---:|
> | `G-AB` | 1.0000 (39/39) | 0.6000 (9/15) | 0 | 9 |
> | `G-AB+P` | 0.0000 (0/39) | 0.0000 (0/15) | 0 | 0 |
> | `G-AB+S` | 0.0000 (0/39) | 0.0000 (0/15) | 0 | 0 |
> | `G-AB+P+E` | **1.0000 (39/39)** | **0.0000 (0/15)** | **39** | **0** |
>
> `G-AB+P+E` recovers **100% of benign dynamic utility** (TCR = 1.0000) while maintaining **0.0000 ASR**, approving
> all 39 legitimate extensions and rejecting all 15 injected tool invocations. Detailed logs are in
> `results/plan_gating_utility.json`.

## Section VI — Context Pruning and Token Optimization

> Context pruning in `TokenOptimizer` preserves a protected class (`user_preference`) alongside intent, policy,
> and constraints. Language, formatting, and style preferences (e.g., "The user prefers concise answers written in
> British English") are unconditionally retained during context pruning, while irrelevant distractors (weather,
> sports, kitchen notes) are pruned based on cosine similarity and budget constraints.
>
> Dynamic token budgeting distributes the total budget across decomposed subtasks; when total budget is smaller
> than subtask needs, non-protected context is truncated, and dropped chunks are logged in `dropped_context`.
>
> A context cutoff sweep over $\text{cutoff} \in [0.00, 0.60]$ evaluated on **200 validation cases** (prototype
> plus InjecAgent base) with `vulnerable` backend and WordPiece counting yielded:
> - **G-AB Baseline:** ASR = 0.0000, TCR (Benign) = 0.8879, TCR (Attack) = 1.0000, $R_{\text{tok}} = 0.0000$.
> - **At $\text{cutoff}=0.30$ (`G-AB+T`):** 275 kept chunks, mean relevance 0.1674, $R_{\text{tok}} = 0.6243$
>   (62.43% token reduction), ASR = 0.0000, TCR (Benign) = 0.8879 (100% utility preserved relative to baseline).
>
> Token counter sources are explicitly recorded across all results: `tiktoken` (cl100k_base), `wordpiece`
> (`tokenizers`), or heuristic approximation. Optional installation of `requirements-ml.txt` enables exact `tiktoken`
> and PyTorch sentence-transformers. Full sweep results are in `results/context_cutoff_sweep.json` and
> `results/context_cutoff_sweep.md`.

## Configuration List

> We evaluate twelve modes: `B0`, `B1`, `G-A`, `G-B-rules`, `G-B-sem`, `G-B`, `G-AB`, `G-AB+P`, `G-AB+P+E`,
> `G-AB+S`, `G-AB+T`, and `FULL`. `B0` bypasses defenses; `B1` is the spotlighting baseline; `G-A` enables token
> validation; the three Stage B modes isolate rules, semantics, or both; `G-AB` combines A and B; `G-AB+P` adds
> trusted-plan gating; `G-AB+P+E` adds trusted-input plan extension; `G-AB+S` adds scope and argument enforcement;
> `G-AB+T` isolates token/context optimization; and `FULL` combines A, B, scope, plan, argument, and optimization
> controls. The obsolete phrase "eight configurations alongside B1 (nine total)" must be removed.

## Table I — Related Work Positioning (IPIGuard Row)

| Defense | Planning / Enforcement | Comparison Status |
|---|---|---|
| **IPIGuard** | Plans execution as traversal over a Tool Dependency Graph (TDG) to constrain tool transitions and separate planning from untrusted tool interaction. | Related plan-constrained defense; evaluated on different benchmarks. DPVG enforces a validated set of planned tools with cryptographic SPTs and Stage B filtering rather than a dependency graph. No head-to-head empirical run was conducted. Do not copy IPIGuard benchmark numbers into DPVG results. |

## Experiments Not Run and Environment Limitations

> The full matrix for live model backends is **skipped: required services/dependencies unavailable in the recorded environment**.
> Specifically:
> 1. **Ollama backend:** Skipped because the local Ollama daemon at `http://localhost:11434` was unreachable.
> 2. **Hosted API backend:** Skipped because `DPVG_HOSTED_API_KEY` and `DPVG_HOSTED_API_MODEL` were unset.
> 3. **AgentDojo workload:** Skipped because the `agentdojo` package and external tool environments were not installed.
>
> All reported matrix numbers were produced using the deterministic `vulnerable` backend on the test split across
> seeds 17, 29, and 43, logging Wilson 95% confidence intervals, per-layer attributions, and exact token counter
> sources in `results/summary_table.md` and `results/scenarios_table.md`.

> To run the live matrix on a provisioned machine with Ollama and hosted backends:
>
> ```powershell
> python -m pip install -r requirements-bench.txt
> $env:DPVG_HOSTED_API_KEY = "<your-key>"
> $env:DPVG_HOSTED_API_MODEL = "gpt-4o-mini"
> python scripts\run_matrix.py --backends ollama hosted vulnerable --workloads injecagent_base injecagent_enhanced adaptive drift demo --limit 100 --sample-seed 42 --seeds 17 29 43 --yes
> ```
