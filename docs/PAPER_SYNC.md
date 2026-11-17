# Paper Sync: Exact Replacement Text

Use the following text to replace the corresponding manuscript passages. It distinguishes implemented mechanisms
from experiments that were not run. The cited measurements are prototype/validation results, not full benchmark
claims.

## Section IV — Architecture and Figure 1 status

> DPVG is implemented as a LangGraph-orchestrated pipeline with a trusted control plane and an instrumented,
> simulated data plane. The implemented modules include the planner and catalog validator, Ed25519 Scoped
> Provenance Token (SPT) issuance and validation, gateway admission checks, Stage B observation filtering, drift
> monitoring, bounded re-planning, and context/token optimization. Figure 1's control/data-plane and verification
> flow corresponds to these modules. Its boxes describe software components and their intended execution order;
> they should not be read as evidence of universal interception, production isolation, or a completed live-model
> evaluation. Tool effects in the reference implementation are simulated.

> The planner receives the original trusted request and the authorized tool catalog. It does not receive tool
> outputs when creating or extending a plan. The optional plan-extension mode asks the planner again using those
> trusted inputs only, validates the resulting plan, and records the decision. Re-planning is bounded. The gateway
> enforces policy only for calls routed through its adapter; integrations that bypass it are outside the enforced
> path.

## Section V.C — Stage B, guarantees, and implementation status

> Stage B consists of two separately ablatable components. **B-rules** applies the configured finite list of
> explicit instruction-override patterns and the supported obfuscation normalization. **B-sem** extracts
> instruction-like candidates and compares their embeddings with the root intent using cosine similarity and
> threshold $\tau$. `G-B-rules` and `G-B-sem` enable each component independently; `G-B` combines them. The
> embedder remains the bundled `all-MiniLM-L6-v2` ONNX model. The override list is documented in the configuration
> and Stage B documentation.

> Stage B is a **soft filter**, not a hard security guarantee. Candidate extraction, finite rules, and embedding
> similarity can miss or misclassify content; similarity is not proof of intent. Deterministic plan, scope, and
> argument checks can reject unauthorized calls only in modes that enable those checks and only on the instrumented
> simulated gateway path.

> Implementation status (Section V.C): SPT signing/validation, gateway checks, the Stage B rule and semantic
> components, drift checks, trusted-input re-planning, and the optional plan extension are implemented and covered
> by tests. These implementation facts do not imply that the complete live-model matrix has been run. In the
> current test run, 189 tests passed and 5 were skipped.

## Section VI — Dataset, split, and threshold calibration

> InjecAgent contains 1,054 raw adversarial rows per setting: 510 Direct Harm and 544 Data Stealing. The repository
> supplies base and enhanced settings, each with 1,054 raw attacks. Because the benchmark rows do not supply clean
> controls, the loader derives two benign cases per attack from its response template: an empty-instruction control
> and a harmless-imperative control. Thus each setting contains 3,162 harness cases (1,054 attacks and 2,108
> derived benign controls); the combined base and enhanced settings contain 6,324 cases.

> The former case-level split leaked repeated attacker instructions and tool groups. The replacement assigns
> complete attacker-tool groups deterministically to validation or test. For each setting, validation and test
> each contain 527 attack rows and 1,054 derived benign controls (1,581 harness cases). The audit checks attacker
> tool, user/attacker tool pair, normalized instruction text, and cross-split near-duplicate text using a
> SequenceMatcher ratio threshold of 0.90. No overlap was found by these checks. The near-duplicate result is a
> heuristic, not a semantic guarantee. See `docs/DATASET_AUDIT.md` for source-file and tool-level counts.

> The semantic threshold is selected on the leak-free **base validation split only**, by maximizing F1 over
> $\tau \in [0.00,1.00]$ in increments of 0.01 subject to $FPR \leq 0.10$. The selected value is $\tau=0.13$
> (precision 0.7275, recall 0.5066, F1 0.5973, FPR 0.0949). The test split is not used for threshold selection.
> This value replaces earlier heuristic values; the embedder was not changed.

## Section VI — Stage B ablation

> On the grouped InjecAgent base validation split, B-rules alone had recall 0.000 and FPR 0.000; B-sem alone at
> $\tau=0.13$ had precision 0.7275, recall 0.5066, F1 0.5973, and FPR 0.0949. On enhanced validation, B-rules
> recall was 1.000 at FPR 0.000 and B-sem recall was 0.6736 at FPR 0.0949. These are validation-only findings.
> Rule-only performance on enhanced data reflects explicit markers in that setting and should not be generalized
> to unmarked attacks. The measured recall at $FPR\leq0.10$ is therefore 0.000 for B-rules and 0.5066 for B-sem
> on base validation; the combined G-B mode is reported separately in the machine-readable ablation results.

> The override-pattern guard acts independently of $\tau$ for candidates matching its configured patterns.
> Matching is not comprehensive: paraphrases, hidden content not exposed by normalization, and unrecognized
> obfuscations may evade it. Stage B is consequently described as a soft filter, while plan/scope/argument
> enforcement supplies the hard denial mechanism for eligible instrumented tool calls.

## Configuration list

> We evaluate twelve modes: `B0`, `B1`, `G-A`, `G-B-rules`, `G-B-sem`, `G-B`, `G-AB`, `G-AB+P`, `G-AB+P+E`,
> `G-AB+S`, `G-AB+T`, and `FULL`. `B0` bypasses defenses; `B1` is the spotlighting baseline; `G-A` enables token
> validation; the three Stage B modes isolate rules, semantics, or both; `G-AB` combines A and B; `G-AB+P` adds
> trusted-plan gating; `G-AB+P+E` adds trusted-input plan extension; `G-AB+S` adds scope and argument enforcement;
> `G-AB+T` isolates token/context optimization; and `FULL` combines A, B, scope, plan, argument, and optimization
> controls. The obsolete phrase "eight configurations alongside B1 (nine total)" must be removed.

## Drift, context pruning, and token accounting

> Drift parameters are selected separately from $\tau$. A sweep over $\tau_{\text{drift}}=0.05$–$0.60$,
> consecutive-step count $N=1$–4, and EMA $\alpha=0.2$–0.8 selected
> $\tau_{\text{drift}}=0.40$, $N=2$, and $\alpha=0.80$. The selection used 11 synthetic validation scenarios
> (six benign and five attacks) and yielded detection 1.00 and false-block rate 0.00 on that small sample. These
> point estimates are not evidence of generalization.

> A context-cutoff prototype sweep covered 0.00–0.60 using 19 validation cases. At cutoff 0.30, 18 chunks were
> retained, mean kept-chunk relevance was 0.3801, and $R_{\mathrm{tok}}=0.7637$; ASR and TCR did not worsen in
> this small evaluation. These results do not establish an optimal cutoff or benchmark-wide utility.

> The total token budget is allocated dynamically across subtasks according to remaining budget and estimated
> subtask size, and prompts are truncated to their allocation. Token reports record their source: Ollama
> `prompt_eval_count`/`eval_count`, hosted usage fields, `tiktoken`, WordPiece, or a documented estimate. Counts
> from fallback estimators are not presented as backend-measured usage.

## Section VI — Plan-gating utility and related-work positioning

> In a two-case simulated dynamic-tool workload, initial plans omitted a required follow-up tool in both cases.
> At $\tau=0$ to isolate gating, G-AB achieved standard and dynamic utility TCR of 1.00, while G-AB+P and G-AB+S
> achieved 0.00. The optional trusted-input extension was approved in 0/2 cases by the deterministic rule planner.
> Plan validity was 2/2, `PlanValidationError` occurred 0/2 times, and the share of benign tasks with a missing
> required tool was 2/2. This fixture is illustrative and is not a general estimate of utility or planner
> reliability.

> **Table I, IPIGuard row:** IPIGuard plans execution as traversal over a Tool Dependency Graph (TDG) to constrain
> tool transitions and separate planning from untrusted tool interaction. DPVG also evaluates plan-constrained
> tool use, but its plan gate is a validated set of tools rather than an equivalent dependency graph. DPVG adds
> signed provenance tokens and a separate Stage B rule/semantic filter. No head-to-head DPVG/IPIGuard experiment
> was conducted, and the comparison does not establish that either approach is superior. See
> `docs/POSITIONING.md` for the fuller comparison with IPIGuard, CaMeL, and DRIFT.

## Phase 6 status and commands

> The full Phase 6 matrix is **skipped: required services/dependencies unavailable in the recorded environment**.
> Ollama was unreachable, `DPVG_HOSTED_API_KEY` and `DPVG_HOSTED_API_MODEL` were unset, and AgentDojo was not
> installed. No live-service matrix results are claimed. Adaptive smoke checks and fixture measurements are not
> substitutes for the matrix.

> To execute the requested matrix on a provisioned Windows PowerShell environment, install
> `requirements-bench.txt`, start the configured Ollama service, set the hosted API credentials/model, and run:

```powershell
python -m pip install -r requirements-bench.txt
$env:DPVG_HOSTED_API_KEY = "<your-key>"
$env:DPVG_HOSTED_API_MODEL = "<your-model>"
python scripts\run_matrix.py --modes B0 B1 G-A G-B-rules G-B-sem G-B G-AB G-AB+P G-AB+P+E G-AB+S G-AB+T FULL --backends ollama hosted vulnerable --workloads demo injecagent_base injecagent_enhanced adaptive drift agentdojo --seeds 17 29 43
```

> If a service or workload is unavailable, the harness must record `skipped: <reason>` rather than fabricate
> results. The exact invocation for regenerating tables from a previously generated matrix file is:

```powershell
python scripts\run_matrix.py --generate-tables-only results\matrix_evaluation.json
```
