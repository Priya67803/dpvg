# DPVG Positioning and Plan-Gating Utility

## Scope and comparison

DPVG, IPIGuard, CaMeL, and DRIFT all constrain how an agent may act after receiving untrusted data, but they enforce different artifacts and make different guarantees. The descriptions below summarize the cited papers; they are not results of a head-to-head experiment.

| System | Main enforcement artifact | Similarity to DPVG | Important difference and boundary |
|---|---|---|---|
| **DPVG** | Ed25519 signed, attenuated tool tokens; optional validated-plan gating, intent-derived scope and argument checks; Stage B rule/semantic filtering | Combines tool authorization with filtering of candidate instructions in tool results and agent handoffs | Stage B is a soft filter. Tool-plan, scope and argument checks are deterministic denials on the instrumented simulated tool path, not an OS-level sandbox or a universal guarantee for tools bypassing the gateway. |
| **IPIGuard** | A planned Tool Dependency Graph (TDG) constrains task execution and tool dependencies | Both can restrict tool use to a plan and reduce actions induced by untrusted observations | IPIGuard's central artifact is a dependency graph. DPVG's plan gate is a set of tools from a validated plan; it does not construct an equivalent graph or establish that a graph-based policy is superior. DPVG adds signed provenance tokens and its separate Stage B filter. |
| **CaMeL** | A program/control-flow plan derived from trusted input, provenance-aware data/control-flow handling, and capability policies | Both separate trusted user intent from untrusted tool data and place enforcement outside the generative model | CaMeL's explicit program and provenance-based flow controls are not equivalent to DPVG's embedding threshold or planned-tool set. DPVG does not claim CaMeL's fine-grained data-flow guarantees. |
| **DRIFT** | Secure Planner trajectory and parameter checklist, Dynamic Validator for deviations, and Injection Isolator for memory-stream content | Both combine planning, runtime checks, and isolation of injected content | DRIFT describes dynamic policy update and memory-stream isolation. DPVG's current optional extension re-plans only from the original trusted request and catalog; it is not DRIFT's dynamic policy mechanism. DPVG uses Ed25519 SPTs and does not reproduce DRIFT's benchmark evaluation. |

Primary references: [IPIGuard (arXiv:2508.15310)](https://arxiv.org/abs/2508.15310), [CaMeL, *Defeating Prompt Injections by Design* (arXiv:2503.18813)](https://arxiv.org/abs/2503.18813), and [DRIFT (arXiv:2506.12104)](https://arxiv.org/abs/2506.12104).

## Proposed Table I wording for IPIGuard

| Defense | Planning / enforcement | Comparison status |
|---|---|---|
| IPIGuard | Plans execution as traversal over a Tool Dependency Graph to constrain tool transitions and decouple planning from untrusted tool interaction. | Related plan-constrained defense; no DPVG/IPIGuard head-to-head run in this repository. Do not copy IPIGuard's reported benchmark numbers into DPVG results. |

## Utility cost of plan gating

`data/dynamic_tool_use.json` contains two benign simulated workflows in which the user asks to read a document and verify a citation/source, but the follow-up search tool is not in the initial rule-planned tool set. The deterministic simulator supplies a citation/source identifier as data and models the agent deciding to use the simulated search tool. The experiment sets `tau=0.0` to isolate tool-plan gating from B-sem's soft score threshold; it is not a production operating point.

Executed by `python scripts/evaluate_plan_gating.py`:

| Mode | Standard benign TCR | Dynamic utility TCR | Initial plans missing a required tool |
|---|---:|---:|---:|
| G-AB | 1.00 | 1.00 (2/2) | 2/2 |
| G-AB+P | 0.00 | 0.00 (0/2) | 2/2 |
| G-AB+S | 0.00 | 0.00 (0/2) | 2/2 |
| G-AB+P+E | 0.00 | 0.00 (0/2) | 2/2 |

These are deterministic fixture measurements, not general estimates of production utility. The G-AB+P+E planner was asked again with the original trusted query only; the rule planner did not add the newly requested tool in either case (0/2 extensions approved). It therefore preserved the plan boundary and did not recover utility in this workload. The extension ablation is switchable as `G-AB+P+E`; an extension is approved only if the trusted planner independently includes the tool in a newly validated plan. Tool output and the agent's proposed arguments are not given to the planner.

The same run measured plan validity at 1.00 (2/2), `PlanValidationError` rate at 0.00 (0/2), and benign tasks with a required tool missing from the initial plan at 1.00 (2/2). The sample is deliberately small and synthetic; it demonstrates the utility trade-off and planner omission metric, not planner reliability at scale. Machine-readable output is in [`results/plan_gating_utility.json`](../results/plan_gating_utility.json).

## What this comparison does not establish

- It does not show that DPVG's plan gate is novel relative to planned call graphs or that any one system dominates the others.
- It does not establish equivalent threat models, evaluation settings, model backends, or benchmark scores.
- It does not turn the soft Stage B similarity/rule filter into a hard security guarantee.
- It does not measure real-world tool effects. The built-in tool plane and the dynamic utility workload are simulated.
