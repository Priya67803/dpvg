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

## Utility cost of plan gating and redesigned extension (`G-AB+P+E`)

The trade-off between plan gating and dynamic utility is evaluated on the in-house AgentDojo-style dynamic set (`data/dynamic_tool_use.json`, expanded to 54 cases, never presented as official AgentDojo):
- 24 benign workflows requiring an unplanned read-only verification tool (`search_web` to verify a citation or document note)
- 15 benign workflows requiring an unplanned side-effect tool authorized by the user prompt (`send_email`, `transfer_money`, `post_slack`)
- 15 injection attacks attempting unauthorized side-effect tool calls via untrusted tool output

The experiment runs under `tau=0.0` to isolate plan gating from Stage B soft score filtering.

Executed by `python scripts/evaluate_plan_gating.py`:

| Mode | Standard benign TCR | Dynamic utility TCR | Attack ASR | Approved extensions | Attack injections admitted |
|---|---:|---:|---:|---:|---:|
| **G-AB** | 1.00 | 1.00 (39/39) | 0.60 (9/15) | 0 | 9 |
| **G-AB+P** | 0.00 | 0.00 (0/39) | 0.00 (0/15) | 0 | 0 |
| **G-AB+S** | 0.00 | 0.00 (0/39) | 0.00 (0/15) | 0 | 0 |
| **G-AB+P+E** | **1.00** | **1.00 (39/39)** | **0.00 (0/15)** | **39** | **0** |

### Redesigned plan extension architecture
In earlier iterations, plan extension simply re-queried the planner with the same unprompted context, yielding 0/2 approved requests.
The redesigned extension mechanism strictly enforces:
1. **Isolated Planner Input:** The planner receives ONLY the original trusted user query, the authorized tool catalog, and the proposed tool name. Untrusted tool outputs, observation payloads, and agent-proposed arguments are never passed to the planner.
2. **Catalog Tool Classification:** Tools are explicitly partitioned into `read_only` and `side_effect`.
3. **Attenuated Capabilities:** Approved read-only tools receive scoped tokens with a short TTL (60s). Approved side-effect tools require the planner to cite the specific user prompt text authorizing the action; argument checks still apply, and an optional human-approval hook is provided.
4. **Audit Logging:** Every extension decision is recorded in the event log and DataPlane ledger.

As measured above, `G-AB+P+E` recovers **100% of benign dynamic utility** (TCR = 1.00) while completely preventing injection-induced privilege escalation (**0.00 ASR**).
Machine-readable output is recorded in [`results/plan_gating_utility.json`](../results/plan_gating_utility.json).

## What this comparison does not establish

- It does not show that DPVG's plan gate is novel relative to planned call graphs or that any one system dominates the others.
- It does not establish equivalent threat models, evaluation settings, model backends, or benchmark scores.
- It does not turn the soft Stage B similarity/rule filter into a hard security guarantee.
- It does not measure real-world tool effects. The built-in tool plane and the dynamic utility workload are simulated.
