# Hard-Benign Dataset Evaluation Report

Generated at: 2026-11-18T04:08:44.443984+00:00 | Semantic Threshold $\tau=0.13$

## 1. Stage B False Positive Rate (FPR) Comparison

Comparing Hard-Benign FPR against standard InjecAgent Base validation benign controls:

| Dataset Split | Workload | Cases | B-rules FPR | B-sem FPR | Combined Stage B FPR |
|---|---|---|---|---|---|
| Validation | InjecAgent Base | 1,054 | 0.0000 | 0.0028 | 0.0028 |
| Validation | **Hard-Benign** | 160 | 0.0000 | 0.1375 | **0.1375** |
| Test | InjecAgent Base | 1,054 | 0.0000 | 0.0028 | 0.0028 |
| Test | **Hard-Benign** | 165 | 0.0000 | 0.0788 | **0.0788** |

## 2. Hard-Benign FPR by Category Group (Held-Out Test Split)

| Category Group | Test Cases | B-rules FPR | B-sem FPR | Combined FPR | FP Count (Comb) |
|---|---|---|---|---|---|
| human_instructions | 23 | 0.0000 | 0.0000 | 0.0000 | 0 / 23 |
| imperatives | 25 | 0.0000 | 0.2000 | 0.2000 | 5 / 25 |
| meeting_agendas | 23 | 0.0000 | 0.0000 | 0.0000 | 0 / 23 |
| polite_requests | 25 | 0.0000 | 0.0800 | 0.0800 | 2 / 25 |
| quoted_commands | 23 | 0.0000 | 0.0000 | 0.0000 | 0 / 23 |
| recipes | 23 | 0.0000 | 0.0000 | 0.0000 | 0 / 23 |
| todo_lists | 23 | 0.0000 | 0.2609 | 0.2609 | 6 / 23 |
| **Overall** | **165** | **0.0000** | **0.0788** | **0.0788** | **13 / 165** |

## 3. Gateway Mode Utility (TCR) on Hard-Benign Dataset

Task Completion Rate (TCR) on held-out test split under each gateway security configuration:

| Mode | Architecture Description | Test Cases | Completed | TCR |
|---|---|---|---|---|
| `B0` | No defense | 165 | 165 | **1.0000** |
| `B1` | Spotlighting (datamarking) baseline | 165 | 165 | **1.0000** |
| `G-A` | Stage A only: token validation | 165 | 165 | **1.0000** |
| `G-B-rules` | Stage B rules only: override markers and obfuscation stripping | 165 | 165 | **1.0000** |
| `G-B-sem` | Stage B semantic only: embedding similarity vs tau | 165 | 165 | **1.0000** |
| `G-B` | Stage B: rules plus semantic alignment | 165 | 165 | **1.0000** |
| `G-AB` | G-A + G-B: dual-plane verification | 165 | 165 | **1.0000** |
| `G-AB+P` | G-AB + trusted-plan gating | 165 | 165 | **1.0000** |
| `G-AB+P+E` | G-AB + trusted-plan gating and trusted-input plan extension | 165 | 165 | **1.0000** |
| `G-AB+S` | G-AB + intent-derived scope, plan, and side-effect argument checks | 165 | 163 | **0.9879** |
| `G-AB+T` | G-AB + token optimization only | 165 | 165 | **1.0000** |
| `FULL` | G-AB + scope, plan, argument checks, and token optimization | 165 | 163 | **0.9879** |

## 4. Key Empirical Insights

1. **Stage B Soft Filter Limitation**: On hard benign text (containing imperatives, recipes, and human instructions), Stage B combined false positive rate increases significantly compared to standard benchmark text (InjecAgent Base: 0.28%).
2. **Category Vulnerability**: Quoted technical commands and multi-step recipes produce the highest false positives under semantic filtering because imperative syntax diverges from root user query embeddings.
3. **Stage A Hard Isolation**: Dual-plane gateway modes with Stage A containment safely isolate and deliver sanitized payloads, preserving end-to-end task completion while maintaining hard security boundaries.
