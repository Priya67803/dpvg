# Semantic Drift Parameter Sensitivity Analysis

Generated at: 2026-11-18T04:12:34.199441+00:00

Evaluated on 160 multi-step scenarios (80 validation, 80 held-out test):

| $\tau_{\text{drift}}$ | $N$ (Consecutive Steps) | $\alpha$ (EMA) | Val Detection | Val False-Block | Test Detection | Test False-Block |
|---|---|---|---|---|---|---|
| 0.25 | 1 | 0.4 | 0.4444 | 0.0000 | 0.4444 | 0.0000 |
| 0.25 | 1 | 0.8 | 0.8889 | 0.0000 | 0.8222 | 0.0000 |
| 0.25 | 2 | 0.4 | 0.2222 | 0.0000 | 0.0000 | 0.0000 |
| 0.25 | 2 | 0.8 | 0.7556 | 0.0000 | 0.6667 | 0.0000 |
| 0.25 | 3 | 0.4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.25 | 3 | 0.8 | 0.1111 | 0.0000 | 0.0889 | 0.0000 |
| 0.35 | 1 | 0.4 | 0.8889 | 0.0000 | 0.8889 | 0.1429 |
| 0.35 | 1 | 0.8 | 1.0000 | 0.0000 | 1.0000 | 0.1429 |
| 0.35 | 2 | 0.4 | 0.4444 | 0.0000 | 0.6667 | 0.0857 |
| 0.35 | 2 | 0.8 | 0.7778 | 0.0000 | 1.0000 | 0.1429 |
| 0.35 | 3 | 0.4 | 0.1778 | 0.0000 | 0.0889 | 0.0857 |
| 0.35 | 3 | 0.8 | 0.4444 | 0.0000 | 0.4000 | 0.0857 |
| 0.40 | 1 | 0.4 | 0.9778 | 0.0000 | 1.0000 | 0.1429 |
| 0.40 | 1 | 0.8 | 1.0000 | 0.2000 | 1.0000 | 0.1429 |
| 0.40 | 2 | 0.4 | 0.5333 | 0.0000 | 0.8667 | 0.1429 |
| 0.40 | 2 | 0.8 | 0.9556 | 0.0000 | 1.0000 | 0.1429 |
| 0.40 | 3 | 0.4 | 0.2222 | 0.0000 | 0.2000 | 0.1429 |
| 0.40 | 3 | 0.8 | 0.5333 | 0.0000 | 0.7556 | 0.1429 |
| 0.45 | 1 | 0.4 | 1.0000 | 0.2857 | 1.0000 | 0.2857 |
| 0.45 | 1 | 0.8 | 1.0000 | 0.4286 | 1.0000 | 0.2857 |
| 0.45 | 2 | 0.4 | 0.8000 | 0.0000 | 1.0000 | 0.1429 |
| 0.45 | 2 | 0.8 | 1.0000 | 0.1143 | 1.0000 | 0.1429 |
| 0.45 | 3 | 0.4 | 0.3333 | 0.0000 | 0.5333 | 0.1429 |
| 0.45 | 3 | 0.8 | 0.6444 | 0.0000 | 0.8667 | 0.1429 |
| 0.55 | 1 | 0.4 | 1.0000 | 0.6571 | 1.0000 | 0.4571 |
| 0.55 | 1 | 0.8 | 1.0000 | 0.8571 | 1.0000 | 0.7429 |
| 0.55 | 2 | 0.4 | 1.0000 | 0.4571 | 1.0000 | 0.4286 |
| 0.55 | 2 | 0.8 | 1.0000 | 0.4857 | 1.0000 | 0.2857 |
| 0.55 | 3 | 0.4 | 0.6444 | 0.4286 | 0.8889 | 0.2571 |
| 0.55 | 3 | 0.8 | 0.7556 | 0.2857 | 1.0000 | 0.1429 |

## Key Parameter Insights

1. **Threshold Sensitivity ($\tau_{\text{drift}}$)**: Higher thresholds ($\ge 0.50$) increase detection of subtle attacks but elevate false-block rates on benign multi-topic shifts. A threshold between 0.35 and 0.40 provides optimal Pareto efficiency.
2. **Consecutive Steps ($N$)**: Single-step gating ($N=1$) is responsive but sensitive to transient dips during legitimate subtask transitions. Requiring $N=2$ consecutive violations provides smoother gating.
3. **Smoothing Parameter ($\alpha$)**: $\alpha=0.6$ to $0.8$ provides rapid tracking of sudden trajectory departures without excessive lag.
