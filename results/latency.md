| Operation / Stage | N | Warmup | p50 ms | p95 ms | p99 ms | Mean ms | % of Ollama inference |
|---|---|---|---|---|---|---|---|
| Token issuance (per task) | 250 | 50 | 0.245 | 0.474 | 0.740 | 0.270 | <0.02% |
| Stage A verify (per call) | 250 | 50 | 0.575 | 1.097 | 1.689 | 0.629 | 0.03% |
| Stage B action-class check | 250 | 50 | 0.028 | 0.049 | 0.097 | 0.030 | <0.01% |
| Stage B embedding & cosine | 250 | 50 | 0.034 | 0.075 | 0.153 | 0.042 | <0.01% |
| Stage B total payload verify | 250 | 50 | 0.704 | 1.123 | 1.862 | 0.733 | 0.04% |
| Single-plan total verify | 250 | 50 | 2.992 | 4.999 | 7.059 | 2.994 | 0.08% |
| Re-plan total verify | 250 | 50 | 4.605 | 7.609 | 10.870 | 4.626 | 0.08% |
