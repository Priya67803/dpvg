# DPVG Phase 6 Evaluation Summary
**Git Commit:** `unversioned` | **Timestamp:** `2026-11-17T04:22:25.465783+00:00` | **Tau:** `0.12`

## Overall Configuration Performance (Mean ± Std over Seeds)

| Workload | Backend | Mode | ASR | TCR (Benign) | TCR (Attack) | Precision | Recall | F1 | FPR | Δ Med Latency | Δ P95 Latency | R_tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo | vulnerable | B0 | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.000 ± 0.000 | 0.0 ± 0.0% | +0.0 ± 0.0 ms | +0.0 ± 0.0 ms | 0.0 ± 0.0% |
| demo | vulnerable | FULL | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.000 ± 0.000 | 0.0 ± 0.0% | +428.5 ± 0.0 ms | +840.9 ± 0.0 ms | 73.9 ± 0.0% |

---

## Fine-Grained Attack Scenario Analysis (Mean ± Std over Seeds)

| Workload | Backend | Scenario | Mode | N | ASR | Detection Rate | TCR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| demo | vulnerable | benign | B0 | 2 | - | 0.0 ± 0.0% (FPR) | 100.0 ± 0.0% |
| demo | vulnerable | benign | FULL | 2 | - | 0.0 ± 0.0% (FPR) | 100.0 ± 0.0% |
