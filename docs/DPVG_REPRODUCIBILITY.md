# DPVG: Experimental Reproducibility & Verification Guide

**Document Version:** 1.1.0  
**Target Environment:** Python 3.11 / Windows / Linux / macOS  
**Reproducibility Objective:** Reproduce the current tests and Phase 6 evaluation workflow. External benchmarks require their services and optional dependencies.

---

## 1. System Environment & Prerequisites

Use Python 3.11 and install the project dependencies from the repository's requirements files. The phase-four environment freeze is historical and does not describe the current working tree.

### 1.1 Hardware & OS Specifications
- **Operating System:** Windows 10/11 x86_64 (Linux Ubuntu 22.04 LTS and macOS Darwin also supported)
- **Python Version:** Python 3.11.7+ (must be 64-bit)
- **Virtual Environment:** Dedicated virtual environment (`.venv`)

### 1.2 Core Dependencies
The primary dependencies pinned in the freeze artifact include:
```
pytest >= 8.3.3
pydantic >= 2.0.0
cryptography >= 41.0.0
onnxruntime >= 1.16.0
httpx >= 0.25.0
fastapi >= 0.100.0
uvicorn >= 0.22.0
numpy >= 1.24.0
```

### 1.3 Local LLM Daemon (Ollama)
For live model evaluation:
1. Install [Ollama](https://ollama.com/).
2. Pull the reference model:
   ```bash
   ollama pull llama3.2:latest
   ```
3. Ensure the Ollama daemon is active:
   ```bash
   ollama serve
   ```
   Verify status at `http://127.0.0.1:11434/api/tags`.

---

## 2. Cryptographic Integrity Verification

Verify the bundled model assets against the recorded SHA-256 digests. Source hashes in the phase-four freeze are historical and are not current expectations.

### 2.1 Model Artifact Hashes
- **`models/all-MiniLM-L6-v2/model.onnx`:**  
  `759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e`
- **`models/all-MiniLM-L6-v2/tokenizer.json`:**  
  `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0`

### 2.2 Source Code Hashes
The source hashes in the phase-four environment freeze describe an earlier code snapshot; they are not current integrity expectations. Record fresh hashes together with any new result artifact rather than comparing today's source against those historical values.

---

## 3. Step-by-Step Replication Protocol

### Step 1: Run the Test Suite
```bash
python -m pytest -q
```
The suite may skip tests that require optional services or packages (for example AgentDojo). The Phase 6 implementation checkpoint recorded 144 passed and 5 skipped; this is a historical observation, not a promise that future runs have the same count.

---

### Step 2: Regenerate Adaptive Attacks
```bash
python scripts/generate_adaptive.py --seed 0 --search-budget 8
```
The generator writes seven deterministic variants per source row, including a bounded similarity search with a score
trace. Payloads target simulated tools and contain only inert markers and the fake `example.invalid` address. It does
not copy the original external side-effect actions or destinations.

### Step 2a: Report Adaptive Attack ASR
```bash
python scripts/evaluate_adaptive.py --setting both --split test
```
This runs the deterministic vulnerable backend across all nine configurations and writes per-attack-type ASR and
the configurations that allowed execution to `results/adaptive_attack_asr.json`. It also records Stage B candidate
acceptance separately from action execution. All actions remain simulated.

---

### Step 3: Calibrate and Run the Phase 6 Evaluation
Run the evaluation runner with explicit backends, seeds, and the complete benchmark selection. Its grid runner
includes B0, B1, G-A, G-B, G-AB, G-AB+P, G-AB+S, G-AB+T, and FULL:
```bash
python scripts/run_phase6_evaluation.py --benchmarks all --backends vulnerable ollama hosted --seeds 17 29 43
```
The runner calibrates tau from InjecAgent base validation data only, sweeping 0.00–1.00 in 0.01 increments and
selecting the best-F1 point with FPR at or below 0.10 by default (`--max-fpr` changes the cap). Test data is never
used for threshold selection. It saves the complete curve and raw benign/injected score distributions before
evaluating the selected test workloads. If no validation point meets the cap, it reports that and does not relax the
cap or update `calibration.json`. Ollama must be reachable and the hosted-backend key must be configured in the
environment. The `all` benchmark selection also requires the optional AgentDojo dependency. Smaller smoke runs can
use `--limit`; do not present them as the full matrix. The AgentDojo adapter is separate and does not yet mirror all
nine component configurations; its output is not a full mode-ablation result.

---

### Step 4: Run AgentDojo Independently (Optional)
Install the optional benchmark dependencies and configure Ollama before running the adapter test:
```bash
python -m pytest tests/test_real.py::test_agentdojo_adapter_runs_real_tasks -v -s
```
If AgentDojo is not installed, the corresponding test is skipped. Results depend on the configured model and benchmark version; no fixed task utility or pass result is asserted here.

---

### Step 5: Check Token Optimization Invariants
```bash
python -m pytest tests -k "token_opt" -v
```
These focused tests cover protected intent/constraint/policy fields and optimization behavior.

---

## 4. Primary Results Artifact Checklist

When audits or peer reviews are conducted, verify the generated JSON outputs in `results/`:

1. `calibration.json`: Current validation-only tau, provenance, the full threshold curve, the FPR-recall curve, and benign/injected candidate-score distributions.
2. `results/calibration_sweep.json` (or `results/phase6_calibration_sweep.json`): Persisted sweep and score distributions for calibration runs.
3. `data/adaptive_attacks.json`: Generated inert adaptive variants, source metadata, and similarity-search traces.
4. `results/adaptive_attack_asr.json`: Per-attack-type ASR for each gateway configuration, when the adaptive report is run.
5. `results/phase6_evaluation.json`: Full evaluation output, created only after the configured runner completes successfully.
6. `results/agentdojo_*.json`: AgentDojo outputs when that optional benchmark is run.

The current calibration has $\tau=0.12$ (precision 0.7189, recall 0.4691, F1 0.5678, FPR 0.0917), selected under
the default maximum FPR of 0.10. The full 0.00–1.00/0.01 curve is stored in the artifacts above. The ONNX embedder
was not changed; tau decreased because the FPR cap was tightened from 0.30 to 0.10. At the time this guide was
updated, the full real-model/AgentDojo matrix had not been run because Ollama, a hosted API key, and the AgentDojo
package were unavailable. Limited vulnerable-backend smoke tests are not substitutes for the full results artifact.
