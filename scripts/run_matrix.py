"""Evaluation Matrix Runner & Summary Table Generator for DPVG.

Executes:
  Configurations (modes) x Backends (vulnerable, ollama, hosted) x
  Workloads (demo, injecagent_base, injecagent_enhanced, adaptive, agentdojo) x Seeds

Reports:
  - Mean +/- std for ASR, TCR (benign and under attack), Precision, Recall, F1, FPR,
    Added end-to-end latency (median and p95 vs B0), and R_tok = 1 - T_opt / T_base.
  - Scenario breakdowns: benign, static injection, forged token, tampered token,
    expired token, replayed token, adaptive/camouflaged, multi-hop handoff propagation.
  - Individual run JSON files with git commit, timestamp, backend, model, tau, and seed.
  - Paper-ready Markdown and CSV summary tables.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config
from app.benchmarks import load_adaptive_attacks, load_injecagent, run_agentdojo
from app.dataset import split_cases
from app.embeddings import default_tau, get_embedder
from app.evaluation import _stopping_layers
from app.gateway import Gateway, MODES, PAPER_MODES

DEFAULT_MODES = list(PAPER_MODES)
DEFAULT_BACKENDS = ["vulnerable", "ollama", "hosted"]
DEFAULT_WORKLOADS = ["demo", "injecagent_base", "injecagent_enhanced", "adaptive", "drift", "agentdojo"]
DEFAULT_SEEDS = [17, 29, 43]
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"

SCENARIOS_ORDER = [
    "benign",
    "static injection",
    "forged token",
    "tampered token",
    "expired token",
    "replayed token",
    "wrong_root token",
    "adaptive/camouflaged",
    "multi-hop handoff propagation",
    "drift",
]


# ==============================================================================
# 1. Provenance & Environment Helpers
# ==============================================================================

def get_git_commit() -> str:
    """Retrieve current git commit hash, falling back to .git inspection or 'unversioned'."""
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        pass

    dot_git = ROOT / ".git"
    if dot_git.exists():
        try:
            head = (dot_git / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref_path = dot_git / head[5:].strip()
                if ref_path.exists():
                    return ref_path.read_text(encoding="utf-8").strip()
            return head
        except Exception:
            pass
    return "unversioned"


def set_random_seed(seed: int) -> None:
    """Fix random seeds across random and numpy."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Calculate mean and sample standard deviation."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(m, 4), "std": round(s, 4)}


def percentile(data: list[float], pct: float) -> float:
    """Calculate percentile value with nearest-rank interpolation."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict[str, float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator
    return {"low": round(max(0.0, center - margin), 4), "high": round(min(1.0, center + margin), 4)}


# ==============================================================================
# 2. Case Scenario Categorization
# ==============================================================================

def categorize_case(case: dict) -> str:
    """Categorize case into exact designated paper scenarios."""
    if not case.get("is_attack"):
        return "benign"

    scn = str(case.get("scenario") or "").lower()
    if scn in ("drift", "gradual_drift", "sudden_jump"):
        return "drift"

    rogue = case.get("rogue") or {}
    tok_attack = rogue.get("token_attack")
    if tok_attack:
        tok_str = str(tok_attack).lower()
        if "forged" in tok_str:
            return "forged token"
        if "tamper" in tok_str:
            return "tampered token"
        if "expire" in tok_str:
            return "expired token"
        if "replay" in tok_str:
            return "replayed token"
        if "wrong_root" in tok_str or "root" in tok_str:
            return "wrong_root token"
        return f"{tok_attack} token"

    if scn in ("propagation", "multi-hop", "handoff"):
        return "multi-hop handoff propagation"
    if scn in ("adaptive", "camouflaged"):
        return "adaptive/camouflaged"
    if scn in ("static", "static injection"):
        return "static injection"

    return scn or "static injection"


# ==============================================================================
# 3. Workload Loaders & Preflight Checkers
# ==============================================================================

def check_backend_available(backend: str) -> tuple[bool, str]:
    """Verify whether a backend is configured and reachable before running."""
    b = backend.lower()
    if b == "vulnerable":
        return True, "ok"

    if b == "ollama":
        url = config.OLLAMA_URL
        try:
            import httpx
            r = httpx.get(f"{url}/api/tags", timeout=5.0)
            if r.status_code >= 400:
                return False, f"Ollama HTTP {r.status_code}"
            models = {m.get("name", "") for m in r.json().get("models", [])}
            want = config.OLLAMA_MODEL
            if want not in models and (want + ":latest") not in models:
                return False, f"Ollama model '{want}' not pulled"
            return True, "ok"
        except Exception as e:
            return False, f"Ollama server not reachable at {url}: {e}"

    if b == "hosted":
        if not config.HOSTED_API_KEY:
            return False, "DPVG_HOSTED_API_KEY is not set"
        if not config.HOSTED_API_MODEL:
            return False, "DPVG_HOSTED_API_MODEL is not set"
        try:
            import httpx
            r = httpx.get(f"{config.HOSTED_API_URL}/models",
                          headers={"Authorization": f"Bearer {config.HOSTED_API_KEY}"},
                          timeout=5.0)
            if r.status_code >= 400:
                return False, f"Hosted API returned HTTP {r.status_code}"
            return True, "ok"
        except Exception as e:
            return False, f"Hosted API not reachable at {config.HOSTED_API_URL}: {e}"

    return False, f"Unknown backend '{backend}'"


def load_workload_cases(workload: str, limit: int | None = None) -> tuple[list[dict] | None, str | None]:
    """Load evaluation cases for a workload, or return (None, skip_reason)."""
    w = workload.lower()
    if w == "demo":
        cases = split_cases("test")
        if limit is not None:
            cases = cases[:limit]
        return cases, None

    if w == "injecagent_base":
        try:
            cases = load_injecagent(setting="base", split="test", limit=limit)
            return cases, None
        except FileNotFoundError as e:
            return None, f"InjecAgent base dataset missing: {e}"

    if w == "injecagent_enhanced":
        try:
            cases = load_injecagent(setting="enhanced", split="test", limit=limit)
            return cases, None
        except FileNotFoundError as e:
            return None, f"InjecAgent enhanced dataset missing: {e}"

    if w == "adaptive":
        try:
            cases = load_adaptive_attacks(setting="both", split="test", limit=limit)
            return cases, None
        except FileNotFoundError as e:
            return None, f"Adaptive attacks dataset missing: {e}"

    if w == "drift":
        path = ROOT / "data" / "drift_scenarios.json"
        if not path.exists():
            return None, f"Drift scenarios dataset missing: {path}"
        scenarios = json.loads(path.read_text(encoding="utf-8")).get("scenarios", [])
        cases = []
        for scenario in scenarios:
            if scenario.get("split") != "test":
                continue
            query = str(scenario["root_intent"])
            cases.append({
                "id": scenario["id"],
                "scenario": "drift",
                "attack_type": scenario.get("attack_type"),
                "query": query,
                "goal": query,
                "is_attack": scenario.get("label") == "attack",
                "split": scenario["split"],
                "subtasks": [{"text": str(step), "tool": None, "args": {}} for step in scenario["steps"]],
            })
        if limit is not None:
            cases = cases[:limit]
        if not cases:
            return None, "Drift scenarios have no held-out test cases"
        return cases, None

    if w == "agentdojo":
        try:
            import agentdojo  # noqa: F401
            return [], None  # Special workload executed via run_agentdojo
        except ImportError:
            return None, "agentdojo package is not installed in the Python environment"

    return None, f"Unknown workload '{workload}'"


# ==============================================================================
# 4. Single-Seed Mode Evaluator
# ==============================================================================

def evaluate_single_mode(gateway: Gateway, mode_key: str, cases: list[dict]) -> dict:
    """Evaluate one gateway configuration mode over a case set on a single seed."""
    runs = []
    lat_total = []

    for c in cases:
        res = gateway.run(c, mode_key)
        runs.append((c, res))
        lat_total.append(res["metrics"]["total_ms"])

    atk = [(c, r) for c, r in runs if c.get("is_attack")]
    ben = [(c, r) for c, r in runs if not c.get("is_attack")]
    n_atk = len(atk)
    n_ben = len(ben)

    tp = sum(bool(r["metrics"]["detected"]) for _, r in atk)
    fn = n_atk - tp
    fp = sum(bool(r["metrics"]["detected"]) for _, r in ben)
    tn = n_ben - fp

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    asr = sum(bool(r["metrics"]["attack_success"]) for _, r in atk) / n_atk if n_atk else 0.0
    tcr_benign = sum(bool(r["metrics"]["completed"]) and not bool(r["metrics"]["detected"]) for _, r in ben) / n_ben if n_ben else 0.0
    tcr_attack = sum(bool(r["metrics"]["completed"]) for _, r in atk) / n_atk if n_atk else 0.0

    # Token reduction R_tok
    rtok_values = [r["metrics"]["r_tok"] for _, r in runs]
    r_tok = statistics.mean(rtok_values) if rtok_values else 0.0

    # Backend-reported token usage
    backend_tokens = [
        r["metrics"]["llm_total_tokens"] for _, r in runs
        if isinstance(r["metrics"].get("llm_total_tokens"), int)
    ]
    token_sources = sorted({
        r["metrics"].get("llm_token_source", "unavailable") for _, r in runs
    })
    token_source = token_sources[0] if len(token_sources) == 1 else ("mixed" if token_sources else "unavailable")
    prompt_tokens = [
        r["metrics"]["llm_prompt_tokens"] for _, r in runs
        if isinstance(r["metrics"].get("llm_prompt_tokens"), int)
    ]
    completion_tokens = [
        r["metrics"]["llm_completion_tokens"] for _, r in runs
        if isinstance(r["metrics"].get("llm_completion_tokens"), int)
    ]

    # Latencies
    med_lat = statistics.median(lat_total) if lat_total else 0.0
    p95_lat = percentile(lat_total, 95) if lat_total else 0.0

    # Scenario breakdown
    scenarios = {}
    grouped: dict[str, list] = {}
    for c, r in runs:
        cat = categorize_case(c)
        grouped.setdefault(cat, []).append((c, r))

    for cat, sc_runs in grouped.items():
        n = len(sc_runs)
        if cat == "benign":
            sc_tcr = sum(bool(r["metrics"]["completed"]) and not bool(r["metrics"]["detected"]) for _, r in sc_runs) / n
            sc_fpr = sum(bool(r["metrics"]["detected"]) for _, r in sc_runs) / n
            scenarios[cat] = {"n": n, "tcr": round(sc_tcr, 4), "fpr": round(sc_fpr, 4), "asr": None, "detected": None}
        else:
            sc_asr = sum(bool(r["metrics"]["attack_success"]) for _, r in sc_runs) / n
            sc_det = sum(bool(r["metrics"]["detected"]) for _, r in sc_runs) / n
            sc_tcr_atk = sum(bool(r["metrics"]["completed"]) for _, r in sc_runs) / n
            scenarios[cat] = {
                "n": n, "asr": round(sc_asr, 4), "detected": round(sc_det, 4),
                "tcr_attack": round(sc_tcr_atk, 4), "tcr": round(sc_tcr_atk, 4), "fpr": None,
            }

    stopped_by_layer: dict[str, int] = {}
    for case, result in atk:
        for layer in _stopping_layers(result):
            stopped_by_layer[layer] = stopped_by_layer.get(layer, 0) + 1
    by_attack_type: dict[str, dict[str, Any]] = {}
    for case, result in atk:
        attack_type = case.get("attack_type")
        if not attack_type:
            continue
        outcome = by_attack_type.setdefault(
            attack_type, {"n": 0, "executed": 0, "stopped_by_layer": {}, "case_outcomes": []}
        )
        layers = _stopping_layers(result)
        outcome["n"] += 1
        outcome["executed"] += bool(result["metrics"]["attack_success"])
        outcome["case_outcomes"].append({
            "id": case.get("id"),
            "executed": bool(result["metrics"]["attack_success"]),
            "stopped_by": layers,
        })
        for layer in layers:
            counts = outcome["stopped_by_layer"]
            counts[layer] = counts.get(layer, 0) + 1
    for outcome in by_attack_type.values():
        outcome["asr"] = round(outcome["executed"] / outcome["n"], 4) if outcome["n"] else 0.0

    return {
        "mode": mode_key,
        "n_cases": len(runs),
        "n_attack": n_atk,
        "n_benign": n_ben,
        "asr": round(asr, 4),
        "tcr_benign": round(tcr_benign, 4),
        "tcr_attack": round(tcr_attack, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "r_tok": round(r_tok, 4),
        "token_source": token_source,
        "token_sources": token_sources or ["unavailable"],
        "llm_prompt_tokens_sum": sum(prompt_tokens) if prompt_tokens else None,
        "llm_completion_tokens_sum": sum(completion_tokens) if completion_tokens else None,
        "backend_tokens_sum": sum(backend_tokens) if backend_tokens else None,
        "latency_ms": {"median": round(med_lat, 3), "p95": round(p95_lat, 3)},
        "scenarios": scenarios,
        "stopped_by_layer": stopped_by_layer,
        "by_attack_type": by_attack_type,
        "counts": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "asr_successes": sum(bool(r["metrics"]["attack_success"]) for _, r in atk),
            "tcr_benign_successes": sum(bool(r["metrics"]["completed"]) and not bool(r["metrics"]["detected"])
                                        for _, r in ben),
            "tcr_attack_successes": sum(bool(r["metrics"]["completed"]) for _, r in atk),
        },
    }


# ==============================================================================
# 5. Full Matrix Runner
# ==============================================================================

def run_matrix(
    modes: list[str] = DEFAULT_MODES,
    backends: list[str] = DEFAULT_BACKENDS,
    workloads: list[str] = DEFAULT_WORKLOADS,
    seeds: list[int] = DEFAULT_SEEDS,
    tau: float | None = None,
    temperature: float = 0.0,
    limit: int | None = None,
    output_path: pathlib.Path | None = None,
    runs_dir: pathlib.Path | None = None,
) -> dict:
    """Execute evaluation matrix across configurations, backends, workloads, and seeds."""
    out_dir = runs_dir or RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    git_commit = get_git_commit()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    embedder = get_embedder()
    effective_tau = default_tau(embedder) if tau is None else tau

    matrix_results: dict[str, Any] = {
        "provenance": {
            "title": "DPVG Phase 6 Comprehensive Evaluation Matrix",
            "git_commit": git_commit,
            "timestamp": timestamp,
            "embedder": embedder.name,
            "tau": effective_tau,
            "temperature": temperature,
            "seeds": seeds,
            "f1_wilson_note": "F1* interval uses effective counts 2TP successes and 2TP+FP+FN trials; it is an approximation, not an exact binomial interval.",
            "drift_parameters": {
                "tau_drift": config.DRIFT_THRESHOLD,
                "consecutive_steps": config.DRIFT_CONSECUTIVE_STEPS,
                "ema_alpha": config.DRIFT_EMA_ALPHA,
                "metric": config.DRIFT_METRIC,
            },
            "modes": modes,
            "backends": backends,
            "workloads": workloads,
            "limit": limit,
        },
        "workloads": {},
        "skipped": {},
    }

    for workload in workloads:
        cases, skip_reason = load_workload_cases(workload, limit=limit)
        if skip_reason is not None:
            matrix_results["skipped"][workload] = {
                "status": "skipped",
                "reason": skip_reason,
            }
            print(f"[-] Workload '{workload}' skipped: {skip_reason}")
            continue

        matrix_results["workloads"][workload] = {}

        # Special handling for agentdojo
        if workload == "agentdojo":
            avail, reason = check_backend_available("ollama")
            if not avail:
                matrix_results["skipped"][f"{workload}:ollama"] = {"status": "skipped", "reason": reason}
                print(f"[-] AgentDojo skipped: {reason}")
                continue
            dojo_results = []
            for seed in seeds:
                set_random_seed(seed)
                res = run_agentdojo(
                    modes=[m for m in modes if m in {"B0", "G-A", "G-B", "G-AB", "FULL"}],
                    tau=effective_tau, seed=seed, temperature=temperature,
                )
                dojo_results.append(res)
            matrix_results["workloads"][workload]["ollama"] = {"seeds": dojo_results}
            continue

        # Standard workloads
        for backend in backends:
            b_avail, b_reason = check_backend_available(backend)
            if not b_avail:
                matrix_results["skipped"][f"{workload}:{backend}"] = {
                    "status": "skipped",
                    "reason": b_reason,
                }
                print(f"[-] Workload '{workload}' on backend '{backend}' skipped: {b_reason}")
                continue

            print(f"[+] Running Workload '{workload}' on '{backend}' across seeds {seeds} (n={len(cases)})...")
            seed_runs = []
            for seed in seeds:
                set_random_seed(seed)
                gw = Gateway(
                    backend=backend,
                    seed=seed,
                    temperature=temperature,
                    tau=effective_tau,
                )
                run_data = {
                    "provenance": {
                        "git_commit": git_commit,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "backend": backend,
                        "model": gw.llm.name,
                        "tau": effective_tau,
                        "drift_parameters": {
                            "tau_drift": gw.drift_threshold,
                            "consecutive_steps": gw.drift_consecutive_steps,
                            "ema_alpha": gw.drift_ema_alpha,
                            "metric": gw.drift_metric,
                        },
                        "seed": seed,
                        "temperature": temperature,
                        "workload": workload,
                    },
                    "modes": {},
                }

                for mode in modes:
                    mode_res = evaluate_single_mode(gw, mode, cases)
                    run_data["modes"][mode] = mode_res

                # Compute added latency relative to B0 for this seed
                b0_lat = run_data["modes"].get("B0", {}).get("latency_ms", {})
                b0_med = b0_lat.get("median", 0.0)
                b0_p95 = b0_lat.get("p95", 0.0)

                for mode, mode_res in run_data["modes"].items():
                    m_lat = mode_res["latency_ms"]
                    mode_res["added_latency_ms"] = {
                        "median": round(m_lat["median"] - b0_med, 3),
                        "p95": round(m_lat["p95"] - b0_p95, 3),
                    }

                # Write individual run JSON
                run_file = out_dir / f"{backend}_{workload}_seed{seed}_{int(time.time())}.json"
                run_file.write_text(json.dumps(run_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                seed_runs.append(run_data)

            # Aggregate across seeds
            mode_summaries = {}
            for mode in modes:
                mode_seeds = [sr["modes"][mode] for sr in seed_runs]
                counts = {
                    key: sum(ms.get("counts", {}).get(key, 0) for ms in mode_seeds)
                    for key in ("tp", "fp", "tn", "fn", "asr_successes",
                                "tcr_benign_successes", "tcr_attack_successes")
                }
                n_attack = sum(ms["n_attack"] for ms in mode_seeds)
                n_benign = sum(ms["n_benign"] for ms in mode_seeds)
                wilson = {
                    "asr": wilson_interval(counts["asr_successes"], n_attack),
                    "tcr_benign": wilson_interval(counts["tcr_benign_successes"], n_benign),
                    "tcr_attack": wilson_interval(counts["tcr_attack_successes"], n_attack),
                    "precision": wilson_interval(counts["tp"], counts["tp"] + counts["fp"]),
                    "recall": wilson_interval(counts["tp"], counts["tp"] + counts["fn"]),
                    "f1": wilson_interval(
                        2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]
                    ),
                    "fpr": wilson_interval(counts["fp"], counts["fp"] + counts["tn"]),
                }
                layer_counts: dict[str, int] = {}
                for ms in mode_seeds:
                    for layer, count in ms.get("stopped_by_layer", {}).items():
                        layer_counts[layer] = layer_counts.get(layer, 0) + count
                type_names = set().union(*(ms.get("by_attack_type", {}).keys() for ms in mode_seeds))
                attack_type_summary = {}
                for attack_type in sorted(type_names):
                    type_runs = [
                        ms["by_attack_type"][attack_type] for ms in mode_seeds
                        if attack_type in ms.get("by_attack_type", {})
                    ]
                    n_type = sum(item["n"] for item in type_runs)
                    executed = sum(item["executed"] for item in type_runs)
                    type_layers: dict[str, int] = {}
                    for item in type_runs:
                        for layer, count in item.get("stopped_by_layer", {}).items():
                            type_layers[layer] = type_layers.get(layer, 0) + count
                    attack_type_summary[attack_type] = {
                        "n": n_type,
                        "executed": executed,
                        "asr": round(executed / n_type, 4) if n_type else 0.0,
                        "stopped_by_layer": type_layers,
                        "case_outcomes": [
                            {**outcome, "seed": seed_run["provenance"]["seed"]}
                            for seed_run in seed_runs
                            for outcome in seed_run["modes"][mode].get("by_attack_type", {}).get(
                                attack_type, {}
                            ).get("case_outcomes", [])
                        ],
                    }
                mode_summaries[mode] = {
                    "mode": mode,
                    "n_cases": mode_seeds[0]["n_cases"],
                    "asr": mean_std([ms["asr"] for ms in mode_seeds]),
                    "tcr_benign": mean_std([ms["tcr_benign"] for ms in mode_seeds]),
                    "tcr_attack": mean_std([ms["tcr_attack"] for ms in mode_seeds]),
                    "precision": mean_std([ms["precision"] for ms in mode_seeds]),
                    "recall": mean_std([ms["recall"] for ms in mode_seeds]),
                    "f1": mean_std([ms["f1"] for ms in mode_seeds]),
                    "fpr": mean_std([ms["fpr"] for ms in mode_seeds]),
                    "r_tok": mean_std([ms["r_tok"] for ms in mode_seeds]),
                    "wilson_95": wilson,
                    "counts": counts,
                    "token_sources": sorted({
                        source for ms in mode_seeds for source in ms.get("token_sources", [])
                    }),
                    "llm_prompt_tokens_sum": sum(
                        ms["llm_prompt_tokens_sum"] for ms in mode_seeds
                        if ms.get("llm_prompt_tokens_sum") is not None
                    ),
                    "llm_completion_tokens_sum": sum(
                        ms["llm_completion_tokens_sum"] for ms in mode_seeds
                        if ms.get("llm_completion_tokens_sum") is not None
                    ),
                    "backend_tokens_sum": sum(
                        ms["backend_tokens_sum"] for ms in mode_seeds
                        if ms.get("backend_tokens_sum") is not None
                    ),
                    "stopped_by_layer": layer_counts,
                    "by_attack_type": attack_type_summary,
                    "added_latency_ms": {
                        "median": mean_std([ms["added_latency_ms"]["median"] for ms in mode_seeds]),
                        "p95": mean_std([ms["added_latency_ms"]["p95"] for ms in mode_seeds]),
                    },
                    "scenarios": {},
                }

                # Aggregate scenarios across seeds
                sc_keys = set().union(*(ms["scenarios"].keys() for ms in mode_seeds))
                for sc in sc_keys:
                    sc_seeds = [ms["scenarios"][sc] for ms in mode_seeds if sc in ms["scenarios"]]
                    if not sc_seeds:
                        continue
                    n = sc_seeds[0]["n"]
                    if sc == "benign":
                        mode_summaries[mode]["scenarios"][sc] = {
                            "n": n,
                            "tcr": mean_std([s["tcr"] for s in sc_seeds]),
                            "fpr": mean_std([s["fpr"] for s in sc_seeds]),
                        }
                    else:
                        mode_summaries[mode]["scenarios"][sc] = {
                            "n": n,
                            "asr": mean_std([s["asr"] for s in sc_seeds]),
                            "detected": mean_std([s["detected"] for s in sc_seeds]),
                            "tcr_attack": mean_std([s["tcr_attack"] for s in sc_seeds]),
                        }

            matrix_results["workloads"][workload][backend] = {
                "summary": mode_summaries,
                "seeds": seed_runs,
            }

    # Write combined matrix JSON
    comb_path = output_path or (RESULTS_DIR / "matrix_evaluation.json")
    comb_path.parent.mkdir(parents=True, exist_ok=True)
    comb_path.write_text(json.dumps(matrix_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[+] Wrote full matrix JSON: {comb_path}")

    # Generate summary tables
    generate_summary_tables(matrix_results, comb_path.parent)

    return matrix_results


# ==============================================================================
# 6. Markdown & CSV Summary Table Generator
# ==============================================================================

def _fmt_stat(val_dict: dict[str, float] | None, pct: bool = False) -> str:
    if not val_dict:
        return "-"
    m, s = val_dict.get("mean", 0.0), val_dict.get("std", 0.0)
    if pct:
        return f"{m * 100:.1f} ± {s * 100:.1f}%"
    return f"{m:.3f} ± {s:.3f}"


def _fmt_lat(lat_dict: dict[str, Any] | None) -> str:
    if not lat_dict:
        return "-"
    m, s = lat_dict.get("mean", 0.0), lat_dict.get("std", 0.0)
    return f"{m:+.1f} ± {s:.1f} ms"


def _fmt_wilson(interval: dict[str, float] | None) -> str:
    if not interval:
        return "-"
    return f"{interval['low'] * 100:.1f}-{interval['high'] * 100:.1f}%"


def generate_summary_tables(data: dict, out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Generate Markdown and CSV summary tables for paper presentation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "summary_table.md"
    csv_path = out_dir / "summary_table.csv"
    sc_md_path = out_dir / "scenarios_table.md"
    sc_csv_path = out_dir / "scenarios_table.csv"
    layer_md_path = out_dir / "layer_stops_table.md"
    layer_csv_path = out_dir / "layer_stops_table.csv"
    type_md_path = out_dir / "attack_types_table.md"
    type_csv_path = out_dir / "attack_types_table.csv"

    # ------------------ Overall Modes Table ------------------
    headers = [
        "Workload", "Backend", "Mode", "ASR", "TCR (Benign)", "TCR (Attack)",
        "Precision", "Recall", "F1", "FPR", "Δ Med Latency", "Δ P95 Latency", "R_tok",
        "95% Wilson CIs (ASR/TCRb/TCRa/P/R/F1*/FPR)", "Token Counter(s)",
        "Prompt Tokens", "Completion Tokens",
    ]

    csv_rows = []
    md_lines = [
        "# DPVG Phase 6 Evaluation Summary",
        f"**Git Commit:** `{data.get('provenance', {}).get('git_commit', 'unknown')}` | "
        f"**Timestamp:** `{data.get('provenance', {}).get('timestamp', 'unknown')}` | "
        f"**Tau:** `{data.get('provenance', {}).get('tau', 'unknown')}`",
        "",
        "## Overall Configuration Performance (Mean ± Std over Seeds)",
        "",
        "F1* Wilson bounds use effective counts and are an approximation; unlike the other intervals, they are not exact binomial coverage.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for workload, backends_data in data.get("workloads", {}).items():
        if workload == "agentdojo":
            continue
        for backend, b_data in backends_data.items():
            summary = b_data.get("summary", {})
            for mode, m_data in summary.items():
                row = [
                    workload,
                    backend,
                    mode,
                    _fmt_stat(m_data.get("asr"), pct=True),
                    _fmt_stat(m_data.get("tcr_benign"), pct=True),
                    _fmt_stat(m_data.get("tcr_attack"), pct=True),
                    _fmt_stat(m_data.get("precision"), pct=True),
                    _fmt_stat(m_data.get("recall"), pct=True),
                    _fmt_stat(m_data.get("f1")),
                    _fmt_stat(m_data.get("fpr"), pct=True),
                    _fmt_lat(m_data.get("added_latency_ms", {}).get("median")),
                    _fmt_lat(m_data.get("added_latency_ms", {}).get("p95")),
                    _fmt_stat(m_data.get("r_tok"), pct=True),
                    "; ".join(
                        f"{label} {_fmt_wilson(m_data.get('wilson_95', {}).get(key))}"
                        for label, key in (
                            ("ASR", "asr"), ("TCRb", "tcr_benign"), ("TCRa", "tcr_attack"),
                            ("P", "precision"), ("R", "recall"), ("F1*", "f1"), ("FPR", "fpr"),
                        )
                    ),
                    ", ".join(m_data.get("token_sources", [])) or "unavailable",
                    str(m_data.get("llm_prompt_tokens_sum")),
                    str(m_data.get("llm_completion_tokens_sum")),
                ]
                csv_rows.append(row)
                md_lines.append("| " + " | ".join(row) + " |")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    # ------------------ Scenarios Breakdown Table ------------------
    sc_headers = [
        "Workload", "Backend", "Scenario", "Mode", "N", "ASR", "Detection Rate", "TCR",
    ]
    sc_csv_rows = []
    sc_md_lines = [
        "# DPVG Scenario Breakdown Table",
        "",
        "## Fine-Grained Attack Scenario Analysis (Mean ± Std over Seeds)",
        "",
        "| " + " | ".join(sc_headers) + " |",
        "| " + " | ".join(["---"] * len(sc_headers)) + " |",
    ]
    layer_headers = ["Workload", "Backend", "Mode", "Stopping Layer", "Attack Cases"]
    layer_csv_rows = []
    layer_md_lines = [
        "# DPVG Per-Layer Attack Outcomes",
        "",
        "| " + " | ".join(layer_headers) + " |",
        "| " + " | ".join(["---"] * len(layer_headers)) + " |",
    ]
    type_headers = ["Workload", "Backend", "Mode", "Attack Type", "N", "Executed", "ASR", "Stopped By"]
    type_csv_rows = []
    type_md_lines = [
        "# DPVG Adaptive/Drift Attack-Type Outcomes",
        "",
        "| " + " | ".join(type_headers) + " |",
        "| " + " | ".join(["---"] * len(type_headers)) + " |",
    ]

    for workload, backends_data in data.get("workloads", {}).items():
        if workload == "agentdojo":
            continue
        for backend, b_data in backends_data.items():
            summary = b_data.get("summary", {})
            for mode, m_data in summary.items():
                for layer, count in sorted(m_data.get("stopped_by_layer", {}).items()):
                    row = [workload, backend, mode, layer, str(count)]
                    layer_csv_rows.append(row)
                    layer_md_lines.append("| " + " | ".join(row) + " |")
                for attack_type, outcome in sorted(m_data.get("by_attack_type", {}).items()):
                    row = [
                        workload, backend, mode, attack_type, str(outcome["n"]),
                        str(outcome["executed"]), f"{outcome['asr']:.4f}",
                        ", ".join(f"{layer}:{count}" for layer, count in sorted(
                            outcome.get("stopped_by_layer", {}).items()
                        )),
                    ]
                    type_csv_rows.append(row)
                    type_md_lines.append("| " + " | ".join(row) + " |")
            for sc in SCENARIOS_ORDER:
                for mode, m_data in summary.items():
                    sc_info = m_data.get("scenarios", {}).get(sc)
                    if not sc_info:
                        continue
                    n = sc_info.get("n", 0)
                    if sc == "benign":
                        asr_str = "-"
                        det_str = _fmt_stat(sc_info.get("fpr"), pct=True) + " (FPR)"
                        tcr_str = _fmt_stat(sc_info.get("tcr"), pct=True)
                    else:
                        asr_str = _fmt_stat(sc_info.get("asr"), pct=True)
                        det_str = _fmt_stat(sc_info.get("detected"), pct=True)
                        tcr_str = _fmt_stat(sc_info.get("tcr_attack"), pct=True)

                    row = [
                        workload, backend, sc, mode, str(n), asr_str, det_str, tcr_str,
                    ]
                    sc_csv_rows.append(row)
                    sc_md_lines.append("| " + " | ".join(row) + " |")

    # Write Scenarios CSV and Markdown
    with open(sc_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(sc_headers)
        writer.writerows(sc_csv_rows)

    sc_md_path.write_text("\n".join(sc_md_lines) + "\n", encoding="utf-8")
    md_lines.extend(["", "---", ""] + sc_md_lines[2:])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with open(layer_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(layer_headers)
        writer.writerows(layer_csv_rows)
    layer_md_path.write_text("\n".join(layer_md_lines) + "\n", encoding="utf-8")
    with open(type_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(type_headers)
        writer.writerows(type_csv_rows)
    type_md_path.write_text("\n".join(type_md_lines) + "\n", encoding="utf-8")

    print(f"[+] Wrote Summary Tables:\n    - {md_path}\n    - {csv_path}\n    - {sc_md_path}\n    - {sc_csv_path}\n    - {layer_md_path}\n    - {layer_csv_path}\n    - {type_md_path}\n    - {type_csv_path}")
    return md_path, csv_path


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=PAPER_MODES, default=DEFAULT_MODES,
                        help="Gateway modes to evaluate.")
    parser.add_argument("--backends", nargs="+", choices=DEFAULT_BACKENDS, default=DEFAULT_BACKENDS,
                        help="LLM backends to evaluate.")
    parser.add_argument("--workloads", nargs="+", choices=DEFAULT_WORKLOADS, default=DEFAULT_WORKLOADS,
                        help="Workloads to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds.")
    parser.add_argument("--tau", type=float, default=None,
                        help="Embedding similarity cutoff tau (default: calibrated default).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for LLM.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of cases per workload.")
    parser.add_argument("--output", type=pathlib.Path, default=RESULTS_DIR / "matrix_evaluation.json",
                        help="Output path for matrix evaluation JSON.")
    parser.add_argument("--runs-dir", type=pathlib.Path, default=RUNS_DIR,
                        help="Directory to save individual run JSONs.")
    parser.add_argument("--generate-tables-only", type=pathlib.Path, default=None,
                        help="Generate markdown/CSV tables from an existing matrix JSON without running.")

    args = parser.parse_args()

    if args.generate_tables_only:
        p = args.generate_tables_only
        if not p.exists():
            parser.error(f"File not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        generate_summary_tables(data, p.parent)
        return

    run_matrix(
        modes=args.modes,
        backends=args.backends,
        workloads=args.workloads,
        seeds=args.seeds,
        tau=args.tau,
        temperature=args.temperature,
        limit=args.limit,
        output_path=args.output,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    main()
