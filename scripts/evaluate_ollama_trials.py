#!/usr/bin/env python3
"""Evaluate Attack Success Rate (ASR) over N trials on local Ollama (Llama 3.2).

Measures:
1. Per-case successes/N with Wilson 95% confidence intervals using:
   from statsmodels.stats.proportion import proportion_confint
   lo, hi = proportion_confint(k, n, method="wilson")
2. Baseline B0 ASR over all cases.
3. Susceptible subset: cases where B0 succeeded in at least one trial (k_B0 >= 1).
4. Defended ASR on the susceptible subset vs all cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config
from app.dataset import CASES
from app.gateway import Gateway
from app.llm_backends import OllamaBackend
from app.stats import proportion_confint


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Compute 95% Wilson score interval for k successes in n trials."""
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def run_trials(
    cases: list[dict],
    modes: list[str],
    n_trials: int = 20,
    temperature: float | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run N trials per case across the specified modes and aggregate results."""
    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    backend = OllamaBackend(temperature=temp)
    backend.check()

    print(f"\n========================================================")
    print(f"DPVG Ollama (llama3.2) Repeated Trials Evaluation")
    print(f"Model: {backend.model} | Temperature: {temp}")
    print(f"Trials per case (N): {n_trials} | Modes: {modes}")
    print(f"Cases ({len(cases)}): {[c['id'] for c in cases]}")
    print(f"========================================================\n")

    gw = Gateway(llm=backend)
    results_by_case = {}

    for c_idx, case in enumerate(cases, 1):
        cid = case["id"]
        label = case.get("label") or case.get("goal", cid)[:50]
        scenario = case.get("scenario", "unknown")
        print(f"[{c_idx}/{len(cases)}] Evaluating Case: {cid} ({scenario}) - {label}")

        case_results = {
            "id": cid,
            "scenario": scenario,
            "label": label,
            "goal": case.get("goal", ""),
            "n_trials": n_trials,
            "modes": {},
        }

        for mode in modes:
            successes = 0
            latencies = []
            trial_details = []

            for t in range(1, n_trials + 1):
                t0 = time.perf_counter()
                run_res = gw.run(case, mode=mode)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)

                is_success = bool(run_res.get("metrics", {}).get("attack_success"))
                if is_success:
                    successes += 1

                trial_details.append({
                    "trial": t,
                    "attack_success": is_success,
                    "latency_ms": round(elapsed_ms, 2),
                    "outcome": run_res.get("outcome", {}).get("status"),
                })
                print(f"    Mode {mode:<5} Trial {t:>2}/{n_trials}: success={is_success} ({elapsed_ms:.1f}ms)", end="\r")

            lo, hi = wilson_ci(successes, n_trials)
            asr = successes / n_trials if n_trials > 0 else 0.0
            case_results["modes"][mode] = {
                "successes": successes,
                "n_trials": n_trials,
                "asr": round(asr, 4),
                "wilson_95": [round(lo, 4), round(hi, 4)],
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "trials": trial_details,
            }
            print(f"    Mode {mode:<5} Result: {successes}/{n_trials} ({asr*100:.1f}%) Wilson 95%: [{lo:.4f}, {hi:.4f}]")

        # Check susceptibility: in B0, succeeded at least once (k >= 1)
        b0_stats = case_results["modes"].get("B0", {})
        case_results["is_susceptible"] = b0_stats.get("successes", 0) >= 1
        results_by_case[cid] = case_results
        print()

    # Aggregate metrics
    total_cases = len(cases)
    susceptible_cases = [cid for cid, res in results_by_case.items() if res["is_susceptible"]]
    susceptible_count = len(susceptible_cases)

    aggregates = {
        "total_cases": total_cases,
        "n_trials_per_case": n_trials,
        "total_trial_runs_per_mode": total_cases * n_trials,
        "susceptible_subset": {
            "case_ids": susceptible_cases,
            "count": susceptible_count,
            "fraction": round(susceptible_count / total_cases, 4) if total_cases > 0 else 0.0,
        },
        "modes": {},
    }

    for mode in modes:
        total_successes_all = sum(res["modes"][mode]["successes"] for res in results_by_case.values())
        total_trials_all = total_cases * n_trials
        asr_all = total_successes_all / total_trials_all if total_trials_all > 0 else 0.0
        lo_all, hi_all = wilson_ci(total_successes_all, total_trials_all)

        # Defended ASR on susceptible subset
        if susceptible_count > 0:
            total_successes_susc = sum(results_by_case[cid]["modes"][mode]["successes"] for cid in susceptible_cases)
            total_trials_susc = susceptible_count * n_trials
            asr_susc = total_successes_susc / total_trials_susc if total_trials_susc > 0 else 0.0
            lo_susc, hi_susc = wilson_ci(total_successes_susc, total_trials_susc)
        else:
            total_successes_susc = 0
            total_trials_susc = 0
            asr_susc = 0.0
            lo_susc, hi_susc = (0.0, 0.0)

        aggregates["modes"][mode] = {
            "all_cases": {
                "successes": total_successes_all,
                "n_trials": total_trials_all,
                "asr": round(asr_all, 4),
                "wilson_95": [round(lo_all, 4), round(hi_all)],
            },
            "susceptible_subset": {
                "successes": total_successes_susc,
                "n_trials": total_trials_susc,
                "asr": round(asr_susc, 4),
                "wilson_95": [round(lo_susc, 4), round(hi_susc)],
            },
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": backend.model,
        "temperature": temp,
        "seed": 42,
        "tau": config.STAGE_B_TAU,
        "embedder": "all-MiniLM-L6-v2",
        "N": n_trials,
        "n_trials": n_trials,
        "aggregates": aggregates,
        "cases": results_by_case,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Print summary table
    print_summary_table(report, modes)
    return report


def print_summary_table(report: dict, modes: list[str]) -> None:
    agg = report["aggregates"]
    susc = agg["susceptible_subset"]
    print("\n" + "=" * 90)
    print("OLLAMA (Llama 3.2) REPEATED TRIALS ASR SUMMARY")
    print("=" * 90)
    print(f"Total Cases: {agg['total_cases']} | Trials Per Case: {agg['n_trials_per_case']} | Total Runs/Mode: {agg['total_trial_runs_per_mode']}")
    print(f"Susceptible Cases (B0 k >= 1): {susc['count']}/{agg['total_cases']} ({susc['fraction']*100:.1f}%) -> {susc['case_ids']}")
    print("-" * 90)
    print(f"{'Mode':<10} | {'All Cases ASR':<28} | {'Susceptible Subset ASR':<32}")
    print(f"{'':<10} | {'(successes/N, %, 95% Wilson)':<28} | {'(successes/N, %, 95% Wilson)':<32}")
    print("-" * 90)

    for mode in modes:
        m_all = agg["modes"][mode]["all_cases"]
        m_susc = agg["modes"][mode]["susceptible_subset"]

        all_str = f"{m_all['successes']}/{m_all['n_trials']} ({m_all['asr']*100:.1f}%) [{m_all['wilson_95'][0]:.3f}, {m_all['wilson_95'][1]:.3f}]"
        susc_str = f"{m_susc['successes']}/{m_susc['n_trials']} ({m_susc['asr']*100:.1f}%) [{m_susc['wilson_95'][0]:.3f}, {m_susc['wilson_95'][1]:.3f}]"

        print(f"{mode:<10} | {all_str:<28} | {susc_str:<32}")
    print("=" * 90 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", "-n", type=int, default=20, help="Number of trials per case per configuration (default: 20)")
    parser.add_argument("--temperature", "-t", type=float, default=None, help="Sampling temperature for Ollama (default from config: 0.7)")
    parser.add_argument("--cases", nargs="+", default=["representative"], help="Case IDs or 'all' or 'representative' (default: representative)")
    parser.add_argument("--modes", nargs="+", default=["B0", "FULL"], help="Configurations to evaluate (default: B0 FULL)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ollama_trials_evaluation.json", help="Output JSON path")
    return parser.parse_args()


def select_cases(case_args: list[str]) -> list[dict]:
    all_attacks = [c for c in CASES if c.get("is_attack")]
    if "all" in case_args:
        return all_attacks

    if "representative" in case_args:
        # Select representative cases across tiers:
        # Tier 1 (off-topic injection): s01, s02
        # Tier 2 (in-domain calendar/email injection): p02, c05
        # Tier 3 (demo passport exfiltration): demo_passport
        # Token attack: t01
        target_ids = ["s01", "s02", "p02", "c05", "demo_passport", "t01"]
        return [c for c in all_attacks if c["id"] in target_ids]

    selected = [c for c in all_attacks if c["id"] in case_args]
    if not selected:
        raise ValueError(f"No attack cases matched {case_args}")
    return selected


def main():
    args = parse_args()
    cases = select_cases(args.cases)
    run_trials(cases, modes=args.modes, n_trials=args.trials, temperature=args.temperature, output_path=args.output)


if __name__ == "__main__":
    main()
