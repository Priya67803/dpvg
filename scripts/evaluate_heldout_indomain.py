#!/usr/bin/env python3
"""Evaluate held-out in-domain injection set (zero static marker phrases).

Compares three detection modes on 24 held-out in-domain injection cases:
1. Rules-only: regex injection pattern matching (Stage B rules).
2. Semantic-only: action-class privilege escalation check + cosine similarity.
3. Combined: Stage B full verification (rules + action-class + semantic similarity).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.alignment import AlignmentEngine
from app.embeddings import get_embedder
from app.heldout_dataset import HELDOUT_IN_DOMAIN_CASES
from app.stats import proportion_confint


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Compute 95% Wilson score interval."""
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def evaluate_heldout(tau: float = 0.13, output_path: Path | None = None) -> dict:
    embedder = get_embedder()
    engine = AlignmentEngine(embedder)
    cases = HELDOUT_IN_DOMAIN_CASES
    total_cases = len(cases)

    print("\n" + "=" * 95)
    print(f"HELD-OUT IN-DOMAIN INJECTION EVALUATION (NO STATIC MARKER PHRASES)")
    print(f"Total Cases: {total_cases} | Tau Threshold: {tau:.2f} | Embedder: {embedder.name}")
    print("=" * 95 + "\n")

    results = []
    rules_detected = 0
    semantic_detected = 0
    combined_detected = 0

    action_class_blocks = 0
    similarity_blocks = 0

    for c in cases:
        cid = c["id"]
        root_intent = c["root_intent"]
        injection = c["injection"]
        domain = c["domain"]
        e_root = embedder.encode(root_intent)

        # 1. Rules-only mode
        res_rules = engine.check_payload(e_root, injection, tau=tau, enable_rules=True, enable_sem=False, root_text=root_intent)
        rules_blocked = res_rules["blocked"]
        if rules_blocked:
            rules_detected += 1

        # 2. Semantic-only mode
        res_sem = engine.check_payload(e_root, injection, tau=tau, enable_rules=False, enable_sem=True, root_text=root_intent)
        sem_blocked = res_sem["blocked"]
        if sem_blocked:
            semantic_detected += 1

        # 3. Combined mode
        res_comb = engine.check_payload(e_root, injection, tau=tau, enable_rules=True, enable_sem=True, root_text=root_intent)
        comb_blocked = res_comb["blocked"]
        if comb_blocked:
            combined_detected += 1

        blocked_by = res_comb.get("blocked_by") or ("allowed" if not comb_blocked else "blocked")
        if "action_class" in str(blocked_by):
            action_class_blocks += 1
        if "sem" in str(blocked_by):
            similarity_blocks += 1

        cand_info = res_comb["candidates"][0] if res_comb["candidates"] else {}
        score = cand_info.get("score")
        cand_classes = cand_info.get("action_classes", [])
        escalation = cand_info.get("action_class_escalation", False)

        results.append({
            "id": cid,
            "domain": domain,
            "root_intent": root_intent,
            "injection": injection,
            "score": score,
            "action_classes": cand_classes,
            "escalation": escalation,
            "rules_only": {"blocked": rules_blocked, "reason": res_rules["candidates"][0]["reason"] if res_rules["candidates"] else "aligned"},
            "semantic_only": {"blocked": sem_blocked, "reason": res_sem["candidates"][0]["reason"] if res_sem["candidates"] else "aligned"},
            "combined": {"blocked": comb_blocked, "blocked_by": blocked_by, "reason": cand_info.get("reason", "aligned")},
        })

    # Summary metrics with Wilson 95% CIs
    r_lo, r_hi = wilson_ci(rules_detected, total_cases)
    s_lo, s_hi = wilson_ci(semantic_detected, total_cases)
    c_lo, c_hi = wilson_ci(combined_detected, total_cases)

    summary = {
        "model": "llama3.2",
        "temperature": 0.7,
        "seed": 42,
        "tau": tau,
        "embedder": embedder.name,
        "N": total_cases,
        "total_cases": total_cases,
        "detection": {
            "rules_only": {
                "detected": rules_detected,
                "total": total_cases,
                "detection_rate": round(rules_detected / total_cases, 4),
                "wilson_95": [round(r_lo, 4), round(r_hi, 4)],
            },
            "semantic_only": {
                "detected": semantic_detected,
                "total": total_cases,
                "detection_rate": round(semantic_detected / total_cases, 4),
                "wilson_95": [round(s_lo, 4), round(s_hi, 4)],
            },
            "combined": {
                "detected": combined_detected,
                "total": total_cases,
                "detection_rate": round(combined_detected / total_cases, 4),
                "wilson_95": [round(c_lo, 4), round(c_hi, 4)],
            },
        },
        "breakdown": {
            "action_class_escalation_blocks": action_class_blocks,
            "similarity_threshold_blocks": similarity_blocks,
        },
        "cases": results,
    }

    # Print comparative table
    print(f"{'Case ID':<10} | {'Domain':<9} | {'Rules-Only':<12} | {'Semantic-Only':<14} | {'Combined':<12} | {'Blocked By Layer':<16}")
    print("-" * 85)
    for r in results:
        r_str = "BLOCKED" if r["rules_only"]["blocked"] else "MISSED (0)"
        s_str = "BLOCKED" if r["semantic_only"]["blocked"] else "MISSED (0)"
        c_str = "BLOCKED" if r["combined"]["blocked"] else "MISSED (0)"
        layer_str = r["combined"]["blocked_by"] or "none"
        print(f"{r['id']:<10} | {r['domain']:<9} | {r_str:<12} | {s_str:<14} | {c_str:<12} | {layer_str:<16}")

    print("\n" + "=" * 85)
    print("DETECTION RATE SUMMARY (HELD-OUT IN-DOMAIN INJECTIONS)")
    print("=" * 85)
    print(f"{'Configuration':<25} | {'Detected':<10} | {'Detection Rate':<16} | {'95% Wilson CI':<18}")
    print("-" * 85)
    print(f"{'Rules-Only (Regex)':<25} | {rules_detected}/{total_cases:<7} | {rules_detected/total_cases*100:>5.1f}%          | [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"{'Semantic-Only (Class+Sim)':<25} | {semantic_detected}/{total_cases:<7} | {semantic_detected/total_cases*100:>5.1f}%          | [{s_lo:.4f}, {s_hi:.4f}]")
    print(f"{'Combined (Full Stage B)':<25} | {combined_detected}/{total_cases:<7} | {combined_detected/total_cases*100:>5.1f}%          | [{c_lo:.4f}, {c_hi:.4f}]")
    print("=" * 85 + "\n")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Detailed results saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=0.13, help="Tau threshold (default: 0.13)")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "heldout_indomain_evaluation.json", help="Output JSON path")
    args = parser.parse_args()
    evaluate_heldout(tau=args.tau, output_path=args.output)


if __name__ == "__main__":
    main()
