"""Sweep context-pruning cutoffs on the local prototype validation split."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dataset import split_cases
from app.embeddings import cosine, default_tau, get_embedder
from app.gateway import Gateway


def _measure(gateway: Gateway, cases: list[dict], mode: str) -> dict:
    runs = [(case, gateway.run(case, mode)) for case in cases]
    attacks = [(case, result) for case, result in runs if case.get("is_attack")]
    benign = [(case, result) for case, result in runs if not case.get("is_attack")]
    asr = sum(result["metrics"]["attack_success"] for _, result in attacks) / len(attacks) if attacks else 0.0
    tcr_benign = (
        sum(result["metrics"]["completed"] and not result["metrics"]["detected"] for _, result in benign) /
        len(benign) if benign else 0.0
    )
    tcr_attack = sum(result["metrics"]["completed"] for _, result in attacks) / len(attacks) if attacks else 0.0
    relevance = []
    for case, result in runs:
        query = str(case.get("query") or case.get("goal") or "")
        for event in result["steps"]:
            for chunks in (event.get("data", {}).get("context_by_subtask") or {}).values():
                relevance.extend(cosine(gateway.embedder.encode(query), gateway.embedder.encode(chunk))
                                 for chunk in chunks)
    counters = sorted({
        result["metrics"].get("llm_token_source", "unavailable")
        for _, result in runs
    })
    return {
        "mode": mode,
        "n_cases": len(runs),
        "asr": round(asr, 4),
        "tcr_benign": round(tcr_benign, 4),
        "tcr_under_attack": round(tcr_attack, 4),
        "r_tok": round(sum(result["metrics"]["r_tok"] for _, result in runs) / len(runs), 4) if runs else 0.0,
        "kept_chunk_count": len(relevance),
        "mean_kept_chunk_relevance": round(sum(relevance) / len(relevance), 4) if relevance else None,
        "token_counter_sources": counters,
    }


def sweep_context_cutoffs(
    cases: list[dict],
    gateway: Gateway,
    cutoffs: list[float],
) -> dict:
    if not cases:
        raise ValueError("context cutoff sweep requires validation cases")
    if any(not 0.0 <= cutoff <= 1.0 for cutoff in cutoffs):
        raise ValueError("context cutoffs must be between 0 and 1")
    baseline = _measure(gateway, cases, "G-AB")
    points = []
    for cutoff in cutoffs:
        gateway.optimizer.min_sim = cutoff
        optimized = _measure(gateway, cases, "G-AB+T")
        points.append({
            "cutoff": cutoff,
            "g_ab_plus_t": optimized,
            "asr_not_worse_than_g_ab": optimized["asr"] <= baseline["asr"],
            "benign_tcr_not_worse_than_g_ab": optimized["tcr_benign"] >= baseline["tcr_benign"],
            "attack_tcr_not_worse_than_g_ab": optimized["tcr_under_attack"] >= baseline["tcr_under_attack"],
        })
    return {
        "dataset": "built-in prototype validation split",
        "split": "val",
        "backend": gateway.llm.name,
        "tau": gateway.tau,
        "g_ab_baseline": baseline,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[round(i / 100, 2) for i in range(0, 61, 5)])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "context_cutoff_sweep.json")
    args = parser.parse_args()
    cases = split_cases("val")
    if args.limit is not None:
        cases = cases[:args.limit]
    embedder = get_embedder()
    gateway = Gateway(embedder=embedder, tau=default_tau(embedder), backend="vulnerable")
    result = sweep_context_cutoffs(cases, gateway, args.cutoffs)
    result["limit"] = args.limit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for point in result["points"]:
        opt = point["g_ab_plus_t"]
        print(
            f"cutoff={point['cutoff']:.2f} relevance={opt['mean_kept_chunk_relevance']} "
            f"R_tok={opt['r_tok']:.3f} ASR={opt['asr']:.3f} TCR={opt['tcr_benign']:.3f} "
            f"not_worse={point['asr_not_worse_than_g_ab'] and point['benign_tcr_not_worse_than_g_ab']}"
        )
    print(f"Saved cutoff sweep to {args.output}")


if __name__ == "__main__":
    main()
