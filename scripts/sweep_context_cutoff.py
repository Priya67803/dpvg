"""Sweep context-pruning cutoffs on validation cases (>= 200 cases)."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks import load_injecagent
from app.dataset import DOMAIN_CTX, NOISE, split_cases
from app.embeddings import cosine, default_tau, get_embedder
from app.gateway import Gateway
from app.token_opt import get_token_counter_source


def load_sweep_cases(n_cases: int = 200, seed: int = 42) -> list[dict]:
    """Load at least n_cases validation cases with context chunks for pruning."""
    cases = list(split_cases("val"))
    if len(cases) < n_cases:
        ia_cases = load_injecagent(setting="base", split="val")
        rng = random.Random(seed)
        shuffled = list(ia_cases)
        rng.shuffle(shuffled)
        for c in shuffled:
            if len(cases) >= n_cases:
                break
            tool = (c.get("subtasks") or [{}])[0].get("tool", "search_web")
            c_copy = dict(c)
            c_copy["context"] = list(DOMAIN_CTX.get(tool, [])) + list(NOISE[: 4 + (len(c["id"]) % 2)])
            cases.append(c_copy)
    return cases[:n_cases]


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
        "token_counter_source": get_token_counter_source(),
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
        "dataset": f"validation split ({len(cases)} cases: prototype + InjecAgent base)",
        "split": "val",
        "n_cases": len(cases),
        "backend": gateway.llm.name,
        "tau": gateway.tau,
        "token_counter_source": get_token_counter_source(),
        "g_ab_baseline": baseline,
        "points": points,
    }


def write_markdown_report(result: dict, out_path: pathlib.Path) -> None:
    lines = [
        "# DPVG Context Pruning Cutoff Sweep",
        "",
        f"- **Dataset**: {result['dataset']}",
        f"- **Backend**: {result['backend']}",
        f"- **Stage B Tau**: {result['tau']}",
        f"- **Token Counter Source**: {result['token_counter_source']}",
        f"- **G-AB Baseline**: ASR={result['g_ab_baseline']['asr']:.4f}, "
        f"TCR (Benign)={result['g_ab_baseline']['tcr_benign']:.4f}, "
        f"TCR (Attack)={result['g_ab_baseline']['tcr_under_attack']:.4f}, "
        f"R_tok={result['g_ab_baseline']['r_tok']:.4f}",
        "",
        "| Cutoff | Kept Chunks | Mean Relevance | R_tok | G-AB+T ASR | G-AB+T TCR (Benign) | G-AB+T TCR (Attack) | Utility Preserved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pt in result["points"]:
        opt = pt["g_ab_plus_t"]
        rel = f"{opt['mean_kept_chunk_relevance']:.4f}" if opt["mean_kept_chunk_relevance"] is not None else "N/A"
        preserved = "Yes" if pt["benign_tcr_not_worse_than_g_ab"] and pt["asr_not_worse_than_g_ab"] else "Degraded"
        lines.append(
            f"| {pt['cutoff']:.2f} | {opt['kept_chunk_count']} | {rel} | "
            f"{opt['r_tok']:.4f} | {opt['asr']:.4f} | {opt['tcr_benign']:.4f} | "
            f"{opt['tcr_under_attack']:.4f} | {preserved} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[round(i / 100, 2) for i in range(0, 61, 5)])
    parser.add_argument("--n-cases", type=int, default=200, help="Number of validation cases (>= 200)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "context_cutoff_sweep.json")
    args = parser.parse_args()
    n_cases = args.limit if args.limit is not None else args.n_cases
    cases = load_sweep_cases(n_cases=n_cases)
    embedder = get_embedder()
    gateway = Gateway(embedder=embedder, tau=default_tau(embedder), backend="vulnerable")
    result = sweep_context_cutoffs(cases, gateway, args.cutoffs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_output = args.output.with_suffix(".md")
    write_markdown_report(result, md_output)

    for point in result["points"]:
        opt = point["g_ab_plus_t"]
        print(
            f"cutoff={point['cutoff']:.2f} relevance={opt['mean_kept_chunk_relevance']} "
            f"R_tok={opt['r_tok']:.3f} ASR={opt['asr']:.3f} TCR={opt['tcr_benign']:.3f} "
            f"not_worse={point['asr_not_worse_than_g_ab'] and point['benign_tcr_not_worse_than_g_ab']}"
        )
    print(f"Saved cutoff sweep to {args.output} and {md_output}")


if __name__ == "__main__":
    main()
