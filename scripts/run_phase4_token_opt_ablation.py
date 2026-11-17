"""Compare G-AB and G-AB+T on attack success, task completion, and token reduction."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dataset import split_cases
from app.embeddings import get_embedder
from app.gateway import Gateway, MODES


def _mode_stats(gateway: Gateway, cases: list[dict], mode: str) -> dict:
    runs = [gateway.run(case, mode) for case in cases]
    attacks = [(case, run) for case, run in zip(cases, runs) if case.get("is_attack")]
    benign = [(case, run) for case, run in zip(cases, runs) if not case.get("is_attack")]
    asr = sum(run["metrics"]["attack_success"] for _, run in attacks) / len(attacks) if attacks else 0.0
    tcr = (
        sum(run["metrics"]["completed"] and not run["metrics"]["detected"] for _, run in benign) / len(benign)
        if benign else 0.0
    )
    mode_obj = MODES[mode]
    token_counts, sources = [], set()
    for run in runs:
        metrics = run["metrics"]
        actual = metrics.get("llm_total_tokens")
        source = metrics.get("llm_token_source", "unavailable")
        if actual is not None and source not in ("unavailable", "none"):
            token_counts.append(actual)
            sources.add(source)
        else:
            token_counts.append(metrics["tokens_opt"] if mode_obj.token_opt else metrics["tokens_base"])
            sources.add("tiktoken-fallback")
    return {
        "mode": mode,
        "asr": round(asr, 4),
        "tcr": round(tcr, 4),
        "tokens": sum(token_counts),
        "token_source": next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unavailable"),
        "n_cases": len(runs),
        "n_attack": len(attacks),
        "n_benign": len(benign),
    }


def compare_modes(gateway: Gateway, cases: list[dict]) -> dict:
    baseline = _mode_stats(gateway, cases, "G-AB")
    optimized = _mode_stats(gateway, cases, "G-AB+T")
    t_base = baseline["tokens"]
    t_opt = optimized["tokens"]
    r_tok = 0.0 if t_base == 0 else 1.0 - t_opt / t_base
    baseline.update(t_base=t_base, t_opt=t_base, r_tok=0.0)
    optimized.update(t_base=t_base, t_opt=t_opt, r_tok=round(r_tok, 4))
    return {
        "title": "G-AB vs G-AB+T token optimization ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": gateway.llm.name,
        "embedder": gateway.embedder.name,
        "tau": gateway.tau,
        "formula": "R_tok = 1 - T_opt / T_base",
        "results": [baseline, optimized],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "val", "test"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "phase4_token_opt_ablation.json")
    args = parser.parse_args()
    cases = split_cases(args.split)
    if args.limit is not None:
        cases = cases[:args.limit]
    gateway = Gateway(embedder=get_embedder())
    check = getattr(gateway.llm, "check", None)
    if check:
        check()
    result = compare_modes(gateway, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in result["results"]:
        print(f'{row["mode"]:8} ASR={row["asr"]:.3f} TCR={row["tcr"]:.3f} '
              f'R_tok={row["r_tok"]:.3f} tokens={row["tokens"]} ({row["token_source"]})')
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
