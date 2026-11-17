"""Report adaptive-attack ASR by attack type and gateway configuration."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks import DATA_DIR, load_adaptive_attacks
from app.evaluation import evaluate_mode
from app.gateway import Gateway, PAPER_MODES


def evaluate_adaptive(cases: list[dict], tau: float | None = None, seed: int = 0, embedder=None) -> dict:
    attacks = [case for case in cases if case.get("is_attack")]
    if not attacks:
        raise ValueError("adaptive evaluation requires at least one attack case")
    gateway = Gateway(backend="vulnerable", tau=tau, seed=seed, embedder=embedder)
    modes = {}
    attack_types = sorted({case["attack_type"] for case in attacks})
    for mode in PAPER_MODES:
        row = evaluate_mode(gateway, mode, attacks)
        by_type = {}
        for attack_type, metrics in row["by_attack_type"].items():
            by_type[attack_type] = {
                "n": metrics["n"],
                "executed": metrics["executed"],
                "attack_success": metrics["attack_success"],
                "detected": metrics["detected"],
                "stage_b_accepted": metrics["stage_b_accepted"],
                "stage_b_acceptance_rate": metrics["stage_b_acceptance_rate"],
                "stopped_by_layer": metrics["stopped_by_layer"],
                "case_outcomes": metrics["case_outcomes"],
            }
        modes[mode] = {
            "label": row["label"],
            "components": row["components"],
            "asr": row["asr"],
            "by_attack_type": by_type,
        }

    by_attack_type = {}
    for attack_type in attack_types:
        count = sum(case["attack_type"] == attack_type for case in attacks)
        per_mode = {
            mode: modes[mode]["by_attack_type"].get(
                attack_type,
                {"n": 0, "executed": 0, "attack_success": 0.0, "detected": 0.0,
             "stage_b_accepted": 0, "stage_b_acceptance_rate": None,
             "stopped_by_layer": {}, "case_outcomes": []},
            )
            for mode in PAPER_MODES
        }
        by_attack_type[attack_type] = {
            "n": count,
            "modes": per_mode,
            "allowing_modes": [mode for mode, metrics in per_mode.items() if metrics["attack_success"] > 0],
            "stage_b_accepting_modes": [
                mode for mode, metrics in per_mode.items()
                if metrics["stage_b_acceptance_rate"] is not None
                and metrics["stage_b_acceptance_rate"] > 0
            ],
        }
    return {
        "schema_version": 1,
        "backend": gateway.llm.name,
        "tau": gateway.tau,
        "seed": seed,
        "n_attacks": len(attacks),
        "mode_order": PAPER_MODES,
        "by_attack_type": by_attack_type,
        "modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setting", choices=("base", "enhanced", "both"), default="both")
    parser.add_argument("--split", choices=("val", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit attack records for a smoke evaluation")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "adaptive_attack_asr.json")
    args = parser.parse_args()
    cases = load_adaptive_attacks(
        setting=args.setting, split=args.split, limit=args.limit, benign=False,
    )
    if not cases:
        parser.error("no adaptive attacks matched the requested setting/split")
    report = evaluate_adaptive(cases, tau=args.tau, seed=args.seed)
    report.update({"setting": args.setting, "split": args.split, "limit": args.limit})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {report['n_attacks']} adaptive attack results to {args.output}")
    print(f"tau={report['tau']:.3f}; ASR and Stage B acceptance by attack type:")
    for attack_type, item in report["by_attack_type"].items():
        summaries = ", ".join(
            f"{mode}={item['modes'][mode]['attack_success']:.3f}"
            for mode in item["allowing_modes"]
        ) or "none"
        stage_b = ", ".join(
            f"{mode}={item['modes'][mode]['stage_b_acceptance_rate']:.3f}"
            for mode in item["stage_b_accepting_modes"]
        ) or "none"
        print(f"  {attack_type}: executed in {summaries}; Stage B accepted candidates in {stage_b}")


if __name__ == "__main__":
    main()
